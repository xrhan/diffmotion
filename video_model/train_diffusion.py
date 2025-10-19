import torch
import datetime
from einops import rearrange, reduce
from .u_vit3d import UViT3D

from omegaconf import DictConfig, OmegaConf
import argparse
import logging
from accelerate import Accelerator
from .denoiser import *
from .video_loader import *
from .u_vit3d_mixer import *


def setup_logging(root_dir, save_name):
    """Configures logging and returns a logger."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    log_filename = f"{root_dir}runs/video_model_training_{timestamp}_{save_name}.log"
    logging.basicConfig(filename=log_filename, level=logging.INFO, 
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def get_training_noise_levels(xs, timesteps, is_continuous):
    """Generate random noise levels for training."""
    batch_size, n_tokens, *_ = xs.shape # non-latent model n_tokens = n_frames
    if is_continuous:
        noise_levels = torch.rand((batch_size, 1)).repeat(1, n_tokens)
    else:
        noise_levels = torch.randint(0, timesteps, (batch_size, 1)).repeat(1, n_tokens)
    return noise_levels


def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # Add more states as needed e.g., scheduler state.
    }
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path, accelerator):
    if os.path.exists(checkpoint_path):
        # When loading models trained on multiple devices, map the checkpoint to the current device.
        checkpoint = torch.load(checkpoint_path, map_location=accelerator.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logging.info(f"Resuming training from epoch {start_epoch}")
        return start_epoch
    else:
        return 0
    

def train_video_diffusion_highres_accelerate(model, optimizer, dataloader, denoiser_cfg, epochs, accelerator,
                                             save_every=50, timesteps=300, log_every=100, pred_channels = 9,
                                             root_dir="", train_save_name="", resume_checkpoint=None,):
    
    checkpoint_dir = os.path.join(root_dir, 'saved_video_models', f'u_vit3d_cont_{train_save_name}.ckpt')

    is_continuous = denoiser_cfg.diffusion.is_continuous
    diffusion_cls = ContinuousDiffusion if is_continuous else DiscreteDiffusion

    start_epoch = 0
    if resume_checkpoint is not None:
        to_load_ckpt = os.path.join(root_dir, 'saved_video_models', f'u_vit3d_cont_{resume_checkpoint}.ckpt')
        if os.path.exists(to_load_ckpt):
            start_epoch = load_checkpoint(model, optimizer, to_load_ckpt, accelerator)

    # Wrap model into the diffusion denoiser.
    diffusion_denoiser = diffusion_cls(
        cfg=denoiser_cfg.diffusion,
        backbone_model=model,
        x_shape=(pred_channels, 256, 256),  # single-image shape as in your code
        max_tokens=3,
        external_cond_dim=3 # rgb conditionin
    )

    model.train()

    for epoch in range(start_epoch, epochs):
        
        epoch_loss_sum = 0.0
        shape_loss_sum = 0.0
        albedo_loss_sum = 0.0
        mat_loss_sum = 0.0

        num_batches = 0

        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # With Accelerate, batches are already on the proper device.
            rgb_frames = batch['rgbs'].float()
            normal_frames = batch['normals'].float()
            albedo_frames = batch['albedos'].float()
            metallic_frames = batch['metallic'].float()
            specular_frames = batch['specular'].float()
            roughness_frames = batch['roughness'].float()

            # B x F x C x H x W concatenate along channel = 2
            target_frames = torch.cat((normal_frames, albedo_frames, metallic_frames, specular_frames, roughness_frames), dim = 2)

            # Compute noise levels and ensure they are on the correct device.
            noise_levels = get_training_noise_levels(target_frames, timesteps, is_continuous).to(target_frames.device)
            xs_pred, loss = diffusion_denoiser(x=target_frames, k=noise_levels, external_cond=rgb_frames)
            
            shape_loss = loss[:, :, 0:3, :, :].mean()
            albedo_loss = loss[:, :, 3:6, :, :].mean()
            mat_loss = loss[:, :, 6:9, :, :].mean()
            
            # loss = (shape_loss + albedo_loss + mat_loss) / 3. # equal weighting
            loss = 0.4 * shape_loss + 0.4 * albedo_loss + 0.2 * mat_loss

            accelerator.backward(loss)
            optimizer.step()

            epoch_loss_sum += loss.item()
            shape_loss_sum += shape_loss.item()
            albedo_loss_sum += albedo_loss.item()
            mat_loss_sum += mat_loss.item()
            num_batches += 1

            if step % log_every == 0:
                local_avg_loss = epoch_loss_sum / num_batches
                avg_shape_loss = shape_loss_sum / num_batches
                avg_albedo_loss = albedo_loss_sum / num_batches
                avg_mat_loss = mat_loss_sum / num_batches
                
                # Use accelerator.gather to collect loss values from all processes.
                loss_tensor = torch.tensor(local_avg_loss, device=target_frames.device)
                global_avg_loss = accelerator.gather(loss_tensor).mean().item()

                shape_loss_tensor = torch.tensor(avg_shape_loss, device=target_frames.device)
                global_avg_shape_loss = accelerator.gather(shape_loss_tensor).mean().item()                

                albedo_loss_tensor = torch.tensor(avg_albedo_loss, device=target_frames.device)
                global_avg_albedo_loss = accelerator.gather(albedo_loss_tensor).mean().item()    

                mat_loss_tensor = torch.tensor(avg_mat_loss, device=target_frames.device)
                global_avg_mat_loss = accelerator.gather(mat_loss_tensor).mean().item()    

                if accelerator.is_main_process:
                                logging.info(
                                    f"Epoch {epoch} | Step {step} - "
                                    f"Avg Loss: {global_avg_loss:.6f} | "
                                    f"Shape: {global_avg_shape_loss:.6f} | "
                                    f"Albedo: {global_avg_albedo_loss:.6f} | "
                                    f"Mat: {global_avg_mat_loss:.6f}"
                                )

        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            if accelerator.is_main_process:
                save_checkpoint(model, optimizer, epoch, checkpoint_dir)
                logging.info("Saving model checkpoint...")

        # Optionally clear CUDA cache.
        # torch.cuda.empty_cache()


def main_accelerate(args, seed = 42):
    # Initialize Accelerator (handles distributed init, device placement, etc.)
    if args.mp16:
        accelerator = Accelerator(mixed_precision="bf16")
        if accelerator.is_main_process:
            logging.info('training with mixed precision bf16.')
    else:
        accelerator = Accelerator()

    # Set seeds for reproducibility.
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create your dataset and dataloader.
    dataloader = DataLoader(
        dataset_train,  # assume dataset_train is defined globally or imported
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )

    # Initialize the model (Accelerate will move it to the correct device).
    model = initialize_highres_model(args, device="cuda")
    denoiser_cfg = initialize_denoiser()

    # Calculate effective batch size and scale learning rate.
    eff_batch_size = args.batch_size * accelerator.num_processes
    eff_lr = 1e-4 * eff_batch_size / 60  # default 1e-4 lr for batch size 60

    optimizer = torch.optim.AdamW(model.parameters(), lr=eff_lr, weight_decay=0.01, betas=(0.9, 0.99))

    # Prepare model, optimizer, and dataloader for distributed training.
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Log setup info on main process.
    if accelerator.is_main_process:
        logger.info(
            f"Starting training with Accelerate: Num GPUs: {accelerator.num_processes}, "
            f"Effective learning rate: {eff_lr}, Effective batch size: {eff_batch_size}"
        )

    # Start training.
    ckpt = None
    if args.ckpt != "":
        ckpt = args.ckpt
    
    train_video_diffusion_highres_accelerate(
        model, optimizer, dataloader, denoiser_cfg, args.epochs, accelerator,
        save_every=args.save_every, timesteps=300, log_every=100,
        root_dir=args.root_dir, train_save_name=args.save_name, resume_checkpoint=ckpt,
    )

def initialize_denoiser():
    denoiser_cfg = OmegaConf.create({
    "diffusion": {
            "is_continuous": True,
            "precond_scale": 0.125,
            "timesteps": 300,
            "beta_schedule": "cosine_simple_diffusion",
            "schedule_fn_kwargs": {
                "shift": 0.125,
                "interpolated": False,
            },
            "use_causal_mask": False,
            "clip_noise": 20.0,
            "objective": "pred_v",
            "loss_weighting": {
                "strategy": "sigmoid",
                "sigmoid_bias" : -1.0,
                "snr_clip": 5.0,
                "cum_snr_decay": 0.9,
            },
            "training_schedule" : {
                "name": "cosine",
                "shift": 0.125,
            },
            "sampling_timesteps": 50,
            "ddim_sampling_eta": 0.0,
            "reconstruction_guidance": 0.0,}})
    
    return denoiser_cfg


def initialize_highres_model(args, device):
    """Initializes and returns the model based on given arguments."""
    cfg = OmegaConf.create({
        "name": "u_vit3d",
        "channels": [64, 128, 256, 512],
        "emb_channels": 512,
        "patch_size": 2,
        "block_types": ["ResBlock", "ResBlock", "TransformerBlock", "TransformerBlock"],
        "block_dropouts": [0.0, 0.0, 0.1, 0.1],
        "num_updown_blocks": [3, 3, 3],
        "num_mid_blocks": 8,
        "num_heads": 4,
        "pos_emb_type": "rope",
        "use_checkpointing": [False, False, False, False],
    })

    cfg_deep = OmegaConf.create({
        "name": "u_vit3d",
        "channels": [128, 256, 512, 1024],
        "emb_channels": 1024,
        "patch_size": 2,
        "block_types": ["ResBlock", "ResBlock", "TransformerBlock", "TransformerBlock"],
        "block_dropouts": [0.0, 0.0, 0.1, 0.1],
        "num_updown_blocks": [3, 3, 3],
        "num_mid_blocks": 8,
        "num_heads": 8,
        "pos_emb_type": "rope",
        "use_checkpointing": [False, False, False, False],
    })

    cfg_mid = OmegaConf.create({
        "name": "u_vit3d",
        "channels": [96, 192, 384, 768],
        "emb_channels": 768,
        "patch_size": 2,
        "block_types": ["ResBlock", "ResBlock", "TransformerBlock", "TransformerBlock"],
        "block_dropouts": [0.0, 0.0, 0.1, 0.1],
        "num_updown_blocks": [3, 3, 3],
        "num_mid_blocks": 8,
        "num_heads": 6,
        "pos_emb_type": "rope",
        "use_checkpointing": [False, False, False, False],
    })

    # replace one ResBlock with Nattn (and optionally tattn)
    cfg_nattn_mid = OmegaConf.create({
        "channels": [96, 192, 384, 768],
        "emb_channels": 768,
        "patch_size": 2,
        "block_types": ["ResBlock", "ResBlock", "TransformerBlock", "TransformerBlock"],
        "block_dropouts": [0.0, 0.0, 0.1, 0.1],
        "num_updown_blocks": [2, 2, 3],
        "num_mid_blocks": 8,
        "num_heads": 6,
        "pos_emb_type": "rope",
        "use_checkpointing": [False, False, False, False],
    })   

    if args.train_mode == "uvit3d_all3": 
        model = UViT3D(cfg_mid, 
                       resolution=256, 
                       in_channels=12, # 3 img, 3 shape, 3 albedo, 3 brdf
                       out_channels=9, 
                       max_tokens=3, 
                       external_cond_dim=3)
        model.to(device)
        return model

    elif args.train_mode == "uvit3d_mixer_all3": 
        model = UViT3D_Mixer(cfg_nattn_mid, 
                       resolution=256, 
                       in_channels=12, # 3 img, 3 shape, 3 albedo, 3 brdf
                       out_channels=9, 
                       max_tokens=3, 
                       external_cond_dim=3)
        model.to(device)
        return model

    else:
        raise ValueError(f"Unsupported train mode: {args.train_mode}")


def get_arg_parser_highres():
    """Returns an argument parser for training configurations."""
    parser = argparse.ArgumentParser(description="Train a model with specified parameters")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--save-name", type=str, default="model", help="Suffix of the saved model")
    parser.add_argument("--loss-type", type=str, default="huber", help="Loss function: l1, l2, or huber")
    parser.add_argument("--lr", type=float, default=2e-4, help="Loss function: l1, l2, or huber")
    parser.add_argument("--scheduler-type", type=str, default="cosine", help="Scheduler type: cosine or linear")
    parser.add_argument("--diffusion-timestep", type=int, default=300, help="Diffusion scheduler timestep")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-mode", type=str, default="uvit3d")
    parser.add_argument("--img-channels", type=int, default=3) # default: rgb image
    parser.add_argument("--out-channels", type=int, default=3)
    parser.add_argument("--root-dir", type=str, default='./')
    parser.add_argument("--dataset-root-dir", type=str, default='./')
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument('--aug-static', action='store_true', help="Augment with static video frames")
    parser.add_argument('--aug-reverse', action='store_true', help="Augment with reversed video frames")
    parser.add_argument('--distributed', action='store_true', default=False)
    parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay (default: 0.02)')
    parser.add_argument('--mp16', action='store_true', default=False, help='bf16 mixed precision training.')
    parser.add_argument('--ckpt', type=str, default="", help='')
    
    return parser


if __name__ == "__main__":
    parser = get_arg_parser_highres()
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.root_dir, args.save_name)

    # Load dataset
    dataset_dir = args.dataset_root_dir + 'Dataset/render_test_0329_processed/'
    dataset_json_path = dataset_dir + '0329_fixed.json' 
    dataset_train = ObjectMotionDataset(dataset_dir=dataset_dir, video_paths_json=dataset_json_path, 
                                  num_frames=3, augment_bg=False, resize_to=None, # (256, 256)
                                  augment_static=args.aug_static, augment_reverse=args.aug_reverse)
    
    # Device setup and model init
    device = "cuda" if torch.cuda.is_available() else "cpu"

    main_accelerate(args)