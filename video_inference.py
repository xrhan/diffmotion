import torch
import numpy as np
import random
import os
import cv2
from omegaconf import OmegaConf
from einops import rearrange, repeat, reduce
from video_model.denoiser import ContinuousDiffusion
from collections import OrderedDict
from typing import Union, Sequence
from PIL import Image
from torchvision import transforms
from video_model import train_diffusion as hvd
from video_model.u_vit3d import *
from video_model.channel_mixer import *

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -------------- Model Initialization ---------------

_BASE_CFG = {
    "name": "u_vit3d",
    "patch_size": 2,
    "block_types": ["ResBlock","ResBlock","TransformerBlock","TransformerBlock"],
    "block_dropouts": [0.0,0.0,0.1,0.1],
    "num_updown_blocks": [3,3,3],
    "num_mid_blocks": 8,
    "pos_emb_type": "rope",
    "use_checkpointing": [False,False,False,False],
}
_PRESETS = {
    "default":   dict(channels=[64,128,256,512],   emb_channels=512,  num_heads=4),
    "mid" : dict(channels=[96,192,384,768],   emb_channels=768,  num_heads=6),
    "deep": dict(channels=[128,256,512, 1024],   emb_channels=1024,  num_heads=8),
    "nattn_mid": dict(channels=[96,192,384,768],   emb_channels=768,  num_heads=6, num_updown_blocks=[2,2,3]),
}
_MODEL_REGISTRY = {
    "uvit3d_all3":       ("UViT3D",               "default",   12,  9), # ModelClassName, preset_key, in_ch, out_ch
    "uvit3d_mid_all3":   ("UViT3D",               "mid",   12,  9),
    "uvit3d_deep_all3":  ("UViT3D",               "deep",   12,  9),
    "uvit3d_mixer_all3": ("UViT3D_NT_ResDoubleMixer", "nattn_mid", 12, 9),
    # add other modes as needed...
}


def build_cfg(preset_key: str) -> OmegaConf:
    if preset_key not in _PRESETS:
        raise ValueError(f"Unknown preset {preset_key}")
    return OmegaConf.merge(_BASE_CFG, _PRESETS[preset_key])


def load_checkpoint_model_only(model, checkpoint_path, device):
    if os.path.exists(checkpoint_path):
        # When loading models trained on multiple devices, map the checkpoint to the current device.
        checkpoint = torch.load(checkpoint_path, device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    else:
        return 0
    

def load_model(train_mode: str, ckpt_path: str, device: str, ddp: bool = True, full_ckpt = False):
    """
    Instantiate and load a model:
    - train_mode: key in _MODEL_REGISTRY
    - ckpt_path: path to checkpoint
    - device: 'cuda' or 'cpu'
    - ddp: if True, strip 'module.' prefixes; if False, load directly
    """
    # lookup
    if train_mode not in _MODEL_REGISTRY:
        raise ValueError(f"unsupported train_mode {train_mode}")
    ModelClassName, preset_key, in_ch, out_ch = _MODEL_REGISTRY[train_mode]
    ModelClass = globals()[ModelClassName]

    # build cfg & instantiate
    cfg = build_cfg(preset_key)
    model = ModelClass(
        cfg,
        resolution=256,
        in_channels=in_ch,
        out_channels=out_ch,
        max_tokens=3,
        external_cond_dim=3,
    ).to(device)

    # load checkpoint
    if full_ckpt:
      raw_state = torch.load(ckpt_path, map_location=device)['model_state_dict']
    else:
      raw_state = torch.load(ckpt_path, map_location=device)

    if ddp:
        state = OrderedDict((k.replace("module.", ""), v) for k, v in raw_state.items())
    else:
        state = raw_state
    model.load_state_dict(state)
    return model.eval()

# -------------- Locading Conditional Image ---------------

def preprocess_video_framepaths(frame_paths: Union[str, Sequence[str]], 
                                device: str,
                                horizon: int = 3,):
    """
    Load [f1,f2,f3] as normalized tensors in shape [1,horizon,C,H,W].
    """

    # make it take either static
    if isinstance(frame_paths, str):
            paths = [frame_paths] * horizon
    # or series of images
    elif isinstance(frame_paths, (list, tuple)):
        if len(frame_paths) == 0:
            raise ValueError("You must provide at least one frame path")
        if len(frame_paths) == 1:
            paths = frame_paths * horizon
        else:
            paths = list(frame_paths)

    to_tensor = transforms.ToTensor()
    imgs = []
    for p in paths:
        arr = np.array(Image.open(p).convert("RGB")) / 255.0 * 2.0 - 1.0
        if arr.shape[0] != 256 or arr.shape[1] != 256:
            arr = cv2.resize(arr, (256, 256), interpolation=cv2.INTER_AREA)
            arr = np.clip(arr, -1., 1.)
        imgs.append(to_tensor(arr))
    # [horizon, C, H, W] → [1, horizon, C, H, W]
    video = torch.stack(imgs, dim=0).unsqueeze(0).to(device)
    return video

# -------------- Diffusion Denoiser Setup ---------------

def build_diffusion(model, x_shape, device: str):
    """
    Returns a ContinuousDiffusion denoiser on `device`.
    """
    denoiser_cfg = hvd.initialize_denoiser()
    diff = ContinuousDiffusion(
        cfg=denoiser_cfg.diffusion,
        backbone_model=model,
        x_shape=x_shape,
        max_tokens=3,
        external_cond_dim=3,
    )
    return diff.to(device)


def ddim_idx_to_noise_level(timesteps, sampling_timesteps, indices: torch.Tensor):
    shape = indices.shape
    real_steps = torch.linspace(-1, timesteps - 1, sampling_timesteps + 1)
    real_steps = real_steps.long().to(indices.device)
    k = real_steps[indices.flatten()]
    return k.view(shape)


def make_scheduling_matrix(
    horizon: int, sampling_timesteps: int, timesteps: int, batch_size: int, device: str
):
    # reverse schedule
    sched = np.arange(sampling_timesteps, -1, -1)[:, None].repeat(horizon, axis=1)
    sched = torch.from_numpy(sched).long().to(device)
    # map idx→noise level
    sched = ddim_idx_to_noise_level(timesteps, sampling_timesteps, sched)
    # add batch dim: [m, horizon] → [m, batch, horizon]
    sched = repeat(sched, "m h -> m b h", b=batch_size)
    return sched


# -------------- Sampling Loop ---------------

def set_seed(seed: int):
    """Fix random seeds for python, numpy, torch (CPU & all GPUs) and enforce CuDNN determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

@torch.no_grad
def sample_video_ddim(
    diffusion: ContinuousDiffusion,
    video_cond: torch.Tensor,
    scheduling_matrix: torch.Tensor,
    x_shape=(9, 256, 256),
    clip_noise=20.0,
    seed: int = None,
):
    
    if seed is not None:
        set_seed(seed)

    m_steps, batch_size, horizon = scheduling_matrix.shape
    # init noise
    xs = torch.randn((batch_size, horizon, *x_shape), device=video_cond.device)
    xs = torch.clamp(xs, -clip_noise, clip_noise)

    diffusion.eval()
    with torch.no_grad():
        for m in range(m_steps - 1):
            from_lv = scheduling_matrix[m]
            to_lv   = scheduling_matrix[m + 1]
            xs = diffusion.sample_step(xs, from_lv, to_lv, video_cond)
    return xs


def extract_output_slices(
    xs_pred: torch.Tensor,
    group_slices=[(0,3), (3,6), (6,9)],
    group_names=["shape", "albedo", "material"],
) -> dict:
    '''
    Normalize all outputs to (0, 1) range in numpy arrays
    
    Returns:
      outputs: dict mapping each name to a numpy array of shape [B, T, H, W, 3]
    '''

    assert xs_pred.dim() == 5, "xs_pred must be [B, T, C, H, W]"
    B, T, C, H, W = xs_pred.shape
    assert len(group_slices) == len(group_names)
    outputs = {}

    # move to CPU & numpy once
    xs_np = xs_pred.detach().cpu().numpy()  # [B, T, C, H, W]

    for (start, end), name in zip(group_slices, group_names):
        # slice out C=[start:end] → shape [B, T, 3, H, W]
        grp = xs_np[:, :, start:end, :, :]
        # map from [-1,1] to [0,1]
        grp = (grp + 1.0) * 0.5
        np.clip(grp, 0.0, 1.0, out=grp)
        # reorder to [B, T, H, W, 3]
        grp = np.moveaxis(grp, 2, -1)
        outputs[name] = grp

    return outputs

# -------------- Visualization ---------------

def create_side_by_side_gif(array1, array2, output_path, sequence = [0, 1, 2, 1, 0], duration=200):
    frames = []  # Sequence of frames
    array1 = (array1 * 255).astype(np.uint8)
    array2 = (array2 * 255).astype(np.uint8)

    for i in sequence:
        # Concatenate images horizontally
        combined_array = np.concatenate((array1[i], array2[i]), axis=1)  # Shape: (64, 128, 3)
        combined_img = Image.fromarray(combined_array)
        combined_img = combined_img.convert("P", palette=Image.ADAPTIVE, colors=256)  # Reduce variation
        frames.append(combined_img)

    # Save as GIF
    frames[0].save(output_path, save_all=True, append_images=frames[1:], loop=0, duration=duration)


import imageio
def create_side_by_side_mp4(array1, array2, array3,
                            output_path,
                            sequence=[0, 1, 2, 1, 0],
                            fps=5):
    """
    Create an MP4 video by concatenating three image sequences horizontally.
    
    Parameters
    ----------
    array1, array2, array3 : np.ndarray
        Shape (T, H, W, C), with values in [0,1] or [0,255].
    output_path : str
        Path to write the .mp4 file (e.g. "out.mp4").
    sequence : list of int
        Frame indices to play (you can loop back and forth).
    fps : int
        Frames per second for the output video.
    """
    # Ensure uint8 in [0,255]
    def prepare(arr):
        arr = arr.copy()
        if arr.dtype != np.uint8:
            arr = (arr * 255).clip(0,255).astype(np.uint8)
        return arr

    a1 = prepare(array1)
    a2 = prepare(array2)
    a3 = prepare(array3)

    # Check that heights and channels match
    H1, W1, C1 = a1.shape[1:]
    H2, W2, C2 = a2.shape[1:]
    H3, W3, C3 = a3.shape[1:]
    assert (H1,C1)==(H2,C2)==(H3,C3), "All arrays must have same height and channels"

    # Open a video writer
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', 
                                ffmpeg_params=['-pix_fmt', 'yuv420p'])

    for idx in sequence:
        # Concatenate horizontally: (H, W1+W2+W3, C)
        frame = np.concatenate((a1[idx], a2[idx], a3[idx]), axis=1)
        writer.append_data(frame)

    writer.close()
    print(f"Saved MP4 to {output_path}")


# -------------- Eval Loops ---------------

def sample_single_video_with_seeds(
    diffusion,
    video_cond: torch.Tensor,
    scheduling_matrix: torch.Tensor,
    seeds: list[int],
    x_shape=(9, 256, 256),
    clip_noise=20.0,
    group_slices=[(0,3), (3,6), (6,9)],
    group_names=["shape", "albedo", "material"],
) -> dict[int, torch.Tensor]:
    
    results = {}
    for seed in seeds:
        xs_pred = sample_video_ddim(
            diffusion,
            video_cond,
            scheduling_matrix,
            x_shape=x_shape,
            clip_noise=clip_noise,
            seed=seed
        )  # out: [B, T, *x_shape]
        out = extract_output_slices(xs_pred, group_slices, group_names) # single batch
        results[seed] = out
        
    return results

