from video_model.video_loader import *
from inference_utils import *
import datetime
import glob
import argparse
from pathlib import Path
from typing import Iterable, List

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **kwargs):  # fallback: no-op
        return x

def setup_model_denoiser(train_mode, ckpt_path, device = 'cuda', ddp = True, full_ckpt=True):
    model = load_model(train_mode = train_mode, 
                       ckpt_path = ckpt_path, 
                       device = device, 
                       ddp = ddp, 
                       full_ckpt=full_ckpt)

    denoiser = build_diffusion(model, (9, 256, 256), 'cuda')
    denoise_scheduler = make_scheduling_matrix(horizon=3, 
                                               sampling_timesteps=50, 
                                               timesteps=300, 
                                               batch_size=1, 
                                               device='cuda')
    return denoiser, denoise_scheduler


def load_static_image_from_folder(
    folder_path: str,
    exts: Sequence[str] = ('png', 'jpg', 'jpeg', 'bmp', 'tiff'),):
    """
    Reads all image files in `folder_path`, sorts them,    
    """
    # gather all matching files
    file_paths = []
    for ext in exts:
        pattern = os.path.join(folder_path, f'*.{ext}')
        file_paths.extend(glob.glob(pattern))
    if not file_paths:
        raise ValueError(f'No image files found in {folder_path} with extensions {exts!r}')
    
    # sort so frame order is consistent
    file_paths = sorted(file_paths)
    return file_paths


def test_videos_paths(folder_path):
    '''
    Specify the test video frames here.
    '''
    to_test = [
        ['teapot1.png', 'teapot2.png', 'teapot3.png'],
        ['duck1.png', 'duck2.png', 'duck3.png'],
        ['vase1.png', 'vase2.png', 'vase3.png'],
        ['mickey.png', 'mickey.png', 'mickey.png'] # repeat image 3 times for static scene
    ]

    full_paths = [
        [os.path.join(folder_path, rel_path) for rel_path in group]
        for group in to_test
    ]
    return full_paths


def test_videos_run(diffusion, scheduling_matrix, static_files_dir, save_dir, 
                      seeds, device, save_name, folder_name='test_data'):
    
    # 1) get test videos paths (specify frame 0, 1, 2)
    load_files_dir = os.path.join(static_files_dir, folder_name)
    file_paths = test_videos_paths(load_files_dir)

    # 2) make timestamped results folder
    date_str    = datetime.datetime.now().strftime("%Y%m%d")
    results_dir = os.path.join(save_dir, f"eval_{date_str}_{save_name}")
    os.makedirs(results_dir, exist_ok=True)

    mapping = {}
    # 3) iterate and predict
    for idx, p in tqdm(enumerate(file_paths), total=len(file_paths),
                       desc=f"Evaluating ({save_name})", unit="vid"):
        mapping[idx] = p
        rgb_frames = preprocess_video_framepaths(p, device, horizon = 3).float()

        curr_results = sample_single_video_with_seeds(
            diffusion,
            rgb_frames,
            scheduling_matrix,
            seeds,
            x_shape=(9, 256, 256),
            clip_noise=20.0,
            group_slices=[(0,3),(3,6),(6,9)],
            group_names=["shape","albedo","material"],
        )

        for s, preds in curr_results.items(): # prepare display arrays

            preds_shape = preds['shape'][0]; preds_albedo = preds['albedo'][0]; preds_mat = preds['material'][0]
            disp_normals = [(p * 255).astype(np.uint8) for p in preds_shape] # 0 assumes batch size 1
            disp_albedos = [(p * 255).astype(np.uint8) for p in preds_albedo]
            disp_materials = [(p * 255).astype(np.uint8) for p in preds_mat]

            # combine into a 2-row image
            norm_row   = np.concatenate(disp_normals, axis=1)  # [H, F*W, 3]
            albedo_row = np.concatenate(disp_albedos, axis=1)  # [H, F*W, 3]
            material_row = np.concatenate(disp_materials, axis=1)
            combined   = np.vstack([norm_row, albedo_row, material_row])    # [3*H, F*W, 3]

            # save PNG
            fname = f"step{idx}_seed{s}_normals_albedo.png"
            Image.fromarray(combined).save(os.path.join(results_dir, fname))
    
    # save filepath meta-data
    json_path = os.path.join(results_dir, 'index_to_filepath.json')
    with open(json_path, 'w') as f:
        json.dump(mapping, f, indent=2)

    print(f"[INFO] Saved to {results_dir}.")


################### running inference experiment ###################
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # high-level mode
    p.add_argument("--test_mode", default="videos", choices=["videos"], help="Evaluation mode.")
    # data
    p.add_argument("--test_root_dir", default="./",
                   help="Root directory of user-specified test videos.")
    p.add_argument("--dataset_dir", default="./Dataset/video_minitest/motion_standard/test/",
                   help="Root directory of eval videos dataset.")
    p.add_argument("--dataset_json_path", default="./Dataset/video_minitest/eval_dataset_precompute_pairs.json",
                   help="Optional JSON with precomputed pairs or metadata (if used downstream).")
    p.add_argument("--folder_name", default="test_data",
                   help="Subfolder name used by the evaluator for reading test data.")
    # outputs
    p.add_argument("--save_dir", default="./evals", help="Where to save evaluation outputs.")
    p.add_argument("--save_name", default=None,
                   help="Override the default save name (defaults to train_mode).")
    # model
    p.add_argument("--train_mode", default="uvit3d_mixer_all3", help="Model/training variant identifier.")
    p.add_argument("--ckpt_path", default="./ckpts/u_vit3d_mixer_e2.ckpt",
                   help="Path to model checkpoint.")
    p.add_argument("--full_ckpt", action="store_true", default=True,
                   help="Load checkpoint as a full state dict (vs. partial).")
    # runtime
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device type to use.")
    p.add_argument("--ddp", action="store_true", default=True, help="Assume DDP-trained checkpoint if required.")
    return p

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def main(argv: Iterable[str] | None = None):
    seeds = [int(x) for x in np.arange(0, 10)]
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    
    dataset_dir = Path(args.dataset_dir).expanduser()
    dataset_json_path = Path(args.dataset_json_path).expanduser()
    save_dir = Path(args.save_dir).expanduser()
    ckpt_path = Path(args.ckpt_path).expanduser()
    _ensure_dir(save_dir)

    diffusion, scheduling_matrix = setup_model_denoiser(args.train_mode,
                                                        str(ckpt_path),
                                                        device=args.device,
                                                        ddp=bool(args.ddp),
                                                        full_ckpt=bool(args.full_ckpt),)
    
    #TODO: add eval mode
    save_name = args.save_name or args.train_mode

    if args.test_mode == 'videos':
        test_root_dir = args.test_root_dir
        test_videos_run(
            diffusion=diffusion,
            scheduling_matrix=scheduling_matrix,
            static_files_dir=test_root_dir,
            save_dir=str(save_dir), 
            seeds=seeds,
            device='cuda',
            save_name=save_name,
            folder_name=args.folder_name)

if __name__ == "__main__":
    main()