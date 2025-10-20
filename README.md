# Generative Perception of Shape and Material from Differential Motion

(repo work in progress)

[**arXiv**](https://arxiv.org/pdf/2506.02473) | [**Project Page**](https://xrhan.github.io/diffmotion/) 

## 🍇 Introduction
We introduce a generative perception model that, given a few frames of an object undergoing motion, produces diverse and plausible interpretations of its shape and material.

## 🍊 Usage

### Dependencies
Create a new environment with conda, and install pytorch. Make sure to use a pytorch version compatiable with [NATTEN](https://natten.org/install/). Our project used the NATTEN 0.17.5 release with torch=2.5.1.

```bash
conda create -n diffmotion python=3.10
conda activate diffmotion
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

Install other dependencies as follows
```bash
pip install -r requirements.txt
```

To use the UViT3D-Mixer model with shift-invariant neighborhood attention, you will need to install custom CUDA kernels via [NATTEN](https://github.com/SHI-Labs/NATTEN/tree/main).

## 🍬 Inference
We include a few sample videos under `test_data/`.  
To run on your own inputs, edit `run_inference.py` and list the **three frames in order** for each test clip.

### What the script does
- Uses random seeds **0–9** by default (see `main()` in `run_inference.py`).
- Saves each result as a **3×3 image grid**:
  - **Rows:** diffuse albedo, surface normals, materials
  - **Columns:** frame 1 → frame 3 (left to right)
- Outputs are written to the `evals/` directory.

### Quick start
```bash
# 1) Create an output folder (once)
mkdir -p evals

# 2) Run inference with your checkpoint and a custom save name
python run_inference.py \
  --ckpt_path "./ckpts/u_vit3d_mixer_e2.ckpt" \
  --save_name "exp1"
```

### Model checkpoint
You can download our pretrained [model checkpoint](https://drive.google.com/file/d/1YCtgWeevOqW1ZDLwgpqK3jXJGRkT6Syy/view?usp=sharing) as follows:
```bash
mkdir ckpts
cd ckpts
gdown 1YCtgWeevOqW1ZDLwgpqK3jXJGRkT6Syy
```

## 🫐 Training
To train the model from scratch, we use the script in `video_model/train_diffusion.py`
Our model is trained with 4 A100 or H100 GPUs for around 200 epochs and with a batch size of 16 per GPU (so effective batch size 64). Training can be performed efficiently with mixed precision training using bf16.

For instance,
```bash
mkdir runs # logging files
mkdir saved_video_models # saved training checkpoints
accelerate launch -m video_model.train_diffusion \
  --train-mode "uvit3d_mixer_all3" \
  --distributed --mp16\
  --epochs 1 \
  --batch-size 16 \
  --aug-static \
  --aug-reverse \
  --save-name "mixer_test" \
  --root-dir './' \
  --save-every 10 \
  --dataset-root-dir './'
```

### Dataset
Please email `xinranhan@g.harvard.edu` for the training dataset we used in the paper.

## 🍒 Synthetic Data Generation
We provide the code to generate textured, synthetic data from Mitsuba3, with ground truth labels of the geometry and materials using Mitsuba3.
Please refer to [DiffMotion-DataGen](https://github.com/xrhan/diffmotion_datagen).

## 🌰 Citation
If you find this repo useful, please consider citing:

```bibtex
@article{han2025generative,
  title={Generative Perception of Shape and Material from Differential Motion},
  author={Han, Xinran Nicole and Nishino, Ko and Zickler, Todd},
  journal={arXiv preprint arXiv:2506.02473},
  year={2025}
}
```

## 🍰 Acknowledgement
This project builds upon several excellent open source projects:
- [Diffusion Forcing Transformer](https://github.com/kwsong0113/diffusion-forcing-transformer)
- [Neighborhood Attention Transformer](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer)
- [Hourglass Diffusion Transformer](https://github.com/crowsonkb/k-diffusion?tab=readme-ov-file)

We thanks the authors of those projects and the developers of Mitsuba3 for their valuable contributions to the open source community. 
