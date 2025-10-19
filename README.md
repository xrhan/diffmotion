# Generative Perception of Shape and Material from Differential Motion

(repo work in progress)

[**arXiv**](https://arxiv.org/pdf/2506.02473) | [**Project Page**](https://xrhan.github.io/diffmotion/) 

## 🍇 Introduction
We introduce a generative perception model that, given a few frames of an object undergoing motion, produces diverse and plausible interpretations of its shape and material.

## 🍊 Usage
To use the UViT3D-Mixer model with shift-invariant neighborhood attention, you will need to install custom CUDA kernels via [NATTEN](https://github.com/SHI-Labs/NATTEN/tree/main).

For training the model, we include an example slurm script in the scripts folder for mixed precision (bf16) training on 4 A100-80G GPUs.

## Inference


### model checkpoint
(coming soon)

## Training
(dataset)

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
- [Diffusion Forcing Transformer](https://github.com/kwsong0113/diffusion-forcing-transformer), 
- [Neighborhood Attention Transformer](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer),  
- [Hourglass Diffusion Transformer](https://github.com/crowsonkb/k-diffusion?tab=readme-ov-file),
We thanks the authors of those projects and the developers of Mitsuba3 for their valuable contributions to the open source community. 
