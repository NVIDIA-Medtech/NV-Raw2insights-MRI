# NV-Raw2insights-MRI

**NV-Raw2insights-MRI** is a universal MRI reconstruction model based on the Scalable Deep Unrolled Model (SDUM) framework. It produces high-quality reconstructions from undersampled k-space across diverse protocols—anatomical targets, contrasts, sampling patterns, and acceleration factors—without task-specific fine-tuning. This project was conducted by NVIDIA in collaboration with the [CMRxRecon Team](https://github.com/CmrxRecon), [Fudan University](https://hupi.fudan.edu.cn/en/rcdw/rc_content.jsp?urltype=news.NewsContentUrl&wbtreeid=1105&wbnewsid=1370), and [Johns Hopkins University](https://profiles.hopkinsmedicine.org/provider/shanshan-jiang/2777746). We thank the CMRxRecon Team for providing the dataset used in this work, enabling the release of NV-Raw2Insights-MRI under the commercially usable [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). We kindly ask users to cite the CMRxRecon dataset papers in any related publications.

<p align="center">
 🤗 <a href="https://huggingface.co/nvidia/NV-Raw2insights-MRI">Hugging Face</a>&nbsp | <a href="https://arxiv.org/abs/2512.17137">Paper</a> | <a href="https://github.com/NVIDIA-Medtech/NV-Raw2insights-MRI">Repository</a>
</p>

<p align="center">
<img width="902.5" height="200" alt="Image" src="https://github.com/user-attachments/assets/27ce9dea-c592-4dd5-b984-38542e1cf8e6" />
</p>

## News
* [March 16, 2026] As part of the NVIDIA Clara Open Models family, we released [NV-Raw2insights-MRI](https://github.com/NVIDIA-Medtech/NV-Raw2insights-MRI)
* [Feb 4, 2026] **NV-Raw2Insights-MRI** achieved 🏆 **1st** place across all four tracks in the [CMRxRecon2025 Challenge](https://www.synapse.org/Synapse:syn59814210/wiki/634966) without task-specific fine-tuning.

## Overview

Clinical MRI encompasses diverse imaging protocols—spanning anatomical targets (cardiac, brain, knee), contrasts (T1, T2, mapping), sampling patterns (Cartesian, radial, spiral, kt-space), and acceleration factors—yet current deep learning reconstructions are typically protocol-specific. NV-Raw2insights-MRI (SDUM) combines:

- **Restormer-based reconstructor** with cascaded unrolled architecture
- **Learned coil sensitivity map estimator (CSME)** per cascade
- **Sampling-aware weighted data consistency (SWDC)**
- **Universal conditioning (UC)** on cascade index and protocol metadata
- **Progressive cascade expansion training** with foundation-model-like scaling

A single model trained on heterogeneous data achieves state-of-the-art results across CMRxRecon2025 challenge tracks (multi-center, multi-disease, 5T, pediatric) and generalizes to CMRxRecon2024 and fastMRI brain.

## Model Family

| Model Name | Description | Cascades | Size |
|------------------|------|:-------------:|:-----:|
| [NV-Raw2insights-MRI-Small](https://huggingface.co/nvidia/NV-Raw2insights-MRI) | Lightweight variant | 6 | 230M |
| [NV-Raw2insights-MRI-Base](https://huggingface.co/nvidia/NV-Raw2insights-MRI)| Default balanced variant | 18 | 760M |
|[NV-Raw2insights-MRI-Large](https://huggingface.co/nvidia/NV-Raw2insights-MRI) | High-capacity variant | 34 | 1.4B |

Checkpoints are automatically downloaded from [Hugging Face](https://huggingface.co/nvidia/NV-Raw2insights-MRI) when not provided locally.

## User Guide

- [Setup Guide](docs/setup.md)
- [Inference](docs/inference.md)
- [Training](docs/training.md)

## Quick Start

After [setup](docs/setup.md), run inference with the base model:

```bash
python scripts/inference.py -c configs/nv_raw2insights_mri_base.json -i example -o outputs/example_output_base
```

For multi-GPU inference:

```bash
torchrun --nproc_per_node=8 scripts/inference.py -c configs/nv_raw2insights_mri_base.json -i /path/to/input -o /path/to/output
```

## Citation

If you use NV-Raw2insights-MRI or the SDUM method in your work, please cite:

```bibtex
@article{wang2025sdum,
  title={SDUM: A Scalable Deep Unrolled Model for Universal MRI Reconstruction},
  author={Wang, Puyang and Guo, Pengfei and Chai, Keyi and Zhou, Jinyuan and Xu, Daguang and Jiang, Shanshan},
  journal={arXiv preprint arXiv:2512.17137},
  year={2025}
}
```
Please also cite the [CMRxRecon dataset](https://www.synapse.org/Synapse:syn59814210/wiki/) papers.

## License and Contact

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

NV-Raw2insights-MRI source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

NV-Raw2insights-MRI models are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license).