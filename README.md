# NV-Raw2Insights-MRI

[![License](https://img.shields.io/badge/Code-Apache%202.0-blue.svg)](LICENSE)
[![Weights](https://img.shields.io/badge/Weights-NVIDIA%20Open%20Model-green.svg)](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow.svg)](https://huggingface.co/nvidia/NV-Raw2Insights-MRI)
[![Paper](https://img.shields.io/badge/arXiv-2512.17137-red.svg)](https://arxiv.org/abs/2512.17137)

Universal MRI reconstruction from undersampled k-space. A single model handles diverse protocols, anatomies, contrasts, and acceleration factors without task-specific fine-tuning.

<p align="center">
<img width="900" alt="NV-Raw2Insights-MRI" src="https://github.com/user-attachments/assets/27ce9dea-c592-4dd5-b984-38542e1cf8e6" />
</p>

## Overview

NV-Raw2Insights-MRI is built on the Scalable Deep Unrolled Model (SDUM) framework. It combines a Restormer-based cascaded unrolled architecture with learned coil sensitivity estimation, sampling-aware weighted data consistency, and universal conditioning on protocol metadata. Trained on heterogeneous data from CMRxRecon2024, CMRxRecon2025, and fastMRI brain datasets, a single model achieves state-of-the-art results across cardiac, brain, and knee MRI reconstruction.

This project was conducted by NVIDIA in collaboration with the [CMRxRecon Team](https://github.com/CmrxRecon), [Fudan University](https://hupi.fudan.edu.cn/en/), and [Johns Hopkins University](https://profiles.hopkinsmedicine.org/provider/shanshan-jiang/2777746).

## News

- **[March 2026]** — Released NV-Raw2Insights-MRI as part of the NVIDIA MedTech Open Models
- **[February 2026]** — Achieved 1st place across all four tracks in the [CMRxRecon2025 Challenge](https://www.synapse.org/Synapse:syn59814210/wiki/634966) without task-specific fine-tuning

## Model Variants

| Model | Cascades | Parameters | HuggingFace | License |
|-------|:--------:|:----------:|-------------|---------|
| [NV-Raw2Insights-MRI-Small](https://huggingface.co/nvidia/NV-Raw2Insights-MRI) | 6 | 230M | [Download](https://huggingface.co/nvidia/NV-Raw2Insights-MRI) | [NVIDIA Open Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |
| [NV-Raw2Insights-MRI-Base](https://huggingface.co/nvidia/NV-Raw2Insights-MRI) | 18 | 760M | [Download](https://huggingface.co/nvidia/NV-Raw2Insights-MRI) | [NVIDIA Open Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |
| [NV-Raw2Insights-MRI-Large](https://huggingface.co/nvidia/NV-Raw2Insights-MRI) | 34 | 1.4B | [Download](https://huggingface.co/nvidia/NV-Raw2Insights-MRI) | [NVIDIA Open Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |

Checkpoints are automatically downloaded from [HuggingFace](https://huggingface.co/nvidia/NV-Raw2Insights-MRI) when not provided locally.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Inference

```bash
python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i example \
  -o outputs/example_output_base
```

For multi-GPU inference:

```bash
torchrun --nproc_per_node=8 scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i /path/to/input \
  -o /path/to/output
```

## Documentation

| Guide | Description |
|-------|-------------|
| [Setup](docs/setup.md) | Full installation guide |
| [Inference](docs/inference.md) | Inference options, configs, multi-GPU |
| [Training](docs/training.md) | Training and fine-tuning guide |

## Performance

### Model Scaling (PSNR vs cascade depth)

| Cascades (T) | PSNR (dB) | Parameters |
|:------------:|:---------:|:----------:|
| 1 | 28.73 | 42M |
| 3 | 30.21 | 126M |
| 6 | 32.09 | 253M |
| 10 | 32.54 | 422M |
| 18 | 33.18 | 759M |

### Inference Compute (per slice, NVIDIA H100, T=18)

| Input Size | Time (s) | Memory (GB) |
|:----------:|:--------:|:-----------:|
| 128x128 | 0.32 | 4.78 |
| 256x256 | 1.03 | 6.07 |
| 256x512 | 2.06 | 7.98 |
| 328x512 | 2.67 | 9.26 |
| 328x640 | 3.30 | 9.62 |
| 328x768 | 3.97 | 10.83 |

## License

| Component | License |
|-----------|---------|
| Source code | [Apache 2.0](LICENSE) |
| Model weights | [NVIDIA Open Model License](LICENSE.weights) |

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

## Citation

```bibtex
@article{wang2025sdum,
  title={SDUM: A Scalable Deep Unrolled Model for Universal MRI Reconstruction},
  author={Wang, Puyang and Guo, Pengfei and Chai, Keyi and Zhou, Jinyuan and Xu, Daguang and Jiang, Shanshan},
  journal={arXiv preprint arXiv:2512.17137},
  year={2025}
}
```

Please also cite the [CMRxRecon dataset](https://www.synapse.org/Synapse:syn59814210/wiki/) papers.

## Resources

- [SDUM Paper](https://arxiv.org/abs/2512.17137) — arXiv
- [HuggingFace Model](https://huggingface.co/nvidia/NV-Raw2Insights-MRI) — Weights and model card
- [CMRxRecon2025 Challenge](https://www.synapse.org/Synapse:syn59814210/wiki/634966) — Benchmark
