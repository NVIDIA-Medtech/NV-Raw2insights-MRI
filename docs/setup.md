# Setup Guide

- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Downloading Checkpoints](#downloading-checkpoints)

## System Requirements

- **NVIDIA GPUs** with sufficient VRAM for the chosen model variant (see [Inference](inference.md) for memory guidance)
- **NVIDIA driver** compatible with CUDA 12.4 (used by PyTorch wheel)
- **Linux** x86-64 (recommended); other platforms may work with appropriate PyTorch/CUDA installs
- **Python** 3.10+ recommended

## Installation

### 1. Install [git lfs](https://git-lfs.com/):

```bash
sudo apt install git-lfs
git lfs install
```

### 2. Clone the repository

```bash
git clone https://github.com/NVIDIA-Medtech/NV-Raw2insights-MRI
cd NV-Raw2insights-MRI
git lfs pull
```

### 3. Create an environment

```bash
conda create --name nv-raw2insights-mri python=3.12
conda activate nv-raw2insights-mri
pip install --upgrade pip setuptools wheel
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Optional: dataset preparation

For training or inference on CMRxRecon-style data, place or symlink datasets as expected by the config (e.g. `dataset/CMRxRecon2025/ChallengeDataTrain`, `ChallengeDataValR1`). Use `scripts/create_cmrxrecon_dataset.py` if you need to build prepare dataset for `.mat` files. See [Training](training.md) for data preparation.

## Downloading Checkpoints

Pre-trained NV-Raw2insights-MRI checkpoints are hosted on Hugging Face and are **downloaded automatically** when you run inference or training without specifying a local checkpoint.

### Optional: manual setup

1. Get a [Hugging Face Access Token](https://huggingface.co/settings/tokens) with `Read` permission (required only if the model is gated).
2. Install the Hugging Face CLI: `pip install -U "huggingface_hub[cli]"`
3. Log in: `hf auth login`
4. Accept any model license on the [NV-Raw2insights-MRI model page](https://huggingface.co/nvidia/NV-Raw2insights-MRI) if applicable.

Checkpoints are cached under the Hugging Face cache directory. To change it, set the `HF_HOME` environment variable:

```bash
export HF_HOME=/path/to/your/hf/cache
```

Then run inference or training as usual; the correct checkpoint for the config’s `model_variant` will be resolved automatically.
