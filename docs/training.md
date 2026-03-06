# Training Guide

This guide describes how to train or fine-tune **NV-Raw2insights-MRI** (SDUM) for CMR reconstruction.

## Prerequisites

### 1. Environment and data

- Follow the [Setup guide](setup.md) for environment setup and dependency installation.
- Prepare your training and validation data (e.g. CMRxRecon-style directory layout with JSON descriptors and `.mat` files). Point the config’s `data_path_train` and `data_path_val` to these directories.

### 2. Hugging Face (for pre-trained checkpoints)

If you start from a pre-trained NV-Raw2insights-MRI checkpoint, it will be downloaded automatically when not present locally. For gated or private repos:

```bash
hf auth login
export HF_HOME=/path/to/your/hf/cache   # optional
```

### 3. Dataset preparation

For **CMRxRecon**-style data (raw kspace and mask `.mat` files in a challenge directory layout), use `scripts/create_cmrxrecon_dataset.py` to build the JSON descriptors that training expects. The script scans `MultiCoil`/`AccFactor` trees under the source and mask directories, matches kspace files to their masks, and writes one JSON per case with `"kspace"` and `"mask"` paths. The resulting `output_dir` can be used as `data_path_train` or `data_path_val` in your config.

### Arguments

| Argument | Description |
|----------|-------------|
| `--source_dir` | Root directory containing kspace `.mat` files (e.g. path to `ChallengeData` or validation task folder). |
| `--mask_dir`   | Root directory containing mask `.mat` files; typically the same as the challenge data root (e.g. `path/to/ChallengeData`). |
| `--output_dir` | Directory where JSON descriptor files will be written. |
| `--training_set` | If set, processes the **training** set (`TrainingSet`); otherwise processes the **validation** set (`ValidationSet`). |

### Example: [CMRxRecon 2025](https://www.synapse.org/Synapse:syn59814210/wiki/)

**Training set** — output for training data (e.g. `dataset/CMRxRecon2025/ChallengeDataTrain`):

```bash
python scripts/create_cmrxrecon_dataset.py \
  --source_dir /path/to/CMRxRecon2025_Data/ChallengeData \
  --mask_dir /path/to/CMRxRecon2025_Data/ChallengeData \
  --output_dir dataset/CMRxRecon2025/ChallengeDataTrain \
  --training_set
```

**Validation set** — output for validation (e.g. `dataset/CMRxRecon2025/ChallengeDataValR1`):

```bash
python scripts/create_cmrxrecon_dataset.py \
  --source_dir /path/to/CMRxRecon2025_Data/ChallengeDataValSet/TaskR1 \
  --mask_dir /path/to/CMRxRecon2025_Data/ChallengeDataValSet/TaskR1 \
  --output_dir dataset/CMRxRecon2025/ChallengeDataValR1
```

Then set `data_path_train` and `data_path_val` in your training config to these `output_dir` paths (e.g. `dataset/CMRxRecon2025/ChallengeDataTrain` and `dataset/CMRxRecon2025/ChallengeDataValR1`).

### JSON format (ChallengeDataTrain)

The `output_dir` (e.g. `dataset/CMRxRecon2025/ChallengeDataTrain`) contains one `.json` file per case. Each JSON describes a single kspace acquisition and its associated mask(s):

| Key | Type | Description |
|-----|------|-------------|
| `kspace` | string | Path to the kspace `.mat` file (multi-coil raw data). |
| `mask` | array of strings | Paths to the corresponding undersampling mask `.mat` file(s). There can be multiple entries when several mask types (e.g. Uniform, Gaussian, Radial) exist for the same case |

Paths may be absolute or relative. The training pipeline reads these keys to load kspace and mask and to derive acquisition type and acceleration from the filenames.

**Example** — one case with multiple masks (one kspace, several mask types/accelerations):

```json
{
    "kspace": "/path/to/ChallengeData/.../P001/cine_lax_3ch.mat",
    "mask": [
        "/path/to/ChallengeData/.../P001/cine_lax_3ch_mask_ktGaussian24.mat",
        "/path/to/ChallengeData/.../P001/cine_lax_3ch_mask_Uniform16.mat",
        "/path/to/ChallengeData/.../P001/cine_lax_3ch_mask_Uniform8.mat"
    ]
}
```

### 4. Training output directory

Checkpoints and logs are written to the path specified in the config (e.g. `exp_dir` and `exp`). Ensure that path has sufficient disk space. You can override or set a global root via environment variables if your training code supports it.

## Weights & Biases (W&B) logging

Training uses **Weights & Biases** for experiment tracking when enabled. Options:

### Enable W&B

1. Create an account at [wandb.ai](https://wandb.ai) and get your API key from [https://wandb.ai/authorize](https://wandb.ai/authorize).
2. Set the API key:

   ```bash
   export WANDB_API_KEY=your_api_key_here
   ```

3. Run training as in [Running training](#running-training) below. The config’s `enable_onelogger` (and related flags) control whether W&B is used.

### Disable W&B

If the code supports a “disabled” mode (e.g. a config flag or environment variable), set it so that training does not attempt to log to W&B. For example, some setups respect `WANDB_MODE=disabled` or a config option like `job.wandb_mode=disabled`. Check `scripts/train.py` and your config for the exact knob.

## Running training

### Multi-node, multi-GPU (recommended)

From the repository root:

```bash
# Set the number of GPUs per node
NUM_GPUS_PER_NODE=8
# Set the number of nodes
NUM_NODES=1
torchrun --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES cripts/train.py --config configs/nv_raw2insights_mri_base.json
```

- `--config`: Path to the training config JSON (e.g. `nv_raw2insights_mri_base`, `nv_raw2insights_mri_small`, `nv_raw2insights_mri_large`).
- `-d` / `--debug`: Optional debug mode (e.g. deterministic seeds).
- `--val`: Optional; run validation only when supported by the script.

Adjust `--nproc_per_node` to the number of GPUs on the node. For multi-node, set `--nnodes` and ensure `MASTER_ADDR` and `MASTER_PORT` are set (the script may auto-select a port if `MASTER_PORT` is unset).

### Single-GPU

The training script supports non-distributed mode (e.g. when `torchrun` is not used), you can run:

```bash
python scripts/train.py --config configs/nv_raw2insights_mri_base.json
```

Check the config for `ddp` and any batch-size or world-size assumptions.

## Config overview

Training configs (e.g. `configs/nv_raw2insights_mri_base.json`) define:

- **Data**: `data_path_train`, `data_path_val`, `dataset`, `train_mask_types`, `val_mask_types`, `fixed_mask_types`, `acq_types`, `accelerations`, etc.
- **Model**: `model_variant`, `model_type`, `num_cascades`, `num_frames`, channels, blocks, and other architecture fields.
- **Optimization**: `num_epochs`, `lr`, `weight_decay`, `lr_schedule`, `warmup_epochs`, `min_lr`, `batch_size`, `amp`.
- **Loss**: `loss_type` (e.g. SSIM).
- **Logging and checkpointing**: `exp_dir`, `exp`, `model_filename`, `val_interval`, W&B-related flags.

The config also specifies which pre-trained checkpoint to load (via `model_variant`); the training code resolves it using the same Hugging Face registry as inference (see [Setup](setup.md)).

## Checkpointing

- Training saves checkpoints under the experiment directory (e.g. `exp/nv_raw2insights_mri_base_posttrain/`). Exact naming (e.g. `model_filename` or step/epoch-based) is defined in the config and `scripts/train.py`.
- To **resume** from a run, the train script supports auto-resume: it will detect existing checkpoints in the experiment directory and continue from the latest one when you start training again with the same config.

## Example

Base model, 8 GPUs, one node:

```bash
torchrun --nproc_per_node=8 --nnodes=1 scripts/train.py --config configs/nv_raw2insights_mri_base.json
```

Ensure `data_path_train` and `data_path_val` in the config point to your dataset paths (e.g. `dataset/CMRxRecon2025/ChallengeDataTrain` and `ChallengeDataValR1`).
