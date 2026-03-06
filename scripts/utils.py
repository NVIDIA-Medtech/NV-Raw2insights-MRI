# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
import random
import socket
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import scipy
import torch
import torch.distributed as dist
import torchvision
from monai.apps.reconstruction.complex_utils import complex_abs, complex_conj_t, complex_mul_t
from monai.data.fft_utils import fftn_centered, ifftn_centered
from monai.networks.utils import copy_model_state
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer

# Default Hugging Face repo for nv_raw2insights MRI checkpoints. Override with env HF_NV_RAW2INSIGHTS_REPO.
HF_NV_RAW2INSIGHTS_REPO = os.environ.get("HF_NV_RAW2INSIGHTS_REPO", "nvidia/NV-Raw2insights-MRI")

# Map checkpoint filename -> (repo_id, filename_on_hub). Enables auto-download when local path missing.
CHECKPOINT_HF_REGISTRY = {
    "nv_raw2insights_mri_small": (HF_NV_RAW2INSIGHTS_REPO, "nv_raw2insights_mri_small.pt"),
    "nv_raw2insights_mri_base": (HF_NV_RAW2INSIGHTS_REPO, "nv_raw2insights_mri_base.pt"),
    "nv_raw2insights_mri_large": (HF_NV_RAW2INSIGHTS_REPO, "nv_raw2insights_mri_large.pt"),
}

MODALITY_MAPPING = {
    "aorta": "Aorta",
    "aorta_sag": "Aorta",
    "aorta_tra": "Aorta",
    "cine": "Cine",
    "cine_ax": "Cine",
    "cine_ot": "Cine",
    "cine_sax": "Cine",
    "cine_lax": "Cine",
    "cine_lax_2ch": "Cine",
    "cine_lax_3ch": "Cine",
    "cine_lax_4ch": "Cine",
    "cine_lax_r2ch": "Cine",
    "cine_lvot": "Cine",
    "cine_rvot": "Cine",
    "cine_lv": "Cine",
    "t1map": "Mapping",
    "t1map_lax_4ch": "Mapping",
    "t2map": "Mapping",
    "t2smap": "Mapping",
    "t1mappost": "Mapping",  # Assuming T1 map post-contrast is a type of mapping
    "t1rho": "T1rho",
    "lge": "LGE",
    "lge_sax": "LGE",
    "lge_lax": "LGE",
    "lge_lax_2ch": "LGE",
    "lge_lax_3ch": "LGE",
    "lge_lax_4ch": "LGE",
    "perfusion": "Perfusion",
    "t2w": "T2w",
    "blackblood": "BlackBlood",  # As per guidelines
    "flow2d": "Flow2d",  # As per guidelines
    "flow2d_inplane": "Flow2d",  # As per guidelines
    "flow2d_throughplane_m": "Flow2d",  # As per guidelines
    "flow2d_throughplane_d": "Flow2d",  # As per guidelines
    "t1w": "T1w",  # As per guidelines
    "t2w_lax_2ch": "T2w",
    "t2w_lax_4ch": "T2w",
    "tagging": "Tagging",
}

__all__ = [
    "adjust_learning_rate",
    "visualize",
    "save_args_to_file_json",
    "find_free_network_port",
    "is_ddp_enabled",
    "mini_dataloader",
    "sensitivity_map_reduce",
    "sensitivity_map_expand",
    "reshape_complex_to_channel_dim",
    "load_config",
    "resolve_checkpoint_path",
    "reshape_channel_complex_to_last_dim",
    "complex_normalize",
    "MultiEpochsDataLoader",
    "load_net",
    "save_checkpoint",
    "get_acs_image",
    "save_img4ranking",
    "ind2xy",
    "xy2ind",
    "reshape_batch_channel_to_channel_dim",
    "reshape_channel_to_batch_dim",
    "Lookahead",
    "windowed_input",
    "complex_zscore",
    "get_training_set",
    "report_nan_from_any_rank",
    "zero_grad_scalar_fast",
    "MODALITY_MAPPING",
]


def adjust_learning_rate(optimizer, epoch, args, is_resume_first_ten=False):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if is_resume_first_ten:
        lr = 0.0
    else:
        if epoch < args.warmup_epochs:
            lr = args.lr * epoch / args.warmup_epochs
        else:
            if args.lr_schedule == "constant":
                lr = args.lr
            elif args.lr_schedule == "cosine":
                lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
                    1.0 + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.num_epochs - args.warmup_epochs))
                )
            else:
                raise NotImplementedError
    for param_group in optimizer.param_groups:
        if "use_muon" in param_group and param_group["use_muon"]:
            lr = lr * args.muon_scale
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def save_img4ranking(img4ranking, folder_path, file_path):
    os.makedirs(folder_path, exist_ok=True)
    scipy.io.savemat(os.path.join(folder_path, file_path), {"img4ranking": img4ranking})


def visualize(input, output, target, epoch, writer):
    def make_grid(image, normalize=True):
        if normalize:
            image -= image.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values
            image /= image.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values

        grid = torchvision.utils.make_grid(image, nrow=1, pad_value=1)
        return grid

    input, output, target = (
        torch.Tensor(input),
        torch.Tensor(output),
        torch.Tensor(target),
    )

    # See PR: https://github.com/Project-MONAI/MONAI/pull/8407
    output_k = fftn_centered(output, spatial_dims=2, is_complex=(output.shape[-1] == 2))
    input_k = fftn_centered(input, spatial_dims=2, is_complex=(input.shape[-1] == 2))
    target_k = fftn_centered(target, spatial_dims=2, is_complex=(target.shape[-1] == 2))

    output = complex_abs(output).unsqueeze(1) if output.shape[-1] == 2 else output.unsqueeze(1)
    target = complex_abs(target).unsqueeze(1) if target.shape[-1] == 2 else target.unsqueeze(1)
    input = complex_abs(input).unsqueeze(1) if input.shape[-1] == 2 else input.unsqueeze(1)

    error = make_grid(torch.abs(target - output))
    input = make_grid(input)
    target = make_grid(target)
    output = make_grid(output)

    input_k_mag = torch.log10(complex_abs(input_k) + 1e-9).unsqueeze(1)
    target_k_mag = torch.log10(complex_abs(target_k) + 1e-9).unsqueeze(1)
    output_k_mag = torch.log10(complex_abs(output_k) + 1e-9).unsqueeze(1)

    error_k = make_grid(torch.abs(target_k_mag - output_k_mag))
    input_k = make_grid(input_k_mag)
    target_k = make_grid(target_k_mag)
    output_k = make_grid(output_k_mag)

    canvas = torch.cat((input, target, output, error, input_k, target_k, output_k, error_k), -1)

    writer.add_image("Result", canvas, epoch)
    return [input, target, output, error, input_k, target_k, output_k, error_k]


def save_args_to_file_json(args, filename):
    # Convert args dictionary to be JSON serializable
    args_dict = vars(args)
    for key, value in args_dict.items():
        if isinstance(value, Path):
            args_dict[key] = str(value)

    with open(filename, "w") as f:
        json.dump(args_dict, f, indent=4)


class Config:
    def __init__(self, d):
        setattr(self, "data_path_train", None)
        setattr(self, "data_path_val", None)
        setattr(self, "pretrained_csm", None)
        setattr(self, "pretrained_recon", None)
        setattr(self, "pp_z_score_norm", False)
        setattr(self, "fixed_mask_types", None)
        setattr(self, "uniform_input_kspace", False)
        setattr(self, "val_interval", 4)
        setattr(self, "num_samples_per_case", 16)
        setattr(self, "data_aug", True)
        setattr(self, "resume_rng_state", False)
        setattr(self, "do_mapping_shuffle", False)
        setattr(self, "do_center_crop", True)
        setattr(self, "seed", None)
        setattr(self, "lookahead", False)
        setattr(self, "muon", False)
        setattr(self, "muon_scale", 5)
        setattr(self, "enable_onelogger", False)
        setattr(self, "flow", False)
        setattr(self, "balance_data", False)
        setattr(self, "enable_cas_skips", True)
        setattr(self, "finetune_ms", False)
        setattr(self, "adaptive_batch_size", False)
        setattr(self, "constant_input_flow", False)
        setattr(self, "amp", True)
        setattr(self, "acs_lines", 20)
        setattr(self, "pp_norm", True)
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)

        if hasattr(self, "mask_types") and not hasattr(self, "train_mask_types"):
            setattr(self, "train_mask_types", self.mask_types)
            setattr(self, "val_mask_types", self.mask_types)
            delattr(self, "mask_types")

    def to_dict(self) -> dict:
        return {k: v.to_dict() if isinstance(v, Config) else v for k, v in vars(self).items()}


def load_config(file_path: Path):
    """
    Loads JSON data from a file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: A Python dictionary representing the JSON data, or None if an error occurs.
    """
    try:
        with open(file_path, "r") as file:
            config_dict = json.load(file)
            config = Config(config_dict)
            config.data_path_train = (
                [config.data_path_train] if isinstance(config.data_path_train, str) else config.data_path_train
            )
            config.data_path_val = (
                [config.data_path_val] if isinstance(config.data_path_val, str) else config.data_path_val
            )
            return config
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in file: {file_path}: {e}")
        return None


def resolve_checkpoint_path(model_variant: str, model_ckpt: str | Path | None = None) -> Path:
    """
    Resolve checkpoint path: if the file exists, return it; otherwise try to download
    from Hugging Face Hub. Uses CHECKPOINT_HF_REGISTRY and
    HF_NV_RAW2INSIGHTS_REPO (or env HF_NV_RAW2INSIGHTS_REPO).

    Args:
        model_variant: Model variant (e.g. "nv_raw2insights_mri_small", "nv_raw2insights_mri_base",
                "nv_raw2insights_mri_large").
        model_ckpt: Local path (e.g. pretrained_ckpt/nv_raw2insights_mri_small_modified.pt). Default is None.
    Returns:
        Path to the checkpoint file (existing or downloaded).
    """
    if model_ckpt is not None:
        path = Path(model_ckpt)
        if path.exists():
            return path

    if model_variant not in CHECKPOINT_HF_REGISTRY:
        raise ValueError(f"Model variant {model_variant} not found in CHECKPOINT_HF_REGISTRY")
    repo_id, filename = CHECKPOINT_HF_REGISTRY[model_variant]
    try:
        from huggingface_hub import hf_hub_download

        local_path = hf_hub_download(repo_id=repo_id, filename=filename)
        print(f"Downloaded checkpoint from Hugging Face ({repo_id}): {filename} -> {local_path}")
        return Path(local_path)
    except BaseException as e:
        raise ValueError(f"Could not download checkpoint from Hugging Face: {e}") from e


def find_free_network_port() -> int:
    """Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def is_ddp_enabled():
    # Check for PyTorch DDP environment variables that torchrun sets
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    # If world_size > 1 and rank is set, we're in a DDP environment
    return world_size > 1 and rank >= 0 and local_rank >= 0


def mini_dataloader(
    data,
    batch_size,
    drop_last=False,
    pad_last=True,
    shuffle=False,
    max_num_batches=None,
    infinite=False,
):
    if len(data) == 0:
        raise ValueError("`data` must be non-empty.")

    indices = list(range(len(data)))
    batch_count = 0

    while True:  # epoch loop (repeats if `infinite=True`)
        if shuffle:
            random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]

            # Handle short last batch
            if len(batch_indices) < batch_size:
                if drop_last:
                    continue  # skip the short last batch
                if pad_last:
                    # pad with random samples (with replacement) to keep batch size
                    batch_indices.extend(random.choices(indices, k=batch_size - len(batch_indices)))

            yield [data[j] for j in batch_indices], i  # keep original return shape

            batch_count += 1
            if max_num_batches is not None and batch_count >= max_num_batches:
                return  # stop after N batches overall (across epochs)

        # one epoch done
        if not infinite:
            break  # stop after a single pass if not infinite


def sensitivity_map_reduce(img: torch.Tensor, sens_maps: torch.Tensor, k: int = 1, mode: str = "rand") -> torch.Tensor:
    if k != 1:
        res = complex_mul_t(img, complex_conj_t(sens_maps))
        coils = list(range(sens_maps.shape[1]))
        if mode == "rand":
            random.shuffle(coils)
        return torch.stack([res[:, i::k].sum(dim=1) for i in range(k)], dim=1)
    else:
        return complex_mul_t(img, complex_conj_t(sens_maps)).sum(dim=-4, keepdim=True)


def sensitivity_map_expand(img: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    return complex_mul_t(img.sum(dim=-4, keepdim=True), sens_maps)


def reshape_complex_to_channel_dim(x: torch.Tensor, temporal=False) -> torch.Tensor:
    if x.shape[-1] != 2:
        raise ValueError(f"last dim must be 2, but x.shape[-1] is {x.shape[-1]}.")

    if len(x.shape) == 5:  # this is 2D
        b, c, h, w, two = x.shape
        return x.permute(0, 4, 1, 2, 3).contiguous().view(b, 2 * c, h, w)

    elif len(x.shape) == 6:
        if not temporal:  # this is 3D
            b, c, h, w, d, two = x.shape
            return x.permute(0, 5, 1, 2, 3, 4).contiguous().view(b, 2 * c, h, w, d)
        else:
            b, t, c, h, w, two = x.shape
            return x.permute(0, 1, 5, 2, 3, 4).contiguous().view(b, t, 2 * c, h, w)
    else:
        raise ValueError(f"only 2D (B,C,H,W,2) and 3D (B,C,H,W,D,2) data are supported but x has shape {x.shape}")


def reshape_channel_complex_to_last_dim(x: torch.Tensor, temporal=False) -> torch.Tensor:

    if len(x.shape) == 4:  # this is 2D
        if x.shape[1] % 2 != 0:
            raise ValueError(f"channel dimension should be even but ({x.shape[1]}) is odd.")
        b, c2, h, w = x.shape  # c2 means c*2
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1)

    elif len(x.shape) == 5:  # this is 3D
        if not temporal:  # this is 3D
            if x.shape[1] % 2 != 0:
                raise ValueError(f"channel dimension should be even but ({x.shape[1]}) is odd.")
            b, c2, h, w, d = x.shape  # c2 means c*2
            c = c2 // 2
            return x.view(b, 2, c, h, w, d).permute(0, 2, 3, 4, 5, 1)
        else:
            if x.shape[2] % 2 != 0:
                raise ValueError(f"channel dimension should be even but ({x.shape[1]}) is odd.")
            b, t, c2, h, w = x.shape  # c2 means c*2
            c = c2 // 2
            return x.view(b, t, 2, c, h, w).permute(0, 1, 3, 4, 5, 2)
    else:
        raise ValueError(f"only 2D (B,C*2,H,W) and 3D (B,C*2,H,W,D) data are supported but x has shape {x.shape}")


def complex_normalize(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(x.shape) == 5:  # this is 2D
        b, c, h, w, cpx = x.shape
        assert cpx == 2, "last dim must be 2, but cpx is {cpx}".format(cpx=cpx)
        x = x.contiguous().view(b, c * h * w, 2)
        mean = x.mean(dim=1).view(b, 1, 1, 1, 2).expand(b, c, 1, 1, 2).contiguous()
        std = x.std(dim=1, unbiased=False).view(b, 1, 1, 1, 2).expand(b, c, 1, 1, 2).contiguous()
        x = x.view(b, c, h, w, 2)
        return (x - mean) / std, mean, std
    elif len(x.shape) == 6:  # this is 3D
        b, c, h, w, d, cpx = x.shape
        assert cpx == 2, "last dim must be 2, but cpx is {cpx}".format(cpx=cpx)
        x = x.contiguous().view(b, c * h * w * d, 2)
        mean = x.mean(dim=1).view(b, 1, 1, 1, 1, 2).expand(b, c, 1, 1, 1, 2).contiguous()
        std = x.std(dim=1, unbiased=False).view(b, 1, 1, 1, 1, 2).expand(b, c, 1, 1, 1, 2).contiguous()
        x = x.view(b, c, h, w, d, 2)
        return (x - mean) / std, mean, std
    else:
        raise ValueError(f"only 2D (B,C,H,W,2) and 3D (B,C,H,W,D,2) data are supported but x has shape {x.shape}")


def complex_zscore(real_imag: torch.Tensor, dim=None, unbiased: bool = False):
    """
    Z‑score normalize a real‑imag tensor along the given spatial dims,
    preserving phase exactly.

    Params:
      real_imag: Tensor of shape (..., 2) in last dim = (real, imag)
      dim:      tuple of dims to reduce over (e.g. spatial dims). If None,
                will reduce over all dims except the last.
      unbiased: if True, uses Bessel’s correction in variance.
    Returns:
      normed:   same shape as input, real‑imag normalized
      mean:     per‑reduction complex mean (real‑imag stacked in last dim)
      std:      per‑reduction real std = sqrt(E[|x-μ|²])
    """
    # view as complex
    x_c = torch.view_as_complex(real_imag)

    # figure out which dims to reduce: exclude last dim and batch/channel dims
    if dim is None:
        dim = tuple(range(x_c.ndim - 1))  # everything except the imaginary axis
    mean = x_c.mean(dim=dim, keepdim=True)  # complex
    # variance = E[ |x - μ|² ]
    var = ((x_c - mean).abs() ** 2).mean(dim=dim, keepdim=True)
    if unbiased:
        # apply Bessel’s correction: var *= N/(N-1)
        n = 1
        for d in dim:
            n *= x_c.size(d)
        if n > 1:
            var = var * (n / (n - 1))
    std = torch.sqrt(var)  # real
    # z‑score (complex shift & real‑scale)
    z_c = (x_c - mean) / std

    # convert everything back to real‑imag pairs
    z_ri = torch.view_as_real(z_c)
    mean_ri = torch.view_as_real(mean)
    std_r = std.real.unsqueeze(-1)  # keep it real
    return z_ri, mean_ri, std_r


# https://discuss.pytorch.org/t/enumerate-dataloader-slow/87778
class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        if self.batch_sampler is None:
            self.sampler = _RepeatSampler(self.sampler)
        else:
            self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.sampler) if self.batch_sampler is None else len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def load_net(
    net,
    resume_training_ckpt,
    device,
    is_ddp=False,
    find_unused_parameters=False,
    resume_rng_state=False,
):
    """
    Load the Net model.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to load the model on.
        is_ddp (bool): Whether to use distributed data parallel.
        find_unused_parameters (bool): Whether to find unused parameters.
        resume_rng_state (bool): Whether to resume the random number generator state.

    Returns:
        torch.nn.Module: Loaded Net model.
    """

    start_epoch = 0
    start_global_step = 0
    optimizer_state_dict = None
    scheduler_state_dict = None
    scaler_state_dict = None
    best_metric = -1
    best_metric_epoch = -1
    wandb_run_id = None

    if not os.path.exists(resume_training_ckpt):
        print("Training from scratch or pretrained model.")
    else:
        optimizer_state_dict = None
        scheduler_state_dict = None
        scaler_state_dict = None
        start_epoch = 0
        start_global_step = 0

        checkpoint_net = torch.load(resume_training_ckpt, map_location=device, weights_only=False)
        print(f"resume training net from {resume_training_ckpt}.")
        ignore_schduler_opt_state = False
        print(f"resume training net, ignore_schduler_opt_state: {ignore_schduler_opt_state}")

        _, updated_keys, unchanged_keys = copy_model_state(net, checkpoint_net["net_state_dict"])

        print(f"net updated_keys: {len(updated_keys)}")
        print(f"net unchanged_keys: {len(unchanged_keys)}")

        if not ignore_schduler_opt_state:
            try:
                if checkpoint_net["epoch_finished"]:
                    start_epoch = checkpoint_net["epoch"]
                else:
                    start_epoch = checkpoint_net["epoch"] - 1
                start_global_step = checkpoint_net["global_step"]
            except BaseException:
                start_epoch = 0
                start_global_step = 0

            if "optimizer_state_dict" in checkpoint_net:
                print("load optimizer_state_dict from ckpt.")
                optimizer_state_dict = checkpoint_net["optimizer_state_dict"]
            if "scaler_state_dict" in checkpoint_net:
                print("load scaler_state_dict from ckpt.")
                scaler_state_dict = checkpoint_net["scaler_state_dict"]
            if "scheduler_state_dict" in checkpoint_net:
                print("load scheduler_state_dict from ckpt.")
                scheduler_state_dict = checkpoint_net["scheduler_state_dict"]
            if "best_metric" in checkpoint_net:
                best_metric = checkpoint_net["best_metric"]
            if "best_metric_epoch" in checkpoint_net:
                best_metric_epoch = checkpoint_net["best_metric_epoch"]
            if "wandb_run_id" in checkpoint_net:
                wandb_run_id = checkpoint_net["wandb_run_id"]
            if resume_rng_state:
                if "python_rng_state" in checkpoint_net:
                    random.setstate(checkpoint_net["python_rng_state"])
                if "numpy_rng_state" in checkpoint_net:
                    np.random.set_state(checkpoint_net["numpy_rng_state"])
                if "torch_rng_state" in checkpoint_net:
                    torch.random.set_rng_state(checkpoint_net["torch_rng_state"].cpu())
                if "cuda_rng_state" in checkpoint_net:
                    torch.cuda.set_rng_state(checkpoint_net["cuda_rng_state"].cpu())

    if is_ddp and torch.cuda.device_count() > 1:
        # Get device index from device object
        device_index = device.index if isinstance(device, torch.device) else 0

        # Ensure all processes are synchronized before converting to DDP
        dist.barrier(device_ids=[device_index])

        # Convert to SyncBatchNorm first
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

        # Ensure all processes are synchronized before DDP initialization
        dist.barrier(device_ids=[device_index])

        # Initialize DDP
        net = DistributedDataParallel(
            net,
            device_ids=[device],
            find_unused_parameters=find_unused_parameters,
        )

        # Final synchronization after DDP initialization
        dist.barrier(device_ids=[device_index])
    elif torch.cuda.is_available():
        net = net.to(device)

    return (
        net,
        optimizer_state_dict,
        scheduler_state_dict,
        scaler_state_dict,
        start_epoch,
        start_global_step,
        best_metric,
        best_metric_epoch,
        wandb_run_id,
    )


def save_checkpoint(
    epoch: int,
    global_step: int,
    net: torch.nn.Module,
    optimizer,
    lr_scheduler,
    scaler,
    ckpt_folder: str,
    best_metric,
    best_metric_epoch,
    model_filename: str,
    epoch_finished: bool = True,
    is_ddp: bool = False,
    wandb_run_id: str = None,
    python_rng_state=None,
    numpy_rng_state=None,
    torch_rng_state=None,
    cuda_rng_state=None,
) -> None:
    """
    Save checkpoint.

    Args:
        epoch (int): Current epoch number.
        net (torch.nn.Module): net model.
        ckpt_folder (str): Checkpoint folder path.
        model_filename (str): model filename.
        epoch_finished (bool): epoch finished
    """
    net_state_dict = net.module.state_dict() if is_ddp else net.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    scaler_state_dict = scaler.state_dict()
    if lr_scheduler is not None:
        scheduler_state_dict = lr_scheduler.state_dict()
    else:
        scheduler_state_dict = None
    torch.save(
        {
            "epoch": epoch + 1,
            "global_step": global_step + 1,
            "net_state_dict": net_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "scheduler_state_dict": scheduler_state_dict,
            "scaler_state_dict": scaler_state_dict,
            "best_metric": best_metric,
            "best_metric_epoch": best_metric_epoch,
            "epoch_finished": epoch_finished,
            "wandb_run_id": wandb_run_id,
            "python_rng_state": python_rng_state,
            "numpy_rng_state": numpy_rng_state,
            "torch_rng_state": torch_rng_state,
            "cuda_rng_state": cuda_rng_state,
        },
        f"{ckpt_folder}/{model_filename}",
    )
    print(f"Save ckpt to {ckpt_folder}/{model_filename}.")


def get_acs_region(mask: torch.Tensor) -> tuple[int, int, int, int]:
    """
    Given a 2D binary mask,
    finds the maximum acs region containing only positive (nonzero)
    elements that includes the center of the mask.

    Returns:
        A tuple (left, right, top, bottom) where:
          - left and right are the minimum and maximum x indices,
          - top and bottom are the minimum and maximum y indices.
    """
    h, w = mask.shape[-3], mask.shape[-2]
    # Determine the center coordinates.
    cy, cx = h // 2, w // 2

    # Ensure the center is positive.
    if not torch.all(mask[..., cy, cx, :]):
        raise ValueError("The center of the mask is not positive.")

    # Initialize the bounding box to the center pixel.
    left, right = cx, cx
    top, bottom = cy, cy

    # Try to expand the box outward while the newly included row/column is all positive.
    expanded = True
    while expanded:
        expanded = False

        # Attempt to expand to the left.
        if left > 0 and torch.all(mask[..., top : bottom + 1, left - 1, :] > 0):
            left -= 1
            expanded = True

        # Attempt to expand to the right.
        if right < w - 1 and torch.all(mask[..., top : bottom + 1, right + 1, :] > 0):
            right += 1
            expanded = True

        # Attempt to expand upward.
        if top > 0 and torch.all(mask[..., top - 1, left : right + 1, :] > 0):
            top -= 1
            expanded = True

        # Attempt to expand downward.
        if bottom < h - 1 and torch.all(mask[..., bottom + 1, left : right + 1, :] > 0):
            bottom += 1
            expanded = True

    return top, bottom, left, right


def get_acs_image(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    top, bottom, left, right = get_acs_region(mask)
    x_k = fftn_centered(x, spatial_dims=2, is_complex=True)
    num_low_freqs_x = right - left  # size of the fully-sampled center
    num_low_freqs_y = bottom - top

    # take out the fully-sampled region and set the rest of the data to zero
    x_k_acs = torch.zeros_like(x_k)
    start_x = (mask.shape[-2] - num_low_freqs_x + 1) // 2  # this marks the start of center extraction
    start_y = (mask.shape[-3] - num_low_freqs_y + 1) // 2

    x_k_acs[..., start_y : start_y + num_low_freqs_y, start_x : start_x + num_low_freqs_x, :] = x_k[
        ..., start_y : start_y + num_low_freqs_y, start_x : start_x + num_low_freqs_x, :
    ]
    x_acs = ifftn_centered(x_k_acs, spatial_dims=2, is_complex=True)
    return x_acs


def ind2xy(idx, num_frames, num_slices):
    """
    Convert a flattened frame×slice index to (slice_idx, frame_idx).
    """
    frame_idx = idx // num_slices
    slice_idx = idx % num_slices
    return slice_idx, frame_idx


def xy2ind(slice_idx, frame_idx, num_frames, num_slices):
    """
    Convert (slice_idx, frame_idx) back to a flattened index.
    """
    return frame_idx * num_slices + slice_idx


def windowed_input(input, micro_b, final_shape, num_frames, slice_window_single_frame=False):
    """
    Given:
      input       – a Tensor of shape [N, …] where N == total_frames * total_slices
      micro_b     – list or 1D Tensor of length B with flattened frame×slice indices
      final_shape – the shape of the volume, where
                     final_shape[-5] == total number of frames,
                     final_shape[-4] == total number of slices
      num_frames  – desired window size (can be odd or even)
    Returns:
      inp – Tensor of shape [B, num_frames, …],
            where for each b, we’ve gathered around the center frame (or slice, if single-frame input).
      window_idx – LongTensor of shape [B, num_frames] containing the gathered indices
    """
    total_frames = final_shape[-5]
    total_slices = final_shape[-4]
    half = num_frames // 2
    # symmetric offsets: e.g. num_frames=4 → offsets = [-2,-1,0,1]
    offsets = range(-half, num_frames - half)

    single_frame = total_frames == 1
    window_idx = []

    for idx in micro_b:
        slice_i, frame_i = ind2xy(idx, total_frames, total_slices)

        if single_frame and slice_window_single_frame:
            # sliding window in the slice dimension
            centers = slice_i
            neighbors = [(centers + off) % total_slices for off in offsets]
            idxs = [xy2ind(s, frame_i, total_frames, total_slices) for s in neighbors]
        else:
            # sliding window in the frame dimension (original behavior)
            centers = frame_i
            neighbors = [(centers + off) % total_frames for off in offsets]
            idxs = [xy2ind(slice_i, f, total_frames, total_slices) for f in neighbors]

        window_idx.append(idxs)

    # make a LongTensor [B, num_frames]
    window_idx = torch.tensor(window_idx, dtype=torch.long, device=input.device)

    # gather: shape [B, num_frames, …rest of input.shape…]
    inp = torch.Tensor(input[window_idx])
    return inp, window_idx


def reshape_channel_to_batch_dim(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """
    Combines batch and channel dimensions.

    Args:
        x: input of shape (B,C,H,W,2) for 2D data or (B,C,H,W,D,2) for 3D data

    Returns:
        A tuple containing:
            (1) output of shape (B*C,1,...)
            (2) batch size
    """

    if len(x.shape) == 5:  # this is 2D
        b, c, h, w, two = x.shape
        return x.contiguous().view(b * c, 1, h, w, two), b

    elif len(x.shape) == 6:  # this is 3D
        b, c, h, w, d, two = x.shape
        return x.contiguous().view(b * c, 1, h, w, d, two), b

    else:
        raise ValueError(f"only 2D (B,C,H,W,2) and 3D (B,C,H,W,D,2) data are supported but x has shape {x.shape}")


def reshape_batch_channel_to_channel_dim(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Detaches batch and channel dimensions.

    Args:
        x: input of shape (B*C,1,H,W,2) for 2D data or (B*C,1,H,W,D,2) for 3D data
        batch_size: batch size

    Returns:
        output of shape (B,C,...)
    """
    if len(x.shape) == 4:  # this is 2D real
        bc, chan, h, w = x.shape  # bc represents B*C
        c = bc // batch_size
        return x.view(batch_size, c, chan, h, w)
    if len(x.shape) == 5:  # this is 2D
        bc, one, h, w, two = x.shape  # bc represents B*C
        c = bc // batch_size
        return x.view(batch_size, c, h, w, two)

    elif len(x.shape) == 6:  # this is 3D
        bc, one, h, w, d, two = x.shape  # bc represents B*C
        c = bc // batch_size
        return x.view(batch_size, c, h, w, d, two)

    else:
        raise ValueError(f"only 2D (B*C,1,H,W,2) and 3D (B*C,1,H,W,D,2) data are supported but x has shape {x.shape}")


class Lookahead(Optimizer):
    """
    PyTorch implementation of the lookahead wrapper.
    Lookahead Optimizer: https://arxiv.org/abs/1907.08610
    """

    def __init__(self, optimizer, alpha=0.5, k=6, pullback_momentum="none"):
        """
        :param optimizer:inner optimizer
        :param k (int): number of lookahead steps
        :param alpha(float): linear interpolation factor. 1.0 recovers the inner optimizer.
        :param pullback_momentum (str): change to inner optimizer momentum on interpolation update
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid slow update rate: {alpha}")
        if not 1 <= k:
            raise ValueError(f"Invalid lookahead steps: {k}")
        self.optimizer = optimizer
        self.param_groups = self.optimizer.param_groups
        self.alpha = alpha
        self.k = k
        self.step_counter = 0
        assert pullback_momentum in ["reset", "pullback", "none"]
        self.pullback_momentum = pullback_momentum
        self.state = defaultdict(dict)

        # Cache the current optimizer parameters
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["cached_params"] = torch.zeros_like(p.data)
                param_state["cached_params"].copy_(p.data)

    def __getstate__(self):
        return {
            "state": self.state,
            "optimizer": self.optimizer,
            "alpha": self.alpha,
            "step_counter": self.step_counter,
            "k": self.k,
            "pullback_momentum": self.pullback_momentum,
        }

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def _backup_and_load_cache(self):
        """Useful for performing evaluation on the slow weights (which typically generalize better)"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["backup_params"] = torch.zeros_like(p.data)
                param_state["backup_params"].copy_(p.data)
                p.data.copy_(param_state["cached_params"])

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                p.data.copy_(param_state["backup_params"])
                del param_state["backup_params"]

    def step(self, closure=None):
        """Performs a single Lookahead optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = self.optimizer.step(closure)
        self.step_counter += 1

        if self.step_counter >= self.k:
            self.step_counter = 0
            # Lookahead and cache the current optimizer parameters
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    param_state = self.state[p]
                    p.data.mul_(self.alpha).add_(1.0 - self.alpha, param_state["cached_params"])  # crucial line
                    param_state["cached_params"].copy_(p.data)
                    if self.pullback_momentum == "pullback":
                        internal_momentum = self.optimizer.state[p]["momentum_buffer"]
                        self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.alpha).add_(
                            1.0 - self.alpha, param_state["cached_mom"]
                        )
                        param_state["cached_mom"] = self.optimizer.state[p]["momentum_buffer"]
                    elif self.pullback_momentum == "reset":
                        self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)

        return loss


def get_training_set(
    filenames: List[str | os.PathLike],
    modality_mapping: Dict[str, str],
    seed: Optional[int] = None,
    balanced: bool = True,
    shuffle: bool = False,
    cmrxrecon_23_data_ratio: float = 1.0,
    acq_types: List[str] = None,
) -> List[str]:
    """
    Given a list of filenames and a mapping from file-suffix to modality,
    returns a balanced subset of filenames (undersampling majority classes)
    so each modality has the same number of samples (min class size).

    Args:
        filenames: List of file paths or names.
        modality_mapping: dict mapping suffix keys (e.g. 'cine_sax') to modality names.
        seed: Optional random seed for reproducibility.

    Returns:
        A list of filenames forming a balanced training set.
    """
    # Prepare
    if seed is not None:
        random.seed(seed)
    # Sort suffixes by length to match the longest first
    filenames = sorted(filenames)

    if acq_types is not None:
        acq_types = set([acq.lower() for acq in acq_types])
        for k in [k for k, v in modality_mapping.items() if v.lower() not in acq_types]:
            del modality_mapping[k]
    suffixes = sorted(modality_mapping.keys(), key=len, reverse=True)

    # Group files by modality
    groups = defaultdict(list)
    for fname in filenames:
        stem, _ = os.path.splitext(os.path.basename(fname))
        modality = None
        for suf in suffixes:
            if stem.lower().endswith(suf):
                modality = modality_mapping[suf]
                break
        if modality:
            if "CMRxRecon2023_Data" in str(fname):
                if random.random() > cmrxrecon_23_data_ratio:
                    continue
            groups[modality].append(fname)
        else:
            # if you prefer to catch unmapped files:
            # raise ValueError(f"No modality mapping for file '{fname}'")
            continue

    # find smallest class size
    if balanced:
        num_files = sum(len(v) for v in groups.values())
        num_modality = 9
        if not groups:
            return []
        n_count_per_modality = num_files // num_modality

        # sample min_count from each group
        balanced = []
        for _, files in groups.items():
            while len(files) < n_count_per_modality:
                files.extend(files)
            files = random.sample(files, n_count_per_modality)
            balanced.extend(files)
        # optional: shuffle final list
        if shuffle:
            random.shuffle(balanced)
        return balanced
    else:
        all = []
        for _, files in groups.items():
            all.extend(files)
        if shuffle:
            random.shuffle(all)
        return all


def report_nan_from_any_rank(loss, file_name, micro_b, is_ddp=False):
    if is_ddp:
        device = loss.device
        rank = dist.get_rank()
        world = dist.get_world_size()

        local_bad = loss.isnan()
        # Step 1: detect if any rank is bad
        any_bad = torch.tensor([int(local_bad)], device=device, dtype=torch.int32)
        dist.all_reduce(any_bad, op=dist.ReduceOp.MAX)
        any_bad = bool(any_bad.item())

        # Step 2: optionally collect details to rank 0
        if any_bad:
            # gather arbitrary python objects (PyTorch 1.8+)
            mine = {"rank": rank, "file": file_name, "micro_b": micro_b} if local_bad else None
            gather_list = [None] * world if rank == 0 else None
            dist.gather_object(mine, gather_list, dst=0)

            if rank == 0:
                offenders = [o for o in gather_list if o is not None]
                ranks = [o["rank"] for o in offenders]
                print(f"NaN/Inf loss detected on ranks {ranks}. Skipping this step.")
                for o in offenders:
                    print(f"  • rank {o['rank']}: file={o['file']} slice={o['micro_b']}")
    else:
        any_bad = loss.isnan()
        if any_bad:
            print(f"NaN Loss detected for {file_name} at slice {micro_b}. Skipping this step.")

    return any_bad


def zero_grad_scalar_fast(model):
    # touches exactly one element from each *trainable* parameter
    s = 0.0
    for p in model.parameters():
        if p.requires_grad:
            s = s + p.view(-1)[0]  # one element per param
    return s * 0.0


if __name__ == "__main__":
    # Suppose these are your training files:
    train_files = [file for file in Path("data/train").iterdir()]

    balanced = get_training_set(train_files, MODALITY_MAPPING, balanced=False)
    print("Balanced subset:")
    for f in balanced:
        print(f)
