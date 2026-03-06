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

import argparse
import logging
import os
import sys
import time
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import tqdm
from models.latent_recon import create_mri_recon_model
from monai.apps.reconstruction.complex_utils import complex_abs, convert_to_tensor_complex
from monai.apps.reconstruction.transforms.dictionary import ExtractDataKeyFromMetaKeyd
from monai.data import Dataset, partition_dataset
from monai.data.fft_utils import fftn_centered, ifftn_centered
from monai.transforms import Compose, EnsureTyped, Identityd, Lambdad, LoadImaged, ResizeWithPadOrCropd
from monai.utils import set_determinism
from mri_data.data_utils import crop_k_space, get_reader, postprocess_mri_recon, rearrange_mri_data
from torch.amp import autocast
from torch.distributed.elastic.multiprocessing.errors import record
from transforms import *
from utils import *

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

warnings.filterwarnings("ignore")


@record
def infer(args):
    if args.ddp:
        # initialize the distributed training process, every GPU runs in a process
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # Set device before initializing process group
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

        # Initialize process group with timeout and device mapping
        timeout = timedelta(seconds=1800)  # 30 minutes timeout

        # Initialize process group with explicit device mapping
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=timeout,
        )

        # Ensure all processes are synchronized after initialization
        dist.barrier(device_ids=[local_rank])
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rank = 0
        world_size = 1

    if args.seed is not None:
        set_determinism(seed=args.seed)
    if args.debug is True:
        set_determinism(seed=0)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    if rank != 0:
        f = open(os.devnull, "w")
        sys.stdout = sys.stderr = f
    Path(args.output_path).mkdir(parents=True, exist_ok=True)  # create output directory to store model checkpoints

    # Add barrier to ensure rank 1 finishes wandb init before rank 0 starts
    if args.ddp:
        dist.barrier()

    test_files = [file for file in Path(args.data_path_test).iterdir() if str(file).endswith(".json")]
    print(f"#Total test files before filtering: {len(test_files)}")
    # filter out already processed files
    test_files = [
        f
        for f in test_files
        if not (Path(args.output_path) / "val_img4ranking" / f.name.replace(".json", ".mat")).exists()
    ]
    print(f"#Total test files after filtering: {len(test_files)}")
    test_files = [dict([("kspace", test_files[i])]) for i in range(len(test_files))]
    print(f"#Test files: {len(test_files)}")
    test_files = partition_dataset(data=test_files, num_partitions=world_size, shuffle=False)[rank]

    # debug
    if args.debug:
        test_files = test_files[:1]

    # create the model
    model = create_mri_recon_model(args).to(device)
    try:
        args.is_multi_coil = model.use_csm or model.use_latent_csm
    except BaseException:
        args.is_multi_coil = True

    # Auto resume
    # Load the model, optimizer, and scheduler
    (
        model,
        optimizer_state_dict,
        scheduler_state_dict,
        scaler_state_dict,
        start_epoch,
        start_global_step,
        best_metric,
        best_metric_epoch,
        wandb_run_id,
    ) = load_net(
        model,
        args.model_ckpt,
        device,
        is_ddp=args.ddp,
        resume_rng_state=args.resume_rng_state,
    )
    model = torch.compile(model) if args.uniform_input_kspace else model
    print(f"#model_params: {np.sum([len(p.flatten()) for p in model.parameters()]) * 1.0e-6:.2f}M")

    test_transforms = Compose(
        [
            LoadImaged(
                keys=["kspace"],
                reader=get_reader(args, is_testing=True),
                image_only=False,
                dtype=np.complex64,
            ),
            # user can also add other random transforms but remember to disable randomness for val_transforms
            ExtractDataKeyFromMetaKeyd(keys=["mask", "acquisition"], meta_key="kspace_meta_dict"),
            KspaceMaskd(
                keys=["kspace"],
                mask_types=(["fixed"] if not hasattr(args, "val_mask_types") else args.val_mask_types),
                center_fractions=([1.0] if not hasattr(args, "center_fractions") else args.center_fractions),
                accelerations=([0.0] if not hasattr(args, "accelerations") else args.accelerations),
                spatial_dims=2,
                is_complex=True,
            ),
            Lambdad(keys=["kspace"], func=lambda x: convert_to_tensor_complex(x)),
            (
                ResizeWithPadOrCropd(
                    keys=["kspace", "mask", "kspace_masked"],
                    spatial_size=[
                        -1,
                        -1,
                        args.uniform_input_kspace[0],
                        args.uniform_input_kspace[1],
                        2,
                    ],
                )
                if args.uniform_input_kspace
                else Identityd(keys=["kspace"])
            ),
            EnsureTyped(keys=["kspace", "kspace_masked", "mask"]),
            Lambdad(
                keys=["kspace", "kspace_masked"],
                overwrite=["kspace_ifft", "kspace_masked_ifft"],
                func=lambda x: ifftn_centered(x, spatial_dims=2, is_complex=True),
            ),
            RearrangeAndNormalizeMRI(keys=["kspace_masked_ifft", "kspace_ifft", "mask"], args=args),
        ]
    )

    test_ds = Dataset(data=test_files, transform=test_transforms)
    test_loader = MultiEpochsDataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    args.model_structure = str(model).split("\n")
    save_args_to_file_json(args, os.path.join(args.output_path, "config.json"))

    # Test
    model.eval()
    with torch.no_grad():
        tic_val = time.time()
        for test_data in tqdm.tqdm(test_loader):
            (
                input,
                mask,
                mask_type,
                acc_factor,
                acq_type,
                mean,
                std,
                temporal_shuffle,
                file_name,
                final_shape,
            ) = (
                test_data["kspace_masked_ifft"][0],
                test_data["mask"][0],
                test_data["mask_type"][0],
                test_data["acc_factor"][0],
                test_data["acquisition"][0],
                test_data["mean"][0],
                test_data["std"][0],
                test_data["temporal_shuffle"][0],
                test_data["kspace_meta_dict"]["filename"][0],
                test_data["kspace_meta_dict"]["shape"][0],
            )
            if args.debug:
                print(file_name, mask_type, acc_factor, acq_type)
            final_shape = [int(s) for s in final_shape]
            input = (
                fftn_centered(input, spatial_dims=2, is_complex=True)
                if args.model_type.lower() in ["varnet", "kspace_mar"]
                else input
            )

            # iterate through all samples:
            num_samples = input.shape[0]
            outputs = []
            for micro_b, _ in mini_dataloader(
                list(range(num_samples)),
                args.batch_size,
                shuffle=False,
                drop_last=False,
                pad_last=False,
            ):
                # forward pass
                inp, window_idx = windowed_input(input, micro_b, final_shape, num_frames=args.num_frames)
                mas = torch.Tensor(mask[window_idx])
                inp, mas, mean, std = (
                    inp.to(device),
                    mas.to(device),
                    mean.to(device),
                    std.to(device),
                )

                with autocast("cuda", torch.bfloat16, enabled=args.amp):
                    output = model(inp, mas.bool(), mask_type, acc_factor, acq_type)

                output = output[:, args.num_frames // 2]
                output = output * std[micro_b] + mean[micro_b]  # [1, c/1, 320, 320, 2]
                output = complex_abs(crop_k_space(output, (final_shape[-2], final_shape[-1])))  # [b, c/1, 320, 320]

                outputs.append(output.data.cpu().numpy())

            outputs = rearrange_mri_data(
                [np.vstack(outputs)],
                args,
                is_complex=False,
                reverse=True,
                num_slices=final_shape[-4],
                num_coils=final_shape[-3],
                temporal_shuffle=temporal_shuffle,
            )  # (time), slice, coil, h, w

            outputs_rss = np.sqrt(np.sum(outputs[0] ** 2, axis=-3))  # RSS: (time), slice, h, w

            outputs_pp = postprocess_mri_recon(
                outputs_rss,
                args,
                file_name,
                is_training=False,
                pp_z_score_norm=args.pp_z_score_norm,
            )  # w, h, slice, (time)

            save_img4ranking(
                outputs_pp,
                os.path.join(args.output_path, "val_img4ranking"),
                file_name.replace(".json", ".mat"),
            )

        if args.ddp:
            # wait for all processes to finish
            dist.barrier()
    torch.cuda.empty_cache()

    print(f"inference completed! test elapsed time: {(time.time() - tic_val) / 60:.2f} mins")

    if args.ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=Path,
        required=True,
        help="Path to the config file",
    )
    parser.add_argument(
        "-m",
        "--model_ckpt",
        default=None,  # Optional, will be auto-downloaded from Hugging Face if not provided
        type=Path,
        required=False,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "-i",
        "--input_path",
        default=None,
        type=Path,
        required=True,
        help="Path to the input folder",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default=None,
        type=Path,
        required=True,
        help="Path to the output folder",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="Debug mode",
    )

    args = parser.parse_args()
    config = load_config(args.config)
    config.ddp = is_ddp_enabled()
    model_variant = config.model_variant
    config.model_ckpt = resolve_checkpoint_path(model_variant, args.model_ckpt)
    config.output_path = args.output_path
    config.data_path_test = args.input_path
    config.debug = args.debug
    if config.ddp and ("MASTER_PORT" not in os.environ.keys()):
        port = str(find_free_network_port())
        print(f"using port {port}")
        os.environ["MASTER_PORT"] = port  # str(port)
    infer(config)
