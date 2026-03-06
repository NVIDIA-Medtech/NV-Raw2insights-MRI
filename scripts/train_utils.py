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

import numpy as np
import torch
from monai.apps.reconstruction.complex_utils import convert_to_tensor_complex
from monai.apps.reconstruction.transforms.dictionary import ExtractDataKeyFromMetaKeyd
from monai.data.fft_utils import ifftn_centered
from monai.transforms import Compose, EnsureTyped, Identityd, Lambdad, LoadImaged, RandFlipd, ResizeWithPadOrCropd
from mri_data.data_utils import get_reader
from muon import MuonWithAuxAdam
from transforms import *
from utils import *


def get_train_transforms(args):
    """
    Get the training transforms.
    """
    train_transforms = Compose(
        [
            LoadImaged(
                keys=["kspace"],
                reader=get_reader(args, is_testing=True),
                image_only=False,
                dtype=np.complex64,
            ),
            (RandFlipd(keys=["kspace"], spatial_axis=-2, prob=0.25) if args.data_aug else Identityd(keys=["kspace"])),
            (RandFlipd(keys=["kspace"], spatial_axis=-1, prob=0.25) if args.data_aug else Identityd(keys=["kspace"])),
            (
                RandShiftKspaced(keys=["kspace"], prob=0.25, shift=(16, 32))
                if args.data_aug
                else Identityd(keys=["kspace"])
            ),
            (
                RandPhaseShiftKspaced(keys=["kspace"], prob=0.25, angle=45)
                if args.data_aug
                else Identityd(keys=["kspace"])
            ),
            (
                RandAdjustContrastKspaced(keys=["kspace"], prob=0.25, gamma=(0.5, 2))
                if args.data_aug
                else Identityd(keys=["kspace"])
            ),
            (
                RandResizeWithPadOrCropd(keys=["kspace"], prob=0.25, spatial_size=(16, 32))
                if args.data_aug
                else Identityd(keys=["kspace"])
            ),
            (
                RandSimulateNoReadoutOversampleKspaced(keys=["kspace"], prob=0.25)
                if args.dataset.lower() == "cmrxrecon"
                else Identityd(keys=["kspace"])
            ),
            ExtractDataKeyFromMetaKeyd(keys=["mask", "acquisition"], meta_key="kspace_meta_dict"),
            KspaceMaskd(
                keys=["kspace"],
                mask_types=args.train_mask_types,
                center_fractions=args.center_fractions,
                accelerations=args.accelerations,
                acs_lines=args.acs_lines,
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
    return train_transforms


def get_val_transforms(args):
    """
    Get the validation transforms.
    """
    val_transforms = Compose(
        [
            LoadImaged(
                keys=["kspace"],
                reader=get_reader(args, is_testing=True),
                image_only=False,
                dtype=np.complex64,
            ),
            ExtractDataKeyFromMetaKeyd(keys=["mask", "acquisition"], meta_key="kspace_meta_dict"),
            KspaceMaskd(
                keys=["kspace"],
                mask_types=args.val_mask_types,
                center_fractions=args.center_fractions,
                accelerations=args.accelerations,
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
    return val_transforms


def get_optimizer(args, model):
    """
    Get the optimizer for the model.
    """
    if args.muon:
        excluded_patterns = [
            "unet.conv_0.conv_0.conv",
            "embed_conv",
            "dc_weight_map",
            "dc_weight",
            "feat_extract",
        ]
        muon_params = [
            p
            for n, p in model.named_parameters()
            if p.ndim >= 2 and not any(exclude in n for exclude in excluded_patterns)
        ]
        adam_params = [
            p for n, p in model.named_parameters() if p.ndim < 2 or any(include in n for include in excluded_patterns)
        ]
        param_groups = [
            dict(
                params=filter(lambda p: p.requires_grad, muon_params),
                use_muon=True,
                lr=args.lr,
                weight_decay=args.weight_decay,
            ),
            dict(
                params=filter(lambda p: p.requires_grad, adam_params),
                use_muon=False,
                lr=args.lr,
                weight_decay=args.weight_decay,
                lr_scale=1.0,
            ),
        ]
        optimizer = MuonWithAuxAdam(param_groups)
    elif args.lookahead:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        optimizer = Lookahead(optimizer)
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    print("using optimizer: ", optimizer.__class__.__name__)
    return optimizer
