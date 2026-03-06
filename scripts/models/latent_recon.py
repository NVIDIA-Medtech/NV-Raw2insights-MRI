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

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
from models.restormer.restormer import restormer_mri
from monai.networks.utils import copy_model_state
from torch.utils.checkpoint import checkpoint

__all__ = ["create_mri_recon_model"]


class Cascaded_MRI_Recon(nn.Module):
    def __init__(self, recon_models, pretrain_recon=None):
        super().__init__()
        self.recon_models = recon_models
        self.cascades = nn.ModuleList([recon_model for recon_model in recon_models])
        self.pretrain_recon = pretrain_recon
        if self.pretrain_recon is not None:
            self.load_recon_model()

    def strip_prefix(self, state_dict, prefix="cascades.0."):
        new_state = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_key = k[len(prefix) :]
            else:
                new_key = k
            new_state[new_key] = v
        return new_state

    def load_recon_model(self):
        try:
            recon_state_dict = torch.load(self.pretrain_recon, map_location="cpu", weights_only=True)
        except BaseException:
            recon_state_dict = torch.load(self.pretrain_recon, map_location="cpu", weights_only=False)["net_state_dict"]
        if len(set([int(s.split(".")[1]) for s in list(recon_state_dict.keys())])) == 1:  # if single cascade
            for i in range(len(self.cascades)):
                _, updated_keys, unchanged_keys = copy_model_state(
                    self.cascades[i], self.strip_prefix(recon_state_dict)
                )
        else:
            _, updated_keys, unchanged_keys = copy_model_state(self, recon_state_dict)
        print(f"Loaded pretrained Recon model from {self.pretrain_recon}")

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        ref_image = x.clone()
        for _, recon in enumerate(self.recon_models):
            if self.training:
                x = checkpoint(recon, x, ref_image, mask, use_reentrant=False)
            else:
                x = recon(x, ref_image, mask)
        return x


class Cascaded_SkipConnected_MRI_Recon(nn.Module):
    def __init__(self, recon_models, pretrain_recon=None, enable_cas_skips=True):
        super().__init__()
        self.recon_models = recon_models
        self.cascades = nn.ModuleList([recon_model for recon_model in recon_models])
        self.pretrain_recon = pretrain_recon
        if self.pretrain_recon is not None:
            self.load_recon_model()
        self.enable_cas_skips = enable_cas_skips

    def strip_prefix(self, state_dict, prefix="cascades.0."):
        new_state = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_key = k[len(prefix) :]
            else:
                new_key = k
            new_state[new_key] = v
        return new_state

    def load_recon_model(self):
        try:
            recon_state_dict = torch.load(self.pretrain_recon, map_location="cpu", weights_only=True)
        except BaseException:
            recon_state_dict = torch.load(self.pretrain_recon, map_location="cpu", weights_only=False)["net_state_dict"]
        _, updated_keys, unchanged_keys = copy_model_state(self, recon_state_dict)
        print(f"Loaded pretrained Recon model from {self.pretrain_recon}")

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        mask_type: str = None,
        acc_factor: int = None,
        acq_type: str = None,
        cas_skips=None,
        sensitivity_maps=None,
        ref_image=None,
        timestep=None,
    ) -> Any:
        ref_image = x.clone() if ref_image is None else ref_image
        for i, recon in enumerate(self.recon_models):
            actual_timestep = timestep * len(self.recon_models) + i if timestep is not None else None
            if not self.enable_cas_skips:
                cas_skips = None
            if self.training:
                x, cas_skips, sensitivity_maps = checkpoint(
                    recon,
                    x,
                    ref_image,
                    mask,
                    cas_skips,
                    mask_type,
                    acc_factor,
                    acq_type,
                    sensitivity_maps,
                    actual_timestep,
                    use_reentrant=False,
                )
            else:
                x, cas_skips, sensitivity_maps = recon(
                    x,
                    ref_image,
                    mask,
                    cas_skips,
                    mask_type,
                    acc_factor,
                    acq_type,
                    sensitivity_maps,
                    actual_timestep,
                )
        if timestep is not None:
            return x, cas_skips, sensitivity_maps
        else:
            return x


class Flow_MRI_Recon(nn.Module):
    def __init__(self, recon_model, pretrain_recon=None, num_steps=10):
        super().__init__()
        self.recon_model = recon_model
        self.pretrain_recon = pretrain_recon
        self.num_steps = num_steps
        if self.pretrain_recon is not None:
            self.load_recon_model()

    def strip_prefix(self, state_dict, prefix="cascades.0."):
        new_state = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_key = k[len(prefix) :]
            else:
                new_key = k
            new_state[new_key] = v
        return new_state

    def load_recon_model(self):
        try:
            recon_state_dict = torch.load(self.pretrain_recon, map_location="cpu", weights_only=True)
        except BaseException:
            recon_state_dict = torch.load(self.pretrain_recon, map_location="cpu", weights_only=False)["net_state_dict"]
        if len(set([int(s.split(".")[1]) for s in list(recon_state_dict.keys())])) == 1:  # if single cascade
            _, updated_keys, unchanged_keys = copy_model_state(self.recon_model, self.strip_prefix(recon_state_dict))
        else:
            _, updated_keys, unchanged_keys = copy_model_state(self, recon_state_dict)
        print(f"Loaded pretrained Recon model from {self.pretrain_recon}")

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        ref_image = x.clone()
        for _ in range(self.num_steps):
            if self.training:
                x = checkpoint(self.recon_model, x, ref_image, mask, use_reentrant=False)
            else:
                x = self.recon_model(x, ref_image, mask)
        return x


class Flow_SkipConnected_MRI_Recon(nn.Module):
    def __init__(
        self,
        recon_model,
        pretrain_recon=None,
        enable_cas_skips=True,
        num_steps=10,
        constant_input_flow=False,
    ):
        super().__init__()
        self.recon_model = recon_model
        self.pretrain_recon = pretrain_recon
        self.num_steps = num_steps
        if self.pretrain_recon is not None:
            self.load_recon_model()
        self.enable_cas_skips = enable_cas_skips
        self.constant_input_flow = constant_input_flow

    def load_recon_model(self):
        try:
            recon_state_dict = torch.load(self.pretrain_recon, map_location="cpu", weights_only=True)
        except BaseException:
            recon_state_dict = torch.load(self.pretrain_recon, map_location="cpu", weights_only=False)["net_state_dict"]
        _, updated_keys, unchanged_keys = copy_model_state(self, recon_state_dict)
        print(f"Loaded pretrained Recon model from {self.pretrain_recon}")

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        mask_type: str = None,
        acc_factor: int = None,
        acq_type: str = None,
    ) -> torch.Tensor:
        ref_image = x.clone()
        x0 = x
        cas_skips = None
        for i in range(self.num_steps):
            sensitivity_maps = None
            if self.constant_input_flow:
                if self.training:
                    x, cas_skips, sensitivity_maps = checkpoint(
                        self.recon_model,
                        x0,
                        mask,
                        mask_type,
                        acc_factor,
                        acq_type,
                        cas_skips,
                        sensitivity_maps,
                        ref_image,
                        i,
                        use_reentrant=False,
                    )
                else:
                    x, cas_skips, sensitivity_maps = self.recon_model(
                        x0,
                        mask,
                        mask_type,
                        acc_factor,
                        acq_type,
                        cas_skips,
                        sensitivity_maps,
                        ref_image,
                        i,
                    )
            else:
                if self.training:
                    x, cas_skips, sensitivity_maps = checkpoint(
                        self.recon_model,
                        x,
                        mask,
                        mask_type,
                        acc_factor,
                        acq_type,
                        cas_skips,
                        sensitivity_maps,
                        ref_image,
                        i,
                        use_reentrant=False,
                    )
                else:
                    x, cas_skips, sensitivity_maps = self.recon_model(
                        x,
                        mask,
                        mask_type,
                        acc_factor,
                        acq_type,
                        cas_skips,
                        sensitivity_maps,
                        ref_image,
                        i,
                    )
        return x


def create_mri_recon_model(args):
    if not hasattr(args, "num_cascades"):
        args.num_cascades = 1
    if args.model_type.lower() == "restormer":
        restormers = []
        if args.flow:
            for i in range(args.num_cascades):
                restormers.append(
                    restormer_mri(
                        args,
                        use_acs_region=args.use_acs_region if i == 0 else False,
                        last_cascade=(i == args.num_cascades - 1),
                        first_cascade=(i == 0),
                    )
                )
            cascades = Cascaded_SkipConnected_MRI_Recon(
                restormers,
                pretrain_recon=args.pretrained_recon,
                enable_cas_skips=args.enable_cas_skips,
            )
            return Flow_SkipConnected_MRI_Recon(
                cascades,
                pretrain_recon=args.pretrained_recon,
                num_steps=args.num_steps,
                enable_cas_skips=args.enable_cas_skips,
                constant_input_flow=args.constant_input_flow,
            )
        else:
            for i in range(args.num_cascades):
                restormers.append(
                    restormer_mri(
                        args,
                        use_acs_region=args.use_acs_region if i == 0 else False,
                        last_cascade=(i == args.num_cascades - 1),
                        first_cascade=(i == 0),
                    )
                )
            return Cascaded_SkipConnected_MRI_Recon(
                restormers,
                pretrain_recon=args.pretrained_recon,
                enable_cas_skips=args.enable_cas_skips,
            )
    else:
        raise ValueError
