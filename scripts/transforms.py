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

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Optional

import numpy as np
import torch
from monai.apps.reconstruction.complex_utils import complex_abs, convert_to_tensor, convert_to_tensor_complex
from monai.apps.reconstruction.mri_utils import root_sum_of_squares
from monai.config import KeysCollection, SequenceStr
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.fft_utils import fftn_centered, ifftn_centered
from monai.data.meta_obj import get_track_meta
from monai.transforms import RandAdjustContrast, ResizeWithPadOrCrop, SpatialPad
from monai.transforms.transform import MapTransform, RandomizableTransform, Transform
from monai.utils import FastMRIKeys, Method, PytorchPadMode, ensure_tuple_rep
from monai.utils.enums import TransformBackends
from mri_data.data_utils import circular_shift_in_kspace, rearrange_mri_data
from mri_data.ktSampling import kt_gaussian_sampling, kt_radial_sampling, kt_uniform_sampling, uniform_sampling
from readers import CMRxReconKeys
from torch import Tensor
from utils import complex_zscore

__all__ = [
    "ConvertPseudo3Dto3D",
    "KspaceMaskd",
    "RandShiftKspaced",
    "RandPhaseShiftKspaced",
    "RandSimulateNoReadoutOversampleKspaced",
    "RandAdjustContrastKspaced",
    "RearrangeAndNormalizeMRI",
    "SimulateLowerAccFactorKspaceMask",
    "RandResizeWithPadOrCropd",
]


class KspaceMask(RandomizableTransform):
    """
    A basic class for under-sampling mask setup. It provides common
    features for under-sampling mask generators.
    For example, RandomMaskFunc and EquispacedMaskFunc (two mask
    transform objects defined right after this module)
    both inherit MaskFunc to properly setup properties like the
    acceleration factor.
    """

    def __init__(
        self,
        center_fractions: Sequence[float],
        accelerations: Sequence[float],
        acs_lines: int = 20,
        spatial_dims: int = 2,
        is_complex: bool = True,
    ):
        """
        Args:
            center_fractions: Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers
                is chosen uniformly each time.
            accelerations: Amount of under-sampling. This should have the
                same length as center_fractions. If multiple values are
                provided, then one of these is chosen uniformly each time.
            spatial_dims: Number of spatial dims (e.g., it's 2 for a 2D data;
                it's also 2 for pseudo-3D datasets like the fastMRI dataset).
                The last spatial dim is selected for sampling. For the fastMRI
                dataset, k-space has the form (...,num_slices,num_coils,H,W)
                and sampling is done along W. For a general 3D data with the
                shape (...,num_coils,H,W,D), sampling is done along D.
            is_complex: if True, then the last dimension will be reserved for
                real/imaginary parts.
        """
        if len(center_fractions) != len(accelerations):
            raise ValueError(
                "Number of center fractions \
                should match number of accelerations"
            )

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.spatial_dims = spatial_dims
        self.is_complex = is_complex
        self.acs_lines = acs_lines

    @abstractmethod
    def __call__(self, kspace: NdarrayOrTensor) -> Sequence[Tensor]:
        """
        This is an extra instance to allow for defining new mask generators.
        For creating other mask transforms, define a new class and simply
        override __call__. See an example of this in
        :py:class:`monai.apps.reconstruction.transforms.array.RandomKspacemask`.

        Args:
            kspace: The input k-space data. The shape is (...,num_coils,H,W,2)
                for complex 2D inputs and (...,num_coils,H,W,D) for real 3D
                data.
        """
        raise NotImplementedError

    def randomize_choose_acceleration(self) -> Sequence[float]:
        """
        If multiple values are provided for center_fractions and
        accelerations, this function selects one value uniformly
        for each training/test sample.

        Returns:
            A tuple containing
                (1) center_fraction: chosen fraction of center kspace
                lines to exclude from under-sampling
                (2) acceleration: chosen acceleration factor
        """
        choice = np.random.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]
        return center_fraction, acceleration


class ConvertPseudo3Dto3D(MapTransform):
    def __init__(self, keys: KeysCollection, target_slices: int = 16, allow_missing_keys: bool = False) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.target_slices = target_slices

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, Tensor]:
        """
        Args:
            data: is a dictionary containing (key,value) pairs from the
                loaded dataset

        Returns:
            the new data dictionary
        """
        d = dict(data)

        for key in self.key_iterator(d):
            kspace_t = convert_to_tensor_complex(d[key])
            recon = ifftn_centered(kspace_t, spatial_dims=2, is_complex=True).permute(1, 2, 3, 0, 4)
            padder = SpatialPad(spatial_size=[-1, -1, self.target_slices, -1])
            # recon_rss_padder = SpatialPad(spatial_size=[-1, self.target_slices])
            recon_padded = padder(recon)
            d[key] = torch.view_as_complex(fftn_centered(recon_padded, spatial_dims=3, is_complex=True))

            # d['reconstruction_rss'] = recon_rss_padder(d['reconstruction_rss'].transpose(1, 2, 0))
            d["reconstruction_rss"] = complex_abs(recon_padded)

        return d


class RandomKspaceMask(KspaceMask):
    """
    This k-space mask transform under-samples the k-space according to a
    random sampling pattern. For 2D data (e.g. fastMRI, shape (...,num_coils,H,W,2)
    for complex inputs) the mask is applied along a single dimension.

    For general 3D data (shape (...,num_coils,H,W,D) for real data or
    (...,num_coils,H,W,D,2) for complex data) the mask is applied along
    the two phase-encoding dimensions (W and D), leaving H fully sampled.

    The mask always fully samples a centered region whose size is determined by
    center_fraction. Outside that region, locations are selected at random so that
    the expected number of samples is equal to the total number of locations divided
    by acceleration.

    Modified and adopted from:
        https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    """

    backend = [TransformBackends.TORCH]

    def __call__(self, kspace: NdarrayOrTensor) -> Sequence[Tensor]:
        """
        Args:
            kspace: The input k-space data.
                For complex 2D inputs: (..., num_coils, H, W, 2) and sampling is done along W.
                For real 3D data: (..., num_coils, H, W, D) and sampling is done along W and D.
                (For complex 3D data the expected shape is (..., num_coils, H, W, D, 2).)

        Returns:
            A tuple containing:
              (1) the under-sampled kspace,
              (2) the absolute value of the inverse fourier of the under-sampled kspace.
        """
        kspace_t = convert_to_tensor_complex(kspace)
        spatial_size = kspace_t.shape

        # Decide whether we are dealing with 2D vs. 3D (masking along one or two dims)
        # Here we assume that when 3D data is present the k-space shape will have one extra
        # spatial dimension (or an extra trailing channel in the complex case).
        is_3d = self.spatial_dims == 3

        center_fraction, acceleration = self.randomize_choose_acceleration()

        if is_3d:
            # --- Create a 2D mask along the two phase-encoding dims (W and D) ---
            if self.is_complex:
                # For complex 3D: expected shape (..., num_coils, H, W, D, 2)
                num_w = spatial_size[-3]  # W dimension
                num_d = spatial_size[-2]  # D dimension
                # The mask will be inserted at dims -3 and -2.
            else:
                # For real 3D: expected shape (..., num_coils, H, W, D)
                num_w = spatial_size[-2]  # W dimension
                num_d = spatial_size[-1]  # D dimension

            total_points = num_w * num_d

            # Determine the size of the fully-sampled central region along W and D.
            num_low_freqs_w = int(round(num_w * center_fraction))
            num_low_freqs_d = int(round(num_d * center_fraction))
            center_points = num_low_freqs_w * num_low_freqs_d

            # Compute the probability for sampling outside the center.
            prob = (total_points / acceleration - center_points) / (total_points - center_points)

            # Create a 2D mask: randomly sample points with probability `prob`
            mask_2d = self.R.uniform(size=(num_w, num_d)) < prob

            # Force the center of k-space to be fully sampled.
            pad_w = (num_w - num_low_freqs_w + 1) // 2
            pad_d = (num_d - num_low_freqs_d + 1) // 2
            mask_2d[pad_w : pad_w + num_low_freqs_w, pad_d : pad_d + num_low_freqs_d] = True

            # Expand the mask to the full k-space shape.
            mask_shape = [1] * len(spatial_size)
            if self.is_complex:
                mask_shape[-3] = num_w
                mask_shape[-2] = num_d
            else:
                mask_shape[-2] = num_w
                mask_shape[-1] = num_d
            mask = convert_to_tensor(mask_2d.reshape(*mask_shape).astype(np.float32))
        else:
            # --- 2D case (original behavior) ---
            if self.is_complex:
                num_cols = spatial_size[-2]
            else:
                num_cols = spatial_size[-1]

            num_low_freqs = int(round(num_cols * center_fraction))
            prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
            mask = self.R.uniform(size=num_cols) < prob
            pad = (num_cols - num_low_freqs + 1) // 2
            mask[pad : pad + num_low_freqs] = True

            mask_shape = [1 for _ in spatial_size]
            if self.is_complex:
                mask_shape[-2] = num_cols
            else:
                mask_shape[-1] = num_cols
            mask = convert_to_tensor(mask.reshape(*mask_shape).astype(np.float32))

        # Under-sample the k-space.
        masked = mask * kspace_t
        masked_kspace: Tensor = convert_to_tensor(masked)
        self.mask = mask

        # Compute the inverse Fourier transform of the masked k-space.
        masked_kspace_ifft: Tensor = convert_to_tensor(
            ifftn_centered(masked_kspace, spatial_dims=self.spatial_dims, is_complex=self.is_complex)
        )
        # combine coil images (it is assumed that the coil dimension is
        # the first dimension before spatial dimensions)
        masked_kspace_ifft_rss: Tensor = convert_to_tensor(
            root_sum_of_squares(complex_abs(masked_kspace_ifft), spatial_dim=-self.spatial_dims - 1)
        )
        return (
            masked_kspace,
            masked_kspace_ifft,
            masked_kspace_ifft_rss.unsqueeze(-self.spatial_dims - 1).unsqueeze(-1),
            acceleration,
        )


class EquispacedKspaceMask(KspaceMask):
    """
    This k-space mask transform under-samples the k-space according to an
    equi-distant sampling pattern. Precisely, it selects an equi-distant
    subset of columns from the input k-space data. If the k-space data has N
    columns, the mask picks out:

    1. N_low_freqs = (N * center_fraction) columns in the center corresponding
    to low-frequencies

    2. The other columns are selected with equal spacing at a proportion that
    reaches the desired acceleration rate taking into consideration the number
    of low frequencies. This ensures that the expected number of columns
    selected is equal to (N / acceleration)

    It is possible to use multiple center_fractions and accelerations, in
    which case one possible (center_fraction, acceleration) is chosen
    uniformly at random each time the EquispacedMaskFunc object is called.

    Example:
        If accelerations = [4, 8] and center_fractions = [0.08, 0.04],
        then there is a 50% probability that 4-fold acceleration with 8%
        center fraction is selected and a 50% probability that 8-fold
        acceleration with 4% center fraction is selected.

    Modified and adopted from:
        https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    """

    backend = [TransformBackends.TORCH]

    def __call__(self, kspace: NdarrayOrTensor) -> Sequence[Tensor]:
        """
        Args:
            kspace: The input k-space data. The shape is (...,num_coils,H,W,2)
                for complex 2D inputs and (...,num_coils,H,W,D) for real 3D
                data. The last spatial dim is selected for sampling. For the
                fastMRI multi-coil dataset, k-space has the form
                (...,num_slices,num_coils,H,W) and sampling is done along W.
                For a general 3D data with the shape (...,num_coils,H,W,D),
                sampling is done along D.

        Returns:
            A tuple containing
                (1) the under-sampled kspace
                (2) absolute value of the inverse fourier of the under-sampled kspace
        """
        kspace_t = convert_to_tensor_complex(kspace)
        spatial_size = kspace_t.shape
        num_cols = spatial_size[-1]
        if self.is_complex:  # for complex data
            num_cols = spatial_size[-2]

        center_fraction, acceleration = self.randomize_choose_acceleration()
        num_low_freqs = int(round(num_cols * center_fraction))

        # Create the mask
        mask = np.zeros(num_cols, dtype=np.float32)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = True

        # Determine acceleration rate by adjusting for the
        # number of low frequencies
        adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (num_low_freqs * acceleration - num_cols)
        offset = self.R.randint(0, round(adjusted_accel))

        accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
        accel_samples = np.around(accel_samples).astype(np.uint)
        mask[accel_samples] = True

        # Reshape the mask
        mask_shape = [1 for _ in spatial_size]
        if self.is_complex:
            mask_shape[-2] = num_cols
        else:
            mask_shape[-1] = num_cols

        mask = convert_to_tensor(mask.reshape(*mask_shape).astype(np.float32))

        # under-sample the ksapce
        masked = mask * kspace_t
        masked_kspace: Tensor = convert_to_tensor(masked)
        self.mask = mask

        # Compute the inverse Fourier transform of the masked k-space.
        masked_kspace_ifft: Tensor = convert_to_tensor(
            ifftn_centered(masked_kspace, spatial_dims=self.spatial_dims, is_complex=self.is_complex)
        )
        # combine coil images (it is assumed that the coil dimension is
        # the first dimension before spatial dimensions)
        masked_kspace_ifft_rss: Tensor = convert_to_tensor(
            root_sum_of_squares(complex_abs(masked_kspace_ifft), spatial_dim=-self.spatial_dims - 1)
        )
        return (
            masked_kspace,
            masked_kspace_ifft,
            masked_kspace_ifft_rss.unsqueeze(-self.spatial_dims - 1).unsqueeze(-1),
            acceleration,
        )


class FixedKspaceMask(KspaceMask):

    backend = [TransformBackends.TORCH]

    def __call__(self, kspace: NdarrayOrTensor, mask: NdarrayOrTensor) -> Sequence[Tensor]:
        """
        Args:
            kspace: The input k-space data. The shape is (...,num_coils,H,W,2)
                for complex 2D inputs and (...,num_coils,H,W,D) for real 3D
                data. The last spatial dim is selected for sampling. For the
                fastMRI multi-coil dataset, k-space has the form
                (...,num_slices,num_coils,H,W) and sampling is done along W.
                For a general 3D data with the shape (...,num_coils,H,W,D),
                sampling is done along D.

        Returns:
            A tuple containing
                (1) the under-sampled kspace
                (2) absolute value of the inverse fourier of the under-sampled kspace
        """
        kspace_t = convert_to_tensor_complex(kspace)
        mask = (mask > 0).astype(np.float32)
        # under-sample the ksapce
        mask = convert_to_tensor(mask[..., np.newaxis])
        masked = mask * kspace_t
        masked_kspace: Tensor = convert_to_tensor(masked)
        self.mask = mask

        # Compute the inverse Fourier transform of the masked k-space.
        masked_kspace_ifft: Tensor = convert_to_tensor(
            ifftn_centered(masked_kspace, spatial_dims=self.spatial_dims, is_complex=self.is_complex)
        )
        # combine coil images (it is assumed that the coil dimension is
        # the first dimension before spatial dimensions)
        masked_kspace_ifft_rss: Tensor = convert_to_tensor(
            root_sum_of_squares(complex_abs(masked_kspace_ifft), spatial_dim=-self.spatial_dims - 1)
        )
        return (
            masked_kspace,
            masked_kspace_ifft,
            masked_kspace_ifft_rss.unsqueeze(-self.spatial_dims - 1).unsqueeze(-1),
            None,
        )


class SimulateLowerAccFactorKspaceMask(MapTransform):
    def __init__(self, keys: KeysCollection, accelerations: Sequence[float], allow_missing_keys: bool = False) -> None:
        self.accelerations = sorted(accelerations)
        MapTransform.__init__(self, keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, Tensor]:
        """
        Args:
            data: is a dictionary containing (key,value) pairs from the
                loaded dataset

        Returns:
            the new data dictionary
        """
        d = dict(data)
        mask_type = "".join(ch for ch in d[CMRxReconKeys.MASK_TYPE] if not ch.isdigit())
        acc_factor = d["acc_factor"]
        target_acc_factors = [x for x in self.accelerations if x < acc_factor]
        tar_acc = target_acc_factors[0] if len(target_acc_factors) > 0 else acc_factor
        for key in self.key_iterator(d):
            og_mask = d[key]
            nt, nz, nc, ny, nx, cpx = og_mask.shape
            # for tar_acc in target_acc_factors:
            if mask_type == "ktGaussian":
                mask = kt_gaussian_sampling(
                    nx, ny, nt, 20, int(tar_acc), alpha=0.2, seed=np.random.randint(1, 1001)
                )  # nx, ny, nt
            elif mask_type == "ktRadial":
                mask = kt_radial_sampling(
                    nx, ny, nt, 20, tar_acc * 0.6, angle4next=137.5, cropcorner=True
                )  # nx, ny, nt
            elif mask_type == "Uniform":
                mask = uniform_sampling(nx, ny, nt, 20, int(tar_acc))
            else:
                raise ValueError("Unknown mask type, must be ktGaussian, ktRadial or Uniform")
            mask = np.expand_dims(mask.transpose(), axis=(1, 2))
            mask = (mask > 0).astype(np.float32)
            mask = convert_to_tensor(mask[..., np.newaxis]).expand(og_mask.shape)
            d["mask_lower_acc"] = mask
        return d


class KtUniformKspaceMask(KspaceMask):
    backend = [TransformBackends.TORCH]

    def __call__(self, kspace: NdarrayOrTensor) -> Sequence[Tensor]:
        assert kspace.ndim == 5, "Kspace mush has shape of (nt,nz,nc,ny,nx)"
        nt, nz, nc, ny, nx = kspace.shape
        _, acceleration = self.randomize_choose_acceleration()
        kspace_t = convert_to_tensor_complex(kspace)
        mask = kt_uniform_sampling(nx, ny, nt, self.acs_lines, int(acceleration))  # nx, ny, nt
        mask = np.expand_dims(mask.transpose(), axis=(1, 2))
        mask = (mask > 0).astype(np.float32)
        # under-sample the ksapce
        mask = convert_to_tensor(mask[..., np.newaxis])
        masked = mask * kspace_t
        masked_kspace: Tensor = convert_to_tensor(masked)
        self.mask = mask

        # Compute the inverse Fourier transform of the masked k-space.
        masked_kspace_ifft: Tensor = convert_to_tensor(
            ifftn_centered(masked_kspace, spatial_dims=self.spatial_dims, is_complex=self.is_complex)
        )
        # combine coil images (it is assumed that the coil dimension is
        # the first dimension before spatial dimensions)
        masked_kspace_ifft_rss: Tensor = convert_to_tensor(
            root_sum_of_squares(complex_abs(masked_kspace_ifft), spatial_dim=-self.spatial_dims - 1)
        )
        return (
            masked_kspace,
            masked_kspace_ifft,
            masked_kspace_ifft_rss.unsqueeze(-self.spatial_dims - 1).unsqueeze(-1),
            acceleration,
        )


class KtGaussianKspaceMask(KspaceMask):
    backend = [TransformBackends.TORCH]

    def __call__(self, kspace: NdarrayOrTensor) -> Sequence[Tensor]:
        assert kspace.ndim == 5, "Kspace mush has shape of (nt,nz,nc,ny,nx)"
        nt, nz, nc, ny, nx = kspace.shape
        _, acceleration = self.randomize_choose_acceleration()
        kspace_t = convert_to_tensor_complex(kspace)
        mask = kt_gaussian_sampling(
            nx, ny, nt, self.acs_lines, int(acceleration), alpha=0.2, seed=self.R.randint(1, 1001)
        )  # nx, ny, nt
        mask = np.expand_dims(mask.transpose(), axis=(1, 2))
        mask = (mask > 0).astype(np.float32)
        # under-sample the ksapce
        mask = convert_to_tensor(mask[..., np.newaxis])
        masked = mask * kspace_t
        masked_kspace: Tensor = convert_to_tensor(masked)
        self.mask = mask

        # Compute the inverse Fourier transform of the masked k-space.
        masked_kspace_ifft: Tensor = convert_to_tensor(
            ifftn_centered(masked_kspace, spatial_dims=self.spatial_dims, is_complex=self.is_complex)
        )
        # combine coil images (it is assumed that the coil dimension is
        # the first dimension before spatial dimensions)
        masked_kspace_ifft_rss: Tensor = convert_to_tensor(
            root_sum_of_squares(complex_abs(masked_kspace_ifft), spatial_dim=-self.spatial_dims - 1)
        )
        return (
            masked_kspace,
            masked_kspace_ifft,
            masked_kspace_ifft_rss.unsqueeze(-self.spatial_dims - 1).unsqueeze(-1),
            acceleration,
        )


class KtRadialKspaceMask(KspaceMask):
    backend = [TransformBackends.TORCH]

    def __call__(self, kspace: NdarrayOrTensor) -> Sequence[Tensor]:
        assert kspace.ndim == 5, "Kspace mush has shape of (nt,nz,nc,ny,nx)"
        nt, nz, nc, ny, nx = kspace.shape
        _, acceleration = self.randomize_choose_acceleration()
        kspace_t = convert_to_tensor_complex(kspace)
        mask = kt_radial_sampling(
            nx, ny, nt, self.acs_lines, acceleration * 0.6, angle4next=137.5, cropcorner=True
        )  # nx, ny, nt
        mask = np.expand_dims(mask.transpose(), axis=(1, 2))
        mask = (mask > 0).astype(np.float32)
        # under-sample the ksapce
        mask = convert_to_tensor(mask[..., np.newaxis])
        masked = mask * kspace_t
        masked_kspace: Tensor = convert_to_tensor(masked)
        self.mask = mask

        # Compute the inverse Fourier transform of the masked k-space.
        masked_kspace_ifft: Tensor = convert_to_tensor(
            ifftn_centered(masked_kspace, spatial_dims=self.spatial_dims, is_complex=self.is_complex)
        )
        # combine coil images (it is assumed that the coil dimension is
        # the first dimension before spatial dimensions)
        masked_kspace_ifft_rss: Tensor = convert_to_tensor(
            root_sum_of_squares(complex_abs(masked_kspace_ifft), spatial_dim=-self.spatial_dims - 1)
        )
        return (
            masked_kspace,
            masked_kspace_ifft,
            masked_kspace_ifft_rss.unsqueeze(-self.spatial_dims - 1).unsqueeze(-1),
            acceleration,
        )


class UniformKspaceMask(KspaceMask):
    backend = [TransformBackends.TORCH]

    def __call__(self, kspace: NdarrayOrTensor) -> Sequence[Tensor]:
        assert kspace.ndim == 5, "Kspace mush has shape of (nt,nz,nc,ny,nx)"
        nt, nz, nc, ny, nx = kspace.shape
        _, acceleration = self.randomize_choose_acceleration()
        kspace_t = convert_to_tensor_complex(kspace)
        mask = uniform_sampling(nx, ny, nt, self.acs_lines, int(acceleration))  # nx, ny, nt
        mask = np.expand_dims(mask.transpose(), axis=(1, 2))
        mask = (mask > 0).astype(np.float32)
        # under-sample the ksapce
        mask = convert_to_tensor(mask[..., np.newaxis])
        masked = mask * kspace_t
        masked_kspace: Tensor = convert_to_tensor(masked)
        self.mask = mask

        # Compute the inverse Fourier transform of the masked k-space.
        masked_kspace_ifft: Tensor = convert_to_tensor(
            ifftn_centered(masked_kspace, spatial_dims=self.spatial_dims, is_complex=self.is_complex)
        )
        # combine coil images (it is assumed that the coil dimension is
        # the first dimension before spatial dimensions)
        masked_kspace_ifft_rss: Tensor = convert_to_tensor(
            root_sum_of_squares(complex_abs(masked_kspace_ifft), spatial_dim=-self.spatial_dims - 1)
        )
        return (
            masked_kspace,
            masked_kspace_ifft,
            masked_kspace_ifft_rss.unsqueeze(-self.spatial_dims - 1).unsqueeze(-1),
            acceleration,
        )


class KspaceMaskd(RandomizableTransform, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.reconstruction.transforms.array.RandomKspacemask`.
    Other mask transforms can inherit from this class, for example:
    :py:class:`monai.apps.reconstruction.transforms.dictionary.EquispacedKspaceMaskd`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        center_fractions: Fraction of low-frequency columns to be retained.
            If multiple values are provided, then one of these numbers is
            chosen uniformly each time.
        accelerations: Amount of under-sampling. This should have the
            same length as center_fractions. If multiple values are provided,
            then one of these is chosen uniformly each time.
        spatial_dims: Number of spatial dims (e.g., it's 2 for a 2D data; it's
            also 2 for pseudo-3D datasets like the fastMRI dataset).
            The last spatial dim is selected for sampling. For the fastMRI
            dataset, k-space has the form (...,num_slices,num_coils,H,W)
            and sampling is done along W. For a general 3D data with the
            shape (...,num_coils,H,W,D), sampling is done along D.
        is_complex: if True, then the last dimension will be reserved
            for real/imaginary parts.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        keys: KeysCollection,
        mask_types: Sequence[str],
        center_fractions: Sequence[float],
        accelerations: Sequence[float],
        acs_lines: int = 20,
        spatial_dims: int = 2,
        is_complex: bool = True,
        allow_missing_keys: bool = False,
        return_ifft: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.maskers = []
        self.masker_enum = {
            "random": RandomKspaceMask,
            "equispaced": EquispacedKspaceMask,
            "fixed": FixedKspaceMask,
            "kt_uniform": KtUniformKspaceMask,
            "kt_gaussian": KtGaussianKspaceMask,
            "kt_radial": KtRadialKspaceMask,
            "uniform": UniformKspaceMask,
        }
        self.return_ifft = return_ifft
        for m in mask_types:
            self.maskers.append(
                self.masker_enum[m](center_fractions, accelerations, acs_lines, spatial_dims, is_complex)
            )

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> KspaceMaskd:
        super().set_random_state(seed, state)
        for i in range(len(self.maskers)):
            self.maskers[i].set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, Tensor]:
        """
        Args:
            data: is a dictionary containing (key,value) pairs from the
                loaded dataset

        Returns:
            the new data dictionary
        """
        d = dict(data)
        chosen_masker_idx = np.random.randint(0, len(self.maskers))
        for key in self.key_iterator(d):
            if (
                isinstance(self.maskers[chosen_masker_idx], FixedKspaceMask)
                and d[key].shape[-2:] == d[FastMRIKeys.MASK].shape[-2:]
            ):
                if self.return_ifft:
                    d[key + "_masked"], d[key + "_masked_ifft"], d[key + "_masked_ifft_rss"], acc_factor = self.maskers[
                        chosen_masker_idx
                    ](d[key], d[FastMRIKeys.MASK])
                else:
                    d[key + "_masked"], _, _, acc_factor = self.maskers[chosen_masker_idx](d[key], d[FastMRIKeys.MASK])
                mask_type = d["kspace_meta_dict"][CMRxReconKeys.MASK_TYPE]
                acc_factor = int("".join(ch for ch in mask_type if ch.isdigit()))
            elif not isinstance(self.maskers[chosen_masker_idx], FixedKspaceMask):
                if self.return_ifft:
                    d[key + "_masked"], d[key + "_masked_ifft"], d[key + "_masked_ifft_rss"], acc_factor = self.maskers[
                        chosen_masker_idx
                    ](d[key])
                else:
                    d[key + "_masked"], _, _, acc_factor = self.maskers[chosen_masker_idx](d[key])
                mask_type = self.maskers[chosen_masker_idx].__class__.__name__
            else:
                raise RuntimeError(
                    "Augmentation of no readout oversample simulation is not compatible with fixed mask type."
                )
            d[FastMRIKeys.MASK] = self.maskers[chosen_masker_idx].mask.expand(d[key + "_masked"].shape)
            d[CMRxReconKeys.MASK_TYPE] = mask_type
            d["acc_factor"] = acc_factor

        return d  # type: ignore


class RandShiftKspaced(RandomizableTransform, MapTransform):

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.25,
        shift: tuple[int, int] = (16, 32),
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.shift = shift
        self.shifter = circular_shift_in_kspace

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> RandShiftKspaced:
        super().set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, Tensor]:
        """
        Args:
            data: is a dictionary containing (key,value) pairs from the
                loaded dataset

        Returns:
            the new data dictionary
        """
        self.randomize(None)
        d = dict(data)
        if self._do_transform:
            y_shift = self.R.randint(-self.shift[0], self.shift[0])
            x_shift = self.R.randint(-self.shift[1], self.shift[1])
            for key in self.key_iterator(d):
                d[key] = self.shifter(d[key], (y_shift, x_shift))

        return d  # type: ignore


class RandPhaseShiftKspaced(RandomizableTransform, MapTransform):

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.25,
        angle: int = 90,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.angle = angle

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> RandPhaseShiftKspaced:
        super().set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, Tensor]:
        """
        Args:
            data: is a dictionary containing (key,value) pairs from the
                loaded dataset

        Returns:
            the new data dictionary
        """
        self.randomize(None)
        d = dict(data)
        if self._do_transform:
            phi = torch.deg2rad(Tensor([self.R.randint(-self.angle, self.angle)]))
            for key in self.key_iterator(d):
                d[key] = d[key] * torch.exp(1j * phi)

        return d  # type: ignore


class RandSimulateNoReadoutOversampleKspaced(RandomizableTransform, MapTransform):

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        ratio: tuple[int] = (1, 0.5),
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.ratio = ratio

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> RandSimulateNoReadoutOversampleKspaced:
        super().set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, Tensor]:
        """
        Args:
            data: is a dictionary containing (key,value) pairs from the
                loaded dataset

        Returns:
            the new data dictionary
        """
        self.randomize(None)
        d = dict(data)
        if self._do_transform:
            for key in self.key_iterator(d):
                filename = d.get("kspace_meta_dict", {}).get("filename", "")
                if (
                    any(substring in filename for substring in ["map", "T1w", "T2w"])
                    and d[key].shape[-1] // 2 > d[key].shape[-2]
                ):
                    recon = torch.fft.fftshift(
                        torch.fft.ifftn(torch.fft.ifftshift(d[key], dim=[-2, -1]), dim=[-2, -1], norm="ortho"),
                        dim=[-2, -1],
                    )
                    half_fov_recon = recon[
                        ..., recon.shape[-1] // 4 : recon.shape[-1] // 4 + ((recon.shape[-1] // 2) & ~1)
                    ]
                    d[key] = torch.fft.fftshift(
                        torch.fft.fftn(torch.fft.ifftshift(half_fov_recon, dim=[-2, -1]), dim=[-2, -1], norm="ortho"),
                        dim=[-2, -1],
                    )
                    d["kspace_meta_dict"]["shape"] = np.array(d[key].shape)

        return d  # type: ignore


class RandResizeWithPadOrCropd(RandomizableTransform, MapTransform):

    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Sequence[int] | int,
        prob: float = 0.25,
        mode: SequenceStr = PytorchPadMode.CONSTANT,
        allow_missing_keys: bool = False,
        method: str = Method.SYMMETRIC,
        **pad_kwargs,
    ) -> None:

        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.method = method
        self.pad_kwargs = pad_kwargs
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.spatial_size = ensure_tuple_rep(spatial_size, 2)

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        self.randomize(None)
        d = dict(data)
        if self._do_transform:
            y = self.spatial_size[0]
            x = self.spatial_size[1]
            for key, m in self.key_iterator(d, self.mode):
                before_shape = d[key].shape
                after_shape = [
                    -1,
                    -1,
                    min(before_shape[-2] + 2 * self.R.randint(-(y // 2), (y + 1) // 2), 328),
                    min(before_shape[-1] + 2 * self.R.randint(-(x // 2), (x + 1) // 2), 806),
                ]
                padcropper = ResizeWithPadOrCrop(spatial_size=after_shape, method=self.method, **self.pad_kwargs)
                d[key] = padcropper(d[key], mode=m)
                d["kspace_meta_dict"]["shape"] = np.array(d[key].shape)

        return d


class RandAdjustContrastKspaced(RandomizableTransform, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandAdjustContrast`.
    Randomly changes image intensity with gamma transform. Each pixel/voxel intensity is updated as:

        `x = ((x - min) / intensity_range) ^ gamma * intensity_range + min`

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        prob: Probability of adjustment.
        gamma: Range of gamma values.
            If single number, value is picked from (0.5, gamma), default is (0.5, 4.5).
        invert_image: whether to invert the image before applying gamma augmentation. If True, multiply all intensity
            values with -1 before the gamma transform and again after the gamma transform. This behaviour is mimicked
            from `nnU-Net <https://www.nature.com/articles/s41592-020-01008-z>`_, specifically `this
            <https://github.com/MIC-DKFZ/batchgenerators/blob/7fb802b28b045b21346b197735d64f12fbb070aa/batchgenerators/augmentations/color_augmentations.py#L107>`_
            function.
        retain_stats: if True, applies a scaling factor and an offset to all intensity values after gamma transform to
            ensure that the output intensity distribution has the same mean and standard deviation as the intensity
            distribution of the input. This behaviour is mimicked from `nnU-Net
            <https://www.nature.com/articles/s41592-020-01008-z>`_, specifically `this
            <https://github.com/MIC-DKFZ/batchgenerators/blob/7fb802b28b045b21346b197735d64f12fbb070aa/batchgenerators/augmentations/color_augmentations.py#L107>`_
            function.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = RandAdjustContrast.backend

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        gamma: tuple[float, float] | float = (0.5, 4.5),
        invert_image: bool = False,
        retain_stats: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.adjuster = RandAdjustContrastComplex(
            gamma=gamma, prob=1.0, invert_image=invert_image, retain_stats=retain_stats
        )
        self.invert_image = invert_image

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> RandAdjustContrastKspaced:
        super().set_random_state(seed, state)
        self.adjuster.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if self._do_transform:
            # all the keys share the same random gamma value
            self.adjuster.randomize(None)
            for key in self.key_iterator(d):
                recon = torch.fft.fftshift(
                    torch.fft.ifftn(torch.fft.ifftshift(d[key], dim=[-2, -1]), dim=[-2, -1], norm="ortho"), dim=[-2, -1]
                )
                d[key] = torch.fft.fftshift(
                    torch.fft.fftn(
                        torch.fft.ifftshift(self.adjuster(recon, randomize=False), dim=[-2, -1]),
                        dim=[-2, -1],
                        norm="ortho",
                    ),
                    dim=[-2, -1],
                )
        return d


class AdjustContrastComplex(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, gamma: float, invert_image: bool = False, retain_stats: bool = False):
        super().__init__()
        if not isinstance(gamma, (int, float)):
            raise ValueError(f"gamma must be a float or int, got {type(gamma)}.")
        self.gamma = gamma
        self.invert_image = invert_image
        self.retain_stats = retain_stats

    def __call__(self, img: Tensor, gamma: Optional[float] = None) -> Tensor:
        gamma = gamma if gamma is not None else self.gamma

        # ensure Torch tensor
        img = convert_to_tensor(img, track_meta=get_track_meta())

        # If complex, split magnitude & phase
        if torch.is_complex(img):
            mag = torch.abs(img)
            phase = torch.angle(img)
            x = mag
        else:
            x = img

        # optional inversion
        if self.invert_image:
            x = -x

        # optional stats retention
        if self.retain_stats:
            mn, sd = x.mean(), x.std()

        # standard gamma contrast
        eps = 1e-7
        x_min = x.min()
        x_range = x.max() - x_min
        y = ((x - x_min) / (x_range + eps)) ** gamma * x_range + x_min

        if self.retain_stats:
            y = (y - y.mean()) / (y.std() + 1e-8)  # normalize
            y = y * sd + mn  # restore stats

        if self.invert_image:
            y = -y

        # Re-attach phase if was complex
        if torch.is_complex(img):
            # torch.polar(magnitude, angle) returns complex tensor
            return torch.polar(y, phase)
        else:
            return y


class RandAdjustContrastComplex(RandomizableTransform):
    """
    Randomly changes image intensity with gamma transform. Each pixel/voxel intensity is updated as:

        x = ((x - min) / intensity_range) ^ gamma * intensity_range + min

    Args:
        prob: Probability of adjustment.
        gamma: Range of gamma values.
            If single number, value is picked from (0.5, gamma), default is (0.5, 4.5).
        invert_image: whether to invert the image before applying gamma augmentation. If True, multiply all intensity
            values with -1 before the gamma transform and again after the gamma transform. This behaviour is mimicked
            from `nnU-Net <https://www.nature.com/articles/s41592-020-01008-z>`_, specifically `this
            <https://github.com/MIC-DKFZ/batchgenerators/blob/7fb802b28b045b21346b197735d64f12fbb070aa/batchgenerators/augmentations/color_augmentations.py#L107>`_
            function.
        retain_stats: if True, applies a scaling factor and an offset to all intensity values after gamma transform to
            ensure that the output intensity distribution has the same mean and standard deviation as the intensity
            distribution of the input. This behaviour is mimicked from `nnU-Net
            <https://www.nature.com/articles/s41592-020-01008-z>`_, specifically `this
            <https://github.com/MIC-DKFZ/batchgenerators/blob/7fb802b28b045b21346b197735d64f12fbb070aa/batchgenerators/augmentations/color_augmentations.py#L107>`_
            function.
    """

    backend = AdjustContrastComplex.backend

    def __init__(
        self,
        prob: float = 0.1,
        gamma: Sequence[float] | float = (0.5, 4.5),
        invert_image: bool = False,
        retain_stats: bool = False,
    ) -> None:
        RandomizableTransform.__init__(self, prob)

        if isinstance(gamma, (int, float)):
            if gamma <= 0.5:
                raise ValueError(
                    f"if gamma is a number, must greater than 0.5 and value is picked from (0.5, gamma), got {gamma}"
                )
            self.gamma = (0.5, gamma)
        elif len(gamma) != 2:
            raise ValueError("gamma should be a number or pair of numbers.")
        else:
            self.gamma = (min(gamma), max(gamma))

        self.gamma_value: float = 1.0
        self.invert_image: bool = invert_image
        self.retain_stats: bool = retain_stats

        self.adjust_contrast = AdjustContrastComplex(
            self.gamma_value, invert_image=self.invert_image, retain_stats=self.retain_stats
        )

    def randomize(self, data: Any | None = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.gamma_value = self.R.uniform(low=self.gamma[0], high=self.gamma[1])

    def __call__(self, img: NdarrayOrTensor, randomize: bool = True) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        if randomize:
            self.randomize()

        if not self._do_transform:
            return img

        if self.gamma_value is None:
            raise RuntimeError("gamma_value is not set, please call `randomize` function first.")

        return self.adjust_contrast(img, self.gamma_value)


class RearrangeAndNormalizeMRI(MapTransform):
    def __init__(self, keys: KeysCollection, args, allow_missing_keys: bool = False) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.keys_list = keys
        self.args = args
        assert len(keys) == 3, "Keys should have 3 keys (input, target, mask), got {}".format(len(keys))

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, Tensor]:
        d = dict(data)

        temporal_shuffle = (
            torch.randperm(int(d["kspace_meta_dict"]["shape"][0]))
            if "map" in d["kspace_meta_dict"]["filename"] and self.args.do_mapping_shuffle
            else torch.arange(int(d["kspace_meta_dict"]["shape"][0]))
        )
        input, target, mask = rearrange_mri_data(
            [d[k][None, ...] for k in self.keys_list], self.args, temporal_shuffle=temporal_shuffle
        )
        input, mean, std = complex_zscore(input, dim=[1, 2, 3])

        d[self.keys_list[0]] = input.contiguous()
        d[self.keys_list[1]] = target.contiguous()
        d[self.keys_list[2]] = mask.contiguous()
        d["mean"] = mean.contiguous()
        d["std"] = std.contiguous()
        d["temporal_shuffle"] = temporal_shuffle

        return d
