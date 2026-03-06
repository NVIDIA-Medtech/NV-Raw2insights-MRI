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

import math
from typing import Sequence

import torch
import torch.nn as nn
from einops import rearrange
from monai.data.fft_utils import fftn_centered, ifftn_centered
from monai.metrics import SSIMMetric
from monai.utils import LossReduction
from readers import CestMRIReader, CMRxReconReader, FastMRIReader
from run4ranking import run4Ranking
from torch.nn.modules.loss import _Loss
from torchvision.transforms.functional import center_crop


class FrequencyDownsampling(nn.Module):
    def __init__(self, stride=(2, 2)):
        """
        Initializes the FrequencyDownsampling module.

        Args:
            stride (tuple): A tuple of two integers representing the subsampling stride
                            in the height and width dimensions, respectively.
        """
        super(FrequencyDownsampling, self).__init__()
        self.stride = stride

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Downsamples the input tensor by performing a strided subsampling in the frequency domain.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channel, height, width).

        Returns:
            torch.Tensor: Downsampled tensor of shape
                          (batch, channel, height//stride[0], width//stride[1]).
        """
        # Check that the height and width dimensions are divisible by the stride values.
        if x.shape[2] % self.stride[0] != 0 or x.shape[3] % self.stride[1] != 0:
            raise ValueError(
                "The height and width of the tensor must be divisible by the respective stride dimensions."
            )

        # Compute the Fourier transform along the spatial dimensions.
        x_freq = torch.fft.fft2(x.float())

        # Subsample the frequency domain representation.
        x_freq_subsampled = x_freq[:, :, :: self.stride[0], :: self.stride[1]]

        # Compute the inverse Fourier transform to get back to the spatial domain.
        x_downsampled = torch.fft.ifft2(x_freq_subsampled)

        # Since the original input is real-valued, return the real part of the output.
        return x_downsampled.real


class FrequencyUpsampling(nn.Module):
    def __init__(self, stride=(2, 2)):
        """
        Initializes the FrequencyUpsampling module.

        Args:
            stride (tuple): A tuple of two integers representing the upsampling factor
                            along the height and width dimensions, respectively.
        """
        super(FrequencyUpsampling, self).__init__()
        self.stride = stride

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Upsamples the input tensor by inserting zeros between frequency components
        in the Fourier domain and then applying the inverse FFT.

        The procedure is as follows:
          1. Compute the FFT along the spatial dimensions.
          2. Create a new frequency-domain tensor of the upsampled size (by multiplying the
             low-resolution dimensions with the stride factors) filled with zeros.
          3. Insert the computed frequency coefficients into the new tensor at positions that
             are multiples of the stride (i.e. every stride-th index).
          4. Apply the inverse FFT to obtain the spatial domain upsampled signal.
          5. Return the real part.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channel, H_low, W_low).

        Returns:
            torch.Tensor: Upsampled tensor of shape
                          (batch, channel, H_low * stride[0], W_low * stride[1]).
        """
        batch, channel, H_low, W_low = x.shape
        new_height = H_low * self.stride[0]
        new_width = W_low * self.stride[1]

        # Compute the Fourier transform of the low-resolution input along spatial dimensions.
        x_freq = torch.fft.fft2(x.float())

        # Create an empty frequency tensor of the new size with the same dtype and device.
        upsampled_freq = torch.zeros(batch, channel, new_height, new_width, dtype=x_freq.dtype, device=x.device)

        # Insert the computed frequency coefficients into the upsampled frequency tensor.
        # This places each coefficient at the corresponding location and inserts zeros in between.
        upsampled_freq[..., :: self.stride[0], :: self.stride[1]] = x_freq

        # Compute the inverse FFT to transform back to the spatial domain.
        x_upsampled = torch.fft.ifft2(upsampled_freq)

        # Return the real part of the upsampled signal.
        return x_upsampled.real


def rearrange_mri_data(
    mri_data,
    args,
    temporal_shuffle=None,
    num_slices=None,
    num_coils=None,
    is_complex=True,
    reverse=False,
):
    if not isinstance(mri_data, list):
        mri_data = list(mri_data)

    if args.do_mapping_shuffle and temporal_shuffle and not reverse:
        for i in range(len(mri_data)):
            mri_data[i] = mri_data[i][:, temporal_shuffle, :, :, :, :, :]

    ds = args.dataset.lower()

    if reverse and (num_slices is None or num_coils is None):
        raise ValueError("num_slices and num_coils must be specified for reversing rearrange.")

    # Define canonical patterns (for non-reversed rearrangement)
    # Key: (is_multi_coil, dataset)
    patterns = {
        (True, "fastmri"): ("batch time slice coil", "(time slice) (coil batch)"),
        (True, "cmrxrecon"): ("batch time slice coil", "(time slice) (coil batch)"),
        (True, "cest"): ("batch time slice coil", "(time slice) (coil batch)"),
        (False, "fastmri"): ("batch time slice coil", "(time slice coil) (batch)"),
        (False, "cmrxrecon"): ("batch time slice coil", "(time slice coil) (batch)"),
        (False, "cest"): ("batch time slice coil", "(time slice coil) (batch)"),
    }

    key = (args.is_multi_coil, ds)
    try:
        left, right = patterns[key]
    except KeyError as err:
        raise ValueError("Unsupported configuration") from err

    # If reverse, swap left and right and require additional kwargs.
    if reverse:
        left, right = right, left
        rearrange_kwargs = {"slice": num_slices, "coil": num_coils}
    else:
        rearrange_kwargs = {}

    extra = " c" if is_complex else ""
    pattern = f"{left} h w{extra} -> {right} h w{extra}"

    rearranged = [rearrange(data, pattern, **rearrange_kwargs) for data in mri_data]

    if args.do_mapping_shuffle and temporal_shuffle and reverse:
        for i in range(len(rearranged)):
            rearranged[i] = rearranged[i][:, temporal_shuffle.argsort(), :, :, :, :, :]

    return rearranged if len(rearranged) > 1 else rearranged[0]


class SSIML1Loss(object):
    def __init__(self, device, ssim_scale=10, l1_scale=1):
        self.ssim_loss = SSIMLoss(spatial_dims=2).to(device)
        self.l1_loss = torch.nn.L1Loss().to(device)
        self.ssim_scale = ssim_scale
        self.l1_scale = l1_scale

    def __call__(self, output, target):
        ssim_loss = self.ssim_loss(output, target)
        l1_loss = self.l1_loss(output, target)
        # print(f"ssim_loss: {ssim_loss}, l1_loss: {l1_loss}")
        return {
            "combined_loss": ssim_loss * self.ssim_scale + l1_loss * self.l1_scale,
            "ssim_loss": ssim_loss.detach().cpu().item(),
            "l1_loss": l1_loss.detach().cpu().item(),
        }


def get_loss_function(args, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.loss_type == "l1":
        loss_function = torch.nn.L1Loss().to(device)
    elif args.loss_type == "l2":
        loss_function = torch.nn.MSELoss().to(device)
    elif args.loss_type == "ssim":
        loss_function = SSIMLoss(spatial_dims=2, ssim_scale=10).to(device)
    elif args.loss_type == "ssim_l1":
        loss_function = SSIML1Loss(device)
    else:
        raise ValueError(f"Loss function {args.loss_type} not supported.")

    return loss_function


class SSIMLoss(_Loss):
    """
    Compute 1 - SSIM, allowing per-sample data_range.

    If `data_range` is a float or 0‑D tensor, behaves exactly as before.
    If you set `loss.data_range = tensor_of_shape(batch,)`, then each
    sample i will be evaluated with data_range = data_range[i].
    """

    def __init__(
        self,
        spatial_dims: int,
        data_range: float = 1.0,
        kernel_type: str = "gaussian",
        win_size: int | Sequence[int] = 11,
        kernel_sigma: float | Sequence[float] = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        reduction: str = "mean",
        ssim_scale: float = 1.0,
    ):
        super().__init__(reduction=LossReduction(reduction).value)
        self.spatial_dims = spatial_dims

        # _data_range can now be float or 1-D tensor
        self._data_range = data_range

        self.kernel_type = kernel_type
        self.kernel_size = win_size
        self.kernel_sigma = kernel_sigma
        self.k1 = k1
        self.k2 = k2
        self.ssim_scale = ssim_scale

        # underlying metric uses a single scalar data_range at init
        self.ssim_metric = SSIMMetric(
            spatial_dims=spatial_dims,
            data_range=float(data_range),
            kernel_type=kernel_type,
            win_size=win_size,
            kernel_sigma=kernel_sigma,
            k1=k1,
            k2=k2,
        )

    @property
    def data_range(self) -> float | torch.Tensor:
        return self._data_range

    @data_range.setter
    def data_range(self, value: float | torch.Tensor) -> None:
        # store whatever the user gives us
        self._data_range = value
        # only push scalar values into the metric
        if not isinstance(value, torch.Tensor):
            self.ssim_metric.data_range = float(value)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input:  (B, C, H, W[, D])
            target: (B, C, H, W[, D])
        Returns:
            loss = 1 - SSIM, averaged/summed per `self.reduction`
        """
        dr = self._data_range
        orig_metric_dr = self.ssim_metric.data_range

        # case 1: per-sample data_range
        if isinstance(dr, torch.Tensor) and dr.dim() == 1:
            bs = input.shape[0]
            # collect per-sample SSIMs
            ssim_vals = []
            for i in range(bs):
                # pull out scalar for this sample
                dr_i = dr[i].item()
                # temporarily override metric
                self.ssim_metric.data_range = dr_i
                # compute SSIM for this single-sample batch
                # _compute_tensor yields shape [1, 1]
                ssim_i = self.ssim_metric._compute_tensor(input[i : i + 1], target[i : i + 1]).view(1, 1)
                ssim_vals.append(ssim_i)
            # restore the metric’s data_range
            self.ssim_metric.data_range = orig_metric_dr
            # stack into (B,1)
            ssim_value = torch.cat(ssim_vals, dim=0)

        # case 2: global scalar data_range
        else:
            ssim_value = self.ssim_metric._compute_tensor(input, target).view(-1, 1)

        loss = (1.0 - ssim_value) * self.ssim_scale

        if self.reduction == LossReduction.MEAN.value:
            return loss.mean()
        if self.reduction == LossReduction.SUM.value:
            return loss.sum()
        return loss  # "none"


def gather_metric(metric_value, device, world_size):
    """Convert a scalar metric to a tensor, gather across processes, and compute the mean."""
    tensor_val = torch.tensor(metric_value, device=device)
    gathered = [torch.empty_like(tensor_val) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, tensor_val)
    return torch.cat(gathered).mean().item()


def postprocess_mri_recon(recon, args, file_type=None, is_training=False, pp_z_score_norm=False, pp_norm=False):
    dataset = args.dataset.lower()
    if is_training and pp_norm:
        # Training mode postprocessing
        if dataset == "cmrxrecon" or dataset == "fastmri":
            center = center_crop(recon, [recon.shape[-2] // 2, recon.shape[-1] // 3])
            if args.do_center_crop:
                recon = center
            # z score normalization...
            if pp_z_score_norm:
                recon = (recon - center.mean(dim=[-2, -1], keepdim=True)) / (
                    center.std(dim=[-2, -1], keepdim=True) + 1e-8
                )
            else:
                recon = recon / torch.quantile(center.flatten(start_dim=1), q=0.995, dim=-1, keepdim=True).view(
                    -1, *([1] * (recon.dim() - 1))
                ).clamp_min(1e-8)

    elif not is_training:
        # Evaluation mode postprocessing
        recon = recon.transpose()
        if dataset == "cmrxrecon":
            # if 'Center' in file_type: # CMRxRecon 2025
            #     recon = run4Ranking2025(recon, file_type)
            # else:                     # CMRxRecon 2024
            recon = run4Ranking(recon, file_type)

    return recon


def crop_k_space(recon, k_crop_size):
    if recon.shape[-3] == k_crop_size[0] and recon.shape[-2] == k_crop_size[1]:
        return recon
    recon_k = torch.view_as_complex(fftn_centered(recon, spatial_dims=2, is_complex=True))
    cropped_recon_k = center_crop(recon_k, k_crop_size)
    cropped_recon = ifftn_centered(torch.view_as_real(cropped_recon_k), spatial_dims=2, is_complex=True)

    return cropped_recon


def circular_shift_in_kspace(kspace: torch.Tensor, shift: tuple) -> torch.Tensor:
    """
    Applies a circular shift to an image whose frequency representation is given in an FFT-shifted domain.

    The input is assumed to be:
      input_fftshifted = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(image), norm=norm))

    The spatial shift is applied via phase modulation using:
         exp(-2πi * (shift_y * f_y + shift_x * f_x))
    where f_y and f_x are the frequency components in the shifted frequency grid.

    Parameters:
    -----------
    input_fftshifted : torch.Tensor
        A complex-valued tensor of shape (H, W) (or with extra batch/channels dimensions)
        that represents the FFT-shifted frequency image.
    shift : tuple of int
        (shift_y, shift_x) spatial shifts.
    norm : str, optional
        Normalization mode used in FFT/IFFT, e.g. "ortho". Default is "ortho".

    Returns:
    --------
    torch.Tensor
        The circularly shifted image in the spatial domain (real part).
    """
    # Get the spatial dimensions from the last two dims.
    H, W = kspace.shape[-2], kspace.shape[-1]

    # Compute the frequency bins using torch.fft.fftfreq.
    # These are in the "natural" order: e.g. [0, 1, ..., -1]
    freq_y = torch.fft.fftfreq(H, d=1.0).to(kspace.device)  # for rows
    freq_x = torch.fft.fftfreq(W, d=1.0).to(kspace.device)  # for cols

    # Create 2D grids for the frequencies.
    grid_y, grid_x = torch.meshgrid(freq_y, freq_x, indexing="ij")

    # Since the input is FFT-shifted, shift the frequency grids so they match the ordering.
    # That is, negative frequencies will appear in the left/top parts as in the input.
    grid_y = torch.fft.fftshift(grid_y)
    grid_x = torch.fft.fftshift(grid_x)

    # Create the phase modulation factor using the Fourier shift theorem:
    # The phase factor is: exp(-2πi*(shift_y * f_y + shift_x * f_x))
    phase = torch.exp(-2j * math.pi * (grid_y * shift[0] + grid_x * shift[1]))

    # Apply the phase modulation in the frequency domain.
    modulated_kspace = kspace * phase

    return modulated_kspace


def get_reader(args, is_testing=False):
    if args.dataset.lower() == "fastmri":
        data_reader = FastMRIReader(is_testing=is_testing, uniform_input_kspace=(384, 384))
    elif args.dataset.lower() == "cmrxrecon":
        data_reader = CMRxReconReader(fixed_mask_types=args.fixed_mask_types)
    elif args.dataset.lower() == "cest":
        data_reader = CestMRIReader()
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported.")
    return data_reader


if __name__ == "__main__":
    # Create a sample image (for instance a 2D gradient).
    H, W = 128, 128
    im = torch.linspace(0, 100, steps=H * W).reshape(H, W)

    # Compute the FFT with appropriate shifts.
    input_fftshifted = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(im), norm="ortho"))

    # Specify the desired spatial shift: (shift_y, shift_x)
    shift = (-10, 20)

    # Apply the circular shift using the frequency domain (already fftshifted input).
    shifted_image = ifftn_centered(
        circular_shift_in_kspace(input_fftshifted, shift),
        spatial_dims=2,
        is_complex=False,
    )[:, :, 0]

    # For validation, you can compare with torch.roll which performs a circular shift directly.
    rolled_image = torch.roll(im, shifts=shift, dims=(-2, -1))

    # Check the maximum difference to see if they're nearly identical.
    print("Max difference:", (shifted_image - rolled_image).abs().max().item())
