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

import copy
from collections.abc import Sequence

import torch
import torch.nn as nn
from monai.apps.reconstruction.complex_utils import complex_abs_t
from monai.apps.reconstruction.mri_utils import root_sum_of_squares_t
from monai.apps.reconstruction.networks.nets.utils import (
    complex_normalize,
    divisible_pad_t,
    inverse_divisible_pad_t,
    reshape_channel_complex_to_last_dim,
    reshape_complex_to_channel_dim,
    sensitivity_map_expand,
    sensitivity_map_reduce,
)
from monai.networks.blocks.fft_utils_t import ifftn_centered_t
from monai.networks.nets import DynUNet
from monai.networks.nets.basic_unet import BasicUNet
from torch import Tensor
from utils import reshape_batch_channel_to_channel_dim, reshape_channel_to_batch_dim


class ComplexUnet(nn.Module):
    """
    This variant of U-Net handles complex-value input/output. It can be
    used as a model to learn sensitivity maps in multi-coil MRI data. It is
    built based on :py:class:`monai.networks.nets.BasicUNet` by default but the user
    can input their convolutional model as well.
    ComplexUnet also applies default normalization to the input which makes it more stable to train.

    The data being a (complex) 2-channel tensor is a requirement for using this model.

    Modified and adopted from: https://github.com/facebookresearch/fastMRI

    Args:
        spatial_dims: number of spatial dimensions.
        features: six integers as numbers of features. denotes number of channels in each layer.
        act: activation type and arguments. Defaults to LeakyReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        bias: whether to have a bias term in convolution blocks. Defaults to True.
        dropout: dropout ratio. Defaults to 0.0.
        upsample: upsampling mode, available options are
            ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
        pad_factor: an integer denoting the number which each padded dimension will be divisible to.
            For example, 16 means each dimension will be divisible by 16 after padding
        conv_net: the learning model used inside the ComplexUnet. The default
            is :py:class:`monai.networks.nets.basic_unet`. The only requirement on the model is to
            have 2 as input and output number of channels.
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256),
        strides: Sequence = (1, 2, 2, (2, 2, 1), (2, 2, 1)),
        kernel_size: Sequence = ((3, 3, 3), (3, 3, 3), (3, 3, 1), (3, 3, 1), (3, 3, 1)),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
        upsample: str = "deconv",
        pad_factor: int = 16,
        conv_net: nn.Module | None = None,
        use_dyn_unet: bool = False,
    ):
        super().__init__()
        self.unet: nn.Module
        if conv_net is None:
            if use_dyn_unet:
                if spatial_dims == 2:
                    strides = (1, 2, 2, 2, 2)
                    kernel_size = ((3, 3), (3, 3), (3, 3), (3, 3), (3, 3))
                elif spatial_dims == 3:
                    strides = (1, 2, 2, (2, 2, 1), (2, 2, 1))
                    kernel_size = (
                        (3, 3, 3),
                        (3, 3, 3),
                        (3, 3, 1),
                        (3, 3, 1),
                        (3, 3, 1),
                    )
                self.unet = DynUNet(
                    spatial_dims=spatial_dims,
                    in_channels=2,
                    out_channels=2,
                    kernel_size=kernel_size,
                    strides=strides,
                    upsample_kernel_size=strides[1:],
                    filters=features,
                    act_name=act,
                    norm_name=norm,
                    res_block=True,
                )
            else:
                self.unet = BasicUNet(
                    spatial_dims=spatial_dims,
                    in_channels=2,
                    out_channels=2,
                    features=list(features) + [features[0]],
                    act=act,
                    norm=norm,
                    bias=bias,
                    dropout=dropout,
                    upsample=upsample,
                )
        else:
            # assume the first layer is convolutional and
            # check whether in_channels == 2
            params = [p.shape for p in conv_net.parameters()]
            if params[0][1] != 2:
                raise ValueError(f"in_channels should be 2 but it's {params[0][1]}.")
            self.unet = conv_net

        self.pad_factor = pad_factor

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: input of shape (B,C,H,W,2) for 2D data or (B,C,H,W,D,2) for 3D data

        Returns:
            output of shape (B,C,H,W,2) for 2D data or (B,C,H,W,D,2) for 3D data
        """
        # suppose the input is 2D, the comment in front of each operator below shows the shape after that operator
        x = reshape_complex_to_channel_dim(x)  # x will be of shape (B,C*2,H,W)
        x, mean, std = complex_normalize(x)  # x will be of shape (B,C*2,H,W)
        # pad input
        x, padding_sizes = divisible_pad_t(
            x, k=self.pad_factor
        )  # x will be of shape (B,C*2,H',W') where H' and W' are for after padding

        x = self.unet(x)
        # inverse padding
        x = inverse_divisible_pad_t(x, padding_sizes)  # x will be of shape (B,C*2,H,W)

        x = x * std + mean
        x = reshape_channel_complex_to_last_dim(x)  # x will be of shape (B,C,H,W,2)
        return x


class CoilSensitivityModel(nn.Module):
    """
    This class uses a convolutional model to learn coil sensitivity maps for multi-coil MRI reconstruction.
    The convolutional model is :py:class:`monai.apps.reconstruction.networks.nets.complex_unet` by default
    but can be specified by the user as well. Learning is done on the center of the under-sampled
    kspace (that region is fully sampled).

    The data being a (complex) 2-channel tensor is a requirement for using this model.

    Modified and adopted from: https://github.com/facebookresearch/fastMRI

    Args:
        spatial_dims: number of spatial dimensions.
        features: six integers as numbers of features. denotes number of channels in each layer.
        act: activation type and arguments. Defaults to LeakyReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        bias: whether to have a bias term in convolution blocks. Defaults to True.
        dropout: dropout ratio. Defaults to 0.0.
        upsample: upsampling mode, available options are
            ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
        coil_dim: coil dimension in the data
        conv_net: the learning model used to estimate the coil sensitivity maps. default
            is :py:class:`monai.apps.reconstruction.networks.nets.complex_unet`. The only
            requirement on the model is to have 2 as input and output number of channels.
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.01, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
        upsample: str = "deconv",
        coil_dim: int = 1,
        conv_net: nn.Module | None = None,
        use_fully_sampled_region_only: bool = False,
        use_dyn_unet: bool = False,
    ):
        super().__init__()
        if conv_net is None:
            self.conv_net = ComplexUnet(
                spatial_dims=spatial_dims,
                features=features,
                act=act,
                norm=norm,
                bias=bias,
                dropout=dropout,
                upsample=upsample,
                use_dyn_unet=use_dyn_unet,
            )
        else:
            # assume the first layer is convolutional and
            # check whether in_channels == 2
            params = [p.shape for p in conv_net.parameters()]
            if params[0][1] != 2:
                raise ValueError(f"in_channels should be 2 but it's {params[0][1]}.")
            self.conv_net = conv_net  # type: ignore
        self.spatial_dims = spatial_dims
        self.coil_dim = coil_dim
        self.use_fully_sampled_region_only = use_fully_sampled_region_only

    def get_fully_sampled_region(self, mask: Tensor) -> tuple[int, int]:
        left = right = mask.shape[-2] // 2
        while mask[..., right, :]:
            right += 1

        while mask[..., left, :]:
            left -= 1

        if (right - left) <= 5:
            return 0, mask.shape[-2] - 1
        else:
            return left + 1, right

    def forward(self, masked_kspace: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            masked_kspace: the under-sampled kspace (which is the input measurement). Its shape
                is (B,C,H,W,2) for 2D data or (B,C,H,W,D,2) for 3D data.
            mask: the under-sampling mask with shape (1,1,1,W,1) for 2D data or (1,1,1,1,D,1) for 3D data.

        Returns:
            predicted coil sensitivity maps with shape (B,C,H,W,2) for 2D data or (B,C,H,W,D,2) for 3D data.
        """
        if self.use_fully_sampled_region_only:
            left, right = self.get_fully_sampled_region(mask)
            num_low_freqs = right - left  # size of the fully-sampled center

            # take out the fully-sampled region and set the rest of the data to zero
            x = torch.zeros_like(masked_kspace)
            start = (mask.shape[-2] - num_low_freqs + 1) // 2  # this marks the start of center extraction
            x[..., start : start + num_low_freqs, :] = masked_kspace[..., start : start + num_low_freqs, :]
        else:
            x = masked_kspace

        # apply inverse fourier to the extracted fully-sampled data
        x = ifftn_centered_t(x, spatial_dims=self.spatial_dims, is_complex=True)

        x, b = reshape_channel_to_batch_dim(x)  # shape of x will be (B*C,1,...)
        x = self.conv_net(x)
        x = reshape_batch_channel_to_channel_dim(x, b)  # shape will be (B,C,...)
        # normalize the maps
        x = x / root_sum_of_squares_t(x, spatial_dim=self.coil_dim).unsqueeze(self.coil_dim)

        return x


class VarNetBlock(nn.Module):
    """
    A variational block based on Sriram et. al., "End-to-end variational networks for accelerated MRI reconstruction".
    It applies data consistency and refinement to the intermediate kspace and combines those results.

    Modified and adopted from: https://github.com/facebookresearch/fastMRI

    Args:
        refinement_model: the model used for refinement (typically a U-Net but can be any deep learning model
            that performs well when the input and output are in image domain (e.g., a convolutional network).
        spatial_dims: is 2 for 2D data and is 3 for 3D data
    """

    def __init__(self, refinement_model: nn.Module, spatial_dims: int = 2):
        super().__init__()
        self.model = refinement_model
        self.spatial_dims = spatial_dims
        self.dc_weight = nn.Parameter(torch.ones(1))  # learned scalar as the multiplier of the DC block

        buffer_shape = [1 for _ in range(spatial_dims + 3)]  # 3 denotes the batch, channel, and real/complex dimensions
        self.register_buffer("zeros", torch.zeros(buffer_shape))

    def soft_dc(self, x: Tensor, ref_kspace: Tensor, mask: Tensor) -> Tensor:
        """
        Applies data consistency to input x. Suppose x is an intermediate estimate of the kspace and ref_kspace
        is the reference under-sampled measurement. This function returns mask * (x - ref_kspace). View this as the
        residual between the original under-sampled kspace and the estimate given by the network.

        Args:
            x: 2D kspace (B,C,H,W,2) with the last dimension being 2 (for real/imaginary parts) and C denoting the
                coil dimension. 3D data will have the shape (B,C,H,W,D,2).
            ref_kspace: original under-sampled kspace with the same shape as x.
            mask: the under-sampling mask with shape (1,1,1,W,1) for 2D data or (1,1,1,1,D,1) for 3D data.

        Returns:
            Output of DC block with the same shape as x
        """
        return torch.where(mask, x - ref_kspace, self.zeros) * self.dc_weight

    def forward(
        self,
        current_kspace: Tensor,
        ref_kspace: Tensor,
        mask: Tensor,
        sens_maps: Tensor,
    ) -> Tensor:
        """
        Args:
            current_kspace: Predicted kspace from the previous block. It's a 2D kspace (B,C,H,W,2)
                with the last dimension being 2 (for real/imaginary parts) and C denoting the
                coil dimension. 3D data will have the shape (B,C,H,W,D,2).
            ref_kspace: reference kspace for applying data consistency (is the under-sampled kspace in MRI reconstruction).
                Its shape is the same as current_kspace.
            mask: the under-sampling mask with shape (1,1,1,W,1) for 2D data or (1,1,1,1,D,1) for 3D data.
            sens_maps: coil sensitivity maps with the same shape as current_kspace

        Returns:
            Output of VarNetBlock with the same shape as current_kspace
        """
        dc_out = self.soft_dc(current_kspace, ref_kspace, mask)  # output of DC block
        refinement_out = sensitivity_map_expand(
            self.model(sensitivity_map_reduce(current_kspace, sens_maps, spatial_dims=self.spatial_dims)),
            sens_maps,
            spatial_dims=self.spatial_dims,
        )  # output of refinement model
        output = current_kspace - dc_out - refinement_out
        return output


class VariationalNetworkModel(nn.Module):
    """
    The end-to-end variational network (or simply e2e-VarNet) based on Sriram et. al., "End-to-end variational
    networks for accelerated MRI reconstruction".
    It comprises several cascades each consisting of refinement and data consistency steps. The network takes in
    the under-sampled kspace and estimates the ground-truth reconstruction.

    Modified and adopted from: https://github.com/facebookresearch/fastMRI

    Args:
        coil_sensitivity_model: A convolutional model for learning coil sensitivity maps. An example is
            :py:class:`monai.apps.reconstruction.networks.nets.coil_sensitivity_model.CoilSensitivityModel`.
        refinement_model: A convolutional network used in the refinement step of e2e-VarNet. An example
            is :py:class:`monai.apps.reconstruction.networks.nets.complex_unet.ComplexUnet`.
        num_cascades: Number of cascades. Each cascade is a
            :py:class:`monai.apps.reconstruction.networks.blocks.varnetblock.VarNetBlock` which consists of
            refinement and data consistency steps.
        spatial_dims: number of spatial dimensions.
    """

    def __init__(
        self,
        coil_sensitivity_model: nn.Module,
        refinement_model: nn.Module,
        num_cascades: int = 12,
        spatial_dims: int = 2,
        disable_coil_sensitivity_model: bool = False,
        return_multi_coil_complex: bool = False,
    ):
        super().__init__()
        if disable_coil_sensitivity_model:
            self.coil_sensitivity_model = None
        else:
            self.coil_sensitivity_model = coil_sensitivity_model
        self.cascades = nn.ModuleList(
            [VarNetBlock(copy.deepcopy(refinement_model), spatial_dims=spatial_dims) for i in range(num_cascades)]
        )
        self.spatial_dims = spatial_dims
        self.disable_coil_sensitivity_model = disable_coil_sensitivity_model
        self.return_multi_coil_complex = return_multi_coil_complex

    def forward(self, masked_kspace: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            masked_kspace: The under-sampled kspace. It's a 2D kspace (B,C,H,W,2)
                with the last dimension being 2 (for real/imaginary parts) and C denoting the
                coil dimension. 3D data will have the shape (B,C,H,W,D,2).
            mask: The under-sampling mask with shape (1,1,1,W,1) for 2D data or (1,1,1,1,D,1) for 3D data.

        Returns:
            The reconstructed image which is the root sum of squares (rss) of the absolute value
                of the inverse fourier of the predicted kspace (note that rss combines coil images into one image).
        """
        if self.disable_coil_sensitivity_model:
            sensitivity_maps = torch.ones_like(masked_kspace)  # shape is similar to masked_kspace
        else:
            sensitivity_maps = self.coil_sensitivity_model(masked_kspace, mask)  # shape is similar to masked_kspace
        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sensitivity_maps)

        if self.return_multi_coil_complex:
            output_image = ifftn_centered_t(kspace_pred, spatial_dims=self.spatial_dims)
        else:
            output_image = root_sum_of_squares_t(
                complex_abs_t(ifftn_centered_t(kspace_pred, spatial_dims=self.spatial_dims)),
                spatial_dim=1,  # 1 is for C which is the coil dimension
            )  # shape is (B,H,W) for 2D and (B,H,W,D) for 3D data.
        return output_image


class ComplexUnet_DCAE(nn.Module):
    """
    This variant of U-Net handles complex-value input/output. It can be
    used as a model to learn sensitivity maps in multi-coil MRI data. It is
    built based on :py:class:`monai.networks.nets.BasicUNet` by default but the user
    can input their convolutional model as well.
    ComplexUnet also applies default normalization to the input which makes it more stable to train.

    The data being a (complex) 2-channel tensor is a requirement for using this model.

    Modified and adopted from: https://github.com/facebookresearch/fastMRI

    Args:
        spatial_dims: number of spatial dimensions.
        features: six integers as numbers of features. denotes number of channels in each layer.
        act: activation type and arguments. Defaults to LeakyReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        bias: whether to have a bias term in convolution blocks. Defaults to True.
        dropout: dropout ratio. Defaults to 0.0.
        upsample: upsampling mode, available options are
            ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
        pad_factor: an integer denoting the number which each padded dimension will be divisible to.
            For example, 16 means each dimension will be divisible by 16 after padding
        conv_net: the learning model used inside the ComplexUnet. The default
            is :py:class:`monai.networks.nets.basic_unet`. The only requirement on the model is to
            have 2 as input and output number of channels.
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256),
        strides: Sequence = (1, 2, 2, (2, 2, 1), (2, 2, 1)),
        kernel_size: Sequence = ((3, 3, 3), (3, 3, 3), (3, 3, 1), (3, 3, 1), (3, 3, 1)),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
        upsample: str = "deconv",
        pad_factor: int = 16,
        conv_net: nn.Module | None = None,
        use_dyn_unet: bool = False,
    ):
        super().__init__()
        self.unet: nn.Module
        if conv_net is None:
            if use_dyn_unet:
                if spatial_dims == 2:
                    strides = (1, 2, 2, 2, 2)
                    kernel_size = ((3, 3), (3, 3), (3, 3), (3, 3), (3, 3))
                elif spatial_dims == 3:
                    strides = (1, 2, 2, (2, 2, 1), (2, 2, 1))
                    kernel_size = (
                        (3, 3, 3),
                        (3, 3, 3),
                        (3, 3, 1),
                        (3, 3, 1),
                        (3, 3, 1),
                    )
                self.unet = DynUNet(
                    spatial_dims=spatial_dims,
                    in_channels=2,
                    out_channels=2,
                    kernel_size=kernel_size,
                    strides=strides,
                    upsample_kernel_size=strides[1:],
                    filters=features,
                    act_name=act,
                    norm_name=norm,
                    res_block=True,
                )
            else:
                self.unet = BasicUNet(
                    spatial_dims=spatial_dims,
                    in_channels=2,
                    out_channels=2,
                    features=list(features) + [features[0]],
                    act=act,
                    norm=norm,
                    bias=bias,
                    dropout=dropout,
                    upsample=upsample,
                )
        else:
            # assume the first layer is convolutional and
            # check whether in_channels == 2
            params = [p.shape for p in conv_net.parameters()]
            if params[0][1] != 2:
                raise ValueError(f"in_channels should be 2 but it's {params[0][1]}.")
            self.unet = conv_net

        self.pad_factor = pad_factor

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: input of shape (B,C,H,W,2) for 2D data or (B,C,H,W,D,2) for 3D data

        Returns:
            output of shape (B,C,H,W,2) for 2D data or (B,C,H,W,D,2) for 3D data
        """
        # suppose the input is 2D, the comment in front of each operator below shows the shape after that operator
        x = reshape_complex_to_channel_dim(x)  # x will be of shape (B,C*2,H,W)
        # pad input
        # possible_no_oversample = x.shape[-1]/x.shape[-2] < 1.75 and x.shape[-2] <= 224
        # if possible_no_oversample:
        #     x = torch.nn.functional.pad(x, (x.shape[-1]//2, x.shape[-1] - x.shape[-1]//2, 0, 0))
        x, padding_sizes = divisible_pad_t(
            x, k=self.pad_factor
        )  # x will be of shape (B,C*2,H',W') where H' and W' are for after padding

        x = self.unet(x)
        # inverse padding
        x = inverse_divisible_pad_t(x, padding_sizes)  # x will be of shape (B,C*2,H,W)
        # if possible_no_oversample:
        #     x = x[..., x.shape[-1] // 4: x.shape[-1] // 4 + x.shape[-1] // 2]
        x = reshape_channel_complex_to_last_dim(x)  # x will be of shape (B,C,H,W,2)
        return x


class CoilSensitivityModel_DCAE(nn.Module):
    """
    This class uses a convolutional model to learn coil sensitivity maps for multi-coil MRI reconstruction.
    The convolutional model is :py:class:`monai.apps.reconstruction.networks.nets.complex_unet` by default
    but can be specified by the user as well. Learning is done on the center of the under-sampled
    kspace (that region is fully sampled).

    The data being a (complex) 2-channel tensor is a requirement for using this model.

    Modified and adopted from: https://github.com/facebookresearch/fastMRI

    Args:
        spatial_dims: number of spatial dimensions.
        features: six integers as numbers of features. denotes number of channels in each layer.
        act: activation type and arguments. Defaults to LeakyReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        bias: whether to have a bias term in convolution blocks. Defaults to True.
        dropout: dropout ratio. Defaults to 0.0.
        upsample: upsampling mode, available options are
            ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
        coil_dim: coil dimension in the data
        conv_net: the learning model used to estimate the coil sensitivity maps. default
            is :py:class:`monai.apps.reconstruction.networks.nets.complex_unet`. The only
            requirement on the model is to have 2 as input and output number of channels.
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        features: Sequence[int] = (12, 24, 48, 96, 192),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.01, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
        upsample: str = "deconv",
        coil_dim: int = 1,
        conv_net: nn.Module | None = None,
        use_dyn_unet: bool = False,
        pad_factor: int = 2**4,
    ):
        super().__init__()
        if conv_net is None:
            self.conv_net = ComplexUnet_DCAE(
                spatial_dims=spatial_dims,
                features=features,
                act=act,
                norm=norm,
                bias=bias,
                dropout=dropout,
                upsample=upsample,
                use_dyn_unet=use_dyn_unet,
                pad_factor=pad_factor,
            )
        else:
            # assume the first layer is convolutional and
            # check whether in_channels == 2
            params = [p.shape for p in conv_net.parameters()]
            if params[0][1] != 2:
                raise ValueError(f"in_channels should be 2 but it's {params[0][1]}.")
            self.conv_net = conv_net  # type: ignore
        self.spatial_dims = spatial_dims
        self.coil_dim = coil_dim

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            masked_kspace: the under-sampled kspace (which is the input measurement). Its shape
                is (B,C,H,W,2) for 2D data or (B,C,H,W,D,2) for 3D data.
            mask: the under-sampling mask with shape (1,1,1,W,1) for 2D data or (1,1,1,1,D,1) for 3D data.

        Returns:
            predicted coil sensitivity maps with shape (B,C,H,W,2) for 2D data or (B,C,H,W,D,2) for 3D data.
        """
        x, b = reshape_channel_to_batch_dim(input)  # shape of x will be (B*C,1,...)
        x = self.conv_net(x)
        x = reshape_batch_channel_to_channel_dim(x, b)  # shape will be (B,C,...)
        # normalize the maps
        x = x / root_sum_of_squares_t(x, spatial_dim=self.coil_dim).unsqueeze(self.coil_dim)

        return x
