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

import json
import os
import random
import re
from collections.abc import Sequence

import numpy as np
import scipy
from monai.config import PathLike
from monai.data.image_reader import ImageReader
from monai.data.utils import is_supported_format
from monai.utils import StrEnum, optional_import, require_pkg
from numpy import ndarray
from scipy.fft import fftn, fftshift, ifftn, ifftshift

h5py, has_h5py = optional_import("h5py")

__all__ = ["FastMRIReader", "CestMRIReader", "CMRxReconReader"]


class FastMRIKeys(StrEnum):
    """
    The keys to be used for extracting data from the fastMRI dataset
    """

    KSPACE = "kspace"
    MASK = "mask"
    FILENAME = "filename"
    RECON = "reconstruction_rss"
    ACQUISITION = "acquisition"
    MAX = "max"
    NORM = "norm"
    PID = "patient_id"


class CestMRIKeys(StrEnum):
    """
    The keys to be used for extracting data from the CestMRI dataset
    """

    KSPACE = "kspace"
    MASK = "mask"
    FILENAME = "filename"
    CSM = "sensitivity_maps"
    RECON_RSS = "reconstruction_rss"
    RECON_SENSE = "reconstruction_sense"
    ACQUISITION = "acquisition"
    MAX = "max"
    NORM = "norm"
    PID = "patient_id"
    SHAPE = "shape"


class CMRxReconKeys(StrEnum):
    """
    The keys to be used for extracting data from the CMRxRecon dataset
    """

    KSPACE = "kspace_full"
    MASK = "mask"
    MASK_TYPE = "mask_type"
    FILENAME = "filename"
    RECON = "reconstruction_rss"
    RECON_RAW = "reconstruction"
    ACQUISITION = "acquisition"
    MAX = "max"
    NORM = "norm"
    PID = "patient_id"
    NUM_SLICES = "num_slices"
    NUM_COILS = "num_coils"
    NUM_FRAMES = "num_frames"
    SHAPE = "shape"


@require_pkg(pkg_name="h5py")
class FastMRIReader(ImageReader):
    """
    Load fastMRI files with '.h5' suffix. fastMRI files, when loaded with "h5py",
    are HDF5 dictionary-like datasets. The keys are:

    - kspace: contains the fully-sampled kspace
    - reconstruction_rss: contains the root sum of squares of ifft of kspace. This
        is the ground-truth image.

    It also has several attributes with the following keys:

    - acquisition (str): acquisition mode of the data (e.g., AXT2 denotes T2 brain MRI scans)
    - max (float): dynamic range of the data
    - norm (float): norm of the kspace
    - patient_id (str): the patient's id whose measurements were recorded
    """

    def verify_suffix(self, filename: Sequence[PathLike] | PathLike) -> bool:
        """
         Verify whether the specified file format is supported by h5py reader.

        Args:
             filename: file name
        """
        suffixes: Sequence[str] = [".h5"]
        return has_h5py and is_supported_format(filename, suffixes)

    def read(self, data: Sequence[PathLike] | PathLike) -> dict:  # type: ignore
        """
        Read data from specified h5 file.
        Note that the returned object is a dictionary.

        Args:
            data: file name to read.
        """
        if isinstance(data, (tuple, list)):
            data = data[0]

        with h5py.File(data, "r") as f:
            # extract everything from the ht5 file
            dat = dict(
                [(key, f[key][()]) for key in f]
                + [(key, f.attrs[key]) for key in f.attrs]
                + [(FastMRIKeys.FILENAME, os.path.basename(data))]  # type: ignore
            )
        f.close()

        return dat

    def get_data(self, dat: dict) -> tuple[ndarray, dict]:
        """
        Extract data array and metadata from the loaded data and return them.
        This function returns two objects, first is numpy array of image data, second is dict of metadata.

        Args:
            dat: a dictionary loaded from an h5 file
        """
        header = self._get_meta_dict(dat)
        data: ndarray = np.array(dat[FastMRIKeys.KSPACE])[np.newaxis, ...]
        header[FastMRIKeys.MASK] = (
            np.expand_dims(np.array(dat[FastMRIKeys.MASK]), 0)[None, ..., None]
            if FastMRIKeys.MASK in dat.keys()
            else np.zeros(data.shape)
        )
        data_shape = data.shape
        header[CMRxReconKeys.NUM_FRAMES] = data_shape[0]
        header[CMRxReconKeys.NUM_SLICES] = data_shape[1]
        header[CMRxReconKeys.NUM_COILS] = data_shape[2]
        header[CMRxReconKeys.SHAPE] = np.array(data_shape)
        mask = np.ones([1] * data.ndim)
        header[CMRxReconKeys.MASK] = mask.astype(np.float32)
        header[FastMRIKeys.ACQUISITION] = header[FastMRIKeys.ACQUISITION].replace("AXT1PRE", "AXT1")
        return data, header

    def _get_meta_dict(self, dat: dict) -> dict:
        """
        Get all the metadata of the loaded dict and return the meta dict.

        Args:
            dat: a dictionary object loaded from an h5 file.
        """
        return {k.value: dat[k.value] for k in FastMRIKeys if k.value in dat}


class CestMRIReader(ImageReader):
    """
    Load fastMRI files with '.h5' suffix. fastMRI files, when loaded with "h5py",
    are HDF5 dictionary-like datasets. The keys are:

    - kspace: contains the fully-sampled kspace
    - reconstruction_rss: contains the root sum of squares of ifft of kspace. This
        is the ground-truth image.

    It also has several attributes with the following keys:

    - acquisition (str): acquisition mode of the data (e.g., AXT2 denotes T2 brain MRI scans)
    - max (float): dynamic range of the data
    - norm (float): norm of the kspace
    - patient_id (str): the patient's id whose measurements were recorded
    """

    def verify_suffix(self, filename: Sequence[PathLike] | PathLike) -> bool:
        """
         Verify whether the specified file format is supported by h5py reader.

        Args:
             filename: file name
        """
        suffixes: Sequence[str] = [".h5"]
        return has_h5py and is_supported_format(filename, suffixes)

    def read(self, data: Sequence[PathLike] | PathLike) -> dict:  # type: ignore
        """
        Read data from specified h5 file.
        Note that the returned object is a dictionary.

        Args:
            data: file name to read.
        """
        if isinstance(data, (tuple, list)):
            data = data[0]

        with h5py.File(data, "r") as f:
            # extract everything from the ht5 file
            dat = dict(
                [(key, f[key][()]) for key in f]
                + [(key, f.attrs[key]) for key in f.attrs]
                + [(CestMRIKeys.FILENAME, os.path.basename(data))]  # type: ignore
            )
        f.close()

        return dat

    def get_data(self, dat: dict) -> tuple[ndarray, dict]:
        """
        Extract data array and metadata from the loaded data and return them.
        This function returns two objects, first is numpy array of image data, second is dict of metadata.

        Args:
            dat: a dictionary loaded from an h5 file
        """
        header = self._get_meta_dict(dat)
        data: ndarray = np.array(dat[CestMRIKeys.KSPACE])
        data = fftshift(
            ifftn(ifftshift(data, axes=[-3, -2, -1]), axes=[-3, -2, -1], norm="ortho"),
            axes=[-3, -2, -1],
        ).transpose(0, 4, 1, 2, 3)
        data = fftshift(
            fftn(ifftshift(data, axes=[-2, -1]), axes=[-2, -1], norm="ortho"),
            axes=[-2, -1],
        )
        header[CestMRIKeys.MASK] = (
            np.array(dat[CestMRIKeys.MASK]) if CestMRIKeys.MASK in dat.keys() else np.zeros(data.shape)
        )
        header[CestMRIKeys.SHAPE] = np.array(data.shape)
        return data, header

    def _get_meta_dict(self, dat: dict) -> dict:
        """
        Get all the metadata of the loaded dict and return the meta dict.

        Args:
            dat: a dictionary object loaded from an h5 file.
        """
        return {k.value: dat[k.value] for k in CestMRIKeys if k.value in dat}


class CMRxReconReader(ImageReader):
    def __init__(self, fixed_mask_types=None):
        super().__init__()
        self.fixed_mask_types = fixed_mask_types if isinstance(fixed_mask_types, list) else [fixed_mask_types]

    def verify_suffix(self, filename: Sequence[PathLike] | PathLike) -> bool:
        """
         Verify whether the specified file format is supported by h5py reader.

        Args:
             filename: file name
        """
        suffixes: Sequence[str] = [".json"]
        return has_h5py and is_supported_format(filename, suffixes)

    def read_mat(self, mat_file: Sequence[PathLike]) -> list:
        try:
            with h5py.File(mat_file, "r", swmr=True) as f:
                data_kv = [(key, f[key][()]) for key in f]
        except BaseException:
            data = scipy.io.loadmat(mat_file)
            data_kv = [(key, data[key]) for key in data]

        return data_kv

    def filter_masks_by_types(self, masks, fixed_mask_types):
        result = []
        if not all(fixed_mask_types):
            return masks
        for mask in masks:
            if any(mask_type in mask for mask_type in fixed_mask_types):
                result.append(mask)

        return result

    def read(self, data: Sequence[PathLike] | PathLike) -> dict:  # type: ignore
        """
        Read data from specified json file.
        Note that the returned object is a dictionary.

        Args:
            data: file name to read.
        """
        if isinstance(data, (tuple, list)):
            data = data[0]

        with open(data, "r") as f:
            json_data = json.load(f)
            kspace = json_data["kspace"]
            masks = self.filter_masks_by_types(json_data["mask"], self.fixed_mask_types)
            mask = random.choice(masks) if json_data["mask"] else ""
            mask_type = mask.split("_mask_")[-1][:-4]
            acquisition_type = re.search(r"(?:^|[/\\])MultiCoil[/\\]([^/\\]+)", kspace, flags=re.I).group(1)

        kspace_kv = self.read_mat(kspace)
        mask_kv = self.read_mat(mask) if mask else [(None, None)]

        dat = dict(
            kspace_kv
            + mask_kv
            + [
                (CMRxReconKeys.FILENAME, os.path.basename(data)),
                (CMRxReconKeys.MASK_TYPE, mask_type),
                (CMRxReconKeys.ACQUISITION, acquisition_type),
            ]
        )
        return dat

    def get_data(self, dat: dict) -> tuple[ndarray, dict]:
        """
        Extract data array and metadata from the loaded data and return them.
        This function returns two objects, first is numpy array of image data, second is dict of metadata.

        Args:
            dat: a dictionary loaded from a mat file
        """
        header = self._get_meta_dict(dat)
        if "kus" in dat:
            kspace_key = "kus"
        else:
            kspace_key = CMRxReconKeys.KSPACE if CMRxReconKeys.KSPACE in dat else "kspace"
        if np.issubdtype(
            dat[kspace_key].dtype, np.complexfloating
        ):  # return from scipy.io.loadmat is complex ndarray with transposed shape
            data_shape = dat[kspace_key].shape[::-1]
            data_shape = (1,) * (5 - len(data_shape)) + data_shape  # [t, z, c, y, x] or [1, z, c, y, x]
            data: ndarray = dat[kspace_key].transpose()  # .reshape(-1, data_shape[-3], data_shape[-2], data_shape[-1])
        else:
            data_shape = dat[kspace_key]["real"].shape
            data_shape = (1,) * (5 - len(data_shape)) + data_shape  # [t, z, c, y, x] or [1, z, c, y, x]
            data: ndarray = np.array(
                dat[kspace_key]["real"] + 1j * dat[kspace_key]["imag"]
            )
        data = data.reshape(data_shape)

        header[CMRxReconKeys.PID] = os.path.splitext(dat[CMRxReconKeys.FILENAME])[0].split("_")[0]
        header[CMRxReconKeys.NUM_FRAMES] = data_shape[0]
        header[CMRxReconKeys.NUM_SLICES] = data_shape[1]
        header[CMRxReconKeys.NUM_COILS] = data_shape[2]
        header[CMRxReconKeys.SHAPE] = np.array(data_shape)
        if CMRxReconKeys.MASK in dat.keys():
            mask = np.array(dat[CMRxReconKeys.MASK])
            if mask.ndim == 2:  # 2D sampling
                mask = np.expand_dims(mask, axis=(0, 1))
            elif mask.ndim == 3:  # 3D k-t sampling
                mask = np.expand_dims(mask, axis=(1, 2))
        else:
            mask = np.ones([1] * data.ndim)
        header[CMRxReconKeys.MASK] = mask.astype(np.float32)
        return data, header

    def _get_meta_dict(self, dat: dict) -> dict:
        """
        Get all the metadata of the loaded dict and return the meta dict.

        Args:
            dat: a dictionary object loaded from a mat file.
        """
        return {k.value: dat[k.value] for k in CMRxReconKeys if k.value in dat and k != CMRxReconKeys.KSPACE}
