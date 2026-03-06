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

from typing import Optional

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred) ** 2)


def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.array(np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2)


def psnr(gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


def ssim(gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if maxval is None:
        maxval = gt.max()
    return structural_similarity(gt, pred, data_range=maxval)


def normalize_std(array):
    """Normalize an array using mean and standard deviation."""
    mean = np.mean(array)
    std = np.std(array) + 1e-8
    return (array - mean) / std


def normalize_percentile(array):
    # get magnitude
    data = np.abs(array)
    # calculate 99% percentile
    percentile99 = np.percentile(data, 99.5)
    # avoid dividing 0
    if percentile99 == 0:
        return data
    # normalization
    # data_normalize = np.clip(data / percentile99, 0, 1)
    data_normalize = data / percentile99

    return data_normalize


def calmetric(pred_recon, gt_recon, z_score=False):
    if gt_recon.ndim == 4:  # w, h, slice, time
        psnr_array = np.zeros((gt_recon.shape[-2], gt_recon.shape[-1]))
        ssim_array = np.zeros((gt_recon.shape[-2], gt_recon.shape[-1]))
        nmse_array = np.zeros((gt_recon.shape[-2], gt_recon.shape[-1]))

        for i in range(gt_recon.shape[-2]):
            for j in range(gt_recon.shape[-1]):
                pred, gt = pred_recon[:, :, i, j], gt_recon[:, :, i, j]
                # revise the normlization to a more stable way
                if z_score:
                    pred_normalized = normalize_std(pred)
                    gt_normalized = normalize_std(gt)
                else:
                    pred_normalized = normalize_percentile(pred)
                    gt_normalized = normalize_percentile(gt)

                psnr_array[i, j] = psnr(gt_normalized, pred_normalized)
                ssim_array[i, j] = ssim(gt_normalized, pred_normalized)
                nmse_array[i, j] = nmse(gt_normalized, pred_normalized)
    else:  # w, h, slice
        psnr_array = np.zeros((1, gt_recon.shape[-1]))
        ssim_array = np.zeros((1, gt_recon.shape[-1]))
        nmse_array = np.zeros((1, gt_recon.shape[-1]))

        for j in range(gt_recon.shape[-1]):
            pred, gt = pred_recon[:, :, j], gt_recon[:, :, j]

            # revise the normlization to a more stable way
            if z_score:
                pred_normalized = normalize_std(pred)
                gt_normalized = normalize_std(gt)
            else:
                pred_normalized = normalize_percentile(pred)
                gt_normalized = normalize_percentile(gt)

            psnr_array[0, j] = psnr(pred_normalized, gt_normalized)
            ssim_array[0, j] = ssim(pred_normalized, gt_normalized)
            nmse_array[0, j] = nmse(pred_normalized, gt_normalized)

    return psnr_array, ssim_array, nmse_array
