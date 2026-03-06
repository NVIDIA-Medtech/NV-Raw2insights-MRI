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

from math import ceil, floor
from typing import Tuple, Union

import numpy as np
from scipy.ndimage import rotate

# -----------------------------
# Helper functions
# -----------------------------


def center_crop(x: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Center-crop array x to the desired target_shape.
    """
    # For each dimension, compute start and end indices such that
    # start = (dim - target) // 2 and end = start + target.
    slices = tuple(
        (
            slice((dim - target) // 2, (dim - target) // 2 + target)
            if target % 2 == 0
            else slice((dim - target) // 2 + 1, (dim - target) // 2 + 1 + target)
        )
        for dim, target in zip(x.shape, target_shape)
    )
    return x[slices]


def zero_pad(x: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Zero-pad array x to center it in an array of shape target_shape.
    """
    x = np.array(x)
    if x.shape == target_shape:
        return x
    result = np.zeros(target_shape, dtype=x.dtype)
    slices = []
    for orig, target in zip(x.shape, target_shape):
        start = (target - orig) // 2
        slices.append(slice(start, start + orig))
    result[tuple(slices)] = x
    return result


def ind_to_xy(ind: Union[np.ndarray, int], X: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Matlab-style linear indices (1-indexed) to (x, y) coordinates.

    Parameters:
      ind: integer or array of indices (1-indexed).
      X: Number of rows (used for modulo arithmetic).

    Returns:
      Tuple of (x, y) coordinates as numpy arrays (still 1-indexed).
    """
    ind = np.asarray(ind)
    x = ((ind - 1) % X) + 1
    y = ((ind - 1) // X) + 1
    return x, y


def nearest_vacant(dupind: int, empind: np.ndarray, ny: int) -> int:
    """
    Given a duplicate linear index, find the nearest available index among empind,
    considering only candidates with the same column (y-coordinate).
    """
    x0, y0 = ind_to_xy(np.array([dupind]), ny)
    x0, y0 = x0[0], y0[0]
    x, y = ind_to_xy(empind, ny)
    # Allow only candidates with nearly the same y coordinate.
    same_y = np.abs(y - y0) < 1e-10
    if np.any(same_y):
        candidate_indices = np.where(same_y)[0]
    else:
        candidate_indices = np.arange(len(empind))
    candidate_x = x[candidate_indices]
    distances = np.abs(candidate_x - x0)
    min_idx = candidate_indices[np.argmin(distances)]
    return empind[min_idx]


def resolve_duplicate_indices(ph: np.ndarray, ti: np.ndarray, ny: int, nt: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adjust duplicate (phase, time) sample positions in the k-t grid by moving
    duplicates to the nearest vacant positions.
    """
    ph = np.array(ph, dtype=np.int64).copy()
    ti = np.array(ti, dtype=np.int64).copy()

    offset_x = int(ceil((ny + 1) / 2))
    offset_t = int(ceil((nt + 1) / 2))
    ph += offset_x
    ti += offset_t

    # Compute linear indices (1-indexed as in Matlab)
    pt = (ti - 1) * ny + ph

    # Identify duplicate indices
    unique_pts, counts = np.unique(pt, return_counts=True)
    repeated_values = unique_pts[counts != 1]
    dup_indices = []
    for rep in repeated_values:
        indices = np.where(pt == rep)[0]
        if indices.size > 1:
            dup_indices.extend(indices[1:])  # skip first occurrence
    dup_indices = np.array(dup_indices, dtype=np.int64)

    # Determine vacant positions on the grid.
    all_indices = np.arange(1, ny * nt + 1)
    empind = np.setdiff1d(all_indices, pt)

    # For each duplicate, find a nearby available index (using same y-coordinate)
    for idx in dup_indices:
        newind = nearest_vacant(pt[idx], empind, ny)
        pt[idx] = newind
        empind = empind[empind != newind]

    # Convert linear indices back to (phase, time) coordinates.
    new_ph, new_ti = ind_to_xy(pt, ny)
    new_ph -= offset_x
    new_ti -= offset_t
    return new_ph, new_ti


def random_weighted_choice(P: np.ndarray, seed: int, size: Union[int, Tuple[int, ...]]) -> np.ndarray:
    """
    Draw random integers from 1 to len(P) with probabilities proportional to P.

    Parameters:
      P: 1D array of nonnegative probabilities.
      seed: Integer seed for reproducibility.
      size: Output shape (tuple or integer).

    Returns:
      Array of integers (1-indexed) drawn according to the distribution.
    """
    P = np.asarray(P, dtype=np.float64).flatten()
    rng = np.random.default_rng(seed)
    choices = rng.choice(np.arange(1, len(P) + 1), size=size, p=P / np.sum(P))
    return choices


def round_away_from_zero(x):
    # MATLAB-style rounding
    if x > 0:
        return floor(x + 0.5)
    else:
        return ceil(x - 0.5)


# -----------------------------
# Sampling functions
# -----------------------------


def kt_radial_sampling(
    nx: int,
    ny: int,
    nt: int,
    ncalib: int,
    R: float,
    angle4next: float,
    cropcorner: bool,
) -> np.ndarray:
    """
    Generate a k-t sampling mask using radial (rotated line) trajectories.
    """
    rate = 1.0 / R
    beams = floor(rate * 180)
    a = max(nx, ny) if cropcorner else int(ceil(np.sqrt(2) * max(nx, ny)))

    # Create an auxiliary image with a horizontal line at its center.
    aux = np.zeros((a, a))
    row_idx = int(round_away_from_zero(a / 2)) - 1  # adjust for 0-indexing
    aux[row_idx, :] = 1

    angle_step = 180.0 / beams
    ktus = np.zeros((nx, ny, nt))
    for i in range(nt):
        angle_offset = angle4next * i
        # Create a set of angles for this frame.
        angles = np.arange(angle_offset, 180 + angle_offset + 1e-9, angle_step)
        image = np.zeros((nx, ny))
        for ang in angles:
            rotated = rotate(aux, ang, reshape=False, order=0, mode="grid-constant", cval=0)
            cropped = center_crop(rotated, (nx, ny))
            image += cropped
        ktus[:, :, i] = image

    # Add calibration region (ACS)
    acs = zero_pad(np.ones((ncalib, ncalib, nt)), (nx, ny, nt))
    mask = (ktus + acs > 0).astype(np.float64)
    return mask


def kt_uniform_sampling(nx: int, ny: int, nt: int, ncalib: int, R: int) -> np.ndarray:
    """
    Generate a k-t sampling mask using uniform undersampling.
    """
    # Create uniform patterns in phase and time directions.
    ptmp = np.zeros(ny)
    ptmp[::R] = 1
    ttmp = np.zeros(nt)
    ttmp[::R] = 1

    # Construct a Toeplitz-like matrix using broadcasting.
    i = np.arange(ny)[:, None]  # column vector, shape (ny,1)
    j = np.arange(nt)[None, :]  # row vector, shape (1,nt)
    k = i - j
    Top = np.empty_like(k, dtype=ptmp.dtype)
    mask = k >= 0
    Top[mask] = ptmp[k[mask]]
    Top[~mask] = ttmp[-k[~mask]]

    # Find nonzero indices.
    rows, cols = np.nonzero(Top)
    ph = rows - ny // 2
    ti = cols - nt // 2

    # Correct duplicate sample positions.
    ph, ti = resolve_duplicate_indices(ph, ti, ny, nt)

    samp = np.zeros((ny, nt))
    row_indices = np.clip(ph + ny // 2, 0, ny - 1)
    col_indices = np.clip(ti + nt // 2, 0, nt - 1)
    samp[row_indices, col_indices] = 1

    # Add ACS region.
    acs = zero_pad(np.ones((nx, ncalib, nt)), (nx, ny, nt))
    ktus = np.tile(samp, (nx, 1, 1))
    mask = (ktus + acs > 0).astype(np.float64)
    return mask


def kt_gaussian_sampling(nx: int, ny: int, nt: int, ncalib: int, R: int, alpha: float, seed: int) -> np.ndarray:
    """
    Generate a k-t sampling mask using a Gaussian density (variable density) sampling.
    """
    # Define phase positions.
    p1 = np.arange(-ny // 2, int(ceil(ny / 2)))
    tr = int(round(ny / R))  # number of readout lines per frame
    ph_list, ti_list = [], []

    sig = ny / 5.0  # standard deviation for the Gaussian envelope
    prob = 0.1 + alpha / (1 - alpha + 1e-10) * np.exp(-(p1**2) / (sig**2))

    rng = np.random.default_rng(seed)
    tmpSd = np.round(rng.random(nt) * 1e6).astype(np.int64)

    offset_t = nt // 2
    offset_ny = ny // 2
    start_t = -int(floor(nt / 2))
    end_t = int(ceil(nt / 2))

    for i_val in range(start_t, end_t):
        n_tmp = tr
        # Randomly choose phase indices with weighted probability.
        p_tmp = random_weighted_choice(prob, int(tmpSd[i_val + offset_t]), (n_tmp,)) - (offset_ny + 1)
        ti_list.extend([i_val] * n_tmp)
        ph_list.extend(p_tmp.tolist())

    ph_arr = np.array(ph_list, dtype=np.int64)
    ti_arr = np.array(ti_list, dtype=np.int64)

    # Resolve duplicate sample positions.
    ph_arr, ti_arr = resolve_duplicate_indices(ph_arr, ti_arr, ny, nt)

    samp = np.zeros((ny, nt))
    row_indices = np.clip(ph_arr + offset_ny, 0, ny - 1)
    col_indices = np.clip(ti_arr + offset_t, 0, nt - 1)
    samp[row_indices, col_indices] = 1

    # Add ACS region.
    acs = zero_pad(np.ones((nx, ncalib, nt)), (nx, ny, nt))
    ktus = np.tile(samp, (nx, 1, 1))
    mask = (ktus + acs > 0).astype(np.float64)
    return mask


def uniform_sampling(nx: int, ny: int, nt: int, ncalib: int, R: int) -> np.ndarray:
    """
    Generate a uniform sampling mask without temporal interleave.
    """
    # Create a calibration region (ones) of shape (nx, ncalib) and center-pad it to (nx, ny)
    mask = zero_pad(np.ones((nx, ncalib), dtype=np.float64), (nx, ny))
    # Set every R-th column (starting at the first) to one.
    mask[:, ::R] = 1
    return mask[..., np.newaxis].repeat(nt, -1)
