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

import numpy as np


def crop(image, crop_size):
    sx, sy, sz, t = image.shape
    cx, cy, cz, ct = crop_size
    start_x = math.floor(sx / 2)
    start_y = math.floor(sy / 2)
    start_z = math.floor(sz / 2)
    start_t = math.floor(t / 2)
    return image[
        start_x + math.ceil(-cx / 2) : start_x + math.ceil(cx / 2),
        start_y + math.ceil(-cy / 2) : start_y + math.ceil(cy / 2),
        start_z + math.ceil(-cz / 2) : start_z + math.ceil(cz / 2),
        start_t + math.ceil(-ct / 2) : start_t + math.ceil(ct / 2),
    ]


def crop_map(image, crop_size):
    # attention: when it comes to the mapping, index changes (weired but checked.)
    sx, sy, sz, t = image.shape
    cx, cy, cz = crop_size
    start_x = math.floor(sx / 2)
    start_y = math.floor(sy / 2)
    if sz % 2 == 0:
        start_z = math.floor(sz / 2) - 1
    else:
        start_z = math.floor(sz / 2)

    return image[
        start_x + math.ceil(-cx / 2) : start_x + math.ceil(cx / 2),
        start_y + math.ceil(-cy / 2) : start_y + math.ceil(cy / 2),
        start_z + math.ceil(-cz / 2) : start_z + math.ceil(cz / 2),
    ]


def run4Ranking(img, filetype, do_center_crop_infer=True):
    sx, sy, sz, st = img.shape

    img = np.reshape(img, (sx, sy, sz, st))

    if do_center_crop_infer:
        if "map" in filetype or "rho" in filetype:
            # nocropping for mapping data]
            reconImg = img[:, :, :, :]
            if sz < 3:
                crop_size = (round(sx / 3), round(sy / 2 + 0.00001), sz)
            else:
                crop_size = (round(sx / 3), round(sy / 2 + 0.00001), 2)
            img4ranking = crop_map(np.abs(reconImg), crop_size).astype(np.float32)
        else:
            if sz < 3:
                # take the first 3 frames
                reconImg = img[..., :3]
                img4ranking = crop(np.abs(reconImg), (round(sx / 3), round(sy / 2 + 0.00001), sz, 3)).astype(np.float32)
            else:
                reconImg = img[:, :, round(sz / 2 + 0.00001) - 2 : round(sz / 2 + 0.00001), :3]
                img4ranking = crop(np.abs(reconImg), (round(sx / 3), round(sy / 2 + 0.00001), 2, 3)).astype(np.float32)
    else:
        img4ranking = np.abs(img).astype(np.float32)
    if "blackblood" in filetype or "T1w" in filetype or "T2w" in filetype or ("cine" in filetype and st == 1):
        assert st == 1
        img4ranking = np.squeeze(img4ranking, axis=-1)
        if sz == 1:
            img4ranking = np.squeeze(img4ranking, axis=-1)
    return img4ranking


# import decimal # For precise rounding if needed, see note below


def crop_center(img_to_crop, target_shape_tuple):
    """
    Crops the input numpy array from the center to achieve the target_shape_tuple.
    Assumes img_to_crop.ndim == len(target_shape_tuple).
    Assumes target_shape_tuple[i] <= img_to_crop.shape[i] for all i.
    """
    current_shape = img_to_crop.shape
    if len(current_shape) != len(target_shape_tuple):
        raise ValueError(
            f"Dimension mismatch: img shape {current_shape} ({len(current_shape)}D), "
            f"target crop shape {target_shape_tuple} ({len(target_shape_tuple)}D)"
        )

    slices_for_crop = []
    for i in range(len(target_shape_tuple)):
        dim_current = current_shape[i]
        dim_target = target_shape_tuple[i]
        if dim_target > dim_current:
            raise ValueError(f"Target dim {dim_target} at axis {i} is larger than current dim {dim_current}")
        start = (dim_current - dim_target) // 2
        end = start + dim_target
        slices_for_crop.append(slice(start, end))
    return img_to_crop[tuple(slices_for_crop)]


def run4Ranking2025(sos_img, filetype):
    """
    Converts complex image data for ranking, analogous to the MATLAB function.
    - Reduces computing burden by evaluating central 2 slices (if available).
    - For cine: uses the first 3 time frames.
    - For mapping: uses all weightings/time frames.
    - Crops the middle 1/3 (sx) and 1/2 (sy) of the original image.

    Args:
        sos_img (np.ndarray): Complex images (sx, sy, sz, t/w) or (sx, sy, sz).
                                  scc is coil dimension. sz is slices. t/w is time/weightings.
        filetype (str): String indicating the type of image (e.g., 'cine_lax', 'T1map', 'blackblood_sax').

    Returns:
        np.ndarray: "single" format (float32) images for ranking.
                    Shape depends on input and processing, e.g., (sx/3, sy/2, num_slices_used, num_times_used).
    """
    img_shape = sos_img.shape

    # Determine sx, sy, s_z (slices), t_frames (time/weightings)
    # MATLAB's size(A) behavior when requesting more outputs than ndims(A) is to return 1 for trailing dimensions.
    s_x = img_shape[0]
    s_y = img_shape[1]

    s_z = img_shape[2]
    # t_frames default to 1 if not specified (e.g. 4D input)
    # In MATLAB, if [...,t]=size(img_4D), t becomes 1.
    # If [...,sz]=size(img_3D), sz becomes 1.

    is_black_blood = False
    filetype_lower = filetype.lower()  # For case-insensitive matching

    if "blackblood" in filetype_lower or "t1w" in filetype_lower or "t2w" in filetype_lower:
        # These are single-frame modalities, sz is the 4th dim.
        # img_shape assumed to be (sx, sy, scc, sz) or (sx, sy, scc) -> sz=1
        t_frames = 1
        is_black_blood = True
    else:
        # Cine, Mapping etc. sz is 4th dim, t is 5th dim (if present)
        # img_shape assumed to be (sx, sy, scc, sz, t) or (sx, sy, scc, sz) -> t=1
        # or (sx, sy, scc) -> sz=1, t=1
        t_frames = img_shape[3]

    # Detect mapping modalities
    detect_map_keywords = ["t1map", "t2map", "t2smap", "t1mappost"]
    is_mapping = any(keyword in filetype_lower for keyword in detect_map_keywords)

    # Detect T1rho modality
    is_t1rho = "t1rho" in filetype_lower

    # --- Clipping slices (select central 2 slices, or all if < 3) ---
    if s_z < 3:
        # Use all slices: 0 to s_z-1
        slice_indices_obj = slice(0, s_z)
        num_slices_to_use = s_z
    else:
        # Use central 2 slices
        # MATLAB: center = round(sz/2); sliceToUse = (center-1):(center); (1-based)
        # e.g., sz=5, center_matlab=3. Slices 2:3 (1-based) -> Python indices 1, 2 -> slice(1, 3)
        # e.g., sz=4, center_matlab=2. Slices 1:2 (1-based) -> Python indices 0, 1 -> slice(0, 2)
        center_matlab_style = int(round(s_z / 2.0))  # round half up for positive

        # Python 0-based indices for (center-1) and (center) from 1-based MATLAB
        slice_start_py = center_matlab_style - 2
        slice_end_py = center_matlab_style  # Exclusive end for slice object
        slice_indices_obj = slice(slice_start_py, slice_end_py)
        num_slices_to_use = 2

    if s_z == 1 and num_slices_to_use == 2:  # Correction if s_z=1 or s_z=2 led to num_slices_to_use=2
        num_slices_to_use = s_z

    # --- Clipping time frames ---
    if is_black_blood or t_frames == 1:
        time_indices_obj = slice(0, 1)  # Select the first (and only) time frame
        num_time_frames_to_use = 1
    elif is_mapping or is_t1rho:
        time_indices_obj = slice(0, t_frames)  # Select all time frames/weightings
        num_time_frames_to_use = t_frames
    else:  # Cine
        limit = min(3, t_frames)
        time_indices_obj = slice(0, limit)  # Select first min(3, t_frames)
        num_time_frames_to_use = limit

    if is_black_blood:  # or t_frames == 1 (implicitly handled by sos_img shape after s_z=1 logic)
        # sos_img shape: (sx,sy, actual_slices_in_sos, actual_times_in_sos=1) if s_z=1 originally
        # or (sx,sy, actual_slices_in_sos) if s_z > 1 originally and t_frames=1
        # slice_indices_obj is for the "slice" dimension (axis 2 of 4D view, or axis 2 of 3D view)
        # time_indices_obj is slice(0,1)

        # We expect sos_img to have a slice dimension at axis 2.
        # If original s_z was >1, sos_img is (sx,sy,s_z_after_sos_squeeze). We select along axis 2.
        # If original s_z was 1, sos_img is (sx,sy,1,1). We select along axis 2.
        selected_img = sos_img[:, :, slice_indices_obj, time_indices_obj]
        # After selection, selected_img will be (sx, sy, num_slices_to_use, num_time_frames_to_use=1)
    else:  # Cine, Mapping (t_frames > 1 or potentially t_frames=1 if not BB but single time)
        # sos_img shape: (sx,sy,actual_slices,actual_times)
        selected_img = sos_img[:, :, slice_indices_obj, time_indices_obj]
        # After selection, selected_img is (sx, sy, num_slices_to_use, num_time_frames_to_use)

    # --- Prepare for cropping: ensure selected_img has the correct Ndim for crop_center ---
    # Target crop dimensions for sx, sy
    # MATLAB's round(x) rounds .5 away from zero. For positive x, int(x + 0.5).
    # Python's round(x) rounds .5 to nearest even.
    # Using int(x + 0.5) for MATLAB-like rounding for positive numbers.
    target_crop_sx = int(s_x / 3.0 + 0.5)
    target_crop_sy = int(s_y / 2.0 + 0.5)

    img_to_be_cropped = np.abs(selected_img)

    # Determine the final shape for the cropped image based on MATLAB logic
    if is_black_blood:  # t_frames_to_use is 1
        if num_slices_to_use > 1:
            # Output: (tsx, tsy, num_slices)
            # selected_img is (sx, sy, num_slices, 1). Squeeze last dim.
            final_crop_target_shape = (
                target_crop_sx,
                target_crop_sy,
                num_slices_to_use,
            )
            if img_to_be_cropped.shape[-1] == 1 and img_to_be_cropped.ndim == 4:
                img_to_be_cropped = np.squeeze(img_to_be_cropped, axis=3)
        else:  # num_slices_to_use == 1
            # Output: (tsx, tsy)
            # selected_img is (sx, sy, 1, 1). Squeeze last two dims.
            final_crop_target_shape = (target_crop_sx, target_crop_sy)
            img_to_be_cropped = np.squeeze(img_to_be_cropped)  # Squeezes all size 1 dims
            if img_to_be_cropped.ndim != 2:  # Ensure it became 2D
                # This might happen if sx or sy was 1, then squeeze reduces further.
                # This case needs more specific handling if sx=1 or sy=1.
                # Assuming sx, sy > 1 for typical images.
                raise ValueError(f"Image for 2D crop is not 2D after squeeze: {img_to_be_cropped.shape}")

    else:  # Not BlackBlood (Cine/Mapping)
        if num_time_frames_to_use > 1 and num_slices_to_use > 1:
            # Output: (tsx, tsy, num_slices, num_times)
            # selected_img is (sx, sy, num_slices, num_times). Matches.
            final_crop_target_shape = (
                target_crop_sx,
                target_crop_sy,
                num_slices_to_use,
                num_time_frames_to_use,
            )
        elif num_time_frames_to_use > 1 and num_slices_to_use == 1:
            # Output: (tsx, tsy, 1, num_times)
            # selected_img is (sx, sy, 1, num_times). Matches.
            final_crop_target_shape = (
                target_crop_sx,
                target_crop_sy,
                1,
                num_time_frames_to_use,
            )
        elif num_time_frames_to_use == 1 and num_slices_to_use > 1:
            # Output: (tsx, tsy, num_slices)
            # selected_img is (sx, sy, num_slices, 1). Squeeze last dim.
            final_crop_target_shape = (
                target_crop_sx,
                target_crop_sy,
                num_slices_to_use,
            )
            if img_to_be_cropped.shape[-1] == 1 and img_to_be_cropped.ndim == 4:
                img_to_be_cropped = np.squeeze(img_to_be_cropped, axis=3)
        elif (
            num_time_frames_to_use == 1 and num_slices_to_use == 1
        ):  # This case was missing in the MATLAB if/else chain
            # Output: (tsx, tsy)
            # selected_img is (sx, sy, 1, 1). Squeeze.
            final_crop_target_shape = (target_crop_sx, target_crop_sy)
            img_to_be_cropped = np.squeeze(img_to_be_cropped)
            if img_to_be_cropped.ndim != 2:
                raise ValueError(f"Image for 2D crop is not 2D after squeeze: {img_to_be_cropped.shape}")
        else:  # Should not be reached
            raise RuntimeError("Unhandled case in final crop shape determination.")

    # Perform the crop
    img4ranking_np = crop_center(img_to_be_cropped, final_crop_target_shape)

    return img4ranking_np.astype(np.float32)


def run4Ranking_v2(img: np.ndarray, filetype: str) -> np.ndarray:
    """
    Select central slices/time-frames and crop the middle 1/6 of img for ranking.

    Parameters
    ----------
    img : np.ndarray
        Array of shape (sx, sy, sz, t), already coil-combined.
    filetype : str
        Identifier string (e.g. filename or modality).

    Returns
    -------
    np.ndarray (float32)
        Cropped array of dims:
          - (sx/3, sy/2)                if sz=1, t=1
          - (sx/3, sy/2, 2)             if sz>=2, t=1
          - (sx/3, sy/2, 2, 3)          if sz>=2, t>1 & not mapping/T1rho
          - (sx/3, sy/2, 2, t)          if mapping/T1rho (t>1)
    """
    ft = filetype.lower()

    # flags
    is_blackblood = any(x in ft for x in ("blackblood", "t1w", "t2w"))
    detect_map = ("t1map", "t2map", "t2smap", "t1mappost")
    is_mapping = any(x in ft for x in detect_map)
    is_t1rho = "t1rho" in ft

    sx, sy, sz, t = img.shape

    # choose central slices
    if sz < 3:
        slice_idx = list(range(sz))
    else:
        c = int(round(sz / 2.0))
        slice_idx = [c - 1, c]

    # choose time frames
    if t == 1:
        time_idx = [0]
    elif is_mapping or is_t1rho:
        time_idx = list(range(t))
    else:
        time_idx = list(range(min(3, t)))

    # central‐crop helper
    def center_crop(a, shape):
        """Crop array `a` to `shape`, centering on each axis."""
        slices = []
        for dim, sz_crop in enumerate(shape):
            L = a.shape[dim]
            start = (L - sz_crop) // 2
            slices.append(slice(start, start + sz_crop))
        return a[tuple(slices)]

    # target crop sizes in x,y
    cx = int(round(sx / 3.0))
    cy = int(round(sy / 2.0))

    S = len(slice_idx)
    T = len(time_idx)

    # select & drop axes as needed
    if T == 1 and S == 1:
        sel = img[:, :, slice_idx[0], time_idx[0]]  # (sx, sy)
        crop_shape = (cx, cy)

    elif T == 1 and S > 1:
        sel = img[:, :, slice_idx, time_idx[0]]  # (sx, sy, S)
        crop_shape = (cx, cy, S)

    elif T > 1 and S == 1:
        sel = img[:, :, slice_idx[0], time_idx]  # (sx, sy, T)
        # keep a singleton slice‐axis to match MATLAB logic:
        sel = sel[..., np.newaxis, :]  # → (sx, sy, 1, T)
        crop_shape = (cx, cy, 1, T)

    else:  # T > 1 and S > 1
        sel = img[:, :, slice_idx, time_idx]  # (sx, sy, S, T)
        crop_shape = (cx, cy, S, T)

    # crop, cast, squeeze
    out = center_crop(sel, crop_shape)
    if is_blackblood:
        assert T == 1
        out = np.squeeze(out, axis=-1)
    return out.astype(np.float32)
