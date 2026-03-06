#!/usr/bin/env python3
"""
Visualize .mat files containing 'img4ranking' data.
Mimics logic from check_file.ipynb: read with h5py/scipy, normalize, show slices.
"""

import argparse
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io


def read_mat(mat_file):
    """Read .mat file; try h5py first (v7.3), then scipy."""
    try:
        with h5py.File(mat_file, "r", swmr=True) as f:
            data_kv = [(key, f[key][()]) for key in f]
        print("read by h5py", file=sys.stderr)
    except Exception:
        data = scipy.io.loadmat(mat_file)
        data_kv = [(key, data[key]) for key in data if not key.startswith("__")]
        print("read by scipy", file=sys.stderr)
    return data_kv


def get_img4ranking(data_kv):
    """Extract 'img4ranking' array from (key, value) list."""
    for key, data in data_kv:
        if key == "img4ranking":
            return np.asarray(data)
    return None


def normalize_01(arr):
    """Normalize array to [0, 1] for display."""
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin == 0:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr.astype(np.float64) - vmin) / (vmax - vmin)


def _normalize_frames(img, normalize):
    """Normalize per-frame (per slice and per time for 4D)."""
    if not normalize:
        return img.astype(np.float64) if img.dtype != np.float64 else img
    img = img.astype(np.float64)
    if img.ndim == 3:
        for i in range(img.shape[-1]):
            img[:, :, i] = normalize_01(img[:, :, i])
    elif img.ndim == 4:
        for s in range(img.shape[2]):
            for t in range(img.shape[3]):
                img[:, :, s, t] = normalize_01(img[:, :, s, t])
    return img


def visualize_mat(
    mat_path,
    out_path=None,
    key="img4ranking",
    normalize=True,
    cmap="gray",
    show_colorbar=True,
    figsize_per_slice=5,
    dpi=100,
):
    """
    Load one .mat file and show all slices (and all time points) of key.

    Handles:
      - 2D (H, W) -> 1 slice
      - 3D (H, W, C) -> C slices in a row
      - 4D (H, W, num_slices, num_times) -> grid: x=time, y=slices (rows=slices, cols=time).
    """
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(mat_path)

    data_kv = read_mat(mat_path)
    img = None
    for k, v in data_kv:
        if k == key:
            img = np.asarray(v)
            break
    if img is None:
        raise KeyError(f"Key '{key}' not found in {mat_path}. Keys: {[k for k, _ in data_kv]}")

    # Interpret shape: (H, W), (H, W, n_slices), or (H, W, num_slices, num_times)
    img = np.asarray(img)
    while img.ndim > 4:
        img = img.squeeze()
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    if img.ndim == 3:
        # (H, W, n_slices)
        n_slices = img.shape[-1]
        n_times = 1
        img_4d = img[:, :, :, np.newaxis]
    elif img.ndim == 4:
        # (H, W, num_slices, num_times)
        n_slices = img.shape[2]
        n_times = img.shape[3]
        img_4d = img
    else:
        raise ValueError(f"Expected 2D, 3D or 4D array, got shape {img.shape}")

    img_4d = _normalize_frames(img_4d, normalize)

    # Grid: x = time, y = slices -> rows = slices, cols = time
    n_rows = n_slices
    n_cols = n_times
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_slice * n_cols, figsize_per_slice * n_rows),
        squeeze=False,
    )

    for s in range(n_slices):
        for t in range(n_times):
            ax = axes[s, t]
            im = ax.imshow(img_4d[:, :, s, t], cmap=cmap)
            ax.set_title(f"t={t}, slice={s}")
            ax.axis("off")
            if show_colorbar:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(os.path.basename(mat_path), fontsize=10)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print("Saved:", out_path)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize .mat files (img4ranking or other key)."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to .mat files or a directory (will use all .mat inside).",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="If set, save figures here instead of showing (filename: <mat_basename>.png).",
    )
    parser.add_argument(
        "-k", "--key",
        default="img4ranking",
        help="Key to visualize (default: img4ranking).",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Do not normalize to [0,1] for display.",
    )
    parser.add_argument(
        "--cmap",
        default="gray",
        help="Matplotlib colormap (default: gray).",
    )
    parser.add_argument(
        "--no-colorbar",
        action="store_true",
        help="Hide colorbars.",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        default=5,
        help="Figure size per slice in inches (default: 5).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="DPI for saved figures (default: 100).",
    )
    args = parser.parse_args()

    # Collect all .mat files
    mat_files = []
    for p in args.paths:
        if os.path.isfile(p) and p.lower().endswith(".mat"):
            mat_files.append(p)
        elif os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if f.lower().endswith(".mat"):
                    mat_files.append(os.path.join(p, f))
        else:
            print("Not a .mat file or directory, skipping:", p, file=sys.stderr)

    if not mat_files:
        print("No .mat files found.", file=sys.stderr)
        sys.exit(1)

    for mat_path in mat_files:
        out_path = None
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(mat_path))[0]
            out_path = os.path.join(args.output_dir, base + ".png")
        try:
            visualize_mat(
                mat_path,
                out_path=out_path,
                key=args.key,
                normalize=not args.no_normalize,
                cmap=args.cmap,
                show_colorbar=not args.no_colorbar,
                figsize_per_slice=args.figsize,
                dpi=args.dpi,
            )
        except Exception as e:
            print("Error visualizing", mat_path, ":", e, file=sys.stderr)


if __name__ == "__main__":
    main()

# One file (interactive window)
# python scripts/visualize_mat.py path/to/file.mat -o ./outputs/

# Several files
# python scripts/visualize_mat.py file1.mat file2.mat

# All .mat in a directory
# python scripts/visualize_mat.py path/to/val_img4ranking/

# Save figures instead of showing
# python scripts/visualize_mat.py path/to/file.mat -o ./figs

# Options
# python scripts/visualize_mat.py file.mat -k img4ranking --cmap gray --figsize 5 --dpi 150