#!/usr/bin/env python3
"""
Save the same CIFAR-10-C image under four corruptions: gaussian_noise, motion_blur, fog, jpeg_compression.

Usage:
    python scripts/save_cifar10c_examples.py --index 0 --severity 3 --outdir outputs/cifar_examples

Produces individual PNGs and a combined grid PNG suitable for LaTeX inclusion.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


CORRUPTIONS = {
    "gaussian_noise": "data/CIFAR-10-C/gaussian_noise.npy",
    "motion_blur": "data/CIFAR-10-C/motion_blur.npy",
    "fog": "data/CIFAR-10-C/fog.npy",
    "jpeg_compression": "data/CIFAR-10-C/jpeg_compression.npy",
}


def load_corruption(path: str) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def select_image(arr: np.ndarray, idx: int, severity: int) -> np.ndarray:
    # CIFAR-10-C arrays are laid out as (5 * 10000, 32, 32, 3)
    # Reshape to (severities, n_images, H, W, C) when possible.
    if arr.ndim != 4:
        raise ValueError(f"Unexpected array shape: {arr.shape}")
    total = arr.shape[0]
    if total % 10000 == 0:
        severities = total // 10000
        n_images = 10000
    else:
        # Fallback: try to guess 5 severities
        severities = 5
        n_images = total // severities
    if not (1 <= severity <= severities):
        raise ValueError(f"severity must be between 1 and {severities}")
    if not (0 <= idx < n_images):
        raise ValueError(f"index must be between 0 and {n_images-1}")
    reshaped = arr.reshape((severities, n_images) + arr.shape[1:])
    img = reshaped[severity - 1, idx]
    return img


def to_pil(img: np.ndarray) -> Image.Image:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def make_grid(images: list[Image.Image], titles: list[str], outpath: Path):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(n * 2.5, 2.5))
    if n == 1:
        axes = [axes]
    for ax, im, title in zip(axes, images, titles):
        ax.imshow(im)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="Base image index in CIFAR-10 test set (0..9999)")
    parser.add_argument("--severity", type=int, default=3, help="Severity level (1..5)")
    parser.add_argument("--outdir", type=str, default="outputs/cifar_examples", help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    images = []
    titles = []

    for name, p in CORRUPTIONS.items():
        arr = load_corruption(p)
        img = select_image(arr, args.index, args.severity)
        pil = to_pil(img)
        fn = outdir / f"cifar10c_{name}_idx{args.index}_sev{args.severity}.png"
        pil.save(fn)
        images.append(pil)
        titles.append(name.replace("_", " "))
        print(f"Saved: {fn}")

    grid_path = outdir / f"cifar10c_grid_idx{args.index}_sev{args.severity}.png"
    make_grid(images, titles, grid_path)
    print(f"Saved grid: {grid_path}")


if __name__ == "__main__":
    main()
