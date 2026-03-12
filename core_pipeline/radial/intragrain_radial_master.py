#!/usr/bin/env python
"""
Intragrain radial master script (F:-paths version).

Combines the behavior of:
- intragrain_radial_descriptors_from_listjson_v2.py
- intragrain_radial_polar_kinetic_from_listjson.py

and additionally computes a radial defect fraction profile.

Metrics (all as functions of normalized radius r/R):
- Mean intensity (from heat-map or fallback .jpg)
- Entropy
- Texture anisotropy A_tex
- Polar anisotropy
- Kinetic heterogeneity (coefficient of variation, CV)
- Defect fraction (fraction of pixels above a given threshold)

Assumptions
-----------
- Each JSON file (e.g. FAPI_0.json) is a *list* of dicts with keys:
    - 'segmentation': COCO RLE dict {'size': [H, W], 'counts': <str>}
    - optionally 'mask_name', 'bbox', 'score' (ignored here)
- For each "<stem>.json" there is at least one intensity image:
    - preferred: "<stem><intensity_suffix>", e.g. "FAPI_0_heatmap.png"
    - fallback:  "<stem>.jpg"
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

try:
    from pycocotools import mask as maskUtils
except ImportError as e:
    raise ImportError(
        "pycocotools is required. On Windows you may need:\n"
        "    pip install pycocotools-windows\n"
        "On Linux/macOS:\n"
        "    pip install pycocotools"
    ) from e


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def load_mask_from_segmentation(seg: dict) -> np.ndarray:
    """Decode COCO RLE segmentation dict -> boolean mask."""
    if not isinstance(seg, dict) or "size" not in seg or "counts" not in seg:
        raise ValueError("Segmentation must be an RLE dict with 'size' and 'counts'.")
    rle = {
        "size": seg["size"],
        "counts": seg["counts"].encode("utf-8")
        if isinstance(seg["counts"], str)
        else seg["counts"],
    }
    m = maskUtils.decode(rle)  # H x W or H x W x 1
    if m.ndim == 3:
        m = m[..., 0]
    return m.astype(bool)


def load_intensity(img_path: Path) -> np.ndarray:
    """Load intensity image as float32 in [0, 1]."""
    img = imageio.imread(img_path)
    if img.ndim == 3:
        # convert RGB to luminance
        img = (
            0.2126 * img[..., 0]
            + 0.7152 * img[..., 1]
            + 0.0722 * img[..., 2]
        )
    img = img.astype(np.float32)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    return img


def radial_bins_for_mask(mask: np.ndarray, n_bins: int):
    """Return bin indices (0..n_bins-1) and valid-pixel coordinates inside grain."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None, None, None, None

    cy = ys.mean()
    cx = xs.mean()
    dy = ys - cy
    dx = xs - cx
    r = np.sqrt(dx * dx + dy * dy)
    R = r.max()
    if R == 0:
        return None, None, None, None

    r_norm = r / R
    bins = np.clip((r_norm * n_bins).astype(int), 0, n_bins - 1)
    return ys, xs, bins, r_norm


def annular_entropy(values: np.ndarray, n_bins_hist: int = 32) -> float:
    """Shannon entropy of intensity values in [0,1]."""
    if values.size == 0:
        return np.nan
    hist, _ = np.histogram(
        values, bins=n_bins_hist, range=(0.0, 1.0), density=True
    )
    p = hist[hist > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def annular_anisotropy(theta: np.ndarray) -> float:
    """
    Texture anisotropy from gradient orientations θ:
    A_tex = |⟨exp(2iθ)⟩|
    """
    if theta.size == 0:
        return np.nan
    z = np.exp(2j * theta)
    return float(np.abs(z.mean()))


def compute_image_gradients(intensity: np.ndarray):
    """Simple finite-difference gradients."""
    gy, gx = np.gradient(intensity)
    mag = np.hypot(gx, gy)
    theta = np.arctan2(gy, gx)
    return mag, theta


# -------------------------------------------------------------------
# Dataset processing: descriptors (intensity, entropy, texture anisotropy)
# -------------------------------------------------------------------

def process_dataset_descriptors(
    dir_path: Path,
    n_bins: int = 25,
    min_pixels: int = 200,
    intensity_suffix: str = "_heatmap.png",
    base_intensity_suffix_fallback: str = ".jpg",
):
    """
    Returns:
        r_centers, m_I, s_I, m_H, s_H, m_A, s_A, n_grains
    """
    all_profiles_I = []
    all_profiles_H = []
    all_profiles_A = []

    json_files = sorted(dir_path.glob("*.json"))
    print(f"[INFO]   [desc] Found {len(json_files)} JSON files in {dir_path}")

    for jf in json_files:
        base = jf.stem  # e.g. FAPI_0

        # choose intensity image: first try suffix (heatmap), then jpg
        img_path = dir_path / f"{base}{intensity_suffix}"
        if not img_path.exists():
            img_path = dir_path / f"{base}{base_intensity_suffix_fallback}"
        if not img_path.exists():
            print(f"[WARN]   [desc] no intensity image for {base}; skipping whole file")
            continue

        intensity = load_intensity(img_path)
        _, grad_theta = compute_image_gradients(intensity)

        with open(jf, "r") as f:
            entries = json.load(f)

        if not isinstance(entries, list):
            print(f"[WARN]   [desc] {jf.name} is not a list; skipping")
            continue

        for entry in entries:
            seg = entry.get("segmentation", None)
            if not isinstance(seg, dict) or "size" not in seg or "counts" not in seg:
                continue

            try:
                mask = load_mask_from_segmentation(seg)
            except Exception as e:
                print(f"[WARN]   [desc] RLE decode failed in {jf.name}: {e}")
                continue

            # sanity: crop to image size if needed
            H, W = intensity.shape
            if mask.shape != (H, W):
                h2 = min(H, mask.shape[0])
                w2 = min(W, mask.shape[1])
                mask_cropped = np.zeros_like(intensity, dtype=bool)
                mask_cropped[:h2, :w2] = mask[:h2, :w2]
                mask = mask_cropped

            ys, xs, bin_idx, _ = radial_bins_for_mask(mask, n_bins)
            if ys is None:
                continue
            if ys.size < min_pixels:
                continue

            I_bins = np.full(n_bins, np.nan, dtype=float)
            H_bins = np.full(n_bins, np.nan, dtype=float)
            A_bins = np.full(n_bins, np.nan, dtype=float)

            min_per_ring = max(10, min_pixels // n_bins)

            for b in range(n_bins):
                sel = bin_idx == b
                if np.count_nonzero(sel) < min_per_ring:
                    continue

                yy = ys[sel]
                xx = xs[sel]

                vals = intensity[yy, xx]
                I_bins[b] = float(vals.mean())

                H_bins[b] = annular_entropy(vals, n_bins_hist=32)

                theta_vals = grad_theta[yy, xx]
                A_bins[b] = annular_anisotropy(theta_vals)

            all_profiles_I.append(I_bins)
            all_profiles_H.append(H_bins)
            all_profiles_A.append(A_bins)

    if not all_profiles_I:
        raise RuntimeError(f"No valid grains found in {dir_path} (descriptors)")

    all_profiles_I = np.stack(all_profiles_I, axis=0)
    all_profiles_H = np.stack(all_profiles_H, axis=0)
    all_profiles_A = np.stack(all_profiles_A, axis=0)

    r_centers = (np.arange(n_bins) + 0.5) / n_bins

    def nanmean_std(arr):
        return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)

    m_I, s_I = nanmean_std(all_profiles_I)
    m_H, s_H = nanmean_std(all_profiles_H)
    m_A, s_A = nanmean_std(all_profiles_A)

    n_grains = all_profiles_I.shape[0]
    print(f"[INFO]   [desc] used {n_grains} grains from {dir_path}")

    return r_centers, m_I, s_I, m_H, s_H, m_A, s_A, n_grains


# -------------------------------------------------------------------
# Dataset processing: polar anisotropy + kinetic heterogeneity
# -------------------------------------------------------------------

def decode_segmentation_polar(seg):
    """Decode a COCO RLE segmentation dict into a boolean mask."""
    if not isinstance(seg, dict) or "size" not in seg or "counts" not in seg:
        raise ValueError("Segmentation must be an RLE dict with 'size' and 'counts'.")

    rle = {
        "size": seg["size"],
        "counts": seg["counts"].encode("ascii") if isinstance(seg["counts"], str) else seg["counts"],
    }
    m = maskUtils.decode(rle)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(bool)


def process_dataset_polar_kinetic(
    dir_path: Path,
    intensity_suffix: str,
    n_bins: int,
    min_pixels: int,
):
    """
    Returns
    -------
    r_centres : (n_bins,) array, bin centres in r/R
    polar_mean, polar_std : (n_bins,) arrays
    kinetic_mean, kinetic_std : (n_bins,) arrays
    n_grains : int, number of grains used
    """
    dir_path = Path(dir_path)
    json_files = sorted(dir_path.glob("*.json"))

    if not json_files:
        raise RuntimeError(f"No JSON files found in {dir_path}")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)

    polar_profiles = []
    kinetic_profiles = []

    print(f"[INFO]   [pk] Found {len(json_files)} JSON files in {dir_path}")

    from PIL import Image  # local import

    for jf in json_files:
        stem = jf.stem  # e.g. "FAPI_0" or "FAPI_TEMPO_0"

        img_path = dir_path / f"{stem}{intensity_suffix}"
        if not img_path.is_file():
            print(f"[WARN]   [pk] no intensity image for {stem}; skipping")
            continue

        img = np.array(Image.open(img_path).convert("F"), dtype=np.float32)
        img = img - img.min()
        denom_img = img.max()
        if denom_img > 0:
            img /= denom_img

        with open(jf, "r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"[WARN]   [pk] {jf.name} is not list-style JSON; skipping")
            continue

        for item in data:
            seg = item.get("segmentation", None)
            if seg is None:
                continue

            try:
                mask = decode_segmentation_polar(seg)
            except Exception as e:
                print(f"[WARN]   [pk] segmentation decode failed in {jf.name}: {e}")
                continue

            ys, xs = np.nonzero(mask)
            if ys.size < min_pixels:
                continue

            vals = img[ys, xs]
            vals = vals - vals.min()
            denom = vals.max()
            if denom > 0:
                vals = vals / denom

            cy = ys.mean()
            cx = xs.mean()
            dy = ys - cy
            dx = xs - cx
            r = np.sqrt(dx * dx + dy * dy)
            R = r.max()
            if R <= 0:
                continue

            r_norm = r / R
            theta = np.arctan2(dy, dx)

            polar_profile = np.full(n_bins, np.nan, dtype=np.float64)
            kinetic_profile = np.full(n_bins, np.nan, dtype=np.float64)

            min_per_ring = max(10, min_pixels // n_bins)

            for b in range(n_bins):
                r0 = bin_edges[b]
                r1 = bin_edges[b + 1]
                in_ring = (r_norm >= r0) & (r_norm < r1)
                if not np.any(in_ring):
                    continue

                ring_vals = vals[in_ring]
                if ring_vals.size < min_per_ring:
                    continue

                ring_theta = theta[in_ring]

                mu = float(ring_vals.mean())
                sigma = float(ring_vals.std())
                kinetic_profile[b] = sigma / (mu + 1e-6)

                v_center = ring_vals - mu
                denom_a = float(np.mean(np.abs(v_center))) + 1e-6
                if denom_a <= 0:
                    A = 0.0
                else:
                    A = np.abs(np.mean(v_center * np.exp(1j * ring_theta))) / denom_a
                polar_profile[b] = A

            if not np.all(np.isnan(polar_profile)):
                polar_profiles.append(polar_profile)
                kinetic_profiles.append(kinetic_profile)

    if len(polar_profiles) == 0:
        raise RuntimeError(f"No valid grains found in {dir_path} (polar/kinetic)")

    polar_arr = np.array(polar_profiles, dtype=np.float64)
    kinetic_arr = np.array(kinetic_profiles, dtype=np.float64)

    r_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    polar_mean = np.nanmean(polar_arr, axis=0)
    polar_std = np.nanstd(polar_arr, axis=0)

    kinetic_mean = np.nanmean(kinetic_arr, axis=0)
    kinetic_std = np.nanstd(kinetic_arr, axis=0)

    n_grains = polar_arr.shape[0]
    print(f"[INFO]   [pk] used {n_grains} grains from {dir_path}")

    return r_centres, polar_mean, polar_std, kinetic_mean, kinetic_std, n_grains


# -------------------------------------------------------------------
# Dataset processing: defect fraction
# -------------------------------------------------------------------

def process_dataset_defect_fraction(
    dir_path: Path,
    n_bins: int,
    min_pixels: int,
    intensity_suffix: str,
    base_intensity_suffix_fallback: str,
    defect_threshold: float,
):
    """
    Compute per-grain radial defect fraction profile.

    Defect fraction in a ring = fraction of pixels whose intensity
    (from the *image-level normalized* intensity field) exceeds
    `defect_threshold` in [0,1].
    """
    all_profiles_D = []

    json_files = sorted(dir_path.glob("*.json"))
    print(f"[INFO]   [def] Found {len(json_files)} JSON files in {dir_path}")

    for jf in json_files:
        base = jf.stem

        img_path = dir_path / f"{base}{intensity_suffix}"
        if not img_path.exists():
            img_path = dir_path / f"{base}{base_intensity_suffix_fallback}"
        if not img_path.exists():
            print(f"[WARN]   [def] no intensity image for {base}; skipping whole file")
            continue

        intensity = load_intensity(img_path)

        with open(jf, "r") as f:
            entries = json.load(f)

        if not isinstance(entries, list):
            print(f"[WARN]   [def] {jf.name} is not a list; skipping")
            continue

        for entry in entries:
            seg = entry.get("segmentation", None)
            if not isinstance(seg, dict) or "size" not in seg or "counts" not in seg:
                continue

            try:
                mask = load_mask_from_segmentation(seg)
            except Exception as e:
                print(f"[WARN]   [def] RLE decode failed in {jf.name}: {e}")
                continue

            H, W = intensity.shape
            if mask.shape != (H, W):
                h2 = min(H, mask.shape[0])
                w2 = min(W, mask.shape[1])
                mask_cropped = np.zeros_like(intensity, dtype=bool)
                mask_cropped[:h2, :w2] = mask[:h2, :w2]
                mask = mask_cropped

            ys, xs, bin_idx, _ = radial_bins_for_mask(mask, n_bins)
            if ys is None:
                continue
            if ys.size < min_pixels:
                continue

            D_bins = np.full(n_bins, np.nan, dtype=float)
            min_per_ring = max(10, min_pixels // n_bins)

            for b in range(n_bins):
                sel = bin_idx == b
                if np.count_nonzero(sel) < min_per_ring:
                    continue

                yy = ys[sel]
                xx = xs[sel]
                vals = intensity[yy, xx]
                D_bins[b] = float(np.mean(vals >= defect_threshold))

            all_profiles_D.append(D_bins)

    if not all_profiles_D:
        raise RuntimeError(f"No valid grains found in {dir_path} (defect fraction)")

    all_profiles_D = np.stack(all_profiles_D, axis=0)
    r_centers = (np.arange(n_bins) + 0.5) / n_bins

    m_D = np.nanmean(all_profiles_D, axis=0)
    s_D = np.nanstd(all_profiles_D, axis=0)
    n_grains = all_profiles_D.shape[0]

    print(f"[INFO]   [def] used {n_grains} grains from {dir_path}")
    return r_centers, m_D, s_D, n_grains


# -------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------

def plot_profile(
    r,
    mean_fapi,
    std_fapi,
    mean_tempo,
    std_tempo,
    ylabel: str,
    title: str,
    out_path: Path,
):
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(r, mean_fapi, label="FAPI")
    ax.fill_between(r, mean_fapi - std_fapi, mean_fapi + std_fapi, alpha=0.25)

    ax.plot(r, mean_tempo, label="FAPI–TEMPO")
    ax.fill_between(r, mean_tempo - std_tempo, mean_tempo + std_tempo, alpha=0.25)

    ax.set_xlabel(r"Normalized radius $r/R$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] saved {out_path}")


# -------------------------------------------------------------------
# CLI (with F:-path defaults)
# -------------------------------------------------------------------

def parse_args():
    # Default paths for your setup
    default_fapi_dir = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\comparative datasets\FAPI"
    default_tempo_dir = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\comparative datasets\FAPI-TEMPO"
    default_out_prefix = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\intragrain_radial"

    p = argparse.ArgumentParser(
        description=(
            "Master intragrain radial script: intensity, entropy, texture anisotropy, "
            "polar anisotropy, kinetic heterogeneity, and defect fraction, "
            "computed from list-style JSON masks and heat-map / optical images.\n\n"
            "If no arguments are given, defaults are set to the F: paths used in your project."
        )
    )
    p.add_argument(
        "--fapi-dir",
        default=default_fapi_dir,
        help=f"Directory with FAPI_*.json and intensity images "
             f"(default: {default_fapi_dir})",
    )
    p.add_argument(
        "--tempo-dir",
        default=default_tempo_dir,
        help=f"Directory with FAPI_TEMPO_*.json and intensity images "
             f"(default: {default_tempo_dir})",
    )
    p.add_argument(
        "--out-prefix",
        default=default_out_prefix,
        help=f"Output prefix (directory + base name) for PNG/CSV outputs "
             f"(default: {default_out_prefix})",
    )
    p.add_argument(
        "--intensity-suffix",
        default="_heatmap.png",
        help="Suffix for intensity images (default: '_heatmap.png'). "
             "If not found, falls back to .jpg for descriptor/defect metrics.",
    )
    p.add_argument(
        "--n-bins",
        type=int,
        default=25,
        help="Number of radial bins between r/R=0 and 1 (default: 25).",
    )
    p.add_argument(
        "--min-pixels",
        type=int,
        default=200,
        help="Minimum number of pixels per grain to be used (default: 200).",
    )
    p.add_argument(
        "--defect-threshold",
        type=float,
        default=0.6,
        help="Threshold in [0,1] for defining a 'defect' pixel (default: 0.6).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_prefix = Path(args.out_prefix)
    base_intensity_suffix_fallback = ".jpg"

    fapi_dir = Path(args.fapi_dir)
    tempo_dir = Path(args.tempo_dir)

    # --- descriptors: intensity, entropy, texture anisotropy
    print(f"[INFO] Processing FAPI descriptors in {fapi_dir} ...")
    (
        r_desc_fapi,
        I_m_fapi,
        I_s_fapi,
        H_m_fapi,
        H_s_fapi,
        A_m_fapi,
        A_s_fapi,
        n_desc_fapi,
    ) = process_dataset_descriptors(
        fapi_dir,
        n_bins=args.n_bins,
        min_pixels=args.min_pixels,
        intensity_suffix=args.intensity_suffix,
        base_intensity_suffix_fallback=base_intensity_suffix_fallback,
    )

    print(f"[INFO] Processing FAPI–TEMPO descriptors in {tempo_dir} ...")
    (
        r_desc_tempo,
        I_m_tempo,
        I_s_tempo,
        H_m_tempo,
        H_s_tempo,
        A_m_tempo,
        A_s_tempo,
        n_desc_tempo,
    ) = process_dataset_descriptors(
        tempo_dir,
        n_bins=args.n_bins,
        min_pixels=args.min_pixels,
        intensity_suffix=args.intensity_suffix,
        base_intensity_suffix_fallback=base_intensity_suffix_fallback,
    )

    # --- polar + kinetic
    print(f"[INFO] Processing FAPI polar/kinetic in {fapi_dir} ...")
    (
        r_pk_fapi,
        polar_mean_fapi,
        polar_std_fapi,
        kinetic_mean_fapi,
        kinetic_std_fapi,
        n_pk_fapi,
    ) = process_dataset_polar_kinetic(
        fapi_dir,
        intensity_suffix=args.intensity_suffix,
        n_bins=args.n_bins,
        min_pixels=args.min_pixels,
    )

    print(f"[INFO] Processing FAPI–TEMPO polar/kinetic in {tempo_dir} ...")
    (
        r_pk_tempo,
        polar_mean_tempo,
        polar_std_tempo,
        kinetic_mean_tempo,
        kinetic_std_tempo,
        n_pk_tempo,
    ) = process_dataset_polar_kinetic(
        tempo_dir,
        intensity_suffix=args.intensity_suffix,
        n_bins=args.n_bins,
        min_pixels=args.min_pixels,
    )

    # --- defect fraction
    print(f"[INFO] Processing FAPI defect fraction in {fapi_dir} ...")
    (
        r_def_fapi,
        D_m_fapi,
        D_s_fapi,
        n_def_fapi,
    ) = process_dataset_defect_fraction(
        fapi_dir,
        n_bins=args.n_bins,
        min_pixels=args.min_pixels,
        intensity_suffix=args.intensity_suffix,
        base_intensity_suffix_fallback=base_intensity_suffix_fallback,
        defect_threshold=args.defect_threshold,
    )

    print(f"[INFO] Processing FAPI–TEMPO defect fraction in {tempo_dir} ...")
    (
        r_def_tempo,
        D_m_tempo,
        D_s_tempo,
        n_def_tempo,
    ) = process_dataset_defect_fraction(
        tempo_dir,
        n_bins=args.n_bins,
        min_pixels=args.min_pixels,
        intensity_suffix=args.intensity_suffix,
        base_intensity_suffix_fallback=base_intensity_suffix_fallback,
        defect_threshold=args.defect_threshold,
    )

    # --- sanity: radial grids should all match
    r = r_desc_fapi
    if not (np.allclose(r, r_desc_tempo) and np.allclose(r, r_pk_fapi) and
            np.allclose(r, r_pk_tempo) and np.allclose(r, r_def_fapi) and
            np.allclose(r, r_def_tempo)):
        raise RuntimeError("Radius bins for the two datasets / metrics do not match.")

    base = out_prefix

    # --- plots
    plot_profile(
        r,
        I_m_fapi,
        I_s_fapi,
        I_m_tempo,
        I_s_tempo,
        ylabel="Normalized intensity",
        title="Radial mean intensity",
        out_path=base.parent / (base.name + "_radial_intensity_profiles.png"),
    )

    plot_profile(
        r,
        H_m_fapi,
        H_s_fapi,
        H_m_tempo,
        H_s_tempo,
        ylabel="Entropy H(r/R) [bits]",
        title="Radial entropy profile",
        out_path=base.parent / (base.name + "_radial_entropy_profiles.png"),
    )

    plot_profile(
        r,
        A_m_fapi,
        A_s_fapi,
        A_m_tempo,
        A_s_tempo,
        ylabel=r"Texture anisotropy $A_{tex}(r/R)$",
        title="Radial texture anisotropy",
        out_path=base.parent / (base.name + "_radial_anisotropy_profiles.png"),
    )

    plot_profile(
        r,
        polar_mean_fapi,
        polar_std_fapi,
        polar_mean_tempo,
        polar_std_tempo,
        ylabel="Radial polar anisotropy",
        title="Radial polar anisotropy",
        out_path=base.parent / (base.name + "_radial_polar_anisotropy.png"),
    )

    plot_profile(
        r,
        kinetic_mean_fapi,
        kinetic_std_fapi,
        kinetic_mean_tempo,
        kinetic_std_tempo,
        ylabel="Radial kinetic heterogeneity (CV)",
        title="Radial kinetic heterogeneity",
        out_path=base.parent / (base.name + "_radial_kinetic_heterogeneity.png"),
    )

    plot_profile(
        r,
        D_m_fapi,
        D_s_fapi,
        D_m_tempo,
        D_s_tempo,
        ylabel=f"Defect fraction (threshold={args.defect_threshold:.2f})",
        title="Radial defect fraction",
        out_path=base.parent / (base.name + "_radial_defect_fraction.png"),
    )

    # --- CSV for polar + kinetic
    out_csv = base.parent / (base.name + "_radial_polar_kinetic_profiles.csv")
    header = ",".join(
        [
            "r_over_R",
            "polar_mean_FAPI",
            "polar_std_FAPI",
            "polar_mean_FAPI_TEMPO",
            "polar_std_FAPI_TEMPO",
            "kinetic_mean_FAPI",
            "kinetic_std_FAPI",
            "kinetic_mean_FAPI_TEMPO",
            "kinetic_std_FAPI_TEMPO",
        ]
    )
    data = np.column_stack(
        [
            r,
            polar_mean_fapi,
            polar_std_fapi,
            polar_mean_tempo,
            polar_std_tempo,
            kinetic_mean_fapi,
            kinetic_std_fapi,
            kinetic_mean_tempo,
            kinetic_std_tempo,
        ]
    )
    np.savetxt(out_csv, data, delimiter=",", header=header, comments="")
    print(f"[OK] saved {out_csv}")


if __name__ == "__main__":
    main()
