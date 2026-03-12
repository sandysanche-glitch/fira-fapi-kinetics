#!/usr/bin/env python
"""
Compute canonical-disk radial descriptors for FAPI and FAPI–TEMPO
from list-style JSON + mask PNGs + an intensity image per field.

Descriptors (per dataset):
- Radial mean intensity         -> underlying field for ΔH, φ
- Radial entropy profile        -> radial decomposition of ΔH
- Radial texture anisotropy     -> radial decomposition of A_tex

Usage example (heat-map as intensity):
python intragrain_radial_descriptors_from_listjson.py ^
  --fapi-dir "D:\SWITCHdrive\Institution\Sts_grain morphology_ML\comparative datasets\FAPI" ^
  --tempo-dir "D:\SWITCHdrive\Institution\Sts_grain morphology_ML\comparative datasets\FAPI-TEMPO" ^
  --out-prefix "D:\SWITCHdrive\Institution\Sts_grain morphology_ML\comparative datasets\intragrain_descriptors" ^
  --intensity-suffix "_heatmap.png" ^
  --n-bins 25 ^
  --min-pixels 200
"""

import argparse
from pathlib import Path
import json

import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt


# ---------- basic utilities ----------

def load_mask(mask_path: Path) -> np.ndarray:
    """Load binary mask (True = inside grain)."""
    img = imageio.imread(mask_path)
    if img.ndim == 3:
        img = img[..., 0]
    mask = img > 0
    return mask


def load_intensity(img_path: Path) -> np.ndarray:
    """Load intensity image as float32 in [0, 1]."""
    img = imageio.imread(img_path)
    if img.ndim == 3:
        # convert RGB to luminance
        img = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    img = img.astype(np.float32)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    return img


def radial_bins_for_mask(mask: np.ndarray, n_bins: int):
    """Return bin indices (0..n_bins-1) and valid-pixel mask inside grain."""
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
    hist, _ = np.histogram(values, bins=n_bins_hist, range=(0.0, 1.0), density=True)
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


# ---------- per-dataset processing ----------

def process_dataset(
    dir_path: Path,
    n_bins: int = 25,
    min_pixels: int = 200,
    intensity_suffix: str = "_heatmap.png",
    base_intensity_suffix_fallback: str = ".jpg",
):
    """
    For each JSON file in dir_path, use the list of masks and a single
    intensity image (heat-map or bright-field) to compute radial profiles.

    Returns:
        r_centers, mean_intensity, std_intensity,
        mean_entropy,   std_entropy,
        mean_aniso,     std_aniso
    """
    all_profiles_I = []
    all_profiles_H = []
    all_profiles_A = []

    json_files = sorted(dir_path.glob("*.json"))
    print(f"[INFO]   Found {len(json_files)} JSON files in {dir_path}")

    for jf in json_files:
        base = jf.stem  # e.g. FAPI_0

        # choose intensity image: first try suffix (heatmap), then jpg
        img_path = dir_path / f"{base}{intensity_suffix}"
        if not img_path.exists():
            img_path = dir_path / f"{base}{base_intensity_suffix_fallback}"
        if not img_path.exists():
            print(f"[WARN]   no intensity image for {base}; skipping")
            continue

        intensity = load_intensity(img_path)
        grad_mag, grad_theta = compute_image_gradients(intensity)

        with open(jf, "r") as f:
            entries = json.load(f)

        if not isinstance(entries, list):
            print(f"[WARN]   {jf.name} is not a list; skipping")
            continue

        for entry in entries:
            mask_name = entry.get("mask_name", None)
            if mask_name is None:
                continue
            mask_path = dir_path / mask_name
            if not mask_path.exists():
                print(f"[WARN]   mask {mask_name} missing; skipping this grain")
                continue

            mask = load_mask(mask_path)
            ys, xs, bin_idx, r_norm = radial_bins_for_mask(mask, n_bins)
            if ys is None:
                continue
            if ys.size < min_pixels:
                continue

            # per-bin accumulators
            I_bins = np.full(n_bins, np.nan, dtype=float)
            H_bins = np.full(n_bins, np.nan, dtype=float)
            A_bins = np.full(n_bins, np.nan, dtype=float)

            for b in range(n_bins):
                sel = bin_idx == b
                if np.count_nonzero(sel) < max(10, min_pixels // n_bins):
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
        raise RuntimeError(f"No valid grains found in {dir_path}")

    all_profiles_I = np.stack(all_profiles_I, axis=0)  # (n_grains, n_bins)
    all_profiles_H = np.stack(all_profiles_H, axis=0)
    all_profiles_A = np.stack(all_profiles_A, axis=0)

    r_centers = (np.arange(n_bins) + 0.5) / n_bins

    def nanmean_std(arr):
        return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)

    m_I, s_I = nanmean_std(all_profiles_I)
    m_H, s_H = nanmean_std(all_profiles_H)
    m_A, s_A = nanmean_std(all_profiles_A)

    print(f"[INFO]   used {all_profiles_I.shape[0]} grains from {dir_path}")

    return r_centers, m_I, s_I, m_H, s_H, m_A, s_A


# ---------- plotting ----------

def plot_two_profiles(
    r,
    m1,
    s1,
    m2,
    s2,
    label1,
    label2,
    ylabel,
    title,
    out_png: Path,
):
    plt.figure(figsize=(5.0, 4.0))
    plt.plot(r, m1, label=label1)
    plt.fill_between(r, m1 - s1, m1 + s1, alpha=0.2)
    plt.plot(r, m2, label=label2)
    plt.fill_between(r, m2 - s2, m2 + s2, alpha=0.2)
    plt.xlabel("Normalised radius $r/R$")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[OK] saved {out_png}")


# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(
        description="Radial intragrain descriptors from list JSON + masks + images"
    )
    p.add_argument("--fapi-dir", required=True, type=str)
    p.add_argument("--tempo-dir", required=True, type=str)
    p.add_argument("--out-prefix", required=True, type=str)
    p.add_argument("--n-bins", type=int, default=25)
    p.add_argument("--min-pixels", type=int, default=200)
    p.add_argument(
        "--intensity-suffix",
        type=str,
        default="_heatmap.png",
        help="suffix for intensity image (default: _heatmap.png). "
             "If not found, falls back to .jpg",
    )
    return p.parse_args()


def main():
    args = parse_args()

    fapi_dir = Path(args.fapi_dir)
    tempo_dir = Path(args.tempo_dir)
    out_prefix = Path(args.out_prefix)

    print(f"[INFO] Processing FAPI in {fapi_dir} ...")
    (
        r_fapi,
        I_m_fapi,
        I_s_fapi,
        H_m_fapi,
        H_s_fapi,
        A_m_fapi,
        A_s_fapi,
    ) = process_dataset(
        fapi_dir,
        n_bins=args.n_bins,
        min_pixels=args.min_pixels,
        intensity_suffix=args.intensity_suffix,
    )

    print(f"[INFO] Processing FAPI–TEMPO in {tempo_dir} ...")
    (
        r_tempo,
        I_m_tempo,
        I_s_tempo,
        H_m_tempo,
        H_s_tempo,
        A_m_tempo,
        A_s_tempo,
    ) = process_dataset(
        tempo_dir,
        n_bins=args.n_bins,
        min_pixels=args.min_pixels,
        intensity_suffix=args.intensity_suffix,
    )

    # sanity: radii are identical if n_bins was the same
    r = r_fapi

    # 1) radial intensity (underlying to ΔH / φ fields)
    plot_two_profiles(
        r,
        I_m_fapi,
        I_s_fapi,
        I_m_tempo,
        I_s_tempo,
        "FAPI",
        "FAPI–TEMPO",
        "Normalised intensity",
        "Radial mean intensity",
        out_prefix.parent / (out_prefix.name + "_radial_intensity_profiles.png"),
    )

    # 2) radial entropy (ΔH-like)
    plot_two_profiles(
        r,
        H_m_fapi,
        H_s_fapi,
        H_m_tempo,
        H_s_tempo,
        "FAPI",
        "FAPI–TEMPO",
        "Entropy H(r/R) [bits]",
        "Radial entropy profile",
        out_prefix.parent / (out_prefix.name + "_radial_entropy_profiles.png"),
    )

    # 3) radial texture anisotropy (A_tex-like)
    plot_two_profiles(
        r,
        A_m_fapi,
        A_s_fapi,
        A_m_tempo,
        A_s_tempo,
        "FAPI",
        "FAPI–TEMPO",
        "Texture anisotropy $A_{tex}(r/R)$",
        "Radial texture anisotropy",
        out_prefix.parent / (out_prefix.name + "_radial_anisotropy_profiles.png"),
    )


if __name__ == "__main__":
    main()
