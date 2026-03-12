#!/usr/bin/env python
"""
Intragrain radial polar-anisotropy and kinetic-heterogeneity profiles
from list-style JSON detections and heat-map images.

Assumptions
-----------
- Each JSON file (e.g. FAPI_0.json) is a list of dicts with keys:
    - 'segmentation': COCO RLE dict {'size': [H, W], 'counts': <str>}
    - (other keys like 'bbox', 'score', 'mask_name' are ignored here)
- For each JSON "<stem>.json" there is a heat-map image
    "<stem><intensity_suffix>", e.g. "FAPI_0_heatmap.png".
- Intensities in the heat-map are proportional to some kinetic field
  (e.g. defect probability, PL quenching, etc.).

Outputs
-------
- <out-prefix>_radial_polar_anisotropy.png
- <out-prefix>_radial_kinetic_heterogeneity.png
- <out-prefix>_radial_polar_kinetic_profiles.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

try:
    from pycocotools import mask as maskUtils
except ImportError as exc:
    raise ImportError(
        "This script requires pycocotools.\n"
        "On Windows try:\n"
        "   pip install pycocotools-windows\n"
        "On Linux/macOS:\n"
        "   pip install pycocotools\n"
    ) from exc


# ---------------------------------------------------------------------
# COCO RLE decoding
# ---------------------------------------------------------------------
def decode_segmentation(seg):
    """
    Decode a COCO RLE segmentation dict into a boolean mask.

    Parameters
    ----------
    seg : dict
        {'size': [H, W], 'counts': <str or bytes>}

    Returns
    -------
    mask : ndarray (H, W), bool
    """
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


# ---------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------
def process_dataset(
    dir_path: Path,
    intensity_suffix: str,
    n_bins: int,
    min_pixels: int,
):
    """
    Process one dataset (FAPI or FAPI–TEMPO).

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

    polar_profiles = []    # list of (n_bins,) arrays
    kinetic_profiles = []  # list of (n_bins,) arrays

    print(f"[INFO]   Found {len(json_files)} JSON files in {dir_path}")

    for jf in json_files:
        stem = jf.stem  # e.g. "FAPI_0" or "FAPI_TEMPO_0"

        # Heat-map image
        img_path = dir_path / f"{stem}{intensity_suffix}"
        if not img_path.is_file():
            print(f"[WARN]   no intensity image for {stem}; skipping")
            continue

        img = np.array(Image.open(img_path).convert("F"), dtype=np.float32)
        # Normalise image to [0, 1] to reduce drift between fields of view
        img = img - img.min()
        denom_img = img.max()
        if denom_img > 0:
            img /= denom_img

        # JSON: list of grain detections
        with open(jf, "r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"[WARN]   {jf.name} is not list-style JSON; skipping")
            continue

        for item in data:
            seg = item.get("segmentation", None)
            if seg is None:
                continue

            try:
                mask = decode_segmentation(seg)
            except Exception as e:
                print(f"[WARN]   segmentation decode failed in {jf.name}: {e}")
                continue

            ys, xs = np.nonzero(mask)
            if ys.size < min_pixels:
                # too small – probably noise or very tiny grain
                continue

            vals = img[ys, xs]

            # Normalise per grain to [0, 1] so curves are relative
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

            # require at least a small number of pixels per ring, scaled with grain size
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

                # --- kinetic heterogeneity: coefficient of variation within ring
                mu = float(ring_vals.mean())
                sigma = float(ring_vals.std())
                kinetic_profile[b] = sigma / (mu + 1e-6)

                # --- polar anisotropy: angular modulation of intensity within ring
                # centre the signal in the ring
                v_center = ring_vals - mu
                denom_a = float(np.mean(np.abs(v_center))) + 1e-6
                if denom_a <= 0:
                    A = 0.0
                else:
                    # complex "dipole" moment of intensity around the ring
                    A = np.abs(np.mean(v_center * np.exp(1j * ring_theta))) / denom_a
                polar_profile[b] = A

            if not np.all(np.isnan(polar_profile)):
                polar_profiles.append(polar_profile)
                kinetic_profiles.append(kinetic_profile)

    if len(polar_profiles) == 0:
        raise RuntimeError(f"No valid grains found in {dir_path}")

    polar_arr = np.array(polar_profiles, dtype=np.float64)
    kinetic_arr = np.array(kinetic_profiles, dtype=np.float64)

    r_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    polar_mean = np.nanmean(polar_arr, axis=0)
    polar_std = np.nanstd(polar_arr, axis=0)

    kinetic_mean = np.nanmean(kinetic_arr, axis=0)
    kinetic_std = np.nanstd(kinetic_arr, axis=0)

    n_grains = polar_arr.shape[0]
    return r_centres, polar_mean, polar_std, kinetic_mean, kinetic_std, n_grains


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def plot_metric(
    r,
    mean_fapi,
    std_fapi,
    mean_tempo,
    std_tempo,
    ylabel: str,
    out_path: Path,
):
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(r, mean_fapi, label="FAPI")
    ax.fill_between(r, mean_fapi - std_fapi, mean_fapi + std_fapi, alpha=0.25)

    ax.plot(r, mean_tempo, label="FAPI–TEMPO")
    ax.fill_between(r, mean_tempo - std_tempo, mean_tempo + std_tempo, alpha=0.25)

    ax.set_xlabel(r"Normalised radius $r/R$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] saved {out_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Compute radial polar-anisotropy and kinetic-heterogeneity curves "
            "for FAPI and FAPI–TEMPO from list-style JSON masks and heat-maps."
        )
    )
    p.add_argument(
        "--fapi-dir",
        required=True,
        help="Directory containing FAPI_*.json and corresponding *_heatmap images.",
    )
    p.add_argument(
        "--tempo-dir",
        required=True,
        help="Directory containing FAPI_TEMPO_*.json and *_heatmap images.",
    )
    p.add_argument(
        "--out-prefix",
        required=True,
        help="Output prefix (directory + base name) for PNG/CSV outputs.",
    )
    p.add_argument(
        "--intensity-suffix",
        default="_heatmap.png",
        help="Suffix for intensity images (default: '_heatmap.png').",
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
    return p.parse_args()


def main():
    args = parse_args()
    out_prefix = Path(args.out_prefix)

    # --- FAPI
    print(f"[INFO] Processing FAPI in {args.fapi_dir} ...")
    (
        r_fapi,
        polar_mean_fapi,
        polar_std_fapi,
        kinetic_mean_fapi,
        kinetic_std_fapi,
        n_grains_fapi,
    ) = process_dataset(
        Path(args.fapi_dir),
        intensity_suffix=args.intensity_suffix,
        n_bins=args.n_bins,
        min_pixels=args.min_pixels,
    )
    print(f"[INFO]   FAPI: used {n_grains_fapi} grains")

    # --- FAPI–TEMPO
    print(f"[INFO] Processing FAPI–TEMPO in {args.tempo_dir} ...")
    (
        r_tempo,
        polar_mean_tempo,
        polar_std_tempo,
        kinetic_mean_tempo,
        kinetic_std_tempo,
        n_grains_tempo,
    ) = process_dataset(
        Path(args.tempo_dir),
        intensity_suffix=args.intensity_suffix,
        n_bins=args.n_bins,
        min_pixels=args.min_pixels,
    )
    print(f"[INFO]   FAPI–TEMPO: used {n_grains_tempo} grains")

    if not np.allclose(r_fapi, r_tempo):
        raise RuntimeError("Radius bins for the two datasets do not match.")

    # --- plots
    out_polar_png = Path(str(out_prefix) + "_radial_polar_anisotropy.png")
    out_kin_png = Path(str(out_prefix) + "_radial_kinetic_heterogeneity.png")

    plot_metric(
        r_fapi,
        polar_mean_fapi,
        polar_std_fapi,
        polar_mean_tempo,
        polar_std_tempo,
        ylabel="Radial polar anisotropy",
        out_path=out_polar_png,
    )

    plot_metric(
        r_fapi,
        kinetic_mean_fapi,
        kinetic_std_fapi,
        kinetic_mean_tempo,
        kinetic_std_tempo,
        ylabel="Radial kinetic heterogeneity (CV)",
        out_path=out_kin_png,
    )

    # --- CSV with all curves
    out_csv = Path(str(out_prefix) + "_radial_polar_kinetic_profiles.csv")
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
            r_fapi,
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
