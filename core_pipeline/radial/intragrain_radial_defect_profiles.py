#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intragrain_radial_defect_profiles.py

Compute average intra-grain radial defect-density profiles for
FAPI and FAPI–TEMPO from COCO-style JSON annotations.

For each dataset:
  - grains are defined by "grain" category masks
  - defects by "defect" category masks
  - for each grain, we compute the fraction of pixels that are
    defective as a function of normalised radius r* = r / r_max
  - profiles are then averaged across grains

Outputs:
  - PNG plot: FAPI vs FAPI–TEMPO average profiles (+/- 1 std)
  - CSV files with binned means and standard deviations

Usage example:

  python intragrain_radial_defect_profiles.py ^
    --fapi-json D:\...\FAPI_annotations.json ^
    --tempo-json D:\...\FAPITEMPO_annotations.json ^
    --out-prefix D:\...\intragrain_defect_profiles ^
    --n-bins 25

Requires:
  - numpy
  - pandas
  - matplotlib
  - pycocotools (pip install pycocotools)
"""

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pycocotools import mask as maskUtils


# ----------------- helpers ----------------- #

def load_coco(json_path):
    """Load COCO-style JSON and index images/annotations/categories."""
    with open(json_path, "r") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    cats = {cat["id"]: cat for cat in data["categories"]}

    anns_by_img = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        anns_by_img.setdefault(img_id, []).append(ann)

    return images, anns_by_img, cats


def find_category_ids(cats, grain_name="grain", defect_name="defect"):
    """
    Find category_ids for grains and defects by matching category 'name'.
    Adjust grain_name/defect_name if your JSON uses different labels.
    """
    grain_ids = [cid for cid, c in cats.items()
                 if grain_name.lower() in c["name"].lower()]
    defect_ids = [cid for cid, c in cats.items()
                  if defect_name.lower() in c["name"].lower()]

    if not grain_ids:
        raise ValueError(f"Could not find grain category matching '{grain_name}'")
    if not defect_ids:
        raise ValueError(f"Could not find defect category matching '{defect_name}'")

    return grain_ids, defect_ids


def rle_to_mask(rle, height, width):
    """Decode COCO RLE (or polygon) to a full-resolution binary mask."""
    # If segmentation is polygon, convert to RLE first
    if isinstance(rle, list):
        rles = maskUtils.frPyObjects(rle, height, width)
        rle = maskUtils.merge(rles)
    elif isinstance(rle, dict) and "counts" in rle and isinstance(rle["counts"], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(rle, height, width)

    m = maskUtils.decode(rle)
    # maskUtils.decode can return HxW or HxWx1
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(bool)


def compute_union_defect_masks(images, anns_by_img, defect_cat_ids):
    """
    For each image, compute the union of all defect masks.
    Returns dict: img_id -> (defect_mask: bool array of shape [H, W])
    """
    union_defects = {}

    for img_id, img_info in images.items():
        h, w = img_info["height"], img_info["width"]
        m_defect = np.zeros((h, w), dtype=bool)

        for ann in anns_by_img.get(img_id, []):
            if ann["category_id"] in defect_cat_ids:
                m = rle_to_mask(ann["segmentation"], h, w)
                m_defect |= m

        union_defects[img_id] = m_defect

    return union_defects


def radial_defect_profile_for_grain(
    grain_mask: np.ndarray,
    defect_mask: np.ndarray,
    n_bins: int = 25,
):
    """
    Compute radial defect fraction profile for a single grain.

    Parameters
    ----------
    grain_mask : bool array [H, W]
    defect_mask: bool array [H, W]
    n_bins     : number of radial bins

    Returns
    -------
    bin_centres : (n_bins,) array in [0, 1]
    profile     : (n_bins,) array of defect fractions; np.nan where no pixels
    """
    ys, xs = np.where(grain_mask)
    if ys.size == 0:
        return None, None

    # centroid in pixel coordinates
    cy = ys.mean()
    cx = xs.mean()

    # radial distance from centroid
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    r_max = r.max()
    if r_max <= 0:
        return None, None

    r_norm = r / r_max

    # defect flags for these pixels
    defect_flags = defect_mask[ys, xs].astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(r_norm, bins) - 1  # 0..n_bins-1

    profile = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask_b = (bin_ids == b)
        if np.any(mask_b):
            profile[b] = defect_flags[mask_b].mean()

    # bin centres
    bin_centres = 0.5 * (bins[:-1] + bins[1:])
    return bin_centres, profile


def collect_profiles_for_dataset(
    json_path: Path,
    grain_name: str = "grain",
    defect_name: str = "defect",
    n_bins: int = 25,
    min_pixels: int = 200,
):
    """
    Collect radial defect profiles for all grains in a dataset.

    Returns
    -------
    bin_centres: (n_bins,) array
    profiles   : list of (n_bins,) arrays (NaNs where bins empty)
    """
    images, anns_by_img, cats = load_coco(json_path)
    grain_cat_ids, defect_cat_ids = find_category_ids(cats, grain_name, defect_name)

    union_defects = compute_union_defect_masks(images, anns_by_img, defect_cat_ids)

    all_profiles = []
    bin_centres_ref = None

    for img_id, img_info in images.items():
        h, w = img_info["height"], img_info["width"]
        m_def = union_defects[img_id]

        for ann in anns_by_img.get(img_id, []):
            if ann["category_id"] not in grain_cat_ids:
                continue

            grain_mask = rle_to_mask(ann["segmentation"], h, w)
            if grain_mask.sum() < min_pixels:
                # skip tiny grains
                continue

            bin_centres, prof = radial_defect_profile_for_grain(
                grain_mask, m_def, n_bins=n_bins
            )
            if bin_centres is None:
                continue

            if bin_centres_ref is None:
                bin_centres_ref = bin_centres
            all_profiles.append(prof)

    if not all_profiles:
        raise RuntimeError(f"No valid grain profiles found in {json_path}")

    profiles_arr = np.vstack(all_profiles)  # (N_grains, n_bins)
    return bin_centres_ref, profiles_arr


def summarise_profiles(bin_centres, profiles_arr):
    """Compute mean, std and count per bin, ignoring NaNs."""
    mean = np.nanmean(profiles_arr, axis=0)
    std = np.nanstd(profiles_arr, axis=0)
    count = np.sum(~np.isnan(profiles_arr), axis=0)
    return pd.DataFrame(
        {
            "r_norm": bin_centres,
            "mean_defect_fraction": mean,
            "std_defect_fraction": std,
            "n_grains": count,
        }
    )


def plot_profiles(df_fapi, df_tempo, out_path: Path):
    """Plot FAPI vs FAPI–TEMPO mean ± std radial profiles."""
    fig, ax = plt.subplots(figsize=(5.0, 4.0))

    for df, label, color in [
        (df_fapi, "FAPI", "C0"),
        (df_tempo, "FAPI–TEMPO", "C1"),
    ]:
        r = df["r_norm"].values
        m = df["mean_defect_fraction"].values
        s = df["std_defect_fraction"].values

        ax.plot(r, m, color=color, lw=2, label=label)
        ax.fill_between(
            r,
            m - s,
            m + s,
            color=color,
            alpha=0.2,
            linewidth=0,
        )

    ax.set_xlabel("normalised radius $r^*$")
    ax.set_ylabel("defect fraction per shell")
    ax.set_title("Average radial defect profile")
    ax.legend()
    ax.grid(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] saved radial profile plot: {out_path}")


# ----------------- main ----------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Compute intra-grain radial defect profiles from COCO JSON."
    )
    parser.add_argument(
        "--fapi-json",
        required=True,
        help="COCO JSON for FAPI dataset",
    )
    parser.add_argument(
        "--tempo-json",
        required=True,
        help="COCO JSON for FAPI–TEMPO dataset",
    )
    parser.add_argument(
        "--out-prefix",
        required=True,
        help="Prefix for output files (PNG and CSVs)",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=25,
        help="Number of radial bins between 0 and 1",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=200,
        help="Minimum number of pixels for a grain to be included",
    )
    parser.add_argument(
        "--grain-name",
        default="grain",
        help="Substring of category name for grains (default: 'grain')",
    )
    parser.add_argument(
        "--defect-name",
        default="defect",
        help="Substring of category name for defects (default: 'defect')",
    )

    args = parser.parse_args()

    out_prefix = Path(args.out_prefix)
    out_dir = out_prefix.parent if out_prefix.parent != Path("") else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_prefix.name

    # FAPI
    bins_fapi, profs_fapi = collect_profiles_for_dataset(
        Path(args.fapi_json),
        grain_name=args.grain_name,
        defect_name=args.defect_name,
        n_bins=args.n_bins,
        min_pixels=args.min_pixels,
    )
    df_fapi = summarise_profiles(bins_fapi, profs_fapi)
    csv_fapi = out_dir / f"{base}_FAPI_radial_defect_profile.csv"
    df_fapi.to_csv(csv_fapi, index=False)
    print(f"[OK] saved FAPI radial profile CSV: {csv_fapi}")

    # FAPI–TEMPO
    bins_tempo, profs_tempo = collect_profiles_for_dataset(
        Path(args.tempo_json),
        grain_name=args.grain_name,
        defect_name=args.defect_name,
        n_bins=args.n_bins,
        min_pixels=args.min_pixels,
    )
    df_tempo = summarise_profiles(bins_tempo, profs_tempo)
    csv_tempo = out_dir / f"{base}_FAPITEMPO_radial_defect_profile.csv"
    df_tempo.to_csv(csv_tempo, index=False)
    print(f"[OK] saved FAPITEMPO radial profile CSV: {csv_tempo}")

    # Plot comparison
    png_path = out_dir / f"{base}_radial_defect_profile.png"
    plot_profiles(df_fapi, df_tempo, png_path)

    print("[DONE] intra-grain radial profiles computed.")


if __name__ == "__main__":
    main()
