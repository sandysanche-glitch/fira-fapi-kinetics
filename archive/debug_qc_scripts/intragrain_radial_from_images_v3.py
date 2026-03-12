#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intragrain_radial_from_images_v3.py

Compute average intragrain radial defect profiles for FAPI and FAPI–TEMPO
using:
  - JSON annotation files (Detectron2/COCO-like per-image JSON)
  - Heatmap images (same basename + user-specified suffix)

Assumptions for your dataset:
  - Grains:  category_id = 1
  - Nuclei:  category_id = 2
  - Defects: category_id = 3
  - JSONs are lists of annotations (one file per image).
  - Heatmaps live in the same folder as the JSONs and are named:
        <basename><heatmap_suffix>
    e.g. FAPI_0.json -> FAPI_0_heatmap.png

Radial coordinate:
  - We compute r_norm in [0, 1] (normalised by each grain radius).
  - We also compute an approximate physical radius in µm using px_per_um.

Outputs:
  - <out_prefix>_FAPI_radial_defects_norm.csv
  - <out_prefix>_FAPITEMPO_radial_defects_norm.csv
    (columns: r_norm, defect_fraction)
  - <out_prefix>_radial_defects_norm.png (FAPI vs FAPI–TEMPO)
  - <out_prefix>_radial_defects_um.png   (FAPI vs FAPI–TEMPO, µm)

Run example (your paths):

  python intragrain_radial_from_images_v3.py ^
    --fapi-dir "D:\\SWITCHdrive\\Institution\\Sts_grain morphology_ML\\comparative datasets\\FAPI" ^
    --tempo-dir "D:\\SWITCHdrive\\Institution\\Sts_grain morphology_ML\\comparative datasets\\FAPI-TEMPO" ^
    --out-prefix "D:\\SWITCHdrive\\Institution\\Sts_grain morphology_ML\\comparative datasets\\intragrain_from_images" ^
    --heatmap-suffix "_heatmap.png" ^
    --grain-cat-id 1 ^
    --nuc-cat-id 2 ^
    --defect-cat-id 3 ^
    --n-bins 25 ^
    --min-pixels 200 ^
    --px-per-um 2.20014

"""

import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

try:
    from pycocotools import mask as maskUtils
except ImportError:
    maskUtils = None
    print("[WARN] pycocotools not found. RLE segmentations will not work.")


def decode_segmentation(segmentation, img_h, img_w):
    """
    Decode a COCO-like segmentation into a boolean mask of shape (H, W).

    Handles:
      - RLE dict (pycocotools)
      - polygon list (single polygon)
    """
    if isinstance(segmentation, dict):
        if maskUtils is None:
            raise RuntimeError(
                "pycocotools is required to decode RLE segmentation but is not installed."
            )
        rle = maskUtils.frPyObjects(segmentation, img_h, img_w)
        m = maskUtils.decode(rle)
        if m.ndim == 3:
            m = m[:, :, 0]
        return m.astype(bool)

    if isinstance(segmentation, list):
        # Assume polygon(s) in [x0,y0,x1,y1,...] format
        import cv2

        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        pts_list = []

        for poly in segmentation:
            coords = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts = coords.astype(np.int32)
            pts_list.append(pts)

        if pts_list:
            cv2.fillPoly(mask, pts_list, 1)
        return mask.astype(bool)

    raise ValueError(f"Unsupported segmentation format: {type(segmentation)}")


def radial_bins_for_mask(mask, center, n_bins):
    """
    Given a boolean mask (grain) and a center (cy, cx),
    compute integer bin indices 0..(n_bins-1) for each pixel inside the mask,
    based on NORMALISED radius: r_norm = r / r_max.

    Returns:
      bin_idx[mask] : array of bin indices for mask pixels
      r_norm[mask]  : array of normalised radii for mask pixels
      r_max         : maximum radius within the mask (in pixels)
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None, None, 0.0

    dy = ys - center[0]
    dx = xs - center[1]
    r = np.sqrt(dx**2 + dy**2)
    r_max = r.max()
    if r_max <= 0:
        return None, None, 0.0

    r_norm = r / r_max
    # avoid exactly 1.0 -> clip slightly
    r_norm = np.clip(r_norm, 0.0, 0.999999)

    bin_idx = (r_norm * n_bins).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    return bin_idx, r_norm, r_max


def process_single_json(
    json_path,
    heatmap_path,
    grain_cat_id,
    nuc_cat_id,
    defect_cat_id,
    n_bins,
    min_pixels,
):
    """
    Process one JSON + heatmap pair and return:

      - grain_bin_defect_counts: shape (n_bins,)
      - grain_bin_counts:       shape (n_bins,)
      - list of grain radii (pixels) used
    """
    # Load annotations list
    with open(json_path, "r") as f:
        coco_list = json.load(f)

    # Load heatmap
    import cv2

    hm = cv2.imread(str(heatmap_path), cv2.IMREAD_GRAYSCALE)
    if hm is None:
        print(f"[WARN] Could not read heatmap {heatmap_path}, skipping.")
        return None, None, None

    H, W = hm.shape

    # Build masks by category
    grain_masks = []
    nuc_masks = []
    defect_mask = np.zeros((H, W), dtype=bool)

    # Category sets:
    grain_cats = {grain_cat_id}
    if nuc_cat_id is not None:
        grain_cats.add(nuc_cat_id)

    for ann in coco_list:
        cid = ann.get("category_id", -1)
        seg = ann.get("segmentation", None)
        if seg is None:
            continue

        try:
            m = decode_segmentation(seg, H, W)
        except Exception as e:
            print(f"[WARN] Segmentation decode failed in {json_path}: {e}")
            continue

        area = m.sum()
        if area < min_pixels:
            continue

        if cid in grain_cats:
            grain_masks.append((cid, m))
        elif cid == defect_cat_id:
            # accumulate all defects in a single mask
            defect_mask |= m

    if len(grain_masks) == 0:
        return None, None, None

    grain_bin_defect_counts = np.zeros(n_bins, dtype=np.float64)
    grain_bin_counts = np.zeros(n_bins, dtype=np.float64)
    radii_pixels = []

    for cid, gm in grain_masks:
        # find center: prefer nucleus if available (cid == nuc_cat_id),
        # otherwise centroid of this grain
        ys, xs = np.nonzero(gm)
        if ys.size == 0:
            continue

        if cid == nuc_cat_id:
            cy = ys.mean()
            cx = xs.mean()
        else:
            # centroid of grain
            cy = ys.mean()
            cx = xs.mean()

        bin_idx, r_norm, r_max = radial_bins_for_mask(gm, (cy, cx), n_bins)
        if bin_idx is None:
            continue

        radii_pixels.append(r_max)

        # defect pixels inside this grain
        defect_in_grain = gm & defect_mask
        ys_g, xs_g = np.nonzero(gm)
        ys_d, xs_d = np.nonzero(defect_in_grain)

        # for grain pixels:
        grain_bins = bin_idx.copy()

        # defect bins: we must compute their bins via same center
        dy_d = ys_d - cy
        dx_d = xs_d - cx
        r_d = np.sqrt(dx_d**2 + dy_d**2)
        if r_max > 0:
            r_d_norm = np.clip(r_d / r_max, 0.0, 0.999999)
            defect_bins = (r_d_norm * n_bins).astype(int)
            defect_bins = np.clip(defect_bins, 0, n_bins - 1)
        else:
            defect_bins = np.zeros_like(r_d, dtype=int)

        # accumulate counts in radial bins
        for b in grain_bins:
            grain_bin_counts[b] += 1.0
        for b in defect_bins:
            grain_bin_defect_counts[b] += 1.0

    if grain_bin_counts.sum() == 0:
        return None, None, None

    return grain_bin_defect_counts, grain_bin_counts, radii_pixels


def process_dataset(
    dir_path,
    heatmap_suffix,
    grain_cat_id,
    nuc_cat_id,
    defect_cat_id,
    n_bins,
    min_pixels,
    px_per_um,
    label,
    out_prefix,
):
    """
    Process all JSON + heatmap pairs in a dataset folder.

    Returns:
      r_norm_centers, avg_defect_norm,
      r_um_centers,   avg_defect_norm (same fractions, different x-axis)
    """
    dir_path = Path(dir_path)
    json_files = sorted(dir_path.glob("*.json"))

    # accumulate over all images
    total_defect_counts = np.zeros(n_bins, dtype=np.float64)
    total_grain_counts = np.zeros(n_bins, dtype=np.float64)
    all_radii_pixels = []

    for jf in json_files:
        base = jf.stem  # e.g. FAPI_0
        heatmap_path = dir_path / f"{base}{heatmap_suffix}"
        if not heatmap_path.is_file():
            # you can comment this if you want fewer warnings
            print(f"[WARN] heatmap not found for {jf.name}, skipping")
            continue

        res = process_single_json(
            jf,
            heatmap_path,
            grain_cat_id=grain_cat_id,
            nuc_cat_id=nuc_cat_id,
            defect_cat_id=defect_cat_id,
            n_bins=n_bins,
            min_pixels=min_pixels,
        )
        if res is None:
            continue
        gb_def, gb_tot, radii = res
        total_defect_counts += gb_def
        total_grain_counts += gb_tot
        all_radii_pixels.extend(radii)

    if total_grain_counts.sum() == 0:
        raise RuntimeError(f"No valid grains found in {dir_path}")

    defect_fraction = total_defect_counts / np.maximum(total_grain_counts, 1.0)
    r_norm_centers = (np.arange(n_bins) + 0.5) / n_bins

    # approximate physical radius axis:
    if len(all_radii_pixels) > 0:
        mean_r_pix = float(np.mean(all_radii_pixels))
    else:
        mean_r_pix = 1.0

    r_pix_centers = r_norm_centers * mean_r_pix
    r_um_centers = r_pix_centers / px_per_um

    # save CSV
    import pandas as pd

    df_norm = pd.DataFrame(
        {
            "r_norm": r_norm_centers,
            "defect_fraction": defect_fraction,
        }
    )
    out_csv_norm = f"{out_prefix}_{label}_radial_defects_norm.csv"
    df_norm.to_csv(out_csv_norm, index=False)
    print(f"[OK] saved {out_csv_norm}")

    df_um = pd.DataFrame(
        {
            "r_um": r_um_centers,
            "defect_fraction": defect_fraction,
        }
    )
    out_csv_um = f"{out_prefix}_{label}_radial_defects_um.csv"
    df_um.to_csv(out_csv_um, index=False)
    print(f"[OK] saved {out_csv_um}")

    return r_norm_centers, defect_fraction, r_um_centers


def main():
    parser = argparse.ArgumentParser(
        description="Intragrain radial defect profiles from JSON + heatmap images."
    )
    parser.add_argument(
        "--fapi-dir",
        required=True,
        help="Folder with FAPI JSON and heatmap images",
    )
    parser.add_argument(
        "--tempo-dir",
        required=True,
        help="Folder with FAPI–TEMPO JSON and heatmap images",
    )
    parser.add_argument(
        "--out-prefix",
        required=True,
        help="Output prefix (no extension)",
    )
    parser.add_argument(
        "--heatmap-suffix",
        default="_heatmap.png",
        help="Suffix appended to JSON basename to get heatmap filename",
    )
    parser.add_argument(
        "--grain-cat-id",
        type=int,
        default=1,
        help="Category ID for grains (outer spherulite)",
    )
    parser.add_argument(
        "--nuc-cat-id",
        type=int,
        default=2,
        help="Category ID for nuclei (used as centre if present)",
    )
    parser.add_argument(
        "--defect-cat-id",
        type=int,
        default=3,
        help="Category ID for defects",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=25,
        help="Number of radial bins",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=200,
        help="Minimum grain area in pixels to be considered",
    )
    parser.add_argument(
        "--px-per-um",
        type=float,
        default=2.20014,
        help="Pixel-per-micrometre calibration",
    )

    args = parser.parse_args()

    out_prefix = args.out_prefix

    print(f"[INFO] Processing FAPI dataset in {args.fapi_dir} ...")
    r_norm_fapi, dfrac_fapi, r_um_fapi = process_dataset(
        dir_path=args.fapi_dir,
        heatmap_suffix=args.heatmap_suffix,
        grain_cat_id=args.grain_cat_id,
        nuc_cat_id=args.nuc_cat_id,
        defect_cat_id=args.defect_cat_id,
        n_bins=args.n_bins,
        min_pixels=args.min_pixels,
        px_per_um=args.px_per_um,
        label="FAPI",
        out_prefix=out_prefix,
    )

    print(f"[INFO] Processing FAPI–TEMPO dataset in {args.tempo_dir} ...")
    r_norm_tempo, dfrac_tempo, r_um_tempo = process_dataset(
        dir_path=args.tempo_dir,
        heatmap_suffix=args.heatmap_suffix,
        grain_cat_id=args.grain_cat_id,
        nuc_cat_id=args.nuc_cat_id,
        defect_cat_id=args.defect_cat_id,
        n_bins=args.n_bins,
        min_pixels=args.min_pixels,
        px_per_um=args.px_per_um,
        label="FAPITEMPO",
        out_prefix=out_prefix,
    )

    # --- Plot in normalised radius ---
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    ax.plot(r_norm_fapi, dfrac_fapi, "-o", label="FAPI")
    ax.plot(r_norm_tempo, dfrac_tempo, "-s", label="FAPI–TEMPO")
    ax.set_xlabel("normalised radius $r / R$")
    ax.set_ylabel("defect fraction")
    ax.set_title("Intragrain defect profiles (normalised)")
    ax.legend()
    ax.grid(False)
    fig.tight_layout()
    out_png_norm = f"{out_prefix}_radial_defects_norm.png"
    fig.savefig(out_png_norm, dpi=300)
    plt.close(fig)
    print(f"[OK] saved {out_png_norm}")

    # --- Plot vs approximate physical radius (µm) ---
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    ax.plot(r_um_fapi, dfrac_fapi, "-o", label="FAPI")
    ax.plot(r_um_tempo, dfrac_tempo, "-s", label="FAPI–TEMPO")
    ax.set_xlabel("radius (µm) [approx.]")
    ax.set_ylabel("defect fraction")
    ax.set_title("Intragrain defect profiles (µm scale)")
    ax.legend()
    ax.grid(False)
    fig.tight_layout()
    out_png_um = f"{out_prefix}_radial_defects_um.png"
    fig.savefig(out_png_um, dpi=300)
    plt.close(fig)
    print(f"[OK] saved {out_png_um}")

    print("[DONE] Intragrain radial defect profiles computed.")


if __name__ == "__main__":
    main()
