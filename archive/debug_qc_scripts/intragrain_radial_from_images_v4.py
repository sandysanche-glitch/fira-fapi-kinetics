#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intragrain_radial_from_images_v4.py

Compute intragrain radial defect profiles from COCO JSONs + heatmap images.

For each dataset (FAPI / FAPI–TEMPO):
  - For every grain (category == grain_cat_id):
      * Decode grain mask (COCO RLE, including compressed strings).
      * Optionally find nucleus mask (category == nuc_cat_id) and use
        its centroid as radial origin; otherwise use grain centroid.
      * Form union of defect masks (category == defect_cat_id).
      * Intersect grain mask with defect mask -> defect pixels.
      * Load heatmap image and ensure same H x W.
      * For each pixel in grain:
           r_px  = distance from origin (in pixels)
           r_um  = r_px / px_per_um
        Bin r_um into n_bins radial bins.
        Accumulate:
           total_area[bin]      += 1
           total_defect[bin]    += 1 if pixel is defect else 0

  - After all grains:
      defect_fraction[bin] = total_defect[bin] / total_area[bin]

Outputs:
  - <out-prefix>_FAPI_radial_defect_profile.csv
  - <out-prefix>_TEMPO_radial_defect_profile.csv
  - <out-prefix>_radial_defect_profile.png (comparison plot)
"""

import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

try:
    from pycocotools import mask as mask_utils
except ImportError as e:
    raise ImportError(
        "pycocotools is required for this script. "
        "Install with: pip install pycocotools"
    ) from e


# ----------------------------------------------------------------------
# COCO helpers
# ----------------------------------------------------------------------

def load_coco(json_path):
    """Load a COCO JSON file into a dict."""
    with open(json_path, "r") as f:
        coco = json.load(f)
    if not isinstance(coco, dict) or "images" not in coco or "annotations" not in coco:
        raise ValueError(f"{json_path} does not appear to be a standard COCO dict.")
    return coco


def build_index(coco):
    """Build simple indices: image_id -> image, image_id -> list(annotations)."""
    images = {img["id"]: img for img in coco.get("images", [])}
    anns_by_img = {}
    for ann in coco.get("annotations", []):
        img_id = ann["image_id"]
        anns_by_img.setdefault(img_id, []).append(ann)
    return images, anns_by_img


def decode_segmentation(segm, height, width):
    """
    Decode a COCO segmentation into a binary mask (H x W; dtype=uint8).

    Handles:
      - list of polygons
      - dict RLE
      - compressed RLE string (detectron2-style)
    """
    if isinstance(segm, list):
        # Polygon(s)
        rles = mask_utils.frPyObjects(segm, height, width)
        rle = mask_utils.merge(rles)
        m = mask_utils.decode(rle)
    elif isinstance(segm, dict):
        # RLE dict
        m = mask_utils.decode(segm)
    elif isinstance(segm, str):
        # Compressed RLE string: we need size from (height, width)
        rle = {"size": [height, width], "counts": segm.encode("ascii")}
        m = mask_utils.decode(rle)
    else:
        raise TypeError(f"Unsupported segmentation type: {type(segm)}")

    # pycocotools returns mask with shape (H, W, 1) or (H, W)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(np.uint8)


def centroid_from_mask(mask):
    """Compute centroid (y,x) of a binary mask."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(ys.mean()), float(xs.mean())


# ----------------------------------------------------------------------
# Core per-dataset processing
# ----------------------------------------------------------------------

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
    out_prefix
):
    """
    Process one dataset folder (FAPI or FAPI–TEMPO) containing:
      - multiple COCO JSON files (one per image).
      - for each image file_name, a corresponding heatmap image:
          <file_name base> + heatmap_suffix

    Returns:
      bin_centers_um (1D array), defect_fraction (1D array)
    and saves a CSV file.
    """
    dir_path = Path(dir_path)
    json_files = sorted(dir_path.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"No JSON files found in {dir_path}")

    all_r_um = []
    all_defect_flags = []

    n_decode_fail = 0
    n_grains_used = 0

    for jf in json_files:
        coco = load_coco(str(jf))
        images, anns_by_img = build_index(coco)

        for img_id, img in images.items():
            # Find associated annotations
            anns = anns_by_img.get(img_id, [])
            if not anns:
                continue

            file_name = img["file_name"]
            H = img["height"]
            W = img["width"]

            # Figure out heatmap path (same base name + suffix)
            base = Path(file_name).stem
            heatmap_path = dir_path / f"{base}{heatmap_suffix}"
            if not heatmap_path.is_file():
                # No heatmap -> skip this image (we need the intensity map)
                # You can comment this out if you want to run without heatmaps.
                # print(f"[WARN] heatmap not found for {file_name}, skipping image")
                continue

            # Load heatmap just to verify size / potential later use
            heat_img = Image.open(heatmap_path).convert("L")
            heat = np.array(heat_img)
            if heat.shape != (H, W):
                print(f"[WARN] Heatmap shape {heat.shape} != ({H},{W}) for {heatmap_path}")
                continue

            # Group annotations by category
            grain_anns = [a for a in anns if a.get("category_id") == grain_cat_id]
            nuc_anns   = [a for a in anns if a.get("category_id") == nuc_cat_id] if nuc_cat_id is not None else []
            defect_anns = [a for a in anns if a.get("category_id") == defect_cat_id]

            # Pre-decode nucleus masks once per image
            nuc_masks = []
            for na in nuc_anns:
                try:
                    nm = decode_segmentation(na["segmentation"], H, W)
                    nuc_masks.append(nm)
                except Exception as e:
                    n_decode_fail += 1

            # Pre-decode defect masks into union
            defect_union = np.zeros((H, W), dtype=np.uint8)
            for da in defect_anns:
                try:
                    dm = decode_segmentation(da["segmentation"], H, W)
                    defect_union |= (dm > 0).astype(np.uint8)
                except Exception as e:
                    n_decode_fail += 1

            # For each grain
            for ga in grain_anns:
                try:
                    gm = decode_segmentation(ga["segmentation"], H, W)
                except Exception as e:
                    n_decode_fail += 1
                    continue

                gm = (gm > 0).astype(np.uint8)
                area_px = int(gm.sum())
                if area_px < min_pixels:
                    # Too small, skip
                    continue

                # Origin: use nucleus centroid if available, else grain centroid
                if nuc_masks:
                    # Choose the nucleus most overlapped with this grain
                    best_nuc = None
                    best_overlap = 0
                    for nm in nuc_masks:
                        overlap = int(((nm > 0) & (gm > 0)).sum())
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_nuc = nm
                    if best_nuc is not None and best_overlap > 0:
                        cy_cx = centroid_from_mask(best_nuc & gm)
                    else:
                        cy_cx = centroid_from_mask(gm)
                else:
                    cy_cx = centroid_from_mask(gm)

                if cy_cx is None:
                    continue
                cy, cx = cy_cx

                # Construct masks for coordinates
                ys, xs = np.nonzero(gm)
                if len(xs) == 0:
                    continue

                # Radii in pixels and micrometres
                dy = ys.astype(np.float32) - cy
                dx = xs.astype(np.float32) - cx
                r_px = np.sqrt(dx * dx + dy * dy)
                r_um = r_px / float(px_per_um)

                # Defect flag for those pixels
                defect_mask = defect_union[ys, xs] > 0
                defect_flag = defect_mask.astype(np.int8)

                all_r_um.append(r_um)
                all_defect_flags.append(defect_flag)
                n_grains_used += 1

    if n_grains_used == 0:
        raise RuntimeError(f"No valid grains found in {dir_path}")

    # Concatenate all grains
    all_r_um = np.concatenate(all_r_um).astype(np.float64)
    all_defect_flags = np.concatenate(all_defect_flags).astype(np.float64)

    # Define radial bins in µm
    r_max = all_r_um.max()
    bin_edges = np.linspace(0.0, r_max, n_bins + 1, dtype=np.float64)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Accumulate per-bin
    total_area = np.zeros(n_bins, dtype=np.float64)
    total_def = np.zeros(n_bins, dtype=np.float64)

    inds = np.digitize(all_r_um, bin_edges) - 1  # bin index in [0, n_bins-1]
    valid = (inds >= 0) & (inds < n_bins)
    inds = inds[valid]
    flags = all_defect_flags[valid]

    # Use bincount to accumulate
    area_counts = np.bincount(inds, minlength=n_bins)
    defect_counts = np.bincount(inds, weights=flags, minlength=n_bins)

    total_area += area_counts.astype(np.float64)
    total_def += defect_counts.astype(np.float64)

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        defect_fraction = np.where(total_area > 0, total_def / total_area, np.nan)

    # Save CSV
    out_csv = f"{out_prefix}_{label}_radial_defect_profile.csv"
    import csv
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["r_um", "defect_fraction"])
        for r, df in zip(bin_centers, defect_fraction):
            writer.writerow([r, df])

    print(
        f"[OK] {label}: used {n_grains_used} grains, "
        f"{n_decode_fail} segmentation decode warnings. "
        f"Saved radial profile to {out_csv}"
    )

    return bin_centers, defect_fraction


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def plot_comparison(
    r_fapi,
    df_fapi,
    r_tempo,
    df_tempo,
    out_png,
):
    plt.figure(figsize=(4.5, 3.5))
    plt.plot(r_fapi, df_fapi, marker="o", linestyle="-", label="FAPI")
    plt.plot(r_tempo, df_tempo, marker="s", linestyle="-", label="FAPI–TEMPO")

    plt.xlabel("radius from nucleus (µm)")
    plt.ylabel("average defect fraction")
    plt.title("Intragrain radial defect profile")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[OK] comparison plot saved: {out_png}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Intragrain radial defect profiles from COCO JSON + heatmaps."
    )
    parser.add_argument(
        "--fapi-dir",
        required=True,
        help="Folder with FAPI COCO JSONs + heatmaps",
    )
    parser.add_argument(
        "--tempo-dir",
        required=True,
        help="Folder with FAPI–TEMPO COCO JSONs + heatmaps",
    )
    parser.add_argument(
        "--out-prefix",
        required=True,
        help="Output prefix (directory + base name)",
    )
    parser.add_argument(
        "--heatmap-suffix",
        default="_heatmap.png",
        help="Suffix appended to image base name to get heatmap path "
             "(default: _heatmap.png)",
    )
    parser.add_argument(
        "--grain-cat-id",
        type=int,
        default=1,
        help="Category id for grains (default: 1)",
    )
    parser.add_argument(
        "--nuc-cat-id",
        type=int,
        default=2,
        help="Category id for nuclei (default: 2)",
    )
    parser.add_argument(
        "--defect-cat-id",
        type=int,
        default=3,
        help="Category id for defects (default: 3)",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=25,
        help="Number of radial bins (default: 25)",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=200,
        help="Minimum grain area in pixels (default: 200)",
    )
    parser.add_argument(
        "--px-per-um",
        type=float,
        default=2.20014,
        help="Pixel-per-micrometre calibration (default: 2.20014)",
    )

    args = parser.parse_args()
    out_prefix = args.out_prefix

    print(f"[INFO] Processing FAPI dataset in {args.fapi_dir} ...")
    r_fapi, df_fapi = process_dataset(
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
    r_tempo, df_tempo = process_dataset(
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

    # For the plot, resample TEMPO onto FAPI radii if needed (simple interp)
    # so both curves share the same x-axis.
    if not np.allclose(r_fapi, r_tempo):
        df_tempo_interp = np.interp(
            r_fapi,
            r_tempo,
            df_tempo,
            left=np.nan,
            right=np.nan,
        )
        r_plot = r_fapi
        df_fapi_plot = df_fapi
        df_tempo_plot = df_tempo_interp
    else:
        r_plot = r_fapi
        df_fapi_plot = df_fapi
        df_tempo_plot = df_tempo

    out_png = f"{out_prefix}_radial_defect_profile.png"
    plot_comparison(r_plot, df_fapi_plot, r_plot, df_tempo_plot, out_png)


if __name__ == "__main__":
    main()
