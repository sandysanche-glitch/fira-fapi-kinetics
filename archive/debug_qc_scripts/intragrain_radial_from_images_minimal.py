#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
intragrain_radial_from_images_minimal.py

Compute average intragrain radial defect profiles for two datasets
(FAPI and FAPI–TEMPO) using COCO-style JSON with RLE segmentations.

Requirements:
  - numpy
  - matplotlib
  - pycocotools

Usage (Windows, from your project folder):

  python intragrain_radial_from_images_minimal.py ^
    --fapi-dir "D:\SWITCHdrive\Institution\Sts_grain morphology_ML\comparative datasets\FAPI" ^
    --tempo-dir "D:\SWITCHdrive\Institution\Sts_grain morphology_ML\comparative datasets\FAPI-TEMPO" ^
    --out-prefix "D:\SWITCHdrive\Institution\Sts_grain morphology_ML\comparative datasets\intragrain_profiles" ^
    --grain-cat-id 1 ^
    --defect-cat-id 3 ^
    --n-bins 25 ^
    --min-pixels 200 ^
    --px-per-um 2.20014
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    from pycocotools import mask as mask_util
except ImportError:
    raise ImportError(
        "pycocotools is required. Install with:\n"
        "  pip install pycocotools-windows\n"
        "or a suitable wheel for your environment."
    )


# -------------------------- COCO helpers -------------------------- #

def load_coco_dict(json_path: Path):
    """Load a COCO-style dict from a JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "annotations" in data:
        return data

    raise ValueError(f"{json_path} is not a COCO dict with 'annotations'.")


def build_image_index(coco):
    """Index images and annotations by image_id."""
    # images
    img_by_id = {}
    if "images" in coco:
        for img in coco["images"]:
            img_by_id[img["id"]] = img
    # annotations
    anns_by_img = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        anns_by_img.setdefault(img_id, []).append(ann)
    return img_by_id, anns_by_img


def decode_rle(segmentation, height=None, width=None):
    """
    Decode a segmentation that is COCO RLE.

    Handles:
      - dict with "size" and "counts" (possibly string)
      - list of such dicts (we take union)
    """
    if isinstance(segmentation, dict):
        rle = segmentation
        # ensure counts is bytes
        counts = rle.get("counts")
        if isinstance(counts, str):
            rle = {"size": rle["size"], "counts": counts.encode("ascii")}
        m = mask_util.decode(rle)
        return m.astype(np.uint8)

    if isinstance(segmentation, list):
        # polygon or list of RLEs; here we assume RLE list
        rles = []
        for seg in segmentation:
            if isinstance(seg, dict) and "counts" in seg:
                c = seg["counts"]
                if isinstance(c, str):
                    rles.append(
                        {"size": seg["size"], "counts": c.encode("ascii")}
                    )
        if not rles:
            raise ValueError("Unsupported segmentation format.")
        m = mask_util.decode(rles)
        # mask_util.decode returns (H, W, N); take union
        if m.ndim == 3:
            m = (m.sum(axis=2) > 0).astype(np.uint8)
        return m

    raise ValueError("Unsupported segmentation type for RLE decode.")


# ---------------------- radial profile logic ---------------------- #

def radial_profile_for_grain(grain_mask: np.ndarray,
                             defect_mask: np.ndarray,
                             n_bins: int):
    """
    Compute radial defect fraction profile for a single grain mask.

    Returns:
      bin_centers (array, shape [n_bins])
      defect_counts (array, shape [n_bins])
      total_counts (array, shape [n_bins])
    """
    ys, xs = np.nonzero(grain_mask)
    if ys.size == 0:
        return None, None, None

    # centroid
    cy = ys.mean()
    cx = xs.mean()

    # radial distances
    dy = ys - cy
    dx = xs - cx
    r = np.sqrt(dx * dx + dy * dy)
    r_max = r.max()
    if r_max <= 0:
        return None, None, None

    r_norm = r / r_max  # 0..1
    # bin indices
    bin_idx = np.floor(r_norm * n_bins).astype(int)
    bin_idx[bin_idx == n_bins] = n_bins - 1

    # defect mask restricted to grain pixels
    defect_pixels = defect_mask[ys, xs] > 0

    defect_counts = np.zeros(n_bins, dtype=np.float64)
    total_counts = np.zeros(n_bins, dtype=np.float64)

    for b, is_def in zip(bin_idx, defect_pixels):
        total_counts[b] += 1.0
        if is_def:
            defect_counts[b] += 1.0

    # bin centers in normalised radius
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return bin_centers, defect_counts, total_counts


def process_dataset(dir_path: Path,
                    grain_cat_id: int,
                    defect_cat_id: int,
                    n_bins: int,
                    min_pixels: int,
                    px_per_um: float,
                    label: str,
                    out_prefix: Path):
    """
    Process all JSONs in dir_path and compute average radial defect profile.

    Returns:
      bin_centers (np.ndarray)
      mean_defect_prob (np.ndarray)
    """
    json_files = sorted(dir_path.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"No JSON files found in {dir_path}")

    total_defects = np.zeros(n_bins, dtype=np.float64)
    total_pixels = np.zeros(n_bins, dtype=np.float64)
    grains_used = 0

    for jf in json_files:
        try:
            coco = load_coco_dict(jf)
        except Exception as e:
            print(f"[WARN] Skipping {jf.name}: {e}")
            continue

        img_by_id, anns_by_img = build_image_index(coco)

        for img_id, anns in anns_by_img.items():
            img_info = img_by_id.get(img_id, None)
            if img_info is not None:
                h = img_info.get("height")
                w = img_info.get("width")
            else:
                # fallback: try to infer from first ann's bbox or rle size
                h = w = None

            # union defect mask for this image
            defect_union = None
            for ann in anns:
                if ann.get("category_id") != defect_cat_id:
                    continue
                try:
                    m_def = decode_rle(ann["segmentation"], height=h, width=w)
                except Exception as e:
                    print(f"[WARN] Defect decode failed in {jf.name}: {e}")
                    continue
                if defect_union is None:
                    defect_union = (m_def > 0)
                else:
                    defect_union |= (m_def > 0)

            if defect_union is None:
                # no defects in this image; still possible to compute profile,
                # but it will be zero everywhere.
                pass

            # now loop over grains
            for ann in anns:
                if ann.get("category_id") != grain_cat_id:
                    continue
                try:
                    m_grain = decode_rle(ann["segmentation"], height=h, width=w)
                except Exception as e:
                    print(f"[WARN] Grain decode failed in {jf.name}: {e}")
                    continue

                # grain area
                area = (m_grain > 0).sum()
                if area < min_pixels:
                    continue

                if defect_union is None:
                    m_def_grain = np.zeros_like(m_grain, dtype=bool)
                else:
                    m_def_grain = defect_union & (m_grain > 0)

                _, def_counts, tot_counts = radial_profile_for_grain(
                    m_grain > 0,
                    m_def_grain,
                    n_bins=n_bins,
                )
                if def_counts is None:
                    continue

                total_defects += def_counts
                total_pixels += tot_counts
                grains_used += 1

    if grains_used == 0 or total_pixels.sum() == 0:
        raise RuntimeError(f"No valid grains found in {dir_path}")

    defect_prob = total_defects / np.maximum(total_pixels, 1e-9)

    # bin centers (normalised radius)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Save CSV
    out_csv = out_prefix.with_suffix(".csv")
    arr = np.column_stack([bin_centers, defect_prob])
    np.savetxt(
        out_csv,
        arr,
        delimiter=",",
        header="r_norm,defect_prob",
        comments="",
    )
    print(f"[OK] Saved radial profile CSV for {label}: {out_csv}")

    # Plot
    out_png = out_prefix.with_suffix(".png")
    plt.figure(figsize=(4.0, 3.0))
    plt.plot(bin_centers, defect_prob, marker="o", lw=1.8)
    plt.xlabel("Normalised radius r/R")
    plt.ylabel("Defect probability")
    plt.title(f"Intragrain radial defect profile ({label})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[OK] Saved radial profile PNG for {label}: {out_png}")

    return bin_centers, defect_prob


# ------------------------------ main ------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Intragrain radial defect profiles for FAPI vs FAPI–TEMPO"
    )
    parser.add_argument(
        "--fapi-dir", required=True,
        help="Directory containing FAPI *.json COCO files"
    )
    parser.add_argument(
        "--tempo-dir", required=True,
        help="Directory containing FAPI–TEMPO *.json COCO files"
    )
    parser.add_argument(
        "--out-prefix", required=True,
        help="Prefix for output files (folder + basename)"
    )
    parser.add_argument(
        "--grain-cat-id", type=int, default=1,
        help="COCO category_id for grains"
    )
    parser.add_argument(
        "--defect-cat-id", type=int, default=3,
        help="COCO category_id for defects"
    )
    parser.add_argument(
        "--n-bins", type=int, default=25,
        help="Number of radial bins"
    )
    parser.add_argument(
        "--min-pixels", type=int, default=200,
        help="Minimum pixel area per grain"
    )
    parser.add_argument(
        "--px-per-um", type=float, default=2.20014,
        help="Pixel-per-micron calibration (for axis annotation only)"
    )

    args = parser.parse_args()

    fapi_dir = Path(args.fapi_dir)
    tempo_dir = Path(args.tempo_dir)
    out_prefix = Path(args.out_prefix)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Processing FAPI dataset in {fapi_dir} ...")
    r_fapi, prob_fapi = process_dataset(
        fapi_dir,
        grain_cat_id=args.grain_cat_id,
        defect_cat_id=args.defect_cat_id,
        n_bins=args.n_bins,
        min_pixels=args.min_pixels,
        px_per_um=args.px_per_um,
        label="FAPI",
        out_prefix=out_prefix.with_name(out_prefix.name + "_FAPI"),
    )

    print(f"[INFO] Processing FAPI–TEMPO dataset in {tempo_dir} ...")
    r_tempo, prob_tempo = process_dataset(
        tempo_dir,
        grain_cat_id=args.grain_cat_id,
        defect_cat_id=args.defect_cat_id,
        n_bins=args.n_bins,
        min_pixels=args.min_pixels,
        px_per_um=args.px_per_um,
        label="FAPI–TEMPO",
        out_prefix=out_prefix.with_name(out_prefix.name + "_FAPITEMPO"),
    )

    # Joint comparison plot
    out_png = out_prefix.with_name(out_prefix.name + "_comparison.png").with_suffix(".png")
    plt.figure(figsize=(4.0, 3.0))
    plt.plot(r_fapi, prob_fapi, marker="o", lw=1.8, label="FAPI")
    plt.plot(r_tempo, prob_tempo, marker="s", lw=1.8, label="FAPI–TEMPO")
    plt.xlabel("Normalised radius r/R")
    plt.ylabel("Defect probability")
    plt.title("Intragrain radial defect profiles")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[OK] Saved FAPI vs FAPI–TEMPO comparison plot: {out_png}")


if __name__ == "__main__":
    main()
