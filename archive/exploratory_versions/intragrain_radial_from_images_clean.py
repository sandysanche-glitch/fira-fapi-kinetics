#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intragrain_radial_from_images_clean.py

Compute intragrain radial defect profiles for two datasets (FAPI, FAPI–TEMPO)
using JSON instance annotations + corresponding heatmap images.

Assumptions:
- Each dataset directory contains:
    FAPI_0.json, FAPI_0.jpg, FAPI_0_heatmap.png, ...
    FAPI_TEMPO_0.json, FAPI_TEMPO_0.jpg, FAPI_TEMPO_0_heatmap.png, ...
- JSON files are either:
    (a) standard COCO dict: {"images": [...], "annotations": [...], "categories": [...]}
    (b) Detectron2-style list of dicts:
        [
          {
            "file_name": "...",
            "height": H,
            "width": W,
            "annotations": [ {category_id, segmentation, ...}, ... ]
          },
          ...
        ]
- Category IDs:
    grain_cat_id  = grains (spherulites)
    nuc_cat_id    = nuclei   (optional; used to locate centre if present)
    defect_cat_id = defects

We compute, for each dataset:
  - Radial bins (0..1, normalised radius)
  - For each bin: fraction of pixels that are labeled as defect inside the grain

Outputs:
  - <out_prefix>_FAPI_radial_defect_profile.png
  - <out_prefix>_TEMPO_radial_defect_profile.png
  - <out_prefix>_radial_defect_profile_both.png
"""

import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

try:
    from pycocotools import mask as mask_utils
except ImportError as e:
    raise ImportError(
        "pycocotools is required. On Windows, try:\n"
        "  pip install pycocotools-windows\n"
        "or:\n"
        "  pip install pycocotools"
    ) from e


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_coco_any(json_path: str):
    """
    Load either a standard COCO dict or a Detectron2-style list of dicts,
    and return a canonical COCO-like dict:
        {"images": [...], "annotations": [...], "categories": [...]}
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Case 1: full COCO
    if isinstance(data, dict) and "annotations" in data:
        # If "images" is missing, we can't map image_id -> filename reliably,
        # but for intragrain profiles we only need masks, not filenames.
        # Still, try to keep "images" if present.
        images = data.get("images", [])
        anns = data["annotations"]
        cats = data.get("categories", [])
        return {"images": images, "annotations": anns, "categories": cats}

    # Case 2: Detectron2-style list of per-image dicts
    if isinstance(data, list):
        images = []
        anns = []
        cat_ids = set()
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            file_name = item.get("file_name", f"img_{idx}.png")
            h = item.get("height", None)
            w = item.get("width", None)
            image_id = idx + 1  # make 1-based ID

            images.append(
                {
                    "id": image_id,
                    "file_name": file_name,
                    "height": h,
                    "width": w,
                }
            )
            for ann in item.get("annotations", []):
                ann = dict(ann)
                ann["image_id"] = image_id
                anns.append(ann)
                cid = ann.get("category_id", None)
                if cid is not None:
                    cat_ids.add(cid)

        cats = [{"id": cid, "name": f"cat_{cid}"} for cid in sorted(cat_ids)]
        return {"images": images, "annotations": anns, "categories": cats}

    raise ValueError(f"{json_path} is neither a COCO dict nor a Detectron2-style list.")


def decode_segmentation(ann, height: int, width: int) -> np.ndarray:
    """
    Decode COCO-style segmentation into a boolean mask.

    Supports:
      - RLE dict: {"counts": "...", "size": [h, w]}
      - Polygon list: [ [x1,y1, x2,y2, ...], ... ]
    """
    seg = ann.get("segmentation", None)
    if seg is None:
        raise ValueError("Annotation has no 'segmentation' field.")

    if isinstance(seg, dict):
        # COCO RLE dict
        rle = seg
        if "size" not in rle:
            rle["size"] = [height, width]
        mask = mask_utils.decode(rle)
        return mask.astype(bool)

    if isinstance(seg, list):
        # Polygon(s)
        rles = mask_utils.frPyObjects(seg, height, width)
        rle = mask_utils.merge(rles)
        mask = mask_utils.decode(rle)
        return mask.astype(bool)

    raise ValueError("Unsupported segmentation format.")


def compute_centroid(mask: np.ndarray):
    """
    Simple centroid of a binary mask (y, x).
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(ys.mean()), float(xs.mean())


def normalised_radial_bins(mask: np.ndarray, centre_y: float, centre_x: float, n_bins: int):
    """
    For all True pixels in mask, compute their normalised radius r/R_max in [0,1],
    then return bin indices and bin edges.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None, None, None

    dy = ys - centre_y
    dx = xs - centre_x
    r = np.sqrt(dx*dx + dy*dy)
    r_max = r.max()
    if r_max <= 0:
        return None, None, None

    r_norm = r / r_max
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.digitize(r_norm, edges) - 1
    bins[bins < 0] = 0
    bins[bins >= n_bins] = n_bins - 1
    return bins, edges, len(xs)


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_dataset(
    dir_path: str,
    grain_cat_id: int,
    nuc_cat_id: int,
    defect_cat_id: int,
    n_bins: int,
    min_pixels: int,
    px_per_um: float,
):
    """
    Process all JSON files in a directory and compute a global
    radial defect profile (normalised radius 0–1) for that dataset.
    """
    dir_path = Path(dir_path)
    json_files = sorted(dir_path.glob("*.json"))

    total_counts = np.zeros(n_bins, dtype=float)
    defect_counts = np.zeros(n_bins, dtype=float)

    grains_used = 0

    for jf in json_files:
        try:
            coco = load_coco_any(str(jf))
        except Exception as e:
            print(f"[WARN] Skipping {jf.name}: cannot parse JSON ({e})")
            continue

        anns = coco["annotations"]
        # Build image size lookup (may be None but we try)
        img_info = {}
        for img in coco.get("images", []):
            img_info[img["id"]] = (img.get("height", None), img.get("width", None))

        # Group annotations by image
        anns_by_img = defaultdict(list)
        for ann in anns:
            img_id = ann.get("image_id", 1)
            anns_by_img[img_id].append(ann)

        # For each image
        for img_id, ann_list in anns_by_img.items():
            height, width = img_info.get(img_id, (None, None))

            # Split by category
            grain_anns = [a for a in ann_list if a.get("category_id") == grain_cat_id]
            nuc_anns   = [a for a in ann_list if a.get("category_id") == nuc_cat_id]
            defect_anns= [a for a in ann_list if a.get("category_id") == defect_cat_id]

            # Pre-decode nucleus & defect masks to reuse
            nuc_masks = []
            for a in nuc_anns:
                try:
                    nm = decode_segmentation(a, height, width)
                    nuc_masks.append(nm)
                except Exception as e:
                    print(f"[WARN] nucleus decode failed in {jf.name}: {e}")

            defect_masks = []
            for a in defect_anns:
                try:
                    dm = decode_segmentation(a, height, width)
                    defect_masks.append(dm)
                except Exception as e:
                    print(f"[WARN] defect decode failed in {jf.name}: {e}")

            # Now process each grain
            for g_ann in grain_anns:
                try:
                    gm = decode_segmentation(g_ann, height, width)
                except Exception as e:
                    print(f"[WARN] grain decode failed in {jf.name}: {e}")
                    continue

                # Min size filter
                if gm.sum() < min_pixels:
                    continue

                # Find nucleus with max overlap, if any
                best_nuc = None
                best_overlap = 0
                for nm in nuc_masks:
                    ov = np.logical_and(gm, nm).sum()
                    if ov > best_overlap:
                        best_overlap = ov
                        best_nuc = nm

                if best_nuc is not None and best_overlap > 0:
                    cy, cx = compute_centroid(best_nuc)
                else:
                    # Fallback: grain centroid
                    cent = compute_centroid(gm)
                    if cent is None:
                        continue
                    cy, cx = cent

                # Union of defect masks intersecting this grain
                if defect_masks:
                    d_union = np.zeros_like(gm, dtype=bool)
                    for dm in defect_masks:
                        if np.logical_and(gm, dm).sum() > 0:
                            d_union |= dm
                    d_in_grain = np.logical_and(gm, d_union)
                else:
                    d_in_grain = np.zeros_like(gm, dtype=bool)

                # Radial bins for all grain pixels
                bins_all, edges, n_pix = normalised_radial_bins(gm, cy, cx, n_bins)
                if bins_all is None:
                    continue

                # Radial bins for defect pixels
                ys_d, xs_d = np.nonzero(d_in_grain)
                if len(xs_d) > 0:
                    dy_d = ys_d - cy
                    dx_d = xs_d - cx
                    r_d = np.sqrt(dy_d*dy_d + dx_d*dx_d)
                    r_max = np.sqrt(((np.nonzero(gm)[1] - cx)**2 + (np.nonzero(gm)[0] - cy)**2)).max()
                    r_norm_d = r_d / r_max
                    bins_def = np.digitize(r_norm_d, edges) - 1
                    bins_def[bins_def < 0] = 0
                    bins_def[bins_def >= n_bins] = n_bins - 1
                else:
                    bins_def = np.array([], dtype=int)

                # Accumulate
                for b in bins_all:
                    total_counts[b] += 1.0
                for b in bins_def:
                    defect_counts[b] += 1.0

                grains_used += 1

    if grains_used == 0:
        raise RuntimeError(f"No valid grains found in {dir_path}")

    # Compute radial coordinate (bin centers 0..1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    frac = np.zeros_like(r_centers)
    nonzero = total_counts > 0
    frac[nonzero] = defect_counts[nonzero] / total_counts[nonzero]

    print(f"[INFO] {dir_path}: used {grains_used} grains.")
    return r_centers, frac


def plot_profiles(r_fapi, f_fapi, r_tempo, f_tempo, out_prefix: Path):
    out_prefix = Path(out_prefix)

    # Separate plots
    for label, r, f in [
        ("FAPI", r_fapi, f_fapi),
        ("FAPI–TEMPO", r_tempo, f_tempo),
    ]:
        fig, ax = plt.subplots(figsize=(4.0, 3.2))
        ax.plot(r, f, "-o", lw=1.8, ms=4)
        ax.set_xlabel("normalised radius $r / R_{\\max}$")
        ax.set_ylabel("defect fraction (probability)")
        ax.set_title(f"Radial defect profile ({label})")
        ax.set_ylim(0, max(0.05, 1.05 * f.max()))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_prefix.parent / f"{out_prefix.name}_{label}_radial_defect_profile.png", dpi=300)
        plt.close(fig)

    # Combined
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(r_fapi, f_fapi, "-o", lw=1.8, ms=4, label="FAPI")
    ax.plot(r_tempo, f_tempo, "-s", lw=1.8, ms=4, label="FAPI–TEMPO")
    ax.set_xlabel("normalised radius $r / R_{\\max}$")
    ax.set_ylabel("defect fraction (probability)")
    ax.set_title("Average intragrain radial defect profile")
    ax.set_ylim(0, max(0.05, 1.05 * max(f_fapi.max(), f_tempo.max())))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_prefix.parent / f"{out_prefix.name}_radial_defect_profile_both.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute intragrain radial defect profiles from JSON + heatmap images."
    )
    parser.add_argument(
        "--fapi-dir",
        required=True,
        help="Directory with FAPI JSON + images + heatmaps",
    )
    parser.add_argument(
        "--tempo-dir",
        required=True,
        help="Directory with FAPI–TEMPO JSON + images + heatmaps",
    )
    parser.add_argument(
        "--out-prefix",
        required=True,
        help="Output prefix (directory + base name) for PNG files",
    )
    parser.add_argument(
        "--grain-cat-id",
        type=int,
        default=1,
        help="Category ID for grains",
    )
    parser.add_argument(
        "--nuc-cat-id",
        type=int,
        default=2,
        help="Category ID for nuclei (currently only used to find centre if overlap)",
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
        help="Number of radial bins between 0 and 1",
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
        help="Pixel-per-micron calibration (used only for physical interpretation)",
    )

    args = parser.parse_args()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Processing FAPI dataset in {args.fapi_dir} ...")
    r_fapi, f_fapi = process_dataset(
        dir_path=args.fapi_dir,
        grain_cat_id=args.grain_cat_id,
        nuc_cat_id=args.nuc_cat_id,
        defect_cat_id=args.defect_cat_id,
        n_bins=args.n_bins,
        min_pixels=args.min_pixels,
        px_per_um=args.px_per_um,
    )

    print(f"[INFO] Processing FAPI–TEMPO dataset in {args.tempo_dir} ...")
    r_tempo, f_tempo = process_dataset(
        dir_path=args.tempo_dir,
        grain_cat_id=args.grain_cat_id,
        nuc_cat_id=args.nuc_cat_id,
        defect_cat_id=args.defect_cat_id,
        n_bins=args.n_bins,
        min_pixels=args.min_pixels,
        px_per_um=args.px_per_um,
    )

    plot_profiles(r_fapi, f_fapi, r_tempo, f_tempo, out_prefix=out_prefix)
    print("[DONE] Intragrain radial defect profiles saved.")


if __name__ == "__main__":
    main()
