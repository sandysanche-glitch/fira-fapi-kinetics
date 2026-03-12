#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intragrain_radial_from_images_v2.py

Compute *intragrain* radial profiles from your JSON + heatmap images:
  - For each grain mask, we compute radial defect fraction (defect pixels / grain pixels)
    as a function of normalised radius r/R_max.
  - Optionally, if a heatmap image "<basename>_heatmap.png" exists, we compute
    the radial mean heatmap intensity.

We then:
  - Average these profiles over all grains for each dataset (FAPI, FAPI–TEMPO).
  - Plot FAPI vs FAPI–TEMPO radial profiles.
  - Save CSV with the averaged profiles.

Assumptions for your dataset:
  - Each JSON file is a list of annotation dicts (NOT a full COCO dict).
  - category_id:
        1 -> grain
        2 -> nucleus (optional, not required)
        3 -> defects
  - Heatmap images live next to the JSON/optical image, with suffix "_heatmap.png".
  - Pixel-to-micrometre calibration: px_per_um = 2.20014
"""

import argparse
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

# if pycocotools is available, we can use it
try:
    from pycocotools import mask as maskUtils
    HAVE_COCO = True
except ImportError:
    HAVE_COCO = False
    print("[WARN] pycocotools not found; RLE decoding will fall back to a naive method "
          "(will only work if 'segmentation' is already a binary mask array or similar).")


# ----------------------------------------------------------------------
# Helper: decode RLE or binary segmentation
# ----------------------------------------------------------------------
def decode_segmentation(segmentation):
    """
    Decode a COCO-style segmentation.

    We expect your JSON 'segmentation' to be an RLE dict:
        {"size": [H, W], "counts": <bytes or str>}

    If pycocotools is available, we use maskUtils.decode.
    Otherwise, we assume 'segmentation' is already a binary mask-like array.

    Returns
    -------
    mask : np.ndarray of shape (H, W), dtype=bool
    """
    if HAVE_COCO and isinstance(segmentation, dict) and "counts" in segmentation:
        rle = segmentation
        if isinstance(rle["counts"], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects([rle], rle["size"][0], rle["size"][1])[0]
        m = maskUtils.decode(rle)
        # maskUtils.decode returns uint8 array with 0/1
        return m.astype(bool)
    else:
        # Fallback: if segmentation is already a 2D list/array
        seg = np.array(segmentation)
        if seg.ndim == 2:
            return seg.astype(bool)
        raise ValueError("Unsupported segmentation format without pycocotools.")


# ----------------------------------------------------------------------
# Per-grain radial profiles
# ----------------------------------------------------------------------
def radial_profile_for_grain(grain_mask, defect_mask, heatmap=None,
                             n_bins=25, min_pixels=200, px_per_um=2.20014):
    """
    Compute radial defect fraction and (optionally) heatmap mean intensity for ONE grain.

    Parameters
    ----------
    grain_mask : 2D bool array
    defect_mask : 2D bool array (same shape), union of all defect masks
    heatmap : 2D float/uint8 array or None
    n_bins : int
    min_pixels : int, minimum number of pixels to accept a grain
    px_per_um : float, pixels per micrometre

    Returns
    -------
    r_norm_centers : 1D array of bin centres, in [0,1]
    defect_frac : 1D array, defect fraction per radial bin
    heatmap_mean : 1D array or None, radial mean heatmap intensity
    r_um_centers : 1D array, radial bin centres in micrometres
    """
    # indices of grain pixels
    ys, xs = np.where(grain_mask)
    if len(xs) < min_pixels:
        return None

    # centroid as centre
    cx = xs.mean()
    cy = ys.mean()

    # radial distances in pixels
    rs_pix = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    r_max = rs_pix.max()
    if r_max <= 0:
        return None

    # bin edges in normalised radius
    r_norm = rs_pix / r_max
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idxs = np.digitize(r_norm, bin_edges) - 1  # 0..n_bins-1

    defect_frac = np.zeros(n_bins, dtype=float)
    heatmap_mean = np.zeros(n_bins, dtype=float) if heatmap is not None else None

    # pre-get defect flags for grain pixels
    defect_flags = defect_mask[ys, xs]

    # if heatmap is given, get intensities at grain pixels
    if heatmap is not None:
        hm_vals = heatmap[ys, xs].astype(float)

    for b in range(n_bins):
        mask_b = (bin_idxs == b)
        if not np.any(mask_b):
            # leave zeros; will handle later
            continue

        # defect fraction
        n_tot = mask_b.sum()
        n_def = defect_flags[mask_b].sum()
        defect_frac[b] = n_def / max(n_tot, 1)

        # heatmap mean
        if heatmap is not None:
            heatmap_mean[b] = hm_vals[mask_b].mean()

    # radial bin centres in normalised units
    r_norm_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # convert to micrometres (using the *absolute* radius scale)
    r_pix_centers = r_norm_centers * r_max
    r_um_centers = r_pix_centers / px_per_um

    return r_norm_centers, defect_frac, heatmap_mean, r_um_centers


# ----------------------------------------------------------------------
# Process one dataset folder (FAPI or FAPI–TEMPO)
# ----------------------------------------------------------------------
def process_dataset(dir_path, heatmap_suffix="_heatmap.png",
                    grain_cat_id=1, defect_cat_id=3,
                    n_bins=25, min_pixels=200, px_per_um=2.20014):
    """
    Loop over all JSON files in a folder, compute per-grain radial profiles, and
    return the *averaged* profiles over all grains.

    Returns
    -------
    r_norm : 1D array
    defect_frac_mean : 1D array
    defect_frac_std : 1D array
    hm_mean : 1D array or None
    hm_std : 1D array or None
    r_um : 1D array (same as r_norm, but in µm for the average r_max across grains)
    """
    dir_path = Path(dir_path)
    json_files = sorted(dir_path.glob("*.json"))

    all_defect_profiles = []
    all_heatmap_profiles = []
    all_r_norm = []
    all_r_um = []

    for jf in json_files:
        base = jf.stem  # e.g. FAPI_0
        heatmap_path = jf.with_name(base + heatmap_suffix)

        # Load annotations list
        with open(jf, "r") as f:
            anns = json.load(f)

        # build union defect mask & find grain masks
        grain_anns = []
        defect_anns = []

        for ann in anns:
            cid = ann.get("category_id", None)
            if cid == grain_cat_id:
                grain_anns.append(ann)
            elif cid == defect_cat_id:
                defect_anns.append(ann)

        if not grain_anns:
            continue

        # decode one grain to get image size
        first_seg = grain_anns[0]["segmentation"]
        first_mask = decode_segmentation(first_seg)
        H, W = first_mask.shape

        # union defect mask
        defect_mask = np.zeros((H, W), dtype=bool)
        for ann in defect_anns:
            dmask = decode_segmentation(ann["segmentation"])
            defect_mask |= dmask

        # optional heatmap
        heatmap = None
        if heatmap_path.is_file():
            hm = plt.imread(str(heatmap_path))
            # ensure 2D float
            if hm.ndim == 3:
                hm = hm[..., 0]
            heatmap = hm.astype(float)

        # per-grain profiles
        for ann in grain_anns:
            gmask = decode_segmentation(ann["segmentation"])
            res = radial_profile_for_grain(
                gmask,
                defect_mask,
                heatmap=heatmap,
                n_bins=n_bins,
                min_pixels=min_pixels,
                px_per_um=px_per_um,
            )
            if res is None:
                continue
            r_norm, dprof, hprof, r_um = res

            all_r_norm.append(r_norm)
            all_r_um.append(r_um)
            all_defect_profiles.append(dprof)
            if hprof is not None:
                all_heatmap_profiles.append(hprof)

    if not all_defect_profiles:
        raise RuntimeError(f"No valid grains found in {dir_path}")

    # Stack & average
    all_defect_profiles = np.vstack(all_defect_profiles)
    defect_mean = all_defect_profiles.mean(axis=0)
    defect_std = all_defect_profiles.std(axis=0)

    # For r_norm & r_um, we can just take the mean over grains
    all_r_norm = np.vstack(all_r_norm)
    r_norm_mean = all_r_norm.mean(axis=0)

    all_r_um = np.vstack(all_r_um)
    r_um_mean = all_r_um.mean(axis=0)

    hm_mean = hm_std = None
    if all_heatmap_profiles:
        all_heatmap_profiles = np.vstack(all_heatmap_profiles)
        hm_mean = all_heatmap_profiles.mean(axis=0)
        hm_std = all_heatmap_profiles.std(axis=0)

    return r_norm_mean, defect_mean, defect_std, hm_mean, hm_std, r_um_mean


# ----------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------
def plot_radial_defect(fapi_r_norm, fapi_d_mean, tempo_r_norm, tempo_d_mean,
                       out_png):
    plt.figure(figsize=(4.5, 4.0))
    plt.plot(fapi_r_norm, fapi_d_mean, "-o", ms=3, label="FAPI")
    plt.plot(tempo_r_norm, tempo_d_mean, "-o", ms=3, label="FAPI–TEMPO")
    plt.xlabel("normalised radius r/R$_\\mathrm{max}$")
    plt.ylabel("radial defect fraction")
    plt.title("Intragrain radial defect profile")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[OK] saved radial defect plot: {out_png}")


def plot_radial_heatmap(fapi_r_norm, fapi_hm, tempo_r_norm, tempo_hm, out_png):
    plt.figure(figsize=(4.5, 4.0))
    plt.plot(fapi_r_norm, fapi_hm, "-o", ms=3, label="FAPI")
    plt.plot(tempo_r_norm, tempo_hm, "-o", ms=3, label="FAPI–TEMPO")
    plt.xlabel("normalised radius r/R$_\\mathrm{max}$")
    plt.ylabel("mean heatmap intensity (a.u.)")
    plt.title("Intragrain radial heatmap profile")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[OK] saved radial heatmap plot: {out_png}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute intragrain radial defect/heatmap profiles from JSON + heatmap images."
    )
    parser.add_argument(
        "--fapi-dir",
        required=True,
        help="Folder with FAPI JSON/heatmap files",
    )
    parser.add_argument(
        "--tempo-dir",
        required=True,
        help="Folder with FAPI–TEMPO JSON/heatmap files",
    )
    parser.add_argument(
        "--out-prefix",
        required=True,
        help="Prefix for output files (PNG + CSV)",
    )
    parser.add_argument(
        "--heatmap-suffix",
        default="_heatmap.png",
        help="Suffix for heatmap images (default: '_heatmap.png')",
    )
    parser.add_argument(
        "--grain-cat-id",
        type=int,
        default=1,
        help="category_id for grains (default: 1)",
    )
    parser.add_argument(
        "--defect-cat-id",
        type=int,
        default=3,
        help="category_id for defects (default: 3)",
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
        help="Minimum grain area (pixels) to be included (default: 200)",
    )
    parser.add_argument(
        "--px-per-um",
        type=float,
        default=2.20014,
        help="Pixels per micrometre (default: 2.20014)",
    )

    args = parser.parse_args()

    out_prefix = Path(args.out_prefix)
    out_dir = out_prefix.parent if out_prefix.parent != Path("") else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    base = out_prefix.name

    # --- FAPI ---
    print(f"[INFO] Processing FAPI dataset in {args.fapi_dir} ...")
    (fapi_r_norm, fapi_def_mean, fapi_def_std,
     fapi_hm_mean, fapi_hm_std, fapi_r_um) = process_dataset(
        args.fapi_dir,
        heatmap_suffix=args.heatmap_suffix,
        grain_cat_id=args.grain_cat_id,
        defect_cat_id=args.defect_cat_id,
        n_bins=args.n_bins,
        min_pixels=args.min_pixels,
        px_per_um=args.px_per_um,
    )

    # --- FAPI–TEMPO ---
    print(f"[INFO] Processing FAPI–TEMPO dataset in {args.tempo_dir} ...")
    (tempo_r_norm, tempo_def_mean, tempo_def_std,
     tempo_hm_mean, tempo_hm_std, tempo_r_um) = process_dataset(
        args.tempo_dir,
        heatmap_suffix=args.heatmap_suffix,
        grain_cat_id=args.grain_cat_id,
        defect_cat_id=args.defect_cat_id,
        n_bins=args.n_bins,
        min_pixels=args.min_pixels,
        px_per_um=args.px_per_um,
    )

    # --- Save CSV with averaged profiles ---
    import pandas as pd

    df = pd.DataFrame({
        "r_norm": fapi_r_norm,
        "r_um_FAPI": fapi_r_um,
        "defect_frac_FAPI": fapi_def_mean,
        "defect_frac_FAPI_std": fapi_def_std,
        "defect_frac_FAPITEMPO": tempo_def_mean,
        "defect_frac_FAPITEMPO_std": tempo_def_std,
    })

    if fapi_hm_mean is not None and tempo_hm_mean is not None:
        df["heatmap_mean_FAPI"] = fapi_hm_mean
        df["heatmap_mean_FAPITEMPO"] = tempo_hm_mean

    csv_path = out_dir / f"{base}_intragrain_radial_profiles.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] saved radial profiles CSV: {csv_path}")

    # --- Plots ---
    plot_radial_defect(
        fapi_r_norm, fapi_def_mean,
        tempo_r_norm, tempo_def_mean,
        out_dir / f"{base}_radial_defect.png",
    )

    if fapi_hm_mean is not None and tempo_hm_mean is not None:
        plot_radial_heatmap(
            fapi_r_norm, fapi_hm_mean,
            tempo_r_norm, tempo_hm_mean,
            out_dir / f"{base}_radial_heatmap.png",
        )

    print("[DONE] Intragrain radial analysis complete.")


if __name__ == "__main__":
    main()
