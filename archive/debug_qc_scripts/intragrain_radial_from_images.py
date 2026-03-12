#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intragrain_radial_from_images.py

Compute average intragrain radial profiles for FAPI and FAPI–TEMPO
directly from per-image JSON annotation files + heatmap images.

Assumes:
  - Each dataset directory contains many JSON files, e.g.
      FAPI_0.json, FAPI_1.json, ...
  - For each JSON 'FAPI_0.json' there is:
      FAPI_0.jpg             (optional, not used here)
      FAPI_0_heatmap.png     (or other suffix via --heatmap-suffix)
  - Each JSON is EITHER:
      * a list of annotation dicts, OR
      * a dict with key "annotations" holding that list.
  - Each annotation has at least:
      "category_id"
      "segmentation"  (COCO RLE dict: {"size": [H,W], "counts": ...})

Category IDs (can be changed via CLI):
  grain_cat_id   (default: 1)
  nucleus_cat_id (default: 2)
  defect_cat_id  (default: 3)

Outputs:
  - <out-prefix>_radial_profiles.csv
  - <out-prefix>_radial_profiles.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from pycocotools import mask as mask_utils


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_per_image_annotations(json_path):
    """
    Load a single-image annotation file.

    Supports two cases:
      1) Top-level dict with an 'annotations' list
      2) Top-level list of annotation dicts

    Returns:
        anns : list of annotation dicts
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "annotations" in data:
            return data["annotations"]
        # If needed, you can add other dict-based layouts here.

    if isinstance(data, list):
        return data

    raise ValueError(f"Unrecognised JSON structure in {json_path}")


def decode_rle_to_mask(segmentation, height=None, width=None):
    """
    Decode a COCO RLE segmentation into a boolean mask.

    segmentation : dict with keys 'size' and 'counts', or
                   already in RLE form accepted by pycocotools.
    Returns:
        mask : (H, W) boolean array
    """
    # If segmentation is already in "RLE dict" with 'size' & 'counts'
    if isinstance(segmentation, dict) and "counts" in segmentation:
        rle = segmentation
        m = mask_utils.decode(rle)
        # pycocotools returns uint8, shape (H, W) or (H,W,1)
        if m.ndim == 3:
            m = m[:, :, 0]
        return m.astype(bool)

    # If it's a list (e.g. polygon format), you would need extra logic.
    # For this project we assume RLE; if not, raise for now:
    raise ValueError("Unexpected segmentation format (non-RLE). "
                     "Polygon decoding not implemented in this script.")


def compute_radial_profile_for_grain(grain_mask,
                                     defect_mask,
                                     heatmap,
                                     n_bins=25,
                                     center=None,
                                     min_pixels=200):
    """
    Compute radial profiles for a single grain:

      - defect_fraction(r): fraction of pixels that are defect inside the grain
      - heat_intensity(r): mean heatmap intensity inside the grain

    r is expressed in normalised units [0, 1], where 1 corresponds to
    the maximum radial extent of the grain.

    Returns:
      r_centers : (n_bins,) array in [0,1]
      defect_profile : (n_bins,) array, NaN where empty bins
      heat_profile   : (n_bins,) array, NaN where empty bins
    """
    if grain_mask.sum() < min_pixels:
        return None

    H, W = grain_mask.shape

    # Determine center: either provided, or use grain centroid
    if center is None:
        ys, xs = np.nonzero(grain_mask)
        cy = ys.mean()
        cx = xs.mean()
    else:
        cy, cx = center  # (row, col)

    # Build coordinate grids
    yy, xx = np.indices((H, W))
    rr = np.sqrt((xx - cx)**2 + (yy - cy)**2)

    # Radial extent only within the grain
    r_in_grain = rr[grain_mask]
    if r_in_grain.size == 0:
        return None

    r_max = r_in_grain.max()
    if r_max <= 0:
        return None

    r_norm = rr / r_max  # normalised radius

    # Bin edges in [0,1]
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    r_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Arrays for profiles
    defect_prof = np.full(n_bins, np.nan, dtype=float)
    heat_prof = np.full(n_bins, np.nan, dtype=float)

    # Masks for grain, defect, etc.
    grain_idx = grain_mask
    defect_idx = defect_mask & grain_mask

    # For each bin, collect pixels of that shell inside the grain
    r_norm_flat = r_norm[grain_idx]
    defect_flat = defect_idx[grain_idx]
    heat_flat = heatmap[grain_idx].astype(float)

    # Digitize normalised radius in the grain
    bin_ids = np.digitize(r_norm_flat, bin_edges) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    for k in range(n_bins):
        in_bin = bin_ids == k
        if not np.any(in_bin):
            continue

        # Shell area (grain pixels in this radial bin)
        n_shell = in_bin.sum()
        # Defect pixels in that shell
        n_def = defect_flat[in_bin].sum()
        defect_prof[k] = n_def / n_shell

        # Mean heatmap intensity
        heat_prof[k] = heat_flat[in_bin].mean()

    return r_centers, defect_prof, heat_prof


def process_dataset(dataset_dir,
                    heatmap_suffix="_heatmap.png",
                    n_bins=25,
                    min_pixels=200,
                    grain_cat_id=1,
                    nucleus_cat_id=2,
                    defect_cat_id=3):
    """
    Process all JSON files in 'dataset_dir' and compute average
    intragrain radial profiles for defect fraction and heatmap intensity.

    Returns:
      r_centers
      mean_defect_profile
      std_defect_profile
      mean_heat_profile
      std_heat_profile
    """
    dataset_dir = Path(dataset_dir)
    json_files = sorted(dataset_dir.glob("*.json"))

    all_defect_profiles = []
    all_heat_profiles = []
    r_centers_ref = None

    for jf in json_files:
        base = jf.stem  # e.g. "FAPI_0"
        heat_path = dataset_dir / f"{base}{heatmap_suffix}"

        heat = plt.imread(str(heat_path)) if heat_path.exists() else None
        if heat is None:
            print(f"[WARN] heatmap not found for {jf.name}, skipping")
            continue

        # If heatmap is RGB, convert to grayscale
        if heat.ndim == 3:
            # Simple luminance or mean
            heat = heat.mean(axis=2)

        anns = load_per_image_annotations(str(jf))

        # Build masks
        H, W = heat.shape
        grain_anns = [a for a in anns if a.get("category_id") == grain_cat_id]
        nucleus_anns = [a for a in anns if a.get("category_id") == nucleus_cat_id]
        defect_anns = [a for a in anns if a.get("category_id") == defect_cat_id]

        if not grain_anns:
            continue

        # Union defect mask for this image
        defect_mask = np.zeros((H, W), dtype=bool)
        for da in defect_anns:
            seg = da.get("segmentation", None)
            if seg is None:
                continue
            try:
                m_def = decode_rle_to_mask(seg)
            except Exception as e:
                print(f"[WARN] could not decode defect in {jf.name}: {e}")
                continue
            if m_def.shape != defect_mask.shape:
                # if mismatch, try to resize or skip; we skip for robustness
                print(f"[WARN] defect mask size mismatch in {jf.name}, skipping defect")
                continue
            defect_mask |= m_def.astype(bool)

        # Optional: build a nucleus mask dict by overlap if needed
        # For now we only use grain centroid; you can refine later using nucleus_anns.

        for ga in grain_anns:
            seg = ga.get("segmentation", None)
            if seg is None:
                continue
            try:
                grain_mask = decode_rle_to_mask(seg)
            except Exception as e:
                print(f"[WARN] could not decode grain in {jf.name}: {e}")
                continue

            if grain_mask.shape != defect_mask.shape:
                print(f"[WARN] grain mask size mismatch in {jf.name}, skipping grain")
                continue

            prof = compute_radial_profile_for_grain(
                grain_mask=grain_mask,
                defect_mask=defect_mask,
                heatmap=heat,
                n_bins=n_bins,
                center=None,
                min_pixels=min_pixels,
            )
            if prof is None:
                continue

            r_centers, d_prof, h_prof = prof

            if r_centers_ref is None:
                r_centers_ref = r_centers
            else:
                # sanity check: same binning
                if len(r_centers_ref) != len(r_centers):
                    print("[WARN] inconsistent radial binning; skipping this grain.")
                    continue

            all_defect_profiles.append(d_prof)
            all_heat_profiles.append(h_prof)

    if not all_defect_profiles:
        raise RuntimeError(f"No valid grains found in {dataset_dir}")

    all_defect_profiles = np.vstack(all_defect_profiles)  # (N_grains, n_bins)
    all_heat_profiles = np.vstack(all_heat_profiles)

    mean_def = np.nanmean(all_defect_profiles, axis=0)
    std_def = np.nanstd(all_defect_profiles, axis=0)

    mean_heat = np.nanmean(all_heat_profiles, axis=0)
    std_heat = np.nanstd(all_heat_profiles, axis=0)

    return r_centers_ref, mean_def, std_def, mean_heat, std_heat


def plot_radial_profiles(r,
                         mean_def_fapi,
                         std_def_fapi,
                         mean_def_tempo,
                         std_def_tempo,
                         mean_heat_fapi,
                         std_heat_fapi,
                         mean_heat_tempo,
                         std_heat_tempo,
                         out_path):
    """
    Make a figure with two panels:

      (a) radial defect fraction vs normalised radius (FAPI vs FAPI–TEMPO)
      (b) radial mean heatmap intensity vs normalised radius
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))

    ax1, ax2 = axes

    # Panel (a): defect fraction
    ax1.plot(r, mean_def_fapi, color="C0", lw=2, label="FAPI")
    ax1.fill_between(r,
                     mean_def_fapi - std_def_fapi,
                     mean_def_fapi + std_def_fapi,
                     color="C0",
                     alpha=0.2)

    ax1.plot(r, mean_def_tempo, color="C2", lw=2, label="FAPI–TEMPO")
    ax1.fill_between(r,
                     mean_def_tempo - std_def_tempo,
                     mean_def_tempo + std_def_tempo,
                     color="C2",
                     alpha=0.2)

    ax1.set_xlabel("normalised radius r/R$_\\mathrm{max}$")
    ax1.set_ylabel("defect fraction")
    ax1.set_title("Radial defect profile")
    ax1.set_xlim(0, 1)
    ax1.legend(frameon=False)
    ax1.grid(False)

    # Panel (b): heatmap intensity
    ax2.plot(r, mean_heat_fapi, color="C0", lw=2, label="FAPI")
    ax2.fill_between(r,
                     mean_heat_fapi - std_heat_fapi,
                     mean_heat_fapi + std_heat_fapi,
                     color="C0",
                     alpha=0.2)

    ax2.plot(r, mean_heat_tempo, color="C2", lw=2, label="FAPI–TEMPO")
    ax2.fill_between(r,
                     mean_heat_tempo - std_heat_tempo,
                     mean_heat_tempo + std_heat_tempo,
                     color="C2",
                     alpha=0.2)

    ax2.set_xlabel("normalised radius r/R$_\\mathrm{max}$")
    ax2.set_ylabel("mean heatmap intensity (a.u.)")
    ax2.set_title("Radial heat profile")
    ax2.set_xlim(0, 1)
    ax2.legend(frameon=False)
    ax2.grid(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] saved radial profile figure: {out_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute intragrain radial defect / heat profiles for "
                    "FAPI vs FAPI–TEMPO from per-image JSON + heatmaps."
    )
    parser.add_argument(
        "--fapi-dir",
        required=True,
        help="Folder with FAPI JSONs + heatmaps"
    )
    parser.add_argument(
        "--tempo-dir",
        required=True,
        help="Folder with FAPI–TEMPO JSONs + heatmaps"
    )
    parser.add_argument(
        "--out-prefix",
        required=True,
        help="Prefix for output files (CSV + PNG)"
    )
    parser.add_argument(
        "--heatmap-suffix",
        default="_heatmap.png",
        help="Suffix for heatmap images (default: _heatmap.png)"
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=25,
        help="Number of radial bins (default: 25)"
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=200,
        help="Minimum grain area in pixels (default: 200)"
    )
    parser.add_argument(
        "--grain-cat-id",
        type=int,
        default=1,
        help="category_id for grain masks (default: 1)"
    )
    parser.add_argument(
        "--nuc-cat-id",
        type=int,
        default=2,
        help="category_id for nucleus masks (default: 2; currently unused)"
    )
    parser.add_argument(
        "--defect-cat-id",
        type=int,
        default=3,
        help="category_id for defect masks (default: 3)"
    )

    args = parser.parse_args()

    fapi_dir = args.fapi_dir
    tempo_dir = args.tempo_dir
    out_prefix = Path(args.out_prefix)
    out_dir = out_prefix.parent if out_prefix.parent != Path("") else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_prefix.name

    print(f"[INFO] Processing FAPI dataset in {fapi_dir} ...")
    r_fapi, mdef_fapi, sdef_fapi, mhe_fapi, she_fapi = process_dataset(
        dataset_dir=fapi_dir,
        heatmap_suffix=args.heatmap_suffix,
        n_bins=args.n_bins,
        min_pixels=args.min_pixels,
        grain_cat_id=args.grain_cat_id,
        nucleus_cat_id=args.nuc_cat_id,
        defect_cat_id=args.defect_cat_id,
    )

    print(f"[INFO] Processing FAPI–TEMPO dataset in {tempo_dir} ...")
    r_tempo, mdef_tempo, sdef_tempo, mhe_tempo, she_tempo = process_dataset(
        dataset_dir=tempo_dir,
        heatmap_suffix=args.heatmap_suffix,
        n_bins=args.n_bins,
        min_pixels=args.min_pixels,
        grain_cat_id=args.grain_cat_id,
        nucleus_cat_id=args.nuc_cat_id,
        defect_cat_id=args.defect_cat_id,
    )

    # Sanity check on r
    if not np.allclose(r_fapi, r_tempo):
        print("[WARN] radial grids differ slightly between datasets; "
              "using FAPI's r as reference.")
    r = r_fapi

    # Save CSV
    import pandas as pd
    csv_path = out_dir / f"{base}_radial_profiles.csv"
    df = pd.DataFrame({
        "r_norm": r,
        "defect_FAPI_mean": mdef_fapi,
        "defect_FAPI_std": sdef_fapi,
        "defect_FAPITEMPO_mean": mdef_tempo,
        "defect_FAPITEMPO_std": sdef_tempo,
        "heat_FAPI_mean": mhe_fapi,
        "heat_FAPI_std": she_fapi,
        "heat_FAPITEMPO_mean": mhe_tempo,
        "heat_FAPITEMPO_std": she_tempo,
    })
    df.to_csv(csv_path, index=False)
    print(f"[OK] saved radial profile CSV: {csv_path}")

    # Save figure
    fig_path = out_dir / f"{base}_radial_profiles.png"
    plot_radial_profiles(
        r=r,
        mean_def_fapi=mdef_fapi,
        std_def_fapi=sdef_fapi,
        mean_def_tempo=mdef_tempo,
        std_def_tempo=sdef_tempo,
        mean_heat_fapi=mhe_fapi,
        std_heat_fapi=she_fapi,
        mean_heat_tempo=mhe_tempo,
        std_heat_tempo=she_tempo,
        out_path=fig_path,
    )

    print("[DONE] All intragrain radial profiles computed.")


if __name__ == "__main__":
    main()
