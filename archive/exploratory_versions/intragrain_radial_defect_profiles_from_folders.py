#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intragrain_radial_defect_profiles_from_folders.py

Compute average intragrain radial defect-fraction profiles
for FAPI and FAPI–TEMPO directly from your JSON folders.

Assumptions for each JSON file:
- It is a list of objects, each with:
    segmentation: {size: [H, W], counts: RLE-string}
    bbox, score, mask_name, and sometimes category_id.
- Grain masks are those where category_id is missing or not 2/3.
- Nucleus masks: category_id == 2
- Defect masks: category_id == 3

For each grain:
- find its nucleus by maximum overlap of masks,
- compute distance from nucleus centroid to all pixels in the grain,
- normalise radius by the grain’s max radius,
- compute fraction of defect pixels in N radial bins.

Outputs:
- <out_prefix>_radial_defect_profiles.csv:
    columns: r_norm, phi_FAPI_mean, phi_FAPI_std,
             phi_FAPITEMPO_mean, phi_FAPITEMPO_std
- <out_prefix>_radial_defect_profiles.png:
    plot of defect fraction vs normalised radius (0–1) for both datasets.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pycocotools import mask as mask_utils


def decode_mask(segmentation):
    """Decode COCO RLE segmentation to a boolean mask."""
    # segmentation is expected to be dict(size=[H,W], counts=...)
    rle = {
        "size": segmentation["size"],
        "counts": segmentation["counts"].encode("utf-8")
    }
    m = mask_utils.decode(rle)  # shape (H, W, 1) or (H, W)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(bool)


def pair_grains_and_nuclei(grain_masks, nucleus_masks):
    """
    Pair each grain mask with the nucleus mask that has the
    largest overlap (in pixels). Returns list of (grain_mask, nucleus_mask)
    pairs; grains with zero overlap to all nuclei are skipped.
    """
    pairs = []
    if len(nucleus_masks) == 0:
        return pairs

    nuc_arrays = [nm for nm in nucleus_masks]

    for g in grain_masks:
        overlaps = [np.count_nonzero(g & nm) for nm in nuc_arrays]
        max_ov = max(overlaps)
        if max_ov <= 0:
            # no nucleus inside this grain
            continue
        idx = int(np.argmax(overlaps))
        pairs.append((g, nuc_arrays[idx]))
    return pairs


def radial_defect_profile_for_grain(grain_mask, nucleus_mask, defect_union,
                                    n_bins=20):
    """
    Compute radial defect-fraction profile for one grain.

    - grain_mask: boolean array (H, W)
    - nucleus_mask: boolean array (H, W)
    - defect_union: boolean array (H, W) with all defects in the image
    - n_bins: number of radial bins (from 0 to 1 in normalised radius)

    Returns:
    - profile: np.array of length n_bins (defect fraction per bin),
               np.nan where bin has no grain pixels.
    """
    # centroid of nucleus (true nucleation centre)
    ys, xs = np.nonzero(nucleus_mask)
    if len(xs) == 0:
        return np.full(n_bins, np.nan)
    cx = xs.mean()
    cy = ys.mean()

    # pixel coordinates grid
    H, W = grain_mask.shape
    yy, xx = np.indices((H, W))
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    # restrict to grain pixels
    grain_idx = np.nonzero(grain_mask)
    r_grain = r[grain_idx]
    if r_grain.size == 0:
        return np.full(n_bins, np.nan)

    # normalised radius in [0,1] (per grain)
    r_max = r_grain.max()
    if r_max <= 0:
        return np.full(n_bins, np.nan)
    r_norm = r_grain / r_max

    # defect pixels inside grain
    defect_grain = defect_union & grain_mask
    defect_flags = defect_grain[grain_idx]

    # radial binning
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    profile = np.full(n_bins, np.nan, dtype=float)

    for i in range(n_bins):
        mask_bin = (r_norm >= edges[i]) & (r_norm < edges[i + 1])
        n_pix = mask_bin.sum()
        if n_pix == 0:
            continue
        n_def = (defect_flags & mask_bin).sum()
        profile[i] = n_def / float(n_pix)

    return profile


def collect_profiles_from_folder(
    folder,
    n_bins=20,
    min_pixels=200,
):
    """
    Walk a folder of JSON files, build intragrain radial defect profiles
    for all grains with >= min_pixels.

    Returns:
    - profiles: list of 1D np.arrays (length n_bins) for each valid grain
    """
    folder = Path(folder)
    json_files = sorted(folder.glob("*.json"))
    profiles = []

    for jf in json_files:
        with open(jf, "r") as f:
            anns = pd.read_json(f)

        # ensure we have direct records, not a nested structure
        if isinstance(anns, pd.DataFrame):
            records = anns.to_dict(orient="records")
        else:
            records = anns

        # decode masks and split by category
        grain_masks = []
        nucleus_masks = []
        defect_masks = []
        H = W = None

        for obj in records:
            seg = obj.get("segmentation", None)
            if seg is None:
                continue
            m = decode_mask(seg)
            H, W = m.shape
            cat = obj.get("category_id", None)

            # category_id mapping:
            #   2 -> nucleus
            #   3 -> defect
            #   others / None -> grain
            if cat == 2:
                nucleus_masks.append(m)
            elif cat == 3:
                defect_masks.append(m)
            else:
                grain_masks.append(m)

        if H is None:
            continue  # no valid masks in this file

        if len(grain_masks) == 0 or len(nucleus_masks) == 0:
            continue

        # union of all defect masks in this image
        if len(defect_masks) > 0:
            defect_union = np.logical_or.reduce(defect_masks)
        else:
            defect_union = np.zeros((H, W), dtype=bool)

        # pair grains and nuclei by overlap
        pairs = pair_grains_and_nuclei(grain_masks, nucleus_masks)

        for g, n in pairs:
            if g.sum() < min_pixels:
                continue
            prof = radial_defect_profile_for_grain(
                grain_mask=g,
                nucleus_mask=n,
                defect_union=defect_union,
                n_bins=n_bins,
            )
            profiles.append(prof)

    return profiles


def avg_profile(profiles, n_bins):
    """
    Compute mean and std over a list of radial profiles (arrays of length n_bins).
    NaNs are ignored.
    """
    if len(profiles) == 0:
        return (np.full(n_bins, np.nan), np.full(n_bins, np.nan))

    arr = np.vstack(profiles)  # (n_grains, n_bins)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    return mean, std


def main():
    parser = argparse.ArgumentParser(
        description="Intragrain radial defect-fraction profiles from JSON folders."
    )
    parser.add_argument(
        "--fapi-dir",
        required=True,
        help="Folder with FAPI JSON files (e.g. ...\\comparative datasets\\FAPI)",
    )
    parser.add_argument(
        "--tempo-dir",
        required=True,
        help="Folder with FAPI–TEMPO JSON files (e.g. ...\\FAPI-TEMPO)",
    )
    parser.add_argument(
        "--out-prefix",
        required=True,
        help="Output prefix for CSV and PNG, e.g. ...\\intragrain_defect_profiles",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=20,
        help="Number of radial bins (default 20)",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=200,
        help="Minimum grain area in pixels to be included (default 200)",
    )

    args = parser.parse_args()

    n_bins = args.n_bins

    print(f"[INFO] Collecting profiles for FAPI from {args.fapi_dir} ...")
    profs_fapi = collect_profiles_from_folder(
        folder=args.fapi_dir,
        n_bins=n_bins,
        min_pixels=args.min_pixels,
    )
    print(f"[INFO] Got {len(profs_fapi)} grain profiles for FAPI.")

    print(f"[INFO] Collecting profiles for FAPI–TEMPO from {args.tempo_dir} ...")
    profs_tempo = collect_profiles_from_folder(
        folder=args.tempo_dir,
        n_bins=n_bins,
        min_pixels=args.min_pixels,
    )
    print(f"[INFO] Got {len(profs_tempo)} grain profiles for FAPI–TEMPO.")

    # average and std
    r_centers = 0.5 * (np.linspace(0.0, 1.0, n_bins, endpoint=False) +
                       np.linspace(0.0, 1.0, n_bins + 1)[1:])

    mean_f, std_f = avg_profile(profs_fapi, n_bins)
    mean_t, std_t = avg_profile(profs_tempo, n_bins)

    # save CSV
    out_prefix = Path(args.out_prefix)
    out_dir = out_prefix.parent if out_prefix.parent != Path("") else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_prefix.name

    df_out = pd.DataFrame(
        {
            "r_norm": r_centers,
            "phi_FAPI_mean": mean_f,
            "phi_FAPI_std": std_f,
            "phi_FAPITEMPO_mean": mean_t,
            "phi_FAPITEMPO_std": std_t,
        }
    )

    csv_path = out_dir / f"{base}_radial_defect_profiles.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"[OK] Saved radial defect profiles CSV: {csv_path}")

    # plot
    png_path = out_dir / f"{base}_radial_defect_profiles.png"
    fig, ax = plt.subplots(figsize=(5.0, 4.0))

    ax.plot(
        r_centers,
        mean_f,
        "-o",
        ms=3,
        lw=1.8,
        label="FAPI",
    )
    ax.fill_between(
        r_centers,
        mean_f - std_f,
        mean_f + std_f,
        alpha=0.2,
    )

    ax.plot(
        r_centers,
        mean_t,
        "-s",
        ms=3,
        lw=1.8,
        label="FAPI–TEMPO",
    )
    ax.fill_between(
        r_centers,
        mean_t - std_t,
        mean_t + std_t,
        alpha=0.2,
    )

    ax.set_xlabel("normalised radius $r/\\langle r_{\\max} \\rangle$")
    ax.set_ylabel("defect fraction")
    ax.set_title("Intragrain radial defect profiles")
    ax.set_xlim(0, 1.0)
    ax.legend()
    ax.grid(False)

    plt.tight_layout()
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved radial defect profiles plot: {png_path}")
    print("[DONE] All intragrain profiles computed.")


if __name__ == "__main__":
    main()
