#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intragrain_radial_profiles_multi_mask.py

Generalisation of intragrain_radial_defect_profiles_from_folders.py:

- Uses JSON folders for FAPI and FAPI–TEMPO.
- Grains: category_id is None or not in {2, 3, ...extra_mask_cats}
- Nuclei: category_id == 2
- Defects: category_id == 3
- Optionally: more mask-based descriptors via --extra-mask-cat, e.g.
    --extra-mask-cat 4 microcracks
    --extra-mask-cat 5 high_entropy

For each grain:
- pair with nucleus (max overlap),
- compute radial profiles of:
    * defect fraction (default, cat 3)
    * each extra mask category (fraction of pixels per radial bin)

Outputs:
- <out_prefix>_radial_profiles.csv:
    columns: r_norm,
             phi_defect_FAPI_mean, phi_defect_FAPI_std,
             [phi_cat<k>_FAPI_mean, phi_cat<k>_FAPI_std, ...],
             phi_defect_FAPITEMPO_mean, phi_defect_FAPITEMPO_std,
             [phi_cat<k>_FAPITEMPO_mean, phi_cat<k>_FAPITEMPO_std, ...]
- <out_prefix>_radial_profiles_defect.png:
    plot of defect fraction vs normalised radius for both datasets.

NOTE: true entropy / intensity profiles require per-pixel maps,
not just JSON; this script handles only mask-based descriptors.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pycocotools import mask as mask_utils


def decode_mask(segmentation):
    """Decode COCO RLE segmentation to a boolean mask."""
    rle = {
        "size": segmentation["size"],
        "counts": segmentation["counts"].encode("utf-8"),
    }
    m = mask_utils.decode(rle)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(bool)


def pair_grains_and_nuclei(grain_masks, nucleus_masks):
    """
    Pair each grain mask with the nucleus mask that has the
    largest overlap (in pixels). Returns list of (grain_mask, nucleus_mask).
    Grains with zero overlap to all nuclei are skipped.
    """
    pairs = []
    if len(nucleus_masks) == 0:
        return pairs

    nuc_arrays = [nm for nm in nucleus_masks]

    for g in grain_masks:
        overlaps = [np.count_nonzero(g & nm) for nm in nuc_arrays]
        max_ov = max(overlaps)
        if max_ov <= 0:
            continue
        idx = int(np.argmax(overlaps))
        pairs.append((g, nuc_arrays[idx]))
    return pairs


def radial_fraction_for_grain(grain_mask, nucleus_mask, mask_union, n_bins=20):
    """
    Generic radial fraction profile for one grain w.r.t. a given union mask.

    - grain_mask: boolean array (H, W)
    - nucleus_mask: boolean array (H, W)
    - mask_union: boolean array (H, W) of the descriptor (defects, etc.)
    - n_bins: number of radial bins

    Returns:
    - profile: array length n_bins with fraction of "mask_union" pixels
               in each radial bin (NaN if no grain pixels in bin).
    """
    ys, xs = np.nonzero(nucleus_mask)
    if len(xs) == 0:
        return np.full(n_bins, np.nan)
    cx = xs.mean()
    cy = ys.mean()

    H, W = grain_mask.shape
    yy, xx = np.indices((H, W))
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    # restrict to grain pixels
    gy, gx = np.nonzero(grain_mask)
    r_grain = r[gy, gx]
    if r_grain.size == 0:
        return np.full(n_bins, np.nan)

    r_max = r_grain.max()
    if r_max <= 0:
        return np.full(n_bins, np.nan)
    r_norm = r_grain / r_max

    # descriptor mask inside grain
    desc_inside = mask_union[gy, gx]

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    prof = np.full(n_bins, np.nan, dtype=float)

    for i in range(n_bins):
        sel = (r_norm >= edges[i]) & (r_norm < edges[i + 1])
        n_pix = sel.sum()
        if n_pix == 0:
            continue
        n_hit = (desc_inside & sel).sum()
        prof[i] = n_hit / float(n_pix)

    return prof


def collect_profiles_from_folder(folder, n_bins=20, min_pixels=200,
                                 extra_mask_cats=None):
    """
    Walk a folder of JSON files and build intragrain radial profiles
    for:
      - defects (category_id == 3)
      - each extra category in extra_mask_cats

    Returns:
    - profiles_defect: list of arrays (n_bins) for each grain
    - profiles_extra: dict {cat_id: [arrays (n_bins) ...]}
    """
    if extra_mask_cats is None:
        extra_mask_cats = []

    folder = Path(folder)
    json_files = sorted(folder.glob("*.json"))

    profiles_defect = []
    profiles_extra = {cid: [] for cid in extra_mask_cats}

    for jf in json_files:
        with open(jf, "r") as f:
            anns = pd.read_json(f)

        if isinstance(anns, pd.DataFrame):
            records = anns.to_dict(orient="records")
        else:
            records = anns

        grain_masks = []
        nucleus_masks = []
        defect_masks = []
        extra_masks = {cid: [] for cid in extra_mask_cats}

        H = W = None

        for obj in records:
            seg = obj.get("segmentation", None)
            if seg is None:
                continue
            m = decode_mask(seg)
            H, W = m.shape
            cat = obj.get("category_id", None)

            if cat == 2:
                nucleus_masks.append(m)
            elif cat == 3:
                defect_masks.append(m)
            elif cat in extra_mask_cats:
                extra_masks[cat].append(m)
            else:
                # treat as grain
                grain_masks.append(m)

        if H is None:
            continue
        if len(grain_masks) == 0 or len(nucleus_masks) == 0:
            continue

        # unions
        defect_union = (
            np.logical_or.reduce(defect_masks) if len(defect_masks) > 0
            else np.zeros((H, W), dtype=bool)
        )

        extra_unions = {}
        for cid, masks in extra_masks.items():
            if len(masks) > 0:
                extra_unions[cid] = np.logical_or.reduce(masks)
            else:
                extra_unions[cid] = np.zeros((H, W), dtype=bool)

        # pair grains and nuclei
        pairs = pair_grains_and_nuclei(grain_masks, nucleus_masks)

        for g, n in pairs:
            if g.sum() < min_pixels:
                continue

            # defects
            prof_def = radial_fraction_for_grain(
                grain_mask=g,
                nucleus_mask=n,
                mask_union=defect_union,
                n_bins=n_bins,
            )
            profiles_defect.append(prof_def)

            # extras
            for cid in extra_mask_cats:
                prof_ex = radial_fraction_for_grain(
                    grain_mask=g,
                    nucleus_mask=n,
                    mask_union=extra_unions[cid],
                    n_bins=n_bins,
                )
                profiles_extra[cid].append(prof_ex)

    return profiles_defect, profiles_extra


def avg_profile(profile_list, n_bins):
    """Mean and std of a list of 1D profiles, ignoring NaN."""
    if len(profile_list) == 0:
        return np.full(n_bins, np.nan), np.full(n_bins, np.nan)
    arr = np.vstack(profile_list)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    return mean, std


def main():
    parser = argparse.ArgumentParser(
        description="Intragrain radial profiles for defect and extra mask-based descriptors."
    )
    parser.add_argument(
        "--fapi-dir",
        required=True,
        help="Folder with FAPI JSON files",
    )
    parser.add_argument(
        "--tempo-dir",
        required=True,
        help="Folder with FAPI–TEMPO JSON files",
    )
    parser.add_argument(
        "--out-prefix",
        required=True,
        help="Output prefix for CSV/PNGs",
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
        help="Minimum grain area (pixels) to include (default 200)",
    )
    parser.add_argument(
        "--extra-mask-cat",
        type=int,
        nargs="*",
        default=[],
        help="Additional category_id values to treat as mask-based descriptors.",
    )

    args = parser.parse_args()
    n_bins = args.n_bins
    extra_cats = args.extra_mask_cat

    print(f"[INFO] Extra mask categories: {extra_cats}")

    # collect FAPI
    print(f"[INFO] Collecting profiles for FAPI from {args.fapi_dir} ...")
    fapi_def, fapi_extra = collect_profiles_from_folder(
        folder=args.fapi_dir,
        n_bins=n_bins,
        min_pixels=args.min_pixels,
        extra_mask_cats=extra_cats,
    )
    print(f"[INFO] Got {len(fapi_def)} FAPI grains with profiles.")

    # collect FAPI–TEMPO
    print(f"[INFO] Collecting profiles for FAPI–TEMPO from {args.tempo_dir} ...")
    tempo_def, tempo_extra = collect_profiles_from_folder(
        folder=args.tempo_dir,
        n_bins=n_bins,
        min_pixels=args.min_pixels,
        extra_mask_cats=extra_cats,
    )
    print(f"[INFO] Got {len(tempo_def)} FAPI–TEMPO grains with profiles.")

    # radial bin centres
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    r_centers = 0.5 * (edges[:-1] + edges[1:])

    # mean/std for defect
    f_def_mean, f_def_std = avg_profile(fapi_def, n_bins)
    t_def_mean, t_def_std = avg_profile(tempo_def, n_bins)

    data = {
        "r_norm": r_centers,
        "phi_defect_FAPI_mean": f_def_mean,
        "phi_defect_FAPI_std": f_def_std,
        "phi_defect_FAPITEMPO_mean": t_def_mean,
        "phi_defect_FAPITEMPO_std": t_def_std,
    }

    # extras
    for cid in extra_cats:
        f_list = fapi_extra.get(cid, [])
        t_list = tempo_extra.get(cid, [])
        f_mean, f_std = avg_profile(f_list, n_bins)
        t_mean, t_std = avg_profile(t_list, n_bins)
        data[f"phi_cat{cid}_FAPI_mean"] = f_mean
        data[f"phi_cat{cid}_FAPI_std"] = f_std
        data[f"phi_cat{cid}_FAPITEMPO_mean"] = t_mean
        data[f"phi_cat{cid}_FAPITEMPO_std"] = t_std

    # save CSV
    out_prefix = Path(args.out_prefix)
    out_dir = out_prefix.parent if out_prefix.parent != Path("") else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_prefix.name

    df_out = pd.DataFrame(data)
    csv_path = out_dir / f"{base}_radial_profiles.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"[OK] Saved radial profiles CSV: {csv_path}")

    # plot defect-only profile
    png_def = out_dir / f"{base}_radial_profiles_defect.png"
    fig, ax = plt.subplots(figsize=(5.0, 4.0))

    ax.plot(r_centers, f_def_mean, "-o", ms=3, lw=1.8, label="FAPI")
    ax.fill_between(
        r_centers,
        f_def_mean - f_def_std,
        f_def_mean + f_def_std,
        alpha=0.2,
    )

    ax.plot(r_centers, t_def_mean, "-s", ms=3, lw=1.8, label="FAPI–TEMPO")
    ax.fill_between(
        r_centers,
        t_def_mean - t_def_std,
        t_def_mean + t_def_std,
        alpha=0.2,
    )

    ax.set_xlabel("normalised radius $r/\\langle r_{\\max} \\rangle$")
    ax.set_ylabel("defect fraction")
    ax.set_title("Intragrain radial defect profiles")
    ax.set_xlim(0, 1.0)
    ax.legend()
    ax.grid(False)

    plt.tight_layout()
    fig.savefig(png_def, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved defect radial profile plot: {png_def}")

    print("[DONE] All intragrain radial profiles computed.")


if __name__ == "__main__":
    main()
