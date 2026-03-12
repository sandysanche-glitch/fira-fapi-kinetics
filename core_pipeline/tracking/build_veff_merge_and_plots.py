#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_v_eff_merge_and_plots.py

1) Compute per-grain effective growth rates v_eff for FAPI and FAPI–TEMPO
   directly from COCO-style JSON masks.
2) Merge with crystal_metrics*.csv via 'file_name'.
3) Produce correlation plots:
     - v_eff vs circularity_distortion
     - v_eff vs entropy
     - v_eff vs defect_fraction

Outputs:
  - *_kinetics_only.csv                  (per-grain kinetics from JSON)
  - *_morpho_kinetics_merged.csv         (merged with crystal_metrics)
  - *_veff_vs_circ_dist.png
  - *_veff_vs_entropy.png                (if entropy column found)
  - *_veff_vs_defect_fraction.png        (if defect columns found)

Example usage (Windows):

python build_v_eff_merge_and_plots.py ^
  --fapi-json-dir "D:\\...\\FAPI" ^
  --tempo-json-dir "D:\\...\\FAPI-TEMPO" ^
  --cm-fapi "D:\\...\\crystal_metrics.csv" ^
  --cm-tempo "D:\\...\\crystal_metrics 1.csv" ^
  --out-prefix "D:\\...\\per_grain_morpho_kinetics" ^
  --px-per-um 2.20014 ^
  --t-max-ms 600 ^
  --t-win-ms 60 ^
  --grain-cat-id 2 ^
  --defect-cat-id 3 ^
  --alpha 2.0 ^
  --beta 3.0 ^
  --circ-col circularity_distortion ^
  --entropy-col entropy_hm_(bits) ^
  --defect-area-col "defects_area_(µm²)" ^
  --grain-area-col "area_(µm²)"
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from pycocotools import mask as mask_utils
except Exception as e:
    mask_utils = None
    _PYCOCO_ERR = e
else:
    _PYCOCO_ERR = None


# ------------------ low-level helpers ------------------

def decode_rle_mask(segmentation):
    """Decode COCO-style RLE segmentation to boolean mask."""
    if mask_utils is None:
        raise RuntimeError(
            "pycocotools is required to decode RLE masks but could not be imported: "
            f"{_PYCOCO_ERR}"
        )
    if not isinstance(segmentation, dict):
        raise ValueError(f"Unsupported segmentation format: {type(segmentation)}")

    if "size" not in segmentation or "counts" not in segmentation:
        raise ValueError("Segmentation dict must contain 'size' and 'counts' keys.")

    rle = {"size": segmentation["size"], "counts": segmentation["counts"]}
    m = mask_utils.decode(rle)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(bool)


def compute_area(mask):
    return float(np.count_nonzero(mask))


def compute_perimeter(mask):
    """
    Crude perimeter as number of boundary pixels (8-connected).
    """
    m = mask.astype(bool)
    if m.size == 0:
        return 0.0
    # pad
    m_pad = np.pad(m, 1, mode="constant", constant_values=False)
    center = m_pad[1:-1, 1:-1]
    neighbors = [
        m_pad[:-2, 1:-1],   # up
        m_pad[2:, 1:-1],    # down
        m_pad[1:-1, :-2],   # left
        m_pad[1:-1, 2:],    # right
        m_pad[:-2, :-2],    # up-left
        m_pad[:-2, 2:],     # up-right
        m_pad[2:, :-2],     # down-left
        m_pad[2:, 2:],      # down-right
    ]
    interior = center.copy()
    for n in neighbors:
        interior &= n
    boundary = center & (~interior)
    return float(boundary.sum())


def compute_circularity(area_px, perimeter_px):
    if perimeter_px <= 0 or area_px <= 0:
        return math.nan
    C = 4.0 * math.pi * area_px / (perimeter_px ** 2)
    return float(min(max(C, 0.0), 1.0))


# ------------------ dataset processing ------------------

def process_dataset(json_dir,
                    label,
                    px_per_um=2.20014,
                    t_max_ms=600.0,
                    t_win_ms=60.0,
                    grain_cat_id=2,
                    defect_cat_id=3,
                    alpha=2.0,
                    beta=3.0):
    """
    Process all JSONs in json_dir and compute per-grain v_eff.

    Returns:
        df_sorted : DataFrame with columns
          ['label','file_name','json_file','grain_index',
           'area_px','circularity_C','defect_frac_phi',
           't0_ms','dt_ms_for_v','R_um_final',
           'v_unpen_um_per_ms','v_eff_um_per_ms',
           'v_unpen_um_per_s','v_eff_um_per_s','penalty_factor']
    """
    json_dir = Path(json_dir)
    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {json_dir}")

    grain_records = []

    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)

        # COCO style: either list of annotations or dict with "annotations"
        if isinstance(data, dict) and "annotations" in data:
            anns = data["annotations"]
        elif isinstance(data, list):
            anns = data
        else:
            raise ValueError(f"Unsupported JSON structure in {jf}")

        defect_masks = []
        grain_objs = []

        for idx, ann in enumerate(anns):
            cat = ann.get("category_id", None)
            if cat is None:
                continue
            seg = ann.get("segmentation", None)
            if seg is None:
                continue
            if cat == defect_cat_id:
                try:
                    dm = decode_rle_mask(seg)
                    defect_masks.append(dm)
                except Exception as e:
                    print(f"[WARN] Defect mask decode failed in {jf.name} idx {idx}: {e}")
            elif cat == grain_cat_id:
                grain_objs.append((idx, ann))

        if defect_masks:
            defect_union = np.any(np.stack(defect_masks, axis=0), axis=0)
        else:
            defect_union = None

        stem = jf.stem  # e.g. FAPI_0

        for idx, ann in grain_objs:
            seg = ann["segmentation"]
            try:
                gmask = decode_rle_mask(seg)
            except Exception as e:
                print(f"[WARN] Grain mask decode failed in {jf.name} idx {idx}: {e}")
                continue

            area_px = compute_area(gmask)
            if area_px <= 0:
                continue

            peri_px = compute_perimeter(gmask)
            C = compute_circularity(area_px, peri_px)

            if defect_union is not None:
                inter = gmask & defect_union
                defect_px = float(inter.sum())
                phi = defect_px / area_px
            else:
                phi = 0.0

            file_name = f"{stem}_{idx}"

            grain_records.append({
                "label": label,
                "file_name": file_name,
                "json_file": jf.name,
                "grain_index": idx,
                "area_px": area_px,
                "circularity_C": C,
                "defect_frac_phi": phi
            })

    if not grain_records:
        raise RuntimeError(f"No grains found in {json_dir}")

    df = pd.DataFrame(grain_records)

    # rank-based nucleation times
    N = len(df)
    df_sorted = df.sort_values("area_px", ascending=False).reset_index(drop=True)
    df_sorted["rank"] = np.arange(N)  # 0..N-1, 0 = largest grain
    df_sorted["t0_ms"] = t_win_ms * df_sorted["rank"] / max(N - 1, 1)
    df_sorted["dt_ms_for_v"] = t_max_ms - df_sorted["t0_ms"]
    df_sorted.loc[df_sorted["dt_ms_for_v"] <= 0, "dt_ms_for_v"] = np.nan

    # equivalent final radius
    df_sorted["R_um_final"] = np.sqrt(df_sorted["area_px"] / math.pi) / px_per_um

    df_sorted["v_unpen_um_per_ms"] = df_sorted["R_um_final"] / df_sorted["dt_ms_for_v"]

    C = df_sorted["circularity_C"].fillna(0.0)
    phi = df_sorted["defect_frac_phi"].fillna(0.0)
    penalty = np.exp(-alpha * (1.0 - C)) * np.exp(-beta * phi)
    df_sorted["penalty_factor"] = penalty
    df_sorted["v_eff_um_per_ms"] = df_sorted["v_unpen_um_per_ms"] * df_sorted["penalty_factor"]

    df_sorted["v_unpen_um_per_s"] = df_sorted["v_unpen_um_per_ms"] * 1000.0
    df_sorted["v_eff_um_per_s"] = df_sorted["v_eff_um_per_ms"] * 1000.0

    return df_sorted


# ------------------ plotting helpers ------------------

def scatter_with_binning(ax, x_f, y_f, x_t, y_t,
                         label_f="FAPI", label_t="FAPI–TEMPO",
                         nbins=25):
    """
    Make a light scatter + binned mean/std envelopes for both datasets.
    """
    ax.scatter(x_f, y_f, s=5, alpha=0.15, label=f"{label_f} (points)")
    ax.scatter(x_t, y_t, s=5, alpha=0.15, label=f"{label_t} (points)")

    def binned_stats(x, y, nbins):
        x = np.asarray(x)
        y = np.asarray(y)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if len(x) == 0:
            return None, None, None
        xmin, xmax = x.min(), x.max()
        if xmin == xmax:
            xmin -= 1e-6
            xmax += 1e-6
        edges = np.linspace(xmin, xmax, nbins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        y_mean = np.full(nbins, np.nan)
        y_std = np.full(nbins, np.nan)
        for i in range(nbins):
            m_bin = (x >= edges[i]) & (x < edges[i+1])
            if np.any(m_bin):
                y_mean[i] = np.mean(y[m_bin])
                y_std[i] = np.std(y[m_bin])
        return centers, y_mean, y_std

    cf, mf, sf = binned_stats(x_f, y_f, nbins)
    if cf is not None:
        ax.plot(cf, mf, "-", color="C0", lw=2, label=f"{label_f} (mean)")
        ax.fill_between(cf, mf-sf, mf+sf, color="C0", alpha=0.15)

    ct, mt, st = binned_stats(x_t, y_t, nbins)
    if ct is not None:
        ax.plot(ct, mt, "-", color="C1", lw=2, label=f"{label_t} (mean)")
        ax.fill_between(ct, mt-st, mt+st, color="C1", alpha=0.15)


# ------------------ main ------------------

def main():
    ap = argparse.ArgumentParser(
        description="Compute per-grain v_eff from JSONs, merge with crystal_metrics, and plot correlations."
    )
    ap.add_argument("--fapi-json-dir", required=True,
                    help="Folder with FAPI JSON files.")
    ap.add_argument("--tempo-json-dir", required=True,
                    help="Folder with FAPI–TEMPO JSON files.")
    ap.add_argument("--cm-fapi", required=True,
                    help="crystal_metrics.csv for FAPI.")
    ap.add_argument("--cm-tempo", required=True,
                    help="crystal_metrics for FAPI–TEMPO (e.g. 'crystal_metrics 1.csv').")
    ap.add_argument("--out-prefix", required=True,
                    help="Output prefix for merged CSVs and figures.")
    ap.add_argument("--px-per-um", type=float, default=2.20014,
                    help="Pixels per micrometre (default 2.20014).")
    ap.add_argument("--t-max-ms", type=float, default=600.0,
                    help="Maximum observation time t_max in ms (default 600).")
    ap.add_argument("--t-win-ms", type=float, default=60.0,
                    help="Nucleation window t_win in ms (default 60).")
    ap.add_argument("--grain-cat-id", type=int, default=2,
                    help="category_id for grains (default 2).")
    ap.add_argument("--defect-cat-id", type=int, default=3,
                    help="category_id for defects (default 3).")
    ap.add_argument("--alpha", type=float, default=2.0,
                    help="Penalty weight alpha for circularity (default 2.0).")
    ap.add_argument("--beta", type=float, default=3.0,
                    help="Penalty weight beta for defect fraction (default 3.0).")
    # columns from crystal_metrics
    ap.add_argument("--circ-col", type=str, default="circularity_distortion",
                    help="Column name for circularity distortion (default: circularity_distortion).")
    ap.add_argument("--entropy-col", type=str, default="entropy_hm_(bits)",
                    help="Column name for entropy (default: entropy_hm_(bits)).")
    ap.add_argument("--defect-area-col", type=str, default="defects_area_(µm²)",
                    help="Column for defects area (default: 'defects_area_(µm²)').")
    ap.add_argument("--grain-area-col", type=str, default="area_(µm²)",
                    help="Column for grain area in µm² (default: 'area_(µm²)').")
    ap.add_argument("--nbins", type=int, default=25,
                    help="Number of bins for binned means in plots (default 25).")

    args = ap.parse_args()

    out_prefix = Path(args.out_prefix)
    out_dir = out_prefix.parent if out_prefix.parent != Path("") else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_prefix.name

    # --- compute kinetics from JSONs ---
    print(f"[INFO] Processing FAPI JSONs in {args.fapi_json_dir} ...")
    fapi_df = process_dataset(
        args.fapi_json_dir,
        label="FAPI",
        px_per_um=args.px_per_um,
        t_max_ms=args.t_max_ms,
        t_win_ms=args.t_win_ms,
        grain_cat_id=args.grain_cat_id,
        defect_cat_id=args.defect_cat_id,
        alpha=args.alpha,
        beta=args.beta,
    )
    print(f"[INFO] Processing FAPI–TEMPO JSONs in {args.tempo_json_dir} ...")
    tempo_df = process_dataset(
        args.tempo_json_dir,
        label="FAPI-TEMPO",
        px_per_um=args.px_per_um,
        t_max_ms=args.t_max_ms,
        t_win_ms=args.t_win_ms,
        grain_cat_id=args.grain_cat_id,
        defect_cat_id=args.defect_cat_id,
        alpha=args.alpha,
        beta=args.beta,
    )

    fapi_kinetics_csv = out_dir / f"{base}_FAPI_kinetics_only.csv"
    tempo_kinetics_csv = out_dir / f"{base}_FAPITEMPO_kinetics_only.csv"
    fapi_df.to_csv(fapi_kinetics_csv, index=False)
    tempo_df.to_csv(tempo_kinetics_csv, index=False)
    print(f"[OK] Saved per-grain kinetics:\n  {fapi_kinetics_csv}\n  {tempo_kinetics_csv}")

    # --- merge with crystal_metrics ---
    print("[INFO] Merging with crystal_metrics CSVs...")
    cm_fapi = pd.read_csv(args.cm_fapi)
    cm_tempo = pd.read_csv(args.cm_tempo)

    if "file_name" not in cm_fapi.columns or "file_name" not in cm_tempo.columns:
        raise ValueError("crystal_metrics CSVs must contain a 'file_name' column.")

    merged_fapi = pd.merge(fapi_df, cm_fapi, on="file_name", how="inner", suffixes=("", "_cm"))
    merged_tempo = pd.merge(tempo_df, cm_tempo, on="file_name", how="inner", suffixes=("", "_cm"))

    merged_fapi_csv = out_dir / f"{base}_FAPI_morpho_kinetics_merged.csv"
    merged_tempo_csv = out_dir / f"{base}_FAPITEMPO_morpho_kinetics_merged.csv"
    merged_fapi.to_csv(merged_fapi_csv, index=False)
    merged_tempo.to_csv(merged_tempo_csv, index=False)
    print(f"[OK] Saved merged CSVs:\n  {merged_fapi_csv}\n  {merged_tempo_csv}")

    # --- add defect_fraction if possible ---
    def add_defect_fraction(df, label):
        if args.defect_area_col in df.columns and args.grain_area_col in df.columns:
            num = df[args.defect_area_col].astype(float)
            den = df[args.grain_area_col].astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                df["defect_fraction"] = np.where(den > 0, num / den, np.nan)
            print(f"[INFO] Added defect_fraction column for {label}.")
        else:
            print(f"[WARN] Could not find defect and/or grain area columns in {label}; "
                  "no defect_fraction will be added.")

    add_defect_fraction(merged_fapi, "FAPI")
    add_defect_fraction(merged_tempo, "FAPI–TEMPO")

    # --- build correlation plots ---

    def finite_pair(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        return x[m], y[m]

    veff_f = merged_fapi["v_eff_um_per_ms"].to_numpy()
    veff_t = merged_tempo["v_eff_um_per_ms"].to_numpy()

    # 1) v_eff vs circularity_distortion
    if args.circ_col in merged_fapi.columns and args.circ_col in merged_tempo.columns:
        circ_f = merged_fapi[args.circ_col].to_numpy()
        circ_t = merged_tempo[args.circ_col].to_numpy()
        circ_f, veff_f_circ = finite_pair(circ_f, veff_f)
        circ_t, veff_t_circ = finite_pair(circ_t, veff_t)

        fig, ax = plt.subplots(figsize=(6, 4))
        scatter_with_binning(ax, circ_f, veff_f_circ, circ_t, veff_t_circ,
                             label_f="FAPI", label_t="FAPI–TEMPO",
                             nbins=args.nbins)
        ax.set_xlabel("Circularity distortion (from nucleation center)")
        ax.set_ylabel(r"Effective growth rate $v_{\rm eff}$ ($\mu$m/ms)")
        ax.set_title(r"$v_{\rm eff}$ vs circularity distortion")
        ax.legend()
        plt.tight_layout()
        out1 = out_dir / f"{base}_veff_vs_circ_dist.png"
        plt.savefig(out1, dpi=300)
        plt.close()
        print(f"[OK] Saved: {out1}")
    else:
        print("[WARN] circularity_distortion column not found in one or both CSVs; "
              "skipping v_eff vs circularity_distortion plot.")

    # 2) v_eff vs entropy
    if args.entropy_col in merged_fapi.columns and args.entropy_col in merged_tempo.columns:
        ent_f = merged_fapi[args.entropy_col].to_numpy()
        ent_t = merged_tempo[args.entropy_col].to_numpy()
        ent_f, veff_f_ent = finite_pair(ent_f, veff_f)
        ent_t, veff_t_ent = finite_pair(ent_t, veff_t)

        fig, ax = plt.subplots(figsize=(6, 4))
        scatter_with_binning(ax, ent_f, veff_f_ent, ent_t, veff_t_ent,
                             label_f="FAPI", label_t="FAPI–TEMPO",
                             nbins=args.nbins)
        ax.set_xlabel("Shannon entropy (grain texture)")
        ax.set_ylabel(r"Effective growth rate $v_{\rm eff}$ ($\mu$m/ms)")
        ax.set_title(r"$v_{\rm eff}$ vs entropy")
        ax.legend()
        plt.tight_layout()
        out2 = out_dir / f"{base}_veff_vs_entropy.png"
        plt.savefig(out2, dpi=300)
        plt.close()
        print(f"[OK] Saved: {out2}")
    else:
        print("[WARN] Entropy column not found in one or both CSVs; "
              "skipping v_eff vs entropy plot.")

    # 3) v_eff vs defect_fraction (if present)
    if "defect_fraction" in merged_fapi.columns and "defect_fraction" in merged_tempo.columns:
        df_f = merged_fapi["defect_fraction"].to_numpy()
        df_t = merged_tempo["defect_fraction"].to_numpy()
        df_f, veff_f_def = finite_pair(df_f, veff_f)
        df_t, veff_t_def = finite_pair(df_t, veff_t)

        fig, ax = plt.subplots(figsize=(6, 4))
        scatter_with_binning(ax, df_f, veff_f_def, df_t, veff_t_def,
                             label_f="FAPI", label_t="FAPI–TEMPO",
                             nbins=args.nbins)
        ax.set_xlabel(r"Defect fraction $\phi$")
        ax.set_ylabel(r"Effective growth rate $v_{\rm eff}$ ($\mu$m/ms)")
        ax.set_title(r"$v_{\rm eff}$ vs defect fraction")
        ax.legend()
        plt.tight_layout()
        out3 = out_dir / f"{base}_veff_vs_defect_fraction.png"
        plt.savefig(out3, dpi=300)
        plt.close()
        print(f"[OK] Saved: {out3}")
    else:
        print("[WARN] defect_fraction column not found in one or both CSVs; "
              "skipping v_eff vs defect_fraction plot.")

    print("[DONE] v_eff computation, merging and plotting completed.")


if __name__ == "__main__":
    main()
