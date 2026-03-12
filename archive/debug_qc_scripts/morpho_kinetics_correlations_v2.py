#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
morpho_kinetics_correlations_v2.py

Use existing per-grain kinetics (per_grain_metrics_*.csv) and
crystal_metrics*.csv to build:

  1) v_eff vs circularity_distortion
  2) v_eff vs entropy
  3) v_eff vs defect_fraction

The two CSV families are assumed to correspond 1:1 in row order
(i.e. row i in per_grain_metrics_* is the same grain as row i in
crystal_metrics*), which is true for the files you provided.

Outputs:
  - morpho_kinetics_v2_FAPI_merged.csv
  - morpho_kinetics_v2_FAPITEMPO_merged.csv
  - morpho_kinetics_v2_veff_vs_circ_dist.png
  - morpho_kinetics_v2_veff_vs_entropy.png
  - morpho_kinetics_v2_veff_vs_defect_fraction.png

Example usage (Windows):

python morpho_kinetics_correlations_v2.py ^
  --fapi-kinetics per_grain_metrics_FAPI.csv ^
  --tempo-kinetics per_grain_metrics_FAPI-TEMPO.csv ^
  --cm-fapi "crystal_metrics.csv" ^
  --cm-tempo "crystal_metrics 1.csv" ^
  --out-prefix morpho_kinetics_v2 ^
  --circ-col circularity_distortion ^
  --entropy-col entropy_hm_(bits) ^
  --defect-area-col "defects_area_(µm²)" ^
  --grain-area-col "area_(µm²)"

"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def finite_pair(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def scatter_with_binning(ax, x_f, y_f, x_t, y_t,
                         label_f="FAPI", label_t="FAPI–TEMPO",
                         nbins=25):
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


def main():
    ap = argparse.ArgumentParser(
        description="Correlate v_eff with morphology metrics using per_grain_metrics_* and crystal_metrics*."
    )
    ap.add_argument("--fapi-kinetics", required=True,
                    help="per_grain_metrics_FAPI.csv")
    ap.add_argument("--tempo-kinetics", required=True,
                    help="per_grain_metrics_FAPI-TEMPO.csv")
    ap.add_argument("--cm-fapi", required=True,
                    help="crystal_metrics.csv for FAPI")
    ap.add_argument("--cm-tempo", required=True,
                    help="crystal_metrics 1.csv for FAPI–TEMPO")
    ap.add_argument("--out-prefix", required=True,
                    help="Output prefix for merged CSVs and figures")
    ap.add_argument("--circ-col", default="circularity_distortion",
                    help="Column name for circularity distortion in crystal_metrics")
    ap.add_argument("--entropy-col", default="entropy_hm_(bits)",
                    help="Column name for entropy in crystal_metrics")
    ap.add_argument("--defect-area-col", default="defects_area_(µm²)",
                    help="Column name for defects area in crystal_metrics")
    ap.add_argument("--grain-area-col", default="area_(µm²)",
                    help="Column name for grain area (µm²) in crystal_metrics")
    ap.add_argument("--nbins", type=int, default=25,
                    help="Number of bins for binned means in plots")
    args = ap.parse_args()

    out_prefix = Path(args.out_prefix)
    out_dir = out_prefix.parent if out_prefix.parent != Path("") else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_prefix.name

    # --- load CSVs ---
    k_fapi = pd.read_csv(args.fapi_kinetics)
    k_tempo = pd.read_csv(args.tempo_kinetics)
    cm_fapi = pd.read_csv(args.cm_fapi)
    cm_tempo = pd.read_csv(args.cm_tempo)

    # sanity check: same length
    if len(k_fapi) != len(cm_fapi):
        raise ValueError(f"FAPI lengths differ: {len(k_fapi)} vs {len(cm_fapi)}")
    if len(k_tempo) != len(cm_tempo):
        raise ValueError(f"TEMPO lengths differ: {len(k_tempo)} vs {len(cm_tempo)}")

    # index-wise concat
    merged_fapi = pd.concat([k_fapi.reset_index(drop=True),
                             cm_fapi.reset_index(drop=True)], axis=1)
    merged_tempo = pd.concat([k_tempo.reset_index(drop=True),
                              cm_tempo.reset_index(drop=True)], axis=1)

    merged_fapi_csv = out_dir / f"{base}_FAPI_merged.csv"
    merged_tempo_csv = out_dir / f"{base}_FAPITEMPO_merged.csv"
    merged_fapi.to_csv(merged_fapi_csv, index=False)
    merged_tempo.to_csv(merged_tempo_csv, index=False)
    print(f"[OK] Saved merged CSVs:\n  {merged_fapi_csv}\n  {merged_tempo_csv}")

    # add defect_fraction if possible
    def add_defect_fraction(df, label):
        if args.defect_area_col in df.columns and args.grain_area_col in df.columns:
            num = df[args.defect_area_col].astype(float)
            den = df[args.grain_area_col].astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                df["defect_fraction"] = np.where(den > 0, num / den, np.nan)
            print(f"[INFO] Added defect_fraction for {label}.")
        else:
            print(f"[WARN] No defect_fraction for {label}: missing columns.")

    add_defect_fraction(merged_fapi, "FAPI")
    add_defect_fraction(merged_tempo, "FAPI–TEMPO")

    veff_f = merged_fapi["v_eff_um_per_ms"].to_numpy()
    veff_t = merged_tempo["v_eff_um_per_ms"].to_numpy()

    # -------- 1) v_eff vs circularity_distortion --------
    if args.circ_col in merged_fapi.columns and args.circ_col in merged_tempo.columns:
        circ_f, veff_f_circ = finite_pair(merged_fapi[args.circ_col], veff_f)
        circ_t, veff_t_circ = finite_pair(merged_tempo[args.circ_col], veff_t)

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
        print("[WARN] circularity_distortion column missing; skipping first plot.")

    # -------- 2) v_eff vs entropy --------
    if args.entropy_col in merged_fapi.columns and args.entropy_col in merged_tempo.columns:
        ent_f, veff_f_ent = finite_pair(merged_fapi[args.entropy_col], veff_f)
        ent_t, veff_t_ent = finite_pair(merged_tempo[args.entropy_col], veff_t)

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
        print("[WARN] entropy column missing; skipping second plot.")

    # -------- 3) v_eff vs defect_fraction --------
    if "defect_fraction" in merged_fapi.columns and "defect_fraction" in merged_tempo.columns:
        df_f, veff_f_def = finite_pair(merged_fapi["defect_fraction"], veff_f)
        df_t, veff_t_def = finite_pair(merged_tempo["defect_fraction"], veff_t)

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
        print("[WARN] defect_fraction missing; skipping third plot.")

    print("[DONE] Correlation plots generated.")


if __name__ == "__main__":
    main()
