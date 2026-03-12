#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
morpho_kinetics_from_crystal_metrics.py

Compute an effective radial growth rate v_eff for each grain directly
from the crystal_metrics*.csv files (FAPI and FAPI–TEMPO), then
plot correlations:

  1) v_eff vs circularity_distortion
  2) v_eff vs entropy
  3) v_eff vs defect_fraction = defects_area_(µm²)/area_(µm²)

The nucleation time t0 is inferred from area ranking:

    - Sort grains by area_(µm²) in descending order.
    - Assign ranks r = 0..N-1 (largest grain has r=0).
    - Map to a nucleation window [0, t_win_ms]:
          t0 = t_win_ms * r/(N-1).
    - Growth time: dt = t_max_ms - t0.
    - Equivalent radius: R = sqrt(area_(µm²)/pi).
    - Effective growth rate: v_eff = R/dt  [µm/ms].

Usage example (Windows):

python morpho_kinetics_from_crystal_metrics.py ^
  --cm-fapi "crystal_metrics.csv" ^
  --cm-tempo "crystal_metrics 1.csv" ^
  --out-prefix morpho_kinetics_from_cm ^
  --t-max-ms 600 ^
  --t-win-ms 60 ^
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
import math


def add_veff_from_area(df, t_max_ms=600.0, t_win_ms=60.0):
    """
    Add columns:
      rank_for_t0, t0_ms, dt_ms, R_um_final, v_eff_um_per_ms
    based on area_(µm²).
    """
    df = df.copy()
    if "area_(µm²)" not in df.columns:
        raise KeyError("Input CSV must contain 'area_(µm²)' column.")

    # sort by area descending (largest grains = earliest nucleation)
    df = df.sort_values("area_(µm²)", ascending=False).reset_index(drop=True)
    N = len(df)
    df["rank_for_t0"] = np.arange(N)
    df["t0_ms"] = t_win_ms * df["rank_for_t0"] / max(N - 1, 1)
    df["dt_ms"] = t_max_ms - df["t0_ms"]
    df.loc[df["dt_ms"] <= 0, "dt_ms"] = np.nan

    # equivalent final radius: area already in µm²
    df["R_um_final"] = np.sqrt(df["area_(µm²)"] / math.pi)
    df["v_eff_um_per_ms"] = df["R_um_final"] / df["dt_ms"]

    return df


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
        description="Compute v_eff from crystal_metrics*.csv and plot correlations."
    )
    ap.add_argument("--cm-fapi", required=True,
                    help="crystal_metrics.csv for FAPI")
    ap.add_argument("--cm-tempo", required=True,
                    help="crystal_metrics 1.csv for FAPI–TEMPO")
    ap.add_argument("--out-prefix", required=True,
                    help="Output prefix for CSVs and figures")
    ap.add_argument("--t-max-ms", type=float, default=600.0,
                    help="Maximum observation time t_max in ms (default 600)")
    ap.add_argument("--t-win-ms", type=float, default=60.0,
                    help="Nucleation window t_win in ms (default 60)")
    ap.add_argument("--circ-col", default="circularity_distortion",
                    help="Column name for circularity distortion")
    ap.add_argument("--entropy-col", default="entropy_hm_(bits)",
                    help="Column name for entropy")
    ap.add_argument("--defect-area-col", default="defects_area_(µm²)",
                    help="Column name for defects area (µm²)")
    ap.add_argument("--grain-area-col", default="area_(µm²)",
                    help="Column name for grain area (µm²)")
    ap.add_argument("--nbins", type=int, default=25,
                    help="Number of bins for binned means in plots")
    args = ap.parse_args()

    out_prefix = Path(args.out_prefix)
    out_dir = out_prefix.parent if out_prefix.parent != Path("") else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_prefix.name

    # --- load and add v_eff ---
    cm_fapi = pd.read_csv(args.cm_fapi)
    cm_tempo = pd.read_csv(args.cm_tempo)

    fapi_df = add_veff_from_area(cm_fapi,
                                 t_max_ms=args.t_max_ms,
                                 t_win_ms=args.t_win_ms)
    tempo_df = add_veff_from_area(cm_tempo,
                                  t_max_ms=args.t_max_ms,
                                  t_win_ms=args.t_win_ms)

    fapi_out_csv = out_dir / f"{base}_FAPI_with_veff.csv"
    tempo_out_csv = out_dir / f"{base}_FAPITEMPO_with_veff.csv"
    fapi_df.to_csv(fapi_out_csv, index=False)
    tempo_df.to_csv(tempo_out_csv, index=False)
    print(f"[OK] Saved FAPI with v_eff:      {fapi_out_csv}")
    print(f"[OK] Saved FAPI–TEMPO with v_eff: {tempo_out_csv}")

    # add defect_fraction if possible
    def add_defect_fraction_inplace(df, label):
        if args.defect-area-col in df.columns and args.grain-area-col in df.columns:
            num = df[args.defect-area-col].astype(float)
            den = df[args.grain-area-col].astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                df["defect_fraction"] = np.where(den > 0, num / den, np.nan)
            print(f"[INFO] Added defect_fraction for {label}.")
        else:
            print(f"[WARN] Missing defect/area cols in {label}; no defect_fraction computed.")

    # NOTE: attribute names can't have '-' so we access via getattr:
    defect_area_col = args.defect_area_col
    grain_area_col = args.grain_area_col

    def add_defect_fraction(df, label):
        if defect_area_col in df.columns and grain_area_col in df.columns:
            num = df[defect_area_col].astype(float)
            den = df[grain_area_col].astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                df["defect_fraction"] = np.where(den > 0, num / den, np.nan)
            print(f"[INFO] Added defect_fraction for {label}.")
        else:
            print(f"[WARN] Missing defect/area cols in {label}; no defect_fraction computed.")

    add_defect_fraction(fapi_df, "FAPI")
    add_defect_fraction(tempo_df, "FAPI–TEMPO")

    veff_f = fapi_df["v_eff_um_per_ms"].to_numpy()
    veff_t = tempo_df["v_eff_um_per_ms"].to_numpy()

    # -------- 1) v_eff vs circularity_distortion --------
    if args.circ_col in fapi_df.columns and args.circ_col in tempo_df.columns:
        circ_f, veff_f_circ = finite_pair(fapi_df[args.circ_col], veff_f)
        circ_t, veff_t_circ = finite_pair(tempo_df[args.circ_col], veff_t)

        fig, ax = plt.subplots(figsize=(5.2, 4))
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

    # -------- 2) v_eff vs entropy --------
    if args.entropy_col in fapi_df.columns and args.entropy_col in tempo_df.columns:
        ent_f, veff_f_ent = finite_pair(fapi_df[args.entropy_col], veff_f)
        ent_t, veff_t_ent = finite_pair(tempo_df[args.entropy_col], veff_t)

        fig, ax = plt.subplots(figsize=(5.2, 4))
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
        print("[WARN] entropy column not found; skipping v_eff vs entropy plot.")

    # -------- 3) v_eff vs defect_fraction --------
    if "defect_fraction" in fapi_df.columns and "defect_fraction" in tempo_df.columns:
        df_f, veff_f_def = finite_pair(fapi_df["defect_fraction"], veff_f)
        df_t, veff_t_def = finite_pair(tempo_df["defect_fraction"], veff_t)

        fig, ax = plt.subplots(figsize=(5.2, 4))
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
        print("[WARN] defect_fraction not available; skipping v_eff vs defect_fraction plot.")

    print("[DONE] v_eff computation and correlation plots complete.")


if __name__ == "__main__":
    main()
