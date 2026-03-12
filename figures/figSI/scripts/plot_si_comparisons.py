#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_SI_comparisons.py

Create PNG plots that compare FAPI vs FAPI-TEMPO on the same figures using
the SI outputs generated previously by export_SI_metrics_with_K.py:

Required per label in --si_dir:
  <LABEL>_nucleation_SI.csv   # columns: t_ms, t_s, J_t_events_per_s_um2
  <LABEL>_growth_SI.csv       # columns: v_um_s, v0_um_s_est, etc.
Optional:
  <LABEL>_growth_summary.csv  # (for reporting/future use)
Optional bulk transformation overlay:
  --x_pred_csv (columns: t_ms, X_pred_FAPI, X_pred_FAPI-TEMPO)
  --n_avrami (float) and (optionally) --K_ms_FAPI and --K_ms_TEMPO to draw Avrami curves

Outputs (PNG to --outdir):
  1) nucleation_J_both.png      (J(t) vs time, both datasets)
  2) growth_speed_hist_both.png (histogram of v_um_s, both datasets)
  3) base_speed_hist_both.png   (histogram of v0_um_s_est, both datasets)
  4) X_pred_avrami_both.png     (optional, if --x_pred_csv provided)

Example:
  python plot_SI_comparisons.py ^
    --si_dir  "D:\...\comparative_outputs_any\outputs_SI" ^
    --labels  FAPI FAPI-TEMPO ^
    --outdir  "D:\...\comparative_outputs_any\plots_SI" ^
    --x_pred_csv "D:\...\comparative_outputs_any\X_pred_both.csv" ^
    --n_avrami 2.5
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_si(si_dir: Path, label: str):
    nuc = pd.read_csv(si_dir / f"{label}_nucleation_SI.csv")
    grw = pd.read_csv(si_dir / f"{label}_growth_SI.csv")
    summ_path = si_dir / f"{label}_growth_summary.csv"
    summ = pd.read_csv(summ_path) if summ_path.exists() else None
    return nuc, grw, summ

def nice_bins(x, nbins=60):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.linspace(0, 1, 10)
    lo, hi = np.nanpercentile(x, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = x.min(), x.max()
    return np.linspace(lo, hi, nbins)

def plot_nucleation_both(nucA, labelA, nucB, labelB, outdir: Path):
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=160)
    # Prefer seconds for SI; still annotate ms on top axis if desired
    tA = nucA.get("t_s", nucA["t_ms"]/1000.0)
    tB = nucB.get("t_s", nucB["t_ms"]/1000.0)
    JA = nucA["J_t_events_per_s_um2"].to_numpy()
    JB = nucB["J_t_events_per_s_um2"].to_numpy()

    ax.plot(tA, JA, lw=2, label=f"{labelA}")
    ax.plot(tB, JB, lw=2, ls="--", label=f"{labelB}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Nucleation rate density $J(t)$  [events s$^{-1}$ $\mu$m$^{-2}$]")
    ax.set_title("Nucleation rate density (SI)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    outpath = outdir / "nucleation_J_both.png"
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"[OK] {outpath}")

def plot_growth_histograms(grwA, labelA, grwB, labelB, outdir: Path):
    # Histogram of effective observed growth speed v_um_s
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=160)
    vA = pd.to_numeric(grwA["v_um_s"], errors="coerce")
    vB = pd.to_numeric(grwB["v_um_s"], errors="coerce")
    bins = nice_bins(pd.concat([vA, vB], ignore_index=True), nbins=60)

    ax.hist(vA.dropna(), bins=bins, alpha=0.55, density=True, label=f"{labelA}")
    ax.hist(vB.dropna(), bins=bins, alpha=0.55, density=True, label=f"{labelB}")
    ax.set_xlabel(r"Observed growth speed $v$  [$\mu$m s$^{-1}$]")
    ax.set_ylabel("Density")
    ax.set_title("Growth speed distribution (observed)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    out1 = outdir / "growth_speed_hist_both.png"
    fig.savefig(out1, dpi=200)
    plt.close(fig)
    print(f"[OK] {out1}")

    # Histogram of base speed v0_um_s_est (de-penalized)
    if "v0_um_s_est" in grwA.columns and "v0_um_s_est" in grwB.columns:
        fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=160)
        v0A = pd.to_numeric(grwA["v0_um_s_est"], errors="coerce")
        v0B = pd.to_numeric(grwB["v0_um_s_est"], errors="coerce")
        bins = nice_bins(pd.concat([v0A, v0B], ignore_index=True), nbins=60)

        ax.hist(v0A.dropna(), bins=bins, alpha=0.55, density=True, label=f"{labelA}")
        ax.hist(v0B.dropna(), bins=bins, alpha=0.55, density=True, label=f"{labelB}")
        ax.set_xlabel(r"Base growth speed $v_0$  [$\mu$m s$^{-1}$]")
        ax.set_ylabel("Density")
        ax.set_title("Base speed distribution (de-penalized)")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        out2 = outdir / "base_speed_hist_both.png"
        fig.savefig(out2, dpi=200)
        plt.close(fig)
        print(f"[OK] {out2}")

def avrami_curve(t_ms, K_ms, n):
    t = np.asarray(t_ms, float)
    return 1.0 - np.exp(-K_ms * (t**n))

def plot_X_pred_and_avrami(x_pred_csv: Path, labelA: str, labelB: str,
                           n_avrami: float, K_ms_A: float, K_ms_B: float,
                           outdir: Path):
    df = pd.read_csv(x_pred_csv)
    # pick columns
    if "t_ms" in df.columns:
        t_ms = pd.to_numeric(df["t_ms"], errors="coerce").to_numpy()
    else:
        # first column fallback
        t_ms = pd.to_numeric(df.iloc[:,0], errors="coerce").to_numpy()

    # Try “X_pred_<LABEL>” first; then any numeric with matching suffix
    def pick_col(label):
        name = f"X_pred_{label}"
        if name in df.columns:
            return df[name]
        # fallback: case-insensitive
        for c in df.columns:
            if c.lower() == name.lower():
                return df[c]
        # last resort: if there are only two numeric cols, pick the second
        nums = [c for c in df.columns if c != "t_ms" and np.issubdtype(df[c].dtype, np.number)]
        if len(nums) >= 1:
            return df[nums[0]] if label == labelA else (df[nums[1]] if len(nums) > 1 else df[nums[0]])
        raise ValueError(f"Could not find X_pred column for label '{label}' in {x_pred_csv}")

    XA = pd.to_numeric(pick_col(labelA), errors="coerce").to_numpy()
    XB = pd.to_numeric(pick_col(labelB), errors="coerce").to_numpy()

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=160)
    ax.plot(t_ms/1000.0, XA, lw=2, label=f"X_pred {labelA}")
    ax.plot(t_ms/1000.0, XB, lw=2, label=f"X_pred {labelB}", ls="--")

    if np.isfinite(K_ms_A) and K_ms_A > 0:
        ax.plot(t_ms/1000.0, avrami_curve(t_ms, K_ms_A, n_avrami),
                lw=1.8, label=f"Avrami fit {labelA} (n={n_avrami:g})", alpha=0.9)
    if np.isfinite(K_ms_B) and K_ms_B > 0:
        ax.plot(t_ms/1000.0, avrami_curve(t_ms, K_ms_B, n_avrami),
                lw=1.8, label=f"Avrami fit {labelB} (n={n_avrami:g})", ls="--", alpha=0.9)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Transformed fraction, X(t)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Bulk fraction with Avrami overlays")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    outp = outdir / "X_pred_avrami_both.png"
    fig.savefig(outp, dpi=200)
    plt.close(fig)
    print(f"[OK] {outp}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--si_dir", required=True, help="Folder with *_growth_SI.csv and *_nucleation_SI.csv")
    ap.add_argument("--labels", nargs=2, required=True, help="Two dataset labels, e.g., FAPI FAPI-TEMPO")
    ap.add_argument("--outdir", required=True, help="Output folder for PNGs")
    ap.add_argument("--x_pred_csv", default=None,
                    help="Optional CSV with t_ms and X_pred_<LABEL> columns to draw bulk overlays")
    ap.add_argument("--n_avrami", type=float, default=2.5, help="Avrami exponent for overlays")
    ap.add_argument("--K_ms_FAPI", type=float, default=np.nan, help="(Optional) K in ms^-n for label 'FAPI'")
    ap.add_argument("--K_ms_TEMPO", type=float, default=np.nan, help="(Optional) K in ms^-n for label 'FAPI-TEMPO'")
    args = ap.parse_args()

    si_dir = Path(args.si_dir)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    labelA, labelB = args.labels

    # Load SI products
    nucA, grwA, summA = load_si(si_dir, labelA)
    nucB, grwB, summB = load_si(si_dir, labelB)

    # 1) Nucleation J(t) on same plot
    plot_nucleation_both(nucA, labelA, nucB, labelB, outdir)

    # 2) Growth histograms
    plot_growth_histograms(grwA, labelA, grwB, labelB, outdir)

    # 3) Optional X_pred overlays with Avrami curves
    if args.x_pred_csv:
        # pick K if provided via CLI; otherwise try growth_summary files
        K_A = args.K_ms_FAPI if labelA.upper()=="FAPI" else (args.K_ms_TEMPO if "TEMPO" in labelA.upper() else np.nan)
        K_B = args.K_ms_TEMPO if "TEMPO" in labelB.upper() else (args.K_ms_FAPI if labelB.upper()=="FAPI" else np.nan)

        def try_summary(label, current):
            if np.isfinite(current): return current
            fp = si_dir / f"{label}_growth_summary.csv"
            if fp.exists():
                df = pd.read_csv(fp)
                if "K_fit_ms_units" in df.columns and np.isfinite(df["K_fit_ms_units"].iloc[0]):
                    return float(df["K_fit_ms_units"].iloc[0])
            return np.nan

        K_A = try_summary(labelA, K_A)
        K_B = try_summary(labelB, K_B)

        plot_X_pred_and_avrami(Path(args.x_pred_csv), labelA, labelB,
                               n_avrami=args.n_avrami, K_ms_A=K_A, K_ms_B=K_B,
                               outdir=outdir)

if __name__ == "__main__":
    main()
