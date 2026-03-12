#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load(label, indir):
    """Load X_pred, dn/dt, and metrics."""
    XA = pd.read_csv(indir / f"{label}_X_pred_Avrami.csv")
    dA = pd.read_csv(indir / f"{label}_nucleation_dn_dt.csv")
    mA = pd.read_csv(indir / f"{label}_SI_metrics.csv")
    return XA, dA, mA

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Folder containing per-dataset CSVs")
    ap.add_argument("--labelA", default="FAPI")
    ap.add_argument("--labelB", default="FAPI-TEMPO")
    ap.add_argument("--out", required=True, help="Output folder for combined exports/plots")
    args = ap.parse_args()

    indir = Path(args.dir)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    XA, dA, mA = load(args.labelA, indir)
    XB, dB, mB = load(args.labelB, indir)

    # Ensure t_s exists
    for X in (XA, XB):
        if "t_s" not in X.columns:
            X["t_s"] = X["t_ms"] / 1000.0

    # ------------------------------------------------------------
    # === Combined CSV exports ===
    # ------------------------------------------------------------

    # Combined X(t)
    X_comb = pd.DataFrame({
        "t_ms": XA["t_ms"],
        f"{args.labelA}_X_pred": XA["X_pred"],
        f"{args.labelA}_X_Avrami": XA["X_Avrami"],
        f"{args.labelB}_X_pred": XB["X_pred"],
        f"{args.labelB}_X_Avrami": XB["X_Avrami"]
    })
    X_comb.to_csv(outdir / "combined_X_pred_Avrami.csv", index=False)

    # Combined dn/dt
    dn_comb = pd.DataFrame({
        "t_ms": dA["t_ms"],
        f"{args.labelA}_dn_dt": dA["dn_dt_per_ms_mm2"],
        f"{args.labelB}_dn_dt": dB["dn_dt_per_ms_mm2"]
    })
    dn_comb.to_csv(outdir / "combined_dn_dt.csv", index=False)

    # Combined histograms (growth rate)
    growthA = mA["v_eff_um_per_s"].replace([np.inf, -np.inf], np.nan).dropna()
    growthB = mB["v_eff_um_per_s"].replace([np.inf, -np.inf], np.nan).dropna()

    hist_growth = pd.DataFrame({
        args.labelA: growthA.values,
        args.labelB: growthB.values
    })
    hist_growth.to_csv(outdir / "combined_growth_rates.csv", index=False)

    # ------------------------------------------------------------
    # === PLOTS ===
    # ------------------------------------------------------------

    # === 1. Combined X(t) (ms) ===
    plt.figure()
    plt.plot(XA["t_ms"], XA["X_pred"], label=f"{args.labelA} X_pred")
    plt.plot(XA["t_ms"], XA["X_Avrami"], "--", label=f"{args.labelA} Avrami-fit")
    plt.plot(XB["t_ms"], XB["X_pred"], label=f"{args.labelB} X_pred")
    plt.plot(XB["t_ms"], XB["X_Avrami"], "--", label=f"{args.labelB} Avrami-fit")
    plt.xlabel("t (ms)")
    plt.ylabel("X(t)")
    plt.title("X(t) vs Avrami — Combined")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "X_combined_ms.png", dpi=250)
    plt.close()

    # === 2. Combined X(t) (seconds) ===
    plt.figure()
    plt.plot(XA["t_s"], XA["X_pred"], label=f"{args.labelA} X_pred")
    plt.plot(XA["t_s"], XA["X_Avrami"], "--", label=f"{args.labelA} Avrami-fit")
    plt.plot(XB["t_s"], XB["X_pred"], label=f"{args.labelB} X_pred")
    plt.plot(XB["t_s"], XB["X_Avrami"], "--", label=f"{args.labelB} Avrami-fit")
    plt.xlabel("t (s)")
    plt.ylabel("X(t)")
    plt.title("X(t) vs Avrami — Combined (seconds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "X_combined_s.png", dpi=250)
    plt.close()

    # === 3. Combined dn/dt ===
    plt.figure()
    plt.plot(dA["t_ms"], dA["dn_dt_per_ms_mm2"], label=args.labelA)
    plt.plot(dB["t_ms"], dB["dn_dt_per_ms_mm2"], label=args.labelB)
    plt.xlabel("t (ms)")
    plt.ylabel("dn/dt   [events / (ms·mm²)]")
    plt.title("Nucleation rate dn/dt — Combined")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "dn_dt_combined.png", dpi=250)
    plt.close()

    # === 4. Growth-rate histogram combined ===
    plt.figure()
    bins = np.linspace(
        min(growthA.min(), growthB.min()),
        max(growthA.max(), growthB.max()),
        50
    )
    plt.hist(growthA, bins=bins, alpha=0.5, label=args.labelA)
    plt.hist(growthB, bins=bins, alpha=0.5, label=args.labelB)
    plt.xlabel("growth rate (µm/s)   [effective]")
    plt.ylabel("count")
    plt.title("Growth-rate distributions — Combined")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "growth_hist_combined.png", dpi=250)
    plt.close()

    print("\n[OK] All combined CSVs and plots written to:", outdir, "\n")

if __name__ == "__main__":
    main()
