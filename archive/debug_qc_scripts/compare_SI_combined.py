#!/usr/bin/env python3
"""
compare_SI_combined.py

Combine SI outputs for two datasets (e.g. FAPI vs FAPI-TEMPO):
- Refit K for each label from X_pred(t) with fixed Avrami exponent n
- Export combined CSVs:
    * combined_dn_dt.csv
    * combined_X_pred_Avrami.csv
    * combined_growth_hist.csv
    * combined_K_fits.csv
- Export combined PNG plots:
    * dn_dt_both_ms.png
    * X_overlay_both_ms.png
    * X_overlay_both_s.png
    * growth_hist_both.png

Usage:
    python compare_SI_combined.py \
        --dir path/to/per_label_outputs \
        --labelA FAPI \
        --labelB FAPI-TEMPO \
        --out path/to/combined_out \
        --n_avrami 2.5
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def fit_K_from_X(t_ms: np.ndarray, X_pred: np.ndarray, n: float) -> float:
    """
    Fit K in Avrami law X_A(t) = 1 - exp(-K * t^n)
    using the same median estimator as in export_SI_metrics_rank_v2.py.
    """
    t_ms = np.asarray(t_ms, dtype=float)
    X_pred = np.asarray(X_pred, dtype=float)

    y = np.clip(X_pred, 0.0, 0.999999)
    t_pow = t_ms ** n

    valid = (t_pow > 0) & (y > 0) & (y < 0.99)
    if np.any(valid):
        K_vals = -np.log(1.0 - y[valid]) / t_pow[valid]
        K_vals = K_vals[np.isfinite(K_vals)]
        if K_vals.size > 0:
            return float(np.nanmedian(K_vals))
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True,
                    help="Folder where per-dataset CSVs were exported "
                         "(*_X_pred_Avrami.csv, *_nucleation_dn_dt.csv, *_SI_metrics.csv)")
    ap.add_argument("--labelA", default="FAPI")
    ap.add_argument("--labelB", default="FAPI-TEMPO")
    ap.add_argument("--out", required=True,
                    help="Output folder for combined CSVs and plots")
    ap.add_argument("--n_avrami", type=float, default=2.5,
                    help="Fixed Avrami exponent n used to refit K for both datasets")
    args = ap.parse_args()

    indir = Path(args.dir)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    labelA = args.labelA
    labelB = args.labelB
    n = float(args.n_avrami)

    # ------------------------------------------------------------------
    # Load per-label CSVs
    # ------------------------------------------------------------------
    # X(t)
    XA = pd.read_csv(indir / f"{labelA}_X_pred_Avrami.csv")
    XB = pd.read_csv(indir / f"{labelB}_X_pred_Avrami.csv")

    # Ensure t_s exists; if not, create
    for X in (XA, XB):
        if "t_s" not in X.columns:
            X["t_s"] = X["t_ms"] / 1000.0

    # dn/dt
    dA = pd.read_csv(indir / f"{labelA}_nucleation_dn_dt.csv")
    dB = pd.read_csv(indir / f"{labelB}_nucleation_dn_dt.csv")

    # SI metrics (for growth-rate histogram)
    mA = pd.read_csv(indir / f"{labelA}_SI_metrics.csv")
    mB = pd.read_csv(indir / f"{labelB}_SI_metrics.csv")

    # ------------------------------------------------------------------
    # Refit K for each dataset from X_pred and build Avrami curves
    # ------------------------------------------------------------------
    t_ms_A = XA["t_ms"].to_numpy(dtype=float)
    t_ms_B = XB["t_ms"].to_numpy(dtype=float)
    X_pred_A = XA["X_pred"].to_numpy(dtype=float)
    X_pred_B = XB["X_pred"].to_numpy(dtype=float)

    K_A = fit_K_from_X(t_ms_A, X_pred_A, n=n)
    K_B = fit_K_from_X(t_ms_B, X_pred_B, n=n)

    X_Avrami_A = 1.0 - np.exp(-K_A * (t_ms_A ** n))
    X_Avrami_B = 1.0 - np.exp(-K_B * (t_ms_B ** n))

    # Add / overwrite Avrami columns with refitted K
    XA["X_Avrami_refit"] = X_Avrami_A
    XB["X_Avrami_refit"] = X_Avrami_B

    # ------------------------------------------------------------------
    # Combined CSVs
    # ------------------------------------------------------------------

    # 1) Combined dn/dt
    dn_comb = pd.merge(
        dA[["t_ms", "dn_dt_per_ms_mm2"]].rename(
            columns={"dn_dt_per_ms_mm2": f"dn_dt_per_ms_mm2_{labelA}"}),
        dB[["t_ms", "dn_dt_per_ms_mm2"]].rename(
            columns={"dn_dt_per_ms_mm2": f"dn_dt_per_ms_mm2_{labelB}"}),
        on="t_ms",
        how="outer",
        sort=True,
    )
    dn_comb.to_csv(outdir / "combined_dn_dt.csv", index=False)

    # 2) Combined X_pred and Avrami (refitted K)
    X_comb = pd.DataFrame({
        "t_ms": XA["t_ms"],
        "t_s": XA["t_s"],
        f"X_pred_{labelA}": XA["X_pred"],
        f"X_Avrami_refit_{labelA}": XA["X_Avrami_refit"]
    })
    # Align XB on t_ms by merge then append columns
    X_comb = pd.merge(
        X_comb,
        XB[["t_ms", "X_pred", "X_Avrami_refit"]].rename(
            columns={
                "X_pred": f"X_pred_{labelB}",
                "X_Avrami_refit": f"X_Avrami_refit_{labelB}"
            }),
        on="t_ms",
        how="outer",
        sort=True,
    )
    # Recompute t_s in case merge expanded t_ms
    X_comb["t_s"] = X_comb["t_ms"] / 1000.0
    X_comb.to_csv(outdir / "combined_X_pred_Avrami.csv", index=False)

    # 3) Combined growth-rate histogram data
    #    Use effective growth rate (µm/s)
    vA = mA.get("v_eff_um_per_s", pd.Series([], dtype=float))
    vB = mB.get("v_eff_um_per_s", pd.Series([], dtype=float))

    vA = vA.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    vB = vB.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)

    if vA.size > 0 or vB.size > 0:
        v_all = np.concatenate([vA, vB]) if (vA.size > 0 and vB.size > 0) else (vA if vA.size > 0 else vB)
        v_min, v_max = np.nanmin(v_all), np.nanmax(v_all)
        if v_min == v_max:
            v_min, v_max = 0.0, v_max * 1.1 if v_max > 0 else 1.0
        bins = np.linspace(v_min, v_max, 61)  # 60 bins

        def hist_df(v, label):
            if v.size == 0:
                return pd.DataFrame(columns=["label", "bin_left", "bin_right", "count"])
            counts, edges = np.histogram(v, bins=bins)
            return pd.DataFrame({
                "label": label,
                "bin_left": edges[:-1],
                "bin_right": edges[1:],
                "count": counts
            })

        histA = hist_df(vA, labelA)
        histB = hist_df(vB, labelB)
        hist_comb = pd.concat([histA, histB], ignore_index=True)
        hist_comb.to_csv(outdir / "combined_growth_hist.csv", index=False)
    else:
        hist_comb = pd.DataFrame(columns=["label", "bin_left", "bin_right", "count"])
        hist_comb.to_csv(outdir / "combined_growth_hist.csv", index=False)

    # 4) Combined K fits
    pd.DataFrame([
        {"label": labelA, "n_avrami": n, "K_fit": K_A},
        {"label": labelB, "n_avrami": n, "K_fit": K_B},
    ]).to_csv(outdir / "combined_K_fits.csv", index=False)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt

        # --- X overlay (ms) ---
        plt.figure()
        plt.plot(X_comb["t_ms"], X_comb[f"X_pred_{labelA}"], label=f"{labelA} X_pred")
        plt.plot(X_comb["t_ms"], X_comb[f"X_Avrami_refit_{labelA}"],
                 linestyle="--", label=f"{labelA} Avrami (n={n:.2f}, K={K_A:.3g})")
        plt.plot(X_comb["t_ms"], X_comb[f"X_pred_{labelB}"], label=f"{labelB} X_pred")
        plt.plot(X_comb["t_ms"], X_comb[f"X_Avrami_refit_{labelB}"],
                 linestyle="--", label=f"{labelB} Avrami (n={n:.2f}, K={K_B:.3g})")
        plt.xlabel("t (ms)")
        plt.ylabel("X(t) (fraction)")
        plt.title("X(t) vs Avrami — both datasets (ms)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "X_overlay_both_ms.png", dpi=200)
        plt.close()

        # --- X overlay (s) ---
        plt.figure()
        plt.plot(X_comb["t_s"], X_comb[f"X_pred_{labelA}"], label=f"{labelA} X_pred")
        plt.plot(X_comb["t_s"], X_comb[f"X_Avrami_refit_{labelA}"],
                 linestyle="--", label=f"{labelA} Avrami (n={n:.2f}, K={K_A:.3g})")
        plt.plot(X_comb["t_s"], X_comb[f"X_pred_{labelB}"], label=f"{labelB} X_pred")
        plt.plot(X_comb["t_s"], X_comb[f"X_Avrami_refit_{labelB}"],
                 linestyle="--", label=f"{labelB} Avrami (n={n:.2f}, K={K_B:.3g})")
        plt.xlabel("t (s)")
        plt.ylabel("X(t) (fraction)")
        plt.title("X(t) vs Avrami — both datasets (seconds)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "X_overlay_both_s.png", dpi=200)
        plt.close()

        # --- dn/dt combined (ms) ---
        plt.figure()
        if f"dn_dt_per_ms_mm2_{labelA}" in dn_comb:
            plt.plot(dn_comb["t_ms"], dn_comb[f"dn_dt_per_ms_mm2_{labelA}"], label=labelA)
        if f"dn_dt_per_ms_mm2_{labelB}" in dn_comb:
            plt.plot(dn_comb["t_ms"], dn_comb[f"dn_dt_per_ms_mm2_{labelB}"], label=labelB)
        plt.xlabel("t (ms)")
        plt.ylabel("dn/dt  [events / (ms·mm²)]")
        plt.title("Nucleation density rate dn/dt — both datasets")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "dn_dt_both_ms.png", dpi=200)
        plt.close()

        # --- Growth-rate histogram combined ---
        if vA.size > 0 or vB.size > 0:
            plt.figure()
            if vA.size > 0:
                plt.hist(vA, bins=bins, histtype="step", label=labelA)
            if vB.size > 0:
                plt.hist(vB, bins=bins, histtype="step", label=labelB)
            plt.xlabel("growth rate (µm/s) [effective]")
            plt.ylabel("count")
            plt.title("Growth-rate distribution — both datasets")
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir / "growth_hist_both.png", dpi=200)
            plt.close()

    except Exception as e:
        print("[WARN] Plotting skipped:", e)

    print(f"[OK] Wrote combined CSVs and plots to: {outdir}")


if __name__ == "__main__":
    main()
