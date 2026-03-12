#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def avrami_X(t: np.ndarray, t0: float, k: float, n: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    y = np.zeros_like(t, dtype=float)
    m = t > t0
    tau = t[m] - t0
    y[m] = 1.0 - np.exp(-k * np.power(tau, n))
    return np.clip(y, 0.0, 1.0)


def avrami_dXdt(t: np.ndarray, t0: float, k: float, n: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    y = np.zeros_like(t, dtype=float)
    m = t > t0
    tau = t[m] - t0
    y[m] = k * n * np.power(tau, n - 1.0) * np.exp(-k * np.power(tau, n))
    return y


def infer_tfrac(t: np.ndarray, X: np.ndarray, frac: float) -> float:
    t = np.asarray(t, float)
    X = np.asarray(X, float)

    ok = np.isfinite(t) & np.isfinite(X)
    t = t[ok]
    X = X[ok]

    if len(t) < 2:
        return np.nan

    Xm = np.maximum.accumulate(np.clip(X, 0, 1))
    ux, idx = np.unique(Xm, return_index=True)
    ut = t[idx]

    if ux[0] > 0:
        ux = np.r_[0.0, ux]
        ut = np.r_[t[0], ut]
    if ux[-1] < 1:
        ux = np.r_[ux, 1.0]
        ut = np.r_[ut, t[-1]]

    return float(np.interp(frac, ux, ut))


def halfmax_width(t: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    t = np.asarray(t, float)
    y = np.asarray(y, float)

    if len(t) < 3 or not np.isfinite(y).any():
        return np.nan, np.nan, np.nan

    peak_idx = int(np.nanargmax(y))
    peak_val = y[peak_idx]
    if not np.isfinite(peak_val) or peak_val <= 0:
        return np.nan, np.nan, np.nan

    half = 0.5 * peak_val
    above = np.where(y >= half)[0]
    if len(above) < 2:
        return np.nan, np.nan, np.nan

    left_idx = above[0]
    right_idx = above[-1]
    left_w = t[peak_idx] - t[left_idx]
    right_w = t[right_idx] - t[peak_idx]
    fwhm = t[right_idx] - t[left_idx]
    return float(left_w), float(right_w), float(fwhm)


def fit_avrami_continuous(t: np.ndarray, X: np.ndarray) -> dict:
    t = np.asarray(t, float)
    X = np.asarray(X, float)

    ok = np.isfinite(t) & np.isfinite(X)
    t = t[ok]
    X = np.clip(X[ok], 0, 1)

    if len(t) < 10:
        raise ValueError("Not enough valid points for continuous Avrami fit.")

    t05 = infer_tfrac(t, X, 0.05)
    t10 = infer_tfrac(t, X, 0.10)
    t50 = infer_tfrac(t, X, 0.50)
    t90 = infer_tfrac(t, X, 0.90)

    if not np.isfinite(t05):
        t05 = float(np.nanmin(t))
    if not np.isfinite(t10):
        t10 = float(np.nanpercentile(t, 10))
    if not np.isfinite(t50):
        t50 = float(np.nanmedian(t))
    if not np.isfinite(t90):
        t90 = float(np.nanpercentile(t, 90))

    t0_guess = max(0.0, min(t05, t10) - 0.5)
    span = max(t90 - t10, 1e-6)
    p0 = [t0_guess, 1.0 / span, 1.0]

    lower_bounds = [0.0, 1e-8, 0.2]
    upper_bounds = [max(t50, 5.0), 1e3, 8.0]

    popt, pcov = curve_fit(
        avrami_X,
        t,
        X,
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
        maxfev=200000,
    )

    t0, k, n = [float(v) for v in popt]
    X_fit = avrami_X(t, t0, k, n)
    dXdt_fit = avrami_dXdt(t, t0, k, n)

    rmse = float(np.sqrt(np.mean((X_fit - X) ** 2)))
    ss_res = float(np.sum((X - X_fit) ** 2))
    ss_tot = float(np.sum((X - np.mean(X)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    perr = np.sqrt(np.diag(pcov)) if pcov is not None and np.all(np.isfinite(pcov)) else np.array([np.nan, np.nan, np.nan])

    fit_t10 = infer_tfrac(t, X_fit, 0.10)
    fit_t50 = infer_tfrac(t, X_fit, 0.50)
    fit_t90 = infer_tfrac(t, X_fit, 0.90)

    peak_idx = int(np.nanargmax(dXdt_fit))
    fit_peak_time = float(t[peak_idx])
    fit_peak_height = float(dXdt_fit[peak_idx])

    _, _, fit_fwhm = halfmax_width(t, dXdt_fit)

    return {
        "t0": t0,
        "k": k,
        "n": n,
        "t0_se": float(perr[0]),
        "k_se": float(perr[1]),
        "n_se": float(perr[2]),
        "rmse_X": rmse,
        "r2_X": r2,
        "fit_t10_ms": fit_t10,
        "fit_t50_ms": fit_t50,
        "fit_t90_ms": fit_t90,
        "fit_rise_10_90_ms": fit_t90 - fit_t10 if np.isfinite(fit_t10) and np.isfinite(fit_t90) else np.nan,
        "fit_peak_time_dXdt_ms": fit_peak_time,
        "fit_peak_height_dXdt": fit_peak_height,
        "fit_FWHM_dXdt_ms": fit_fwhm,
        "X_fit": X_fit,
        "dXdt_fit": dXdt_fit,
    }


def load_area_weighted_curves(path: Path) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(path)
    req = {"time_ms", "X", "dX_dt", "sample", "weighting"}
    missing = req.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = {}
    for sample in ["FAPI", "FAPI-TEMPO"]:
        sub = df[(df["sample"] == sample) & (df["weighting"] == "area_weighted")].copy()
        sub = sub.sort_values("time_ms")
        if len(sub) == 0:
            raise ValueError(f"No area_weighted transported curves found for {sample}")
        out[sample] = sub
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curves", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("transported_avrami_continuous.png"))
    parser.add_argument("--params_out", type=Path, default=Path("transported_avrami_continuous_fit_params.csv"))
    parser.add_argument("--si_table_out", type=Path, default=Path("SI_Table_Avrami_Transported.csv"))
    parser.add_argument("--fig_width_cm", type=float, default=24.0)
    parser.add_argument("--fig_height_cm", type=float, default=10.5)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    curves = load_area_weighted_curves(args.curves)

    fit_rows = []
    fit_store = {}

    for sample in ["FAPI", "FAPI-TEMPO"]:
        sub = curves[sample]
        t = sub["time_ms"].to_numpy(float)
        X = sub["X"].to_numpy(float)

        fit = fit_avrami_continuous(t, X)
        fit_store[sample] = fit

        fit_rows.append(
            {
                "sample": sample,
                "weighting": "area_weighted",
                "avrami_t0": fit["t0"],
                "avrami_k": fit["k"],
                "avrami_n": fit["n"],
                "rmse_X": fit["rmse_X"],
                "r2_X": fit["r2_X"],
                "fit_t10_ms": fit["fit_t10_ms"],
                "fit_t50_ms": fit["fit_t50_ms"],
                "fit_t90_ms": fit["fit_t90_ms"],
                "fit_rise_10_90_ms": fit["fit_rise_10_90_ms"],
                "fit_peak_time_dXdt_ms": fit["fit_peak_time_dXdt_ms"],
                "fit_peak_height_dXdt": fit["fit_peak_height_dXdt"],
                "fit_FWHM_dXdt_ms": fit["fit_FWHM_dXdt_ms"],
            }
        )

    params_df = pd.DataFrame(
        [
            {
                "sample": sample,
                "weighting": "area_weighted",
                "avrami_t0": fit_store[sample]["t0"],
                "avrami_k": fit_store[sample]["k"],
                "avrami_n": fit_store[sample]["n"],
                "avrami_t0_se": fit_store[sample]["t0_se"],
                "avrami_k_se": fit_store[sample]["k_se"],
                "avrami_n_se": fit_store[sample]["n_se"],
                "rmse_X": fit_store[sample]["rmse_X"],
                "r2_X": fit_store[sample]["r2_X"],
            }
            for sample in ["FAPI", "FAPI-TEMPO"]
        ]
    )

    si_df = pd.DataFrame(fit_rows)

    params_df.to_csv(args.params_out, index=False)
    si_df.to_csv(args.si_table_out, index=False)

    cm_to_in = 1.0 / 2.54
    fig, axes = plt.subplots(
        1, 2,
        figsize=(args.fig_width_cm * cm_to_in, args.fig_height_cm * cm_to_in),
        dpi=args.dpi,
        facecolor="white",
    )

    ax1, ax2 = axes
    color_map = {"FAPI": "tab:blue", "FAPI-TEMPO": "tab:orange"}

    for sample in ["FAPI", "FAPI-TEMPO"]:
        sub = curves[sample]
        fit = fit_store[sample]
        t = sub["time_ms"].to_numpy(float)
        c = color_map[sample]

        ax1.plot(t, sub["X"], color=c, linewidth=2.6, label=f"{sample} transported")
        ax1.plot(t, fit["X_fit"], color=c, linestyle="--", linewidth=2.0, label=f"{sample} Avrami fit")

        ax2.plot(t, sub["dX_dt"], color=c, linewidth=2.6, label=f"{sample} transported")
        ax2.plot(t, fit["dXdt_fit"], color=c, linestyle="--", linewidth=2.0, label=f"{sample} Avrami fit")

    ax1.set_title("Transported X(t) with continuous Avrami fits", fontsize=13)
    ax1.set_xlabel("Effective time (ms)")
    ax1.set_ylabel("Transformed fraction")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(frameon=False, fontsize=8)

    ax2.set_title("Transported dX/dt(t) with continuous Avrami-fit derivatives", fontsize=13)
    ax2.set_xlabel("Effective time (ms)")
    ax2.set_ylabel("Transformation rate")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved figure: {args.out.resolve()}")
    print(f"Saved fit parameters: {args.params_out.resolve()}")
    print(f"Saved SI table: {args.si_table_out.resolve()}")
    print("\nSI Avrami table:")
    print(si_df.to_string(index=False))


if __name__ == "__main__":
    main()