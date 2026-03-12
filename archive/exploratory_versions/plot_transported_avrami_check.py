#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def avrami_X(t: np.ndarray, t0: float, k: float, n: float) -> np.ndarray:
    t = np.asarray(t, float)
    y = np.zeros_like(t, dtype=float)
    m = t > t0
    tau = t[m] - t0
    y[m] = 1.0 - np.exp(-k * np.power(tau, n))
    return np.clip(y, 0.0, 1.0)


def avrami_dXdt(t: np.ndarray, t0: float, k: float, n: float) -> np.ndarray:
    t = np.asarray(t, float)
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


def fit_avrami_grid(t: np.ndarray, X: np.ndarray) -> dict:
    """
    Simple robust grid-search fit, avoids SciPy dependency.
    Fits X(t) = 1 - exp[-k (t-t0)^n] for t > t0.
    """
    t = np.asarray(t, float)
    X = np.asarray(X, float)

    ok = np.isfinite(t) & np.isfinite(X)
    t = t[ok]
    X = np.clip(X[ok], 0, 1)

    if len(t) < 10:
        raise ValueError("Not enough valid points for Avrami fit.")

    # practical onset guesses from the curve itself
    t01 = infer_tfrac(t, X, 0.01)
    t05 = infer_tfrac(t, X, 0.05)
    t10 = infer_tfrac(t, X, 0.10)
    t50 = infer_tfrac(t, X, 0.50)

    # fallback if early fractions are unstable
    if not np.isfinite(t05):
        t05 = np.nanmin(t)
    if not np.isfinite(t10):
        t10 = np.nanmin(t)
    if not np.isfinite(t50):
        t50 = np.nanmedian(t)

    # search ranges
    t0_min = max(np.nanmin(t), t05 - 5.0)
    t0_max = min(t10, t50)
    if not np.isfinite(t0_min) or not np.isfinite(t0_max) or t0_max <= t0_min:
        t0_min = np.nanmin(t)
        t0_max = np.nanpercentile(t, 20)

    t0_grid = np.linspace(t0_min, t0_max, 40)
    n_grid = np.linspace(0.8, 6.0, 70)
    k_grid = np.logspace(-5, 2, 100)

    best = None
    best_sse = np.inf

    for t0 in t0_grid:
        tau = np.clip(t - t0, 0, None)
        for n in n_grid:
            # vectorized over k
            tau_n = np.power(tau, n)
            # shape: (len(k), len(t))
            pred = 1.0 - np.exp(-np.outer(k_grid, tau_n))
            sse = np.sum((pred - X[None, :]) ** 2, axis=1)
            idx = int(np.argmin(sse))
            if sse[idx] < best_sse:
                best_sse = float(sse[idx])
                best = {
                    "t0": float(t0),
                    "k": float(k_grid[idx]),
                    "n": float(n),
                }

    if best is None:
        raise RuntimeError("Avrami fit failed.")

    X_fit = avrami_X(t, best["t0"], best["k"], best["n"])
    dXdt_fit = avrami_dXdt(t, best["t0"], best["k"], best["n"])

    rmse = float(np.sqrt(np.mean((X_fit - X) ** 2)))
    ss_res = float(np.sum((X - X_fit) ** 2))
    ss_tot = float(np.sum((X - np.mean(X)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    best["rmse_X"] = rmse
    best["r2_X"] = r2
    best["X_fit"] = X_fit
    best["dXdt_fit"] = dXdt_fit
    return best


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
    parser.add_argument("--curves", type=Path, required=True,
                        help="Path to reconstructed_curves_with_veff.csv")
    parser.add_argument("--out", type=Path, default=Path("transported_avrami_check.png"))
    parser.add_argument("--params_out", type=Path, default=Path("transported_avrami_fit_params.csv"))
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

        fit = fit_avrami_grid(t, X)
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
            }
        )

    params_df = pd.DataFrame(fit_rows)
    params_df.to_csv(args.params_out, index=False)

    cm_to_in = 1.0 / 2.54
    fig, axes = plt.subplots(
        1, 2,
        figsize=(args.fig_width_cm * cm_to_in, args.fig_height_cm * cm_to_in),
        dpi=args.dpi,
        facecolor="white",
    )

    ax1, ax2 = axes

    for sample in ["FAPI", "FAPI-TEMPO"]:
        sub = curves[sample]
        fit = fit_store[sample]
        t = sub["time_ms"].to_numpy(float)

        # transported
        ax1.plot(t, sub["X"], linewidth=2.4, label=f"{sample} transported")
        ax2.plot(t, sub["dX_dt"], linewidth=2.4, label=f"{sample} transported")

        # avrami fit
        ax1.plot(t, fit["X_fit"], linestyle="--", linewidth=2.0, label=f"{sample} Avrami fit")
        ax2.plot(t, fit["dXdt_fit"], linestyle="--", linewidth=2.0, label=f"{sample} Avrami fit")

    ax1.set_title("Transported X(t) with Avrami fits", fontsize=13)
    ax1.set_xlabel("Effective time (ms)")
    ax1.set_ylabel("Transformed fraction")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(frameon=False, fontsize=8)

    ax2.set_title("Transported dX/dt(t) with Avrami-fit derivatives", fontsize=13)
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
    print("\nFit parameters:")
    print(params_df.to_string(index=False))


if __name__ == "__main__":
    main()