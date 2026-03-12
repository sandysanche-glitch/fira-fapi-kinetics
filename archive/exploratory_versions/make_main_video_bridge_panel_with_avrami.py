#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit


# -----------------------------------
# generic helpers
# -----------------------------------
def robust_num(s):
    return pd.to_numeric(s, errors="coerce")


def moving_average(y: np.ndarray, win: int = 5) -> np.ndarray:
    y = np.asarray(y, float)
    if win <= 1:
        return y.copy()
    pad = win // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=float) / float(win)
    return np.convolve(ypad, kernel, mode="valid")


def derivative(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if len(t) < 2:
        return np.full_like(y, np.nan, dtype=float)
    return np.gradient(y, t)


# -----------------------------------
# video side
# -----------------------------------
def infer_video_sample_col(df: pd.DataFrame) -> str:
    for c in ["sample", "dataset", "condition", "group"]:
        if c in df.columns:
            return c
    raise ValueError(
        "Could not find a sample column in counts_shifted.csv. "
        "Expected one of: sample, dataset, condition, group."
    )


def canonical_sample_name(x: str) -> str:
    s = str(x).upper().replace("_", "-")
    if "TEMPO" in s:
        return "FAPI-TEMPO"
    if "FAPI" in s:
        return "FAPI"
    return str(x)


def normalize_video_sample(g: pd.DataFrame, smooth_win: int = 5) -> pd.DataFrame:
    g = g.copy()

    time_col = "t_shifted_ms" if "t_shifted_ms" in g.columns else "time_ms"
    x_col = "max_bbox_frac_kept"

    if x_col not in g.columns:
        raise ValueError(f"Expected video X column '{x_col}' in counts_shifted.csv")

    g[time_col] = robust_num(g[time_col])
    g[x_col] = robust_num(g[x_col])
    g = g[g[time_col].notna()].sort_values(time_col)

    # final agreed workflow: floor at 0 ms
    g = g[g[time_col] >= 0].copy()

    t = g[time_col].to_numpy(float)
    x_raw = g[x_col].to_numpy(float)

    valid = x_raw[np.isfinite(x_raw)]
    p99 = np.nanpercentile(valid, 99) if len(valid) else 1.0
    if not np.isfinite(p99) or p99 <= 0:
        p99 = np.nanmax(valid) if len(valid) else 1.0
    if not np.isfinite(p99) or p99 <= 0:
        p99 = 1.0

    x_norm = np.clip(x_raw / p99, 0, 1)
    x_smooth = moving_average(x_norm, smooth_win)
    dxdt = derivative(t, x_smooth)

    return pd.DataFrame(
        {
            "time_ms": t,
            "X_norm": x_norm,
            "X_smooth": x_smooth,
            "dXdt": dxdt,
        }
    )


def load_video_curves(counts_csv: Path, smooth_win: int = 5) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(counts_csv)
    sample_col = infer_video_sample_col(df)

    curves = {}
    for sval, g in df.groupby(sample_col):
        name = canonical_sample_name(sval)
        curves[name] = normalize_video_sample(g, smooth_win=smooth_win)

    needed = {"FAPI", "FAPI-TEMPO"}
    missing = needed.difference(curves.keys())
    if missing:
        raise ValueError(f"Missing video samples in counts_shifted.csv: {sorted(missing)}")

    return curves


# -----------------------------------
# transported side
# -----------------------------------
def load_bridge_curves(curves_csv: Path, weighting: str = "area_weighted") -> dict[str, pd.DataFrame]:
    df = pd.read_csv(curves_csv)

    required = {"time_ms", "X", "dX_dt", "sample", "weighting"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Bridge curves CSV missing columns: {sorted(missing)}")

    out = {}
    for sample in ["FAPI", "FAPI-TEMPO"]:
        sub = df[(df["sample"] == sample) & (df["weighting"] == weighting)].copy()
        sub = sub.sort_values("time_ms")
        if len(sub) == 0:
            raise ValueError(f"No bridge curves found for sample={sample}, weighting={weighting}")
        out[sample] = sub
    return out


# -----------------------------------
# continuous Avrami fit
# -----------------------------------
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

    return {
        "t0": t0,
        "k": k,
        "n": n,
        "X_fit": X_fit,
        "dXdt_fit": dXdt_fit,
    }


# -----------------------------------
# plot styling
# -----------------------------------
def style_axis(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


def annotate_video_peak(ax, x_peak: float, label: str, color: str, text_xy: tuple[float, float]):
    ax.axvline(x_peak, linestyle="--", linewidth=1.5, color=color, alpha=0.95)
    ax.annotate(
        label,
        xy=(x_peak, 0),
        xycoords=("data", "axes fraction"),
        xytext=text_xy,
        textcoords="axes fraction",
        ha="left",
        va="top",
        fontsize=10,
        color=color,
        arrowprops=dict(arrowstyle="-", lw=1.0, color=color),
    )


# -----------------------------------
# main
# -----------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts_shifted", type=Path, required=True)
    parser.add_argument("--bridge_curves", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("main_video_bridge_panel_with_avrami.png"))
    parser.add_argument("--video_smooth_win", type=int, default=5)
    parser.add_argument("--bridge_weighting", type=str, default="area_weighted",
                        choices=["count_weighted", "area_weighted", "R2_weighted"])
    parser.add_argument("--fig_width_cm", type=float, default=46.7)
    parser.add_argument("--fig_height_cm", type=float, default=19.9)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    # agreed video burst peaks
    fapi_peak_ms = 46.0
    tempo_peak_ms = 170.0

    video = load_video_curves(args.counts_shifted, smooth_win=args.video_smooth_win)
    bridge = load_bridge_curves(args.bridge_curves, weighting=args.bridge_weighting)

    # fit Avrami to transported X(t)
    avrami = {}
    for sample in ["FAPI", "FAPI-TEMPO"]:
        t = bridge[sample]["time_ms"].to_numpy(float)
        X = bridge[sample]["X"].to_numpy(float)
        avrami[sample] = fit_avrami_continuous(t, X)

    cm_to_in = 1.0 / 2.54
    fig = plt.figure(
        figsize=(args.fig_width_cm * cm_to_in, args.fig_height_cm * cm_to_in),
        dpi=args.dpi,
        facecolor="white",
    )

    # ~40% more separated middle gutter and narrower both blocks
    gs = GridSpec(
        2, 3,
        figure=fig,
        width_ratios=[0.92, 0.48, 0.84],
        height_ratios=[1.0, 1.0],
        wspace=0.0,
        hspace=0.03,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax_a)
    ax_mid = fig.add_subplot(gs[:, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 2], sharex=ax_c)

    # blank middle gutter
    ax_mid.set_facecolor("white")
    ax_mid.set_xticks([])
    ax_mid.set_yticks([])
    for s in ax_mid.spines.values():
        s.set_visible(False)

    # colors
    c_fapi = "tab:blue"
    c_tempo = "tab:orange"

    # -------- left block: measured/video --------
    ax_a.plot(video["FAPI"]["time_ms"], video["FAPI"]["X_smooth"], color=c_fapi, linewidth=2.4, label="FAPI")
    ax_a.plot(video["FAPI-TEMPO"]["time_ms"], video["FAPI-TEMPO"]["X_smooth"], color=c_tempo, linewidth=2.4, label="FAPI-TEMPO")
    ax_a.set_ylabel("X (normalized)")
    style_axis(ax_a)
    ax_a.legend(frameon=False, loc="lower right", fontsize=11)

    # subtle peak guides in top video panel
    ax_a.axvline(fapi_peak_ms, linestyle="--", linewidth=1.0, alpha=0.22, color=c_fapi)
    ax_a.axvline(tempo_peak_ms, linestyle="--", linewidth=1.0, alpha=0.22, color=c_tempo)

    ax_b.plot(video["FAPI"]["time_ms"], video["FAPI"]["dXdt"], color=c_fapi, linewidth=2.4, label="FAPI")
    ax_b.plot(video["FAPI-TEMPO"]["time_ms"], video["FAPI-TEMPO"]["dXdt"], color=c_tempo, linewidth=2.4, label="FAPI-TEMPO")
    ax_b.set_xlabel("Time (ms)")
    ax_b.set_ylabel("dX/dt (per ms)")
    style_axis(ax_b)

    annotate_video_peak(
        ax_b, fapi_peak_ms,
        "FAPI peak\n~46 ms",
        color=c_fapi,
        text_xy=(0.08, 0.94),
    )
    annotate_video_peak(
        ax_b, tempo_peak_ms,
        "FAPI-TEMPO peak\n~170 ms",
        color=c_tempo,
        text_xy=(0.56, 0.94),
    )

    # -------- right block: transported + Avrami --------
    t_f = bridge["FAPI"]["time_ms"].to_numpy(float)
    t_t = bridge["FAPI-TEMPO"]["time_ms"].to_numpy(float)

    ax_c.plot(t_f, bridge["FAPI"]["X"], color=c_fapi, linewidth=2.4, label="FAPI")
    ax_c.plot(t_f, avrami["FAPI"]["X_fit"], color=c_fapi, linestyle="--", linewidth=2.0, alpha=0.95, label="FAPI Avrami")
    ax_c.plot(t_t, bridge["FAPI-TEMPO"]["X"], color=c_tempo, linewidth=2.4, label="FAPI-TEMPO")
    ax_c.plot(t_t, avrami["FAPI-TEMPO"]["X_fit"], color=c_tempo, linestyle="--", linewidth=2.0, alpha=0.95, label="FAPI-TEMPO Avrami")
    ax_c.set_ylabel("Transformed fraction")
    style_axis(ax_c)
    ax_c.legend(frameon=False, loc="lower right", fontsize=10)

    ax_d.plot(t_f, bridge["FAPI"]["dX_dt"], color=c_fapi, linewidth=2.4, label="FAPI")
    ax_d.plot(t_f, avrami["FAPI"]["dXdt_fit"], color=c_fapi, linestyle="--", linewidth=2.0, alpha=0.95, label="FAPI Avrami")
    ax_d.plot(t_t, bridge["FAPI-TEMPO"]["dX_dt"], color=c_tempo, linewidth=2.4, label="FAPI-TEMPO")
    ax_d.plot(t_t, avrami["FAPI-TEMPO"]["dXdt_fit"], color=c_tempo, linestyle="--", linewidth=2.0, alpha=0.95, label="FAPI-TEMPO Avrami")
    ax_d.set_xlabel("Effective time (ms)")
    ax_d.set_ylabel("Transformation rate")
    style_axis(ax_d)

    # no a/b/c/d labels and no subplot titles
    plt.setp(ax_a.get_xticklabels(), visible=False)
    plt.setp(ax_c.get_xticklabels(), visible=False)

    # manual margins to keep the right block visually a bit left-shifted
    fig.subplots_adjust(left=0.055, right=0.985, top=0.965, bottom=0.11)

    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure to: {args.out.resolve()}")


if __name__ == "__main__":
    main()