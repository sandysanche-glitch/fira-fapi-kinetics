#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit


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


def infer_video_sample_col(df: pd.DataFrame) -> str:
    for c in ["sample", "dataset", "condition", "group"]:
        if c in df.columns:
            return c
    raise ValueError("Could not find a sample column in counts_shifted.csv.")


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

    g[time_col] = robust_num(g[time_col])
    g[x_col] = robust_num(g[x_col])
    g = g[g[time_col].notna()].sort_values(time_col)
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

    return pd.DataFrame({"time_ms": t, "X_smooth": x_smooth, "dXdt": dxdt})


def load_video_curves(counts_csv: Path, smooth_win: int = 5) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(counts_csv)
    sample_col = infer_video_sample_col(df)

    curves = {}
    for sval, g in df.groupby(sample_col):
        curves[canonical_sample_name(sval)] = normalize_video_sample(g, smooth_win=smooth_win)
    return curves


def load_bridge_curves(curves_csv: Path, weighting: str = "area_weighted") -> dict[str, pd.DataFrame]:
    df = pd.read_csv(curves_csv)
    out = {}
    for sample in ["FAPI", "FAPI-TEMPO"]:
        sub = df[(df["sample"] == sample) & (df["weighting"] == weighting)].copy()
        out[sample] = sub.sort_values("time_ms")
    return out


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
    t05 = infer_tfrac(t, X, 0.05)
    t10 = infer_tfrac(t, X, 0.10)
    t50 = infer_tfrac(t, X, 0.50)
    t90 = infer_tfrac(t, X, 0.90)

    t0_guess = max(0.0, min(t05, t10) - 0.5)
    span = max(t90 - t10, 1e-6)
    p0 = [t0_guess, 1.0 / span, 1.0]

    popt, _ = curve_fit(
        avrami_X, t, X, p0=p0,
        bounds=([0.0, 1e-8, 0.2], [max(t50, 5.0), 1e3, 8.0]),
        maxfev=200000
    )
    t0, k, n = [float(v) for v in popt]
    return {"X_fit": avrami_X(t, t0, k, n), "dXdt_fit": avrami_dXdt(t, t0, k, n)}


def style_axis(ax, ticksize: float):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", labelsize=ticksize)


def style_legend(ax, fontsize: float, loc: str):
    ax.legend(
        frameon=False,
        loc=loc,
        fontsize=fontsize,
        handlelength=2.2,
        handletextpad=0.7,
        labelspacing=0.45,
        borderaxespad=0.2,
    )


def annotate_video_peak(ax, x_peak: float, label: str, color: str, text_xy: tuple[float, float], fontsize: float):
    ax.axvline(x_peak, linestyle="--", linewidth=1.5, color=color, alpha=0.95)
    ax.annotate(
        label,
        xy=(x_peak, 0),
        xycoords=("data", "axes fraction"),
        xytext=text_xy,
        textcoords="axes fraction",
        ha="left",
        va="top",
        fontsize=fontsize,
        color=color,
        arrowprops=dict(arrowstyle="-", lw=1.0, color=color),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts_shifted", type=Path, required=True)
    parser.add_argument("--bridge_curves", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("main_video_bridge_panel_with_avrami_fonts_v2.png"))
    parser.add_argument("--video_smooth_win", type=int, default=5)
    parser.add_argument("--bridge_weighting", type=str, default="area_weighted")
    parser.add_argument("--fig_width_cm", type=float, default=46.7)
    parser.add_argument("--fig_height_cm", type=float, default=19.9)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    # ~5% lower than previous version
    label_fs = 15.3
    tick_fs = 13.1
    legend_fs = 12.1
    annot_fs = 10.9

    fapi_peak_ms = 46.0
    tempo_peak_ms = 170.0

    video = load_video_curves(args.counts_shifted, smooth_win=args.video_smooth_win)
    bridge = load_bridge_curves(args.bridge_curves, weighting=args.bridge_weighting)

    avrami = {}
    for sample in ["FAPI", "FAPI-TEMPO"]:
        t = bridge[sample]["time_ms"].to_numpy(float)
        X = bridge[sample]["X"].to_numpy(float)
        avrami[sample] = fit_avrami_continuous(t, X)

    cm_to_in = 1.0 / 2.54
    fig = plt.figure(figsize=(args.fig_width_cm * cm_to_in, args.fig_height_cm * cm_to_in), dpi=args.dpi, facecolor="white")

    gs = GridSpec(
        2, 3, figure=fig,
        width_ratios=[0.92, 0.48, 0.84],
        height_ratios=[1.0, 1.0],
        wspace=0.0, hspace=0.03
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax_a)
    ax_mid = fig.add_subplot(gs[:, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 2], sharex=ax_c)

    ax_mid.set_facecolor("white")
    ax_mid.set_xticks([])
    ax_mid.set_yticks([])
    for s in ax_mid.spines.values():
        s.set_visible(False)

    c_fapi = "tab:blue"
    c_tempo = "tab:orange"

    ax_a.plot(video["FAPI"]["time_ms"], video["FAPI"]["X_smooth"], color=c_fapi, linewidth=2.4, label="FAPI")
    ax_a.plot(video["FAPI-TEMPO"]["time_ms"], video["FAPI-TEMPO"]["X_smooth"], color=c_tempo, linewidth=2.4, label="FAPI-TEMPO")
    ax_a.set_ylabel("X (normalized)", fontsize=label_fs)
    style_axis(ax_a, tick_fs)
    style_legend(ax_a, legend_fs, "lower right")
    ax_a.axvline(fapi_peak_ms, linestyle="--", linewidth=1.0, alpha=0.22, color=c_fapi)
    ax_a.axvline(tempo_peak_ms, linestyle="--", linewidth=1.0, alpha=0.22, color=c_tempo)

    ax_b.plot(video["FAPI"]["time_ms"], video["FAPI"]["dXdt"], color=c_fapi, linewidth=2.4)
    ax_b.plot(video["FAPI-TEMPO"]["time_ms"], video["FAPI-TEMPO"]["dXdt"], color=c_tempo, linewidth=2.4)
    ax_b.set_xlabel("Time (ms)", fontsize=label_fs)
    ax_b.set_ylabel("dX/dt (per ms)", fontsize=label_fs)
    style_axis(ax_b, tick_fs)

    annotate_video_peak(ax_b, fapi_peak_ms, "FAPI peak\n~46 ms", c_fapi, (0.08, 0.94), annot_fs)
    annotate_video_peak(ax_b, tempo_peak_ms, "FAPI-TEMPO peak\n~170 ms", c_tempo, (0.56, 0.94), annot_fs)

    t_f = bridge["FAPI"]["time_ms"].to_numpy(float)
    t_t = bridge["FAPI-TEMPO"]["time_ms"].to_numpy(float)

    ax_c.plot(t_f, bridge["FAPI"]["X"], color=c_fapi, linewidth=2.4, label="FAPI")
    ax_c.plot(t_f, avrami["FAPI"]["X_fit"], color=c_fapi, linestyle="--", linewidth=2.0, alpha=0.95, label="FAPI Avrami")
    ax_c.plot(t_t, bridge["FAPI-TEMPO"]["X"], color=c_tempo, linewidth=2.4, label="FAPI-TEMPO")
    ax_c.plot(t_t, avrami["FAPI-TEMPO"]["X_fit"], color=c_tempo, linestyle="--", linewidth=2.0, alpha=0.95, label="FAPI-TEMPO Avrami")
    ax_c.set_ylabel("Transformed fraction", fontsize=label_fs)
    style_axis(ax_c, tick_fs)
    style_legend(ax_c, legend_fs, "lower right")

    ax_d.plot(t_f, bridge["FAPI"]["dX_dt"], color=c_fapi, linewidth=2.4)
    ax_d.plot(t_f, avrami["FAPI"]["dXdt_fit"], color=c_fapi, linestyle="--", linewidth=2.0, alpha=0.95)
    ax_d.plot(t_t, bridge["FAPI-TEMPO"]["dX_dt"], color=c_tempo, linewidth=2.4)
    ax_d.plot(t_t, avrami["FAPI-TEMPO"]["dXdt_fit"], color=c_tempo, linestyle="--", linewidth=2.0, alpha=0.95)
    ax_d.set_xlabel("Effective time (ms)", fontsize=label_fs)
    ax_d.set_ylabel("Transformation rate", fontsize=label_fs)
    style_axis(ax_d, tick_fs)

    plt.setp(ax_a.get_xticklabels(), visible=False)
    plt.setp(ax_c.get_xticklabels(), visible=False)

    fig.subplots_adjust(left=0.055, right=0.985, top=0.965, bottom=0.11)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure to: {args.out.resolve()}")


if __name__ == "__main__":
    main()