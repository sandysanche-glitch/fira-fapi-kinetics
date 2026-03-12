#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrow


# -----------------------------
# helpers
# -----------------------------
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


# -----------------------------
# video curves
# -----------------------------
def normalize_video_sample(g: pd.DataFrame, smooth_win: int = 5) -> pd.DataFrame:
    g = g.copy()

    time_col = "t_shifted_ms" if "t_shifted_ms" in g.columns else "time_ms"
    x_col = "max_bbox_frac_kept"

    if x_col not in g.columns:
        raise ValueError(f"Expected video X column '{x_col}' in counts_shifted.csv")

    g[time_col] = robust_num(g[time_col])
    g[x_col] = robust_num(g[x_col])
    g = g[g[time_col].notna()].sort_values(time_col)

    # final agreed workflow: time floor at 0 ms
    g = g[g[time_col] >= 0].copy()

    t = g[time_col].to_numpy(float)
    x_raw = g[x_col].to_numpy(float)

    # final agreed workflow: normalize by p99, no running-max enforcement
    valid = x_raw[np.isfinite(x_raw)]
    p99 = np.nanpercentile(valid, 99) if len(valid) else 1.0
    if not np.isfinite(p99) or p99 <= 0:
        p99 = np.nanmax(valid) if len(valid) else 1.0
    if not np.isfinite(p99) or p99 <= 0:
        p99 = 1.0

    x_norm = np.clip(x_raw / p99, 0, 1)
    x_smooth = moving_average(x_norm, smooth_win)
    dxdt = derivative(t, x_smooth)

    out = pd.DataFrame(
        {
            "time_ms": t,
            "X_norm": x_norm,
            "X_smooth": x_smooth,
            "dXdt": dxdt,
        }
    )
    return out


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


# -----------------------------
# bridge curves
# -----------------------------
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


# -----------------------------
# plotting style
# -----------------------------
def style_axis(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


def add_panel_label(ax, label: str):
    ax.text(
        0.015, 0.98, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=13
    )


def annotate_video_peak(ax, x_peak: float, label: str, color: str, text_xy: tuple[float, float]):
    ax.axvline(x_peak, linestyle="--", linewidth=1.6, color=color, alpha=0.95)
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
        arrowprops=dict(arrowstyle="-", lw=1.2, color=color),
    )


# -----------------------------
# main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts_shifted", type=Path, required=True,
                        help="Path to counts_shifted.csv for video anchor")
    parser.add_argument("--bridge_curves", type=Path, required=True,
                        help="Path to reconstructed_curves_with_veff.csv")
    parser.add_argument("--out", type=Path, default=Path("main_video_bridge_panel_final.png"))
    parser.add_argument("--video_smooth_win", type=int, default=5)
    parser.add_argument("--bridge_weighting", type=str, default="area_weighted",
                        choices=["count_weighted", "area_weighted", "R2_weighted"])
    parser.add_argument("--fig_width_cm", type=float, default=46.7)
    parser.add_argument("--fig_height_cm", type=float, default=19.9)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--transport_band_color", type=str, default="#4A76C2")
    args = parser.parse_args()

    # agreed burst peak locations from finalized video metrics
    fapi_peak_ms = 46.0
    tempo_peak_ms = 170.0

    video = load_video_curves(args.counts_shifted, smooth_win=args.video_smooth_win)
    bridge = load_bridge_curves(args.bridge_curves, weighting=args.bridge_weighting)

    cm_to_in = 1.0 / 2.54
    fig = plt.figure(
        figsize=(args.fig_width_cm * cm_to_in, args.fig_height_cm * cm_to_in),
        dpi=args.dpi,
        facecolor="white",
    )

    gs = GridSpec(
        2, 3,
        figure=fig,
        width_ratios=[1.0, 0.23, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.0,
        hspace=0.0,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax_a)
    ax_mid = fig.add_subplot(gs[:, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 2], sharex=ax_c)

    # ---------------- left: video anchor ----------------
    ax_a.plot(video["FAPI"]["time_ms"], video["FAPI"]["X_smooth"], linewidth=2.5, label="FAPI")
    ax_a.plot(video["FAPI-TEMPO"]["time_ms"], video["FAPI-TEMPO"]["X_smooth"], linewidth=2.5, label="FAPI-TEMPO")
    ax_a.set_ylabel("X (normalized)")
    ax_a.set_title("Video X(t)", fontsize=15)
    style_axis(ax_a)
    add_panel_label(ax_a, "a)")
    ax_a.legend(frameon=False, loc="lower right", fontsize=11)

    # optional subtle timing markers in X panel
    ax_a.axvline(fapi_peak_ms, linestyle="--", linewidth=1.2, alpha=0.35)
    ax_a.axvline(tempo_peak_ms, linestyle="--", linewidth=1.2, alpha=0.35)

    ax_b.plot(video["FAPI"]["time_ms"], video["FAPI"]["dXdt"], linewidth=2.5, label="FAPI")
    ax_b.plot(video["FAPI-TEMPO"]["time_ms"], video["FAPI-TEMPO"]["dXdt"], linewidth=2.5, label="FAPI-TEMPO")
    ax_b.set_xlabel("Time (ms)")
    ax_b.set_ylabel("dX/dt (per ms)")
    ax_b.set_title("Video dX/dt", fontsize=15)
    style_axis(ax_b)
    add_panel_label(ax_b, "b)")

    # automatic peak annotations using agreed peak times
    annotate_video_peak(
        ax_b, fapi_peak_ms,
        "FAPI peak\n~46 ms",
        color="tab:blue",
        text_xy=(0.10, 0.94),
    )
    annotate_video_peak(
        ax_b, tempo_peak_ms,
        "FAPI-TEMPO peak\n~170 ms",
        color="tab:orange",
        text_xy=(0.56, 0.94),
    )

    # ---------------- middle transport band ----------------
    ax_mid.set_facecolor(args.transport_band_color)
    ax_mid.set_xticks([])
    ax_mid.set_yticks([])
    for s in ax_mid.spines.values():
        s.set_visible(False)

    ax_mid.text(
        0.5, 0.5, "Morphology transported",
        ha="center",
        va="center",
        rotation=90,
        color="white",
        fontsize=16,
        fontweight="bold",
        transform=ax_mid.transAxes,
    )

    arr1 = FancyArrow(
        0.28, 0.74, 0.38, 0.0,
        width=0.020, head_width=0.085, head_length=0.13,
        length_includes_head=True,
        facecolor="white", edgecolor="#264B8A", linewidth=0.9,
        transform=ax_mid.transAxes,
    )
    arr2 = FancyArrow(
        0.28, 0.26, 0.38, 0.0,
        width=0.020, head_width=0.085, head_length=0.13,
        length_includes_head=True,
        facecolor="white", edgecolor="#264B8A", linewidth=0.9,
        transform=ax_mid.transAxes,
    )
    ax_mid.add_patch(arr1)
    ax_mid.add_patch(arr2)

    # ---------------- right: bridge curves ----------------
    ax_c.plot(bridge["FAPI"]["time_ms"], bridge["FAPI"]["X"], linewidth=2.5, label="FAPI")
    ax_c.plot(bridge["FAPI-TEMPO"]["time_ms"], bridge["FAPI-TEMPO"]["X"], linewidth=2.5, label="FAPI-TEMPO")
    ax_c.set_ylabel("Transformed fraction")
    ax_c.set_title("Bridge X(t)", fontsize=15)
    style_axis(ax_c)
    add_panel_label(ax_c, "c)")
    ax_c.legend(frameon=False, loc="lower right", fontsize=11)

    ax_d.plot(bridge["FAPI"]["time_ms"], bridge["FAPI"]["dX_dt"], linewidth=2.5, label="FAPI")
    ax_d.plot(bridge["FAPI-TEMPO"]["time_ms"], bridge["FAPI-TEMPO"]["dX_dt"], linewidth=2.5, label="FAPI-TEMPO")
    ax_d.set_xlabel("Effective time (ms)")
    ax_d.set_ylabel("Transformation rate")
    ax_d.set_title("Bridge dX/dt(t)", fontsize=15)
    style_axis(ax_d)
    add_panel_label(ax_d, "d)")

    # tidy top row x tick labels
    plt.setp(ax_a.get_xticklabels(), visible=False)
    plt.setp(ax_c.get_xticklabels(), visible=False)

    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure to: {args.out.resolve()}")


if __name__ == "__main__":
    main()