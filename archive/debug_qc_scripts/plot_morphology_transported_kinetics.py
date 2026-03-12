#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def robust_num(s):
    return pd.to_numeric(s, errors="coerce")


def smooth(y, sigma_pts=2.0):
    y = np.asarray(y, float)
    radius = max(1, int(round(4 * sigma_pts)))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-(x**2) / (2 * sigma_pts**2))
    k /= k.sum()
    return np.convolve(y, k, mode="same")


def parse_file_name(name: str, sample: str):
    s = str(name)
    nums = re.findall(r"(\d+)", s)
    mg = nums[0] if len(nums) >= 1 else s
    roi = nums[1] if len(nums) >= 2 else "0"
    return f"{sample}_MG_{mg}", f"{sample}_ROI_{mg}_{roi}"


def prep(path: Path, sample: str) -> pd.DataFrame:
    raw = pd.read_csv(path)

    out = pd.DataFrame(index=raw.index)
    out["sample"] = sample
    out["file_name"] = raw["file_name"].astype(str)

    parsed = out["file_name"].apply(lambda x: parse_file_name(x, sample))
    out["micrograph_id"] = [a for a, b in parsed]
    out["roi_id"] = [b for a, b in parsed]

    colmap = {
        "area_(µm²)": "area_um2",
        "perimeter_(µm)": "perimeter_um",
        "circularity_distortion": "circularity_distortion",
        "defects_count": "defects_count",
        "defects_area_(µm²)": "defects_area_um2",
        "entropy(bits)": "entropy_bits",
        "entropy_norm(bits)": "entropy_norm_bits",
        "entropy_hm_(bits)": "entropy_hm_bits",
        "nucleus_circularity": "nucleus_circularity",
        "rank_for_t0": "rank_for_t0",
        "t0_ms": "t0_ms",
        "dt_ms": "dt_ms",
        "R_um_final": "R_um_final",
        "v_eff_um_per_ms": "v_eff_um_per_ms",
    }

    for src, dst in colmap.items():
        out[dst] = robust_num(raw[src]) if src in raw.columns else np.nan

    out["t0_ms"] = out["t0_ms"].clip(lower=0)
    out["count_weight"] = 1.0
    out["area_weight"] = out["area_um2"]
    out["R2_weight"] = out["R_um_final"] ** 2
    return out


def reconstruct(gs: pd.DataFrame, weight_col: str, n_grid: int = 500) -> pd.DataFrame:
    g = gs[["t0_ms", weight_col]].dropna().sort_values("t0_ms")
    g = g[g[weight_col] >= 0]

    if len(g) < 2:
        raise ValueError(f"Not enough rows to reconstruct using {weight_col}")

    t = g["t0_ms"].to_numpy(float)
    w = g[weight_col].to_numpy(float)

    t_grid = np.linspace(t.min(), t.max(), n_grid)
    n = np.searchsorted(t, t_grid, side="right").astype(float)

    cws = np.cumsum(w)
    idx = np.searchsorted(t, t_grid, side="right") - 1
    x_num = np.where(idx >= 0, cws[np.clip(idx, 0, len(cws) - 1)], 0.0)
    X = x_num / w.sum() if w.sum() > 0 else np.full_like(t_grid, np.nan)

    dn = smooth(np.gradient(n, t_grid), 2.0)
    dX = smooth(np.gradient(X, t_grid), 2.0)

    return pd.DataFrame(
        {
            "time_ms": t_grid,
            "n": n,
            "dn_dt": dn,
            "X": X,
            "dX_dt": dX,
        }
    )


def build_curves(fapi_path: Path, tempo_path: Path) -> pd.DataFrame:
    fapi = prep(fapi_path, "FAPI")
    tempo = prep(tempo_path, "FAPI-TEMPO")
    df = pd.concat([fapi, tempo], ignore_index=True)

    curve_list = []
    for sample in ["FAPI", "FAPI-TEMPO"]:
        gs = df[df["sample"] == sample].copy()
        for label, wcol in [
            ("count_weighted", "count_weight"),
            ("area_weighted", "area_weight"),
            ("R2_weighted", "R2_weight"),
        ]:
            c = reconstruct(gs, wcol)
            c["sample"] = sample
            c["weighting"] = label
            curve_list.append(c)

    return pd.concat(curve_list, ignore_index=True)


def save_single_metric_plot(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    outpath: Path,
    title: str,
    weightings: list[str],
    samples: list[str],
):
    fig, ax = plt.subplots(figsize=(6.5, 4.4))

    for weighting in weightings:
        for sample in samples:
            sub = df[(df["weighting"] == weighting) & (df["sample"] == sample)].sort_values("time_ms")
            if len(sub) == 0:
                continue
            ax.plot(sub["time_ms"], sub[metric], label=f"{sample} | {weighting}", linewidth=2)

    ax.set_xlabel("Effective time (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_primary_2x2_panel(df: pd.DataFrame, outpath: Path, primary_weighting: str, samples: list[str]):
    metrics = [
        ("n", "n(t)", "Count"),
        ("dn_dt", "dn/dt(t)", "Rate"),
        ("X", "X(t)", "Transformed fraction"),
        ("dX_dt", "dX/dt(t)", "Transformation rate"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2))
    axes = axes.flatten()

    for ax, (metric, title, ylabel) in zip(axes, metrics):
        for sample in samples:
            sub = df[(df["weighting"] == primary_weighting) & (df["sample"] == sample)].sort_values("time_ms")
            if len(sub) == 0:
                continue
            ax.plot(sub["time_ms"], sub[metric], label=sample, linewidth=2)

        ax.set_title(title)
        ax.set_xlabel("Effective time (ms)")
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(samples), frameon=False)

    fig.suptitle(f"Morphology-transported kinetics ({primary_weighting})", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_weighting_comparison_panel(
    df: pd.DataFrame,
    outpath: Path,
    metric: str,
    ylabel: str,
    samples: list[str],
    weightings: list[str],
):
    fig, axes = plt.subplots(1, len(samples), figsize=(10.0, 4.0), sharey=False)
    if len(samples) == 1:
        axes = [axes]

    for ax, sample in zip(axes, samples):
        for weighting in weightings:
            sub = df[(df["sample"] == sample) & (df["weighting"] == weighting)].sort_values("time_ms")
            if len(sub) == 0:
                continue
            ax.plot(sub["time_ms"], sub[metric], label=weighting, linewidth=2)

        ax.set_title(sample)
        ax.set_xlabel("Effective time (ms)")
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(weightings), frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fapi", type=Path, required=True, help="Path to morpho_kinetics_from_cm_full_FAPI_with_veff.csv")
    parser.add_argument("--tempo", type=Path, required=True, help="Path to morpho_kinetics_from_cm_full_FAPITEMPO_with_veff.csv")
    parser.add_argument("--outdir", type=Path, default=Path("qc_plots"))
    parser.add_argument(
        "--primary_weighting",
        type=str,
        default="area_weighted",
        choices=["count_weighted", "area_weighted", "R2_weighted"],
    )
    parser.add_argument("--save_curves_csv", action="store_true", help="Also save reconstructed_curves_with_veff.csv")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    curves = build_curves(args.fapi, args.tempo)
    samples = list(curves["sample"].dropna().unique())
    weightings = ["count_weighted", "area_weighted", "R2_weighted"]

    if args.save_curves_csv:
        curves.to_csv(args.outdir / "reconstructed_curves_with_veff.csv", index=False)

    save_primary_2x2_panel(
        df=curves,
        outpath=args.outdir / f"main_panel_{args.primary_weighting}.png",
        primary_weighting=args.primary_weighting,
        samples=samples,
    )

    save_single_metric_plot(
        df=curves,
        metric="n",
        ylabel="Count",
        outpath=args.outdir / "n_t_all_weightings.png",
        title="Morphology-transported n(t)",
        weightings=weightings,
        samples=samples,
    )
    save_single_metric_plot(
        df=curves,
        metric="dn_dt",
        ylabel="Rate",
        outpath=args.outdir / "dn_dt_all_weightings.png",
        title="Morphology-transported dn/dt(t)",
        weightings=weightings,
        samples=samples,
    )
    save_single_metric_plot(
        df=curves,
        metric="X",
        ylabel="Transformed fraction",
        outpath=args.outdir / "X_t_all_weightings.png",
        title="Morphology-transported X(t)",
        weightings=weightings,
        samples=samples,
    )
    save_single_metric_plot(
        df=curves,
        metric="dX_dt",
        ylabel="Transformation rate",
        outpath=args.outdir / "dX_dt_all_weightings.png",
        title="Morphology-transported dX/dt(t)",
        weightings=weightings,
        samples=samples,
    )

    save_weighting_comparison_panel(
        df=curves,
        outpath=args.outdir / "SI_weighting_comparison_X_t.png",
        metric="X",
        ylabel="Transformed fraction",
        samples=samples,
        weightings=weightings,
    )
    save_weighting_comparison_panel(
        df=curves,
        outpath=args.outdir / "SI_weighting_comparison_dX_dt.png",
        metric="dX_dt",
        ylabel="Transformation rate",
        samples=samples,
        weightings=weightings,
    )

    print(f"Plots written to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()