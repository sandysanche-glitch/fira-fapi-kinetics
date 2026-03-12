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
        "t0_ms": "t0_ms",
        "R_um_final": "R_um_final",
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

    t = g["t0_ms"].to_numpy(float)
    w = g[weight_col].to_numpy(float)

    t_grid = np.linspace(t.min(), t.max(), n_grid)

    cws = np.cumsum(w)
    idx = np.searchsorted(t, t_grid, side="right") - 1
    x_num = np.where(idx >= 0, cws[np.clip(idx, 0, len(cws) - 1)], 0.0)
    X = x_num / w.sum() if w.sum() > 0 else np.full_like(t_grid, np.nan)
    dX = smooth(np.gradient(X, t_grid), 2.0)

    return pd.DataFrame(
        {
            "time_ms": t_grid,
            "X": X,
            "dX_dt": dX,
        }
    )


def build_all_curves(fapi_path: Path, tempo_path: Path) -> pd.DataFrame:
    fapi = prep(fapi_path, "FAPI")
    tempo = prep(tempo_path, "FAPI-TEMPO")
    df = pd.concat([fapi, tempo], ignore_index=True)

    rows = []
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
            rows.append(c)
    return pd.concat(rows, ignore_index=True)


def make_panel(df: pd.DataFrame, metric: str, ylabel: str, outpath: Path):
    samples = ["FAPI", "FAPI-TEMPO"]
    weightings = ["count_weighted", "area_weighted", "R2_weighted"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=False)

    for ax, sample in zip(axes, samples):
        for weighting in weightings:
            sub = df[(df["sample"] == sample) & (df["weighting"] == weighting)].sort_values("time_ms")
            ax.plot(sub["time_ms"], sub[metric], label=weighting, linewidth=2.5)

        ax.set_title(sample)
        ax.set_xlabel("Effective time (ms)")
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fapi", type=Path, required=True)
    parser.add_argument("--tempo", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("SI_weighting_plots"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    curves = build_all_curves(args.fapi, args.tempo)

    make_panel(
        df=curves,
        metric="X",
        ylabel="Transformed fraction",
        outpath=args.outdir / "SI_weighting_comparison_X_t.png",
    )
    make_panel(
        df=curves,
        metric="dX_dt",
        ylabel="Transformation rate",
        outpath=args.outdir / "SI_weighting_comparison_dX_dt.png",
    )

    print(f"Saved SI plots to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()