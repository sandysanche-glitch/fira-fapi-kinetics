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
    out["area_weight"] = out["area_um2"]
    return out


def reconstruct_area_weighted(gs: pd.DataFrame, n_grid: int = 500) -> pd.DataFrame:
    g = gs[["t0_ms", "area_weight"]].dropna().sort_values("t0_ms")
    g = g[g["area_weight"] >= 0]

    t = g["t0_ms"].to_numpy(float)
    w = g["area_weight"].to_numpy(float)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fapi", type=Path, required=True)
    parser.add_argument("--tempo", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("main_morphology_transported_kinetics.png"))
    args = parser.parse_args()

    fapi = prep(args.fapi, "FAPI")
    tempo = prep(args.tempo, "FAPI-TEMPO")

    cf = reconstruct_area_weighted(fapi)
    ct = reconstruct_area_weighted(tempo)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    axes[0].plot(cf["time_ms"], cf["X"], label="FAPI", linewidth=2.5)
    axes[0].plot(ct["time_ms"], ct["X"], label="FAPI-TEMPO", linewidth=2.5)
    axes[0].set_title("Area-weighted X(t)")
    axes[0].set_xlabel("Effective time (ms)")
    axes[0].set_ylabel("Transformed fraction")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].plot(cf["time_ms"], cf["dX_dt"], label="FAPI", linewidth=2.5)
    axes[1].plot(ct["time_ms"], ct["dX_dt"], label="FAPI-TEMPO", linewidth=2.5)
    axes[1].set_title("Area-weighted dX/dt(t)")
    axes[1].set_xlabel("Effective time (ms)")
    axes[1].set_ylabel("Transformation rate")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Morphology-transported effective kinetics", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {args.out.resolve()}")


if __name__ == "__main__":
    main()