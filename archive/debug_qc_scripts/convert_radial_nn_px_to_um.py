#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_wide_radial_crowding_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = [
        "r_over_R",
        "FAPI_nn_median_px", "FAPI_nn_q25_px", "FAPI_nn_q75_px",
        "FAPITEMPO_nn_median_px", "FAPITEMPO_nn_q25_px", "FAPITEMPO_nn_q75_px",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns: {missing}\n"
            f"Columns found: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "r_over_R": pd.to_numeric(df["r_over_R"], errors="coerce"),

        "FAPI_nn_median_px": pd.to_numeric(df["FAPI_nn_median_px"], errors="coerce"),
        "FAPI_nn_q25_px": pd.to_numeric(df["FAPI_nn_q25_px"], errors="coerce"),
        "FAPI_nn_q75_px": pd.to_numeric(df["FAPI_nn_q75_px"], errors="coerce"),

        "FAPITEMPO_nn_median_px": pd.to_numeric(df["FAPITEMPO_nn_median_px"], errors="coerce"),
        "FAPITEMPO_nn_q25_px": pd.to_numeric(df["FAPITEMPO_nn_q25_px"], errors="coerce"),
        "FAPITEMPO_nn_q75_px": pd.to_numeric(df["FAPITEMPO_nn_q75_px"], errors="coerce"),
    })

    # keep impingement/QC if present
    for c in df.columns:
        if c not in out.columns:
            out[c] = df[c]

    out = out[np.isfinite(out["r_over_R"])].copy()
    return out


def convert_px_to_um(df: pd.DataFrame, um_per_px: float) -> pd.DataFrame:
    out = df.copy()

    for sample in ["FAPI", "FAPITEMPO"]:
        out[f"{sample}_nn_median_um"] = out[f"{sample}_nn_median_px"] * um_per_px
        out[f"{sample}_nn_q25_um"] = out[f"{sample}_nn_q25_px"] * um_per_px
        out[f"{sample}_nn_q75_um"] = out[f"{sample}_nn_q75_px"] * um_per_px

    return out


def plot_um(df: pd.DataFrame, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    ax.plot(df["r_over_R"], df["FAPI_nn_median_um"], lw=2.8, color="tab:blue", label="FAPI")
    ax.fill_between(
        df["r_over_R"], df["FAPI_nn_q25_um"], df["FAPI_nn_q75_um"],
        color="tab:blue", alpha=0.18
    )

    ax.plot(df["r_over_R"], df["FAPITEMPO_nn_median_um"], lw=2.8, color="tab:orange", label="FAPI-TEMPO")
    ax.fill_between(
        df["r_over_R"], df["FAPITEMPO_nn_q25_um"], df["FAPITEMPO_nn_q75_um"],
        color="tab:orange", alpha=0.18
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(r"Normalized radius $r/R$")
    ax.set_ylabel(r"Nearest-neighbor distance ($\mu$m)")
    ax.set_title(title)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert wide-format radial NN px values to µm.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_png", default=None)
    ap.add_argument("--px_per_10um", type=float, default=22.0014)
    ap.add_argument(
        "--title",
        default=r"Radial crowding proxy: median NN distance vs $r/R$"
    )
    args = ap.parse_args()

    um_per_px = 10.0 / args.px_per_10um
    df = load_wide_radial_crowding_csv(Path(args.csv))
    df = convert_px_to_um(df, um_per_px=um_per_px)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] Saved CSV: {args.out_csv}")
    print(f"[INFO] Calibration: 1 px = {um_per_px:.6f} µm")

    if args.out_png:
        plot_um(df, Path(args.out_png), args.title)
        print(f"[OK] Saved PNG: {args.out_png}")


if __name__ == "__main__":
    main()