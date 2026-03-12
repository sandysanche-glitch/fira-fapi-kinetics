#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_combined_X_and_growth.py

Use the existing CSV outputs from the kinetics workflow to:
  1) Make a combined X(t) overlay with both ideal and shifted Avrami
     for FAPI and FAPI–TEMPO on a slightly wider figure.
  2) Re-plot the growth-rate distribution histogram on a slightly
     narrower figure.

Defaults are set to your folder:
  D:\\SWITCHdrive\\Institution\\Sts_grain morphology_ML\\combined_out_penalties
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_combined_X_plot(x_df: pd.DataFrame, out_path: Path):
    """
    Combined X(t) plot:
      - FAPI:    X_pred, ideal Avrami, shifted Avrami
      - FAPI–TEMPO: same
    Time axis in ms (t_ms).
    """
    t_ms = x_df["t_ms"].values

    # width reduced by ~15% (8.0 -> 6.8), same height
    fig, ax = plt.subplots(figsize=(6.3, 4.5))

    # FAPI
    ax.plot(
        t_ms,
        x_df["X_pred_FAPI"],
        color="C0",
        lw=2,
        label="FAPI $X_{\\mathrm{pred}}$",
    )
    ax.plot(
        t_ms,
        x_df["X_AvramiIdeal_FAPI"],
        "--",
        color="C1",
        lw=2,
        label="FAPI ideal Avrami (n=2.5)",
    )
    ax.plot(
        t_ms,
        x_df["X_AvramiShifted_FAPI"],
        ":",
        color="C1",
        lw=2,
        label="FAPI shifted Avrami (n=2.5)",
    )

    # FAPI–TEMPO
    ax.plot(
        t_ms,
        x_df["X_pred_FAPI-TEMPO"],
        color="C2",
        lw=2,
        label="FAPI–TEMPO $X_{\\mathrm{pred}}$",
    )
    ax.plot(
        t_ms,
        x_df["X_AvramiIdeal_FAPI-TEMPO"],
        "--",
        color="C3",
        lw=2,
        label="FAPI–TEMPO ideal Avrami (n=2.5)",
    )
    ax.plot(
        t_ms,
        x_df["X_AvramiShifted_FAPI-TEMPO"],
        ":",
        color="C3",
        lw=2,
        label="FAPI–TEMPO shifted Avrami (n=2.5)",
    )

    ax.set_xlabel("t (ms)")
    ax.set_ylabel("X(t) (a.u.)")
    ax.set_title(
        "Bulk transformed fraction"
    )
    ax.legend(fontsize=8.5, ncol=2)
    ax.grid(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] saved combined X plot: {out_path}")


def make_narrow_growth_hist(g_df: pd.DataFrame, out_path: Path):
    """
    Growth-rate histogram for both datasets on a slightly narrower figure.
    Uses the pre-binned counts in combined_growth_hist.csv.
    """
    # width reduced by ~15% (6.0 -> 5.1), same height
    fig, ax = plt.subplots(figsize=(4.6, 4.0))

    for label in sorted(g_df["label"].unique()):
        sub = g_df[g_df["label"] == label].sort_values("bin_left")

        x_vals = []
        y_vals = []

        # start at left edge with zero
        x_vals.append(sub["bin_left"].iloc[0])
        y_vals.append(0.0)

        for _, row in sub.iterrows():
            left = row["bin_left"]
            right = row["bin_right"]
            c = row["count"]

            x_vals.extend([left, right])
            y_vals.extend([c, c])

        # return to zero at the far right
        x_vals.append(sub["bin_right"].iloc[-1])
        y_vals.append(0.0)

        ax.plot(
            x_vals,
            y_vals,
            drawstyle="steps-post",
            lw=1.8,
            label=label,
        )

    ax.set_xlabel("growth rate (µm/s) [effective]")
    ax.set_ylabel("count")
    ax.set_title("Growth-rate distribution")
    ax.legend()
    ax.grid(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] saved growth histogram: {out_path}")


def main():
    base_dir = Path(
        r"D:\SWITCHdrive\Institution\Sts_grain morphology_ML\combined_out_penalties"
    )

    parser = argparse.ArgumentParser(
        description="Combine X overlays (ideal + shifted) and re-plot growth histogram."
    )
    parser.add_argument(
        "--x-csv",
        default=str(base_dir / "combined_X_pred_models.csv"),
        help="CSV with X_pred and Avrami models",
    )
    parser.add_argument(
        "--growth-csv",
        default=str(base_dir / "combined_growth_hist.csv"),
        help="CSV with growth-rate histogram data",
    )
    parser.add_argument(
        "--out-prefix",
        default=str(base_dir / "combined_penalised"),
        help="Prefix for output PNG files",
    )

    args = parser.parse_args()

    x_df = pd.read_csv(args.x_csv)
    g_df = pd.read_csv(args.growth_csv)

    out_prefix = Path(args.out_prefix)
    out_dir = out_prefix.parent if out_prefix.parent != Path("") else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_prefix.name

    # 1) Combined X overlay (ideal + shifted)
    out_X = out_dir / f"{base}_X_overlay_ideal_shifted_ms.png"
    make_combined_X_plot(x_df, out_X)

    # 2) Narrow growth-rate histogram
    out_hist = out_dir / f"{base}_growth_hist_narrow.png"
    make_narrow_growth_hist(g_df, out_hist)

    print("[DONE] All plots generated.")


if __name__ == "__main__":
    main()
