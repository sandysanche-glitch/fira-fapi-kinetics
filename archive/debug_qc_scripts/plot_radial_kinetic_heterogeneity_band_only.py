#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# I/O
# -----------------------------
def load_radial_kinetic_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = [
        "r_over_R",
        "FAPI_cv_veff",
        "FAPITEMPO_cv_veff",
        "FAPI_n_grains",
        "FAPITEMPO_n_grains",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns: {missing}\n"
            f"Columns found: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "r_over_R": pd.to_numeric(df["r_over_R"], errors="coerce"),

        "FAPI_cv": pd.to_numeric(df["FAPI_cv_veff"], errors="coerce"),
        "FAPI_n": pd.to_numeric(df["FAPI_n_grains"], errors="coerce"),

        "FAPITEMPO_cv": pd.to_numeric(df["FAPITEMPO_cv_veff"], errors="coerce"),
        "FAPITEMPO_n": pd.to_numeric(df["FAPITEMPO_n_grains"], errors="coerce"),
    })

    # optional columns, only for QC / fallback if ever needed
    optional = [
        "FAPI_std_veff", "FAPITEMPO_std_veff",
        "FAPI_mean_veff", "FAPITEMPO_mean_veff",
    ]
    for col in optional:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")

    out = out[np.isfinite(out["r_over_R"])].copy()
    out = out.sort_values("r_over_R").reset_index(drop=True)
    return out


# -----------------------------
# CV uncertainty band
# -----------------------------
def cv_se_approx(cv: np.ndarray, n: np.ndarray) -> np.ndarray:
    """
    Approximate SE of coefficient of variation.

    Uses:
        SE(CV) ≈ CV * sqrt((1 + 2*CV^2) / (2*(n-1)))

    This is an approximation, but it is far better than using q25/q75 of v_eff
    to fake a CV band.
    """
    cv = np.asarray(cv, dtype=float)
    n = np.asarray(n, dtype=float)

    se = np.full_like(cv, np.nan, dtype=float)
    good = np.isfinite(cv) & np.isfinite(n) & (n > 2) & (cv >= 0)
    se[good] = cv[good] * np.sqrt((1.0 + 2.0 * cv[good] ** 2) / (2.0 * (n[good] - 1.0)))
    return se


def add_cv_bands(
    df: pd.DataFrame,
    z_value: float = 1.0,
    min_se_frac: float = 0.03,
) -> pd.DataFrame:
    """
    Build low/high bands around CV using approximate SE.
    z_value=1.0 gives ~68% visual band.
    """
    out = df.copy()

    for prefix in ["FAPI", "FAPITEMPO"]:
        cv_col = f"{prefix}_cv"
        n_col = f"{prefix}_n"

        se = cv_se_approx(out[cv_col].values, out[n_col].values)

        # tiny fallback so the band is still visible in bins with huge n
        fallback = np.abs(out[cv_col].values) * float(min_se_frac)
        se = np.where(np.isfinite(se), np.maximum(se, fallback), fallback)

        low = out[cv_col].values - z_value * se
        high = out[cv_col].values + z_value * se

        out[f"{prefix}_low"] = np.maximum(low, 0.0)
        out[f"{prefix}_high"] = high

    return out


# -----------------------------
# Plot style
# -----------------------------
def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, which="major", alpha=0.28, linewidth=0.8)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    ax.tick_params(axis="both", labelsize=14, width=1.2, length=6)


def plot_radial_kinetic_cv(
    df: pd.DataFrame,
    out_png: Path,
    title: str,
) -> None:
    # Match the original rectangular look more closely
    fig, ax = plt.subplots(figsize=(9.2, 6.0))

    # Bands first
    ax.fill_between(
        df["r_over_R"].values,
        df["FAPI_low"].values,
        df["FAPI_high"].values,
        color="tab:blue",
        alpha=0.18,
        zorder=1,
    )
    ax.fill_between(
        df["r_over_R"].values,
        df["FAPITEMPO_low"].values,
        df["FAPITEMPO_high"].values,
        color="tab:orange",
        alpha=0.18,
        zorder=1,
    )

    # Main lines on top
    ax.plot(
        df["r_over_R"].values,
        df["FAPI_cv"].values,
        color="tab:blue",
        linewidth=3.6,
        label="FAPI",
        zorder=3,
    )
    ax.plot(
        df["r_over_R"].values,
        df["FAPITEMPO_cv"].values,
        color="tab:orange",
        linewidth=3.6,
        label="FAPI-TEMPO",
        zorder=3,
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(r"Normalized radius $r/R$", fontsize=18)
    ax.set_ylabel(r"Radial kinetic heterogeneity $CV(v_{\mathrm{eff}})$", fontsize=18)
    ax.set_title(title, fontsize=22, pad=14)

    style_axes(ax)

    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=16,
        handlelength=2.6,
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot radial kinetic heterogeneity CV with a proper uncertainty band."
    )
    ap.add_argument("--csv", required=True, help="Input radial kinetic CSV")
    ap.add_argument("--out", required=True, help="Output PNG")
    ap.add_argument(
        "--title",
        default=r"Radial kinetic heterogeneity from grain-level $v_{\mathrm{eff}}$",
        help="Plot title",
    )
    ap.add_argument(
        "--z",
        type=float,
        default=1.0,
        help="Band width multiplier in SE units (default: 1.0)",
    )
    ap.add_argument(
        "--min_se_frac",
        type=float,
        default=0.03,
        help="Minimum visible band as fraction of CV (default: 0.03)",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)

    df = load_radial_kinetic_csv(csv_path)
    df = add_cv_bands(df, z_value=args.z, min_se_frac=args.min_se_frac)
    plot_radial_kinetic_cv(df, out_path, args.title)

    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()