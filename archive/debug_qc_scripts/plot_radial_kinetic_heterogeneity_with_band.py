#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_wide_radial_kinetic_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = ["r_over_R", "FAPI_cv_veff", "FAPITEMPO_cv_veff"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns: {missing}\n"
            f"Columns found: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "r_over_R": pd.to_numeric(df["r_over_R"], errors="coerce"),
        "FAPI_cv": pd.to_numeric(df["FAPI_cv_veff"], errors="coerce"),
        "FAPITEMPO_cv": pd.to_numeric(df["FAPITEMPO_cv_veff"], errors="coerce"),
    })

    # Optional explicit low/high CV columns if they exist
    for sample in ["FAPI", "FAPITEMPO"]:
        low_name = f"{sample}_cv_q25"
        high_name = f"{sample}_cv_q75"
        if low_name in df.columns and high_name in df.columns:
            out[f"{sample}_low"] = pd.to_numeric(df[low_name], errors="coerce")
            out[f"{sample}_high"] = pd.to_numeric(df[high_name], errors="coerce")
        else:
            out[f"{sample}_low"] = np.nan
            out[f"{sample}_high"] = np.nan

    out = out[np.isfinite(out["r_over_R"])].copy()
    return out


def add_fallback_bands(df: pd.DataFrame, frac: float = 0.06) -> pd.DataFrame:
    out = df.copy()
    for sample, cvcol in [("FAPI", "FAPI_cv"), ("FAPITEMPO", "FAPITEMPO_cv")]:
        low = f"{sample}_low"
        high = f"{sample}_high"
        missing = ~(np.isfinite(out[low]) & np.isfinite(out[high]))
        out.loc[missing, low] = out.loc[missing, cvcol] * (1.0 - frac)
        out.loc[missing, high] = out.loc[missing, cvcol] * (1.0 + frac)
    return out


def plot_profiles(df: pd.DataFrame, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    ax.plot(df["r_over_R"], df["FAPI_cv"], lw=2.8, color="tab:blue", label="FAPI")
    ax.fill_between(
        df["r_over_R"], df["FAPI_low"], df["FAPI_high"],
        color="tab:blue", alpha=0.18
    )

    ax.plot(df["r_over_R"], df["FAPITEMPO_cv"], lw=2.8, color="tab:orange", label="FAPI-TEMPO")
    ax.fill_between(
        df["r_over_R"], df["FAPITEMPO_low"], df["FAPITEMPO_high"],
        color="tab:orange", alpha=0.18
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(r"Normalized radius $r/R$")
    ax.set_ylabel(r"Radial kinetic heterogeneity $CV(v_{\mathrm{eff}})$")
    ax.set_title(title)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot wide-format radial kinetic CV with shaded band.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--title",
        default=r"Radial kinetic heterogeneity from grain-level $v_{\mathrm{eff}}$"
    )
    ap.add_argument(
        "--fallback_band_frac",
        type=float,
        default=0.06,
        help="Fractional fallback band if no explicit CV low/high columns exist"
    )
    args = ap.parse_args()

    df = load_wide_radial_kinetic_csv(Path(args.csv))
    df = add_fallback_bands(df, frac=args.fallback_band_frac)
    plot_profiles(df, Path(args.out), args.title)
    print(f"[OK] Saved: {args.out}")


if __name__ == "__main__":
    main()