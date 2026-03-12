#!/usr/bin/env python
"""
Relate per-grain morphology to effective growth rate v_eff.

Inputs
------
Two CSV files, one per composition (FAPI and FAPI–TEMPO), each containing
at least these columns (per grain):

    - grain_id              (optional but nice to have)
    - circ_dist             circularity distortion (scalar)
    - defect_fraction       φ = A_def / A_grain
    - entropy               Shannon entropy H of grain texture
    - veff                  effective growth rate (µm/ms)

Outputs
-------
1) Two "with_veff" CSVs (mostly a cleaned / filtered copy of the inputs):

    morpho_kinetics_from_cm_FAPI_with_veff.csv
    morpho_kinetics_from_cm_FAPITEMPO_with_veff.csv

2) Three PNG figures:

    morpho_kinetics_from_cm_veff_vs_circ_dist.png
    morpho_kinetics_from_cm_veff_vs_defect_fraction.png
    morpho_kinetics_from_cm_veff_vs_entropy.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def bin_xy(x, y, nbins=25, min_count=5):
    """
    Bin y as a function of x into nbins linearly spaced bins.

    Returns
    -------
    x_centers, y_mean, y_std
    """
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return np.array([]), np.array([]), np.array([])

    bins = np.linspace(x.min(), x.max(), nbins + 1)
    idx = np.digitize(x, bins) - 1  # bin index 0..nbins-1

    x_centers = []
    y_mean = []
    y_std = []

    for i in range(nbins):
        in_bin = idx == i
        if np.count_nonzero(in_bin) < min_count:
            continue
        x_centers.append(0.5 * (bins[i] + bins[i + 1]))
        y_mean.append(np.mean(y[in_bin]))
        y_std.append(np.std(y[in_bin]))

    return np.asarray(x_centers), np.asarray(y_mean), np.asarray(y_std)


def scatter_plus_binned(
    ax, df_fapi, df_tempo, xcol, xlabel, ylabel, title,
    out_png, nbins=25
):
    """
    Make a scatter plot of v_eff vs a given x-column, with binned mean ± std.
    """

    # soft colors and alpha
    ax.scatter(
        df_fapi[xcol], df_fapi["veff"],
        s=8, alpha=0.25, edgecolors="none", label="FAPI (points)"
    )
    ax.scatter(
        df_tempo[xcol], df_tempo["veff"],
        s=8, alpha=0.25, edgecolors="none", label="FAPI–TEMPO (points)"
    )

    # binned statistics
    x_f, m_f, s_f = bin_xy(df_fapi[xcol], df_fapi["veff"], nbins=nbins)
    x_t, m_t, s_t = bin_xy(df_tempo[xcol], df_tempo["veff"], nbins=nbins)

    if len(x_f):
        ax.fill_between(
            x_f, m_f - s_f, m_f + s_f, alpha=0.2
        )
        ax.plot(x_f, m_f, lw=2, label="FAPI (mean)")
    if len(x_t):
        ax.fill_between(
            x_t, m_t - s_t, m_t + s_t, alpha=0.2
        )
        ax.plot(x_t, m_t, lw=2, label="FAPI–TEMPO (mean)")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=True)

    ax.set_ylim(bottom=0.0)  # v_eff should not be negative

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"[INFO] saved {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Correlate morphology descriptors with v_eff."
    )
    p.add_argument(
        "--fapi-csv", required=True,
        help="Per-grain CSV for FAPI (must contain veff column)."
    )
    p.add_argument(
        "--tempo-csv", required=True,
        help="Per-grain CSV for FAPI–TEMPO (must contain veff column)."
    )
    p.add_argument(
        "--out-prefix", required=True,
        help="Output prefix (directory + base name)."
    )
    p.add_argument(
        "--nbins", type=int, default=25,
        help="Number of bins for x when computing mean ± std."
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_prefix = Path(args.out_prefix)

    # ------------------------------------------------------------------
    # Load and basic cleaning
    # ------------------------------------------------------------------
    df_fapi = pd.read_csv(args.fapi_csv)
    df_tempo = pd.read_csv(args.tempo_csv)

    # Ensure columns exist (rename if user used slightly different names)
    rename_map = {
        "circularity_distortion": "circ_dist",
        "circ_distortion": "circ_dist",
        "defect_frac": "defect_fraction",
        "H_entropy": "entropy",
        "H": "entropy",
    }

    df_fapi = df_fapi.rename(columns=rename_map)
    df_tempo = df_tempo.rename(columns=rename_map)

    required = ["circ_dist", "defect_fraction", "entropy", "veff"]
    for col in required:
        if col not in df_fapi.columns:
            raise ValueError(f"FAPI CSV missing required column '{col}'")
        if col not in df_tempo.columns:
            raise ValueError(f"FAPI–TEMPO CSV missing required column '{col}'")

    # Drop any all-NaN rows in veff
    df_fapi = df_fapi[np.isfinite(df_fapi["veff"])]
    df_tempo = df_tempo[np.isfinite(df_tempo["veff"])]

    print(f"[INFO] FAPI grains used: {len(df_fapi)}")
    print(f"[INFO] FAPI–TEMPO grains used: {len(df_tempo)}")

    # Save cleaned copies with a standardized schema
    out_fapi_csv = out_prefix.with_name(out_prefix.name + "_FAPI_with_veff.csv")
    out_tempo_csv = out_prefix.with_name(out_prefix.name + "_FAPITEMPO_with_veff.csv")

    df_fapi.to_csv(out_fapi_csv, index=False)
    df_tempo.to_csv(out_tempo_csv, index=False)
    print(f"[INFO] wrote {out_fapi_csv}")
    print(f"[INFO] wrote {out_tempo_csv}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    # 1) v_eff vs circularity distortion
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    scatter_plus_binned(
        ax, df_fapi, df_tempo,
        xcol="circ_dist",
        xlabel="Circularity distortion (from nucleation center)",
        ylabel=r"Effective growth rate $v_{\mathrm{eff}}$ ($\mu\mathrm{m}/\mathrm{ms}$)",
        title=r"$v_{\mathrm{eff}}$ vs circularity distortion",
        out_png=out_prefix.with_name(out_prefix.name + "_veff_vs_circ_dist.png"),
        nbins=args.nbins,
    )

    # 2) v_eff vs defect fraction
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    scatter_plus_binned(
        ax, df_fapi, df_tempo,
        xcol="defect_fraction",
        xlabel=r"Defect fraction $\phi$",
        ylabel=r"Effective growth rate $v_{\mathrm{eff}}$ ($\mu\mathrm{m}/\mathrm{ms}$)",
        title=r"$v_{\mathrm{eff}}$ vs defect fraction",
        out_png=out_prefix.with_name(out_prefix.name + "_veff_vs_defect_fraction.png"),
        nbins=args.nbins,
    )

    # 3) v_eff vs entropy
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    scatter_plus_binned(
        ax, df_fapi, df_tempo,
        xcol="entropy",
        xlabel=r"Shannon entropy (grain texture)",
        ylabel=r"Effective growth rate $v_{\mathrm{eff}}$ ($\mu\mathrm{m}/\mathrm{ms}$)",
        title=r"$v_{\mathrm{eff}}$ vs entropy",
        out_png=out_prefix.with_name(out_prefix.name + "_veff_vs_entropy.png"),
        nbins=args.nbins,
    )


if __name__ == "__main__":
    main()
