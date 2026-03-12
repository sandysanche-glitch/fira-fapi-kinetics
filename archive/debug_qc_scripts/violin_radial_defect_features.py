#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
violin_radial_defect_features.py

Loads:
  - radial_defect_features_FAPI.csv
  - radial_defect_features_FAPI_TEMPO.csv

Plots violin distributions for:
  core_mean, edge_mean, delta_edge_minus_core, auc

Overlays mean ± std for each composition and metric,
and prints a simple effect-size report:
  - Cohen's d
  - Hedges' g
  - Cliff's delta

Usage:
  python violin_radial_defect_features.py ^
    --fapi "D:\...\radial_defect_features_FAPI.csv" ^
    --tempo "D:\...\radial_defect_features_FAPI_TEMPO.csv" ^
    --out-prefix "D:\...\radial_defects_violin"

If --out-prefix is omitted, it will save next to the FAPI csv.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------- effect sizes ----------------------------- #
def cohens_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    if pooled <= 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / np.sqrt(pooled)


def hedges_g(x, y):
    d = cohens_d(x, y)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    nx, ny = len(x), len(y)
    if np.isnan(d) or nx + ny < 4:
        return np.nan
    # small-sample correction
    J = 1 - (3 / (4 * (nx + ny) - 9))
    return J * d


def cliffs_delta(x, y):
    """
    Nonparametric effect size.
    For very large N this O(N*M) approach is too slow.
    We downsample if needed.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan

    # Safety downsample for speed on huge datasets
    max_n = 20000
    rng = np.random.default_rng(0)
    if nx > max_n:
        x = rng.choice(x, size=max_n, replace=False)
    if ny > max_n:
        y = rng.choice(y, size=max_n, replace=False)

    # Compute delta via pairwise comparisons using sorting trick
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)

    # Count how many y are less than each x
    # Using searchsorted (vectorized)
    less = np.searchsorted(y_sorted, x_sorted, side="left")
    greater = len(y_sorted) - np.searchsorted(y_sorted, x_sorted, side="right")

    n_pairs = len(x_sorted) * len(y_sorted)
    delta = (np.sum(less) - np.sum(greater)) / n_pairs
    return delta


# ----------------------------- plotting ----------------------------- #
def add_violin(ax, data, positions, widths=0.22):
    vp = ax.violinplot(
        data,
        positions=positions,
        widths=widths,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    # neutral styling; don't set explicit colors unless asked
    for body in vp["bodies"]:
        body.set_alpha(0.25)
        body.set_edgecolor("black")
        body.set_linewidth(0.8)
    return vp


def overlay_mean_std(ax, means, stds, positions, label=None, marker="o"):
    ax.errorbar(
        positions,
        means,
        yerr=stds,
        fmt=marker,
        capsize=6,
        linewidth=1.5,
        markersize=7,
        label=label,
        zorder=5,
    )


def main():
    parser = argparse.ArgumentParser(description="Violin + mean±std + effect sizes for radial defect features.")
    parser.add_argument("--fapi", required=True, help="Path to radial_defect_features_FAPI.csv")
    parser.add_argument("--tempo", required=True, help="Path to radial_defect_features_FAPI_TEMPO.csv")
    parser.add_argument("--out-prefix", default=None, help="Output prefix for PNG/CSV report")
    parser.add_argument("--method-labels", action="store_true",
                        help="If set, adds extra text labels per metric (may clutter).")
    args = parser.parse_args()

    fapi_path = args.fapi
    tempo_path = args.tempo

    df_f = pd.read_csv(fapi_path)
    df_t = pd.read_csv(tempo_path)

    # Expected columns from your previous script
    metrics = [
        ("core_mean", "Core mean"),
        ("edge_mean", "Edge mean"),
        ("delta_edge_minus_core", r"$\Delta$ (edge$-$core)"),
        ("auc", "AUC"),
    ]

    missing_f = [c for c, _ in metrics if c not in df_f.columns]
    missing_t = [c for c, _ in metrics if c not in df_t.columns]
    if missing_f or missing_t:
        raise KeyError(
            "Missing expected columns.\n"
            f"FAPI missing: {missing_f}\n"
            f"TEMPO missing: {missing_t}\n"
            f"FAPI cols: {list(df_f.columns)}\n"
            f"TEMPO cols: {list(df_t.columns)}"
        )

    # Determine output prefix
    if args.out_prefix is None:
        out_dir = os.path.dirname(os.path.abspath(fapi_path))
        out_prefix = os.path.join(out_dir, "radial_defect_features_violin")
    else:
        out_prefix = args.out_prefix

    # Compute summary + effect sizes
    rows = []
    for col, pretty in metrics:
        x = df_f[col].astype(float).to_numpy()
        y = df_t[col].astype(float).to_numpy()

        mean_f = np.nanmean(x)
        std_f = np.nanstd(x, ddof=1)
        mean_t = np.nanmean(y)
        std_t = np.nanstd(y, ddof=1)

        d = cohens_d(x, y)
        g = hedges_g(x, y)
        cd = cliffs_delta(x, y)

        rows.append({
            "metric": col,
            "label": pretty,
            "FAPI_mean": mean_f,
            "FAPI_std": std_f,
            "TEMPO_mean": mean_t,
            "TEMPO_std": std_t,
            "Cohens_d(FAPI-TEMPO)": d,
            "Hedges_g(FAPI-TEMPO)": g,
            "Cliffs_delta(FAPI-TEMPO)": cd,
            "N_FAPI": np.sum(~np.isnan(x)),
            "N_TEMPO": np.sum(~np.isnan(y)),
        })

    report = pd.DataFrame(rows)
    report_path = out_prefix + "_effect_sizes.csv"
    report.to_csv(report_path, index=False)

    # ----------------------------- figure ----------------------------- #
    # Prepare data for violin
    f_list = [df_f[col].astype(float).to_numpy() for col, _ in metrics]
    t_list = [df_t[col].astype(float).to_numpy() for col, _ in metrics]

    # Positions: grouped by metric
    n = len(metrics)
    base = np.arange(n)
    pos_f = base - 0.15
    pos_t = base + 0.15

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    add_violin(ax, f_list, pos_f, widths=0.25)
    add_violin(ax, t_list, pos_t, widths=0.25)

    # Means/stds
    means_f = [np.nanmean(v) for v in f_list]
    stds_f = [np.nanstd(v, ddof=1) for v in f_list]
    means_t = [np.nanmean(v) for v in t_list]
    stds_t = [np.nanstd(v, ddof=1) for v in t_list]

    overlay_mean_std(ax, means_f, stds_f, pos_f, label="FAPI", marker="o")
    overlay_mean_std(ax, means_t, stds_t, pos_t, label="FAPI–TEMPO", marker="s")

    ax.set_title("Radial defect features per grain")
    ax.set_ylabel("Feature value (a.u.)")

    ax.set_xticks(base)
    ax.set_xticklabels([pretty for _, pretty in metrics], rotation=0)

    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", frameon=True)

    # Optional text annotations of effect sizes
    if args.method_labels:
        for i, r in report.iterrows():
            txt = f"d={r['Cohens_d(FAPI-TEMPO)']:.2f}\nΔ={r['Cliffs_delta(FAPI-TEMPO)']:.2f}"
            # place slightly above the max of both violins for that metric
            vmax = np.nanmax(np.concatenate([f_list[i], t_list[i]]))
            ax.text(i, vmax * 1.02 if vmax > 0 else vmax + 0.02, txt,
                    ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    png_path = out_prefix + ".png"
    pdf_path = out_prefix + ".pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    # ----------------------------- console summary ----------------------------- #
    print("=== Radial defect features: summary + effect sizes ===")
    print(report[[
        "metric", "FAPI_mean", "FAPI_std", "TEMPO_mean", "TEMPO_std",
        "Cohens_d(FAPI-TEMPO)", "Cliffs_delta(FAPI-TEMPO)",
        "N_FAPI", "N_TEMPO"
    ]].to_string(index=False))

    print("\n[OK] Saved:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()
