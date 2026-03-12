#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def robust_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def detect_centroid_columns(df: pd.DataFrame) -> tuple[str, str]:
    candidates = [
        ("centroid_x", "centroid_y"),
        ("cx", "cy"),
        ("x_centroid", "y_centroid"),
        ("center_x", "center_y"),
        ("x_center", "y_center"),
        ("centroid-0", "centroid-1"),
        ("centroid_col", "centroid_row"),
        ("xm", "ym"),
        ("x_um", "y_um"),
        ("x", "y"),
    ]
    cols = set(df.columns)
    for xcol, ycol in candidates:
        if xcol in cols and ycol in cols:
            return xcol, ycol
    raise KeyError(
        "Could not detect centroid columns.\n"
        "Expected something like centroid_x/centroid_y, cx/cy, x/y, etc."
    )


def pairwise_nn_distance(x: np.ndarray, y: np.ndarray, block: int = 5000) -> np.ndarray:
    """
    Memory-safe nearest-neighbor distance using blockwise distance evaluation.
    Returns Euclidean NN distance in the same coordinate units as x,y.
    """
    pts = np.column_stack([x, y]).astype(float)
    n = len(pts)
    out = np.full(n, np.nan, dtype=float)

    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        A = pts[i0:i1]  # (m,2)

        # compute distances from block A to all points
        dx = A[:, None, 0] - pts[None, :, 0]
        dy = A[:, None, 1] - pts[None, :, 1]
        d2 = dx * dx + dy * dy

        # ignore self-distance for rows that overlap global indices
        rows = np.arange(i0, i1) - i0
        cols = np.arange(i0, i1)
        d2[rows, cols] = np.inf

        out[i0:i1] = np.sqrt(np.min(d2, axis=1))

    return out


def add_nn_and_crowding(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    file_col: str | None = None,
) -> pd.DataFrame:
    """
    If file_col is provided and exists, compute NN distance within each file/micrograph.
    Otherwise compute globally over the whole table.
    """
    df = df.copy()
    df["nn_distance_um"] = np.nan
    df["crowding_index_um_inv"] = np.nan

    if file_col is not None and file_col in df.columns:
        groups = df.groupby(file_col, sort=False)
    else:
        groups = [(None, df)]

    for _, g in groups:
        idx = g.index
        x = robust_num(g[xcol]).to_numpy(float)
        y = robust_num(g[ycol]).to_numpy(float)

        ok = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(ok) < 2:
            continue

        nn = np.full(len(g), np.nan, dtype=float)
        nn_ok = pairwise_nn_distance(x[ok], y[ok])

        nn[ok] = nn_ok
        with np.errstate(divide="ignore", invalid="ignore"):
            crowd = np.where(nn > 0, 1.0 / nn, np.nan)

        df.loc[idx, "nn_distance_um"] = nn
        df.loc[idx, "crowding_index_um_inv"] = crowd

    return df


def finite_pair(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def binned_stats(x, y, nbins=25):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) == 0:
        return None, None, None

    xmin, xmax = x.min(), x.max()
    if xmin == xmax:
        xmin -= 1e-6
        xmax += 1e-6

    edges = np.linspace(xmin, xmax, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(nbins, np.nan)
    stds = np.full(nbins, np.nan)

    for i in range(nbins):
        if i < nbins - 1:
            m_bin = (x >= edges[i]) & (x < edges[i + 1])
        else:
            m_bin = (x >= edges[i]) & (x <= edges[i + 1])

        if np.any(m_bin):
            means[i] = np.mean(y[m_bin])
            stds[i] = np.std(y[m_bin])

    return centers, means, stds


def scatter_with_binning(ax, x_f, y_f, x_t, y_t, nbins=25):
    ax.scatter(x_f, y_f, s=5, alpha=0.18, label="FAPI (points)")
    ax.scatter(x_t, y_t, s=5, alpha=0.18, label="FAPI-TEMPO (points)")

    cf, mf, sf = binned_stats(x_f, y_f, nbins=nbins)
    if cf is not None:
        ax.plot(cf, mf, "-", color="C0", lw=2, label="FAPI (mean)")
        ax.fill_between(cf, mf - sf, mf + sf, color="C0", alpha=0.15)

    ct, mt, st = binned_stats(x_t, y_t, nbins=nbins)
    if ct is not None:
        ax.plot(ct, mt, "-", color="C1", lw=2, label="FAPI-TEMPO (mean)")
        ax.fill_between(ct, mt - st, mt + st, color="C1", alpha=0.15)


def make_plot(
    x_f,
    y_f,
    x_t,
    y_t,
    xlabel,
    ylabel,
    title,
    out_path: Path,
    nbins=25,
    legend_loc="best",
):
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    scatter_with_binning(ax, x_f, y_f, x_t, y_t, nbins=nbins)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc=legend_loc)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")


def summarize(df: pd.DataFrame, sample: str) -> dict:
    nn = robust_num(df["nn_distance_um"]).dropna().to_numpy(float)
    ci = robust_num(df["crowding_index_um_inv"]).dropna().to_numpy(float)
    return {
        "sample": sample,
        "n_rows": len(df),
        "n_nn_valid": len(nn),
        "nn_distance_um_mean": float(np.mean(nn)) if len(nn) else np.nan,
        "nn_distance_um_median": float(np.median(nn)) if len(nn) else np.nan,
        "crowding_index_um_inv_mean": float(np.mean(ci)) if len(ci) else np.nan,
        "crowding_index_um_inv_median": float(np.median(ci)) if len(ci) else np.nan,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Compute local crowding/impingement descriptor from nearest-neighbor distance."
    )
    ap.add_argument("--fapi", required=True, help="Path to morpho_kinetics_from_cm_full_FAPI_with_veff.csv")
    ap.add_argument("--tempo", required=True, help="Path to morpho_kinetics_from_cm_full_FAPITEMPO_with_veff.csv")
    ap.add_argument("--out-prefix", required=True, help="Output prefix")
    ap.add_argument("--veff-col", default="v_eff_um_per_ms", help="Effective growth-rate column")
    ap.add_argument("--file-col", default="file_name", help="Grouping column for within-image NN calculation")
    ap.add_argument("--x-col", default=None, help="Optional x centroid column override")
    ap.add_argument("--y-col", default=None, help="Optional y centroid column override")
    ap.add_argument("--nbins", type=int, default=25, help="Bins for mean±std overlays")
    ap.add_argument("--skip-plots", action="store_true", help="Only export augmented tables and summary")
    args = ap.parse_args()

    out_prefix = Path(args.out_prefix)
    out_dir = out_prefix.parent if out_prefix.parent != Path("") else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_prefix.name

    fapi = pd.read_csv(args.fapi)
    tempo = pd.read_csv(args.tempo)

    # centroid detection
    if args.x_col is not None and args.y_col is not None:
        xcol_f, ycol_f = args.x_col, args.y_col
        xcol_t, ycol_t = args.x_col, args.y_col
    else:
        xcol_f, ycol_f = detect_centroid_columns(fapi)
        xcol_t, ycol_t = detect_centroid_columns(tempo)

    print(f"[INFO] FAPI centroid columns: {xcol_f}, {ycol_f}")
    print(f"[INFO] FAPI-TEMPO centroid columns: {xcol_t}, {ycol_t}")

    fapi_aug = add_nn_and_crowding(fapi, xcol_f, ycol_f, file_col=args.file_col)
    tempo_aug = add_nn_and_crowding(tempo, xcol_t, ycol_t, file_col=args.file_col)

    # export updated tables
    fapi_out = out_dir / f"{base}_FAPI_with_crowding.csv"
    tempo_out = out_dir / f"{base}_FAPITEMPO_with_crowding.csv"
    fapi_aug.to_csv(fapi_out, index=False)
    tempo_aug.to_csv(tempo_out, index=False)
    print(f"[OK] Exported: {fapi_out}")
    print(f"[OK] Exported: {tempo_out}")

    # summary table
    summary_df = pd.DataFrame(
        [
            summarize(fapi_aug, "FAPI"),
            summarize(tempo_aug, "FAPI-TEMPO"),
        ]
    )
    summary_out = out_dir / f"{base}_crowding_summary.csv"
    summary_df.to_csv(summary_out, index=False)
    print(f"[OK] Exported summary: {summary_out}")
    print("\nCrowding summary:")
    print(summary_df.to_string(index=False))

    if args.skip_plots:
        print("[DONE] Crowding descriptor computed without plots.")
        return

    if args.veff_col not in fapi_aug.columns or args.veff_col not in tempo_aug.columns:
        raise KeyError(f"Missing '{args.veff_col}' in one or both augmented tables.")

    veff_f = robust_num(fapi_aug[args.veff_col]).to_numpy()
    veff_t = robust_num(tempo_aug[args.veff_col]).to_numpy()

    # Plot 1: v_eff vs NN distance
    x_f, y_f = finite_pair(fapi_aug["nn_distance_um"], veff_f)
    x_t, y_t = finite_pair(tempo_aug["nn_distance_um"], veff_t)
    make_plot(
        x_f, y_f, x_t, y_t,
        xlabel=r"Nearest-neighbor distance (µm)",
        ylabel=r"Effective growth rate $v_{\mathrm{eff}}$ (µm/ms)",
        title=r"$v_{\mathrm{eff}}$ vs nearest-neighbor distance",
        out_path=out_dir / f"{base}_veff_vs_nn_distance.png",
        nbins=args.nbins,
        legend_loc="upper right",
    )

    # Plot 2: v_eff vs crowding index
    x_f, y_f = finite_pair(fapi_aug["crowding_index_um_inv"], veff_f)
    x_t, y_t = finite_pair(tempo_aug["crowding_index_um_inv"], veff_t)
    make_plot(
        x_f, y_f, x_t, y_t,
        xlabel=r"Crowding index $C_{\mathrm{local}} = 1/\mathrm{NN}$ (µm$^{-1}$)",
        ylabel=r"Effective growth rate $v_{\mathrm{eff}}$ (µm/ms)",
        title=r"$v_{\mathrm{eff}}$ vs local crowding index",
        out_path=out_dir / f"{base}_veff_vs_crowding_index.png",
        nbins=args.nbins,
        legend_loc="upper right",
    )

    print("[DONE] Crowding descriptor and plots complete.")


if __name__ == "__main__":
    main()