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


def finite_pair(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def add_defect_fraction(
    df: pd.DataFrame,
    defect_area_col: str,
    grain_area_col: str,
) -> pd.DataFrame:
    df = df.copy()
    if defect_area_col in df.columns and grain_area_col in df.columns:
        num = robust_num(df[defect_area_col])
        den = robust_num(df[grain_area_col])
        with np.errstate(divide="ignore", invalid="ignore"):
            df["defect_fraction"] = np.where(den > 0, num / den, np.nan)
    return df


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
    y_mean = np.full(nbins, np.nan)
    y_std = np.full(nbins, np.nan)

    for i in range(nbins):
        if i < nbins - 1:
            m_bin = (x >= edges[i]) & (x < edges[i + 1])
        else:
            m_bin = (x >= edges[i]) & (x <= edges[i + 1])
        if np.any(m_bin):
            y_mean[i] = np.mean(y[m_bin])
            y_std[i] = np.std(y[m_bin])

    return centers, y_mean, y_std


def scatter_with_binning(
    ax,
    x_f,
    y_f,
    x_t,
    y_t,
    label_f="FAPI",
    label_t="FAPI-TEMPO",
    nbins=25,
    point_alpha=0.18,
    point_size=5,
):
    ax.scatter(x_f, y_f, s=point_size, alpha=point_alpha, label=f"{label_f} (points)")
    ax.scatter(x_t, y_t, s=point_size, alpha=point_alpha, label=f"{label_t} (points)")

    cf, mf, sf = binned_stats(x_f, y_f, nbins=nbins)
    if cf is not None:
        ax.plot(cf, mf, "-", color="C0", lw=2, label=f"{label_f} (mean)")
        ax.fill_between(cf, mf - sf, mf + sf, color="C0", alpha=0.15)

    ct, mt, st = binned_stats(x_t, y_t, nbins=nbins)
    if ct is not None:
        ax.plot(ct, mt, "-", color="C1", lw=2, label=f"{label_t} (mean)")
        ax.fill_between(ct, mt - st, mt + st, color="C1", alpha=0.15)


def style_axis(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(False)


def make_single_plot(
    x_f,
    y_f,
    x_t,
    y_t,
    xlabel,
    ylabel,
    title,
    out_path: Path,
    nbins=25,
    figsize=(5.2, 4.0),
    xlim=None,
    ylim=None,
    legend_loc="best",
):
    fig, ax = plt.subplots(figsize=figsize)
    scatter_with_binning(ax, x_f, y_f, x_t, y_t, nbins=nbins)
    style_axis(ax, xlabel, ylabel, title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc=legend_loc)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Plot large-dataset v_eff correlations from *_with_veff.csv files."
    )
    ap.add_argument("--fapi", required=True, help="Path to morpho_kinetics_from_cm_full_FAPI_with_veff.csv")
    ap.add_argument("--tempo", required=True, help="Path to morpho_kinetics_from_cm_full_FAPITEMPO_with_veff.csv")
    ap.add_argument("--out-prefix", required=True, help="Output prefix for plots and exports")
    ap.add_argument("--veff-col", default="v_eff_um_per_ms", help="Effective growth-rate column")
    ap.add_argument("--circ-col", default="circularity_distortion", help="Circularity distortion column")
    ap.add_argument("--entropy-col", default="entropy_hm_(bits)", help="Entropy column")
    ap.add_argument("--defect-area-col", default="defects_area_(µm²)", help="Defects area column")
    ap.add_argument("--grain-area-col", default="area_(µm²)", help="Grain area column")
    ap.add_argument("--nbins", type=int, default=25, help="Number of bins for mean±std overlays")
    ap.add_argument("--export-merged", action="store_true", help="Also export cleaned merged tables")
    args = ap.parse_args()

    out_prefix = Path(args.out_prefix)
    out_dir = out_prefix.parent if out_prefix.parent != Path("") else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_prefix.name

    fapi = pd.read_csv(args.fapi)
    tempo = pd.read_csv(args.tempo)

    fapi = add_defect_fraction(fapi, args.defect_area_col, args.grain_area_col)
    tempo = add_defect_fraction(tempo, args.defect_area_col, args.grain_area_col)

    if args.veff_col not in fapi.columns or args.veff_col not in tempo.columns:
        raise KeyError(f"Missing '{args.veff_col}' in one or both input files.")

    veff_f = robust_num(fapi[args.veff_col]).to_numpy()
    veff_t = robust_num(tempo[args.veff_col]).to_numpy()

    # 1) v_eff vs circularity distortion
    if args.circ_col in fapi.columns and args.circ_col in tempo.columns:
        circ_f, veff_f_c = finite_pair(robust_num(fapi[args.circ_col]), veff_f)
        circ_t, veff_t_c = finite_pair(robust_num(tempo[args.circ_col]), veff_t)

        make_single_plot(
            circ_f, veff_f_c, circ_t, veff_t_c,
            xlabel="Circularity distortion (from nucleation center)",
            ylabel=r"Effective growth rate $v_{\mathrm{eff}}$ ($\mu$m/ms)",
            title=r"$v_{\mathrm{eff}}$ vs circularity distortion",
            out_path=out_dir / f"{base}_veff_vs_circ_dist.png",
            nbins=args.nbins,
            figsize=(5.2, 4.0),
            legend_loc="upper right",
        )
    else:
        print("[WARN] circularity distortion column missing; skipping circularity plot.")

    # 2) v_eff vs defect fraction
    if "defect_fraction" in fapi.columns and "defect_fraction" in tempo.columns:
        df_f, veff_f_d = finite_pair(fapi["defect_fraction"], veff_f)
        df_t, veff_t_d = finite_pair(tempo["defect_fraction"], veff_t)

        make_single_plot(
            df_f, veff_f_d, df_t, veff_t_d,
            xlabel=r"Defect fraction $\phi$",
            ylabel=r"Effective growth rate $v_{\mathrm{eff}}$ ($\mu$m/ms)",
            title=r"$v_{\mathrm{eff}}$ vs defect fraction",
            out_path=out_dir / f"{base}_veff_vs_defect_fraction.png",
            nbins=args.nbins,
            figsize=(5.2, 4.0),
            legend_loc="upper right",
        )
    else:
        print("[WARN] defect_fraction unavailable; skipping defect plot.")

    # 3) v_eff vs entropy
    if args.entropy_col in fapi.columns and args.entropy_col in tempo.columns:
        ent_f, veff_f_e = finite_pair(robust_num(fapi[args.entropy_col]), veff_f)
        ent_t, veff_t_e = finite_pair(robust_num(tempo[args.entropy_col]), veff_t)

        make_single_plot(
            ent_f, veff_f_e, ent_t, veff_t_e,
            xlabel="Shannon entropy (grain texture)",
            ylabel=r"Effective growth rate $v_{\mathrm{eff}}$ ($\mu$m/ms)",
            title=r"$v_{\mathrm{eff}}$ vs entropy",
            out_path=out_dir / f"{base}_veff_vs_entropy.png",
            nbins=args.nbins,
            figsize=(5.2, 4.0),
            legend_loc="upper left",
        )
    else:
        print("[WARN] entropy column missing; skipping entropy plot.")

    # optional cleaned exports
    if args.export_merged:
        fapi_out = out_dir / f"{base}_FAPI_cleaned.csv"
        tempo_out = out_dir / f"{base}_FAPITEMPO_cleaned.csv"
        fapi.to_csv(fapi_out, index=False)
        tempo.to_csv(tempo_out, index=False)
        print(f"[OK] Exported cleaned FAPI table: {fapi_out}")
        print(f"[OK] Exported cleaned FAPI-TEMPO table: {tempo_out}")

    print("[DONE] Large-dataset v_eff correlation plots complete.")


if __name__ == "__main__":
    main()