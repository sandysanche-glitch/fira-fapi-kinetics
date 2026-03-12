#!/usr/bin/env python
"""
Publication-style radial plots matching the original 'Radial kintecis.png' look:
- rectangular figure
- light grid
- thick mean curves + shaded IQR bands
- consistent legend box styling

Supports:
1) Radial kinetic heterogeneity band (CV(veff) vs r/R)
2) Radial median veff band (median veff vs r/R)
3) Radial NN distance band (median NN vs r/R) [px or µm already in CSV]
4) Radial impingement index band (median Req/NN vs r/R)

Expected "wide" CSV formats used in this project:
- r_over_R
- FAPI_* and FAPITEMPO_* columns, e.g.
  FAPI_cv_veff, FAPI_q25_veff, FAPI_q75_veff
  FAPI_median_veff, FAPI_q25_veff, FAPI_q75_veff
  FAPI_nn_median_um, FAPI_nn_q25_um, FAPI_nn_q75_um  (or *_px)
  FAPI_imp_median, FAPI_imp_q25, FAPI_imp_q75
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def set_pub_style(font_scale: float = 1.0) -> None:
    # Global style to mimic the original “Radial kintecis.png”
    base = 16 * font_scale
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.6,
        "axes.grid": True,
        "grid.color": "#d0d0d0",
        "grid.linewidth": 1.0,
        "grid.alpha": 0.65,
        "font.size": base,
        "axes.titlesize": base * 1.25,
        "axes.labelsize": base * 1.20,
        "xtick.labelsize": base * 1.05,
        "ytick.labelsize": base * 1.05,
        "legend.fontsize": base * 1.05,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.fancybox": True,
        "legend.edgecolor": "#bfbfbf",
    })


def require_cols(df: pd.DataFrame, cols: list[str], path: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing columns in {path}:\n"
            f"  missing: {missing}\n"
            f"  found: {list(df.columns)}"
        )


def plot_two_sample_band(
    df: pd.DataFrame,
    xcol: str,
    y1_mean: str, y1_lo: str, y1_hi: str,
    y2_mean: str, y2_lo: str, y2_hi: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_png: Path,
    legend_loc: str = "lower right",
    lw: float = 5.0,
    band_alpha: float = 0.18,
    ylim: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = (0.0, 1.0),
    tight: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.0), dpi=200)  # rectangular like the original

    x = df[xcol].to_numpy()

    # FAPI
    y1 = df[y1_mean].to_numpy()
    y1a = df[y1_lo].to_numpy()
    y1b = df[y1_hi].to_numpy()
    ax.plot(x, y1, lw=lw, label="FAPI")
    ax.fill_between(x, y1a, y1b, alpha=band_alpha)

    # FAPI-TEMPO
    y2 = df[y2_mean].to_numpy()
    y2a = df[y2_lo].to_numpy()
    y2b = df[y2_hi].to_numpy()
    ax.plot(x, y2, lw=lw, label="FAPI-TEMPO")
    ax.fill_between(x, y2a, y2b, alpha=band_alpha)

    ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    # Make grid visible but behind data
    ax.set_axisbelow(True)

    leg = ax.legend(loc=legend_loc)
    # Slightly enlarge legend handles (for thick lines)
    for lh in leg.legend_handles:
        try:
            lh.set_linewidth(lw)
        except Exception:
            pass

    if tight:
        fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinetic_csv", type=str, required=True,
                    help="radial_kinetic_proxy_v3_radial_kinetic_heterogeneity.csv (wide format)")
    ap.add_argument("--crowding_csv", type=str, required=True,
                    help="radial_crowding_proxy_v1_radial_crowding_profiles_um.csv (or *_profiles.csv if still px)")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--font_scale", type=float, default=1.0,
                    help="1.0 matches original style; increase e.g. 1.15 for larger text")
    args = ap.parse_args()

    set_pub_style(font_scale=args.font_scale)

    out_dir = Path(args.out_dir)

    # --- KINETIC HETEROGENEITY (CV) ---
    kin_path = Path(args.kinetic_csv)
    kin = pd.read_csv(kin_path)

    # This project uses: r_over_R, FAPI_cv_veff, FAPI_q25_veff, FAPI_q75_veff, ...
    require_cols(
        kin,
        ["r_over_R", "FAPI_cv_veff", "FAPI_q25_veff", "FAPI_q75_veff",
         "FAPITEMPO_cv_veff", "FAPITEMPO_q25_veff", "FAPITEMPO_q75_veff"],
        kin_path,
    )

    plot_two_sample_band(
        kin,
        xcol="r_over_R",
        y1_mean="FAPI_cv_veff", y1_lo="FAPI_q25_veff", y1_hi="FAPI_q75_veff",
        y2_mean="FAPITEMPO_cv_veff", y2_lo="FAPITEMPO_q25_veff", y2_hi="FAPITEMPO_q75_veff",
        title=r"Radial kinetic heterogeneity from grain-level $v_\mathrm{eff}$",
        xlabel=r"Normalized radius $r/R$",
        ylabel=r"Radial kinetic heterogeneity $\mathrm{CV}(v_\mathrm{eff})$",
        out_png=out_dir / "radial_kinetic_heterogeneity_cv_band_pubstyle.png",
        legend_loc="lower right",
    )

    # --- MEDIAN veff (band) ---
    require_cols(
        kin,
        ["r_over_R", "FAPI_median_veff", "FAPI_q25_veff", "FAPI_q75_veff",
         "FAPITEMPO_median_veff", "FAPITEMPO_q25_veff", "FAPITEMPO_q75_veff"],
        kin_path,
    )

    plot_two_sample_band(
        kin,
        xcol="r_over_R",
        y1_mean="FAPI_median_veff", y1_lo="FAPI_q25_veff", y1_hi="FAPI_q75_veff",
        y2_mean="FAPITEMPO_median_veff", y2_lo="FAPITEMPO_q25_veff", y2_hi="FAPITEMPO_q75_veff",
        title=r"Annulus-conditioned median $v_\mathrm{eff}$ vs $r/R$",
        xlabel=r"Normalized radius $r/R$",
        ylabel=r"Median effective growth rate $v_\mathrm{eff}$ ($\mu$m/ms)",
        out_png=out_dir / "radial_median_veff_band_pubstyle.png",
        legend_loc="upper right",
    )

    # --- CROWDING (NN + impingement) ---
    crowd_path = Path(args.crowding_csv)
    crowd = pd.read_csv(crowd_path)
    require_cols(crowd, ["r_over_R"], crowd_path)

    # Detect whether NN is in um or px based on column presence
    nn_unit = None
    if "FAPI_nn_median_um" in crowd.columns:
        nn_unit = "um"
        nn_med_f, nn_q25_f, nn_q75_f = "FAPI_nn_median_um", "FAPI_nn_q25_um", "FAPI_nn_q75_um"
        nn_med_t, nn_q25_t, nn_q75_t = "FAPITEMPO_nn_median_um", "FAPITEMPO_nn_q25_um", "FAPITEMPO_nn_q75_um"
        nn_ylabel = r"Nearest-neighbor distance ($\mu$m)"
    elif "FAPI_nn_median_px" in crowd.columns:
        nn_unit = "px"
        nn_med_f, nn_q25_f, nn_q75_f = "FAPI_nn_median_px", "FAPI_nn_q25_px", "FAPI_nn_q75_px"
        nn_med_t, nn_q25_t, nn_q75_t = "FAPITEMPO_nn_median_px", "FAPITEMPO_nn_q25_px", "FAPITEMPO_nn_q75_px"
        nn_ylabel = "Nearest-neighbor distance (px)"
    else:
        raise KeyError(
            f"Could not find NN columns in {crowd_path}. "
            f"Expected *_nn_median_um or *_nn_median_px.\nColumns: {list(crowd.columns)}"
        )

    require_cols(crowd, [nn_med_f, nn_q25_f, nn_q75_f, nn_med_t, nn_q25_t, nn_q75_t], crowd_path)

    plot_two_sample_band(
        crowd,
        xcol="r_over_R",
        y1_mean=nn_med_f, y1_lo=nn_q25_f, y1_hi=nn_q75_f,
        y2_mean=nn_med_t, y2_lo=nn_q25_t, y2_hi=nn_q75_t,
        title=r"Radial crowding proxy: median NN distance vs $r/R$",
        xlabel=r"Normalized radius $r/R$",
        ylabel=nn_ylabel,
        out_png=out_dir / f"radial_nn_median_band_pubstyle_{nn_unit}.png",
        legend_loc="upper center",
    )

    # Impingement
    require_cols(
        crowd,
        ["FAPI_imp_median", "FAPI_imp_q25", "FAPI_imp_q75",
         "FAPITEMPO_imp_median", "FAPITEMPO_imp_q25", "FAPITEMPO_imp_q75"],
        crowd_path,
    )

    plot_two_sample_band(
        crowd,
        xcol="r_over_R",
        y1_mean="FAPI_imp_median", y1_lo="FAPI_imp_q25", y1_hi="FAPI_imp_q75",
        y2_mean="FAPITEMPO_imp_median", y2_lo="FAPITEMPO_imp_q25", y2_hi="FAPITEMPO_imp_q75",
        title=r"Radial crowding proxy: median impingement index vs $r/R$",
        xlabel=r"Normalized radius $r/R$",
        ylabel=r"Impingement index ($R_\mathrm{eq}/\mathrm{NN}$)",
        out_png=out_dir / "radial_impingement_median_band_pubstyle.png",
        legend_loc="upper center",
    )

    print(f"[OK] Wrote publication-style radial plots to: {out_dir}")


if __name__ == "__main__":
    main()