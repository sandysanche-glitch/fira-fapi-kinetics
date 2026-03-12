#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------
# Figure-6-panel-i-like style
# ---------------------------------------------------------------------
FIGSIZE = (10, 7.5)   # close to the reference panel aspect
DPI = 300

LINEWIDTH = 3.0
BAND_ALPHA = 0.18

FAPI_COLOR = "#1f77b4"
TEMPO_COLOR = "#ff7f0e"

TITLE_FS = 26
LABEL_FS = 22
TICK_FS = 18
LEGEND_FS = 18


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "r_over_R" not in df.columns:
        raise KeyError(
            f"'r_over_R' not found in {path}. Columns found: {list(df.columns)}"
        )
    return df


def style_axes(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", labelsize=TICK_FS, width=1.8, length=7)
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
    ax.set_xlim(0.0, 1.0)


def pick_cols(
    df: pd.DataFrame,
    sample_prefix: str,
    mean_suffix: str,
    low_suffix: str,
    high_suffix: str,
) -> Tuple[str, str, str]:
    mean_col = f"{sample_prefix}_{mean_suffix}"
    low_col = f"{sample_prefix}_{low_suffix}"
    high_col = f"{sample_prefix}_{high_suffix}"

    missing = [c for c in [mean_col, low_col, high_col] if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing expected columns {missing}. Available columns: {list(df.columns)}"
        )
    return mean_col, low_col, high_col


def make_band_plot(
    df: pd.DataFrame,
    out_png: Path,
    ylabel: str,
    title: Optional[str],
    fapi_mean_suffix: str,
    fapi_low_suffix: str,
    fapi_high_suffix: str,
    tempo_mean_suffix: str,
    tempo_low_suffix: str,
    tempo_high_suffix: str,
    legend_loc: str = "upper right",
) -> None:
    r = df["r_over_R"]

    fapi_mean, fapi_low, fapi_high = pick_cols(
        df, "FAPI", fapi_mean_suffix, fapi_low_suffix, fapi_high_suffix
    )
    tempo_mean, tempo_low, tempo_high = pick_cols(
        df, "FAPITEMPO", tempo_mean_suffix, tempo_low_suffix, tempo_high_suffix
    )

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # FAPI
    ax.fill_between(
        r,
        df[fapi_low],
        df[fapi_high],
        color=FAPI_COLOR,
        alpha=BAND_ALPHA,
        linewidth=0,
    )
    ax.plot(
        r,
        df[fapi_mean],
        color=FAPI_COLOR,
        lw=LINEWIDTH,
        label="FAPI",
    )

    # FAPI-TEMPO
    ax.fill_between(
        r,
        df[tempo_low],
        df[tempo_high],
        color=TEMPO_COLOR,
        alpha=BAND_ALPHA,
        linewidth=0,
    )
    ax.plot(
        r,
        df[tempo_mean],
        color=TEMPO_COLOR,
        lw=LINEWIDTH,
        label="FAPI-TEMPO",
    )

    if title:
        ax.set_title(title, fontsize=TITLE_FS, pad=10)

    ax.set_xlabel(r"Normalized radius $r/R$", fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)

    ax.legend(
        loc=legend_loc,
        frameon=False,
        fontsize=LEGEND_FS,
    )

    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Reformat radial proxy plots to match the Figure 6 panel i style."
    )
    ap.add_argument(
        "--radial-crowding-csv",
        required=True,
        help="CSV containing radial crowding profiles (NN + impingement).",
    )
    ap.add_argument(
        "--radial-kinetic-csv",
        required=True,
        help="CSV containing radial kinetic proxy profiles (median_veff + CV).",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Directory where the formatted PNGs will be written.",
    )
    ap.add_argument(
        "--with-titles",
        action="store_true",
        help="Include titles inside the PNGs. Omit this flag for panel-ready plots.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    crowd = load_csv(Path(args.radial_crowding_csv))
    kin = load_csv(Path(args.radial_kinetic_csv))

    # 1) NN distance in µm
    make_band_plot(
        df=crowd,
        out_png=out_dir / "panel_like_fig6i_radial_nn_um.png",
        ylabel=r"Nearest-neighbor distance ($\mu$m)",
        title="Radial crowding proxy: median NN distance vs $r/R$" if args.with_titles else None,
        fapi_mean_suffix="nn_median_um",
        fapi_low_suffix="nn_q25_um",
        fapi_high_suffix="nn_q75_um",
        tempo_mean_suffix="nn_median_um",
        tempo_low_suffix="nn_q25_um",
        tempo_high_suffix="nn_q75_um",
        legend_loc="upper center",
    )

    # 2) Impingement index
    make_band_plot(
        df=crowd,
        out_png=out_dir / "panel_like_fig6i_radial_impingement.png",
        ylabel=r"Impingement index ($R_{\mathrm{eq}}/\mathrm{NN}$)",
        title="Radial crowding proxy: median impingement index vs $r/R$" if args.with_titles else None,
        fapi_mean_suffix="imp_median",
        fapi_low_suffix="imp_q25",
        fapi_high_suffix="imp_q75",
        tempo_mean_suffix="imp_median",
        tempo_low_suffix="imp_q25",
        tempo_high_suffix="imp_q75",
        legend_loc="upper center",
    )

    # 3) Median v_eff
    make_band_plot(
        df=kin,
        out_png=out_dir / "panel_like_fig6i_radial_median_veff.png",
        ylabel=r"Median effective growth rate $v_{\mathrm{eff}}$ ($\mu$m/ms)",
        title="Annulus-conditioned median $v_{\\mathrm{eff}}$ vs $r/R$" if args.with_titles else None,
        fapi_mean_suffix="median_veff",
        fapi_low_suffix="q25_veff",
        fapi_high_suffix="q75_veff",
        tempo_mean_suffix="median_veff",
        tempo_low_suffix="q25_veff",
        tempo_high_suffix="q75_veff",
        legend_loc="upper right",
    )

    print("[OK] Wrote:")
    print(f"  {out_dir / 'panel_like_fig6i_radial_nn_um.png'}")
    print(f"  {out_dir / 'panel_like_fig6i_radial_impingement.png'}")
    print(f"  {out_dir / 'panel_like_fig6i_radial_median_veff.png'}")


if __name__ == "__main__":
    main()