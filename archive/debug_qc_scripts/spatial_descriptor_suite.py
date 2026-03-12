#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
spatial_descriptor_suite.py

Spatial maps and RGB fusion plots for FAPI and FAPI–TEMPO
using the anisotropy_merged_*.csv files.

- Automatically computes defect_fraction if missing:
    defect_fraction = defects_area_(µm²) / area_(µm²)

- Converts coordinates from px -> µm using px_per_um.

Example usage (Windows, from your folder):

  python spatial_descriptor_suite.py ^
      --fapi-csv "D:\SWITCHdrive\Institution\Segmentation%datset stats\anisotropy_merged_FAPI_merged.csv" ^
      --tempo-csv "D:\SWITCHdrive\Institution\Segmentation%datset stats\anisotropy_merged_FAPITEMPO_merged.csv" ^
      --out-dir  "D:\SWITCHdrive\Institution\Segmentation%datset stats\spatial_outputs"

"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- global calibration: pixels -> microns ---
PX_PER_UM = 2.20014  # from your manuscript
UM_PER_PX = 1.0 / PX_PER_UM


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def minmax_norm(series: pd.Series) -> np.ndarray:
    """Normalize a pandas Series to [0,1]."""
    vals = series.to_numpy(dtype=float)
    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)
    if np.isclose(vmax, vmin):
        return np.zeros_like(vals)
    return (vals - vmin) / (vmax - vmin)


def ensure_defect_fraction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has a 'defect_fraction' column.
    If missing, compute from defects_area_(µm²) / area_(µm²) when available.
    """
    if "defect_fraction" not in df.columns:
        if "defects_area_(µm²)" in df.columns and "area_(µm²)" in df.columns:
            df = df.copy()
            df["defect_fraction"] = (
                df["defects_area_(µm²)"] / df["area_(µm²)"]
            ).fillna(0.0)
            print("[INFO] defect_fraction computed from defects_area_(µm²)/area_(µm²).")
        else:
            # fall back to zeros if we really have nothing
            df = df.copy()
            df["defect_fraction"] = 0.0
            print("[WARN] No defects_area_(µm²) / area_(µm²); setting defect_fraction=0.")
    return df


def get_xy_um(df: pd.DataFrame):
    """
    Get (x,y) coordinates in microns.
    Prefers nucleus coordinates; falls back to centroids.
    """
    if "nuc_x" in df.columns and "nuc_y" in df.columns:
        x_px = df["nuc_x"].to_numpy(float)
        y_px = df["nuc_y"].to_numpy(float)
    elif "centroid_x" in df.columns and "centroid_y" in df.columns:
        x_px = df["centroid_x"].to_numpy(float)
        y_px = df["centroid_y"].to_numpy(float)
    else:
        raise KeyError("No nuc_x/nuc_y or centroid_x/centroid_y columns found.")

    x_um = x_px * UM_PER_PX
    y_um = y_px * UM_PER_PX
    return x_um, y_um


# ---------------------------------------------------------------------
# plotting functions
# ---------------------------------------------------------------------
def plot_spatial_map(df: pd.DataFrame, value_col: str, title: str, out_path: Path):
    """
    Scatter plot in (x,y) with colour given by `value_col`.
    Coordinates are converted from px to microns.
    """
    df = ensure_defect_fraction(df)

    x_um, y_um = get_xy_um(df)
    vals = df[value_col].to_numpy(float)

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(x_um, y_um, c=vals, s=10, cmap="viridis")

    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(value_col)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] spatial map saved: {out_path}")


def plot_rgb_fusion(df: pd.DataFrame, title: str, out_path: Path):
    """
    RGB fusion scatter:
      R = defect_fraction
      G = entropy_hm_(bits)
      B = circularity_distortion
    Coordinates in microns.
    """
    df = ensure_defect_fraction(df)

    # column names in your anisotropy_merged CSVs
    if "entropy_hm_(bits)" not in df.columns:
        raise KeyError("entropy_hm_(bits) column not found in dataframe.")
    if "circularity_distortion" not in df.columns:
        raise KeyError("circularity_distortion column not found in dataframe.")

    r = minmax_norm(df["defect_fraction"])
    g = minmax_norm(df["entropy_hm_(bits)"])
    b = minmax_norm(df["circularity_distortion"])

    rgb = np.stack([r, g, b], axis=1)
    rgb = np.clip(rgb, 0.0, 1.0)

    x_um, y_um = get_xy_um(df)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x_um, y_um, c=rgb, s=10)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] RGB fusion map saved: {out_path}")


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Spatial descriptor maps and RGB fusion for FAPI and FAPI–TEMPO."
    )
    parser.add_argument(
        "--fapi-csv",
        required=True,
        help="Path to anisotropy_merged_FAPI_merged.csv",
    )
    parser.add_argument(
        "--tempo-csv",
        required=True,
        help="Path to anisotropy_merged_FAPITEMPO_merged.csv",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for PNG files",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fapi = pd.read_csv(args.fapi_csv)
    tempo = pd.read_csv(args.tempo_csv)

    # --- spatial maps of A_polar ---
    if "A_polar" not in fapi.columns or "A_polar" not in tempo.columns:
        raise KeyError("A_polar column not found in one of the CSVs.")

    plot_spatial_map(
        df=fapi,
        value_col="A_polar",
        title="FAPI: spatial map of A_polar",
        out_path=out_dir / "FAPI_spatial_A_polar.png",
    )
    plot_spatial_map(
        df=tempo,
        value_col="A_polar",
        title="FAPI–TEMPO: spatial map of A_polar",
        out_path=out_dir / "FAPITEMPO_spatial_A_polar.png",
    )

    # --- RGB fusion maps (φ, entropy, circ_dist) ---
    plot_rgb_fusion(
        df=fapi,
        title="FAPI: RGB fusion (R=φ, G=entropy, B=circ_dist)",
        out_path=out_dir / "FAPI_rgb_phi_entropy_circ.png",
    )
    plot_rgb_fusion(
        df=tempo,
        title="FAPI–TEMPO: RGB fusion (R=φ, G=entropy, B=circ_dist)",
        out_path=out_dir / "FAPITEMPO_rgb_phi_entropy_circ.png",
    )

    # --- example: spatial map of circularity_distortion for TEMPO only ---
    plot_spatial_map(
        df=tempo,
        value_col="circularity_distortion",
        title="FAPI–TEMPO: spatial map of circularity_distortion",
        out_path=out_dir / "FAPITEMPO_spatial_circ_dist.png",
    )

    print("[DONE] All spatial descriptor plots generated.")


if __name__ == "__main__":
    main()
