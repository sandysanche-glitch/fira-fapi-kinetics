#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
merge_anisotropy_and_metrics.py

Merge per-grain anisotropy (from JSON masks) with nucleation-centre-based
metrics (crystal_metrics*.csv) using file_name, and regenerate:

  1) Histogram of circularity_distortion (FAPI vs FAPI–TEMPO)
  2) Average polar profile in Cartesian coordinates
  3) Average polar profile as a rose plot (polar coordinates)

Usage example (Windows):

python merge_anisotropy_and_metrics.py ^
  --ani-fapi anisotropy_out_FAPI_anisotropy_per_grain.csv ^
  --ani-tempo anisotropy_out_FAPITEMPO_anisotropy_per_grain.csv ^
  --metrics-fapi crystal_metrics.csv ^
  --metrics-tempo "crystal_metrics 1.csv" ^
  --polar-fapi anisotropy_out_FAPI_polar_profile.csv ^
  --polar-tempo anisotropy_out_FAPITEMPO_polar_profile.csv ^
  --out-prefix merged_anisotropy

"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_file_name_column(df, dataset_label=""):
    """
    Ensure a 'file_name' column exists in the anisotropy dataframe.

    Expected columns from anisotropy_out_*_anisotropy_per_grain.csv:
      - 'json_file' (e.g. 'FAPI_0.json')
      - 'grain_index' (int)

    We build:
      file_name = stem(json_file) + '_' + grain_index
                 e.g. FAPI_0_3
    """
    if "file_name" in df.columns:
        return df

    if "json_file" not in df.columns or "grain_index" not in df.columns:
        raise ValueError(
            f"{dataset_label}: cannot construct file_name; "
            "expected 'json_file' and 'grain_index' columns."
        )

    stem = df["json_file"].astype(str).str.replace(".json", "", regex=False)
    df["file_name"] = stem + "_" + df["grain_index"].astype(str)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Merge anisotropy_out_*_anisotropy_per_grain.csv with crystal_metrics*.csv and remake plots."
    )
    parser.add_argument("--ani-fapi", required=True,
                        help="anisotropy_out_FAPI_anisotropy_per_grain.csv")
    parser.add_argument("--ani-tempo", required=True,
                        help="anisotropy_out_FAPITEMPO_anisotropy_per_grain.csv")
    parser.add_argument("--metrics-fapi", required=True,
                        help="crystal_metrics.csv (FAPI)")
    parser.add_argument("--metrics-tempo", required=True,
                        help='crystal_metrics for FAPI–TEMPO (e.g. "crystal_metrics 1.csv")')
    parser.add_argument("--polar-fapi", required=True,
                        help="anisotropy_out_FAPI_polar_profile.csv")
    parser.add_argument("--polar-tempo", required=True,
                        help="anisotropy_out_FAPITEMPO_polar_profile.csv")
    parser.add_argument("--out-prefix", required=True,
                        help="Output prefix for merged CSVs and figures")

    args = parser.parse_args()
    out_prefix = Path(args.out_prefix)
    out_dir = out_prefix.parent if out_prefix.parent != Path("") else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_prefix.name

    # ------------------------
    # 1) Load and merge tables
    # ------------------------
    print("[INFO] Loading per-grain anisotropy CSVs...")
    ani_fapi = pd.read_csv(args.ani_fapi)
    ani_tempo = pd.read_csv(args.ani_tempo)

    ani_fapi = ensure_file_name_column(ani_fapi, "FAPI anisotropy")
    ani_tempo = ensure_file_name_column(ani_tempo, "FAPI–TEMPO anisotropy")

    print("[INFO] Loading crystal_metrics CSVs...")
    met_fapi = pd.read_csv(args.metrics_fapi)
    met_tempo = pd.read_csv(args.metrics_tempo)

    if "file_name" not in met_fapi.columns or "file_name" not in met_tempo.columns:
        raise ValueError("crystal_metrics CSVs must contain a 'file_name' column.")

    if "circularity_distortion" not in met_fapi.columns or "circularity_distortion" not in met_tempo.columns:
        raise ValueError("crystal_metrics CSVs must contain 'circularity_distortion' column.")

    print("[INFO] Merging on file_name...")
    merged_fapi = pd.merge(
        ani_fapi, met_fapi, on="file_name", how="inner", suffixes=("", "_metrics")
    )
    merged_tempo = pd.merge(
        ani_tempo, met_tempo, on="file_name", how="inner", suffixes=("", "_metrics")
    )

    merged_fapi_path = out_dir / f"{base}_FAPI_merged.csv"
    merged_tempo_path = out_dir / f"{base}_FAPITEMPO_merged.csv"
    merged_fapi.to_csv(merged_fapi_path, index=False)
    merged_tempo.to_csv(merged_tempo_path, index=False)

    print(f"[OK] Saved merged CSVs:\n  {merged_fapi_path}\n  {merged_tempo_path}")

    # ------------------------
    # 2) Histogram of circularity_distortion
    # ------------------------
    print("[INFO] Plotting circularity_distortion histogram...")
    Af = merged_fapi["circularity_distortion"].dropna()
    At = merged_tempo["circularity_distortion"].dropna()

    max_val = max(Af.max(), At.max())
    min_val = min(Af.min(), At.min(), 0.0)
    bins = np.linspace(min_val, max_val, 40)

    plt.figure(figsize=(6, 4))
    plt.hist(Af, bins=bins, histtype="step", label="FAPI")
    plt.hist(At, bins=bins, histtype="step", label="FAPI–TEMPO")
    plt.xlabel("Circularity distortion (from nucleation center)")
    plt.ylabel("Count")
    plt.title("Grain shape anisotropy (nucleation-center based)")
    plt.legend()
    plt.tight_layout()
    hist_path = out_dir / f"{base}_circularity_distortion_hist.png"
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"[OK] Saved histogram: {hist_path}")

    # ------------------------
    # 3) Average polar profiles (from existing polar_profile CSVs)
    # ------------------------
    print("[INFO] Loading polar profiles...")
    pf_fapi = pd.read_csv(args.polar_fapi)
    pf_tempo = pd.read_csv(args.polar_tempo)

    # Expected columns: theta_deg, r_norm_mean, r_norm_std
    for df, label in [(pf_fapi, "FAPI"), (pf_tempo, "FAPI–TEMPO")]:
        for col in ["theta_deg", "r_norm_mean"]:
            if col not in df.columns:
                raise ValueError(f"{label} polar_profile CSV missing '{col}' column.")

    theta_deg_fapi = pf_fapi["theta_deg"].values
    rnorm_fapi = pf_fapi["r_norm_mean"].values
    theta_deg_tempo = pf_tempo["theta_deg"].values
    rnorm_tempo = pf_tempo["r_norm_mean"].values

    # 3a) Cartesian plot
    plt.figure(figsize=(6, 4))
    plt.plot(theta_deg_fapi, rnorm_fapi, label="FAPI")
    plt.plot(theta_deg_tempo, rnorm_tempo, label="FAPI–TEMPO")
    plt.xlabel("Angle (deg)")
    plt.ylabel("Normalized radius r(θ)/⟨r⟩")
    plt.title("Average grain shape in polar coordinates")
    plt.legend()
    plt.tight_layout()
    cart_path = out_dir / f"{base}_polar_avg_cartesian.png"
    plt.savefig(cart_path, dpi=300)
    plt.close()
    print(f"[OK] Saved Cartesian polar profile plot: {cart_path}")

    # 3b) Rose/polar plot
    theta_rad_fapi = np.radians(theta_deg_fapi)
    theta_rad_tempo = np.radians(theta_deg_tempo)

    # close the curve for nicer polar plot
    th_fapi = np.append(theta_rad_fapi, theta_rad_fapi[0])
    r_fapi = np.append(rnorm_fapi, rnorm_fapi[0])
    th_tempo = np.append(theta_rad_tempo, theta_rad_tempo[0])
    r_tempo = np.append(rnorm_tempo, rnorm_tempo[0])

    fig = plt.figure(figsize=(6, 4.5))
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(th_fapi, r_fapi, label="FAPI")
    ax.plot(th_tempo, r_tempo, label="FAPI–TEMPO")
    ax.set_title("Average normalized grain shape (polar)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    rose_path = out_dir / f"{base}_polar_avg_rose.png"
    plt.savefig(rose_path, dpi=300)
    plt.close()
    print(f"[OK] Saved polar (rose) plot: {rose_path}")

    print("[DONE] All merged plots generated.")


if __name__ == "__main__":
    main()
