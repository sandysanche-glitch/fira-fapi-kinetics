#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
descriptor_radar_from_crystal_metrics_only.py

Compute simple microstructural descriptors for FAPI and FAPI–TEMPO
using ONLY the two crystal_metrics CSV files, and generate a radar
chart comparing them.

Descriptors (per dataset):
  - Mean circularity distortion ± std
  - Mean Shannon entropy H (bits) ± std
  - Mean defect fraction φ ± std

Expected files in base_dir:
  - crystal_metrics.csv        (FAPI)
  - crystal_metrics 1.csv      (FAPI–TEMPO)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # ------------------------------------------------------------------
    # 1) Paths: UPDATED to your folder
    # ------------------------------------------------------------------
    base_dir = Path(r"D:\SWITCHdrive\Institution\Segmentation%datset stats")

    fapi_path = base_dir / "crystal_metrics.csv"
    tempo_path = base_dir / "crystal_metrics 1.csv"

    # ------------------------------------------------------------------
    # 2) Load data
    # ------------------------------------------------------------------
    fapi = pd.read_csv(fapi_path)
    tempo = pd.read_csv(tempo_path)

    # Basic sanity check
    required_cols = [
        "circularity_distortion",
        "entropy_hm_(bits)",
        "defects_area_(µm²)",
        "area_(µm²)",
    ]
    for col in required_cols:
        if col not in fapi.columns:
            raise KeyError(f"Column '{col}' not found in FAPI CSV.")
        if col not in tempo.columns:
            raise KeyError(f"Column '{col}' not found in FAPI–TEMPO CSV.")

    # ------------------------------------------------------------------
    # 3) Compute descriptors
    # ------------------------------------------------------------------
    # 3.1 Circularity distortion
    circ_f_mean = fapi["circularity_distortion"].mean()
    circ_f_std = fapi["circularity_distortion"].std()

    circ_t_mean = tempo["circularity_distortion"].mean()
    circ_t_std = tempo["circularity_distortion"].std()

    # 3.2 Shannon entropy (bits)
    H_f_mean = fapi["entropy_hm_(bits)"].mean()
    H_f_std = fapi["entropy_hm_(bits)"].std()

    H_t_mean = tempo["entropy_hm_(bits)"].mean()
    H_t_std = tempo["entropy_hm_(bits)"].std()

    # 3.3 Defect fraction φ = defects_area / area
    phi_f = fapi["defects_area_(µm²)"] / fapi["area_(µm²)"]
    phi_t = tempo["defects_area_(µm²)"] / tempo["area_(µm²)"]

    phi_f_mean = phi_f.mean()
    phi_f_std = phi_f.std()

    phi_t_mean = phi_t.mean()
    phi_t_std = phi_t.std()

    print("FAPI (from crystal_metrics.csv):")
    print(f"  circularity_distortion = {circ_f_mean:.3f} ± {circ_f_std:.3f}")
    print(f"  entropy_hm_(bits)      = {H_f_mean:.3f} ± {H_f_std:.3f}")
    print(f"  defect fraction φ      = {phi_f_mean:.4f} ± {phi_f_std:.4f}")

    print("\nFAPI–TEMPO (from crystal_metrics 1.csv):")
    print(f"  circularity_distortion = {circ_t_mean:.3f} ± {circ_t_std:.3f}")
    print(f"  entropy_hm_(bits)      = {H_t_mean:.3f} ± {H_t_std:.3f}")
    print(f"  defect fraction φ      = {phi_t_mean:.4f} ± {phi_t_std:.4f}")

    # ------------------------------------------------------------------
    # 4) Prepare data for radar chart (normalized)
    # ------------------------------------------------------------------
    categories = [
        "Circularity\n distortion",
        "Entropy\n$H$ (bits)",
        "Defect\n fraction $\\phi$",
    ]
    n_cat = len(categories)

    vals_f_raw = [
        circ_f_mean,
        H_f_mean,
        phi_f_mean,
    ]
    vals_t_raw = [
        circ_t_mean,
        H_t_mean,
        phi_t_mean,
    ]

    # Normalise each axis by max(FAPI, TEMPO) for that descriptor
    vals_f_norm = []
    vals_t_norm = []
    for vf, vt in zip(vals_f_raw, vals_t_raw):
        maxv = max(vf, vt)
        if maxv > 0.0:
            vals_f_norm.append(vf / maxv)
            vals_t_norm.append(vt / maxv)
        else:
            vals_f_norm.append(0.0)
            vals_t_norm.append(0.0)

    # Close the polygons
    vals_f_plot = vals_f_norm + [vals_f_norm[0]]
    vals_t_plot = vals_t_norm + [vals_t_norm[0]]

    angles = np.linspace(0, 2 * np.pi, n_cat, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    # ------------------------------------------------------------------
    # 5) Plot radar chart
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.0, 5.0), subplot_kw=dict(polar=True))

    ax.plot(angles, vals_f_plot, linewidth=2, label="FAPI")
    ax.fill(angles, vals_f_plot, alpha=0.25)

    ax.plot(angles, vals_t_plot, linewidth=2, linestyle="--", label="FAPI–TEMPO")
    ax.fill(angles, vals_t_plot, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)

    ax.set_yticklabels([])   # hide radial labels for cleaner look
    ax.set_ylim(0, 1.05)

    ax.set_title(
        "Normalized descriptors",
        pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), fontsize=9)

    plt.tight_layout()
    out_path = base_dir / "descriptor_radar_from_crystal_metrics_only.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"\n[OK] Radar chart saved to: {out_path}")


if __name__ == "__main__":
    main()
