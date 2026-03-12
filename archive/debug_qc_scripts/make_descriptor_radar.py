#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_descriptor_radar.py

Compute microstructural descriptors for FAPI and FAPI–TEMPO
from the CSV exports and generate a radar chart comparing them.

Expected files (in base_dir):
  - crystal_metrics.csv                 (FAPI)
  - crystal_metrics 1.csv               (FAPI–TEMPO)
  - anisotropy_out_FAPI_anisotropy_per_grain.csv
  - anisotropy_out_FAPITEMPO_anisotropy_per_grain.csv
  - per_grain_metrics_FAPI.csv
  - per_grain_metrics_FAPI-TEMPO.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # ------------------------------------------------------------------
    # 1) Paths  (UPDATE DONE HERE)
    # ------------------------------------------------------------------
    base_dir = Path(r"D:\SWITCHdrive\Institution\Segmentation%datset stats")

    cm_fapi_path = base_dir / "crystal_metrics.csv"
    cm_tempo_path = base_dir / "crystal_metrics 1.csv"

    aniso_fapi_path = base_dir / "anisotropy_out_FAPI_anisotropy_per_grain.csv"
    aniso_tempo_path = base_dir / "anisotropy_out_FAPITEMPO_anisotropy_per_grain.csv"

    pg_fapi_path = base_dir / "per_grain_metrics_FAPI.csv"
    pg_tempo_path = base_dir / "per_grain_metrics_FAPI-TEMPO.csv"

    # ------------------------------------------------------------------
    # 2) Load data
    # ------------------------------------------------------------------
    cm_fapi = pd.read_csv(cm_fapi_path)
    cm_tempo = pd.read_csv(cm_tempo_path)

    aniso_fapi = pd.read_csv(aniso_fapi_path)
    aniso_tempo = pd.read_csv(aniso_tempo_path)

    pg_fapi = pd.read_csv(pg_fapi_path)
    pg_tempo = pd.read_csv(pg_tempo_path)

    # ------------------------------------------------------------------
    # 3) Compute descriptors
    # ------------------------------------------------------------------
    # 3.1 Shape anisotropy A_polar
    A_polar_f_mean = aniso_fapi["A_polar"].mean()
    A_polar_f_std = aniso_fapi["A_polar"].std()

    A_polar_t_mean = aniso_tempo["A_polar"].mean()
    A_polar_t_std = aniso_tempo["A_polar"].std()

    # 3.2 Circularity distortion
    circ_f_mean = cm_fapi["circularity_distortion"].mean()
    circ_f_std = cm_fapi["circularity_distortion"].std()

    circ_t_mean = cm_tempo["circularity_distortion"].mean()
    circ_t_std = cm_tempo["circularity_distortion"].std()

    # 3.3 Shannon entropy H (bits)
    H_f_mean = cm_fapi["entropy_hm_(bits)"].mean()
    H_f_std = cm_fapi["entropy_hm_(bits)"].std()

    H_t_mean = cm_tempo["entropy_hm_(bits)"].mean()
    H_t_std = cm_tempo["entropy_hm_(bits)"].std()

    # 3.4 Defect fraction φ = defects_area / area
    phi_f = cm_fapi["defects_area_(µm²)"] / cm_fapi["area_(µm²)"]
    phi_t = cm_tempo["defects_area_(µm²)"] / cm_tempo["area_(µm²)"]

    phi_f_mean = phi_f.mean()
    phi_f_std = phi_f.std()

    phi_t_mean = phi_t.mean()
    phi_t_std = phi_t.std()

    # 3.5 Kinetic heterogeneity: CV(v_eff) from per_grain_metrics
    veff_f = pg_fapi["v_eff_um_per_s"]
    veff_t = pg_tempo["v_eff_um_per_s"]

    cv_f = veff_f.std() / veff_f.mean()
    cv_t = veff_t.std() / veff_t.mean()

    print("FAPI descriptors:")
    print(f"  A_polar = {A_polar_f_mean:.3f} ± {A_polar_f_std:.3f}")
    print(f"  circ_dist = {circ_f_mean:.3f} ± {circ_f_std:.3f}")
    print(f"  H = {H_f_mean:.3f} ± {H_f_std:.3f}")
    print(f"  phi = {phi_f_mean:.4f} ± {phi_f_std:.4f}")
    print(f"  CV(v_eff) = {cv_f:.3f}")

    print("\nFAPI–TEMPO descriptors:")
    print(f"  A_polar = {A_polar_t_mean:.3f} ± {A_polar_t_std:.3f}")
    print(f"  circ_dist = {circ_t_mean:.3f} ± {circ_t_std:.3f}")
    print(f"  H = {H_t_mean:.3f} ± {H_t_std:.3f}")
    print(f"  phi = {phi_t_mean:.4f} ± {phi_t_std:.4f}")
    print(f"  CV(v_eff) = {cv_t:.3f}")

    # ------------------------------------------------------------------
    # 4) Prepare data for radar chart (normalised)
    # ------------------------------------------------------------------
    categories = [
        "Shape anisotropy\n$A_{\\mathrm{polar}}$",
        "Circularity\n distortion",
        "Entropy\n$H$ (bits)",
        "Defect\n fraction $\\phi$",
        "Kinetic\nheterogeneity\nCV($v_{\\mathrm{eff}}$)",
    ]
    n_cat = len(categories)

    vals_f_raw = [
        A_polar_f_mean,
        circ_f_mean,
        H_f_mean,
        phi_f_mean,
        cv_f,
    ]
    vals_t_raw = [
        A_polar_t_mean,
        circ_t_mean,
        H_t_mean,
        phi_t_mean,
        cv_t,
    ]

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

    vals_f_plot = vals_f_norm + [vals_f_norm[0]]
    vals_t_plot = vals_t_norm + [vals_t_norm[0]]

    angles = np.linspace(0, 2 * np.pi, n_cat, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    # ------------------------------------------------------------------
    # 5) Plot radar chart
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.0, 5.0), subplot_kw=dict(polar=True))

    ax.plot(angles, vals_f_plot, linewidth=2, label="FAPI")
    ax.fill(angles, vals_f_plot, alpha=0.2)

    ax.plot(angles, vals_t_plot, linewidth=2, linestyle="--", label="FAPI–TEMPO")
    ax.fill(angles, vals_t_plot, alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)

    ax.set_yticklabels([])
    ax.set_ylim(0, 1.05)

    ax.set_title("Normalised microstructural descriptors\nFAPI vs FAPI–TEMPO", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), fontsize=9)

    plt.tight_layout()
    out_path = base_dir / "descriptor_radar_FAPI_FAPITEMPO.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"\n[OK] Radar chart saved to: {out_path}")


if __name__ == "__main__":
    main()
