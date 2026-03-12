#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
spatial_descriptors_and_clustering.py

Tools to explore spatial and descriptor-space structure of FAPI vs FAPI–TEMPO
using per-grain metrics from crystal_metrics*.csv:

  1) Spatial descriptor maps (scatter over (x, y), coloured by descriptor)
  2) Neighbourhood graphs (k-NN edges overlaid on positions)
  3) Moran's I and empirical variogram for a chosen descriptor
  4) UMAP embedding in descriptor space (FAPI vs FAPI–TEMPO)
  5) RGB fusion maps combining 3 descriptors into pseudo-PL/EBSD/profilometry maps

USAGE (example):

  python spatial_descriptors_and_clustering.py ^
      --fapi-csv "crystal_metrics.csv" ^
      --tempo-csv "crystal_metrics 1.csv" ^
      --out-prefix "spatial_desc" ^
      --x-col "cx" ^
      --y-col "cy" ^
      --entropy-col "entropy_hm_(bits)" ^
      --defect-area-col "defects_area_(µm²)" ^
      --area-col "area_(µm²)" ^
      --circ-col "circularity_distortion"

Notes:
- Requires: numpy, pandas, matplotlib, scikit-learn
- Optional: umap-learn (for UMAP embedding)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors

# Try to import UMAP; if not installed, we will skip that part gracefully
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[WARN] umap-learn not installed; UMAP plots will be skipped.")


# ----------------------------------------------------------------------
# Helper / derived descriptors
# ----------------------------------------------------------------------

def add_derived_descriptors(df, args, label):
    """
    Add derived quantities:
      - defect_fraction phi = defects_area / area
      - v_eff (if present in CSV)
    """
    df = df.copy()
    df["label"] = label

    # Defect fraction phi
    if args.defect_area_col in df.columns and args.area_col in df.columns:
        area = df[args.area_col].replace(0, np.nan)
        df["phi"] = df[args.defect_area_col] / area
    else:
        df["phi"] = np.nan
        print(f"[WARN] defect area or area column missing for {label}; phi set to NaN.")

    # v_eff if exists (otherwise NaN)
    if args.veff_col and args.veff_col in df.columns:
        df["v_eff"] = df[args.veff_col]
    else:
        df["v_eff"] = np.nan

    # circularity distortion
    if args.circ_col in df.columns:
        df["circ_dist"] = df[args.circ_col]
    else:
        df["circ_dist"] = np.nan

    # entropy
    if args.entropy_col in df.columns:
        df["entropy"] = df[args.entropy_col]
    else:
        df["entropy"] = np.nan

    # A_polar / A_tex / deltaH if they exist (optional)
    if args.apolar_col and args.apolar_col in df.columns:
        df["A_polar"] = df[args.apolar_col]
    else:
        df["A_polar"] = np.nan

    if args.atex_col and args.atex_col in df.columns:
        df["A_tex"] = df[args.atex_col]
    else:
        df["A_tex"] = np.nan

    if args.dh_col and args.dh_col in df.columns:
        df["Delta_H"] = df[args.dh_col]
    else:
        df["Delta_H"] = np.nan

    return df


# ----------------------------------------------------------------------
# 1) Spatial descriptor maps
# ----------------------------------------------------------------------

def plot_spatial_map(df, x_col, y_col, desc_col, out_path, title=None, vmin=None, vmax=None):
    """Scatter plot of grains in (x,y), coloured by descriptor desc_col."""
    dfv = df.dropna(subset=[x_col, y_col, desc_col]).copy()
    if dfv.empty:
        print(f"[WARN] No valid data for spatial map of {desc_col}. Skipping.")
        return

    x = dfv[x_col].values
    y = dfv[y_col].values
    z = dfv[desc_col].values

    if vmin is None:
        vmin = np.nanpercentile(z, 5)
    if vmax is None:
        vmax = np.nanpercentile(z, 95)
    if vmin == vmax:
        vmax = vmin + 1e-6

    fig, ax = plt.subplots(figsize=(5, 4.5))
    sc = ax.scatter(x, y, c=z, s=8, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    if title:
        ax.set_title(title)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label(desc_col)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] spatial map saved: {out_path}")


# ----------------------------------------------------------------------
# 2) Neighbourhood graphs (k-NN)
# ----------------------------------------------------------------------

def plot_knn_graph(df, x_col, y_col, desc_col, out_path, k=6, title=None):
    """Plot k-NN edges between grains, coloured by descriptor."""
    dfv = df.dropna(subset=[x_col, y_col, desc_col]).copy()
    if len(dfv) < k + 1:
        print(f"[WARN] Not enough points for k-NN (k={k}) for {desc_col}. Skipping.")
        return

    X = dfv[[x_col, y_col]].values
    z = dfv[desc_col].values

    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dists, idxs = nn.kneighbors(X)

    fig, ax = plt.subplots(figsize=(5, 4.5))

    # draw edges
    for i in range(len(dfv)):
        for j_idx in idxs[i, 1:]:  # skip self
            x0, y0 = X[i]
            x1, y1 = X[j_idx]
            ax.plot([x0, x1], [y0, y1], color="lightgray", linewidth=0.5, zorder=1)

    sc = ax.scatter(X[:, 0], X[:, 1], c=z, s=10, cmap="viridis", zorder=2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    if title:
        ax.set_title(title)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label(desc_col)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] k-NN graph saved: {out_path}")


# ----------------------------------------------------------------------
# 3) Moran's I and variogram
# ----------------------------------------------------------------------

def compute_morans_I(values, X, k=6):
    """
    Moran's I with k-NN weights.

    values : 1D array of descriptor values
    X      : (N,2) coordinates
    """
    mask = np.isfinite(values)
    values = values[mask]
    X = X[mask]
    n = len(values)
    if n < k + 1:
        return np.nan

    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dists, idxs = nn.kneighbors(X)

    # Build weight matrix in sparse fashion
    x_bar = np.mean(values)
    num = 0.0
    W = 0.0
    for i in range(n):
        for j_idx in idxs[i, 1:]:
            wij = 1.0
            W += wij
            num += wij * (values[i] - x_bar) * (values[j_idx] - x_bar)

    den = np.sum((values - x_bar) ** 2)
    if den == 0 or W == 0:
        return np.nan

    I = (n / W) * (num / den)
    return I


def compute_variogram(values, X, n_bins=10):
    """
    Empirical variogram: returns (bin_centres, gamma(h)).
    """
    mask = np.isfinite(values)
    values = values[mask]
    X = X[mask]
    n = len(values)
    if n < 2:
        return None, None

    # Use random subsampling for large N to avoid O(N^2) explosion
    max_pairs = 200000
    idx_all = np.arange(n)
    if n > 1000:
        # subsample points
        np.random.seed(0)
        idx_sub = np.random.choice(idx_all, size=1000, replace=False)
        X = X[idx_sub]
        values = values[idx_sub]
        n = len(values)

    # Compute all pair distances and semivariances
    dh_list = []
    gamma_list = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = X[i, 0] - X[j, 0]
            dy = X[i, 1] - X[j, 1]
            d = np.sqrt(dx * dx + dy * dy)
            g = 0.5 * (values[i] - values[j]) ** 2
            dh_list.append(d)
            gamma_list.append(g)

    dh = np.array(dh_list)
    gvals = np.array(gamma_list)

    if len(dh) == 0:
        return None, None

    # Bin
    d_min, d_max = np.min(dh), np.max(dh)
    bins = np.linspace(d_min, d_max, n_bins + 1)
    bin_centres = 0.5 * (bins[:-1] + bins[1:])
    gamma_bin = np.zeros_like(bin_centres)
    for i in range(n_bins):
        mask_bin = (dh >= bins[i]) & (dh < bins[i + 1])
        if np.any(mask_bin):
            gamma_bin[i] = np.mean(gvals[mask_bin])
        else:
            gamma_bin[i] = np.nan

    return bin_centres, gamma_bin


def plot_morans_and_variogram(df, x_col, y_col, desc_col, out_prefix):
    dfv = df.dropna(subset=[x_col, y_col, desc_col]).copy()
    if dfv.empty:
        print(f"[WARN] No data for Moran/variogram of {desc_col}. Skipping.")
        return

    X = dfv[[x_col, y_col]].values
    v = dfv[desc_col].values

    I = compute_morans_I(v, X, k=6)
    print(f"[INFO] Moran's I for {desc_col}: {I:.3f}")

    h, gamma_h = compute_variogram(v, X, n_bins=10)
    if h is None:
        print(f"[WARN] Variogram could not be computed for {desc_col}.")
        return

    # Plot variogram
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    ax.plot(h, gamma_h, "o-", lw=1.5)
    ax.set_xlabel("Distance (px)")
    ax.set_ylabel(r"$\gamma(h)$")
    ax.set_title(f"Variogram of {desc_col}\nMoran's I = {I:.3f}")
    plt.tight_layout()
    fig.savefig(out_prefix, dpi=300)
    plt.close(fig)
    print(f"[OK] variogram plot saved: {out_prefix}")


# ----------------------------------------------------------------------
# 4) UMAP clustering in descriptor space
# ----------------------------------------------------------------------

def plot_umap_embedding(df_combined, out_path, desc_cols):
    if not HAS_UMAP:
        print("[WARN] umap-learn not available; skipping UMAP embedding.")
        return

    dfv = df_combined.dropna(subset=desc_cols).copy()
    if dfv.empty:
        print(f"[WARN] No rows with all descriptors {desc_cols} present. Skipping UMAP.")
        return

    X = dfv[desc_cols].values
    labels = dfv["label"].values

    reducer = umap.UMAP(n_components=2, random_state=0)
    emb = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    for lab in np.unique(labels):
        mask = labels == lab
        ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=10,
            alpha=0.7,
            label=lab,
        )

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("UMAP embedding of grain descriptors")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] UMAP plot saved: {out_path}")


# ----------------------------------------------------------------------
# 5) RGB fusion maps
# ----------------------------------------------------------------------

def plot_rgb_fusion(df, x_col, y_col,
                    r_col, g_col, b_col,
                    out_path, title=None):
    dfv = df.dropna(subset=[x_col, y_col, r_col, g_col, b_col]).copy()
    if dfv.empty:
        print(f"[WARN] No data for RGB fusion map. Skipping.")
        return

    X = dfv[[x_col, y_col]].values
    R = dfv[r_col].values
    G = dfv[g_col].values
    B = dfv[b_col].values

    # Min-max normalisation each channel
    def norm01(arr):
        amin, amax = np.nanmin(arr), np.nanmax(arr)
        if amax == amin:
            return np.zeros_like(arr)
        return (arr - amin) / (amax - amin)

    Rn = norm01(R)
    Gn = norm01(G)
    Bn = norm01(B)

    colors = np.vstack([Rn, Gn, Bn]).T

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    if title:
        ax.set_title(title)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] RGB fusion map saved: {out_path}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Spatial descriptor maps, neighbourhood graphs, Moran's I, UMAP, and RGB fusion for FAPI vs FAPI–TEMPO."
    )
    parser.add_argument("--fapi-csv", required=True,
                        help="Per-grain CSV for FAPI (e.g. crystal_metrics.csv)")
    parser.add_argument("--tempo-csv", required=True,
                        help="Per-grain CSV for FAPI–TEMPO (e.g. crystal_metrics 1.csv)")
    parser.add_argument("--out-prefix", default="spatial_desc",
                        help="Prefix for all output plots")

    parser.add_argument("--x-col", default="cx",
                        help="Column with x-coordinate of grain centroid")
    parser.add_argument("--y-col", default="cy",
                        help="Column with y-coordinate of grain centroid")

    parser.add_argument("--entropy-col", default="entropy_hm_(bits)",
                        help="Column with Shannon entropy per grain")
    parser.add_argument("--defect-area-col", default="defects_area_(µm²)",
                        help="Column with defects area per grain")
    parser.add_argument("--area-col", default="area_(µm²)",
                        help="Column with grain area in µm²")
    parser.add_argument("--circ-col", default="circularity_distortion",
                        help="Column with circularity distortion")

    parser.add_argument("--veff-col", default=None,
                        help="Optional column with v_eff (if available)")
    parser.add_argument("--apolar-col", default=None,
                        help="Optional column with A_polar (if available)")
    parser.add_argument("--atex-col", default=None,
                        help="Optional column with A_tex (if available)")
    parser.add_argument("--dh-col", default=None,
                        help="Optional column with Delta_H (if available)")

    args = parser.parse_args()

    out_prefix = Path(args.out_prefix)
    out_dir = out_prefix.parent if out_prefix.parent != Path("") else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_prefix.name

    # Load CSVs
    fapi  = pd.read_csv(args.fapi_csv)
    tempo = pd.read_csv(args.tempo_csv)

    fapi  = add_derived_descriptors(fapi,  args, label="FAPI")
    tempo = add_derived_descriptors(tempo, args, label="FAPI–TEMPO")

    combined = pd.concat([fapi, tempo], ignore_index=True)

    # ------------------------------------------------------------------
    # 1) Spatial maps for a few key descriptors
    # ------------------------------------------------------------------
    for desc in ["entropy", "phi", "circ_dist"]:
        for df, lab in [(fapi, "FAPI"), (tempo, "FAPI–TEMPO")]:
            out_map = out_dir / f"{base}_spatial_{desc}_{lab.replace(' ', '')}.png"
            title = f"{lab}: spatial map of {desc}"
            plot_spatial_map(df, args.x_col, args.y_col, desc, out_map, title=title)

    # If A_polar, A_tex or Delta_H exist, also map them
    for desc in ["A_polar", "A_tex", "Delta_H"]:
        if desc in combined.columns and combined[desc].notna().any():
            for df, lab in [(fapi, "FAPI"), (tempo, "FAPI–TEMPO")]:
                out_map = out_dir / f"{base}_spatial_{desc}_{lab.replace(' ', '')}.png"
                title = f"{lab}: spatial map of {desc}"
                plot_spatial_map(df, args.x_col, args.y_col, desc, out_map, title=title)

    # ------------------------------------------------------------------
    # 2) Neighbourhood graphs for one key descriptor (e.g. phi)
    # ------------------------------------------------------------------
    for df, lab in [(fapi, "FAPI"), (tempo, "FAPI–TEMPO")]:
        out_knn = out_dir / f"{base}_knn_phi_{lab.replace(' ', '')}.png"
        plot_knn_graph(df, args.x_col, args.y_col, "phi", out_knn,
                       k=6, title=f"{lab}: k-NN graph coloured by φ")

    # ------------------------------------------------------------------
    # 3) Moran's I + variogram for entropy and phi
    # ------------------------------------------------------------------
    for desc in ["entropy", "phi"]:
        for df, lab in [(fapi, "FAPI"), (tempo, "FAPI–TEMPO")]:
            out_var = out_dir / f"{base}_variogram_{desc}_{lab.replace(' ', '')}.png"
            plot_morans_and_variogram(df, args.x_col, args.y_col, desc, out_var)

    # ------------------------------------------------------------------
    # 4) UMAP embedding in descriptor space
    #    Use whatever descriptors exist and are not all-NaN
    # ------------------------------------------------------------------
    desc_candidates = ["A_polar", "A_tex", "Delta_H", "phi", "circ_dist", "entropy"]
    desc_cols = [c for c in desc_candidates
                 if c in combined.columns and combined[c].notna().any()]
    if len(desc_cols) >= 2:
        out_umap = out_dir / f"{base}_umap_descriptors.png"
        plot_umap_embedding(combined, out_umap, desc_cols)
    else:
        print("[WARN] Not enough valid descriptor columns for UMAP (need ≥ 2).")

    # ------------------------------------------------------------------
    # 5) RGB fusion maps: example mapping
    #    R = phi, G = entropy, B = circ_dist
    # ------------------------------------------------------------------
    for df, lab in [(fapi, "FAPI"), (tempo, "FAPI–TEMPO")]:
        out_rgb = out_dir / f"{base}_rgb_phi_entropy_circ_{lab.replace(' ', '')}.png"
        title = f"{lab}: RGB fusion (R=φ, G=entropy, B=circ_dist)"
        plot_rgb_fusion(df, args.x_col, args.y_col,
                        "phi", "entropy", "circ_dist",
                        out_rgb, title=title)

    print("[DONE] All spatial / clustering analyses completed.")


if __name__ == "__main__":
    main()
