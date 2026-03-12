#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from pycocotools import mask as maskUtils
except ImportError as e:
    raise ImportError(
        "pycocotools is required.\n"
        "Windows: pip install pycocotools-windows\n"
        "Linux/macOS: pip install pycocotools"
    ) from e


# ------------------------------------------------------------
# basic helpers
# ------------------------------------------------------------
def robust_num(s):
    return pd.to_numeric(s, errors="coerce")


def normalize_key(x: str) -> str:
    x = str(x).strip().replace("\\", "/").split("/")[-1]
    return x


def decode_segmentation_rle(seg: dict) -> np.ndarray:
    rle = {
        "size": seg["size"],
        "counts": seg["counts"].encode("utf-8") if isinstance(seg["counts"], str) else seg["counts"],
    }
    m = maskUtils.decode(rle)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(bool)


def compute_mask_geometry(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if ys.size < 3:
        return None

    cy = ys.mean()
    cx = xs.mean()
    area_px = float(mask.sum())
    r_eq_px = float(np.sqrt(area_px / np.pi))
    return {
        "centroid_x_px": float(cx),
        "centroid_y_px": float(cy),
        "area_px": area_px,
        "equiv_radius_px": r_eq_px,
    }


def pairwise_nn_distance(x: np.ndarray, y: np.ndarray, block: int = 4000) -> np.ndarray:
    pts = np.column_stack([x, y]).astype(float)
    n = len(pts)
    out = np.full(n, np.nan, dtype=float)

    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        A = pts[i0:i1]
        dx = A[:, None, 0] - pts[None, :, 0]
        dy = A[:, None, 1] - pts[None, :, 1]
        d2 = dx * dx + dy * dy

        rows = np.arange(i0, i1) - i0
        cols = np.arange(i0, i1)
        d2[rows, cols] = np.inf

        out[i0:i1] = np.sqrt(np.min(d2, axis=1))

    return out


# ------------------------------------------------------------
# extract centroids + NN from JSON masks
# ------------------------------------------------------------
def extract_geometry_from_json_dir(json_dir: Path, min_pixels: int = 10) -> pd.DataFrame:
    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"No JSON files found in {json_dir}")

    rows = []

    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            continue

        image_id = jf.stem

        for i, ann in enumerate(data):
            seg = ann.get("segmentation", None)
            if not isinstance(seg, dict) or "counts" not in seg:
                continue

            mask_name = ann.get("mask_name", f"{image_id}_{i}.png")

            try:
                mask_bin = decode_segmentation_rle(seg)
            except Exception as e:
                print(f"[WARN] Failed RLE decode in {jf.name}, idx={i}: {e}")
                continue

            if mask_bin.sum() < min_pixels:
                continue

            geom = compute_mask_geometry(mask_bin)
            if geom is None:
                continue

            rows.append(
                {
                    "json_file": jf.name,
                    "image_id": image_id,
                    "grain_index": i,
                    "mask_name": normalize_key(mask_name),
                    **geom,
                }
            )

    if not rows:
        raise RuntimeError(f"No usable grain masks found in {json_dir}")

    df = pd.DataFrame(rows)

    # NN within each JSON image
    df["nn_distance_px"] = np.nan
    for img_id, g in df.groupby("image_id", sort=False):
        idx = g.index.to_numpy()
        x = g["centroid_x_px"].to_numpy(float)
        y = g["centroid_y_px"].to_numpy(float)

        if len(g) < 2:
            continue

        nn = pairwise_nn_distance(x, y)
        df.loc[idx, "nn_distance_px"] = nn

    # simple impingement/crowding proxies
    # 1) inverse NN
    with np.errstate(divide="ignore", invalid="ignore"):
        df["crowding_index_px_inv"] = np.where(df["nn_distance_px"] > 0, 1.0 / df["nn_distance_px"], np.nan)

    # 2) NN normalized by own grain size
    with np.errstate(divide="ignore", invalid="ignore"):
        df["nn_over_req"] = np.where(df["equiv_radius_px"] > 0, df["nn_distance_px"] / df["equiv_radius_px"], np.nan)

    # 3) impingement index: smaller center spacing relative to radius -> more crowded
    with np.errstate(divide="ignore", invalid="ignore"):
        df["impingement_index"] = np.where(df["nn_over_req"] > 0, 1.0 / df["nn_over_req"], np.nan)

    return df


# ------------------------------------------------------------
# merge with v_eff tables
# ------------------------------------------------------------
def prepare_with_veff(df: pd.DataFrame, file_col: str, veff_col: str) -> pd.DataFrame:
    if file_col not in df.columns:
        raise KeyError(f"Missing file key column '{file_col}'")
    if veff_col not in df.columns:
        raise KeyError(f"Missing v_eff column '{veff_col}'")

    out = df.copy()
    out["_merge_key"] = out[file_col].map(normalize_key)
    return out


def merge_geometry_with_veff(geom_df: pd.DataFrame, veff_df: pd.DataFrame) -> pd.DataFrame:
    g = geom_df.copy()
    g["_merge_key"] = g["mask_name"].map(normalize_key)

    merged = veff_df.merge(
        g.drop(columns=["mask_name"]),
        how="left",
        on="_merge_key",
        suffixes=("", "_geom"),
    )
    return merged


# ------------------------------------------------------------
# plotting
# ------------------------------------------------------------
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


def make_plot(x_f, y_f, x_t, y_t, xlabel, ylabel, title, out_path: Path, nbins=25, legend_loc="best"):
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


# ------------------------------------------------------------
# summary
# ------------------------------------------------------------
def summarize(df: pd.DataFrame, sample: str, veff_col: str) -> dict:
    nn = robust_num(df["nn_distance_px"]).dropna().to_numpy(float)
    imp = robust_num(df["impingement_index"]).dropna().to_numpy(float)
    v = robust_num(df[veff_col]).dropna().to_numpy(float)
    return {
        "sample": sample,
        "n_rows": len(df),
        "n_veff_valid": len(v),
        "n_nn_valid": len(nn),
        "nn_distance_px_mean": float(np.mean(nn)) if len(nn) else np.nan,
        "nn_distance_px_median": float(np.median(nn)) if len(nn) else np.nan,
        "impingement_index_mean": float(np.mean(imp)) if len(imp) else np.nan,
        "impingement_index_median": float(np.median(imp)) if len(imp) else np.nan,
    }


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Compute local crowding/impingement from JSON masks and merge to with_veff tables."
    )
    ap.add_argument("--fapi-json-dir", required=True, help="Directory with FAPI list-style JSON files")
    ap.add_argument("--tempo-json-dir", required=True, help="Directory with FAPI-TEMPO list-style JSON files")
    ap.add_argument("--fapi-csv", required=True, help="FAPI with_veff CSV")
    ap.add_argument("--tempo-csv", required=True, help="FAPI-TEMPO with_veff CSV")
    ap.add_argument("--out-prefix", required=True, help="Output prefix")
    ap.add_argument("--file-col", default="file_name", help="Key column in with_veff CSV for mask_name matching")
    ap.add_argument("--veff-col", default="v_eff_um_per_ms", help="v_eff column")
    ap.add_argument("--min-pixels", type=int, default=10, help="Minimum decoded-mask area")
    ap.add_argument("--nbins", type=int, default=25, help="Bins for mean±std overlays")
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()

    out_prefix = Path(args.out_prefix)
    out_dir = out_prefix.parent if out_prefix.parent != Path("") else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_prefix.name

    # geometry from JSON
    print("[INFO] Extracting FAPI geometry from JSON...")
    geom_fapi = extract_geometry_from_json_dir(Path(args.fapi_json_dir), min_pixels=args.min_pixels)
    print("[INFO] Extracting FAPI-TEMPO geometry from JSON...")
    geom_tempo = extract_geometry_from_json_dir(Path(args.tempo_json_dir), min_pixels=args.min_pixels)

    geom_fapi.to_csv(out_dir / f"{base}_FAPI_geometry_from_json.csv", index=False)
    geom_tempo.to_csv(out_dir / f"{base}_FAPITEMPO_geometry_from_json.csv", index=False)

    # load v_eff tables
    fapi = pd.read_csv(args.fapi_csv)
    tempo = pd.read_csv(args.tempo_csv)
    fapi = prepare_with_veff(fapi, args.file_col, args.veff_col)
    tempo = prepare_with_veff(tempo, args.file_col, args.veff_col)

    # merge
    fapi_m = merge_geometry_with_veff(geom_fapi, fapi)
    tempo_m = merge_geometry_with_veff(geom_tempo, tempo)

    fapi_out = out_dir / f"{base}_FAPI_with_crowding.csv"
    tempo_out = out_dir / f"{base}_FAPITEMPO_with_crowding.csv"
    fapi_m.to_csv(fapi_out, index=False)
    tempo_m.to_csv(tempo_out, index=False)

    # match summary
    summary_df = pd.DataFrame(
        [
            summarize(fapi_m, "FAPI", args.veff_col),
            summarize(tempo_m, "FAPI-TEMPO", args.veff_col),
        ]
    )
    summary_out = out_dir / f"{base}_crowding_summary.csv"
    summary_df.to_csv(summary_out, index=False)

    print(f"[OK] Exported: {fapi_out}")
    print(f"[OK] Exported: {tempo_out}")
    print(f"[OK] Exported: {summary_out}")
    print("\nCrowding summary:")
    print(summary_df.to_string(index=False))

    if args.skip_plots:
        print("[DONE] Geometry extraction + merge complete (plots skipped).")
        return

    # plots
    veff_f = robust_num(fapi_m[args.veff_col]).to_numpy(float)
    veff_t = robust_num(tempo_m[args.veff_col]).to_numpy(float)

    # 1) v_eff vs NN distance
    x_f, y_f = finite_pair(fapi_m["nn_distance_px"], veff_f)
    x_t, y_t = finite_pair(tempo_m["nn_distance_px"], veff_t)
    make_plot(
        x_f, y_f, x_t, y_t,
        xlabel="Nearest-neighbor distance (px)",
        ylabel=r"Effective growth rate $v_{\mathrm{eff}}$ (µm/ms)",
        title=r"$v_{\mathrm{eff}}$ vs nearest-neighbor distance",
        out_path=out_dir / f"{base}_veff_vs_nn_distance.png",
        nbins=args.nbins,
        legend_loc="upper right",
    )

    # 2) v_eff vs impingement index
    x_f, y_f = finite_pair(fapi_m["impingement_index"], veff_f)
    x_t, y_t = finite_pair(tempo_m["impingement_index"], veff_t)
    make_plot(
        x_f, y_f, x_t, y_t,
        xlabel=r"Impingement index $(R_{\mathrm{eq}}/\mathrm{NN})$",
        ylabel=r"Effective growth rate $v_{\mathrm{eff}}$ (µm/ms)",
        title=r"$v_{\mathrm{eff}}$ vs local impingement index",
        out_path=out_dir / f"{base}_veff_vs_impingement_index.png",
        nbins=args.nbins,
        legend_loc="upper right",
    )

    print("[DONE] JSON-based crowding descriptor complete.")


if __name__ == "__main__":
    main()