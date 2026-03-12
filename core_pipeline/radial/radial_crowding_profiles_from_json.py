#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
radial_crowding_profiles_from_json.py

Compute annulus-conditioned radial profiles (proxy) for grain-level crowding metrics:
  - median NN distance vs r/R
  - median impingement index (R_eq/NN) vs r/R
With QC outputs:
  - n_grains_per_bin
  - mean_ring_pixels_per_grain_bin

Inputs:
  - JSON dirs containing list-style or COCO-style annotations with RLE segmentation.
  - "with_crowding" CSVs (recommended): must include file_name (merge key), nn_dist_px, impingement_index.

Merge key convention:
  file_name = <json_stem> + "_" + <annotation_index_in_annotation_list>

Outputs:
  - <out-prefix>_radial_crowding_profiles.csv
  - <out-prefix>_radial_nn_median.png
  - <out-prefix>_radial_impingement_median.png
  - <out-prefix>_radial_crowding_qc.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from pycocotools import mask as maskUtils
except ImportError as e:
    raise ImportError(
        "pycocotools is required.\n"
        "conda install -c conda-forge pycocotools\n"
        "or pip install pycocotools / pycocotools-windows"
    ) from e


# -------------------------
# JSON utilities
# -------------------------
def iter_annotations(obj: Any) -> List[Dict[str, Any]]:
    """Support list-style JSON or COCO dict JSON."""
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict) and isinstance(obj.get("annotations", None), list):
        return [x for x in obj["annotations"] if isinstance(x, dict)]
    return []


def as_rle(seg: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(seg, dict) or "size" not in seg or "counts" not in seg:
        raise ValueError("segmentation must be an RLE dict with keys size, counts")
    rle = {"size": seg["size"], "counts": seg["counts"]}
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return rle


def decode_mask(seg: Dict[str, Any]) -> np.ndarray:
    rle = as_rle(seg)
    m = maskUtils.decode(rle)
    if m.ndim == 3:
        m = m[..., 0]
    return m.astype(bool)


# -------------------------
# Merge-key normalization
# -------------------------
def normalize_key(x: str) -> str:
    x = str(x).strip().replace("\\", "/").split("/")[-1]
    xl = x.lower()
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".json"]:
        if xl.endswith(ext):
            x = x[: -len(ext)]
            break
    return x


# -------------------------
# Annulus membership per grain
# -------------------------
def grain_annulus_membership(mask: np.ndarray, bin_edges: np.ndarray, min_ring_pixels: int) -> Tuple[List[int], List[int]]:
    """
    Return:
      - bins used by this grain (indices)
      - ring pixel counts (same length as bins)
    """
    ys, xs = np.nonzero(mask)
    if xs.size < min_ring_pixels:
        return [], []

    cx = xs.mean()
    cy = ys.mean()

    dx = xs - cx
    dy = ys - cy
    r = np.sqrt(dx * dx + dy * dy)
    R = float(r.max())
    if not np.isfinite(R) or R <= 0:
        return [], []

    rn = r / R
    n_bins = len(bin_edges) - 1

    bins = []
    ring_counts = []
    for b in range(n_bins):
        r0, r1 = bin_edges[b], bin_edges[b + 1]
        if b < n_bins - 1:
            sel = (rn >= r0) & (rn < r1)
        else:
            sel = (rn >= r0) & (rn <= r1)
        n_pix = int(np.count_nonzero(sel))
        if n_pix >= min_ring_pixels:
            bins.append(b)
            ring_counts.append(n_pix)

    return bins, ring_counts


# -------------------------
# Core computation
# -------------------------
def compute_radial_profiles(
    json_dir: Path,
    crowding_df: pd.DataFrame,
    sample_label: str,
    n_bins: int,
    min_grain_pixels: int,
    min_ring_pixels: int,
    nn_col: str,
    imp_col: str,
) -> Dict[str, Any]:
    """
    For each grain (JSON annotation), assign its grain-level NN and impingement to
    the annulus bins it occupies. Aggregate per-bin as median and IQR.
    """
    # Map grain key -> (nn, imp)
    if "file_name" not in crowding_df.columns:
        raise KeyError("crowding CSV must contain a 'file_name' column for merging.")

    crowding_df = crowding_df.copy()
    crowding_df["_k"] = crowding_df["file_name"].map(normalize_key)

    if nn_col not in crowding_df.columns:
        raise KeyError(f"Missing nn column '{nn_col}' in crowding CSV.")
    if imp_col not in crowding_df.columns:
        raise KeyError(f"Missing impingement column '{imp_col}' in crowding CSV.")

    nn_map = dict(zip(crowding_df["_k"], pd.to_numeric(crowding_df[nn_col], errors="coerce")))
    imp_map = dict(zip(crowding_df["_k"], pd.to_numeric(crowding_df[imp_col], errors="coerce")))

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"No JSON files found in {json_dir}")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=float)
    r_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # per-bin collected values
    per_bin_nn: List[List[float]] = [[] for _ in range(n_bins)]
    per_bin_imp: List[List[float]] = [[] for _ in range(n_bins)]
    per_bin_ringpix: List[List[int]] = [[] for _ in range(n_bins)]

    n_json = 0
    n_ann = 0
    n_used = 0
    n_matched = 0

    for jf in json_files:
        n_json += 1
        data = json.loads(jf.read_text(encoding="utf-8"))
        anns = iter_annotations(data)
        if not anns:
            continue

        stem = jf.stem

        for i, ann in enumerate(anns):
            n_ann += 1
            seg = ann.get("segmentation", None)
            if seg is None:
                continue

            try:
                mask = decode_mask(seg)
            except Exception:
                continue

            if int(mask.sum()) < min_grain_pixels:
                continue

            grain_key = normalize_key(f"{stem}_{i}")
            nn = nn_map.get(grain_key, np.nan)
            imp = imp_map.get(grain_key, np.nan)

            if not np.isfinite(nn) or not np.isfinite(imp):
                continue

            n_matched += 1

            bins, ring_counts = grain_annulus_membership(mask, bin_edges, min_ring_pixels=min_ring_pixels)
            if not bins:
                continue

            for b, rp in zip(bins, ring_counts):
                per_bin_nn[b].append(float(nn))
                per_bin_imp[b].append(float(imp))
                per_bin_ringpix[b].append(int(rp))

            n_used += 1

    # Aggregate per bin
    def agg_median_iqr(vals: List[float]) -> Tuple[float, float, float]:
        if len(vals) == 0:
            return (np.nan, np.nan, np.nan)
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return (np.nan, np.nan, np.nan)
        q25, q50, q75 = np.percentile(v, [25, 50, 75])
        return float(q50), float(q25), float(q75)

    nn_med = np.full(n_bins, np.nan)
    nn_q25 = np.full(n_bins, np.nan)
    nn_q75 = np.full(n_bins, np.nan)

    imp_med = np.full(n_bins, np.nan)
    imp_q25 = np.full(n_bins, np.nan)
    imp_q75 = np.full(n_bins, np.nan)

    n_grains = np.zeros(n_bins, dtype=int)
    mean_ringpix = np.full(n_bins, np.nan)

    for b in range(n_bins):
        nn_med[b], nn_q25[b], nn_q75[b] = agg_median_iqr(per_bin_nn[b])
        imp_med[b], imp_q25[b], imp_q75[b] = agg_median_iqr(per_bin_imp[b])
        n_grains[b] = int(len(per_bin_nn[b]))

        if len(per_bin_ringpix[b]) > 0:
            mean_ringpix[b] = float(np.mean(per_bin_ringpix[b]))

    info = {
        "sample": sample_label,
        "n_json_files": n_json,
        "n_annotations_seen": n_ann,
        "n_grains_matched_to_crowding": n_matched,
        "n_grains_used_after_ring_filter": n_used,
    }

    return {
        "r_over_R": r_centres,
        "nn_median": nn_med,
        "nn_q25": nn_q25,
        "nn_q75": nn_q75,
        "imp_median": imp_med,
        "imp_q25": imp_q25,
        "imp_q75": imp_q75,
        "n_grains_per_bin": n_grains,
        "mean_ring_pixels_per_grain_bin": mean_ringpix,
        "info": info,
    }


# -------------------------
# Plotting
# -------------------------
def plot_with_iqr(r, y_med, y_q25, y_q75, label, ylabel, title, out_png: Path):
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(r, y_med, lw=2.2, label=label)
    ax.fill_between(r, y_q25, y_q75, alpha=0.18)
    ax.set_xlabel("Normalized radius $r/R$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_two_samples_with_iqr(r, A, B, ylabel, title, out_png: Path, labels=("FAPI", "FAPI-TEMPO")):
    fig, ax = plt.subplots(figsize=(5.6, 4.2))

    ax.plot(r, A["med"], lw=2.2, label=labels[0])
    ax.fill_between(r, A["q25"], A["q75"], alpha=0.18)

    ax.plot(r, B["med"], lw=2.2, label=labels[1])
    ax.fill_between(r, B["q25"], B["q75"], alpha=0.18)

    ax.set_xlabel("Normalized radius $r/R$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_qc(r, nA, nB, pixA, pixB, out_png: Path, labels=("FAPI", "FAPI-TEMPO")):
    fig, axes = plt.subplots(2, 1, figsize=(5.8, 6.4), sharex=True)

    ax = axes[0]
    ax.plot(r, nA, lw=2, label=labels[0])
    ax.plot(r, nB, lw=2, label=labels[1])
    ax.set_ylabel("Contributing grains per bin")
    ax.set_title("QC support for radial crowding proxy")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(r, pixA, lw=2, label=labels[0])
    ax.plot(r, pixB, lw=2, label=labels[1])
    ax.set_xlabel("Normalized radius $r/R$")
    ax.set_ylabel("Mean ring pixels per grain/bin")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Radial (annulus-conditioned) crowding/impingement profiles from JSON + with_crowding tables.")
    ap.add_argument("--fapi-json-dir", required=True)
    ap.add_argument("--tempo-json-dir", required=True)
    ap.add_argument("--fapi-crowding-csv", required=True, help="e.g. crowding_json_based_v3_FAPI_with_crowding.csv")
    ap.add_argument("--tempo-crowding-csv", required=True, help="e.g. crowding_json_based_v3_FAPITEMPO_with_crowding.csv")
    ap.add_argument("--out-prefix", required=True)

    ap.add_argument("--n-bins", type=int, default=25)
    ap.add_argument("--min-grain-pixels", type=int, default=200, help="Minimum grain mask area to consider")
    ap.add_argument("--min-ring-pixels", type=int, default=20, help="Minimum pixels in a ring for a grain to contribute to that ring")

    ap.add_argument("--nn-col", default="nn_dist_px")
    ap.add_argument("--imp-col", default="impingement_index")
    args = ap.parse_args()

    out_prefix = Path(args.out_prefix)
    out_dir = out_prefix.parent if out_prefix.parent != Path("") else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_prefix.name

    fapi_c = pd.read_csv(args.fapi_crowding_csv)
    tempo_c = pd.read_csv(args.tempo_crowding_csv)

    fapi = compute_radial_profiles(
        json_dir=Path(args.fapi_json_dir),
        crowding_df=fapi_c,
        sample_label="FAPI",
        n_bins=args.n_bins,
        min_grain_pixels=args.min_grain_pixels,
        min_ring_pixels=args.min_ring_pixels,
        nn_col=args.nn_col,
        imp_col=args.imp_col,
    )

    tempo = compute_radial_profiles(
        json_dir=Path(args.tempo_json_dir),
        crowding_df=tempo_c,
        sample_label="FAPI-TEMPO",
        n_bins=args.n_bins,
        min_grain_pixels=args.min_grain_pixels,
        min_ring_pixels=args.min_ring_pixels,
        nn_col=args.nn_col,
        imp_col=args.imp_col,
    )

    r = fapi["r_over_R"]
    if not np.allclose(r, tempo["r_over_R"]):
        raise RuntimeError("Radius grids differ (should not happen).")

    # Export combined table
    out_csv = out_dir / f"{base}_radial_crowding_profiles.csv"
    df_out = pd.DataFrame(
        {
            "r_over_R": r,
            "FAPI_nn_median_px": fapi["nn_median"],
            "FAPI_nn_q25_px": fapi["nn_q25"],
            "FAPI_nn_q75_px": fapi["nn_q75"],
            "FAPI_imp_median": fapi["imp_median"],
            "FAPI_imp_q25": fapi["imp_q25"],
            "FAPI_imp_q75": fapi["imp_q75"],
            "FAPI_n_grains": fapi["n_grains_per_bin"],
            "FAPI_mean_ring_pixels": fapi["mean_ring_pixels_per_grain_bin"],
            "FAPITEMPO_nn_median_px": tempo["nn_median"],
            "FAPITEMPO_nn_q25_px": tempo["nn_q25"],
            "FAPITEMPO_nn_q75_px": tempo["nn_q75"],
            "FAPITEMPO_imp_median": tempo["imp_median"],
            "FAPITEMPO_imp_q25": tempo["imp_q25"],
            "FAPITEMPO_imp_q75": tempo["imp_q75"],
            "FAPITEMPO_n_grains": tempo["n_grains_per_bin"],
            "FAPITEMPO_mean_ring_pixels": tempo["mean_ring_pixels_per_grain_bin"],
        }
    )
    df_out.to_csv(out_csv, index=False)

    # Plots
    plot_two_samples_with_iqr(
        r,
        A={"med": fapi["nn_median"], "q25": fapi["nn_q25"], "q75": fapi["nn_q75"]},
        B={"med": tempo["nn_median"], "q25": tempo["nn_q25"], "q75": tempo["nn_q75"]},
        ylabel="Nearest-neighbor distance (px)",
        title="Radial crowding proxy: median NN distance vs $r/R$",
        out_png=out_dir / f"{base}_radial_nn_median.png",
    )

    plot_two_samples_with_iqr(
        r,
        A={"med": fapi["imp_median"], "q25": fapi["imp_q25"], "q75": fapi["imp_q75"]},
        B={"med": tempo["imp_median"], "q25": tempo["imp_q25"], "q75": tempo["imp_q75"]},
        ylabel=r"Impingement index ($R_{eq}$/NN)",
        title="Radial crowding proxy: median impingement index vs $r/R$",
        out_png=out_dir / f"{base}_radial_impingement_median.png",
    )

    plot_qc(
        r,
        nA=fapi["n_grains_per_bin"],
        nB=tempo["n_grains_per_bin"],
        pixA=fapi["mean_ring_pixels_per_grain_bin"],
        pixB=tempo["mean_ring_pixels_per_grain_bin"],
        out_png=out_dir / f"{base}_radial_crowding_qc.png",
    )

    print("[OK] Wrote:")
    print(" ", out_csv)
    print(" ", out_dir / f"{base}_radial_nn_median.png")
    print(" ", out_dir / f"{base}_radial_impingement_median.png")
    print(" ", out_dir / f"{base}_radial_crowding_qc.png")
    print("\n[INFO] Counts:")
    print(" FAPI:", fapi["info"])
    print(" FAPI-TEMPO:", tempo["info"])
    print("\n[NOTE] This is an annulus-conditioned proxy (grain-level NN/impingement assigned to rings the grain occupies).")


if __name__ == "__main__":
    main()