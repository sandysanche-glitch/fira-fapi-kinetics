#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def robust_num(s):
    return pd.to_numeric(s, errors="coerce")


def normalize_key(x: str) -> str:
    x = str(x).strip().replace("\\", "/").split("/")[-1]
    xl = x.lower()
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".json"]:
        if xl.endswith(ext):
            x = x[: -len(ext)]
            break
    return x


def iter_annotations(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict) and isinstance(obj.get("annotations", None), list):
        return [x for x in obj["annotations"] if isinstance(x, dict)]
    return []


def is_crystal_ann(ann: Dict[str, Any], crystal_category_id: int, accept_all_if_missing: bool) -> bool:
    if "category_id" in ann:
        try:
            return int(ann["category_id"]) == int(crystal_category_id)
        except Exception:
            return False
    return accept_all_if_missing


def decode_mask(seg: Dict[str, Any]) -> np.ndarray:
    if not isinstance(seg, dict) or "size" not in seg or "counts" not in seg:
        raise ValueError("Segmentation must be COCO RLE with size/counts.")
    rle = {"size": seg["size"], "counts": seg["counts"]}
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("utf-8")
    m = maskUtils.decode(rle)
    if m.ndim == 3:
        m = m[..., 0]
    return m.astype(bool)


def grain_annulus_membership(mask: np.ndarray, bin_edges: np.ndarray, min_ring_pixels: int) -> Tuple[List[int], List[int]]:
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


# ------------------------------------------------------------
# core radial proxy computation
# ------------------------------------------------------------
def compute_radial_kinetic_proxy(
    json_dir: Path,
    veff_df: pd.DataFrame,
    sample_label: str,
    veff_col: str,
    n_bins: int,
    min_grain_pixels: int,
    min_ring_pixels: int,
    crystal_category_id: int,
    accept_all_if_missing_category: bool,
) -> Dict[str, Any]:
    if "file_name" not in veff_df.columns:
        raise KeyError("with_veff CSV must contain a 'file_name' column.")
    if veff_col not in veff_df.columns:
        raise KeyError(f"Missing v_eff column '{veff_col}'.")

    veff_df = veff_df.copy()
    veff_df["_grain_key"] = veff_df["file_name"].map(normalize_key)
    veff_map = dict(zip(veff_df["_grain_key"], robust_num(veff_df[veff_col])))

    all_veff = robust_num(veff_df[veff_col]).dropna().to_numpy(float)
    scalar_mean = float(np.mean(all_veff)) if all_veff.size else np.nan
    scalar_std = float(np.std(all_veff, ddof=1)) if all_veff.size > 1 else np.nan
    scalar_cv = float(scalar_std / scalar_mean) if np.isfinite(scalar_mean) and scalar_mean != 0 else np.nan

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"No JSON files found in {json_dir}")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=float)
    r_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    per_bin_veff = [[] for _ in range(n_bins)]
    per_bin_ringpix = [[] for _ in range(n_bins)]

    n_json = 0
    n_ann_total = 0
    n_ann_after_cat = 0
    n_matched = 0
    n_used = 0

    for jf in json_files:
        n_json += 1
        data = json.loads(jf.read_text(encoding="utf-8"))
        anns = iter_annotations(data)
        if not anns:
            continue

        stem = jf.stem

        for i, ann in enumerate(anns):
            n_ann_total += 1

            if not is_crystal_ann(ann, crystal_category_id, accept_all_if_missing_category):
                continue
            n_ann_after_cat += 1

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
            veff = veff_map.get(grain_key, np.nan)

            if not np.isfinite(veff):
                continue
            n_matched += 1

            bins, ring_counts = grain_annulus_membership(mask, bin_edges, min_ring_pixels=min_ring_pixels)
            if not bins:
                continue

            for b, rp in zip(bins, ring_counts):
                per_bin_veff[b].append(float(veff))
                per_bin_ringpix[b].append(int(rp))

            n_used += 1

    mean_veff = np.full(n_bins, np.nan)
    std_veff = np.full(n_bins, np.nan)
    cv_veff = np.full(n_bins, np.nan)
    med_veff = np.full(n_bins, np.nan)
    q25_veff = np.full(n_bins, np.nan)
    q75_veff = np.full(n_bins, np.nan)
    n_grains = np.zeros(n_bins, dtype=int)
    mean_ring_pixels = np.full(n_bins, np.nan)

    for b in range(n_bins):
        vals = np.asarray(per_bin_veff[b], dtype=float)
        vals = vals[np.isfinite(vals)]
        n_grains[b] = int(vals.size)

        if vals.size > 0:
            mean_veff[b] = float(np.mean(vals))
            med_veff[b] = float(np.median(vals))
            q25_veff[b], q75_veff[b] = np.percentile(vals, [25, 75])

        if vals.size > 1:
            std_veff[b] = float(np.std(vals, ddof=1))
            if mean_veff[b] != 0 and np.isfinite(mean_veff[b]):
                cv_veff[b] = float(std_veff[b] / mean_veff[b])
        elif vals.size == 1:
            std_veff[b] = 0.0
            cv_veff[b] = 0.0

        rp = np.asarray(per_bin_ringpix[b], dtype=float)
        rp = rp[np.isfinite(rp)]
        if rp.size > 0:
            mean_ring_pixels[b] = float(np.mean(rp))

    info = {
        "sample": sample_label,
        "n_json_files": n_json,
        "n_annotations_seen": n_ann_total,
        "n_annotations_after_category_filter": n_ann_after_cat,
        "n_grains_matched_to_veff": n_matched,
        "n_grains_used_after_ring_filter": n_used,
        "scalar_mean_veff": scalar_mean,
        "scalar_std_veff": scalar_std,
        "scalar_CV_veff": scalar_cv,
    }

    return {
        "r_over_R": r_centres,
        "mean_veff": mean_veff,
        "std_veff": std_veff,
        "cv_veff": cv_veff,
        "median_veff": med_veff,
        "q25_veff": q25_veff,
        "q75_veff": q75_veff,
        "n_grains_per_bin": n_grains,
        "mean_ring_pixels_per_grain_bin": mean_ring_pixels,
        "info": info,
    }


# ------------------------------------------------------------
# plotting
# ------------------------------------------------------------
def plot_two_samples(r, A_med, A_low, A_high, B_med, B_low, B_high, ylabel, title, out_png: Path, labels=("FAPI", "FAPI-TEMPO")):
    fig, ax = plt.subplots(figsize=(5.6, 4.2))

    ax.plot(r, A_med, lw=2.2, label=labels[0])
    ax.fill_between(r, A_low, A_high, alpha=0.18)

    ax.plot(r, B_med, lw=2.2, label=labels[1])
    ax.fill_between(r, B_low, B_high, alpha=0.18)

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
    ax.set_title("QC support for radial kinetic heterogeneity proxy")
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


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Fixed radial kinetic heterogeneity proxy from JSON masks + grain-level with_veff tables.")
    ap.add_argument("--fapi-json-dir", required=True)
    ap.add_argument("--tempo-json-dir", required=True)
    ap.add_argument("--fapi-csv", required=True)
    ap.add_argument("--tempo-csv", required=True)
    ap.add_argument("--out-prefix", required=True)

    ap.add_argument("--veff-col", default="v_eff_um_per_ms")
    ap.add_argument("--n-bins", type=int, default=25)
    ap.add_argument("--min-grain-pixels", type=int, default=200)
    ap.add_argument("--min-ring-pixels", type=int, default=20)
    ap.add_argument("--crystal-category-id", type=int, default=1)
    ap.add_argument("--accept-all-if-missing-category", action="store_true",
                    help="If category_id is absent, treat all anns as crystals. Recommended for your list-style JSONs.")
    args = ap.parse_args()

    out_prefix = Path(args.out_prefix)
    out_dir = out_prefix.parent if out_prefix.parent != Path("") else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_prefix.name

    fapi_df = pd.read_csv(args.fapi_csv)
    tempo_df = pd.read_csv(args.tempo_csv)

    fapi = compute_radial_kinetic_proxy(
        json_dir=Path(args.fapi_json_dir),
        veff_df=fapi_df,
        sample_label="FAPI",
        veff_col=args.veff_col,
        n_bins=args.n_bins,
        min_grain_pixels=args.min_grain_pixels,
        min_ring_pixels=args.min_ring_pixels,
        crystal_category_id=args.crystal_category_id,
        accept_all_if_missing_category=(args.accept_all_if_missing_category or True),
    )

    tempo = compute_radial_kinetic_proxy(
        json_dir=Path(args.tempo_json_dir),
        veff_df=tempo_df,
        sample_label="FAPI-TEMPO",
        veff_col=args.veff_col,
        n_bins=args.n_bins,
        min_grain_pixels=args.min_grain_pixels,
        min_ring_pixels=args.min_ring_pixels,
        crystal_category_id=args.crystal_category_id,
        accept_all_if_missing_category=(args.accept_all_if_missing_category or True),
    )

    r = fapi["r_over_R"]
    if not np.allclose(r, tempo["r_over_R"]):
        raise RuntimeError("Radius grids do not match.")

    # combined CSV
    out_csv = out_dir / f"{base}_radial_kinetic_heterogeneity.csv"
    df_out = pd.DataFrame(
        {
            "r_over_R": r,
            "FAPI_mean_veff": fapi["mean_veff"],
            "FAPI_std_veff": fapi["std_veff"],
            "FAPI_cv_veff": fapi["cv_veff"],
            "FAPI_median_veff": fapi["median_veff"],
            "FAPI_q25_veff": fapi["q25_veff"],
            "FAPI_q75_veff": fapi["q75_veff"],
            "FAPI_n_grains": fapi["n_grains_per_bin"],
            "FAPI_mean_ring_pixels": fapi["mean_ring_pixels_per_grain_bin"],
            "FAPITEMPO_mean_veff": tempo["mean_veff"],
            "FAPITEMPO_std_veff": tempo["std_veff"],
            "FAPITEMPO_cv_veff": tempo["cv_veff"],
            "FAPITEMPO_median_veff": tempo["median_veff"],
            "FAPITEMPO_q25_veff": tempo["q25_veff"],
            "FAPITEMPO_q75_veff": tempo["q75_veff"],
            "FAPITEMPO_n_grains": tempo["n_grains_per_bin"],
            "FAPITEMPO_mean_ring_pixels": tempo["mean_ring_pixels_per_grain_bin"],
        }
    )
    df_out.to_csv(out_csv, index=False)

    # main CV plot
    out_cv_png = out_dir / f"{base}_radial_kinetic_heterogeneity_cv.png"
    # use simple propagated bands from CV ± 0 here is misleading; use std/mean proxy around CV not ideal.
    # Better: show CV lines only and use IQR on veff in companion plot.
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(r, fapi["cv_veff"], lw=2.2, label="FAPI")
    ax.plot(r, tempo["cv_veff"], lw=2.2, label="FAPI-TEMPO")
    ax.set_xlabel("Normalized radius $r/R$")
    ax.set_ylabel(r"Radial kinetic heterogeneity $CV(v_{\mathrm{eff}})$")
    ax.set_title(r"Radial kinetic heterogeneity from grain-level $v_{\mathrm{eff}}$")
    ax.set_xlim(0.0, 1.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_cv_png, dpi=300)
    plt.close(fig)

    # companion median veff plot
    out_med_png = out_dir / f"{base}_radial_median_veff.png"
    plot_two_samples(
        r,
        fapi["median_veff"], fapi["q25_veff"], fapi["q75_veff"],
        tempo["median_veff"], tempo["q25_veff"], tempo["q75_veff"],
        ylabel=r"Median effective growth rate $v_{\mathrm{eff}}$ (µm/ms)",
        title=r"Annulus-conditioned median $v_{\mathrm{eff}}$ vs $r/R$",
        out_png=out_med_png,
    )

    # QC plot
    out_qc_png = out_dir / f"{base}_radial_kinetic_heterogeneity_qc.png"
    plot_qc(
        r,
        fapi["n_grains_per_bin"],
        tempo["n_grains_per_bin"],
        fapi["mean_ring_pixels_per_grain_bin"],
        tempo["mean_ring_pixels_per_grain_bin"],
        out_qc_png,
    )

    # scalar table
    scalar_df = pd.DataFrame([fapi["info"], tempo["info"]])
    out_scalar = out_dir / f"{base}_tableII_kinetic_heterogeneity.csv"
    scalar_df.to_csv(out_scalar, index=False)

    print("[OK] Wrote:")
    print(" ", out_csv)
    print(" ", out_cv_png)
    print(" ", out_med_png)
    print(" ", out_qc_png)
    print(" ", out_scalar)
    print("\n[INFO]")
    print(scalar_df.to_string(index=False))
    print(
        "\n[NOTE] This is an annulus-conditioned proxy: grain-level scalar v_eff values "
        "are assigned to the normalized-radius bins occupied by each grain. "
        "It is not a true local pixelwise v_eff(r/R) field."
    )


if __name__ == "__main__":
    main()