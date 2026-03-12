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


def robust_num(s):
    return pd.to_numeric(s, errors="coerce")


def load_mask_from_segmentation(seg: dict) -> np.ndarray:
    if not isinstance(seg, dict) or "size" not in seg or "counts" not in seg:
        raise ValueError("Segmentation must be a COCO RLE dict with 'size' and 'counts'.")
    rle = {
        "size": seg["size"],
        "counts": seg["counts"].encode("utf-8") if isinstance(seg["counts"], str) else seg["counts"],
    }
    m = maskUtils.decode(rle)
    if m.ndim == 3:
        m = m[..., 0]
    return m.astype(bool)


def normalize_key(x: str) -> str:
    return str(x).strip().replace("\\", "/").split("/")[-1]


def make_veff_map(df: pd.DataFrame, file_col: str, veff_col: str) -> dict[str, float]:
    if file_col not in df.columns:
        raise KeyError(f"Missing file-name column '{file_col}'")
    if veff_col not in df.columns:
        raise KeyError(f"Missing v_eff column '{veff_col}'")

    out = {}
    for _, row in df[[file_col, veff_col]].dropna().iterrows():
        k = normalize_key(row[file_col])
        v = float(row[veff_col])
        if np.isfinite(v):
            out[k] = v
    return out


def entry_key_from_json_item(stem: str, item: dict, idx: int, index_base: int) -> str:
    for candidate in ["mask_name", "file_name", "name", "id_name"]:
        if candidate in item and item[candidate] is not None:
            return normalize_key(item[candidate])
    return f"{normalize_key(stem)}_{idx + index_base}"


def per_grain_annulus_mask_stats(mask: np.ndarray, bin_edges: np.ndarray, min_pixels: int):
    ys, xs = np.nonzero(mask)
    if ys.size < min_pixels:
        return None, None

    cy = ys.mean()
    cx = xs.mean()
    dy = ys - cy
    dx = xs - cx
    r = np.sqrt(dx * dx + dy * dy)
    R = r.max()
    if R <= 0:
        return None, None

    r_norm = r / R
    valid_bins = []
    ring_counts = []

    n_bins = len(bin_edges) - 1
    for b in range(n_bins):
        r0 = bin_edges[b]
        r1 = bin_edges[b + 1]
        if b < n_bins - 1:
            in_ring = (r_norm >= r0) & (r_norm < r1)
        else:
            in_ring = (r_norm >= r0) & (r_norm <= r1)

        n_pix = int(np.count_nonzero(in_ring))
        if n_pix >= min_pixels:
            valid_bins.append(b)
            ring_counts.append(n_pix)

    return valid_bins, ring_counts


def process_dataset(
    json_dir: Path,
    with_veff_csv: Path,
    n_bins: int,
    min_pixels: int,
    file_col: str,
    veff_col: str,
    index_base: int,
):
    df = pd.read_csv(with_veff_csv)
    veff_map = make_veff_map(df, file_col=file_col, veff_col=veff_col)

    all_veff = robust_num(df[veff_col]).dropna().to_numpy(float)
    scalar_mean = float(np.mean(all_veff)) if all_veff.size else np.nan
    scalar_std = float(np.std(all_veff, ddof=1)) if all_veff.size > 1 else np.nan
    scalar_cv = float(scalar_std / scalar_mean) if np.isfinite(scalar_mean) and scalar_mean != 0 else np.nan

    json_files = sorted(Path(json_dir).glob("*.json"))
    if not json_files:
        raise RuntimeError(f"No JSON files found in {json_dir}")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=float)
    r_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    per_bin_veff = [[] for _ in range(n_bins)]
    per_bin_ringpix = [[] for _ in range(n_bins)]

    n_json = 0
    n_entries = 0
    n_matched = 0
    n_used = 0

    for jf in json_files:
        n_json += 1
        with open(jf, "r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            continue

        stem = jf.stem

        for idx, item in enumerate(data):
            n_entries += 1
            seg = item.get("segmentation", None)
            if seg is None:
                continue

            try:
                mask = load_mask_from_segmentation(seg)
            except Exception:
                continue

            grain_key = entry_key_from_json_item(stem, item, idx, index_base=index_base)
            if grain_key not in veff_map:
                continue

            n_matched += 1
            veff = veff_map[grain_key]
            if not np.isfinite(veff):
                continue

            valid_bins, ring_counts = per_grain_annulus_mask_stats(mask, bin_edges, min_pixels=min_pixels)
            if valid_bins is None:
                continue

            for b, rc in zip(valid_bins, ring_counts):
                per_bin_veff[b].append(veff)
                per_bin_ringpix[b].append(rc)
            n_used += 1

    mean_profile = np.full(n_bins, np.nan)
    std_profile = np.full(n_bins, np.nan)
    cv_profile = np.full(n_bins, np.nan)
    cv_band = np.full(n_bins, np.nan)
    n_profile = np.zeros(n_bins, dtype=int)
    mean_ringpix = np.full(n_bins, np.nan)
    std_ringpix = np.full(n_bins, np.nan)

    for b in range(n_bins):
        vals = np.asarray(per_bin_veff[b], dtype=float)
        vals = vals[np.isfinite(vals)]
        n_profile[b] = vals.size

        if vals.size >= 2:
            mean_profile[b] = float(np.mean(vals))
            std_profile[b] = float(np.std(vals, ddof=1))
            cv_profile[b] = float(std_profile[b] / mean_profile[b]) if mean_profile[b] != 0 else np.nan

            # spread of the underlying grain-level veff values shown as a simple propagated band proxy
            cv_band[b] = float(std_profile[b] / mean_profile[b]) if mean_profile[b] != 0 else np.nan
        elif vals.size == 1:
            mean_profile[b] = float(vals[0])
            std_profile[b] = 0.0
            cv_profile[b] = 0.0
            cv_band[b] = 0.0

        pix = np.asarray(per_bin_ringpix[b], dtype=float)
        pix = pix[np.isfinite(pix)]
        if pix.size:
            mean_ringpix[b] = float(np.mean(pix))
            std_ringpix[b] = float(np.std(pix, ddof=1)) if pix.size > 1 else 0.0

    info = {
        "n_json_files": n_json,
        "n_entries_seen": n_entries,
        "n_entries_matched_to_veff": n_matched,
        "n_entries_used": n_used,
        "scalar_mean_veff": scalar_mean,
        "scalar_std_veff": scalar_std,
        "scalar_cv_veff": scalar_cv,
    }

    return {
        "r_over_R": r_centres,
        "mean_veff": mean_profile,
        "std_veff": std_profile,
        "cv_veff": cv_profile,
        "cv_band": cv_band,
        "n_grains_per_bin": n_profile,
        "mean_ring_pixels": mean_ringpix,
        "std_ring_pixels": std_ringpix,
        "info": info,
    }


def plot_profile_with_band(
    r,
    mean_fapi,
    band_fapi,
    mean_tempo,
    band_tempo,
    ylabel,
    title,
    out_path: Path,
):
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(r, mean_fapi, lw=2.2, label="FAPI")
    ax.fill_between(r, mean_fapi - band_fapi, mean_fapi + band_fapi, alpha=0.18)

    ax.plot(r, mean_tempo, lw=2.2, label="FAPI-TEMPO")
    ax.fill_between(r, mean_tempo - band_tempo, mean_tempo + band_tempo, alpha=0.18)

    ax.set_xlabel("Normalized radius $r/R$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_qc_support(
    r,
    n_fapi,
    n_tempo,
    ringpix_fapi,
    ringpix_tempo,
    out_path: Path,
):
    fig, axes = plt.subplots(2, 1, figsize=(5.4, 6.2), sharex=True)

    ax = axes[0]
    ax.plot(r, n_fapi, lw=2.0, label="FAPI")
    ax.plot(r, n_tempo, lw=2.0, label="FAPI-TEMPO")
    ax.set_ylabel("Contributing grains per bin")
    ax.set_title("QC support for radial kinetic proxy")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(r, ringpix_fapi, lw=2.0, label="FAPI")
    ax.plot(r, ringpix_tempo, lw=2.0, label="FAPI-TEMPO")
    ax.set_xlabel("Normalized radius $r/R$")
    ax.set_ylabel("Mean ring pixels per grain/bin")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Update radial kinetic heterogeneity from grain-level v_eff tables. "
            "This is an annulus-conditioned grain-level proxy, not a true local veff(r/R) field."
        )
    )
    ap.add_argument("--fapi-json-dir", required=True)
    ap.add_argument("--tempo-json-dir", required=True)
    ap.add_argument("--fapi-csv", required=True)
    ap.add_argument("--tempo-csv", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--n-bins", type=int, default=25)
    ap.add_argument("--min-pixels", type=int, default=20)
    ap.add_argument("--file-col", default="file_name")
    ap.add_argument("--veff-col", default="v_eff_um_per_ms")
    ap.add_argument("--index-base", type=int, default=0, choices=[0, 1])
    args = ap.parse_args()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    base = out_prefix.parent / out_prefix.name

    fapi = process_dataset(
        json_dir=Path(args.fapi_json_dir),
        with_veff_csv=Path(args.fapi_csv),
        n_bins=args.n_bins,
        min_pixels=args.min_pixels,
        file_col=args.file_col,
        veff_col=args.veff_col,
        index_base=args.index_base,
    )
    tempo = process_dataset(
        json_dir=Path(args.tempo_json_dir),
        with_veff_csv=Path(args.tempo_csv),
        n_bins=args.n_bins,
        min_pixels=args.min_pixels,
        file_col=args.file_col,
        veff_col=args.veff_col,
        index_base=args.index_base,
    )

    r = fapi["r_over_R"]
    if not np.allclose(r, tempo["r_over_R"]):
        raise RuntimeError("Radius grids do not match between datasets.")

    profile_df = pd.DataFrame(
        {
            "r_over_R": r,
            "mean_veff_FAPI": fapi["mean_veff"],
            "std_veff_FAPI": fapi["std_veff"],
            "cv_veff_FAPI": fapi["cv_veff"],
            "cv_band_FAPI": fapi["cv_band"],
            "n_grains_FAPI": fapi["n_grains_per_bin"],
            "mean_ring_pixels_FAPI": fapi["mean_ring_pixels"],
            "std_ring_pixels_FAPI": fapi["std_ring_pixels"],
            "mean_veff_FAPI_TEMPO": tempo["mean_veff"],
            "std_veff_FAPI_TEMPO": tempo["std_veff"],
            "cv_veff_FAPI_TEMPO": tempo["cv_veff"],
            "cv_band_FAPI_TEMPO": tempo["cv_band"],
            "n_grains_FAPI_TEMPO": tempo["n_grains_per_bin"],
            "mean_ring_pixels_FAPI_TEMPO": tempo["mean_ring_pixels"],
            "std_ring_pixels_FAPI_TEMPO": tempo["std_ring_pixels"],
        }
    )
    out_csv = base.with_name(base.name + "_radial_kinetic_heterogeneity_updated.csv")
    profile_df.to_csv(out_csv, index=False)

    out_png = base.with_name(base.name + "_radial_kinetic_heterogeneity_updated.png")
    plot_profile_with_band(
        r,
        fapi["cv_veff"], fapi["cv_band"],
        tempo["cv_veff"], tempo["cv_band"],
        ylabel=r"Radial kinetic heterogeneity $CV(v_{\mathrm{eff}})$",
        title=r"Radial kinetic heterogeneity from updated $v_{\mathrm{eff}}$",
        out_path=out_png,
    )

    qc_png = base.with_name(base.name + "_radial_kinetic_heterogeneity_qc.png")
    plot_qc_support(
        r,
        fapi["n_grains_per_bin"],
        tempo["n_grains_per_bin"],
        fapi["mean_ring_pixels"],
        tempo["mean_ring_pixels"],
        qc_png,
    )

    scalar_df = pd.DataFrame(
        [
            {
                "sample": "FAPI",
                "mean_veff": fapi["info"]["scalar_mean_veff"],
                "std_veff": fapi["info"]["scalar_std_veff"],
                "CV_veff": fapi["info"]["scalar_cv_veff"],
                "n_json_files": fapi["info"]["n_json_files"],
                "n_entries_seen": fapi["info"]["n_entries_seen"],
                "n_entries_matched_to_veff": fapi["info"]["n_entries_matched_to_veff"],
                "n_entries_used": fapi["info"]["n_entries_used"],
            },
            {
                "sample": "FAPI-TEMPO",
                "mean_veff": tempo["info"]["scalar_mean_veff"],
                "std_veff": tempo["info"]["scalar_std_veff"],
                "CV_veff": tempo["info"]["scalar_cv_veff"],
                "n_json_files": tempo["info"]["n_json_files"],
                "n_entries_seen": tempo["info"]["n_entries_seen"],
                "n_entries_matched_to_veff": tempo["info"]["n_entries_matched_to_veff"],
                "n_entries_used": tempo["info"]["n_entries_used"],
            },
        ]
    )
    out_scalar = base.with_name(base.name + "_tableII_kinetic_heterogeneity_updated.csv")
    scalar_df.to_csv(out_scalar, index=False)

    print(f"[OK] saved profile CSV: {out_csv}")
    print(f"[OK] saved main plot:   {out_png}")
    print(f"[OK] saved QC plot:     {qc_png}")
    print(f"[OK] saved scalar CSV:  {out_scalar}")
    print("\nUpdated Table II / radar kinetic heterogeneity:")
    print(scalar_df.to_string(index=False))
    print(
        "\n[NOTE] This remains an annulus-conditioned grain-level proxy. "
        "The heatmaps support texture/intensity/disorder analysis, but do not provide a true local veff(r/R) field by themselves."
    )


if __name__ == "__main__":
    main()