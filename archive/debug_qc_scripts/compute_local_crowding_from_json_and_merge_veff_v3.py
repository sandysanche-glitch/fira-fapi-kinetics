#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compute_local_crowding_from_json_and_merge_veff_v3.py

JSON-based local crowding/impingement descriptor:
- Decode COCO RLE masks from JSON to compute per-grain centroid + area.
- Compute nearest-neighbor distance per micrograph (JSON stem).
- Compute impingement index = R_eq_px / NN_dist_px.
- Merge into "with_veff" tables using the canonical key:
    file_name = <json_stem> + "_" + <annotation_index_in_json_list>

Works with:
- list-style JSON: top-level is a list of annotations
- COCO dict JSON: {"annotations":[...]} etc.

By default:
- If category_id is missing: we assume every entry is a crystal.
- If category_id exists: we keep category_id == 1 as crystals.
You can override with --accept-all (ignore category_id filtering).

Outputs:
- <out-prefix>_FAPI_geometry_from_json.csv
- <out-prefix>_FAPITEMPO_geometry_from_json.csv
- <out-prefix>_FAPI_with_crowding.csv
- <out-prefix>_FAPITEMPO_with_crowding.csv
- <out-prefix>_crowding_summary.csv
- <out-prefix>_veff_vs_nn_distance.png
- <out-prefix>_veff_vs_impingement_index.png
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
        "pycocotools is required to decode RLE masks.\n"
        "On Windows/conda, try:\n"
        "  conda install -c conda-forge pycocotools\n"
        "or:\n"
        "  pip install pycocotools\n"
    ) from e

try:
    from scipy.spatial import cKDTree
except ImportError as e:
    raise ImportError(
        "scipy is required for fast nearest-neighbor computation.\n"
        "Install with:\n"
        "  conda install scipy\n"
        "or:\n"
        "  pip install scipy\n"
    ) from e


# -----------------------------
# JSON / RLE helpers
# -----------------------------
def _as_rle(seg: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize segmentation dict into pycocotools-compatible RLE."""
    if not isinstance(seg, dict) or "size" not in seg or "counts" not in seg:
        raise ValueError("Segmentation must be an RLE dict with keys: size, counts")
    rle = {"size": seg["size"], "counts": seg["counts"]}
    # pycocotools expects bytes for counts if it's a string
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return rle


def decode_mask(seg: Dict[str, Any]) -> np.ndarray:
    """Decode COCO RLE to boolean mask (H,W)."""
    rle = _as_rle(seg)
    m = maskUtils.decode(rle)
    if m.ndim == 3:
        m = m[..., 0]
    return m.astype(bool)


def iter_annotations(obj: Any) -> List[Dict[str, Any]]:
    """
    Return a flat list of annotation dicts from:
      - list-style JSON: [ {...}, {...} ]
      - COCO dict JSON:  {"annotations":[...], ...}
    """
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        anns = obj.get("annotations", None)
        if isinstance(anns, list):
            return [x for x in anns if isinstance(x, dict)]
    return []


def is_crystal_ann(ann: Dict[str, Any], accept_all_if_missing: bool, accept_all_override: bool) -> bool:
    """
    Decide whether an annotation is a crystal:
      - If --accept-all is set: always True.
      - Else if category_id present: crystal = (category_id == 1)
      - Else: crystal = accept_all_if_missing (default True)
    """
    if accept_all_override:
        return True
    if "category_id" in ann:
        try:
            return int(ann["category_id"]) == 1
        except Exception:
            return False
    return accept_all_if_missing


# -----------------------------
# Geometry extraction
# -----------------------------
def extract_geometry_from_json_dir(
    json_dir: Path,
    dataset_label: str,
    min_pixels: int,
    accept_all_if_missing_category: bool,
    accept_all_override: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extract per-grain geometry from all JSON files in a directory.
    Returns (geometry_df, info_dict).
    geometry_df columns:
      - json_file
      - json_stem
      - ann_index          (index in the full annotation list, 0-based)
      - file_name          (= json_stem + "_" + ann_index)  <-- merge key
      - cx_px, cy_px
      - area_px
      - R_eq_px
    """
    json_paths = sorted(json_dir.glob("*.json"))
    rows: List[Dict[str, Any]] = []

    total_json = len(json_paths)
    total_anns = 0
    used_anns = 0
    skipped_small = 0
    skipped_no_seg = 0
    skipped_decode = 0

    for jf in json_paths:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            # fallback encoding
            data = json.loads(jf.read_text())

        anns = iter_annotations(data)
        if not anns:
            continue

        total_anns += len(anns)
        stem = jf.stem

        for i, ann in enumerate(anns):
            if not is_crystal_ann(ann, accept_all_if_missing_category, accept_all_override):
                continue

            seg = ann.get("segmentation", None)
            if seg is None:
                skipped_no_seg += 1
                continue

            try:
                mask = decode_mask(seg)
            except Exception:
                skipped_decode += 1
                continue

            ys, xs = np.nonzero(mask)
            if xs.size < min_pixels:
                skipped_small += 1
                continue

            cx = float(xs.mean())
            cy = float(ys.mean())
            area_px = float(xs.size)  # pixel count
            R_eq_px = float(math.sqrt(area_px / math.pi))

            rows.append(
                dict(
                    json_file=jf.name,
                    json_stem=stem,
                    ann_index=int(i),  # IMPORTANT: index in the original list
                    file_name=f"{stem}_{i}",
                    cx_px=cx,
                    cy_px=cy,
                    area_px=area_px,
                    R_eq_px=R_eq_px,
                )
            )
            used_anns += 1

    df = pd.DataFrame(rows)
    info = dict(
        dataset=dataset_label,
        json_dir=str(json_dir),
        n_json=total_json,
        n_annotations_total=total_anns,
        n_grains_used=used_anns,
        skipped_no_seg=skipped_no_seg,
        skipped_decode=skipped_decode,
        skipped_small=skipped_small,
        accept_all_if_missing_category=accept_all_if_missing_category,
        accept_all_override=accept_all_override,
    )
    return df, info


# -----------------------------
# Crowding metrics
# -----------------------------
def compute_nn_and_impingement(geom: pd.DataFrame) -> pd.DataFrame:
    """
    Compute nearest-neighbor distance (px) within each json_stem (micrograph),
    and impingement index = R_eq_px / NN_dist_px.
    """
    out = geom.copy()
    out["nn_dist_px"] = np.nan
    out["impingement_index"] = np.nan

    for stem, g in out.groupby("json_stem", sort=False):
        if len(g) < 2:
            continue

        pts = np.vstack([g["cx_px"].values, g["cy_px"].values]).T
        tree = cKDTree(pts)
        # query k=2 because first neighbor is self at distance 0
        dists, idxs = tree.query(pts, k=2)
        nn = dists[:, 1]
        out.loc[g.index, "nn_dist_px"] = nn

        # avoid divide-by-zero
        nn_safe = np.where(nn > 0, nn, np.nan)
        out.loc[g.index, "impingement_index"] = g["R_eq_px"].values / nn_safe

    return out


# -----------------------------
# Merge + plotting
# -----------------------------
def normalize_file_name_series(s: pd.Series) -> pd.Series:
    """
    Normalize file_name values for robust merging:
    - cast to str
    - strip extension if present
    - replace backslashes with forward slashes
    - keep only last path component
    """
    s = s.astype(str)
    s = s.str.replace("\\", "/", regex=False)
    s = s.str.split("/").str[-1]
    s = s.str.replace(".png", "", regex=False)
    s = s.str.replace(".jpg", "", regex=False)
    s = s.str.replace(".tif", "", regex=False)
    s = s.str.replace(".tiff", "", regex=False)
    return s


def binned_mean_std(x: np.ndarray, y: np.ndarray, nbins: int = 25):
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size == 0:
        return None, None, None
    xmin, xmax = float(x.min()), float(x.max())
    if xmin == xmax:
        xmin -= 1e-6
        xmax += 1e-6
    edges = np.linspace(xmin, xmax, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    y_mean = np.full(nbins, np.nan)
    y_std = np.full(nbins, np.nan)
    for i in range(nbins):
        sel = (x >= edges[i]) & (x < edges[i + 1])
        if np.any(sel):
            y_mean[i] = float(np.mean(y[sel]))
            y_std[i] = float(np.std(y[sel], ddof=1)) if sel.sum() > 1 else 0.0
    return centers, y_mean, y_std


def plot_scatter_with_bands(
    out_png: Path,
    x_f: np.ndarray,
    y_f: np.ndarray,
    x_t: np.ndarray,
    y_t: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    nbins: int = 25,
):
    plt.figure(figsize=(6.0, 4.5))
    plt.scatter(x_f, y_f, s=6, alpha=0.15, label="FAPI (points)")
    plt.scatter(x_t, y_t, s=6, alpha=0.15, label="FAPI–TEMPO (points)")

    cf, mf, sf = binned_mean_std(x_f, y_f, nbins=nbins)
    if cf is not None:
        plt.plot(cf, mf, lw=2, label="FAPI (mean)")
        plt.fill_between(cf, mf - sf, mf + sf, alpha=0.15)

    ct, mt, st = binned_mean_std(x_t, y_t, nbins=nbins)
    if ct is not None:
        plt.plot(ct, mt, lw=2, label="FAPI–TEMPO (mean)")
        plt.fill_between(ct, mt - st, mt + st, alpha=0.15)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fapi-json-dir", required=True)
    ap.add_argument("--tempo-json-dir", required=True)
    ap.add_argument("--fapi-csv", required=True)
    ap.add_argument("--tempo-csv", required=True)
    ap.add_argument("--out-prefix", required=True)

    ap.add_argument("--min-pixels", type=int, default=200)
    ap.add_argument("--accept-all-if-missing-category", action="store_true",
                    help="If category_id is missing, treat all anns as crystals (recommended for list-style JSONs).")
    ap.add_argument("--accept-all", action="store_true",
                    help="Ignore category_id completely; treat all anns as crystals.")
    ap.add_argument("--nbins", type=int, default=25)

    ap.add_argument("--veff-col", default=None,
                    help="Name of v_eff column in with_veff CSV. If not set, auto-detect.")
    args = ap.parse_args()

    out_prefix = Path(args.out_prefix)
    out_dir = out_prefix.parent if out_prefix.parent != Path("") else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_prefix.name

    fapi_dir = Path(args.fapi_json_dir)
    tempo_dir = Path(args.tempo_json_dir)

    # 1) geometry extraction
    geom_fapi, info_fapi = extract_geometry_from_json_dir(
        json_dir=fapi_dir,
        dataset_label="FAPI",
        min_pixels=args.min_pixels,
        accept_all_if_missing_category=args.accept_all_if_missing_category or True,
        accept_all_override=args.accept_all,
    )
    geom_tempo, info_tempo = extract_geometry_from_json_dir(
        json_dir=tempo_dir,
        dataset_label="FAPI–TEMPO",
        min_pixels=args.min_pixels,
        accept_all_if_missing_category=args.accept_all_if_missing_category or True,
        accept_all_override=args.accept_all,
    )

    if geom_fapi.empty:
        raise RuntimeError(f"No usable crystal masks found in {fapi_dir} (after decoding + filtering).")
    if geom_tempo.empty:
        raise RuntimeError(f"No usable crystal masks found in {tempo_dir} (after decoding + filtering).")

    # 2) compute nn + impingement per micrograph
    geom_fapi = compute_nn_and_impingement(geom_fapi)
    geom_tempo = compute_nn_and_impingement(geom_tempo)

    # save geometry
    geom_fapi_path = out_dir / f"{base}_FAPI_geometry_from_json.csv"
    geom_tempo_path = out_dir / f"{base}_FAPITEMPO_geometry_from_json.csv"
    geom_fapi.to_csv(geom_fapi_path, index=False)
    geom_tempo.to_csv(geom_tempo_path, index=False)

    # 3) merge to with_veff tables
    fapi = pd.read_csv(args.fapi_csv)
    tempo = pd.read_csv(args.tempo_csv)

    if "file_name" not in fapi.columns or "file_name" not in tempo.columns:
        raise ValueError("Both with_veff CSVs must contain a 'file_name' column for merging.")

    fapi["_file_name_norm"] = normalize_file_name_series(fapi["file_name"])
    tempo["_file_name_norm"] = normalize_file_name_series(tempo["file_name"])
    geom_fapi["_file_name_norm"] = normalize_file_name_series(geom_fapi["file_name"])
    geom_tempo["_file_name_norm"] = normalize_file_name_series(geom_tempo["file_name"])

    fapi_m = pd.merge(fapi, geom_fapi.drop(columns=["file_name"]), on="_file_name_norm", how="inner")
    tempo_m = pd.merge(tempo, geom_tempo.drop(columns=["file_name"]), on="_file_name_norm", how="inner")

    # auto-detect veff column if not provided
    def detect_veff_col(df: pd.DataFrame) -> str:
        if args.veff_col and args.veff_col in df.columns:
            return args.veff_col
        cands = [c for c in df.columns if c.lower() in ("v_eff", "v_eff_um_per_ms", "v_eff_um/ms", "veff", "v_eff_um_per_ms")]
        if cands:
            return cands[0]
        # fallback: contains both v and eff
        cands2 = [c for c in df.columns if ("v" in c.lower() and "eff" in c.lower())]
        if cands2:
            return cands2[0]
        raise KeyError("Could not auto-detect a v_eff column. Use --veff-col to specify.")

    veff_col_f = detect_veff_col(fapi_m)
    veff_col_t = detect_veff_col(tempo_m)

    # outputs
    fapi_out = out_dir / f"{base}_FAPI_with_crowding.csv"
    tempo_out = out_dir / f"{base}_FAPITEMPO_with_crowding.csv"
    fapi_m.to_csv(fapi_out, index=False)
    tempo_m.to_csv(tempo_out, index=False)

    # 4) summary
    summary = pd.DataFrame(
        [
            dict(**info_fapi,
                 n_merged=len(fapi_m),
                 veff_col=veff_col_f,
                 nn_non_nan=int(np.isfinite(fapi_m["nn_dist_px"]).sum())),
            dict(**info_tempo,
                 n_merged=len(tempo_m),
                 veff_col=veff_col_t,
                 nn_non_nan=int(np.isfinite(tempo_m["nn_dist_px"]).sum())),
        ]
    )
    summary_path = out_dir / f"{base}_crowding_summary.csv"
    summary.to_csv(summary_path, index=False)

    # 5) plots
    plot_scatter_with_bands(
        out_png=out_dir / f"{base}_veff_vs_nn_distance.png",
        x_f=fapi_m["nn_dist_px"].values,
        y_f=fapi_m[veff_col_f].values,
        x_t=tempo_m["nn_dist_px"].values,
        y_t=tempo_m[veff_col_t].values,
        xlabel="Nearest-neighbor distance (px)",
        ylabel=r"Effective growth rate $v_{\mathrm{eff}}$ (µm/ms)",
        title=r"$v_{\mathrm{eff}}$ vs nearest-neighbor distance",
        nbins=args.nbins,
    )

    plot_scatter_with_bands(
        out_png=out_dir / f"{base}_veff_vs_impingement_index.png",
        x_f=fapi_m["impingement_index"].values,
        y_f=fapi_m[veff_col_f].values,
        x_t=tempo_m["impingement_index"].values,
        y_t=tempo_m[veff_col_t].values,
        xlabel=r"Impingement index ($R_{eq}$/NN)",
        ylabel=r"Effective growth rate $v_{\mathrm{eff}}$ (µm/ms)",
        title=r"$v_{\mathrm{eff}}$ vs local impingement index",
        nbins=args.nbins,
    )

    print("[OK] Saved:")
    print(f"  {geom_fapi_path}")
    print(f"  {geom_tempo_path}")
    print(f"  {fapi_out}")
    print(f"  {tempo_out}")
    print(f"  {summary_path}")
    print(f"  {out_dir / (base + '_veff_vs_nn_distance.png')}")
    print(f"  {out_dir / (base + '_veff_vs_impingement_index.png')}")


if __name__ == "__main__":
    main()