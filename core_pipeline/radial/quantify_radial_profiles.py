#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class Windows:
    center_max: float = 0.10
    mid_min: float = 0.45
    mid_max: float = 0.55
    edge_min: float = 0.90


def _require_cols(df: pd.DataFrame, cols: List[str], path: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing} in {path}. Found: {list(df.columns)}")


def _clean_xy(r: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(r) & np.isfinite(y)
    r2 = r[mask]
    y2 = y[mask]
    order = np.argsort(r2)
    return r2[order], y2[order]


def _window_stat(r: np.ndarray, y: np.ndarray, w: Windows, which: str) -> float:
    if which == "center":
        mask = (r >= 0.0) & (r <= w.center_max)
    elif which == "mid":
        mask = (r >= w.mid_min) & (r <= w.mid_max)
    elif which == "edge":
        mask = (r >= w.edge_min) & (r <= 1.0)
    else:
        raise ValueError(which)

    yy = y[mask]
    if yy.size == 0:
        return float("nan")
    return float(np.nanmedian(yy))


def _auc(r: np.ndarray, y: np.ndarray) -> float:
    # assumes r sorted
    if r.size < 2:
        return float("nan")
    return float(np.trapz(y, r))


def _r_of_min(r: np.ndarray, y: np.ndarray) -> float:
    if r.size == 0:
        return float("nan")
    return float(r[int(np.nanargmin(y))])


def quantify_curve(
    r: np.ndarray,
    y: np.ndarray,
    w: Windows,
    compute_rmin: bool = True,
) -> Dict[str, float]:
    r, y = _clean_xy(r, y)

    c = _window_stat(r, y, w, "center")
    m = _window_stat(r, y, w, "mid")
    e = _window_stat(r, y, w, "edge")

    out = {
        "center_median": c,
        "mid_median": m,
        "edge_median": e,
        "center_to_mid": m - c,
        "mid_to_edge": e - m,
        "edge_minus_center": e - c,
        "auc_r_over_R": _auc(r, y),
    }
    if compute_rmin:
        out["r_at_min"] = _r_of_min(r, y)
    return out


def quantify_wide_file(
    csv_path: Path,
    curve_specs: List[Tuple[str, str, bool]],
    # list of (label, suffix_column_name, compute_rmin)
    w: Windows,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    _require_cols(df, ["r_over_R"], str(csv_path))
    r = pd.to_numeric(df["r_over_R"], errors="coerce").to_numpy()

    rows = []
    for sample in ["FAPI", "FAPITEMPO"]:
        for label, suffix, do_rmin in curve_specs:
            col = f"{sample}_{suffix}"
            if col not in df.columns:
                # skip gracefully; many files won't have all curves
                continue
            y = pd.to_numeric(df[col], errors="coerce").to_numpy()
            metrics = quantify_curve(r, y, w=w, compute_rmin=do_rmin)

            row = {
                "source_csv": csv_path.name,
                "sample": "FAPI-TEMPO" if sample == "FAPITEMPO" else "FAPI",
                "curve": label,
                "value_column": col,
            }
            row.update(metrics)
            rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No matching curves were found in {csv_path}. "
            f"Columns: {list(df.columns)}"
        )

    return pd.DataFrame(rows)


def make_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    # Pairwise difference: FAPI - FAPI-TEMPO for same source_csv + curve
    key_cols = ["source_csv", "curve"]
    metric_cols = [
        "center_median", "mid_median", "edge_median",
        "center_to_mid", "mid_to_edge", "edge_minus_center",
        "auc_r_over_R", "r_at_min"
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]

    fapi = df[df["sample"] == "FAPI"].set_index(key_cols)
    tempo = df[df["sample"] == "FAPI-TEMPO"].set_index(key_cols)

    common = fapi.index.intersection(tempo.index)
    if common.empty:
        return pd.DataFrame()

    out = []
    for idx in common:
        row = {"source_csv": idx[0], "curve": idx[1], "comparison": "FAPI - FAPI-TEMPO"}
        for mc in metric_cols:
            a = fapi.loc[idx, mc]
            b = tempo.loc[idx, mc]
            row[mc] = float(a - b) if (np.isfinite(a) and np.isfinite(b)) else float("nan")
        out.append(row)

    return pd.DataFrame(out)


def default_curve_specs() -> List[Tuple[str, str, bool]]:
    """
    Configure which curves to quantify, by *suffix* in the wide CSV.
    (label, suffix, compute_rmin)
    """
    return [
        ("NN distance (µm)", "nn_median_um", True),
        ("NN distance (px)", "nn_median_px", True),
        ("Impingement index (Req/NN)", "imp_median", True),
        ("Median v_eff (µm/ms)", "median_veff", True),
        ("Kinetic heterogeneity CV(v_eff)", "cv_veff", True),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Quantify wide-format radial profiles into caption-ready metrics."
    )
    ap.add_argument("--crowding-csv", default=None, help="radial_crowding_profiles*.csv (wide format)")
    ap.add_argument("--kinetic-csv", default=None, help="radial_kinetic_heterogeneity*.csv (wide format)")
    ap.add_argument("--out-dir", required=True, help="Output directory for SI-ready tables")
    ap.add_argument("--center-max", type=float, default=0.10, help="Center window: r/R <= this")
    ap.add_argument("--mid-min", type=float, default=0.45, help="Mid window min")
    ap.add_argument("--mid-max", type=float, default=0.55, help="Mid window max")
    ap.add_argument("--edge-min", type=float, default=0.90, help="Edge window: r/R >= this")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    w = Windows(
        center_max=args.center_max,
        mid_min=args.mid_min,
        mid_max=args.mid_max,
        edge_min=args.edge_min,
    )

    specs = default_curve_specs()

    all_tables = []

    if args.crowding_csv:
        crowd = quantify_wide_file(Path(args.crowding_csv), specs, w=w)
        all_tables.append(crowd)

    if args.kinetic_csv:
        kin = quantify_wide_file(Path(args.kinetic_csv), specs, w=w)
        all_tables.append(kin)

    if not all_tables:
        raise SystemExit("Provide at least one of --crowding-csv or --kinetic-csv")

    df = pd.concat(all_tables, ignore_index=True)

    # Export per-sample table (SI-ready)
    per_sample_out = out_dir / "SI_radial_curve_metrics_per_sample.csv"
    df.to_csv(per_sample_out, index=False)

    # Export pairwise table
    pair = make_pairwise(df)
    pair_out = out_dir / "SI_radial_curve_metrics_pairwise_FAPI_minus_TEMPO.csv"
    pair.to_csv(pair_out, index=False)

    print(f"[OK] Wrote: {per_sample_out}")
    print(f"[OK] Wrote: {pair_out}")
    print("\n[INFO] Windows used:")
    print(f"  center: r/R in [0, {w.center_max}]")
    print(f"  mid:    r/R in [{w.mid_min}, {w.mid_max}]")
    print(f"  edge:   r/R in [{w.edge_min}, 1]")


if __name__ == "__main__":
    main()