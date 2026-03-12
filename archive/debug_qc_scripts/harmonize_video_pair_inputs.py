# harmonize_video_pair_inputs.py
# Robust harmonization for the video-pair checkpoint tree (FAPI vs FAPI-TEMPO)
#
# What this script does:
#   - reads checkpoint canonical_inputs/*
#   - harmonizes active-grains table (from frame_kinetics.csv)
#   - harmonizes growth-rate table (from growth_rate_vs_time.csv)
#   - harmonizes tau table while PRESERVING tau/QC metadata
#   - writes per-dataset qc_summary.csv
#   - emits a tau propagation check (canonical -> harmonized) before saving
#
# Designed to be checkpoint-tree-only and compatible with compare_video_pair_harmonized.py
# (tau histogram logic unchanged; extra QC columns are preserved but optional).
#
# Usage:
#   python harmonize_video_pair_inputs.py
#   python harmonize_video_pair_inputs.py --checkpoint_dir "F:\...\harmonization_checkpoints\v001_video_pair_fapi_vs_fapi_tempo"
#   python harmonize_video_pair_inputs.py --um_per_px 0.065
#
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# =============================================================================
# Config
# =============================================================================

DEFAULT_UM_PER_PX = 0.065
DATASET_NAMES = ["FAPI", "FAPI_TEMPO"]


# =============================================================================
# Helpers
# =============================================================================

def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def err(msg: str) -> None:
    print(f"[ERROR] {msg}")


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def read_csv_safe(path: Path, label: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        warn(f"Failed reading {label}: {path} ({e})")
        return None


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def touch_text(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def remove_if_exists(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def find_time_ms_col(df: pd.DataFrame) -> Optional[str]:
    return first_existing_col(df, ["time_ms", "t_ms", "time", "t"])


def find_frame_index_col(df: pd.DataFrame) -> Optional[str]:
    return first_existing_col(df, ["frame_i", "frame_idx", "frame_index", "frame", "frame_no"])


def find_track_id_col(df: pd.DataFrame) -> Optional[str]:
    return first_existing_col(df, ["track_id", "track", "id_track"])


def summarize_numeric(series: pd.Series) -> Dict[str, float]:
    s = to_num(series).dropna()
    if s.empty:
        return {
            "count": 0,
            "median": math.nan,
            "mean": math.nan,
            "min": math.nan,
            "max": math.nan,
        }
    return {
        "count": int(s.shape[0]),
        "median": float(s.median()),
        "mean": float(s.mean()),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def as_bool_series(s: pd.Series) -> pd.Series:
    # Robust conversion for bool/int/string fields
    if s.dtype == bool:
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return to_num(s).fillna(0) != 0
    x = s.astype(str).str.strip().str.lower()
    true_set = {"1", "true", "t", "yes", "y", "pass", "ok"}
    return x.isin(true_set)


# =============================================================================
# Path resolution
# =============================================================================

def infer_checkpoint_dir(cli_checkpoint_dir: Optional[str]) -> Path:
    if cli_checkpoint_dir:
        return Path(cli_checkpoint_dir).resolve()

    # Prefer script location if run from checkpoint/scripts/
    script_dir = Path(__file__).resolve().parent
    if script_dir.name == "scripts" and script_dir.parent.name.startswith("v001_"):
        return script_dir.parent

    # Fallback: current working directory / nearby search
    cwd = Path.cwd().resolve()
    if cwd.name == "scripts" and cwd.parent.name.startswith("v001_"):
        return cwd.parent

    # Last resort: assume cwd is checkpoint dir
    return cwd


@dataclass
class DatasetPaths:
    name: str
    canonical_dir: Path
    harmonized_dir: Path
    events_csv: Path
    rate_csv: Path
    nt_csv: Path
    active_src_csv: Path
    tracks_src_csv: Path
    tau_csv: Path
    tau_unavailable_flag: Path
    growth_csv: Path
    frame_kinetics_csv: Path
    metadata_yaml: Path
    # outputs
    active_harmonized_csv: Path
    tau_harmonized_csv: Path
    growth_harmonized_csv: Path
    qc_summary_csv: Path


def dataset_paths(checkpoint_dir: Path, dataset_name: str) -> DatasetPaths:
    canonical = checkpoint_dir / "canonical_inputs" / dataset_name
    harm = checkpoint_dir / "harmonized_tables" / dataset_name
    return DatasetPaths(
        name=dataset_name,
        canonical_dir=canonical,
        harmonized_dir=harm,
        events_csv=canonical / "events.csv",
        rate_csv=canonical / "rate_curve.csv",
        nt_csv=canonical / "Nt_curve.csv",
        active_src_csv=canonical / "active_tracks_source.csv",
        tracks_src_csv=canonical / "tracks_source.csv",
        tau_csv=canonical / "tau_fits.csv",
        tau_unavailable_flag=canonical / "tau_unavailable.flag",
        growth_csv=canonical / "growth_rate_vs_time.csv",
        frame_kinetics_csv=canonical / "frame_kinetics.csv",
        metadata_yaml=canonical / "metadata.yaml",
        active_harmonized_csv=harm / "active_table_harmonized.csv",
        tau_harmonized_csv=harm / "tau_table_harmonized.csv",
        growth_harmonized_csv=harm / "growth_table_harmonized.csv",
        qc_summary_csv=harm / "qc_summary.csv",
    )


# =============================================================================
# Active harmonization (same active criterion from frame_kinetics.csv)
# =============================================================================

def harmonize_active_from_frame_kinetics(df: pd.DataFrame, dataset_name: str, um_per_px: float) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("frame_kinetics.csv empty/unreadable")

    time_col = find_time_ms_col(df)
    if time_col is None:
        raise KeyError(f"{dataset_name} frame_kinetics: missing time column (time_ms/t_ms). Columns={list(df.columns)}")

    frame_col = find_frame_index_col(df)
    ann_col = first_existing_col(df, ["annotation_id", "det_id", "id", "obj_id"])
    area_col = first_existing_col(df, ["area_px", "area", "mask_area_px"])
    r_col = first_existing_col(df, ["R_px", "R_mono", "radius_px", "r_px"])

    x = df.copy()
    x[time_col] = to_num(x[time_col])

    if frame_col is None:
        # Build synthetic frame index from sorted unique times
        uniq_times = sorted([v for v in x[time_col].dropna().unique().tolist()])
        time_to_idx = {t: i for i, t in enumerate(uniq_times)}
        x["_frame_i_tmp"] = x[time_col].map(time_to_idx)
        frame_col = "_frame_i_tmp"
    else:
        x[frame_col] = to_num(x[frame_col])

    # Same active criterion for both datasets:
    # active grains = non-background objects per frame.
    # We exclude the largest-area object in each frame (usually the substrate/background region).
    if area_col is not None:
        x[area_col] = to_num(x[area_col])
    else:
        x["_area_px_tmp"] = math.nan
        area_col = "_area_px_tmp"

    if r_col is not None:
        x[r_col] = to_num(x[r_col])
    elif area_col is not None:
        x["_R_px_tmp"] = (x[area_col] / math.pi).clip(lower=0) ** 0.5
        r_col = "_R_px_tmp"
    else:
        x["_R_px_tmp"] = math.nan
        r_col = "_R_px_tmp"

    group_cols = [frame_col, time_col]

    def per_frame(g: pd.DataFrame) -> pd.Series:
        g2 = g.copy()
        n_total = int(len(g2))

        # mark one background-like object = largest area if available
        bg_excluded = 0
        if area_col in g2.columns and g2[area_col].notna().any():
            idx_bg = g2[area_col].idxmax()
            g2["_is_bg"] = False
            if idx_bg in g2.index:
                g2.loc[idx_bg, "_is_bg"] = True
                bg_excluded = 1
        else:
            g2["_is_bg"] = False

        active = g2.loc[~g2["_is_bg"]].copy()
        n_active = int(len(active))

        sum_area_px = float(to_num(active[area_col]).sum(skipna=True)) if area_col in active.columns else math.nan
        mean_r_px = float(to_num(active[r_col]).mean(skipna=True)) if r_col in active.columns else math.nan

        return pd.Series(
            {
                "n_total_objects": n_total,
                "n_background_excluded": bg_excluded,
                "n_active_grains": n_active,
                "sum_area_px": sum_area_px,
                "sum_area_um2": sum_area_px * (um_per_px ** 2) if pd.notna(sum_area_px) else math.nan,
                "mean_R_px": mean_r_px,
                "mean_R_um": mean_r_px * um_per_px if pd.notna(mean_r_px) else math.nan,
            }
        )

    out = (
        x.groupby(group_cols, dropna=False, sort=True)
        .apply(per_frame)
        .reset_index()
        .rename(columns={frame_col: "frame_i", time_col: "time_ms"})
    )

    out["frame_i"] = to_num(out["frame_i"]).astype("Int64")
    out["time_ms"] = to_num(out["time_ms"])
    out["t_s"] = out["time_ms"] / 1000.0
    out["dataset"] = dataset_name

    # Stable column order
    cols = [
        "dataset", "frame_i", "time_ms", "t_s",
        "n_total_objects", "n_background_excluded", "n_active_grains",
        "sum_area_px", "sum_area_um2", "mean_R_px", "mean_R_um",
    ]
    for c in cols:
        if c not in out.columns:
            out[c] = math.nan
    out = out[cols].sort_values(["time_ms", "frame_i"], na_position="last").reset_index(drop=True)
    return out


# =============================================================================
# Growth harmonization
# =============================================================================

def harmonize_growth_table(df: pd.DataFrame, dataset_name: str, um_per_px: float) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("growth_rate_vs_time.csv empty/unreadable")

    x = df.copy()
    time_col = find_time_ms_col(x)
    if time_col is None:
        raise KeyError(f"{dataset_name} growth_rate_vs_time: missing time column. Columns={list(x.columns)}")

    # Accept many possible growth columns (including clipped/raw variants)
    growth_candidates = [
        "median_um_per_s_clipped",
        "median_um_per_s_raw",
        "median_um_per_s",
        "median_growth_um_per_s",
        "growth_um_per_s",
        "median_rate_um_per_s",
        "median_px_per_s",
        "median_growth_px_per_s",
        "growth_px_per_s",
        "median",
        "growth_rate",
        "rate",
    ]
    g_col = first_existing_col(x, growth_candidates)
    if g_col is None:
        raise KeyError(
            f"{dataset_name} growth_rate_vs_time: no supported growth column found. "
            f"Need one of {growth_candidates}. Found={list(x.columns)}"
        )

    x[time_col] = to_num(x[time_col])
    x[g_col] = to_num(x[g_col])

    # infer units from column name
    col_lower = g_col.lower()
    if "um" in col_lower:
        growth_um = x[g_col]
        growth_px = x[g_col] / um_per_px if um_per_px > 0 else math.nan
        units_src = "um_per_s"
    elif "px" in col_lower:
        growth_px = x[g_col]
        growth_um = x[g_col] * um_per_px
        units_src = "px_per_s"
    else:
        # ambiguous -> assume um/s if values are small; otherwise px/s
        med = float(growth_val_median := x[g_col].dropna().median()) if x[g_col].notna().any() else math.nan
        assume_um = pd.notna(med) and med < 50
        if assume_um:
            growth_um = x[g_col]
            growth_px = x[g_col] / um_per_px if um_per_px > 0 else math.nan
            units_src = "assumed_um_per_s"
        else:
            growth_px = x[g_col]
            growth_um = x[g_col] * um_per_px
            units_src = "assumed_px_per_s"

    out = pd.DataFrame(
        {
            "dataset": dataset_name,
            "time_ms": x[time_col],
            "t_s": x[time_col] / 1000.0,
            "growth_rate_um_per_s": to_num(growth_um),
            "growth_rate_px_per_s": to_num(growth_px),
            "growth_source_col": g_col,
            "growth_source_units_inferred": units_src,
        }
    )
    out = out.dropna(subset=["time_ms"]).sort_values("time_ms").reset_index(drop=True)
    return out


# =============================================================================
# TAU harmonization (DROP-IN BLOCK)
# =============================================================================

def _coalesce_numeric(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    """Coalesce multiple candidate numeric columns left-to-right."""
    vals = pd.Series([math.nan] * len(df), index=df.index, dtype="float64")
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        real = c if c in df.columns else lower_map.get(c.lower())
        if real is None:
            continue
        cur = to_num(df[real])
        vals = vals.where(vals.notna(), cur)
    return vals


def _coalesce_text(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    vals = pd.Series([None] * len(df), index=df.index, dtype="object")
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        real = c if c in df.columns else lower_map.get(c.lower())
        if real is None:
            continue
        cur = df[real].astype(str)
        cur = cur.where(df[real].notna(), None)
        vals = vals.where(vals.notna(), cur)
    return vals


def harmonize_tau_table_with_propagation_check(
    canonical_tau_df: pd.DataFrame,
    dataset_name: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Robust tau harmonization that preserves exact tau/QC metadata and emits a propagation check.

    Required minimum:
      - track_id
      - tau_ms (or convertible aliases)

    Preserved when available:
      - tau_qc_pass
      - fit_status
      - tau_source_status
      - fit_r2
      - fit_nrmse_range
      - fit_window_over_tau
      - n_points
      - fit_t_start_ms / fit_t_end_ms (or aliases)
      - bound-hit flags
    """
    if canonical_tau_df is None or canonical_tau_df.empty:
        raise ValueError("tau_fits.csv empty/unreadable")

    x = canonical_tau_df.copy()

    track_col = find_track_id_col(x)
    if track_col is None:
        raise KeyError(f"{dataset_name} tau_fits.csv: missing track_id. Columns={list(x.columns)}")

    # Build canonicalized columns by coalescing aliases
    tau_ms = _coalesce_numeric(x, ["tau_ms", "tau_fit_ms", "tau", "tau_value_ms"])
    # if 'tau' might be in seconds in some legacy tables, only convert if values look tiny
    if tau_ms.notna().any():
        med_tau = float(tau_ms.dropna().median())
        # heuristic: if med < 1.0, probably seconds -> convert to ms
        if med_tau < 1.0:
            tau_ms = tau_ms * 1000.0

    fit_r2 = _coalesce_numeric(x, ["fit_r2", "r2", "R2"])
    fit_nrmse = _coalesce_numeric(x, ["fit_nrmse_range", "nrmse_range", "fit_nrmse", "nrmse"])
    fit_win_over_tau = _coalesce_numeric(x, ["fit_window_over_tau", "window_over_tau", "fit_span_over_tau"])
    n_points = _coalesce_numeric(x, ["n_points", "fit_n_points", "num_points"])
    t_start_ms = _coalesce_numeric(x, ["fit_t_start_ms", "t_start_ms", "fit_start_ms"])
    t_end_ms = _coalesce_numeric(x, ["fit_t_end_ms", "t_end_ms", "fit_end_ms"])

    tau_qc_pass_col = first_existing_col(x, ["tau_qc_pass", "qc_pass", "fit_qc_pass"])
    fit_status = _coalesce_text(x, ["fit_status", "status", "tau_fit_status"])
    tau_source_status = _coalesce_text(x, ["tau_source_status", "source_status"])

    hit_lo = _coalesce_numeric(x, ["hit_tau_lower_bound", "tau_hit_lower_bound", "hit_lower_bound"]).fillna(0)
    hit_hi = _coalesce_numeric(x, ["hit_tau_upper_bound", "tau_hit_upper_bound", "hit_upper_bound"]).fillna(0)

    # Build harmonized table (one row per track, preserving exact values)
    out = pd.DataFrame(
        {
            "track_id": x[track_col],
            "tau_ms": tau_ms,
            "tau_qc_pass": as_bool_series(x[tau_qc_pass_col]) if tau_qc_pass_col else False,
            "fit_status": fit_status,
            "tau_source_status": tau_source_status,
            "fit_r2": fit_r2,
            "fit_nrmse_range": fit_nrmse,
            "fit_window_over_tau": fit_win_over_tau,
            "n_points": n_points,
            "fit_t_start_ms": t_start_ms,
            "fit_t_end_ms": t_end_ms,
            "hit_tau_lower_bound": as_bool_series(hit_lo),
            "hit_tau_upper_bound": as_bool_series(hit_hi),
        }
    )

    # If tau_source_status missing, synthesize (compare script-compatible)
    if out["tau_source_status"].isna().all():
        out["tau_source_status"] = out["tau_qc_pass"].map({True: "qc_pass", False: "qc_fail"})

    # If fit_status missing, synthesize
    if out["fit_status"].isna().all():
        out["fit_status"] = out["tau_ms"].notna().map({True: "fit_ok", False: "fit_failed"})

    # Normalize types
    out["track_id"] = pd.to_numeric(out["track_id"], errors="coerce").astype("Int64")
    for c in ["tau_ms", "fit_r2", "fit_nrmse_range", "fit_window_over_tau", "n_points", "fit_t_start_ms", "fit_t_end_ms"]:
        out[c] = to_num(out[c])

    for c in ["tau_qc_pass", "hit_tau_lower_bound", "hit_tau_upper_bound"]:
        out[c] = as_bool_series(out[c])

    # Drop rows without track_id; de-duplicate track_id while preserving best row
    out = out.dropna(subset=["track_id"]).copy()
    out["track_id"] = out["track_id"].astype(int)

    # ranking for duplicate rows: prefer qc_pass, then tau non-null, then larger n_points, then better r2
    out["_rank_qc"] = out["tau_qc_pass"].astype(int)
    out["_rank_tau"] = out["tau_ms"].notna().astype(int)
    out["_rank_np"] = to_num(out["n_points"]).fillna(-1)
    out["_rank_r2"] = to_num(out["fit_r2"]).fillna(-1)
    out = out.sort_values(
        ["track_id", "_rank_qc", "_rank_tau", "_rank_np", "_rank_r2"],
        ascending=[True, False, False, False, False],
    )
    out = out.drop_duplicates(subset=["track_id"], keep="first").copy()
    out = out.drop(columns=["_rank_qc", "_rank_tau", "_rank_np", "_rank_r2"], errors="ignore")
    out = out.sort_values("track_id").reset_index(drop=True)

    # -----------------------------
    # Built-in propagation check (canonical -> harmonized)
    # -----------------------------
    can = x.copy()
    can_track = pd.to_numeric(can[track_col], errors="coerce")
    can_tau = tau_ms.copy()
    can_join = pd.DataFrame({"track_id": can_track, "tau_ms_canonical": can_tau}).dropna(subset=["track_id"])
    can_join["track_id"] = can_join["track_id"].astype(int)
    # if canonical has duplicates, keep first non-null tau by same ranking idea
    can_join["_tau_nonnull"] = can_join["tau_ms_canonical"].notna().astype(int)
    can_join = can_join.sort_values(["track_id", "_tau_nonnull"], ascending=[True, False])
    can_join = can_join.drop_duplicates("track_id", keep="first").drop(columns=["_tau_nonnull"])

    both = can_join.merge(out[["track_id", "tau_ms"]].rename(columns={"tau_ms": "tau_ms_harmonized"}),
                          on="track_id", how="inner")
    prop = {
        "canonical_unique_tracks": int(can_join["track_id"].nunique()) if not can_join.empty else 0,
        "harmonized_unique_tracks": int(out["track_id"].nunique()) if not out.empty else 0,
        "in_both_tracks": int(both["track_id"].nunique()) if not both.empty else 0,
        "tau_pairs_compared": 0,
        "tau_abs_diff_median_ms": math.nan,
        "tau_abs_diff_max_ms": math.nan,
    }
    if not both.empty:
        diffs = (to_num(both["tau_ms_canonical"]) - to_num(both["tau_ms_harmonized"])).abs()
        diffs = diffs.dropna()
        prop["tau_pairs_compared"] = int(diffs.shape[0])
        if not diffs.empty:
            prop["tau_abs_diff_median_ms"] = float(diffs.median())
            prop["tau_abs_diff_max_ms"] = float(diffs.max())

    return out, prop


# =============================================================================
# QC summaries
# =============================================================================

def build_qc_summary_rows(
    dataset_name: str,
    active_df: Optional[pd.DataFrame],
    growth_df: Optional[pd.DataFrame],
    tau_df: Optional[pd.DataFrame],
    tau_prop: Optional[Dict[str, float]],
    warnings_list: List[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    def add(metric: str, value, section: str = "general"):
        rows.append({"dataset": dataset_name, "section": section, "metric": metric, "value": value})

    # Active
    if active_df is None:
        add("active_table_status", "missing", "active")
    else:
        add("active_table_status", "ok", "active")
        add("active_rows", int(len(active_df)), "active")
        if "n_active_grains" in active_df.columns:
            s = summarize_numeric(active_df["n_active_grains"])
            add("n_active_grains_count", s["count"], "active")
            add("n_active_grains_median", s["median"], "active")
            add("n_active_grains_max", s["max"], "active")
        if "time_ms" in active_df.columns:
            t = summarize_numeric(active_df["time_ms"])
            add("time_ms_min", t["min"], "active")
            add("time_ms_max", t["max"], "active")

    # Growth
    if growth_df is None:
        add("growth_table_status", "missing", "growth")
    else:
        add("growth_table_status", "ok", "growth")
        add("growth_rows", int(len(growth_df)), "growth")
        if "growth_rate_um_per_s" in growth_df.columns:
            s = summarize_numeric(growth_df["growth_rate_um_per_s"])
            add("growth_um_per_s_count", s["count"], "growth")
            add("growth_um_per_s_median", s["median"], "growth")
            add("growth_um_per_s_max", s["max"], "growth")
        if "growth_source_col" in growth_df.columns and not growth_df.empty:
            mode_src = growth_df["growth_source_col"].mode(dropna=True)
            add("growth_source_col_mode", mode_src.iloc[0] if not mode_src.empty else "", "growth")

    # Tau
    if tau_df is None:
        add("tau_table_status", "missing", "tau")
    else:
        add("tau_table_status", "ok", "tau")
        add("tau_rows", int(len(tau_df)), "tau")
        add("tau_unique_tracks", int(pd.Series(tau_df["track_id"]).nunique(dropna=True)) if "track_id" in tau_df.columns else 0, "tau")
        if "tau_ms" in tau_df.columns:
            s_all = summarize_numeric(tau_df["tau_ms"])
            add("tau_nonnull_count", s_all["count"], "tau")
            add("tau_ms_median_all", s_all["median"], "tau")
            add("tau_ms_min_all", s_all["min"], "tau")
            add("tau_ms_max_all", s_all["max"], "tau")
        if "tau_qc_pass" in tau_df.columns:
            q = as_bool_series(tau_df["tau_qc_pass"])
            add("tau_qc_pass_count", int(q.sum()), "tau")
            add("tau_qc_pass_fraction", float(q.mean()) if len(q) else math.nan, "tau")
            if "tau_ms" in tau_df.columns and q.any():
                s_pass = summarize_numeric(tau_df.loc[q, "tau_ms"])
                add("tau_ms_median_qc_pass", s_pass["median"], "tau")
        if "fit_r2" in tau_df.columns and "tau_qc_pass" in tau_df.columns:
            q = as_bool_series(tau_df["tau_qc_pass"])
            if q.any():
                add("fit_r2_median_qc_pass", summarize_numeric(tau_df.loc[q, "fit_r2"])["median"], "tau")
        if "fit_nrmse_range" in tau_df.columns and "tau_qc_pass" in tau_df.columns:
            q = as_bool_series(tau_df["tau_qc_pass"])
            if q.any():
                add("fit_nrmse_range_median_qc_pass", summarize_numeric(tau_df.loc[q, "fit_nrmse_range"])["median"], "tau")
        if "fit_window_over_tau" in tau_df.columns and "tau_qc_pass" in tau_df.columns:
            q = as_bool_series(tau_df["tau_qc_pass"])
            if q.any():
                add("fit_window_over_tau_median_qc_pass", summarize_numeric(tau_df.loc[q, "fit_window_over_tau"])["median"], "tau")

        if "hit_tau_lower_bound" in tau_df.columns:
            add("tau_hit_lower_bound_count", int(as_bool_series(tau_df["hit_tau_lower_bound"]).sum()), "tau")
        if "hit_tau_upper_bound" in tau_df.columns:
            add("tau_hit_upper_bound_count", int(as_bool_series(tau_df["hit_tau_upper_bound"]).sum()), "tau")

        # propagate check metrics
        if tau_prop:
            for k, v in tau_prop.items():
                add(k, v, "tau_propagation")

    # warnings
    add("warning_count", len(warnings_list), "warnings")
    for i, w in enumerate(warnings_list, start=1):
        add(f"warning_{i:02d}", w, "warnings")

    return pd.DataFrame(rows)


# =============================================================================
# Per-dataset pipeline
# =============================================================================

def process_dataset(paths: DatasetPaths, um_per_px: float) -> None:
    info(f"--- {paths.name} ---")
    paths.harmonized_dir.mkdir(parents=True, exist_ok=True)

    warnings_list: List[str] = []

    # -------------------------
    # Active (frame_kinetics-based, symmetric criterion)
    # -------------------------
    active_df_h: Optional[pd.DataFrame] = None
    fk_df = read_csv_safe(paths.frame_kinetics_csv, f"{paths.name} frame_kinetics.csv")
    if fk_df is None:
        w = f"{paths.name}: Missing frame_kinetics.csv in canonical_inputs. Active-grains harmonization cannot proceed robustly."
        warn(w)
        warnings_list.append(w)
        remove_if_exists(paths.active_harmonized_csv)
    else:
        try:
            active_df_h = harmonize_active_from_frame_kinetics(fk_df, paths.name, um_per_px)
            write_csv(active_df_h, paths.active_harmonized_csv)
            ok(f"{paths.name}: active_table_harmonized.csv ({len(active_df_h)} rows)")
        except Exception as e:
            w = f"{paths.name}: active harmonization failed ({e})"
            warn(w)
            warnings_list.append(w)
            remove_if_exists(paths.active_harmonized_csv)

    # -------------------------
    # Growth
    # -------------------------
    growth_df_h: Optional[pd.DataFrame] = None
    g_df = read_csv_safe(paths.growth_csv, f"{paths.name} growth_rate_vs_time.csv")
    if g_df is None:
        w = f"{paths.name}: Missing growth_rate_vs_time.csv in canonical_inputs. Growth harmonization unavailable."
        warn(w)
        warnings_list.append(w)
        remove_if_exists(paths.growth_harmonized_csv)
    else:
        try:
            growth_df_h = harmonize_growth_table(g_df, paths.name, um_per_px)
            write_csv(growth_df_h, paths.growth_harmonized_csv)
            ok(f"{paths.name}: growth_table_harmonized.csv ({len(growth_df_h)} rows)")
        except Exception as e:
            w = f"{paths.name}: growth harmonization failed ({e})"
            warn(w)
            warnings_list.append(w)
            remove_if_exists(paths.growth_harmonized_csv)

    # -------------------------
    # Tau (preserve exact QC metadata + propagation check)
    # -------------------------
    tau_df_h: Optional[pd.DataFrame] = None
    tau_prop: Optional[Dict[str, float]] = None

    # Clear stale tau_unavailable.flag if tau now exists
    if paths.tau_csv.exists() and paths.tau_unavailable_flag.exists():
        try:
            paths.tau_unavailable_flag.unlink()
        except Exception:
            pass

    tau_src_df = read_csv_safe(paths.tau_csv, f"{paths.name} tau_fits.csv")
    if tau_src_df is None:
        # Respect/emit tau_unavailable.flag (important for compare script)
        if not paths.tau_unavailable_flag.exists():
            touch_text(paths.tau_unavailable_flag, "tau_fits.csv unavailable at harmonization time.\n")
        w = f"{paths.name}: Missing tau source (tau_fits.csv). Tau harmonization unavailable."
        warn(w)
        warnings_list.append(w)
        remove_if_exists(paths.tau_harmonized_csv)
    else:
        try:
            tau_df_h, tau_prop = harmonize_tau_table_with_propagation_check(tau_src_df, paths.name)

            # Built-in propagation check line (requested)
            med_delta = tau_prop.get("tau_abs_diff_median_ms", math.nan) if tau_prop else math.nan
            max_delta = tau_prop.get("tau_abs_diff_max_ms", math.nan) if tau_prop else math.nan
            info(
                f"{paths.name}: tau propagation canonical->harmonized | "
                f"median |Δtau| = {med_delta:.6g} ms, max |Δtau| = {max_delta:.6g} ms"
            )

            write_csv(tau_df_h, paths.tau_harmonized_csv)
            ok(f"{paths.name}: tau_table_harmonized.csv ({len(tau_df_h)} rows)")

            # If we successfully wrote harmonized tau, ensure unavailable flag is removed
            remove_if_exists(paths.tau_unavailable_flag)
        except Exception as e:
            w = f"{paths.name}: tau harmonization failed ({e})"
            warn(w)
            warnings_list.append(w)
            remove_if_exists(paths.tau_harmonized_csv)
            # Keep/emit unavailable flag so compare script behaves predictably
            if not paths.tau_unavailable_flag.exists():
                touch_text(paths.tau_unavailable_flag, f"tau harmonization failed: {e}\n")

    # -------------------------
    # QC summary
    # -------------------------
    qc = build_qc_summary_rows(paths.name, active_df_h, growth_df_h, tau_df_h, tau_prop, warnings_list)
    write_csv(qc, paths.qc_summary_csv)
    ok(f"{paths.name}: qc_summary.csv written")


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Harmonize video-pair inputs (active/growth/tau) in checkpoint tree.")
    ap.add_argument("--checkpoint_dir", type=str, default=None,
                    help="Path to checkpoint root (.../harmonization_checkpoints/v001_video_pair_fapi_vs_fapi_tempo)")
    ap.add_argument("--um_per_px", type=float, default=DEFAULT_UM_PER_PX,
                    help="Pixel calibration in µm/px (default: 0.065)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    checkpoint_dir = infer_checkpoint_dir(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        err(f"Checkpoint dir does not exist: {checkpoint_dir}")
        return 2

    # basic tree checks
    canonical_root = checkpoint_dir / "canonical_inputs"
    harm_root = checkpoint_dir / "harmonized_tables"
    scripts_root = checkpoint_dir / "scripts"

    info("Harmonizing video-pair inputs (active/growth/tau) ...")
    info(f"CHECKPOINT_DIR  : {checkpoint_dir}")
    info(f"UM_PER_PX       : {args.um_per_px:.6f}")

    if not canonical_root.exists():
        err(f"Missing canonical_inputs folder: {canonical_root}")
        return 2

    harm_root.mkdir(parents=True, exist_ok=True)
    scripts_root.mkdir(parents=True, exist_ok=True)

    # Process datasets
    for ds in DATASET_NAMES:
        paths = dataset_paths(checkpoint_dir, ds)
        paths.harmonized_dir.mkdir(parents=True, exist_ok=True)
        process_dataset(paths, um_per_px=args.um_per_px)

    ok("Harmonization tables completed.")
    info("Output folder:")
    info(f"  {harm_root}")
    info("Next: run compare_video_pair_harmonized.py on harmonized_tables/ for a fully symmetric comparison.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())