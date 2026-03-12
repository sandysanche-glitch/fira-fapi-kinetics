#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tau_panel_readiness_check.py

Reads:
  1) tempo_tau_recovery_summary.csv
  2) canonical_inputs/FAPI_TEMPO/tau_fits.csv
  3) harmonized_tables/FAPI_TEMPO/tau_table_harmonized.csv

Prints a one-page "tau panel readiness check":
  - pass counts
  - QC medians
  - bound hits
  - propagation consistency (canonical -> harmonized)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


CHECKPOINT_REL = os.path.join("harmonization_checkpoints", "v001_video_pair_fapi_vs_fapi_tempo")
CANONICAL_TAU_REL = os.path.join("canonical_inputs", "FAPI_TEMPO", "tau_fits.csv")
HARMONIZED_TAU_REL = os.path.join("harmonized_tables", "FAPI_TEMPO", "tau_table_harmonized.csv")

ALIASES = {
    "track_id": ["track_id", "track", "id", "grain_id"],
    "tau_ms": ["tau_ms", "tau", "tau_fit_ms", "tau_value_ms"],
    "tau_qc_pass": ["tau_qc_pass", "qc_pass", "pass_qc", "tau_pass"],
    "fit_status": ["fit_status", "status", "tau_fit_status"],
    "tau_source_status": ["tau_source_status", "source_status", "tau_status"],
    "fit_r2": ["fit_r2", "r2", "R2", "fit_R2"],
    "fit_nrmse_range": ["fit_nrmse_range", "nrmse_range", "fit_nrmse", "NRMSE_range"],
    "fit_window_over_tau": ["fit_window_over_tau", "window_over_tau", "fit_span_over_tau"],
    "n_points": ["n_points", "fit_n_points", "n_fit_points"],
    "t_start_ms": ["t_start_ms", "fit_t_start_ms", "fit_start_ms"],
    "t_end_ms": ["t_end_ms", "fit_t_end_ms", "fit_end_ms"],
    "tau_hit_lower_bound": ["tau_hit_lower_bound", "hit_tau_lower_bound", "tau_at_lower_bound"],
    "tau_hit_upper_bound": ["tau_hit_upper_bound", "hit_tau_upper_bound", "tau_at_upper_bound"],
}

SUMMARY_KEY_ALIASES = {
    "tracks_total": ["tracks_total", "n_tracks_total"],
    "rows_total": ["rows_total", "n_rows_total"],
    "fit_ok_count": ["fit_ok_count", "n_fit_ok"],
    "fit_failed_count": ["fit_failed_count", "n_fit_failed"],
    "tau_nonnull_count": ["tau_nonnull_count", "n_tau_nonnull"],
    "tau_qc_pass_count": ["tau_qc_pass_count", "n_tau_qc_pass", "qc_pass_count"],
    "tau_lower_bound_ms": ["tau_lower_bound_ms", "tau_min_bound_ms"],
    "tau_upper_bound_ms": ["tau_upper_bound_ms", "tau_max_bound_ms"],
    "tau_hit_lower_bound_count": ["tau_hit_lower_bound_count", "hit_lower_bound_count"],
    "tau_hit_upper_bound_count": ["tau_hit_upper_bound_count", "hit_upper_bound_count"],
    "tau_ms_ok_median": ["tau_ms_ok_median"],
    "tau_ms_ok_mean": ["tau_ms_ok_mean"],
    "tau_ms_ok_min": ["tau_ms_ok_min"],
    "tau_ms_ok_max": ["tau_ms_ok_max"],
    "fit_r2_ok_median": ["fit_r2_ok_median"],
    "fit_nrmse_range_ok_median": ["fit_nrmse_range_ok_median"],
    "fit_window_over_tau_ok_median": ["fit_window_over_tau_ok_median"],
}


def first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def resolve_cols(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {k: first_existing_col(df, v) for k, v in ALIASES.items()}


def read_csv_safe(path: str, label: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def parse_boolish(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="boolean")
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "1": True, "true": True, "yes": True, "y": True, "pass": True,
        "0": False, "false": False, "no": False, "n": False, "fail": False,
        "nan": pd.NA, "none": pd.NA, "": pd.NA,
    }
    return s.map(mapping).astype("boolean")


def to_num(series: Optional[pd.Series]) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def fmt(x, nd=3) -> str:
    try:
        if pd.isna(x):
            return "NA"
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def fmti(x) -> str:
    try:
        if pd.isna(x):
            return "NA"
        return str(int(round(float(x))))
    except Exception:
        return str(x)


def median_if_any(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.median()) if len(s) else np.nan


def one_row_summary_to_dict(df: pd.DataFrame) -> Dict[str, object]:
    if df is None or df.empty:
        return {}

    if len(df) == 1 and len(df.columns) > 1:
        return df.iloc[0].to_dict()

    # long format fallback
    key_col = None
    val_col = None
    for c in df.columns:
        cl = str(c).lower()
        if cl in ("metric", "key", "name"):
            key_col = c
        elif cl in ("value", "val"):
            val_col = c
    if key_col and val_col:
        out = {}
        for _, r in df.iterrows():
            out[str(r[key_col])] = r[val_col]
        return out

    return {}


def get_summary_value(summary_dict: Dict[str, object], logical_key: str):
    aliases = SUMMARY_KEY_ALIASES.get(logical_key, [logical_key])
    for k in aliases:
        if k in summary_dict:
            return summary_dict[k]
    lower_map = {str(k).lower(): v for k, v in summary_dict.items()}
    for k in aliases:
        if k.lower() in lower_map:
            return lower_map[k.lower()]
    return np.nan


def infer_bound_hits(df: pd.DataFrame, cols: Dict[str, Optional[str]]) -> Tuple[Optional[int], Optional[int]]:
    lo_col = cols.get("tau_hit_lower_bound")
    hi_col = cols.get("tau_hit_upper_bound")

    if lo_col and hi_col and lo_col in df.columns and hi_col in df.columns:
        lo_n = int(parse_boolish(df[lo_col]).fillna(False).sum())
        hi_n = int(parse_boolish(df[hi_col]).fillna(False).sum())
        return lo_n, hi_n

    status_col = cols.get("fit_status")
    if status_col and status_col in df.columns:
        s = df[status_col].astype(str).str.lower()
        return int(s.str.contains("lower_bound", na=False).sum()), int(s.str.contains("upper_bound", na=False).sum())

    return None, None


def qc_subset(df: pd.DataFrame, cols: Dict[str, Optional[str]]) -> pd.DataFrame:
    qcol = cols.get("tau_qc_pass")
    if not qcol or qcol not in df.columns:
        return df.iloc[0:0].copy()
    mask = parse_boolish(df[qcol]).fillna(False)
    return df.loc[mask].copy()


def fit_ok_subset(df: pd.DataFrame, cols: Dict[str, Optional[str]]) -> pd.DataFrame:
    status_col = cols.get("fit_status")
    tau_col = cols.get("tau_ms")

    if status_col and status_col in df.columns:
        s = df[status_col].astype(str).str.strip().str.lower()
        return df.loc[s.eq("ok")].copy()
    if tau_col and tau_col in df.columns:
        return df.loc[to_num(df[tau_col]).notna()].copy()
    return df.iloc[0:0].copy()


def compute_tau_metrics(df: pd.DataFrame, cols: Dict[str, Optional[str]]) -> Dict[str, object]:
    m: Dict[str, object] = {}
    if df is None or df.empty:
        m["rows"] = 0
        return m

    track_col = cols.get("track_id")
    tau_col = cols.get("tau_ms")
    r2_col = cols.get("fit_r2")
    nrmse_col = cols.get("fit_nrmse_range")
    wot_col = cols.get("fit_window_over_tau")
    status_col = cols.get("fit_status")
    source_col = cols.get("tau_source_status")

    m["rows"] = len(df)
    m["unique_tracks"] = int(df[track_col].nunique()) if track_col and track_col in df.columns else np.nan

    if tau_col and tau_col in df.columns:
        tau = to_num(df[tau_col]).dropna()
        m["tau_nonnull"] = int(len(tau))
        m["tau_median_ms"] = float(tau.median()) if len(tau) else np.nan
        m["tau_mean_ms"] = float(tau.mean()) if len(tau) else np.nan
        m["tau_min_ms"] = float(tau.min()) if len(tau) else np.nan
        m["tau_max_ms"] = float(tau.max()) if len(tau) else np.nan
    else:
        m["tau_nonnull"] = np.nan
        m["tau_median_ms"] = np.nan
        m["tau_mean_ms"] = np.nan
        m["tau_min_ms"] = np.nan
        m["tau_max_ms"] = np.nan

    m["fit_r2_median"] = median_if_any(df[r2_col]) if r2_col and r2_col in df.columns else np.nan
    m["fit_nrmse_range_median"] = median_if_any(df[nrmse_col]) if nrmse_col and nrmse_col in df.columns else np.nan
    m["fit_window_over_tau_median"] = median_if_any(df[wot_col]) if wot_col and wot_col in df.columns else np.nan

    if status_col and status_col in df.columns:
        s = df[status_col].astype(str).str.lower().str.strip()
        m["fit_ok_count"] = int((s == "ok").sum())
        m["fit_failed_count"] = int((s != "ok").sum())
    else:
        m["fit_ok_count"] = np.nan
        m["fit_failed_count"] = np.nan

    if source_col and source_col in df.columns:
        vc = df[source_col].astype(str).fillna("NA").value_counts(dropna=False)
        m["tau_source_status_top"] = vc.index[0] if len(vc) else "NA"
    else:
        m["tau_source_status_top"] = "NA"

    return m


def propagation_check(
    can_df: pd.DataFrame, can_cols: Dict[str, Optional[str]],
    harm_df: pd.DataFrame, harm_cols: Dict[str, Optional[str]]
) -> Dict[str, object]:
    out: Dict[str, object] = {}

    can_track = can_cols.get("track_id")
    harm_track = harm_cols.get("track_id")
    can_tau = can_cols.get("tau_ms")
    harm_tau = harm_cols.get("tau_ms")

    if not (isinstance(can_track, str) and isinstance(harm_track, str)
            and isinstance(can_tau, str) and isinstance(harm_tau, str)):
        out["track_join_possible"] = False
        out["reason"] = "missing track_id and/or tau columns in canonical/harmonized"
        return out

    keep_can = [can_track, can_tau]
    for c in [can_cols.get("tau_qc_pass"), can_cols.get("fit_r2"), can_cols.get("fit_nrmse_range"),
              can_cols.get("fit_window_over_tau"), can_cols.get("tau_source_status")]:
        if isinstance(c, str) and c in can_df.columns and c not in keep_can:
            keep_can.append(c)

    keep_har = [harm_track, harm_tau]
    for c in [harm_cols.get("tau_qc_pass"), harm_cols.get("fit_r2"), harm_cols.get("fit_nrmse_range"),
              harm_cols.get("fit_window_over_tau"), harm_cols.get("tau_source_status")]:
        if isinstance(c, str) and c in harm_df.columns and c not in keep_har:
            keep_har.append(c)

    c = can_df[keep_can].copy().rename(columns={can_track: "track_id", can_tau: "tau_ms"})
    h = harm_df[keep_har].copy().rename(columns={harm_track: "track_id", harm_tau: "tau_ms"})

    c = c.drop_duplicates(subset=["track_id"], keep="first")
    h = h.drop_duplicates(subset=["track_id"], keep="first")

    j = c.merge(h, on="track_id", how="outer", suffixes=("_can", "_har"), indicator=True)

    out["track_join_possible"] = True
    out["canonical_unique_tracks"] = int(c["track_id"].nunique())
    out["harmonized_unique_tracks"] = int(h["track_id"].nunique())
    out["joined_rows"] = int(len(j))
    out["in_both"] = int((j["_merge"] == "both").sum())
    out["only_canonical"] = int((j["_merge"] == "left_only").sum())
    out["only_harmonized"] = int((j["_merge"] == "right_only").sum())

    both = j.loc[j["_merge"] == "both"].copy()
    out["tau_pairs_compared"] = 0
    out["tau_abs_diff_median_ms"] = np.nan
    out["tau_abs_diff_max_ms"] = np.nan
    out["tau_exact_equal_count"] = 0
    out["qc_flag_pairs_compared"] = 0
    out["qc_flag_match_count"] = 0
    out["qc_flag_match_fraction"] = np.nan

    if both.empty:
        return out

    # tau comparison
    tau_can_col = "tau_ms_can" if "tau_ms_can" in both.columns else None
    tau_har_col = "tau_ms_har" if "tau_ms_har" in both.columns else None
    if tau_can_col and tau_har_col:
        both[tau_can_col] = pd.to_numeric(both[tau_can_col], errors="coerce")
        both[tau_har_col] = pd.to_numeric(both[tau_har_col], errors="coerce")
        valid = both.dropna(subset=[tau_can_col, tau_har_col]).copy()
        out["tau_pairs_compared"] = int(len(valid))
        if len(valid):
            dtau = (valid[tau_har_col] - valid[tau_can_col]).abs()
            out["tau_abs_diff_median_ms"] = float(dtau.median())
            out["tau_abs_diff_max_ms"] = float(dtau.max())
            out["tau_exact_equal_count"] = int((dtau == 0).sum())

    # QC-flag consistency (None-safe)
    q_can_col = None
    q_har_col = None

    # Prefer suffix-based matches from merged columns
    for ccol in both.columns:
        cstr = str(ccol).lower()
        if "tau_qc_pass" in cstr and str(ccol).endswith("_can"):
            q_can_col = ccol
        elif "tau_qc_pass" in cstr and str(ccol).endswith("_har"):
            q_har_col = ccol

    # Fallback to exact alias-derived names if they survived merge without suffix
    can_qc_name = can_cols.get("tau_qc_pass")
    har_qc_name = harm_cols.get("tau_qc_pass")

    if q_can_col is None and isinstance(can_qc_name, str) and can_qc_name:
        for ccol in both.columns:
            if str(ccol) == can_qc_name or str(ccol).startswith(can_qc_name):
                q_can_col = ccol
                break

    if q_har_col is None and isinstance(har_qc_name, str) and har_qc_name:
        for ccol in both.columns:
            if str(ccol) == har_qc_name or str(ccol).startswith(har_qc_name):
                q_har_col = ccol
                break

    if q_can_col is not None and q_har_col is not None:
        q1 = parse_boolish(both[q_can_col])
        q2 = parse_boolish(both[q_har_col])
        comp = pd.DataFrame({"q1": q1, "q2": q2}).dropna()
        out["qc_flag_pairs_compared"] = int(len(comp))
        if len(comp):
            matches = (comp["q1"] == comp["q2"])
            out["qc_flag_match_count"] = int(matches.sum())
            out["qc_flag_match_fraction"] = float(matches.mean())

    return out


def find_summary_path(checkpoint_dir: str) -> Optional[str]:
    candidates = [
        os.path.join(checkpoint_dir, "compare_outputs", "csv", "tempo_tau_recovery_summary.csv"),
        os.path.join(checkpoint_dir, "compare_outputs", "tempo_tau_recovery_summary.csv"),
        os.path.join(checkpoint_dir, "tempo_tau_recovery_summary.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def print_section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default=None)
    args = parser.parse_args()

    cwd = os.getcwd()
    if args.checkpoint_dir:
        checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    else:
        if os.path.basename(cwd).lower() == "scripts" and "harmonization_checkpoints" in cwd:
            checkpoint_dir = os.path.dirname(cwd)
        elif os.path.basename(cwd).lower() == "v001_video_pair_fapi_vs_fapi_tempo":
            checkpoint_dir = cwd
        else:
            checkpoint_dir = os.path.abspath(os.path.join(cwd, CHECKPOINT_REL))

    summary_path = find_summary_path(checkpoint_dir)
    canonical_tau_path = os.path.join(checkpoint_dir, CANONICAL_TAU_REL)
    harmonized_tau_path = os.path.join(checkpoint_dir, HARMONIZED_TAU_REL)

    print("=" * 88)
    print("TAU PANEL READINESS CHECK (FAPI-TEMPO)")
    print("=" * 88)
    print(f"[INFO] CHECKPOINT_DIR : {checkpoint_dir}")
    print(f"[INFO] SUMMARY        : {summary_path if summary_path else 'NOT FOUND'}")
    print(f"[INFO] CANONICAL_TAU  : {canonical_tau_path}")
    print(f"[INFO] HARMONIZED_TAU : {harmonized_tau_path}")

    errs = []
    if summary_path is None:
        errs.append("tempo_tau_recovery_summary.csv not found")
    if not os.path.exists(canonical_tau_path):
        errs.append(f"Missing canonical tau_fits.csv: {canonical_tau_path}")
    if not os.path.exists(harmonized_tau_path):
        errs.append(f"Missing harmonized tau_table_harmonized.csv: {harmonized_tau_path}")
    if errs:
        print_section("FATAL INPUT CHECK")
        for e in errs:
            print(f"[ERROR] {e}")
        return 1

    summary_df = read_csv_safe(summary_path, "summary")
    can_df = read_csv_safe(canonical_tau_path, "canonical tau")
    harm_df = read_csv_safe(harmonized_tau_path, "harmonized tau")

    summary_dict = one_row_summary_to_dict(summary_df)
    can_cols = resolve_cols(can_df)
    harm_cols = resolve_cols(harm_df)

    can_all = compute_tau_metrics(can_df, can_cols)
    harm_all = compute_tau_metrics(harm_df, harm_cols)

    can_qc_df = qc_subset(can_df, can_cols)
    harm_qc_df = qc_subset(harm_df, harm_cols)
    can_qc = compute_tau_metrics(can_qc_df, can_cols)
    harm_qc = compute_tau_metrics(harm_qc_df, harm_cols)

    can_lo, can_hi = infer_bound_hits(can_df, can_cols)
    har_lo, har_hi = infer_bound_hits(harm_df, harm_cols)

    prop = propagation_check(can_df, can_cols, harm_df, harm_cols)

    s_tau_qc_pass_count = get_summary_value(summary_dict, "tau_qc_pass_count")
    s_fit_ok_count = get_summary_value(summary_dict, "fit_ok_count")
    s_lo_hit_count = get_summary_value(summary_dict, "tau_hit_lower_bound_count")
    s_hi_hit_count = get_summary_value(summary_dict, "tau_hit_upper_bound_count")

    print_section("1) FILE HEALTH / SCHEMA")
    print(f"summary rows, cols      : {len(summary_df)} x {len(summary_df.columns)}")
    print(f"canonical rows, cols    : {len(can_df)} x {len(can_df.columns)}")
    print(f"harmonized rows, cols   : {len(harm_df)} x {len(harm_df.columns)}")
    print(f"canonical key cols      : {can_cols}")
    print(f"harmonized key cols     : {harm_cols}")

    print_section("2) RECOVERY SUMMARY (tempo_tau_recovery_summary.csv)")
    for k in [
        "tracks_total", "rows_total", "fit_ok_count", "fit_failed_count",
        "tau_nonnull_count", "tau_qc_pass_count",
        "tau_lower_bound_ms", "tau_upper_bound_ms",
        "tau_hit_lower_bound_count", "tau_hit_upper_bound_count",
        "tau_ms_ok_median", "tau_ms_ok_mean", "tau_ms_ok_min", "tau_ms_ok_max",
        "fit_r2_ok_median", "fit_nrmse_range_ok_median", "fit_window_over_tau_ok_median",
    ]:
        print(f"{k:28s}: {get_summary_value(summary_dict, k)}")

    print_section("3) CANONICAL tau_fits.csv")
    print(f"rows total                  : {fmti(can_all.get('rows', np.nan))}")
    print(f"unique tracks               : {fmti(can_all.get('unique_tracks', np.nan))}")
    print(f"tau non-null                : {fmti(can_all.get('tau_nonnull', np.nan))}")
    print(f"tau_qc_pass rows            : {fmti(can_qc.get('rows', np.nan))}"
          + ("" if can_cols.get("tau_qc_pass") else "  [tau_qc_pass missing]"))
    print(f"tau median (all non-null)   : {fmt(can_all.get('tau_median_ms', np.nan))} ms")
    print(f"tau median (QC-pass only)   : {fmt(can_qc.get('tau_median_ms', np.nan))} ms")
    print(f"fit_r2 median (QC-pass)     : {fmt(can_qc.get('fit_r2_median', np.nan), 4)}")
    print(f"nrmse_range med (QC-pass)   : {fmt(can_qc.get('fit_nrmse_range_median', np.nan), 4)}")
    print(f"window/tau med (QC-pass)    : {fmt(can_qc.get('fit_window_over_tau_median', np.nan), 4)}")
    print(f"tau bound hits lower/upper  : {can_lo if can_lo is not None else 'NA'} / {can_hi if can_hi is not None else 'NA'}")

    print_section("4) HARMONIZED tau_table_harmonized.csv")
    print(f"rows total                  : {fmti(harm_all.get('rows', np.nan))}")
    print(f"unique tracks               : {fmti(harm_all.get('unique_tracks', np.nan))}")
    print(f"tau non-null                : {fmti(harm_all.get('tau_nonnull', np.nan))}")
    print(f"tau_qc_pass rows            : {fmti(harm_qc.get('rows', np.nan))}"
          + ("" if harm_cols.get("tau_qc_pass") else "  [tau_qc_pass missing]"))
    print(f"tau median (all non-null)   : {fmt(harm_all.get('tau_median_ms', np.nan))} ms")
    print(f"tau median (QC-pass only)   : {fmt(harm_qc.get('tau_median_ms', np.nan))} ms")
    print(f"fit_r2 median (QC-pass)     : {fmt(harm_qc.get('fit_r2_median', np.nan), 4)}")
    print(f"nrmse_range med (QC-pass)   : {fmt(harm_qc.get('fit_nrmse_range_median', np.nan), 4)}")
    print(f"window/tau med (QC-pass)    : {fmt(harm_qc.get('fit_window_over_tau_median', np.nan), 4)}")
    print(f"tau bound hits lower/upper  : {har_lo if har_lo is not None else 'NA'} / {har_hi if har_hi is not None else 'NA'}")
    print(f"top tau_source_status       : {harm_all.get('tau_source_status_top', 'NA')}")

    print_section("5) PROPAGATION CONSISTENCY (canonical -> harmonized)")
    if not prop.get("track_join_possible", False):
        print(f"[WARN] {prop.get('reason', 'join not possible')}")
    else:
        for k in [
            "canonical_unique_tracks", "harmonized_unique_tracks", "joined_rows",
            "in_both", "only_canonical", "only_harmonized",
            "tau_pairs_compared", "tau_abs_diff_median_ms", "tau_abs_diff_max_ms",
            "tau_exact_equal_count", "qc_flag_pairs_compared", "qc_flag_match_count", "qc_flag_match_fraction"
        ]:
            v = prop.get(k, np.nan)
            if "fraction" in k:
                print(f"{k:28s}: {fmt(v, 4)}")
            elif "diff" in k:
                print(f"{k:28s}: {fmt(v, 6)}")
            else:
                print(f"{k:28s}: {fmti(v) if isinstance(v, (int, float, np.number)) else v}")

    print_section("6) READINESS (practical)")
    harm_qc_count = int(harm_qc.get("rows", 0) or 0)
    warnings = []
    positives = []

    if harm_qc_count == 0:
        status = "NOT READY"
        warnings.append("No QC-pass tau rows in harmonized table.")
    elif harm_qc_count < 10:
        status = "MARGINAL"
        warnings.append("Very low QC-pass count; tau histogram unstable.")
    elif harm_qc_count < 20:
        status = "BORDERLINE"
    elif harm_qc_count < 30:
        status = "USABLE"
    else:
        status = "STRONG"

    if prop.get("track_join_possible", False):
        if int(prop.get("only_canonical", 0)) == 0:
            positives.append("All canonical tau tracks propagated to harmonized table.")
        else:
            warnings.append(f"{prop.get('only_canonical')} canonical tracks missing in harmonized tau table.")

        if int(prop.get("tau_pairs_compared", 0)) > 0:
            maxdiff = prop.get("tau_abs_diff_max_ms", np.nan)
            if not pd.isna(maxdiff) and float(maxdiff) == 0.0:
                positives.append("Tau values preserved exactly during harmonization.")
            else:
                warnings.append(f"Tau values changed during harmonization (max |Δ|={fmt(maxdiff,6)} ms).")

    # summary consistency
    try:
        if not pd.isna(s_tau_qc_pass_count):
            if int(round(float(s_tau_qc_pass_count))) == harm_qc_count:
                positives.append("Summary tau_qc_pass_count matches harmonized QC-pass rows.")
            else:
                warnings.append(
                    f"Summary tau_qc_pass_count ({fmti(s_tau_qc_pass_count)}) != harmonized QC-pass rows ({harm_qc_count})."
                )
    except Exception:
        pass

    # upper-bound hit fraction
    try:
        if not pd.isna(s_hi_hit_count) and not pd.isna(s_fit_ok_count) and float(s_fit_ok_count) > 0:
            hi_frac = float(s_hi_hit_count) / float(s_fit_ok_count)
            print(f"upper-bound hit fraction    : {fmt(hi_frac, 3)} of fit_ok")
            if hi_frac > 0.20:
                warnings.append("High upper-bound hit fraction among fit_ok tracks (possible right-censoring).")
    except Exception:
        pass

    print(f"Readiness status            : {status}")
    print(f"QC-pass rows (harmonized)   : {harm_qc_count}")

    if positives:
        print("\nPositive checks:")
        for p in positives:
            print(f"  - {p}")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\nWarnings: none")

    print("\n[DONE] Tau panel readiness check completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())