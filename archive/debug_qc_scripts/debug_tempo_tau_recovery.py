#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_tempo_tau_recovery.py

Targeted debug utility for FAPI-TEMPO tau recovery.

It reads:
  1) tempo_tau_recovery_warnings.csv  (fit/QC diagnostics per track)
  2) tracks.csv                       (per-frame track trajectory table)

and automatically generates:
  - representative failed-fit plots by failure mode
  - summary stats by failure mode
  - heuristic recommendations for bound/window adjustments

Outputs are written into a debug folder (default: ./tau_recovery_debug).

This script is intentionally robust to column-name variation. It tries to detect:
  - warnings table columns for track_id, tau, R2, fit status, pass/fail, reasons, fit window, bounds
  - tracks table columns for track_id, time, and radius/area

If a column is missing, it will warn and continue where possible.

USAGE (example)
---------------
python debug_tempo_tau_recovery.py ^
  --warnings_csv "F:\\...\\tempo_tau_recovery_warnings.csv" ^
  --tracks_csv   "F:\\...\\out\\FAPI_TEMPO\\tracks.csv" ^
  --out_dir      "F:\\...\\tau_recovery_debug"

Optional:
  --max_plots_per_mode 12
  --top_modes 8
  --seed 42
  --um_per_px 0.065

Notes
-----
- Plot fitting overlays are "best effort". If fit metadata is available (tau, t0, R0, model columns),
  the script overlays an exponential rise curve. Otherwise it plots raw R(t) only and annotates metadata.
- Recommendations are heuristic and intended to guide your next fitting/QC iteration, not replace judgment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def ok(msg: str) -> None:
    print(f"[OK] {msg}", flush=True)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_read_csv(path: str, label: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed reading {label} ({path}): {e}") from e


def to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lut = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lut:
            return cols_lut[c.lower()]
    # partial contains fallback
    for c in candidates:
        c_low = c.lower()
        for actual in df.columns:
            if c_low in actual.lower():
                return actual
    return None


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^\w\-.]+", "_", str(name))
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:180] if len(name) > 180 else name


def parse_reason_list(x) -> List[str]:
    """Try to parse reason(s) from strings like:
       - 'low_r2|tau_near_lower_bound'
       - '["low_r2","tau_near_lower_bound"]'
       - 'low_r2, tau_near_lower_bound'
    """
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    # JSON-ish list
    if s.startswith("[") and s.endswith("]"):
        try:
            y = json.loads(s)
            if isinstance(y, list):
                return [str(v).strip() for v in y if str(v).strip()]
        except Exception:
            pass
    # split on common delimiters
    parts = re.split(r"[|,;]+", s)
    out = [p.strip() for p in parts if p.strip()]
    return out if out else [s]


# -----------------------------
# Schema detection
# -----------------------------
@dataclass
class WarningsSchema:
    track_id: Optional[str]
    tau_ms: Optional[str]
    tau_s: Optional[str]
    tau_lower_ms: Optional[str]
    tau_upper_ms: Optional[str]
    r2: Optional[str]
    fit_status: Optional[str]
    qc_pass: Optional[str]
    fail_reason: Optional[str]
    fail_reasons: Optional[str]
    n_points_fit: Optional[str]
    n_points_total: Optional[str]
    t_start_ms: Optional[str]
    t_end_ms: Optional[str]
    fit_window_ms: Optional[str]
    t0_ms: Optional[str]
    R0_px: Optional[str]
    Rinf_px: Optional[str]
    model_name: Optional[str]


@dataclass
class TracksSchema:
    track_id: Optional[str]
    time_ms: Optional[str]
    time_s: Optional[str]
    frame_i: Optional[str]
    radius_px: Optional[str]
    area_px: Optional[str]
    x: Optional[str]
    y: Optional[str]


def detect_warnings_schema(df: pd.DataFrame) -> WarningsSchema:
    return WarningsSchema(
        track_id=first_existing(df, ["track_id", "track", "id"]),
        tau_ms=first_existing(df, ["tau_ms", "tau_fit_ms", "tau_recovered_ms"]),
        tau_s=first_existing(df, ["tau_s", "tau_fit_s"]),
        tau_lower_ms=first_existing(df, ["tau_lower_ms", "tau_lb_ms", "tau_min_ms"]),
        tau_upper_ms=first_existing(df, ["tau_upper_ms", "tau_ub_ms", "tau_max_ms"]),
        r2=first_existing(df, ["r2", "fit_r2", "r_squared", "r2_fit"]),
        fit_status=first_existing(df, ["fit_status", "status", "fit_ok"]),
        qc_pass=first_existing(df, ["tau_qc_pass", "qc_pass", "pass_qc", "tau_pass"]),
        fail_reason=first_existing(df, ["tau_qc_fail_reason", "fail_reason", "qc_fail_reason"]),
        fail_reasons=first_existing(df, ["tau_qc_fail_reasons", "fail_reasons", "qc_fail_reasons"]),
        n_points_fit=first_existing(df, ["n_points_fit", "fit_n_points", "n_fit_points", "n_points_used"]),
        n_points_total=first_existing(df, ["n_points_total", "n_points", "track_n_points"]),
        t_start_ms=first_existing(df, ["fit_t_start_ms", "t_fit_start_ms", "t_start_ms"]),
        t_end_ms=first_existing(df, ["fit_t_end_ms", "t_fit_end_ms", "t_end_ms"]),
        fit_window_ms=first_existing(df, ["fit_window_ms", "window_ms", "fit_range_ms"]),
        t0_ms=first_existing(df, ["t0_ms", "fit_t0_ms", "t_nuc_ms"]),
        R0_px=first_existing(df, ["R0_px", "r0_px", "fit_R0_px"]),
        Rinf_px=first_existing(df, ["Rinf_px", "rinf_px", "fit_Rinf_px", "Rmax_px"]),
        model_name=first_existing(df, ["model", "fit_model", "model_name"]),
    )


def detect_tracks_schema(df: pd.DataFrame) -> TracksSchema:
    return TracksSchema(
        track_id=first_existing(df, ["track_id", "track", "id"]),
        time_ms=first_existing(df, ["time_ms", "t_ms"]),
        time_s=first_existing(df, ["time_s", "t_s"]),
        frame_i=first_existing(df, ["frame_i", "frame", "frame_idx"]),
        radius_px=first_existing(df, ["R_px", "radius_px", "r_px"]),
        area_px=first_existing(df, ["area_px", "area"]),
        x=first_existing(df, ["cx", "x", "x_px"]),
        y=first_existing(df, ["cy", "y", "y_px"]),
    )


# -----------------------------
# Normalization
# -----------------------------
def normalize_warnings(df: pd.DataFrame, sch: WarningsSchema) -> pd.DataFrame:
    out = df.copy()

    # Track ID
    if sch.track_id is None:
        raise KeyError(f"Warnings CSV missing track_id-like column. Columns={list(df.columns)}")
    out["track_id_norm"] = out[sch.track_id].astype(str).str.strip()

    # tau_ms
    out["tau_ms_norm"] = np.nan
    if sch.tau_ms is not None:
        out["tau_ms_norm"] = to_numeric(out[sch.tau_ms])
    elif sch.tau_s is not None:
        out["tau_ms_norm"] = 1000.0 * to_numeric(out[sch.tau_s])

    # bounds
    out["tau_lower_ms_norm"] = to_numeric(out[sch.tau_lower_ms]) if sch.tau_lower_ms else np.nan
    out["tau_upper_ms_norm"] = to_numeric(out[sch.tau_upper_ms]) if sch.tau_upper_ms else np.nan

    # r2
    out["r2_norm"] = to_numeric(out[sch.r2]) if sch.r2 else np.nan

    # fit status
    if sch.fit_status:
        out["fit_status_norm"] = out[sch.fit_status].astype(str).str.strip().str.lower()
    else:
        out["fit_status_norm"] = ""

    # qc pass
    if sch.qc_pass:
        # supports 0/1, True/False, yes/no
        vals = out[sch.qc_pass].astype(str).str.strip().str.lower()
        out["qc_pass_norm"] = vals.map({
            "1": 1, "true": 1, "yes": 1, "y": 1, "pass": 1, "ok": 1,
            "0": 0, "false": 0, "no": 0, "n": 0, "fail": 0
        })
        # preserve numeric if unmapped
        m = out["qc_pass_norm"].isna()
        if m.any():
            out.loc[m, "qc_pass_norm"] = pd.to_numeric(vals[m], errors="coerce")
    else:
        out["qc_pass_norm"] = np.nan

    # reasons
    if sch.fail_reasons:
        out["fail_reasons_norm"] = out[sch.fail_reasons].apply(parse_reason_list)
    elif sch.fail_reason:
        out["fail_reasons_norm"] = out[sch.fail_reason].apply(parse_reason_list)
    else:
        out["fail_reasons_norm"] = [[] for _ in range(len(out))]

    # fit metadata
    for src_col, dst_col in [
        (sch.n_points_fit, "n_points_fit_norm"),
        (sch.n_points_total, "n_points_total_norm"),
        (sch.t_start_ms, "fit_t_start_ms_norm"),
        (sch.t_end_ms, "fit_t_end_ms_norm"),
        (sch.fit_window_ms, "fit_window_ms_norm"),
        (sch.t0_ms, "t0_ms_norm"),
        (sch.R0_px, "R0_px_norm"),
        (sch.Rinf_px, "Rinf_px_norm"),
    ]:
        out[dst_col] = to_numeric(out[src_col]) if src_col else np.nan

    # derive fit window if missing
    if out["fit_window_ms_norm"].isna().all():
        if not out["fit_t_start_ms_norm"].isna().all() and not out["fit_t_end_ms_norm"].isna().all():
            out["fit_window_ms_norm"] = out["fit_t_end_ms_norm"] - out["fit_t_start_ms_norm"]

    # model
    out["model_name_norm"] = out[sch.model_name].astype(str) if sch.model_name else ""

    # classify final row class
    out["row_class_norm"] = "unknown"
    fit_ok = out["fit_status_norm"].eq("ok")
    qc_pass = out["qc_pass_norm"] == 1
    qc_fail = out["qc_pass_norm"] == 0
    out.loc[fit_ok & qc_pass, "row_class_norm"] = "fit_ok_qc_pass"
    out.loc[fit_ok & qc_fail, "row_class_norm"] = "fit_ok_qc_fail"
    out.loc[~fit_ok, "row_class_norm"] = "fit_fail"

    # primary failure mode (for plotting/summary)
    def primary_mode(row) -> str:
        reasons = row["fail_reasons_norm"] if isinstance(row["fail_reasons_norm"], list) else []
        if reasons:
            return reasons[0]
        fs = str(row.get("fit_status_norm", "")).strip().lower()
        if fs and fs != "ok":
            return f"fit_status:{fs}"
        qc = row.get("qc_pass_norm", np.nan)
        if pd.notna(qc) and int(qc) == 1:
            return "qc_pass"
        return "unspecified"

    out["primary_failure_mode"] = out.apply(primary_mode, axis=1)
    out["all_failure_modes_joined"] = out["fail_reasons_norm"].apply(lambda xs: "|".join(xs) if isinstance(xs, list) else "")

    # boundary proximity metrics if possible
    out["tau_lb_ratio"] = np.nan
    out["tau_ub_ratio"] = np.nan
    m = (~out["tau_ms_norm"].isna()) & (~out["tau_lower_ms_norm"].isna()) & (out["tau_lower_ms_norm"] > 0)
    out.loc[m, "tau_lb_ratio"] = out.loc[m, "tau_ms_norm"] / out.loc[m, "tau_lower_ms_norm"]
    m = (~out["tau_ms_norm"].isna()) & (~out["tau_upper_ms_norm"].isna()) & (out["tau_upper_ms_norm"] > 0)
    out.loc[m, "tau_ub_ratio"] = out.loc[m, "tau_ms_norm"] / out.loc[m, "tau_upper_ms_norm"]

    out["fit_window_over_tau"] = np.nan
    m = (~out["fit_window_ms_norm"].isna()) & (~out["tau_ms_norm"].isna()) & (out["tau_ms_norm"] > 0)
    out.loc[m, "fit_window_over_tau"] = out.loc[m, "fit_window_ms_norm"] / out.loc[m, "tau_ms_norm"]

    return out


def normalize_tracks(df: pd.DataFrame, sch: TracksSchema) -> pd.DataFrame:
    out = df.copy()

    if sch.track_id is None:
        raise KeyError(f"tracks.csv missing track_id-like column. Columns={list(df.columns)}")
    out["track_id_norm"] = out[sch.track_id].astype(str).str.strip()

    # time
    if sch.time_ms is not None:
        out["time_ms_norm"] = to_numeric(out[sch.time_ms])
    elif sch.time_s is not None:
        out["time_ms_norm"] = 1000.0 * to_numeric(out[sch.time_s])
    else:
        out["time_ms_norm"] = np.nan
        warn("tracks.csv: no time_ms/time_s column found; plot x-axis may be frame index only.")

    # frame
    out["frame_i_norm"] = to_numeric(out[sch.frame_i]) if sch.frame_i else np.nan

    # radius
    if sch.radius_px is not None:
        out["R_px_norm"] = to_numeric(out[sch.radius_px])
    elif sch.area_px is not None:
        area = to_numeric(out[sch.area_px])
        out["R_px_norm"] = np.sqrt(np.clip(area, 0, None) / math.pi)
    else:
        out["R_px_norm"] = np.nan
        warn("tracks.csv: no radius/area column found; tau-fit debug plots cannot show growth radius.")

    # basic cleanup
    if "time_ms_norm" in out.columns:
        out = out.sort_values(["track_id_norm", "time_ms_norm"], kind="mergesort")
    elif "frame_i_norm" in out.columns:
        out = out.sort_values(["track_id_norm", "frame_i_norm"], kind="mergesort")

    return out


# -----------------------------
# Fit overlay (best effort)
# -----------------------------
def exp_rise_model(t_ms: np.ndarray, t0_ms: float, R0: float, Rinf: float, tau_ms: float) -> np.ndarray:
    """
    Simple exponential rise after t0:
        R(t) = R0                                for t <= t0
             = Rinf - (Rinf - R0) * exp(-(t-t0)/tau)   for t > t0
    """
    t = np.asarray(t_ms, dtype=float)
    y = np.full_like(t, fill_value=np.nan, dtype=float)
    valid = np.isfinite(t)
    if not np.isfinite(t0_ms):
        t0_ms = np.nanmin(t[valid]) if valid.any() else 0.0
    if not np.isfinite(R0):
        R0 = np.nan
    if not np.isfinite(Rinf):
        Rinf = np.nan
    if not np.isfinite(tau_ms) or tau_ms <= 0:
        return y

    y[valid] = R0
    m = valid & (t > t0_ms)
    if np.isfinite(R0) and np.isfinite(Rinf):
        y[m] = Rinf - (Rinf - R0) * np.exp(-(t[m] - t0_ms) / tau_ms)
    return y


def choose_plot_x(track_df: pd.DataFrame) -> Tuple[np.ndarray, str]:
    if "time_ms_norm" in track_df.columns and not track_df["time_ms_norm"].isna().all():
        return track_df["time_ms_norm"].to_numpy(dtype=float), "Time (ms)"
    if "frame_i_norm" in track_df.columns and not track_df["frame_i_norm"].isna().all():
        return track_df["frame_i_norm"].to_numpy(dtype=float), "Frame index"
    return np.arange(len(track_df), dtype=float), "Point index"


# -----------------------------
# Summaries and recommendations
# -----------------------------
def summarize_failure_modes(w: pd.DataFrame) -> pd.DataFrame:
    rows = []

    total = len(w)
    fit_ok = int((w["fit_status_norm"] == "ok").sum()) if "fit_status_norm" in w.columns else np.nan
    qc_pass = int((w["qc_pass_norm"] == 1).sum()) if "qc_pass_norm" in w.columns else np.nan
    qc_fail = int((w["qc_pass_norm"] == 0).sum()) if "qc_pass_norm" in w.columns else np.nan

    # primary modes
    vc = w["primary_failure_mode"].value_counts(dropna=False)
    for mode, count in vc.items():
        sub = w[w["primary_failure_mode"] == mode].copy()

        row = {
            "mode": mode,
            "count": int(count),
            "frac_all": float(count / total) if total > 0 else np.nan,
            "n_fit_ok": int((sub["fit_status_norm"] == "ok").sum()) if "fit_status_norm" in sub else np.nan,
            "n_qc_pass": int((sub["qc_pass_norm"] == 1).sum()) if "qc_pass_norm" in sub else np.nan,
            "n_qc_fail": int((sub["qc_pass_norm"] == 0).sum()) if "qc_pass_norm" in sub else np.nan,
        }

        for col in ["tau_ms_norm", "r2_norm", "fit_window_ms_norm", "fit_window_over_tau", "tau_lb_ratio", "tau_ub_ratio"]:
            if col in sub.columns:
                vals = pd.to_numeric(sub[col], errors="coerce").dropna()
                row[f"{col}_median"] = float(vals.median()) if len(vals) else np.nan
                row[f"{col}_p10"] = float(vals.quantile(0.10)) if len(vals) else np.nan
                row[f"{col}_p90"] = float(vals.quantile(0.90)) if len(vals) else np.nan
            else:
                row[f"{col}_median"] = np.nan
                row[f"{col}_p10"] = np.nan
                row[f"{col}_p90"] = np.nan

        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["count", "mode"], ascending=[False, True]).reset_index(drop=True)

    # prepend global row
    global_row = pd.DataFrame([{
        "mode": "__GLOBAL__",
        "count": total,
        "frac_all": 1.0 if total > 0 else np.nan,
        "n_fit_ok": fit_ok,
        "n_qc_pass": qc_pass,
        "n_qc_fail": qc_fail,
        "tau_ms_norm_median": float(pd.to_numeric(w["tau_ms_norm"], errors="coerce").median()) if "tau_ms_norm" in w else np.nan,
        "r2_norm_median": float(pd.to_numeric(w["r2_norm"], errors="coerce").median()) if "r2_norm" in w else np.nan,
        "fit_window_ms_norm_median": float(pd.to_numeric(w["fit_window_ms_norm"], errors="coerce").median()) if "fit_window_ms_norm" in w else np.nan,
        "fit_window_over_tau_median": float(pd.to_numeric(w["fit_window_over_tau"], errors="coerce").median()) if "fit_window_over_tau" in w else np.nan,
        "tau_lb_ratio_median": float(pd.to_numeric(w["tau_lb_ratio"], errors="coerce").median()) if "tau_lb_ratio" in w else np.nan,
        "tau_ub_ratio_median": float(pd.to_numeric(w["tau_ub_ratio"], errors="coerce").median()) if "tau_ub_ratio" in w else np.nan,
    }])
    out = pd.concat([global_row, out], ignore_index=True)
    return out


def generate_recommendations(w: pd.DataFrame, summary_modes: pd.DataFrame) -> pd.DataFrame:
    """
    Heuristic recommendations based on observed failure signatures.
    """
    recs = []

    total = len(w)
    if total == 0:
        return pd.DataFrame(columns=["priority", "topic", "recommendation", "evidence"])

    # Counts for common reasons
    # Expand reasons list
    exploded = []
    for _, row in w.iterrows():
        tid = row.get("track_id_norm", "")
        for r in row.get("fail_reasons_norm", []):
            exploded.append((tid, str(r)))
    reason_df = pd.DataFrame(exploded, columns=["track_id_norm", "reason"]) if exploded else pd.DataFrame(columns=["track_id_norm", "reason"])
    reason_counts = reason_df["reason"].value_counts() if not reason_df.empty else pd.Series(dtype=int)

    def cnt(patterns: List[str]) -> int:
        if reason_counts.empty:
            return 0
        c = 0
        for k, v in reason_counts.items():
            kl = str(k).lower()
            if any(p in kl for p in patterns):
                c += int(v)
        return c

    c_lb = cnt(["lower_bound", "near_lower", "at_lower"])
    c_ub = cnt(["upper_bound", "near_upper", "at_upper"])
    c_r2 = cnt(["low_r2", "r2"])
    c_short = cnt(["short", "window", "fit_window"])
    c_pts = cnt(["n_points", "too_few_points"])
    c_monotonic = cnt(["monotonic", "nonmonotonic", "decrease"])
    c_amp = cnt(["low_amplitude", "small_amplitude", "delta_r"])

    fit_ok = int((w["fit_status_norm"] == "ok").sum()) if "fit_status_norm" in w.columns else 0
    qc_pass = int((w["qc_pass_norm"] == 1).sum()) if "qc_pass_norm" in w.columns else 0
    qc_pass_frac = qc_pass / total if total > 0 else np.nan

    # Global diagnostics
    med_tau = float(pd.to_numeric(w["tau_ms_norm"], errors="coerce").median()) if "tau_ms_norm" in w else np.nan
    med_ratio = float(pd.to_numeric(w["fit_window_over_tau"], errors="coerce").median()) if "fit_window_over_tau" in w else np.nan
    med_r2 = float(pd.to_numeric(w["r2_norm"], errors="coerce").median()) if "r2_norm" in w else np.nan

    # Recommendation rules
    if fit_ok > 0 and qc_pass == 0:
        recs.append({
            "priority": 1,
            "topic": "QC strictness vs fit formulation",
            "recommendation": "All tracks fail QC despite some fit_status=ok. Inspect representative fits by mode before loosening thresholds. Prioritize diagnosing tau-bound hits and fit-window adequacy.",
            "evidence": f"fit_ok={fit_ok}, qc_pass={qc_pass}, total={total}"
        })

    if c_lb > 0:
        recs.append({
            "priority": 1,
            "topic": "Lower-bound tau hits",
            "recommendation": (
                "Many taus are pinned near the lower bound. Check time units (ms vs s), track-level t0 alignment, and whether fit starts too late. "
                "If units/alignment are correct, consider reducing tau lower bound (e.g., by 2–5x) for diagnostic rerun and compare distributions."
            ),
            "evidence": f"lower-bound-related failures ≈ {c_lb}"
        })

    if c_ub > 0:
        recs.append({
            "priority": 2,
            "topic": "Upper-bound tau hits",
            "recommendation": (
                "Some taus reach upper bound. This often indicates insufficient fit window or tracks not approaching saturation. "
                "Extend fit window, require minimum coverage, or treat these as censored / non-identifiable instead of forcing a bounded fit."
            ),
            "evidence": f"upper-bound-related failures ≈ {c_ub}"
        })

    if np.isfinite(med_ratio):
        if med_ratio < 2.0:
            recs.append({
                "priority": 1,
                "topic": "Fit window too short",
                "recommendation": (
                    "Median fit_window/tau is low. For reliable tau, target fit_window >= ~3*tau (preferably 4–5*tau where possible), "
                    "or tighten QC to reject under-covered tracks explicitly."
                ),
                "evidence": f"median fit_window_over_tau={med_ratio:.2f}"
            })
        elif med_ratio < 3.0:
            recs.append({
                "priority": 2,
                "topic": "Fit window adequacy borderline",
                "recommendation": (
                    "Fit-window coverage is borderline. Consider strengthening QC threshold (e.g., require fit_window/tau >= 3) "
                    "or extending per-track fit ranges if data permit."
                ),
                "evidence": f"median fit_window_over_tau={med_ratio:.2f}"
            })

    if np.isfinite(med_r2) and med_r2 < 0.95:
        recs.append({
            "priority": 2,
            "topic": "Low R² prevalence",
            "recommendation": (
                "R² appears low overall. Inspect whether growth model matches actual kinetics (single-exponential rise may be too simple), "
                "and verify radius extraction smoothness/noise. Consider weighting, smoothing, or monotonic filtering for fit input."
            ),
            "evidence": f"median R²={med_r2:.3f}"
        })

    if c_monotonic > 0:
        recs.append({
            "priority": 2,
            "topic": "Non-monotonic radius traces",
            "recommendation": (
                "Radius traces may contain shrinkage/noise segments. Use monotonic-envelope preprocessing or fit only the monotonic growth portion, "
                "and record QC metadata on excluded points."
            ),
            "evidence": f"monotonicity-related failures ≈ {c_monotonic}"
        })

    if c_pts > 0 or c_short > 0:
        recs.append({
            "priority": 2,
            "topic": "Minimum point / window criteria",
            "recommendation": (
                "Increase minimum fit-point requirement and enforce minimum dynamic range (ΔR) before fitting. "
                "Tracks failing these should be classified as 'insufficient data' rather than assigned unstable taus."
            ),
            "evidence": f"point/window-related failures ≈ {c_pts + c_short}"
        })

    if c_amp > 0:
        recs.append({
            "priority": 3,
            "topic": "Amplitude identifiability",
            "recommendation": (
                "Low-amplitude tracks are weakly identifiable for tau. Add ΔR_min or fractional-growth coverage thresholds "
                "(e.g., require R_end - R_start >= threshold and enough points beyond onset)."
            ),
            "evidence": f"low-amplitude-related failures ≈ {c_amp}"
        })

    if not recs:
        recs.append({
            "priority": 3,
            "topic": "General",
            "recommendation": "No dominant failure mode detected from labels alone. Inspect representative failed tracks and compare with passing tracks.",
            "evidence": f"total={total}, qc_pass_frac={qc_pass_frac:.3f}" if np.isfinite(qc_pass_frac) else f"total={total}"
        })

    return pd.DataFrame(recs).sort_values(["priority", "topic"]).reset_index(drop=True)


# -----------------------------
# Plotting
# -----------------------------
def plot_representative_failures(
    w: pd.DataFrame,
    tr: pd.DataFrame,
    out_dir: str,
    max_plots_per_mode: int = 12,
    top_modes: int = 8,
    seed: int = 42,
    um_per_px: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate representative plots for failure modes.
    Returns:
      (plot_index_df, mode_counts_df)
    """
    ensure_dir(out_dir)
    rng = random.Random(seed)

    # Focus on failed rows (fit_fail or fit_ok_qc_fail)
    failed = w[w["row_class_norm"].isin(["fit_fail", "fit_ok_qc_fail"])].copy()
    if failed.empty:
        warn("No failed rows found in warnings table; no representative failure plots generated.")
        return pd.DataFrame(), pd.DataFrame()

    mode_counts = failed["primary_failure_mode"].value_counts().reset_index()
    mode_counts.columns = ["primary_failure_mode", "count"]
    mode_counts = mode_counts.sort_values(["count", "primary_failure_mode"], ascending=[False, True]).reset_index(drop=True)

    selected_modes = mode_counts["primary_failure_mode"].head(top_modes).tolist()
    plot_rows = []

    # Pre-index tracks by track_id for speed
    tr_groups = {k: g.copy() for k, g in tr.groupby("track_id_norm")} if "track_id_norm" in tr.columns else {}

    for mode in selected_modes:
        mode_dir = os.path.join(out_dir, sanitize_filename(mode))
        ensure_dir(mode_dir)

        sub = failed[failed["primary_failure_mode"] == mode].copy()
        # Sample tracks (prefer unique track ids)
        tids = sub["track_id_norm"].dropna().astype(str).unique().tolist()
        rng.shuffle(tids)
        tids = tids[:max_plots_per_mode]

        for i, tid in enumerate(tids, start=1):
            row = sub[sub["track_id_norm"] == tid].iloc[0]
            if tid not in tr_groups:
                warn(f"track_id={tid} not found in tracks.csv (mode={mode}); skipping plot.")
                continue

            tg = tr_groups[tid].copy()
            if tg.empty:
                continue

            x, xlabel = choose_plot_x(tg)
            y = tg["R_px_norm"].to_numpy(dtype=float) if "R_px_norm" in tg.columns else np.full(len(tg), np.nan)
            if np.isnan(y).all():
                warn(f"track_id={tid}: no radius data available; plotting skipped.")
                continue

            plt.figure(figsize=(6.8, 4.6))
            plt.plot(x, y, marker="o", linewidth=1.2, markersize=2.5, label="R(t) raw [px]")

            # Optional µm overlay on secondary y-axis if calibration provided
            if um_per_px is not None and np.isfinite(um_per_px) and um_per_px > 0:
                ax1 = plt.gca()
                ax2 = ax1.twinx()
                ax2.plot(x, y * um_per_px, alpha=0.0)  # scale only (keeps axis)
                ax2.set_ylabel("Radius (µm)")

            # Fit window highlight if available
            t_start = row.get("fit_t_start_ms_norm", np.nan)
            t_end = row.get("fit_t_end_ms_norm", np.nan)
            # Only add vspan if x-axis is time-like
            if "Time" in xlabel and np.isfinite(t_start) and np.isfinite(t_end):
                plt.axvspan(t_start, t_end, alpha=0.15, label="fit window")

            # Best-effort fit overlay
            tau = row.get("tau_ms_norm", np.nan)
            t0 = row.get("t0_ms_norm", np.nan)
            R0 = row.get("R0_px_norm", np.nan)
            Rinf = row.get("Rinf_px_norm", np.nan)

            if "Time" in xlabel and np.isfinite(tau) and tau > 0 and np.isfinite(R0) and np.isfinite(Rinf):
                yfit = exp_rise_model(x, t0_ms=t0, R0=R0, Rinf=Rinf, tau_ms=tau)
                if np.isfinite(yfit).any():
                    plt.plot(x, yfit, linestyle="--", linewidth=1.4, label="fit (exp rise)")

            # Title and annotation
            fs = str(row.get("fit_status_norm", ""))
            qc = row.get("qc_pass_norm", np.nan)
            r2 = row.get("r2_norm", np.nan)
            qclabel = "NA" if pd.isna(qc) else str(int(qc))
            title = f"{mode} | track {tid}"
            plt.title(title)

            ann_lines = [
                f"fit_status={fs}",
                f"qc_pass={qclabel}",
                f"tau_ms={tau:.3g}" if np.isfinite(tau) else "tau_ms=NA",
                f"R²={r2:.4f}" if np.isfinite(r2) else "R²=NA",
            ]
            if np.isfinite(t_start) and np.isfinite(t_end):
                ann_lines.append(f"fit_window=[{t_start:.3g}, {t_end:.3g}] ms")
            fwt = row.get("fit_window_over_tau", np.nan)
            if np.isfinite(fwt):
                ann_lines.append(f"window/tau={fwt:.2f}")
            lb = row.get("tau_lower_ms_norm", np.nan)
            ub = row.get("tau_upper_ms_norm", np.nan)
            if np.isfinite(lb) or np.isfinite(ub):
                ann_lines.append(f"bounds=[{lb if np.isfinite(lb) else 'NA'}, {ub if np.isfinite(ub) else 'NA'}] ms")

            plt.xlabel(xlabel)
            plt.ylabel("Radius (px)")
            plt.legend(loc="best", fontsize=8)
            plt.tight_layout()

            # add text box after tight_layout
            plt.gca().text(
                0.01, 0.99,
                "\n".join(ann_lines),
                transform=plt.gca().transAxes,
                ha="left", va="top",
                fontsize=8,
                bbox=dict(boxstyle="round", alpha=0.2)
            )

            fname = f"{i:02d}_track_{sanitize_filename(tid)}.png"
            fpath = os.path.join(mode_dir, fname)
            plt.savefig(fpath, dpi=180)
            plt.close()

            plot_rows.append({
                "primary_failure_mode": mode,
                "track_id": tid,
                "plot_path": fpath,
                "fit_status_norm": fs,
                "qc_pass_norm": qclabel,
                "tau_ms_norm": tau if np.isfinite(tau) else np.nan,
                "r2_norm": r2 if np.isfinite(r2) else np.nan,
                "fit_window_ms_norm": row.get("fit_window_ms_norm", np.nan),
                "fit_window_over_tau": row.get("fit_window_over_tau", np.nan),
            })

    plot_index_df = pd.DataFrame(plot_rows)
    return plot_index_df, mode_counts


def plot_global_diagnostic_histograms(w: pd.DataFrame, out_png: str) -> None:
    """
    Create a compact diagnostic histogram panel (single PNG) with common metrics if available.
    """
    metrics = []
    for col, title in [
        ("tau_ms_norm", "tau (ms)"),
        ("r2_norm", "R²"),
        ("fit_window_ms_norm", "fit window (ms)"),
        ("fit_window_over_tau", "fit window / tau"),
    ]:
        if col in w.columns and pd.to_numeric(w[col], errors="coerce").notna().any():
            metrics.append((col, title))

    if not metrics:
        warn("No numeric diagnostics available for histogram panel.")
        return

    n = len(metrics)
    plt.figure(figsize=(4.6 * n, 3.8))
    for i, (col, title) in enumerate(metrics, start=1):
        plt.subplot(1, n, i)
        vals = pd.to_numeric(w[col], errors="coerce").dropna()
        if len(vals):
            plt.hist(vals, bins=30)
        plt.title(title)
        plt.xlabel(col)
        plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Debug FAPI-TEMPO tau recovery fits from warnings + tracks tables.")
    p.add_argument("--warnings_csv", required=True, help="Path to tempo_tau_recovery_warnings.csv")
    p.add_argument("--tracks_csv", required=True, help="Path to tracks.csv")
    p.add_argument("--out_dir", default="tau_recovery_debug", help="Output directory for debug artifacts")
    p.add_argument("--max_plots_per_mode", type=int, default=12, help="Max representative plots per failure mode")
    p.add_argument("--top_modes", type=int, default=8, help="Number of top failure modes to plot")
    p.add_argument("--seed", type=int, default=42, help="Random seed for representative sampling")
    p.add_argument("--um_per_px", type=float, default=np.nan, help="Optional calibration for secondary y-axis in plots (µm/px)")
    return p


def main() -> None:
    args = build_parser().parse_args()

    out_dir = os.path.abspath(args.out_dir)
    plots_dir = os.path.join(out_dir, "failed_fit_plots_by_mode")
    csv_dir = os.path.join(out_dir, "csv")
    png_dir = os.path.join(out_dir, "png")
    ensure_dir(out_dir)
    ensure_dir(plots_dir)
    ensure_dir(csv_dir)
    ensure_dir(png_dir)

    info("Loading inputs...")
    info(f"warnings_csv: {args.warnings_csv}")
    info(f"tracks_csv  : {args.tracks_csv}")
    w_raw = safe_read_csv(args.warnings_csv, "tempo_tau_recovery_warnings.csv")
    t_raw = safe_read_csv(args.tracks_csv, "tracks.csv")

    info(f"warnings rows={len(w_raw)} cols={len(w_raw.columns)}")
    info(f"tracks   rows={len(t_raw)} cols={len(t_raw.columns)}")

    # Detect schema
    w_sch = detect_warnings_schema(w_raw)
    t_sch = detect_tracks_schema(t_raw)

    # Save schema detection report
    schema_report = pd.DataFrame([
        {"table": "warnings", "field": k, "detected_column": v}
        for k, v in vars(w_sch).items()
    ] + [
        {"table": "tracks", "field": k, "detected_column": v}
        for k, v in vars(t_sch).items()
    ])
    schema_report.to_csv(os.path.join(csv_dir, "detected_schema_report.csv"), index=False)

    # Normalize
    info("Normalizing tables...")
    try:
        w = normalize_warnings(w_raw, w_sch)
    except Exception as e:
        raise RuntimeError(f"Failed normalizing warnings CSV: {e}") from e

    try:
        t = normalize_tracks(t_raw, t_sch)
    except Exception as e:
        raise RuntimeError(f"Failed normalizing tracks CSV: {e}") from e

    # Basic global summary
    global_summary = {
        "warnings_rows_total": len(w),
        "warnings_unique_tracks": int(w["track_id_norm"].nunique()) if "track_id_norm" in w else np.nan,
        "tracks_rows_total": len(t),
        "tracks_unique_tracks": int(t["track_id_norm"].nunique()) if "track_id_norm" in t else np.nan,
        "fit_ok_count": int((w["fit_status_norm"] == "ok").sum()) if "fit_status_norm" in w else np.nan,
        "qc_pass_count": int((w["qc_pass_norm"] == 1).sum()) if "qc_pass_norm" in w else np.nan,
        "qc_fail_count": int((w["qc_pass_norm"] == 0).sum()) if "qc_pass_norm" in w else np.nan,
        "tau_nonnull_count": int(pd.to_numeric(w["tau_ms_norm"], errors="coerce").notna().sum()) if "tau_ms_norm" in w else np.nan,
        "r2_nonnull_count": int(pd.to_numeric(w["r2_norm"], errors="coerce").notna().sum()) if "r2_norm" in w else np.nan,
    }
    pd.DataFrame([global_summary]).to_csv(os.path.join(csv_dir, "global_debug_summary.csv"), index=False)

    # Save normalized extracts for inspection
    keep_cols_w = [
        "track_id_norm", "row_class_norm", "fit_status_norm", "qc_pass_norm",
        "primary_failure_mode", "all_failure_modes_joined",
        "tau_ms_norm", "tau_lower_ms_norm", "tau_upper_ms_norm",
        "tau_lb_ratio", "tau_ub_ratio",
        "r2_norm", "n_points_fit_norm", "n_points_total_norm",
        "fit_t_start_ms_norm", "fit_t_end_ms_norm", "fit_window_ms_norm", "fit_window_over_tau",
        "t0_ms_norm", "R0_px_norm", "Rinf_px_norm", "model_name_norm"
    ]
    keep_cols_w = [c for c in keep_cols_w if c in w.columns]
    w[keep_cols_w].to_csv(os.path.join(csv_dir, "warnings_normalized_extract.csv"), index=False)

    keep_cols_t = [c for c in ["track_id_norm", "time_ms_norm", "frame_i_norm", "R_px_norm"] if c in t.columns]
    t[keep_cols_t].to_csv(os.path.join(csv_dir, "tracks_normalized_extract.csv"), index=False)

    # Failure mode summary
    info("Computing failure-mode summaries...")
    mode_summary = summarize_failure_modes(w)
    mode_summary.to_csv(os.path.join(csv_dir, "failure_mode_summary.csv"), index=False)

    # Recommendations
    info("Generating heuristic recommendations...")
    recs = generate_recommendations(w, mode_summary)
    recs.to_csv(os.path.join(csv_dir, "recommended_adjustments.csv"), index=False)

    # Representative failed-fit plots
    info("Generating representative failed-fit plots by failure mode...")
    plot_index_df, mode_counts_df = plot_representative_failures(
        w=w,
        tr=t,
        out_dir=plots_dir,
        max_plots_per_mode=args.max_plots_per_mode,
        top_modes=args.top_modes,
        seed=args.seed,
        um_per_px=(args.um_per_px if np.isfinite(args.um_per_px) else None),
    )
    if not plot_index_df.empty:
        plot_index_df.to_csv(os.path.join(csv_dir, "representative_failed_fit_plot_index.csv"), index=False)
    if not mode_counts_df.empty:
        mode_counts_df.to_csv(os.path.join(csv_dir, "failure_mode_counts_failed_only.csv"), index=False)

    # Diagnostic histograms
    info("Generating global diagnostic histograms...")
    plot_global_diagnostic_histograms(w, os.path.join(png_dir, "tau_recovery_diagnostic_histograms.png"))

    # Write a human-readable text summary
    txt_path = os.path.join(out_dir, "README_debug_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Tau Recovery Debug Summary\n")
        f.write("=========================\n\n")
        f.write(f"warnings_csv: {os.path.abspath(args.warnings_csv)}\n")
        f.write(f"tracks_csv  : {os.path.abspath(args.tracks_csv)}\n\n")
        for k, v in global_summary.items():
            f.write(f"{k}: {v}\n")
        f.write("\nTop failure modes (failed rows only):\n")
        if mode_counts_df.empty:
            f.write("  (none)\n")
        else:
            for _, r in mode_counts_df.head(15).iterrows():
                f.write(f"  - {r['primary_failure_mode']}: {int(r['count'])}\n")
        f.write("\nRecommendations:\n")
        if recs.empty:
            f.write("  (none)\n")
        else:
            for _, r in recs.iterrows():
                f.write(f"  [{int(r['priority'])}] {r['topic']}: {r['recommendation']}\n")
                f.write(f"      evidence: {r['evidence']}\n")

    ok("Debug artifacts generated.")
    log(f"[OK] Output root: {out_dir}")
    log("[OK] Key outputs:")
    log(f"  - {os.path.join(csv_dir, 'failure_mode_summary.csv')}")
    log(f"  - {os.path.join(csv_dir, 'recommended_adjustments.csv')}")
    log(f"  - {os.path.join(csv_dir, 'representative_failed_fit_plot_index.csv')} (if plots generated)")
    log(f"  - {os.path.join(png_dir, 'tau_recovery_diagnostic_histograms.png')}")
    log(f"  - {plots_dir} (representative failed-fit plots by mode)")
    log(f"  - {txt_path}")


if __name__ == "__main__":
    main()