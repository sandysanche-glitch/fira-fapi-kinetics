#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_recover_tempo_tau_fits.py

Recover per-track tau fits for FAPI-TEMPO from tracks.csv (video-pair workflow),
with robust QC metadata and checkpoint-tree compatible outputs.

Writes:
- canonical_inputs/FAPI_TEMPO/tau_fits.csv
- compare_outputs/csv/tempo_tau_recovery_summary.csv
- compare_outputs/csv/tempo_tau_recovery_warnings.csv

Compatibility goals:
- one row per track
- tau in ms (+ tau_s)
- QC metadata columns preserved for harmonizer/compare:
    fit_r2, fit_nrmse_range, fit_window_over_tau, tau_source_status
- tau_qc_pass and tau_qc_failure_reasons included
"""

from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


# ============================================================
# Config
# ============================================================

CHECKPOINT_VERSION = "v001_video_pair_fapi_vs_fapi_tempo"
UM_PER_PX = 0.065  # context only (tau is time-based)

# Practical tau QC defaults (video-pair tuned, conservative but usable)
TAU_QC_CONFIG: Dict[str, float | int | bool] = {
    # Data sufficiency
    "min_points_total": 8,
    "min_points_fit": 6,
    "min_fit_span_ms": 20.0,

    # Radius signal quality
    "min_dynamic_range_px": 3.0,

    # Tau bounds (ms)
    "tau_lower_ms": 2.0,
    "tau_upper_ms": 400.0,
    "tau_grid_size": 240,

    # Fit quality
    "min_fit_r2": 0.75,
    "max_fit_nrmse_range": 0.30,
    "min_fit_window_over_tau": 1.10,
    "max_fit_window_over_tau": 20.0,

    # Shape/noise sanity
    "require_monotonic_time": True,
    "allow_negative_steps_fraction": 0.35,
    "max_tau_over_track_span": 3.0,

    # Bound hits treated as fail for publication-robust tau panel
    "fail_if_hit_tau_bounds": True,
}

FIT_CONFIG: Dict[str, Any] = {
    "use_track_summary_for_t_nuc": True,
    "trim_initial_points": 0,
    "max_points_per_track": None,
    "drop_duplicate_times": True,
    "dedupe_keep": "last",  # "first" or "last"
}

# Input schema aliases (important fix for your current tracks.csv)
TRACKS_SCHEMA_ALIASES = {
    "track_id": ["track_id", "track", "id_track"],
    "time_ms": ["time_ms", "t_ms", "t", "time", "time_msec"],
    "R_px": ["R_px", "R_mono", "radius_px", "r_px"],
    "area_px": ["area_px", "area", "area_pixels"],
}

TRACK_SUMMARY_TNUC_ALIASES = ["t_nuc_ms", "nuc_time_ms", "t0_ms", "t_start_ms"]


# ============================================================
# Logging
# ============================================================

def info(msg: str) -> None:
    print(f"[INFO] {msg}")

def warn(msg: str) -> None:
    print(f"[WARN] {msg}")

def ok(msg: str) -> None:
    print(f"[OK] {msg}")

def err(msg: str) -> None:
    print(f"[ERROR] {msg}")


# ============================================================
# Path helpers
# ============================================================

def find_checkpoint_dir_from_here() -> Path:
    """
    Robustly find:
      .../harmonization_checkpoints/v001_video_pair_fapi_vs_fapi_tempo
    from current working dir or script location.
    """
    starts: List[Path] = []
    try:
        starts.append(Path(__file__).resolve())
    except Exception:
        pass
    starts.append(Path.cwd().resolve())

    for s in starts:
        chain = [s] + list(s.parents)
        for p in chain:
            if p.name == CHECKPOINT_VERSION and p.parent.name == "harmonization_checkpoints":
                return p
            nested = p / "harmonization_checkpoints" / CHECKPOINT_VERSION
            if nested.exists():
                return nested.resolve()

    raise FileNotFoundError(
        f"Could not locate checkpoint directory '{CHECKPOINT_VERSION}'. "
        "Run from Kinetics root or from the checkpoint scripts folder."
    )

def infer_kinetics_root(checkpoint_dir: Path) -> Path:
    # .../Kinetics/harmonization_checkpoints/v001...
    return checkpoint_dir.parent.parent.resolve()

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


# ============================================================
# General helpers
# ============================================================

def pick_first_existing_col(df: pd.DataFrame, candidates: List[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Missing required column for {label}: candidates={candidates}, found={list(df.columns)}")

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def safe_float(v: Any) -> float:
    try:
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")

def safe_int(v: Any, default: int = -1) -> int:
    try:
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return default
        return int(float(v))
    except Exception:
        return default

def write_csv_even_if_empty(df: pd.DataFrame, path: Path, columns_if_empty: Optional[List[str]] = None) -> None:
    ensure_dir(path.parent)
    if df is None or df.empty:
        if columns_if_empty is None:
            pd.DataFrame().to_csv(path, index=False)
        else:
            pd.DataFrame(columns=columns_if_empty).to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


# ============================================================
# QC evaluation (drop-in compatible)
# ============================================================

def evaluate_tau_qc(
    fit_row: Dict[str, Any],
    cfg: Dict[str, float | int | bool] = TAU_QC_CONFIG,
) -> Tuple[bool, List[str]]:
    """
    Returns:
      (tau_qc_pass, qc_failure_reasons)
    """
    reasons: List[str] = []

    fit_status = str(fit_row.get("fit_status", "unknown"))
    if fit_status != "ok":
        reasons.append(f"fit_status:{fit_status}")
        return False, reasons

    n_total = safe_int(fit_row.get("n_points_total"))
    n_fit = safe_int(fit_row.get("n_points_fit"))
    fit_span_ms = safe_float(fit_row.get("fit_span_ms"))
    tau_ms = safe_float(fit_row.get("tau_ms"))
    fit_r2 = safe_float(fit_row.get("fit_r2"))
    fit_nrmse = safe_float(fit_row.get("fit_nrmse_range"))
    fit_window_over_tau = safe_float(fit_row.get("fit_window_over_tau"))
    dynamic_range_px = safe_float(fit_row.get("dynamic_range_px"))
    neg_step_frac = safe_float(fit_row.get("neg_step_frac"))
    tau_over_track_span = safe_float(fit_row.get("tau_over_track_span"))

    # Data sufficiency
    if n_total < int(cfg["min_points_total"]):
        reasons.append("too_few_points_total")
    if n_fit < int(cfg["min_points_fit"]):
        reasons.append("too_few_points_fit")
    if not np.isfinite(fit_span_ms) or fit_span_ms < float(cfg["min_fit_span_ms"]):
        reasons.append("fit_span_too_short")

    # Tau existence and bounds
    if not np.isfinite(tau_ms) or tau_ms <= 0:
        reasons.append("tau_invalid")
    else:
        if tau_ms < float(cfg["tau_lower_ms"]) or tau_ms > float(cfg["tau_upper_ms"]):
            reasons.append("tau_out_of_bounds")

    # Signal quality
    if not np.isfinite(dynamic_range_px) or dynamic_range_px < float(cfg["min_dynamic_range_px"]):
        reasons.append("dynamic_range_too_small")

    # Fit quality
    if not np.isfinite(fit_r2) or fit_r2 < float(cfg["min_fit_r2"]):
        reasons.append("fit_r2_low")
    if not np.isfinite(fit_nrmse) or fit_nrmse > float(cfg["max_fit_nrmse_range"]):
        reasons.append("fit_nrmse_high")

    # Window adequacy
    if not np.isfinite(fit_window_over_tau) or fit_window_over_tau < float(cfg["min_fit_window_over_tau"]):
        reasons.append("fit_window_too_short_vs_tau")
    if np.isfinite(fit_window_over_tau) and fit_window_over_tau > float(cfg["max_fit_window_over_tau"]):
        reasons.append("fit_window_too_long_vs_tau")

    if np.isfinite(tau_over_track_span) and tau_over_track_span > float(cfg["max_tau_over_track_span"]):
        reasons.append("tau_poorly_constrained_vs_track_span")

    # Noise / non-monotonicity proxy
    if np.isfinite(neg_step_frac) and neg_step_frac > float(cfg["allow_negative_steps_fraction"]):
        reasons.append("too_many_negative_steps")

    # Boundary hits
    hit_lo = bool(fit_row.get("hit_tau_lower_bound", False))
    hit_hi = bool(fit_row.get("hit_tau_upper_bound", False))
    if bool(cfg.get("fail_if_hit_tau_bounds", True)):
        if hit_lo:
            reasons.append("hit_tau_lower_bound")
        if hit_hi:
            reasons.append("hit_tau_upper_bound")

    return (len(reasons) == 0), reasons


# ============================================================
# Fit model
# ============================================================

@dataclass
class FitResult:
    fit_status: str
    tau_ms: float
    fit_r2: float
    fit_rmse_px: float
    fit_nrmse_range: float
    fit_window_over_tau: float
    fit_span_ms: float
    tau_over_track_span: float
    n_points_total: int
    n_points_fit: int
    t_nuc_ms: float
    t_start_fit_ms: float
    t_end_fit_ms: float
    dynamic_range_px: float
    R0_px: float
    R_end_px: float
    R_max_px: float
    amp_px: float
    neg_step_frac: float
    hit_tau_lower_bound: bool
    hit_tau_upper_bound: bool
    notes: str

def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return np.nan
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    y_bar = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_bar) ** 2))
    if ss_tot <= 0:
        return np.nan
    return 1.0 - ss_res / ss_tot

def fit_tau_exponential_grid(
    t_ms: np.ndarray,
    r_px: np.ndarray,
    tau_lower_ms: float,
    tau_upper_ms: float,
    tau_grid_size: int,
) -> Tuple[float, np.ndarray, Dict[str, float]]:
    """
    Fit:
      R(t) = R0 + A * (1 - exp(-(t - t0)/tau))
    with:
      - R0 fixed to first point
      - tau via log-grid search
      - A by analytical least squares
    """
    t = np.asarray(t_ms, dtype=float)
    y = np.asarray(r_px, dtype=float)

    if len(t) < 3:
        raise ValueError("Need at least 3 points to fit tau")
    if np.any(np.diff(t) < 0):
        raise ValueError("Time must be non-decreasing")

    dt = t - t[0]
    if float(np.nanmax(dt)) <= 0:
        raise ValueError("Zero time span in fit window")

    R0 = float(y[0])
    y_shift = y - R0

    taus = np.logspace(np.log10(tau_lower_ms), np.log10(tau_upper_ms), int(tau_grid_size))
    best = None

    for tau in taus:
        basis = 1.0 - np.exp(-dt / tau)
        denom = float(np.dot(basis, basis))
        if denom <= 0:
            continue
        A = float(np.dot(basis, y_shift) / denom)
        if A < 0:
            A = 0.0  # non-negative amplitude only
        yhat = R0 + A * basis
        sse = float(np.sum((y - yhat) ** 2))
        if (best is None) or (sse < best["sse"]):
            best = {"tau": float(tau), "A": A, "yhat": yhat, "sse": sse}

    if best is None:
        raise RuntimeError("Grid fitting failed")

    tau = float(best["tau"])
    yhat = np.asarray(best["yhat"], dtype=float)
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    y_range = float(np.nanmax(y) - np.nanmin(y))
    nrmse_range = rmse / y_range if y_range > 0 else np.nan
    r2 = _safe_r2(y, yhat)

    return tau, yhat, {
        "amp_px": float(best["A"]),
        "rmse_px": rmse,
        "nrmse_range": nrmse_range,
        "r2": r2,
    }

def dedupe_and_sort_track(df_track: pd.DataFrame) -> pd.DataFrame:
    out = df_track.copy()
    out = out.sort_values("time_ms")

    if FIT_CONFIG.get("drop_duplicate_times", True):
        keep = str(FIT_CONFIG.get("dedupe_keep", "last")).lower()
        if keep not in {"first", "last"}:
            keep = "last"
        out = out.drop_duplicates(subset=["time_ms"], keep=keep)

    out = out.sort_values("time_ms").reset_index(drop=True)
    return out

def fit_one_track_tau(track_df: pd.DataFrame, t_nuc_hint_ms: Optional[float] = None) -> FitResult:
    """
    track_df must contain standardized columns:
      time_ms, R_px
    """
    df = track_df.copy()
    df["time_ms"] = to_num(df["time_ms"])
    df["R_px"] = to_num(df["R_px"])
    df = df.dropna(subset=["time_ms", "R_px"]).copy()

    if df.empty:
        return FitResult(
            fit_status="empty_track",
            tau_ms=np.nan, fit_r2=np.nan, fit_rmse_px=np.nan, fit_nrmse_range=np.nan,
            fit_window_over_tau=np.nan, fit_span_ms=np.nan, tau_over_track_span=np.nan,
            n_points_total=0, n_points_fit=0, t_nuc_ms=np.nan, t_start_fit_ms=np.nan, t_end_fit_ms=np.nan,
            dynamic_range_px=np.nan, R0_px=np.nan, R_end_px=np.nan, R_max_px=np.nan,
            amp_px=np.nan, neg_step_frac=np.nan,
            hit_tau_lower_bound=False, hit_tau_upper_bound=False,
            notes="no valid points"
        )

    max_pts = FIT_CONFIG.get("max_points_per_track", None)
    if max_pts is not None:
        df = df.iloc[: int(max_pts)].copy()

    df = dedupe_and_sort_track(df)
    n_total = int(len(df))

    if bool(TAU_QC_CONFIG.get("require_monotonic_time", True)):
        if (df["time_ms"].diff().dropna() < 0).any():
            return FitResult(
                fit_status="nonmonotonic_time",
                tau_ms=np.nan, fit_r2=np.nan, fit_rmse_px=np.nan, fit_nrmse_range=np.nan,
                fit_window_over_tau=np.nan, fit_span_ms=np.nan, tau_over_track_span=np.nan,
                n_points_total=n_total, n_points_fit=0, t_nuc_ms=np.nan, t_start_fit_ms=np.nan, t_end_fit_ms=np.nan,
                dynamic_range_px=np.nan, R0_px=np.nan, R_end_px=np.nan, R_max_px=np.nan,
                amp_px=np.nan, neg_step_frac=np.nan,
                hit_tau_lower_bound=False, hit_tau_upper_bound=False,
                notes="time not monotonic"
            )

    trim_n = int(FIT_CONFIG.get("trim_initial_points", 0) or 0)
    if trim_n > 0 and len(df) > trim_n:
        df = df.iloc[trim_n:].copy().reset_index(drop=True)

    t_first = float(df["time_ms"].iloc[0])
    t_start = t_first
    if t_nuc_hint_ms is not None and np.isfinite(t_nuc_hint_ms):
        t_start = max(t_first, float(t_nuc_hint_ms))

    fit_df = df[df["time_ms"] >= t_start].copy().sort_values("time_ms").reset_index(drop=True)
    n_fit = int(len(fit_df))

    if fit_df.empty:
        return FitResult(
            fit_status="empty_fit_window",
            tau_ms=np.nan, fit_r2=np.nan, fit_rmse_px=np.nan, fit_nrmse_range=np.nan,
            fit_window_over_tau=np.nan, fit_span_ms=np.nan, tau_over_track_span=np.nan,
            n_points_total=n_total, n_points_fit=0, t_nuc_ms=t_start, t_start_fit_ms=np.nan, t_end_fit_ms=np.nan,
            dynamic_range_px=np.nan, R0_px=np.nan, R_end_px=np.nan, R_max_px=np.nan,
            amp_px=np.nan, neg_step_frac=np.nan,
            hit_tau_lower_bound=False, hit_tau_upper_bound=False,
            notes="no points after fit start"
        )

    t_arr = fit_df["time_ms"].to_numpy(dtype=float)
    r_arr = fit_df["R_px"].to_numpy(dtype=float)

    fit_span_ms = float(t_arr[-1] - t_arr[0]) if len(t_arr) >= 2 else 0.0
    dynamic_range_px = float(np.nanmax(r_arr) - np.nanmin(r_arr)) if len(r_arr) else np.nan
    R0_px = float(r_arr[0]) if len(r_arr) else np.nan
    R_end_px = float(r_arr[-1]) if len(r_arr) else np.nan
    R_max_px = float(np.nanmax(r_arr)) if len(r_arr) else np.nan

    neg_step_frac = float(np.mean(np.diff(r_arr) < 0)) if len(r_arr) >= 2 else np.nan

    try:
        tau_ms, yhat, fm = fit_tau_exponential_grid(
            t_ms=t_arr,
            r_px=r_arr,
            tau_lower_ms=float(TAU_QC_CONFIG["tau_lower_ms"]),
            tau_upper_ms=float(TAU_QC_CONFIG["tau_upper_ms"]),
            tau_grid_size=int(TAU_QC_CONFIG["tau_grid_size"]),
        )

        fit_window_over_tau = (fit_span_ms / tau_ms) if (np.isfinite(tau_ms) and tau_ms > 0) else np.nan
        track_span_ms = float(df["time_ms"].iloc[-1] - df["time_ms"].iloc[0]) if len(df) >= 2 else np.nan
        tau_over_track_span = (tau_ms / track_span_ms) if (np.isfinite(track_span_ms) and track_span_ms > 0) else np.nan

        # "near bound" detection (grid exact hit or numerical near-hit)
        lo = float(TAU_QC_CONFIG["tau_lower_ms"])
        hi = float(TAU_QC_CONFIG["tau_upper_ms"])
        hit_lo = bool(abs(tau_ms - lo) / max(lo, 1e-9) < 1e-6)
        hit_hi = bool(abs(tau_ms - hi) / max(hi, 1e-9) < 1e-6)

        notes = []
        if hit_lo:
            notes.append("tau_at_lower_bound")
        if hit_hi:
            notes.append("tau_at_upper_bound")

        return FitResult(
            fit_status="ok",
            tau_ms=float(tau_ms),
            fit_r2=float(fm["r2"]),
            fit_rmse_px=float(fm["rmse_px"]),
            fit_nrmse_range=float(fm["nrmse_range"]),
            fit_window_over_tau=float(fit_window_over_tau) if np.isfinite(fit_window_over_tau) else np.nan,
            fit_span_ms=float(fit_span_ms),
            tau_over_track_span=float(tau_over_track_span) if np.isfinite(tau_over_track_span) else np.nan,
            n_points_total=n_total,
            n_points_fit=n_fit,
            t_nuc_ms=float(t_start),
            t_start_fit_ms=float(t_arr[0]) if len(t_arr) else np.nan,
            t_end_fit_ms=float(t_arr[-1]) if len(t_arr) else np.nan,
            dynamic_range_px=float(dynamic_range_px),
            R0_px=float(R0_px),
            R_end_px=float(R_end_px),
            R_max_px=float(R_max_px),
            amp_px=float(fm["amp_px"]),
            neg_step_frac=float(neg_step_frac) if np.isfinite(neg_step_frac) else np.nan,
            hit_tau_lower_bound=hit_lo,
            hit_tau_upper_bound=hit_hi,
            notes=";".join(notes)
        )
    except Exception as e:
        return FitResult(
            fit_status="fit_exception",
            tau_ms=np.nan, fit_r2=np.nan, fit_rmse_px=np.nan, fit_nrmse_range=np.nan,
            fit_window_over_tau=np.nan, fit_span_ms=float(fit_span_ms) if np.isfinite(fit_span_ms) else np.nan,
            tau_over_track_span=np.nan, n_points_total=n_total, n_points_fit=n_fit,
            t_nuc_ms=float(t_start), t_start_fit_ms=float(t_arr[0]) if len(t_arr) else np.nan,
            t_end_fit_ms=float(t_arr[-1]) if len(t_arr) else np.nan,
            dynamic_range_px=float(dynamic_range_px) if np.isfinite(dynamic_range_px) else np.nan,
            R0_px=float(R0_px) if np.isfinite(R0_px) else np.nan,
            R_end_px=float(R_end_px) if np.isfinite(R_end_px) else np.nan,
            R_max_px=float(R_max_px) if np.isfinite(R_max_px) else np.nan,
            amp_px=np.nan, neg_step_frac=float(neg_step_frac) if np.isfinite(neg_step_frac) else np.nan,
            hit_tau_lower_bound=False, hit_tau_upper_bound=False,
            notes=str(e)
        )


# ============================================================
# Input loading / schema normalization
# ============================================================

def load_tracks_csv_robust(path: Path) -> pd.DataFrame:
    """
    Robust loader for FAPI_TEMPO tracks.csv.
    Supports your current schema:
      ['dataset','track_id','frame_idx','t_ms','det_id',...,'R_px',...,'R_mono']
    """
    info(f"Using tracks source: {path}")
    df = pd.read_csv(path)

    # Detect columns
    c_track = pick_first_existing_col(df, TRACKS_SCHEMA_ALIASES["track_id"], "track_id")
    c_time = pick_first_existing_col(df, TRACKS_SCHEMA_ALIASES["time_ms"], "time_ms")
    c_r = None
    for cand in TRACKS_SCHEMA_ALIASES["R_px"]:
        if cand in df.columns:
            c_r = cand
            break

    # If no direct radius, derive from area
    if c_r is None:
        if any(c in df.columns for c in TRACKS_SCHEMA_ALIASES["area_px"]):
            c_area = pick_first_existing_col(df, TRACKS_SCHEMA_ALIASES["area_px"], "area_px")
            df["__R_px_from_area__"] = np.sqrt(pd.to_numeric(df[c_area], errors="coerce") / math.pi)
            c_r = "__R_px_from_area__"
            warn(f"tracks.csv has no direct radius column; derived R_px from {c_area}.")
        else:
            raise KeyError(
                f"tracks.csv: could not find radius column candidates={TRACKS_SCHEMA_ALIASES['R_px']} "
                f"or area column candidates={TRACKS_SCHEMA_ALIASES['area_px']}; found={list(df.columns)}"
            )

    out = pd.DataFrame({
        "track_id": pd.to_numeric(df[c_track], errors="coerce"),
        "time_ms": pd.to_numeric(df[c_time], errors="coerce"),
        "R_px": pd.to_numeric(df[c_r], errors="coerce"),
    })

    # keep optional columns if present (useful for debugging/provenance)
    if "frame_idx" in df.columns:
        out["frame_idx"] = pd.to_numeric(df["frame_idx"], errors="coerce")
    if "det_id" in df.columns:
        out["det_id"] = pd.to_numeric(df["det_id"], errors="coerce")

    before = len(out)
    out = out.dropna(subset=["track_id", "time_ms", "R_px"]).copy()
    out["track_id"] = out["track_id"].astype(int)
    dropped = before - len(out)
    if dropped > 0:
        warn(f"tracks.csv: dropped {dropped} rows with non-numeric track_id/time/radius.")

    if out.empty:
        raise ValueError("tracks.csv normalized table is empty after cleaning.")

    info(
        "tracks.csv schema mapped as: "
        f"track_id={c_track}, time_ms={c_time}, R_px={c_r} "
        f"(rows={len(out)}, tracks={out['track_id'].nunique()})"
    )
    return out

def load_track_summary_optional(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    info(f"Using track_summary source (optional): {path}")
    ts = pd.read_csv(path)
    if "track_id" not in ts.columns:
        warn("track_summary.csv found but missing track_id; ignoring it for t_nuc hints.")
        return None
    ts["track_id"] = pd.to_numeric(ts["track_id"], errors="coerce")
    ts = ts.dropna(subset=["track_id"]).copy()
    ts["track_id"] = ts["track_id"].astype(int)
    return ts

def extract_t_nuc_map(track_summary: Optional[pd.DataFrame]) -> Dict[int, float]:
    if track_summary is None or track_summary.empty:
        return {}
    col = next((c for c in TRACK_SUMMARY_TNUC_ALIASES if c in track_summary.columns), None)
    if col is None:
        warn(
            "track_summary present but no nucleation-time hint column found "
            f"among {TRACK_SUMMARY_TNUC_ALIASES}; fitting starts from first track point."
        )
        return {}
    tmp = track_summary[["track_id", col]].copy()
    tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
    tmp = tmp.dropna(subset=[col])
    return dict(zip(tmp["track_id"].astype(int), tmp[col].astype(float)))


# ============================================================
# Builder
# ============================================================

def build_tau_fits_from_tracks(
    tracks_df: pd.DataFrame,
    t_nuc_hint_map: Optional[Dict[int, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t_nuc_hint_map = t_nuc_hint_map or {}

    rows: List[Dict[str, Any]] = []
    warn_rows: List[Dict[str, Any]] = []

    grouped = tracks_df.groupby("track_id", sort=True)
    n_tracks = len(grouped)
    info(f"Fitting tau per track ... n_tracks={n_tracks}")

    for i, (track_id, g) in enumerate(grouped, start=1):
        hint = t_nuc_hint_map.get(int(track_id), None)
        fit = fit_one_track_tau(g[["time_ms", "R_px"]], t_nuc_hint_ms=hint)

        row: Dict[str, Any] = {
            "track_id": int(track_id),

            # Core tau
            "tau_ms": fit.tau_ms,
            "tau_s": (fit.tau_ms / 1000.0) if np.isfinite(fit.tau_ms) else np.nan,

            # Harmonization / compare compatibility fields
            "tau_source_status": "recovered_from_tracks",
            "fit_status": fit.fit_status,
            "fit_r2": fit.fit_r2,
            "fit_nrmse_range": fit.fit_nrmse_range,
            "fit_window_over_tau": fit.fit_window_over_tau,

            # QC metadata
            "n_points_total": fit.n_points_total,
            "n_points_fit": fit.n_points_fit,
            "fit_span_ms": fit.fit_span_ms,
            "tau_over_track_span": fit.tau_over_track_span,

            # Fit time range metadata
            "t_nuc_ms": fit.t_nuc_ms,
            "t_start_fit_ms": fit.t_start_fit_ms,
            "t_end_fit_ms": fit.t_end_fit_ms,

            # Track geometry metadata (schema style similar to FAPI_tau_fits.csv + extra QC)
            "R0_px": fit.R0_px,
            "R_end_px": fit.R_end_px,
            "R_max_px": fit.R_max_px,
            "dynamic_range_px": fit.dynamic_range_px,
            "fit_rmse_px": fit.fit_rmse_px,
            "amp_px": fit.amp_px,
            "neg_step_frac": fit.neg_step_frac,
            "hit_tau_lower_bound": bool(fit.hit_tau_lower_bound),
            "hit_tau_upper_bound": bool(fit.hit_tau_upper_bound),
            "notes": fit.notes,
        }

        tau_qc_pass, qc_reasons = evaluate_tau_qc(row, TAU_QC_CONFIG)
        row["tau_qc_pass"] = bool(tau_qc_pass)
        row["tau_qc_failure_reasons"] = ";".join(qc_reasons)

        rows.append(row)

        if (fit.fit_status != "ok") or (not tau_qc_pass):
            warn_rows.append({
                "track_id": int(track_id),
                "fit_status": fit.fit_status,
                "tau_qc_pass": bool(tau_qc_pass),
                "tau_qc_failure_reasons": ";".join(qc_reasons),
                "tau_ms": fit.tau_ms,
                "fit_r2": fit.fit_r2,
                "fit_nrmse_range": fit.fit_nrmse_range,
                "fit_window_over_tau": fit.fit_window_over_tau,
                "n_points_total": fit.n_points_total,
                "n_points_fit": fit.n_points_fit,
                "fit_span_ms": fit.fit_span_ms,
                "dynamic_range_px": fit.dynamic_range_px,
                "tau_over_track_span": fit.tau_over_track_span,
                "neg_step_frac": fit.neg_step_frac,
                "hit_tau_lower_bound": bool(fit.hit_tau_lower_bound),
                "hit_tau_upper_bound": bool(fit.hit_tau_upper_bound),
                "notes": fit.notes,
            })

        if (i % 200) == 0:
            info(f"  progress: {i}/{n_tracks} tracks")

    tau_fits_df = pd.DataFrame(rows).sort_values("track_id").reset_index(drop=True)
    warnings_df = pd.DataFrame(warn_rows)

    # Summary table
    if tau_fits_df.empty:
        summary_df = pd.DataFrame([
            {"metric": "n_tracks_total", "value": 0},
            {"metric": "n_fit_status_ok", "value": 0},
            {"metric": "n_tau_qc_pass", "value": 0},
        ])
        return tau_fits_df, warnings_df, summary_df

    ok_mask = tau_fits_df["fit_status"].astype(str).eq("ok")
    pass_mask = tau_fits_df["tau_qc_pass"].fillna(False).astype(bool)

    ok_tau = pd.to_numeric(tau_fits_df.loc[ok_mask, "tau_ms"], errors="coerce").dropna()
    pass_tau = pd.to_numeric(tau_fits_df.loc[pass_mask, "tau_ms"], errors="coerce").dropna()

    def qstat(s: pd.Series, q: float) -> float:
        return float(s.quantile(q)) if not s.empty else np.nan

    summary_rows = [
        {"metric": "n_tracks_total", "value": int(len(tau_fits_df))},
        {"metric": "n_fit_status_ok", "value": int(ok_mask.sum())},
        {"metric": "n_tau_qc_pass", "value": int(pass_mask.sum())},
        {"metric": "frac_tau_qc_pass_of_total", "value": float(pass_mask.mean())},
        {"metric": "frac_tau_qc_pass_of_fit_ok", "value": float((pass_mask & ok_mask).sum() / max(int(ok_mask.sum()), 1))},
        {"metric": "tau_ms_ok_count", "value": int(ok_tau.count())},
        {"metric": "tau_ms_ok_median", "value": float(ok_tau.median()) if not ok_tau.empty else np.nan},
        {"metric": "tau_ms_ok_p10", "value": qstat(ok_tau, 0.10)},
        {"metric": "tau_ms_ok_p90", "value": qstat(ok_tau, 0.90)},
        {"metric": "tau_ms_qcpass_count", "value": int(pass_tau.count())},
        {"metric": "tau_ms_qcpass_median", "value": float(pass_tau.median()) if not pass_tau.empty else np.nan},
        {"metric": "tau_ms_qcpass_p10", "value": qstat(pass_tau, 0.10)},
        {"metric": "tau_ms_qcpass_p90", "value": qstat(pass_tau, 0.90)},
    ]

    # Failure mode counts (top-level)
    if not warnings_df.empty and "tau_qc_failure_reasons" in warnings_df.columns:
        exploded = (
            warnings_df["tau_qc_failure_reasons"]
            .fillna("")
            .astype(str)
            .str.split(";")
            .explode()
            .str.strip()
        )
        exploded = exploded[(exploded != "") & exploded.notna()]
        if not exploded.empty:
            vc = exploded.value_counts()
            for k, v in vc.items():
                summary_rows.append({"metric": f"fail_reason_count::{k}", "value": int(v)})

    summary_df = pd.DataFrame(summary_rows)
    return tau_fits_df, warnings_df, summary_df


# ============================================================
# Main
# ============================================================

def main() -> int:
    try:
        info("Recovering true tau fits for FAPI-TEMPO ...")

        checkpoint_dir = find_checkpoint_dir_from_here()
        kinetics_root = infer_kinetics_root(checkpoint_dir)

        info(f"KINETICS_ROOT   : {kinetics_root}")
        info(f"CHECKPOINT_DIR  : {checkpoint_dir}")

        # Candidate sources (checkpoint-tree compatible)
        tracks_candidates = [
            kinetics_root / "out" / "FAPI_TEMPO" / "tracks.csv",
            checkpoint_dir / "canonical_inputs" / "FAPI_TEMPO" / "tracks_source.csv",
            Path.cwd() / "tracks.csv",
        ]
        track_summary_candidates = [
            kinetics_root / "out" / "FAPI_TEMPO" / "track_summary.csv",
            checkpoint_dir / "raw_snapshot" / "FAPI_TEMPO" / "out_FAPI_TEMPO_track_summary" / "track_summary.csv",
        ]

        tracks_path = first_existing(tracks_candidates)
        if tracks_path is None:
            raise FileNotFoundError(
                "Could not find FAPI_TEMPO tracks.csv. Checked:\n  - "
                + "\n  - ".join(str(p) for p in tracks_candidates)
            )

        track_summary_path = first_existing(track_summary_candidates)

        tracks_df = load_tracks_csv_robust(tracks_path)
        track_summary_df = load_track_summary_optional(track_summary_path)
        t_nuc_map = extract_t_nuc_map(track_summary_df) if bool(FIT_CONFIG.get("use_track_summary_for_t_nuc", True)) else {}

        tau_fits_df, warnings_df, summary_df = build_tau_fits_from_tracks(tracks_df, t_nuc_hint_map=t_nuc_map)

        # Outputs
        canonical_dir = checkpoint_dir / "canonical_inputs" / "FAPI_TEMPO"
        compare_csv_dir = checkpoint_dir / "compare_outputs" / "csv"
        ensure_dir(canonical_dir)
        ensure_dir(compare_csv_dir)

        out_tau = canonical_dir / "tau_fits.csv"
        out_summary = compare_csv_dir / "tempo_tau_recovery_summary.csv"
        out_warn = compare_csv_dir / "tempo_tau_recovery_warnings.csv"

        write_csv_even_if_empty(
            tau_fits_df,
            out_tau,
            columns_if_empty=[
                "track_id", "tau_ms", "tau_s", "tau_source_status", "fit_status",
                "fit_r2", "fit_nrmse_range", "fit_window_over_tau",
                "n_points_total", "n_points_fit", "fit_span_ms", "tau_over_track_span",
                "t_nuc_ms", "t_start_fit_ms", "t_end_fit_ms",
                "R0_px", "R_end_px", "R_max_px", "dynamic_range_px",
                "fit_rmse_px", "amp_px", "neg_step_frac",
                "hit_tau_lower_bound", "hit_tau_upper_bound",
                "tau_qc_pass", "tau_qc_failure_reasons", "notes",
            ],
        )
        write_csv_even_if_empty(summary_df, out_summary, columns_if_empty=["metric", "value"])
        write_csv_even_if_empty(
            warnings_df,
            out_warn,
            columns_if_empty=[
                "track_id", "fit_status", "tau_qc_pass", "tau_qc_failure_reasons",
                "tau_ms", "fit_r2", "fit_nrmse_range", "fit_window_over_tau",
                "n_points_total", "n_points_fit", "fit_span_ms", "dynamic_range_px",
                "tau_over_track_span", "neg_step_frac", "hit_tau_lower_bound", "hit_tau_upper_bound", "notes"
            ],
        )

        # If tau_unavailable.flag exists, remove it now
        tau_unavail_flag = canonical_dir / "tau_unavailable.flag"
        if tau_unavail_flag.exists():
            try:
                tau_unavail_flag.unlink()
                ok(f"Removed flag: {tau_unavail_flag}")
            except Exception as e:
                warn(f"Could not remove tau_unavailable.flag: {e}")

        # Console summary
        n_total = int(len(tau_fits_df))
        n_ok = int((tau_fits_df["fit_status"] == "ok").sum()) if n_total else 0
        n_pass = int(tau_fits_df["tau_qc_pass"].fillna(False).astype(bool).sum()) if n_total else 0

        pass_tau = pd.to_numeric(tau_fits_df.loc[tau_fits_df["tau_qc_pass"] == True, "tau_ms"], errors="coerce").dropna()
        if not pass_tau.empty:
            info(
                "QC-pass tau stats (ms): "
                f"median={pass_tau.median():.3f}, "
                f"p10={pass_tau.quantile(0.10):.3f}, "
                f"p90={pass_tau.quantile(0.90):.3f}"
            )
        else:
            warn("No QC-pass tau fits. Inspect tempo_tau_recovery_warnings.csv and tune TAU_QC_CONFIG if needed.")

        ok("Wrote:")
        print(f"  {out_tau}")
        print(f"  {out_summary}")
        print(f"  {out_warn}")

        info(f"[SUMMARY] tracks_total={n_total}")
        info(f"[SUMMARY] fit_status_ok={n_ok}")
        info(f"[SUMMARY] tau_qc_pass={n_pass}")

        info("Next:")
        print("  1) python harmonize_video_pair_inputs.py")
        print("  2) python compare_video_pair_harmonized.py")
        print("  3) tau panel should activate automatically if tau_fits.csv is accepted")

        return 0

    except Exception as e:
        err("Failed while building recovered tau_fits table:")
        print(str(e))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())