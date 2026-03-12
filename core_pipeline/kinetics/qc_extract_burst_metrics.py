#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
qc_extract_burst_metrics.py

Compute "burst" metrics from time series traces grouped by dataset:
- X(t): growth proxy (e.g. max_bbox_frac_kept)
- n(t): number of grains kept (e.g. n_kept)

Outputs:
- metrics CSV
- QC plots (X, dX/dt, dn/dt)

Key design choices to avoid NA/empty plots:
- Robust column detection + explicit overrides
- Robust normalization (none|max|p99) with safe denominators
- Robust t_ind/t_sat detection with persistence AND fallbacks
- Burst window detection based on dn/dt peak with frac_onset and persistence
- FWHM computed on baseline-corrected curve with fallbacks (NA if no crossings)
- Always plots full traces; window overlays are optional
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------

def _nan_safe(x):
    if x is None:
        return np.nan
    try:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return np.nan
    except Exception:
        pass
    return x


def moving_average(y: np.ndarray, win: int) -> np.ndarray:
    """Centered moving average (odd window recommended)."""
    if win is None or win <= 1:
        return y.astype(float).copy()
    win = int(win)
    if win < 2:
        return y.astype(float).copy()
    # pad by edge to keep length
    pad = win // 2
    ypad = np.pad(y.astype(float), (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=float) / float(win)
    ys = np.convolve(ypad, kernel, mode="valid")
    return ys


def robust_p99(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    p = np.nanpercentile(x, 99)
    if not np.isfinite(p) or p <= 0:
        return float(np.nanmax(x)) if np.nanmax(x) > 0 else 1.0
    return float(p)


def safe_divide(a: np.ndarray, denom: float) -> np.ndarray:
    if denom is None or not np.isfinite(denom) or denom == 0:
        denom = 1.0
    return a.astype(float) / float(denom)


def first_persistent_crossing(t: np.ndarray, y: np.ndarray, thr: float, M: int) -> Optional[float]:
    """
    Return first t where y >= thr for at least M consecutive points.
    """
    if y.size == 0:
        return None
    M = max(1, int(M))
    mask = (y >= thr) & np.isfinite(y)
    if not np.any(mask):
        return None
    # find runs of True
    run = 0
    for i, ok in enumerate(mask):
        if ok:
            run += 1
            if run >= M:
                return float(t[i - M + 1])
        else:
            run = 0
    return None


def derivative(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    dy/dt using numpy gradient with respect to t.
    Handles non-uniform spacing.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size < 2:
        return np.full_like(y, np.nan, dtype=float)
    # gradient returns same length
    return np.gradient(y, t)


def median_baseline(y: np.ndarray, t: np.ndarray, peak_t: float, frac_pre: float = 0.25) -> float:
    """
    Robust baseline: median of y in an early pre-peak region.
    We use t <= (t_min + frac_pre*(peak_t - t_min)) as default.
    Falls back to median of all finite values.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    finite = np.isfinite(y) & np.isfinite(t)
    if not np.any(finite):
        return 0.0
    tmin = float(np.nanmin(t[finite]))
    # if peak_t not finite, just overall median
    if not np.isfinite(peak_t):
        return float(np.nanmedian(y[finite]))
    cutoff = tmin + float(frac_pre) * (float(peak_t) - tmin)
    pre = finite & (t <= cutoff)
    if np.any(pre):
        return float(np.nanmedian(y[pre]))
    return float(np.nanmedian(y[finite]))


def find_peak(t: np.ndarray, y: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    """Return peak_t, peak_y, peak_idx for max y (finite only)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(t) & np.isfinite(y)
    if not np.any(finite):
        return None, None, None
    idxs = np.where(finite)[0]
    imax_local = idxs[np.nanargmax(y[finite])]
    return float(t[imax_local]), float(y[imax_local]), int(imax_local)


def fwhm_from_curve(
    t: np.ndarray,
    y: np.ndarray,
    peak_idx: int,
    baseline: float
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    FWHM around peak on baseline-corrected amplitude:
      target = baseline + 0.5*(peak - baseline)
    Finds left/right crossing by linear interpolation.
    Returns (left_t, right_t, fwhm).
    If crossings can't be found, returns (None, None, None).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if peak_idx is None or peak_idx < 0 or peak_idx >= y.size:
        return None, None, None
    if y.size < 3:
        return None, None, None

    peak = y[peak_idx]
    if not np.isfinite(peak) or not np.isfinite(baseline):
        return None, None, None

    # If peak not above baseline meaningfully, no FWHM
    if peak <= baseline:
        return None, None, None

    half = baseline + 0.5 * (peak - baseline)

    # Search left
    left = None
    for i in range(peak_idx, 0, -1):
        if not (np.isfinite(y[i]) and np.isfinite(y[i - 1]) and np.isfinite(t[i]) and np.isfinite(t[i - 1])):
            continue
        # crossing when y goes from >= half to < half
        if (y[i] >= half) and (y[i - 1] < half):
            # interpolate between i-1 and i
            y0, y1 = y[i - 1], y[i]
            t0, t1 = t[i - 1], t[i]
            if y1 == y0:
                left = float(t0)
            else:
                alpha = (half - y0) / (y1 - y0)
                left = float(t0 + alpha * (t1 - t0))
            break

    # Search right
    right = None
    for i in range(peak_idx, y.size - 1):
        if not (np.isfinite(y[i]) and np.isfinite(y[i + 1]) and np.isfinite(t[i]) and np.isfinite(t[i + 1])):
            continue
        if (y[i] >= half) and (y[i + 1] < half):
            y0, y1 = y[i], y[i + 1]
            t0, t1 = t[i], t[i + 1]
            if y1 == y0:
                right = float(t1)
            else:
                alpha = (half - y0) / (y1 - y0)
                right = float(t0 + alpha * (t1 - t0))
            break

    if left is None or right is None:
        return None, None, None
    return left, right, float(right - left)


def equiv_width_ms(t: np.ndarray, y: np.ndarray, baseline: float) -> Optional[float]:
    """
    Equivalent width = area/(peak-baseline), area computed over (y-baseline)+ only.
    Requires peak>baseline.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(t) & np.isfinite(y)
    if not np.any(finite):
        return None
    peak = float(np.nanmax(y[finite]))
    if not (np.isfinite(peak) and np.isfinite(baseline)) or peak <= baseline:
        return None
    ycorr = np.maximum(0.0, y - baseline)
    area = float(np.trapz(ycorr[finite], t[finite]))
    amp = peak - baseline
    if amp <= 0:
        return None
    return area / amp


# -----------------------------
# Data model
# -----------------------------

@dataclass
class Metrics:
    dataset: str

    n_points: int
    normalize_method: str
    normalize_denom: float

    X_ind: float
    X_sat: float
    persist_M: int
    smooth_win: int
    fwhm_mode: str
    frac_onset: float

    t_ind_ms: Optional[float]
    t_sat_ms: Optional[float]
    induction_time_X05_ms: Optional[float]

    induction_time_burst_ms: Optional[float]
    growth_duration_ms: Optional[float]

    peak_t_ms: Optional[float]
    peak_dXdt_per_ms: Optional[float]

    fwhm_left_ms: Optional[float]
    fwhm_right_ms: Optional[float]
    fwhm_ms: Optional[float]

    w_eq_ms: Optional[float]
    sharpness_X: Optional[float]
    synchrony_X: Optional[float]

    peak_dndt_per_ms: Optional[float]
    sharpness_n: Optional[float]
    synchrony_n: Optional[float]

    peak_ratio_dndt_over_dXdt: Optional[float]

    # Debug flags (helpful when something is NA)
    used_fallback_t_ind: bool
    used_fallback_t_sat: bool
    used_fallback_window: bool


# -----------------------------
# Core computation per dataset
# -----------------------------

def compute_metrics_for_dataset(
    ds_name: str,
    df: pd.DataFrame,
    time_col: str,
    X_col: str,
    n_col: str,
    smooth_win: int,
    X_ind: float,
    X_sat: float,
    persist_M: int,
    normalize_by: str,
    use_running_max: bool,
    t0_floor_ms: Optional[float],
    fwhm_mode: str,
    frac_onset: float,
    out_prefix: str
) -> Metrics:
    df = df.copy()
    # Sort by time
    df = df.sort_values(time_col, kind="mergesort")
    t = df[time_col].to_numpy(dtype=float)

    # Apply optional floor
    if t0_floor_ms is not None and np.isfinite(t0_floor_ms):
        keep = t >= float(t0_floor_ms)
        df = df.loc[keep].copy()
        t = df[time_col].to_numpy(dtype=float)

    Xraw = df[X_col].to_numpy(dtype=float)
    nraw = df[n_col].to_numpy(dtype=float)

    n_points = int(len(df))

    # --- Normalize Xraw ---
    normalize_by = (normalize_by or "none").lower().strip()
    if normalize_by == "none":
        denom = 1.0
        norm_method = "none"
    elif normalize_by == "max":
        mx = float(np.nanmax(Xraw)) if np.isfinite(np.nanmax(Xraw)) else 1.0
        denom = mx if mx > 0 else 1.0
        norm_method = "max"
    elif normalize_by in ("p99", "pct99", "q99"):
        denom = robust_p99(Xraw)
        norm_method = "p99"
    else:
        # fallback
        denom = robust_p99(Xraw)
        norm_method = "p99"

    X = safe_divide(Xraw, denom)

    # --- Running max monotonicization (recommended) ---
    if use_running_max:
        Xmono = np.maximum.accumulate(X)
    else:
        Xmono = X.copy()

    # --- Find t_ind, t_sat (persistent crossings) ---
    t_ind = first_persistent_crossing(t, Xmono, float(X_ind), persist_M)
    t_sat = first_persistent_crossing(t, Xmono, float(X_sat), persist_M)

    used_fallback_t_ind = False
    used_fallback_t_sat = False

    # Fallbacks so we don't cascade into NA-land
    if t_ind is None:
        t_ind = float(np.nanmin(t)) if t.size else None
        used_fallback_t_ind = True
    if t_sat is None:
        t_sat = float(np.nanmax(t)) if t.size else None
        used_fallback_t_sat = True

    induction_time_X05 = t_ind

    # --- Smooth signals ---
    Xmono_s = moving_average(Xmono, smooth_win)
    n_s = moving_average(nraw, smooth_win)

    # --- Derivatives ---
    dXdt_all = derivative(t, Xmono_s)
    dndt_all = derivative(t, n_s)

    # --- Burst window selection ---
    # Default: growth window between t_ind and t_sat
    mask_growth = np.isfinite(t)
    if t_ind is not None and np.isfinite(t_ind) and t_sat is not None and np.isfinite(t_sat):
        mask_growth = (t >= float(t_ind)) & (t <= float(t_sat)) & np.isfinite(t)

    used_fallback_window = False
    mask_window = mask_growth.copy()

    induction_time_burst = None
    growth_duration = None

    if fwhm_mode == "burst":
        # Use dn/dt to define burst window around its main peak.
        peak_t_n, peak_dn, peak_idx_n = find_peak(t, dndt_all)

        if peak_t_n is None or peak_idx_n is None:
            # fallback to growth window
            used_fallback_window = True
        else:
            base_n = median_baseline(dndt_all, t, peak_t_n, frac_pre=0.25)
            # threshold for onset/offset
            thr = base_n + float(frac_onset) * (peak_dn - base_n)

            # Find onset: first persistent point before peak where dndt crosses up above thr
            onset_t = None
            if np.isfinite(thr):
                # search from start to peak
                pre_idx = np.where(t <= peak_t_n)[0]
                if pre_idx.size > 0:
                    ypre = dndt_all[pre_idx]
                    tpre = t[pre_idx]
                    onset_t = first_persistent_crossing(tpre, ypre, thr, persist_M)

            # Find offset: first persistent point after peak where dndt drops below thr
            offset_t = None
            if np.isfinite(thr):
                post_idx = np.where(t >= peak_t_n)[0]
                if post_idx.size > 0:
                    ypost = dndt_all[post_idx]
                    tpost = t[post_idx]
                    # We want crossing DOWN: find first run where y <= thr
                    mask_down = (ypost <= thr) & np.isfinite(ypost)
                    run = 0
                    for i, ok in enumerate(mask_down):
                        if ok:
                            run += 1
                            if run >= persist_M:
                                offset_t = float(tpost[i - persist_M + 1])
                                break
                        else:
                            run = 0

            # If burst detection fails, fallback
            if onset_t is None or offset_t is None:
                used_fallback_window = True
            else:
                # ensure sensible ordering
                if offset_t <= onset_t:
                    used_fallback_window = True
                else:
                    mask_window = (t >= onset_t) & (t <= offset_t) & np.isfinite(t)
                    induction_time_burst = float(onset_t)
                    growth_duration = float(offset_t - onset_t)

    # If not burst mode, define induction_time_burst and growth_duration from growth window
    if fwhm_mode != "burst":
        induction_time_burst = float(t_ind) if t_ind is not None else None
        if t_ind is not None and t_sat is not None and np.isfinite(t_ind) and np.isfinite(t_sat):
            growth_duration = float(t_sat - t_ind)
        else:
            growth_duration = None

    # Always ensure window has points; if not, fallback to growth window; if still not, fallback to all finite
    if not np.any(mask_window):
        used_fallback_window = True
        mask_window = mask_growth.copy()
    if not np.any(mask_window):
        used_fallback_window = True
        mask_window = np.isfinite(t)

    # Windowed arrays
    tw = t[mask_window]
    dXdt_w = dXdt_all[mask_window]
    dndt_w = dndt_all[mask_window]
    Xmono_w = Xmono_s[mask_window]

    # --- Peak dX/dt in window ---
    peak_t, peak_dX, peak_idx_local = find_peak(tw, dXdt_w)
    peak_idx_global = None
    if peak_idx_local is not None:
        # map local index to global by using mask indices
        global_idxs = np.where(mask_window)[0]
        peak_idx_global = int(global_idxs[int(peak_idx_local)])

    # --- Baseline for dX/dt (use pre-peak in the full trace, robust) ---
    if peak_t is None:
        baseline_dX = float(np.nanmedian(dXdt_all[np.isfinite(dXdt_all)])) if np.any(np.isfinite(dXdt_all)) else 0.0
    else:
        baseline_dX = median_baseline(dXdt_all, t, peak_t, frac_pre=0.25)

    # FWHM in window (use windowed curve but baseline from overall)
    fwhm_left = fwhm_right = fwhm = None
    if peak_t is not None:
        # compute on windowed curve for crossings
        _, _, peak_idx_w = find_peak(tw, dXdt_w)
        if peak_idx_w is not None:
            fwhm_left, fwhm_right, fwhm = fwhm_from_curve(tw, dXdt_w, peak_idx_w, baseline_dX)

    # Equivalent width
    w_eq = None
    if peak_t is not None:
        w_eq = equiv_width_ms(tw, dXdt_w, baseline_dX)

    # sharpness_X and synchrony_X (only if fwhm & w_eq exist)
    sharpness_X = None
    if (fwhm is not None) and np.isfinite(fwhm) and fwhm > 0 and (w_eq is not None) and np.isfinite(w_eq) and w_eq > 0:
        # One reasonable sharpness definition: fwhm / w_eq or inverse; choose inverse so "sharper" -> larger
        sharpness_X = float(w_eq / fwhm)

    synchrony_X = None
    if fwhm is not None and np.isfinite(fwhm) and fwhm > 0 and growth_duration is not None and np.isfinite(growth_duration) and growth_duration > 0:
        # Synchrony as growth_duration / fwhm (bigger means more synchronized burst vs total growth time)
        synchrony_X = float(growth_duration / fwhm)

    # --- Peak dn/dt in window ---
    peak_t_nw, peak_dn_w, _ = find_peak(tw, dndt_w)
    # Baseline for dn/dt
    baseline_dn = median_baseline(dndt_all, t, peak_t_nw if peak_t_nw is not None else np.nan, frac_pre=0.25)

    # "sharpness_n": analogous to X if we can define a width
    fwhm_left_n = fwhm_right_n = fwhm_n = None
    if peak_t_nw is not None:
        _, _, pidx_nw = find_peak(tw, dndt_w)
        if pidx_nw is not None:
            fwhm_left_n, fwhm_right_n, fwhm_n = fwhm_from_curve(tw, dndt_w, pidx_nw, baseline_dn)

    w_eq_n = None
    if peak_t_nw is not None:
        w_eq_n = equiv_width_ms(tw, dndt_w, baseline_dn)

    sharpness_n = None
    if (fwhm_n is not None) and np.isfinite(fwhm_n) and fwhm_n > 0 and (w_eq_n is not None) and np.isfinite(w_eq_n) and w_eq_n > 0:
        sharpness_n = float(w_eq_n / fwhm_n)

    synchrony_n = None
    if fwhm_n is not None and np.isfinite(fwhm_n) and fwhm_n > 0 and growth_duration is not None and np.isfinite(growth_duration) and growth_duration > 0:
        synchrony_n = float(growth_duration / fwhm_n)

    # ratio
    ratio = None
    if peak_dn_w is not None and peak_dX is not None and np.isfinite(peak_dn_w) and np.isfinite(peak_dX) and peak_dX != 0:
        ratio = float(peak_dn_w / peak_dX)

    # -----------------------------
    # QC plots (always plot full traces; overlay window if available)
    # -----------------------------
    prefix = out_prefix if out_prefix else "metrics"
    # Make safe filename stem
    ds_safe = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in str(ds_name)])

    # X plot
    fig = plt.figure(figsize=(10, 6))
    plt.plot(t, X, alpha=0.4, label="X raw (normalized)")
    plt.plot(t, Xmono_s, label="X mono (running max)" if use_running_max else "X (smoothed)")
    if fwhm_mode == "burst" and np.any(mask_window):
        plt.axvspan(float(np.nanmin(tw)), float(np.nanmax(tw)), alpha=0.1, label="burst window")
    if t_ind is not None and np.isfinite(t_ind):
        plt.axvline(float(t_ind), linestyle="--", label=f"t_ind={float(t_ind):g} ms")
    if t_sat is not None and np.isfinite(t_sat):
        plt.axvline(float(t_sat), linestyle="--", label=f"t_sat={float(t_sat):g} ms")
    plt.title(f"{ds_name}: X_mono(t)")
    plt.xlabel("time (ms)")
    plt.ylabel("X (normalized)")
    plt.legend(loc="best")
    plt.tight_layout()
    fig.savefig(f"{prefix}_{ds_safe}_X.png", dpi=150)
    plt.close(fig)

    # dX/dt plot
    fig = plt.figure(figsize=(10, 6))
    plt.plot(t, dXdt_all, alpha=0.35, label="dX/dt (per ms) (all)")
    # window plot
    if np.any(mask_window):
        plt.plot(tw, dXdt_w, label="dX/dt (per ms) (window, smoothed)")
    plt.axhline(baseline_dX, linestyle="--", label="baseline")
    if peak_t is not None:
        plt.axvline(float(peak_t), linestyle="--", label=f"peak_t={float(peak_t):g} ms")
    if fwhm_left is not None and fwhm_right is not None:
        plt.axvline(float(fwhm_left), linestyle=":", label="FWHM left")
        plt.axvline(float(fwhm_right), linestyle=":", label="FWHM right")
    ttl = f"{ds_name}: dX/dt | peak={_nan_safe(peak_dX):g}  FWHM={('NA' if fwhm is None else f'{fwhm:g}')} ms"
    plt.title(ttl)
    plt.xlabel("time (ms)")
    plt.ylabel("dX/dt (per ms)")
    plt.legend(loc="best")
    plt.tight_layout()
    fig.savefig(f"{prefix}_{ds_safe}_dXdt.png", dpi=150)
    plt.close(fig)

    # dn/dt plot
    fig = plt.figure(figsize=(10, 6))
    plt.plot(t, dndt_all, alpha=0.35, label="dn/dt (per ms) (all)")
    if np.any(mask_window):
        plt.plot(tw, dndt_w, label="dn/dt (per ms) (window, smoothed)")
    plt.axhline(baseline_dn, linestyle="--", label="baseline")
    if peak_t_nw is not None:
        plt.axvline(float(peak_t_nw), linestyle="--", label=f"peak_t={float(peak_t_nw):g} ms")
    if fwhm_left_n is not None and fwhm_right_n is not None:
        plt.axvline(float(fwhm_left_n), linestyle=":", label="FWHM left")
        plt.axvline(float(fwhm_right_n), linestyle=":", label="FWHM right")
    ttl = f"{ds_name}: dn/dt | peak={_nan_safe(peak_dn_w):g}"
    plt.title(ttl)
    plt.xlabel("time (ms)")
    plt.ylabel("dn/dt (per ms)")
    plt.legend(loc="best")
    plt.tight_layout()
    fig.savefig(f"{prefix}_{ds_safe}_dndt.png", dpi=150)
    plt.close(fig)

    # Build Metrics object
    m = Metrics(
        dataset=str(ds_name),

        n_points=n_points,
        normalize_method=norm_method,
        normalize_denom=float(denom),

        X_ind=float(X_ind),
        X_sat=float(X_sat),
        persist_M=int(persist_M),
        smooth_win=int(smooth_win),
        fwhm_mode=str(fwhm_mode),
        frac_onset=float(frac_onset),

        t_ind_ms=float(t_ind) if t_ind is not None and np.isfinite(t_ind) else None,
        t_sat_ms=float(t_sat) if t_sat is not None and np.isfinite(t_sat) else None,
        induction_time_X05_ms=float(induction_time_X05) if induction_time_X05 is not None and np.isfinite(induction_time_X05) else None,

        induction_time_burst_ms=float(induction_time_burst) if induction_time_burst is not None and np.isfinite(induction_time_burst) else None,
        growth_duration_ms=float(growth_duration) if growth_duration is not None and np.isfinite(growth_duration) else None,

        peak_t_ms=float(peak_t) if peak_t is not None and np.isfinite(peak_t) else None,
        peak_dXdt_per_ms=float(peak_dX) if peak_dX is not None and np.isfinite(peak_dX) else None,

        fwhm_left_ms=float(fwhm_left) if fwhm_left is not None and np.isfinite(fwhm_left) else None,
        fwhm_right_ms=float(fwhm_right) if fwhm_right is not None and np.isfinite(fwhm_right) else None,
        fwhm_ms=float(fwhm) if fwhm is not None and np.isfinite(fwhm) else None,

        w_eq_ms=float(w_eq) if w_eq is not None and np.isfinite(w_eq) else None,
        sharpness_X=float(sharpness_X) if sharpness_X is not None and np.isfinite(sharpness_X) else None,
        synchrony_X=float(synchrony_X) if synchrony_X is not None and np.isfinite(synchrony_X) else None,

        peak_dndt_per_ms=float(peak_dn_w) if peak_dn_w is not None and np.isfinite(peak_dn_w) else None,
        sharpness_n=float(sharpness_n) if sharpness_n is not None and np.isfinite(sharpness_n) else None,
        synchrony_n=float(synchrony_n) if synchrony_n is not None and np.isfinite(synchrony_n) else None,

        peak_ratio_dndt_over_dXdt=float(ratio) if ratio is not None and np.isfinite(ratio) else None,

        used_fallback_t_ind=bool(used_fallback_t_ind),
        used_fallback_t_sat=bool(used_fallback_t_sat),
        used_fallback_window=bool(used_fallback_window),
    )
    return m


# -----------------------------
# Column detection
# -----------------------------

def autodetect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Returns (dataset_col, time_col, X_col, n_col) if possible.
    We still allow user overrides.
    """
    cols = list(df.columns)

    # dataset
    dataset_col = "dataset" if "dataset" in cols else None

    # time
    time_candidates = ["t_shifted_ms", "time_ms", "t_ms", "time", "t"]
    time_col = next((c for c in time_candidates if c in cols), None)

    # X
    X_candidates = ["X", "x", "max_bbox_frac_kept", "sum_bbox_area_frac_kept"]
    X_col = next((c for c in X_candidates if c in cols), None)

    # n
    n_candidates = ["n_kept", "n", "count", "n_grains", "n_active"]
    n_col = next((c for c in n_candidates if c in cols), None)

    return dataset_col, time_col, X_col, n_col


# -----------------------------
# CLI / main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Input CSV path")

    p.add_argument("--out_prefix", default="metrics_burst_metrics", help="Output prefix for CSV + plots")
    p.add_argument("--dataset_col", default=None, help="Dataset column (default: auto or 'dataset')")
    p.add_argument("--time_col", default=None, help="Time column (e.g. t_shifted_ms)")
    p.add_argument("--X_col", default=None, help="X column (e.g. max_bbox_frac_kept)")
    p.add_argument("--n_col", default=None, help="n column (e.g. n_kept)")

    p.add_argument("--smooth_win", type=int, default=7, help="Smoothing window (moving average)")
    p.add_argument("--X_ind", type=float, default=0.05, help="Induction threshold in normalized X")
    p.add_argument("--X_sat", type=float, default=0.9, help="Saturation threshold in normalized X")
    p.add_argument("--persist_M", type=int, default=3, help="Required consecutive points for threshold crossing")

    p.add_argument("--normalize_by", default="p99", choices=["none", "max", "p99"], help="How to normalize X")
    p.add_argument("--no_running_max", action="store_true", help="Disable running max monotonicization of X")

    p.add_argument("--t0_floor_ms", type=float, default=None, help="Drop points earlier than this time (ms)")

    p.add_argument("--fwhm_mode", default="growth", choices=["growth", "burst"],
                   help="growth: use t_ind..t_sat; burst: use dn/dt burst window")
    p.add_argument("--frac_onset", type=float, default=0.05,
                   help="Onset fraction (baseline + frac*(peak-baseline)) for burst window via dn/dt")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)

    ds_auto, t_auto, X_auto, n_auto = autodetect_columns(df)

    dataset_col = args.dataset_col if args.dataset_col else (ds_auto if ds_auto else "dataset")
    time_col = args.time_col if args.time_col else t_auto
    X_col = args.X_col if args.X_col else X_auto
    n_col = args.n_col if args.n_col else n_auto

    if time_col is None or X_col is None or n_col is None:
        raise ValueError(
            "Could not resolve required columns.\n"
            f"Found columns: {list(df.columns)}\n"
            "Need: time_col (e.g. t_shifted_ms), X_col (e.g. max_bbox_frac_kept), n_col (e.g. n_kept)\n"
            "Fix by providing: --time_col ... --X_col ... --n_col ..."
        )

    if dataset_col not in df.columns:
        # If no dataset column, treat everything as one dataset
        df = df.copy()
        dataset_col = "dataset"
        df[dataset_col] = "dataset0"

    metrics: List[Metrics] = []

    for ds_name, g in df.groupby(dataset_col, sort=False):
        m = compute_metrics_for_dataset(
            ds_name=str(ds_name),
            df=g,
            time_col=time_col,
            X_col=X_col,
            n_col=n_col,
            smooth_win=args.smooth_win,
            X_ind=args.X_ind,
            X_sat=args.X_sat,
            persist_M=args.persist_M,
            normalize_by=args.normalize_by,
            use_running_max=(not args.no_running_max),
            t0_floor_ms=args.t0_floor_ms,
            fwhm_mode=args.fwhm_mode,
            frac_onset=args.frac_onset,
            out_prefix=args.out_prefix
        )
        metrics.append(m)

        # Console print in the style you’ve been using
        print(f"\n=== {m.dataset} ===")
        for k, v in asdict(m).items():
            if k == "dataset":
                continue
            if isinstance(v, bool):
                print(f"{k:>24s}: {v}")
            else:
                if v is None or (isinstance(v, float) and not np.isfinite(v)):
                    vv = "NA"
                else:
                    vv = v
                print(f"{k:>24s}: {vv}")

    out_csv = f"{args.out_prefix}.csv"
    out_df = pd.DataFrame([asdict(m) for m in metrics])
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print(f"Saved QC plots: {args.out_prefix}_<dataset>_X.png, {args.out_prefix}_<dataset>_dXdt.png, {args.out_prefix}_<dataset>_dndt.png\n")

    # Windows note (your loop culprit earlier)
    print("Windows CMD tip:")
    print("  - Don't use '\\' for line continuation. Use one line, or use '^' in CMD, or backtick ` in PowerShell.")
    print("Example one-liner:")
    print(
        f"  python qc_extract_burst_metrics.py --csv path\\to\\counts_shifted.csv "
        f"--time_col {time_col} --X_col {X_col} --n_col {n_col} --fwhm_mode {args.fwhm_mode} --frac_onset {args.frac_onset}"
    )


if __name__ == "__main__":
    main()