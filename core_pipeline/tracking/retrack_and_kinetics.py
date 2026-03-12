#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
retrack_and_kinetics.py

Works with tracks.csv like the one you uploaded:
Columns include:
- frame_idx, t_ms, cx, cy, area_px, R_px, bbox_x, bbox_y, bbox_w, bbox_h, track_id, det_id...

What it does:
1) Filters detections (optional) by min area, border margin, max area fraction.
2) Re-links detections into tracks using distance + optional radius-change gating.
3) Produces kinetics outputs: nucleation histogram, active tracks vs time.
4) Optional per-track tau fit of saturating exponential on radius vs time.

Outputs (in out_dir):
- retracked_tracks.csv
- track_summary.csv
- nucleation_bins.csv
- active_tracks.csv
- tau_fits.csv (if fits possible)
- nucleation_rate.png
- active_tracks.png
- tau_hist.png (if tau fits exist)
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional Hungarian assignment via SciPy
try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# -----------------------------
# Helpers
# -----------------------------
def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_float_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").astype(float)


def to_int_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)


def radius_from_area(area_px: np.ndarray) -> np.ndarray:
    # Equivalent disk radius: area = pi*r^2
    return np.sqrt(np.clip(area_px, 0, None) / math.pi)


def infer_image_size_from_bbox(df: pd.DataFrame) -> Tuple[Optional[int], Optional[int]]:
    """
    If bbox_x/bbox_w/bbox_y/bbox_h exist, infer image width/height
    as max(bbox_x + bbox_w) and max(bbox_y + bbox_h).
    """
    needed = {"bbox_x", "bbox_y", "bbox_w", "bbox_h"}
    if not needed.issubset(set(df.columns)):
        return None, None

    bx = pd.to_numeric(df["bbox_x"], errors="coerce").astype(float)
    by = pd.to_numeric(df["bbox_y"], errors="coerce").astype(float)
    bw = pd.to_numeric(df["bbox_w"], errors="coerce").astype(float)
    bh = pd.to_numeric(df["bbox_h"], errors="coerce").astype(float)

    w = np.nanmax((bx + bw).values)
    h = np.nanmax((by + bh).values)
    if not np.isfinite(w) or not np.isfinite(h):
        return None, None
    return int(math.ceil(w)), int(math.ceil(h))


def infer_dt_ms_from_t_ms(df: pd.DataFrame, frame_col: str, t_col: str) -> Optional[float]:
    """
    Infer dt_ms from t_ms increments (robustly).
    Returns None if insufficient.
    """
    if t_col not in df.columns:
        return None
    tmp = df[[frame_col, t_col]].dropna().copy()
    if tmp.empty:
        return None
    tmp[frame_col] = to_int_series(tmp, frame_col)
    tmp[t_col] = to_float_series(tmp, t_col)
    tmp = tmp.drop_duplicates(subset=[frame_col]).sort_values(frame_col)
    if len(tmp) < 2:
        return None
    diffs = np.diff(tmp[t_col].values.astype(float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return None
    return float(np.median(diffs))


# -----------------------------
# Tracking
# -----------------------------
@dataclass
class TrackState:
    track_id: int
    last_frame: int
    x: float
    y: float
    r: Optional[float]  # radius px
    gap: int            # frames since last seen


def compute_cost(
    trk: TrackState,
    det_x: float,
    det_y: float,
    det_r: Optional[float],
    w_dist: float,
    w_dR: float,
) -> float:
    dist = math.hypot(det_x - trk.x, det_y - trk.y)
    if trk.r is None or det_r is None:
        return w_dist * dist
    return w_dist * dist + w_dR * abs(det_r - trk.r)


def build_assignment(
    tracks: List[TrackState],
    det_xy: np.ndarray,
    det_r: Optional[np.ndarray],
    max_dist_px: float,
    max_dR_px: float,
    w_dist: float,
    w_dR: float,
) -> List[Tuple[int, int, float]]:
    """
    Returns list of (track_index, det_index, cost).
    Hungarian if SciPy available, else greedy.
    """
    if len(tracks) == 0 or det_xy.shape[0] == 0:
        return []

    nT, nD = len(tracks), det_xy.shape[0]
    BIG = 1e9
    cost = np.full((nT, nD), BIG, dtype=float)

    for i, trk in enumerate(tracks):
        for j in range(nD):
            dx, dy = float(det_xy[j, 0]), float(det_xy[j, 1])
            rr = None if det_r is None else float(det_r[j])

            dist = math.hypot(dx - trk.x, dy - trk.y)
            if dist > max_dist_px:
                continue
            if (trk.r is not None) and (rr is not None) and (abs(rr - trk.r) > max_dR_px):
                continue

            cost[i, j] = compute_cost(trk, dx, dy, rr, w_dist=w_dist, w_dR=w_dR)

    matches: List[Tuple[int, int, float]] = []

    if HAS_SCIPY:
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            if cost[r, c] >= BIG / 2:
                continue
            matches.append((int(r), int(c), float(cost[r, c])))
        return matches

    # Greedy fallback
    flat = []
    for i in range(nT):
        for j in range(nD):
            if cost[i, j] < BIG / 2:
                flat.append((cost[i, j], i, j))
    flat.sort(key=lambda x: x[0])

    usedT, usedD = set(), set()
    for c, i, j in flat:
        if i in usedT or j in usedD:
            continue
        usedT.add(i)
        usedD.add(j)
        matches.append((i, j, float(c)))

    return matches


def retrack(
    df: pd.DataFrame,
    frame_col: str,
    x_col: str,
    y_col: str,
    area_col: Optional[str],
    r_col: Optional[str],
    min_area_px: float,
    max_area_frac: float,
    border_margin_px: int,
    img_w_px: Optional[int],
    img_h_px: Optional[int],
    max_dist_px: float,
    max_dR_px: float,
    max_gap_frames: int,
    w_dist: float,
    w_dR: float,
    min_track_frames: int,
) -> pd.DataFrame:
    out = df.copy()

    # Enforce types
    out[frame_col] = to_int_series(out, frame_col)
    out[x_col] = to_float_series(out, x_col)
    out[y_col] = to_float_series(out, y_col)

    # Radius source:
    # Prefer r_col if present; else compute from area if present; else disable radius gating.
    r_px_all: Optional[np.ndarray] = None
    if r_col is not None and r_col in out.columns:
        r_px_all = pd.to_numeric(out[r_col], errors="coerce").astype(float).values
    elif area_col is not None and area_col in out.columns:
        area = pd.to_numeric(out[area_col], errors="coerce").astype(float).values
        r_px_all = radius_from_area(area)

    # Filtering mask
    mask = np.ones(len(out), dtype=bool)

    # Min area filter
    if area_col is not None and area_col in out.columns and min_area_px > 0:
        out[area_col] = to_float_series(out, area_col)
        mask &= (out[area_col].values >= float(min_area_px))

    # Max area fraction (requires image size + area)
    if (
        max_area_frac is not None
        and max_area_frac > 0
        and img_w_px is not None
        and img_h_px is not None
        and area_col is not None
        and area_col in out.columns
    ):
        img_area = float(img_w_px) * float(img_h_px)
        frac = out[area_col].astype(float).values / img_area
        mask &= np.isfinite(frac) & (frac <= float(max_area_frac))

    # Border filter (requires image size)
    if border_margin_px > 0 and img_w_px is not None and img_h_px is not None:
        x = out[x_col].values
        y = out[y_col].values
        bm = int(border_margin_px)
        mask &= (x >= bm) & (y >= bm) & (x <= (img_w_px - 1 - bm)) & (y <= (img_h_px - 1 - bm))

    out = out.loc[mask].reset_index(drop=True)
    if r_px_all is not None:
        r_px = r_px_all[mask]
    else:
        r_px = None

    # Sort by frame
    out = out.sort_values(frame_col).reset_index(drop=True)

    frames = out[frame_col].values.astype(int)
    xy = out[[x_col, y_col]].values.astype(float)

    new_track_ids = np.full(len(out), -1, dtype=int)

    active: List[TrackState] = []
    next_tid = 0

    unique_frames = np.unique(frames)
    rows_by_frame: Dict[int, np.ndarray] = {int(fr): np.where(frames == fr)[0] for fr in unique_frames}

    for fr in unique_frames:
        idxs = rows_by_frame[int(fr)]
        det_xy = xy[idxs]
        det_r = None if r_px is None else np.asarray(r_px)[idxs]

        # update gaps & prune
        for t in active:
            if t.last_frame < int(fr):
                t.gap = int(fr) - t.last_frame
        active = [t for t in active if t.gap <= max_gap_frames]

        matches = build_assignment(
            tracks=active,
            det_xy=det_xy,
            det_r=det_r,
            max_dist_px=float(max_dist_px),
            max_dR_px=float(max_dR_px),
            w_dist=float(w_dist),
            w_dR=float(w_dR),
        )

        matched_dets = set()

        # apply matches
        for ti, dj, _c in matches:
            trk = active[ti]
            matched_dets.add(dj)
            row = int(idxs[dj])

            new_track_ids[row] = trk.track_id
            trk.x = float(det_xy[dj, 0])
            trk.y = float(det_xy[dj, 1])
            trk.r = None if det_r is None else float(det_r[dj])
            trk.last_frame = int(fr)
            trk.gap = 0

        # start new tracks for unmatched detections
        for dj in range(det_xy.shape[0]):
            if dj in matched_dets:
                continue
            row = int(idxs[dj])
            new_track_ids[row] = next_tid
            active.append(
                TrackState(
                    track_id=next_tid,
                    last_frame=int(fr),
                    x=float(det_xy[dj, 0]),
                    y=float(det_xy[dj, 1]),
                    r=None if det_r is None else float(det_r[dj]),
                    gap=0,
                )
            )
            next_tid += 1

    out["track_id"] = new_track_ids

    # prune short tracks
    vc = out["track_id"].value_counts()
    keep = set(vc[vc >= int(min_track_frames)].index.tolist())
    out = out[out["track_id"].isin(keep)].copy()

    # remap to 0..N-1
    ids = sorted(out["track_id"].unique().tolist())
    remap = {old: new for new, old in enumerate(ids)}
    out["track_id"] = out["track_id"].map(remap).astype(int)

    return out.reset_index(drop=True)


# -----------------------------
# Kinetics
# -----------------------------
def nucleation_bins(track_summary: pd.DataFrame, dt_ms: float, nuc_bin_ms: float) -> pd.DataFrame:
    births_ms = track_summary["birth_frame"].values.astype(int) * float(dt_ms)
    nuc_bin_ms = float(nuc_bin_ms) if nuc_bin_ms and nuc_bin_ms > 0 else float(dt_ms)

    if len(births_ms) == 0:
        return pd.DataFrame(
            columns=["bin_start_ms", "bin_end_ms", "bin_center_ms", "nucleations", "nucleation_rate_per_ms", "cumulative_nucleations"]
        )

    max_t = float(np.nanmax(births_ms))
    edges = np.arange(0, max_t + nuc_bin_ms + 1e-9, nuc_bin_ms)
    if len(edges) < 2:
        edges = np.array([0.0, nuc_bin_ms], dtype=float)

    counts, _ = np.histogram(births_ms, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = pd.DataFrame({
        "bin_start_ms": edges[:-1],
        "bin_end_ms": edges[1:],
        "bin_center_ms": centers,
        "nucleations": counts.astype(int),
        "nucleation_rate_per_ms": counts.astype(float) / nuc_bin_ms,
    })
    out["cumulative_nucleations"] = np.cumsum(out["nucleations"].values)
    return out


def active_tracks_timeseries(df_tr: pd.DataFrame, frame_col: str, dt_ms: float) -> pd.DataFrame:
    grp = df_tr.groupby(frame_col)["track_id"].nunique().reset_index()
    grp.rename(columns={"track_id": "active_tracks"}, inplace=True)
    grp["time_ms"] = grp[frame_col].astype(int) * float(dt_ms)
    return grp[[frame_col, "time_ms", "active_tracks"]]


def fit_tau_saturating_exp(
    t_ms: np.ndarray,
    r: np.ndarray,
    tau_max_ms: float,
    min_points: int,
) -> Optional[Dict[str, float]]:
    """
    Fit r(t) ≈ r0 + A*(1 - exp(-t/tau)) via tau grid search + linear LS for r0/A.
    """
    if len(t_ms) < min_points:
        return None

    t = t_ms.astype(float)
    y = r.astype(float)

    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if len(t) < min_points:
        return None

    # shift to birth time = 0
    t0 = float(np.min(t))
    t = t - t0

    if tau_max_ms and tau_max_ms > 0:
        mm = t <= float(tau_max_ms)
        t, y = t[mm], y[mm]
        if len(t) < min_points:
            return None

    if np.nanstd(y) < 1e-9:
        return {"r0": float(np.nanmean(y)), "A": 0.0, "tau_ms": 0.0, "rmse": 0.0}

    # tau grid
    tspan = max(float(np.max(t)), 1e-6)
    uniq = np.unique(t)
    dt_guess = float(np.median(np.diff(uniq))) if len(uniq) > 1 else tspan / 10.0
    tau_min = max(1e-3, 0.1 * dt_guess)
    tau_max = max(tau_min * 10.0, 10.0 * tspan)
    taus = np.logspace(np.log10(tau_min), np.log10(tau_max), 200)

    best = None
    best_rmse = float("inf")

    for tau in taus:
        g = 1.0 - np.exp(-t / tau)
        X = np.vstack([np.ones_like(g), g]).T
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r0_hat, A_hat = float(beta[0]), float(beta[1])
        yhat = r0_hat + A_hat * g
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best = {"r0": r0_hat, "A": A_hat, "tau_ms": float(tau), "rmse": rmse}

    return best


def per_track_tau_fits(
    df_tr: pd.DataFrame,
    frame_col: str,
    dt_ms: float,
    r_px_col: str,
    fit_tau_max_ms: float,
    fit_min_points: int,
) -> pd.DataFrame:
    rows = []
    for tid, g in df_tr.groupby("track_id"):
        g = g.sort_values(frame_col)
        t_ms = g[frame_col].values.astype(int) * float(dt_ms)
        r = g[r_px_col].values.astype(float)
        res = fit_tau_saturating_exp(t_ms, r, tau_max_ms=float(fit_tau_max_ms), min_points=int(fit_min_points))
        if res is None:
            continue
        rows.append({
            "track_id": int(tid),
            "tau_ms": res["tau_ms"],
            "r0_px": res["r0"],
            "A_px": res["A"],
            "rmse_px": res["rmse"],
        })
    return pd.DataFrame(rows)


# -----------------------------
# Plotting
# -----------------------------
def save_plot_nucleation(nuc_df: pd.DataFrame, out_path: str) -> None:
    plt.figure()
    if len(nuc_df) > 0:
        plt.plot(nuc_df["bin_center_ms"].values, nuc_df["nucleations"].values)
    plt.xlabel("Time (ms)")
    plt.ylabel("Nucleations per bin")
    plt.title("Nucleation events vs time")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_plot_active(active_df: pd.DataFrame, out_path: str) -> None:
    plt.figure()
    if len(active_df) > 0:
        plt.plot(active_df["time_ms"].values, active_df["active_tracks"].values)
    plt.xlabel("Time (ms)")
    plt.ylabel("Active tracks")
    plt.title("Active tracks vs time")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_plot_tau_hist(tau_df: pd.DataFrame, out_path: str) -> None:
    if tau_df is None or len(tau_df) == 0:
        return
    plt.figure()
    plt.hist(tau_df["tau_ms"].values, bins=30)
    plt.xlabel("Tau (ms)")
    plt.ylabel("Count")
    plt.title("Per-track tau distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument("--tracks_csv", required=True, help="Input tracks/detections CSV")
    p.add_argument("--out_dir", required=True, help="Output directory")

    # You can pass dt_ms, OR set dt_ms=0 to infer from t_ms column (if present).
    p.add_argument("--dt_ms", required=True, type=float, help="Frame time step in ms (set 0 to infer from t_ms)")
    p.add_argument("--px_um", required=True, type=float, help="Pixel size in um/px (must be a numeric float!)")

    # Filtering
    p.add_argument("--min_area_px", type=float, default=0.0)
    p.add_argument("--max_area_frac", type=float, default=0.0, help="Requires image size; 0 disables.")
    p.add_argument("--border_margin_px", type=int, default=0)

    # If you know image size, you can set it; else we'll infer from bbox_* if present.
    p.add_argument("--img_w_px", type=int, default=0)
    p.add_argument("--img_h_px", type=int, default=0)

    # Linking
    p.add_argument("--max_dist_px", type=float, default=10.0)
    p.add_argument("--max_dR_px", type=float, default=5.0)
    p.add_argument("--max_gap_frames", type=int, default=0)
    p.add_argument("--w_dist", type=float, default=1.0)
    p.add_argument("--w_dR", type=float, default=1.0)
    p.add_argument("--min_track_frames", type=int, default=3)

    # Kinetics
    p.add_argument("--nuc_bin_ms", type=float, default=20.0)
    p.add_argument("--fit_tau_max_ms", type=float, default=0.0)
    p.add_argument("--fit_min_points", type=int, default=6)

    args = p.parse_args()
    safe_mkdir(args.out_dir)

    df = pd.read_csv(args.tracks_csv)

    # Your file columns (from upload): frame_idx, cx, cy, area_px, R_px, t_ms...
    frame_col = "frame_idx" if "frame_idx" in df.columns else "frame"
    x_col = "cx" if "cx" in df.columns else "x"
    y_col = "cy" if "cy" in df.columns else "y"
    area_col = "area_px" if "area_px" in df.columns else (None)
    # Prefer your provided radius column if available:
    r_col = "R_px" if "R_px" in df.columns else (None)

    # Preserve original track_id if present
    if "track_id" in df.columns:
        df = df.rename(columns={"track_id": "track_id_input"})

    # dt_ms: allow inference if user passes 0
    dt_ms = float(args.dt_ms)
    if dt_ms <= 0:
        dt_infer = infer_dt_ms_from_t_ms(df, frame_col=frame_col, t_col="t_ms")
        if dt_infer is None:
            raise ValueError("dt_ms was set to 0, but could not infer dt from t_ms column.")
        dt_ms = dt_infer

    # Image size
    img_w_px = int(args.img_w_px) if int(args.img_w_px) > 0 else None
    img_h_px = int(args.img_h_px) if int(args.img_h_px) > 0 else None
    if img_w_px is None or img_h_px is None:
        w_inf, h_inf = infer_image_size_from_bbox(df)
        img_w_px = img_w_px if img_w_px is not None else w_inf
        img_h_px = img_h_px if img_h_px is not None else h_inf

    df_tr = retrack(
        df=df,
        frame_col=frame_col,
        x_col=x_col,
        y_col=y_col,
        area_col=area_col,
        r_col=r_col,
        min_area_px=float(args.min_area_px),
        max_area_frac=float(args.max_area_frac),
        border_margin_px=int(args.border_margin_px),
        img_w_px=img_w_px,
        img_h_px=img_h_px,
        max_dist_px=float(args.max_dist_px),
        max_dR_px=float(args.max_dR_px),
        max_gap_frames=int(args.max_gap_frames),
        w_dist=float(args.w_dist),
        w_dR=float(args.w_dR),
        min_track_frames=int(args.min_track_frames),
    )

    # Add derived columns
    # Ensure r_px exists
    if "r_px" not in df_tr.columns:
        if r_col is not None and r_col in df_tr.columns:
            df_tr["r_px"] = pd.to_numeric(df_tr[r_col], errors="coerce").astype(float)
        elif area_col is not None and area_col in df_tr.columns:
            df_tr["r_px"] = radius_from_area(pd.to_numeric(df_tr[area_col], errors="coerce").astype(float).values)
        else:
            df_tr["r_px"] = np.nan

    px_um = float(args.px_um)
    df_tr["time_ms"] = df_tr[frame_col].astype(int) * dt_ms
    df_tr["x_um"] = df_tr[x_col].astype(float) * px_um
    df_tr["y_um"] = df_tr[y_col].astype(float) * px_um
    df_tr["r_um"] = df_tr["r_px"].astype(float) * px_um

    # Track summary
    g = df_tr.groupby("track_id")
    summary = g.agg(
        birth_frame=(frame_col, "min"),
        last_frame=(frame_col, "max"),
        n_frames=(frame_col, "nunique"),
        n_detections=(frame_col, "size"),
        x0_px=(x_col, "first"),
        y0_px=(y_col, "first"),
        x_end_px=(x_col, "last"),
        y_end_px=(y_col, "last"),
        r0_px=("r_px", "first"),
        r_end_px=("r_px", "last"),
    ).reset_index()

    summary["birth_time_ms"] = summary["birth_frame"].astype(int) * dt_ms
    summary["duration_ms"] = (summary["last_frame"].astype(int) - summary["birth_frame"].astype(int)) * dt_ms

    # Kinetics
    nuc_df = nucleation_bins(summary, dt_ms=dt_ms, nuc_bin_ms=float(args.nuc_bin_ms))
    active_df = active_tracks_timeseries(df_tr, frame_col=frame_col, dt_ms=dt_ms)

    # Tau fits (if radius exists)
    tau_df = pd.DataFrame()
    if np.isfinite(df_tr["r_px"].values).any():
        tau_df = per_track_tau_fits(
            df_tr=df_tr,
            frame_col=frame_col,
            dt_ms=dt_ms,
            r_px_col="r_px",
            fit_tau_max_ms=float(args.fit_tau_max_ms),
            fit_min_points=int(args.fit_min_points),
        )

    # Save outputs
    out_tracks = os.path.join(args.out_dir, "retracked_tracks.csv")
    out_summary = os.path.join(args.out_dir, "track_summary.csv")
    out_nuc = os.path.join(args.out_dir, "nucleation_bins.csv")
    out_active = os.path.join(args.out_dir, "active_tracks.csv")
    out_tau = os.path.join(args.out_dir, "tau_fits.csv")

    df_tr.to_csv(out_tracks, index=False)
    summary.to_csv(out_summary, index=False)
    nuc_df.to_csv(out_nuc, index=False)
    active_df.to_csv(out_active, index=False)
    if len(tau_df) > 0:
        tau_df.to_csv(out_tau, index=False)

    # Plots
    save_plot_nucleation(nuc_df, os.path.join(args.out_dir, "nucleation_rate.png"))
    save_plot_active(active_df, os.path.join(args.out_dir, "active_tracks.png"))
    if len(tau_df) > 0:
        save_plot_tau_hist(tau_df, os.path.join(args.out_dir, "tau_hist.png"))

    # Console summary
    print("Done.")
    print(f"dt_ms used: {dt_ms}")
    if img_w_px is not None and img_h_px is not None:
        print(f"inferred/provided image size: {img_w_px} x {img_h_px} px")
    print(f"Wrote: {out_tracks}")
    print(f"Wrote: {out_summary}")
    print(f"Wrote: {out_nuc}")
    print(f"Wrote: {out_active}")
    if len(tau_df) > 0:
        print(f"Wrote: {out_tau}")
    else:
        print("Tau fits: skipped (no usable radius values or insufficient points).")


if __name__ == "__main__":
    main()
