# 03_track_spherulites.py
# Track spherulites across frames using Hungarian matching (SciPy) + IoU/centroid gating.
# Input: per-frame SAM outputs saved as COCO-style RLE JSONs in sam/<DATASET>/coco_rle
# Output: tracks.csv + per-dataset plots (Ri(t), <R(t)>, cohort R(tau), t_nuc histogram)

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter

from pycocotools import mask as mask_utils
import matplotlib.pyplot as plt


# -----------------------------
# CONFIG (edit if needed)
# -----------------------------
KINETICS_ROOT = Path(r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics")

DATASETS = {
    "FAPI": KINETICS_ROOT / "sam" / "FAPI" / "coco_rle",
    "FAPI_TEMPO": KINETICS_ROOT / "sam" / "FAPI_TEMPO" / "coco_rle",
}

# If timestamps exist in image names like frame_00001_t2.00ms.png, we parse those.
# Otherwise, we fall back to frame index * dt_ms
DEFAULT_FPS = 500.0
DT_MS_FALLBACK = 1000.0 / DEFAULT_FPS

# Filtering masks (adjust to your scale)
MIN_AREA_PX = 100
MAX_AREA_PX = None  # set e.g. 2_000_000 if you want

# Matching / tracking gates
MAX_CENTROID_JUMP_PX = 80.0     # if centroids move more than this between frames, do not match
MIN_IOU_TO_MATCH = 0.05         # require at least this overlap (0 disables)
MAX_MISSES = 3                  # track can survive this many unmatched frames before being closed

# Cost function weights (Hungarian minimizes cost)
W_DIST = 1.0
W_IOU = 40.0   # larger means "prefer overlap strongly"

# Smoothing for plots/derivatives
SMOOTH_WINDOW = 9   # odd integer; will auto-shrink per track
SMOOTH_POLY = 2

# Cohort plot: only show tau where >= this many tracks contribute
MIN_TRACKS_PER_TAU = 20

# Output folder for plots/csv
OUT_ROOT = KINETICS_ROOT / "tracking_outputs"

# -----------------------------
# Utilities
# -----------------------------
TS_RE = re.compile(r"_t(?P<tms>[0-9]+(?:\.[0-9]+)?)ms", re.IGNORECASE)

def parse_time_ms_from_name(name: str) -> Optional[float]:
    m = TS_RE.search(name)
    if not m:
        return None
    return float(m.group("tms"))

def safe_odd(n: int) -> int:
    if n < 3:
        return 3
    return n if n % 2 == 1 else n - 1

def decode_rle_to_mask(rle_obj, height: int, width: int) -> np.ndarray:
    """
    rle_obj can be:
      - dict with "counts" and "size"
      - list of polygons (not expected here)
    """
    if isinstance(rle_obj, dict):
        rle = rle_obj
    else:
        # If it’s polygons, convert using frPyObjects (rare for SAM RLE output)
        rle = mask_utils.frPyObjects(rle_obj, height, width)
        rle = mask_utils.merge(rle)
    m = mask_utils.decode(rle)  # HxW (uint8 0/1) or HxWxN
    if m.ndim == 3:
        m = m[:, :, 0]
    return (m > 0).astype(np.uint8)

def iou_masks(m1: np.ndarray, m2: np.ndarray) -> float:
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter / union) if union > 0 else 0.0

def centroid_from_mask(m: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return (np.nan, np.nan)
    return (float(xs.mean()), float(ys.mean()))

def equiv_radius_from_area(area_px: float) -> float:
    return math.sqrt(area_px / math.pi) if area_px > 0 else 0.0


@dataclass
class Det:
    frame: int
    t_ms: float
    det_id: int
    area: float
    cx: float
    cy: float
    mask: np.ndarray


@dataclass
class TrackState:
    track_id: int
    last_frame: int
    last_t_ms: float
    last_cx: float
    last_cy: float
    last_mask: np.ndarray
    misses: int = 0
    t_birth_ms: float = 0.0


def load_frame_detections(json_path: Path, frame_idx: int, fallback_t_ms: float) -> Tuple[float, List[Det]]:
    """
    Loads one COCO RLE json containing SAM annotations for a single image/frame.
    We expect it to be a list of dicts with keys including:
      - segmentation (RLE)
      - area
      - image_height / image_width OR size in segmentation
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Try time from filename
    t_ms = parse_time_ms_from_name(json_path.stem)
    if t_ms is None:
        t_ms = fallback_t_ms

    dets: List[Det] = []
    # infer H,W if present
    H = None
    W = None

    # Some pipelines store H/W in the annotation; otherwise use segmentation["size"]
    for i, ann in enumerate(data):
        seg = ann.get("segmentation", None)
        if isinstance(seg, dict) and "size" in seg:
            H = int(seg["size"][0])
            W = int(seg["size"][1])
            break

    for i, ann in enumerate(data):
        seg = ann.get("segmentation", None)
        if seg is None:
            continue

        area = float(ann.get("area", 0.0))
        if area < MIN_AREA_PX:
            continue
        if MAX_AREA_PX is not None and area > float(MAX_AREA_PX):
            continue

        # Decode mask
        if isinstance(seg, dict) and "size" in seg:
            H = int(seg["size"][0])
            W = int(seg["size"][1])
        if H is None or W is None:
            # last resort
            raise RuntimeError(f"Cannot infer mask size from {json_path}. Need seg['size'].")

        m = decode_rle_to_mask(seg, H, W)
        if m.sum() < MIN_AREA_PX:
            continue

        cx, cy = centroid_from_mask(m)
        if not np.isfinite(cx) or not np.isfinite(cy):
            continue

        dets.append(
            Det(
                frame=frame_idx,
                t_ms=t_ms,
                det_id=i,
                area=float(m.sum()),  # use decoded area for consistency
                cx=cx,
                cy=cy,
                mask=m,
            )
        )

    return t_ms, dets


def build_cost_matrix(
    tracks: List[TrackState], dets: List[Det]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      cost[T,D]
      dist[T,D]
      iou[T,D]
    """
    T = len(tracks)
    D = len(dets)
    cost = np.full((T, D), fill_value=1e9, dtype=float)
    dist = np.full((T, D), fill_value=np.inf, dtype=float)
    ious = np.zeros((T, D), dtype=float)

    for ti, tr in enumerate(tracks):
        for di, det in enumerate(dets):
            dx = det.cx - tr.last_cx
            dy = det.cy - tr.last_cy
            d = math.sqrt(dx * dx + dy * dy)
            dist[ti, di] = d

            if d > MAX_CENTROID_JUMP_PX:
                continue

            iou = iou_masks(tr.last_mask, det.mask)
            ious[ti, di] = iou
            if iou < MIN_IOU_TO_MATCH:
                continue

            # Hungarian minimizes cost
            cost[ti, di] = W_DIST * d + W_IOU * (1.0 - iou)

    return cost, dist, ious


def track_dataset(dataset_name: str, coco_rle_dir: Path) -> pd.DataFrame:
    json_files = sorted(coco_rle_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"No JSON files found in {coco_rle_dir}")

    tracks_active: List[TrackState] = []
    next_track_id = 1

    rows = []

    for fidx, jp in enumerate(json_files):
        fallback_t = fidx * DT_MS_FALLBACK
        t_ms, dets = load_frame_detections(jp, frame_idx=fidx, fallback_t_ms=fallback_t)

        # Start: if no active tracks, spawn all dets
        if len(tracks_active) == 0:
            for det in dets:
                tr = TrackState(
                    track_id=next_track_id,
                    last_frame=det.frame,
                    last_t_ms=det.t_ms,
                    last_cx=det.cx,
                    last_cy=det.cy,
                    last_mask=det.mask,
                    misses=0,
                    t_birth_ms=det.t_ms,
                )
                next_track_id += 1
                tracks_active.append(tr)

                rows.append({
                    "dataset": dataset_name,
                    "track_id": tr.track_id,
                    "frame": det.frame,
                    "t_ms": det.t_ms,
                    "t_birth_ms": tr.t_birth_ms,
                    "area_px": det.area,
                    "R_px": equiv_radius_from_area(det.area),
                    "cx": det.cx,
                    "cy": det.cy,
                    "matched": 0,
                    "iou_prev": np.nan,
                    "dist_prev": np.nan,
                })
            continue

        # Match
        if len(dets) == 0:
            # no detections: increment misses
            for tr in tracks_active:
                tr.misses += 1
            tracks_active = [tr for tr in tracks_active if tr.misses <= MAX_MISSES]
            continue

        cost, dist, ious = build_cost_matrix(tracks_active, dets)
        ti_idx, di_idx = linear_sum_assignment(cost)

        matched_tracks = set()
        matched_dets = set()

        # Apply assignments that are not "blocked" (cost huge)
        for ti, di in zip(ti_idx, di_idx):
            if cost[ti, di] >= 1e8:
                continue
            tr = tracks_active[ti]
            det = dets[di]

            matched_tracks.add(ti)
            matched_dets.add(di)

            # update track
            prev_mask = tr.last_mask
            prev_cx, prev_cy = tr.last_cx, tr.last_cy

            tr.last_frame = det.frame
            tr.last_t_ms = det.t_ms
            tr.last_cx = det.cx
            tr.last_cy = det.cy
            tr.last_mask = det.mask
            tr.misses = 0

            rows.append({
                "dataset": dataset_name,
                "track_id": tr.track_id,
                "frame": det.frame,
                "t_ms": det.t_ms,
                "t_birth_ms": tr.t_birth_ms,
                "area_px": det.area,
                "R_px": equiv_radius_from_area(det.area),
                "cx": det.cx,
                "cy": det.cy,
                "matched": 1,
                "iou_prev": iou_masks(prev_mask, det.mask),
                "dist_prev": math.sqrt((det.cx - prev_cx)**2 + (det.cy - prev_cy)**2),
            })

        # Unmatched tracks: increase misses
        for ti, tr in enumerate(tracks_active):
            if ti not in matched_tracks:
                tr.misses += 1

        # Remove dead tracks
        tracks_active = [tr for tr in tracks_active if tr.misses <= MAX_MISSES]

        # Unmatched detections: spawn new tracks
        for di, det in enumerate(dets):
            if di in matched_dets:
                continue
            tr = TrackState(
                track_id=next_track_id,
                last_frame=det.frame,
                last_t_ms=det.t_ms,
                last_cx=det.cx,
                last_cy=det.cy,
                last_mask=det.mask,
                misses=0,
                t_birth_ms=det.t_ms,
            )
            next_track_id += 1
            tracks_active.append(tr)

            rows.append({
                "dataset": dataset_name,
                "track_id": tr.track_id,
                "frame": det.frame,
                "t_ms": det.t_ms,
                "t_birth_ms": tr.t_birth_ms,
                "area_px": det.area,
                "R_px": equiv_radius_from_area(det.area),
                "cx": det.cx,
                "cy": det.cy,
                "matched": 0,
                "iou_prev": np.nan,
                "dist_prev": np.nan,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"{dataset_name}: tracking produced no rows. Check area thresholds and JSON content.")
    return df


def smooth_series(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if len(y) < 5:
        return y
    w = min(SMOOTH_WINDOW, len(y))
    w = safe_odd(w)
    if w >= len(y):
        w = safe_odd(len(y) - 1)
    if w < 3:
        return y
    return savgol_filter(y, window_length=w, polyorder=min(SMOOTH_POLY, w - 1))


def make_plots(df: pd.DataFrame, dataset_name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sort
    df = df.sort_values(["track_id", "t_ms"]).copy()
    df["tau_ms"] = df["t_ms"] - df["t_birth_ms"]

    # ---------- Sample trajectories ----------
    tracks = df["track_id"].unique()
    n_show = min(25, len(tracks))
    show_ids = tracks[:n_show]

    plt.figure(figsize=(8, 6))
    for tid in show_ids:
        dfi = df[df["track_id"] == tid]
        y = smooth_series(dfi["R_px"].values)
        plt.plot(dfi["t_ms"].values, y, alpha=0.6)
    plt.title(f"{dataset_name}: sample tracked spherulite trajectories")
    plt.xlabel("Time (ms)")
    plt.ylabel("R_smooth (px)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{dataset_name}_tracks_Ri.png", dpi=200)
    plt.close()

    # ---------- Mean <R(t)> ----------
    g = df.groupby("t_ms")["R_px"].mean().reset_index()
    plt.figure(figsize=(8, 5))
    plt.plot(g["t_ms"].values, smooth_series(g["R_px"].values))
    plt.title(f"{dataset_name}: population mean <R(t)> (can drop with new births)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Mean R(t) (px)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{dataset_name}_Rmean.png", dpi=200)
    plt.close()

    # ---------- Nucleation times histogram ----------
    births = df.groupby("track_id")["t_birth_ms"].min().values
    plt.figure(figsize=(7, 5))
    plt.hist(births, bins=25)
    plt.title(f"{dataset_name}: nucleation times (track births)")
    plt.xlabel("t_nuc (ms)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / f"{dataset_name}_t_nuc_hist.png", dpi=200)
    plt.close()

    # ---------- Cohort growth: median R(tau) ----------
    # Build cohort table by tau (rounded to nearest dt for stability)
    # If your timestamps are exact, this is fine; if not, rounding helps.
    tau = df["tau_ms"].values
    # Round to 0.5 ms to reduce fragmentation
    df["tau_bin"] = np.round(df["tau_ms"].values * 2.0) / 2.0

    cohort = df.groupby("tau_bin")["R_px"].agg(["median", "count"]).reset_index()
    cohort = cohort[cohort["count"] >= MIN_TRACKS_PER_TAU]

    plt.figure(figsize=(8, 5))
    plt.plot(cohort["tau_bin"].values, smooth_series(cohort["median"].values))
    plt.title(f"{dataset_name}: cohort growth (aligned by nucleation)")
    plt.xlabel("τ = t - t_nuc (ms)")
    plt.ylabel("Median R(τ) (px)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{dataset_name}_cohort_Rtau.png", dpi=200)
    plt.close()


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_dfs = []

    for name, coco_dir in DATASETS.items():
        print(f"\n=== Tracking dataset: {name} ===")
        print(f"COCO RLE folder: {coco_dir}")
        df = track_dataset(name, coco_dir)

        ds_out = OUT_ROOT / name
        ds_out.mkdir(parents=True, exist_ok=True)

        tracks_csv = ds_out / "tracks.csv"
        df.to_csv(tracks_csv, index=False)
        print(f"✔ wrote {tracks_csv}")

        # Simple per-track summary (birth + final size)
        summary = (
            df.sort_values(["track_id", "t_ms"])
              .groupby("track_id")
              .agg(
                  t_birth_ms=("t_birth_ms", "min"),
                  t_last_ms=("t_ms", "max"),
                  R_last_px=("R_px", "last"),
                  n_points=("R_px", "count"),
              )
              .reset_index()
        )
        summary_csv = ds_out / "track_summary.csv"
        summary.to_csv(summary_csv, index=False)
        print(f"✔ wrote {summary_csv}")

        make_plots(df, name, ds_out)
        print(f"✔ plots saved in {ds_out}")

        all_dfs.append(df)

    # Combined export
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv(OUT_ROOT / "all_tracks.csv", index=False)
    print(f"\nAll done. Combined CSV: {OUT_ROOT / 'all_tracks.csv'}")


if __name__ == "__main__":
    main()
