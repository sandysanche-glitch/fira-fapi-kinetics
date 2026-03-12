#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import time
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional (polygon IoU)
HAS_SHAPELY = False
try:
    from shapely.geometry import Polygon
    from shapely.errors import TopologicalError
    HAS_SHAPELY = True
except Exception:
    HAS_SHAPELY = False


# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def bbox_center_xywh(b: List[float]) -> Tuple[float, float]:
    x, y, w, h = b
    return (x + 0.5 * w, y + 0.5 * h)

def bbox_iou_xywh(b1: List[float], b2: List[float]) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
    bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = w1 * h1
    area_b = w2 * h2
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0

def segmentation_to_polygon(seg: Any) -> Optional["Polygon"]:
    """
    COCO-style segmentation can be:
      - list of lists: [[x1,y1,x2,y2,...], [...]]  (multiple rings)
      - list: [x1,y1,x2,y2,...]
    We'll take the largest-area ring as polygon (common in your data).
    """
    if not HAS_SHAPELY:
        return None
    try:
        if seg is None:
            return None
        rings: List[List[float]] = []
        if isinstance(seg, list) and len(seg) > 0:
            if isinstance(seg[0], list):
                rings = seg
            else:
                rings = [seg]
        else:
            return None

        polys = []
        for r in rings:
            if not isinstance(r, list) or len(r) < 6:
                continue
            coords = [(float(r[i]), float(r[i + 1])) for i in range(0, len(r) - 1, 2)]
            if len(coords) < 3:
                continue
            p = Polygon(coords)
            if not p.is_valid:
                p = p.buffer(0)
            if p.is_empty:
                continue
            polys.append(p)

        if not polys:
            return None
        polys.sort(key=lambda p: p.area, reverse=True)
        return polys[0]
    except (ValueError, TopologicalError):
        return None
    except Exception:
        return None

def poly_iou(seg_a: Any, seg_b: Any) -> Optional[float]:
    if not HAS_SHAPELY:
        return None
    pa = segmentation_to_polygon(seg_a)
    pb = segmentation_to_polygon(seg_b)
    if pa is None or pb is None:
        return None
    try:
        inter = pa.intersection(pb).area
        union = pa.union(pb).area
        if union <= 0:
            return 0.0
        return float(inter / union)
    except Exception:
        return None


# -----------------------------
# JSON frame loading + caching
# -----------------------------

FRAME_RE = re.compile(r"^frame_(\d+)_.*_idmapped\.json$")

@dataclass
class DetObj:
    oid: Optional[int]
    bbox: List[float]          # [x,y,w,h]
    cx: float
    cy: float
    area_px: Optional[float]
    seg: Any

class FrameCache:
    def __init__(self, json_dir: str):
        self.json_dir = json_dir
        self._map: Dict[int, str] = {}
        self._cache: Dict[int, List[DetObj]] = {}

        for fn in os.listdir(json_dir):
            m = FRAME_RE.match(fn)
            if m:
                fi = int(m.group(1))
                self._map[fi] = os.path.join(json_dir, fn)

    def has_frame(self, frame_i: int) -> bool:
        return frame_i in self._map

    def load(self, frame_i: int) -> List[DetObj]:
        if frame_i in self._cache:
            return self._cache[frame_i]
        path = self._map.get(frame_i, None)
        if path is None:
            self._cache[frame_i] = []
            return []
        try:
            with open(path, "r") as f:
                obj = json.load(f)
        except Exception:
            self._cache[frame_i] = []
            return []

        dets: List[DetObj] = []
        if isinstance(obj, list):
            for a in obj:
                if not isinstance(a, dict):
                    continue
                bbox = a.get("bbox", None)
                if not (isinstance(bbox, list) and len(bbox) == 4):
                    continue
                cx, cy = bbox_center_xywh(bbox)
                dets.append(
                    DetObj(
                        oid=a.get("id", None),
                        bbox=[float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                        cx=float(cx),
                        cy=float(cy),
                        area_px=(float(a["area_px"]) if "area_px" in a and a["area_px"] is not None else None),
                        seg=a.get("segmentation", None),
                    )
                )
        self._cache[frame_i] = dets
        return dets


# -----------------------------
# Offset estimation
# -----------------------------

def estimate_offsets(
    tracks: pd.DataFrame,
    cache: FrameCache,
    out_csv: str,
    min_tracks_per_frame: int = 20,
    smooth_window: int = 21,
    use_prev_if_empty: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Convention (IMPORTANT):
      dx = median(track_cx - json_bbox_cx)
      dy = median(track_cy - json_bbox_cy)

    So to map a track point into JSON coordinate system:
      cx_json = cx_track - dx
      cy_json = cy_track - dy
    """
    if verbose:
        print("[OK] Estimating per-frame offsets (tracks vs JSON bbox centers)...", flush=True)

    frames = sorted(tracks["frame_idx"].unique().tolist())
    rows = []
    prev_dx = 0.0
    prev_dy = 0.0

    for f in frames:
        sub = tracks[tracks["frame_idx"] == f]
        if len(sub) < min_tracks_per_frame:
            rows.append((f, np.nan, np.nan, len(sub)))
            continue

        dets = cache.load(int(f))
        if len(dets) == 0:
            rows.append((f, np.nan, np.nan, len(sub)))
            continue

        # Compare medians (robust against outliers)
        track_cx = float(sub["cx"].median())
        track_cy = float(sub["cy"].median())
        json_cx = float(np.median([d.cx for d in dets]))
        json_cy = float(np.median([d.cy for d in dets]))

        dx = track_cx - json_cx
        dy = track_cy - json_cy
        rows.append((f, dx, dy, len(sub)))

    df = pd.DataFrame(rows, columns=["frame_idx", "dx_raw", "dy_raw", "n_tracks"])
    df["dx_s"] = df["dx_raw"].rolling(smooth_window, center=True, min_periods=1).median()
    df["dy_s"] = df["dy_raw"].rolling(smooth_window, center=True, min_periods=1).median()

    if use_prev_if_empty:
        df["dx_s"] = df["dx_s"].ffill().bfill().fillna(0.0)
        df["dy_s"] = df["dy_s"].ffill().bfill().fillna(0.0)
    else:
        df["dx_s"] = df["dx_s"].fillna(0.0)
        df["dy_s"] = df["dy_s"].fillna(0.0)

    df.to_csv(out_csv, index=False)

    if verbose:
        med_dx = float(np.median(df["dx_s"].values))
        med_dy = float(np.median(df["dy_s"].values))
        print(f"[OK] Offset estimation done. median dx={med_dx:.2f}, dy={med_dy:.2f}. Saved: {out_csv}", flush=True)

    return df[["frame_idx", "dx_s", "dy_s"]].copy()


# -----------------------------
# Matching / Overlap
# -----------------------------

def preselect_prev_candidates(
    prev_dets: List[DetObj],
    cx_json: float,
    cy_json: float,
    pad_px: float,
    prev_max_candidates: int,
    k_closest: int,
) -> List[DetObj]:
    if len(prev_dets) == 0:
        return []
    # window filter
    win = []
    x0, x1 = cx_json - pad_px, cx_json + pad_px
    y0, y1 = cy_json - pad_px, cy_json + pad_px
    for d in prev_dets:
        if x0 <= d.cx <= x1 and y0 <= d.cy <= y1:
            win.append(d)

    if len(win) == 0:
        return []

    # sort by distance and keep top-N
    win.sort(key=lambda d: (d.cx - cx_json) ** 2 + (d.cy - cy_json) ** 2)
    if prev_max_candidates > 0:
        win = win[:prev_max_candidates]
    if k_closest > 0 and len(win) > k_closest:
        win = win[:k_closest]
    return win

def compute_overlap_prev(
    prev_candidates: List[DetObj],
    curr_det: DetObj,
    overlap_prev_max: float,
    strict_overlap: bool,
    overlap_strategy: str,
    overlap_unknown_policy: str,
    per_call_timeout_s: float,
) -> Tuple[Optional[float], str]:
    """
    Returns (overlap_prev_value or None, note)
    None means "unknown" (could not compute). Policy handles it.
    """
    if len(prev_candidates) == 0:
        return (0.0, "prev_nocand_assumed_0")

    t0 = time.time()

    # Strategy:
    #  - "bbox": use bbox IoU only
    #  - "poly": use polygon IoU only (requires shapely)
    #  - "bbox_then_poly": fast bbox IoU gate, then polygon IoU if needed & possible
    strat = overlap_strategy.lower().strip()

    best = -1.0
    best_note = ""
    n = len(prev_candidates)

    for i, p in enumerate(prev_candidates):
        if per_call_timeout_s > 0 and (time.time() - t0) > per_call_timeout_s:
            return (None, f"overlap_timeout(n={n})")

        if strat == "bbox":
            ov = bbox_iou_xywh(p.bbox, curr_det.bbox)
        elif strat == "poly":
            if not HAS_SHAPELY:
                return (None, "no_shapely")
            ovp = poly_iou(p.seg, curr_det.seg)
            if ovp is None:
                return (None, "poly_failed")
            ov = ovp
        else:  # bbox_then_poly
            ov_bbox = bbox_iou_xywh(p.bbox, curr_det.bbox)
            # If bbox IoU already exceeds threshold, that's enough to reject in strict mode
            # and we don't need the slower polygon.
            if strict_overlap and ov_bbox > overlap_prev_max:
                ov = ov_bbox
            else:
                # Try polygon only if available; otherwise use bbox.
                if HAS_SHAPELY:
                    ovp = poly_iou(p.seg, curr_det.seg)
                    ov = ovp if ovp is not None else ov_bbox
                else:
                    ov = ov_bbox

        if ov is None:
            continue
        if ov > best:
            best = ov

    if best < 0:
        return (None, "overlap_all_failed")

    return (float(best), f"prev_max({n})={best:.3f}")


# -----------------------------
# Main pipeline
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks_csv", required=True)
    ap.add_argument("--json_dir", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--L", type=int, default=5)
    ap.add_argument("--amin_px", type=float, default=800)
    ap.add_argument("--rnuc_max", type=float, default=60)

    ap.add_argument("--lookahead_k", type=int, default=150)

    ap.add_argument("--strict_overlap", action="store_true")
    ap.add_argument("--overlap_prev_max", type=float, default=0.3)

    ap.add_argument("--overlap_strategy", type=str, default="bbox_then_poly",
                    choices=["bbox", "poly", "bbox_then_poly"])
    ap.add_argument("--overlap_unknown_policy", type=str, default="keep",
                    choices=["keep", "reject", "assume0"])

    ap.add_argument("--prev_search_pad_px", type=float, default=250)
    ap.add_argument("--prev_max_candidates", type=int, default=50)
    ap.add_argument("--k_closest", type=int, default=15)

    ap.add_argument("--accept_empty_prev", action="store_true")
    ap.add_argument("--accept_no_candidate_prev", action="store_true")

    ap.add_argument("--overlap_per_call_timeout_s", type=float, default=0.60)

    # Offset
    ap.add_argument("--auto_offset", action="store_true")
    ap.add_argument("--offset_min_tracks", type=int, default=20)
    ap.add_argument("--offset_smooth_window", type=int, default=21)
    ap.add_argument("--offset_use_prev_if_empty", action="store_true")

    # Run control
    ap.add_argument("--progress_every", type=int, default=25)
    ap.add_argument("--checkpoint_every", type=int, default=100)
    ap.add_argument("--start_track_idx", type=int, default=0)
    ap.add_argument("--max_tracks", type=int, default=0)  # 0 = all
    ap.add_argument("--resume", action="store_true")

    args = ap.parse_args()

    ensure_dir(args.out_dir)
    state_path = os.path.join(args.out_dir, "state.json")
    ckpt_prefix = os.path.join(args.out_dir, "nucleation_events")

    # Load tracks
    print(f"[OK] Reading tracks: {args.tracks_csv}", flush=True)
    tracks = pd.read_csv(args.tracks_csv)

    # Normalize column names (your CSVs sometimes use frame_idx)
    if "frame_i" in tracks.columns and "frame_idx" not in tracks.columns:
        tracks = tracks.rename(columns={"frame_i": "frame_idx"})
    required = ["track_id", "frame_idx", "cx", "cy"]
    for c in required:
        if c not in tracks.columns:
            raise RuntimeError(f"tracks_csv missing column: {c}")

    # Load/cache JSON
    cache = FrameCache(args.json_dir)

    # Offsets per frame
    offsets_df = None
    offsets_map: Dict[int, Tuple[float, float]] = {}
    if args.auto_offset:
        out_offsets = os.path.join(args.out_dir, "offsets_estimated.csv")
        offsets_df = estimate_offsets(
            tracks=tracks,
            cache=cache,
            out_csv=out_offsets,
            min_tracks_per_frame=args.offset_min_tracks,
            smooth_window=args.offset_smooth_window,
            use_prev_if_empty=args.offset_use_prev_if_empty,
            verbose=True,
        )
        for _, r in offsets_df.iterrows():
            offsets_map[int(r["frame_idx"])] = (float(r["dx_s"]), float(r["dy_s"]))

    # Track list
    track_ids = sorted(tracks["track_id"].unique().tolist())
    print(f"[OK] Tracks: {len(track_ids)} unique track_id", flush=True)

    # Resume support
    start_i = int(args.start_track_idx)
    filtered_rows: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []

    if args.resume and os.path.exists(state_path):
        try:
            with open(state_path, "r") as f:
                st = json.load(f)
            start_i = max(start_i, int(st.get("next_index", start_i)))
            # You can also reload partial CSVs here if you want append-safe.
            print(f"[OK] Resume enabled. Starting at global index {start_i}.", flush=True)
        except Exception:
            pass

    def save_ckpt(global_i: int):
        pd.DataFrame(filtered_rows).to_csv(os.path.join(args.out_dir, "nucleation_events_filtered.csv"), index=False)
        pd.DataFrame(rejected_rows).to_csv(os.path.join(args.out_dir, "nucleation_events_rejected.csv"), index=False)

        if args.checkpoint_every > 0:
            if (global_i % args.checkpoint_every) == 0:
                pd.DataFrame(filtered_rows).to_csv(f"{ckpt_prefix}_filtered_ckpt_{global_i:05d}.csv", index=False)
                pd.DataFrame(rejected_rows).to_csv(f"{ckpt_prefix}_rejected_ckpt_{global_i:05d}.csv", index=False)

        with open(state_path, "w") as f:
            json.dump(
                {
                    "next_index": global_i,
                    "args": vars(args),
                    "has_shapely": HAS_SHAPELY,
                },
                f,
                indent=2,
            )

    kept = 0
    rej = 0

    # main loop
    end_i = len(track_ids) if args.max_tracks <= 0 else min(len(track_ids), start_i + args.max_tracks)

    for gi in range(start_i, end_i):
        tid = track_ids[gi]
        sub = tracks[tracks["track_id"] == tid].sort_values("frame_idx")

        # Lookahead window: only examine first lookahead_k frames of each track
        if args.lookahead_k > 0 and len(sub) > args.lookahead_k:
            sub = sub.iloc[: args.lookahead_k].copy()

        # Identify "passes" frames based on area/R if present; otherwise we use rnuc gate based on bbox only later
        # Your tracks.csv likely has area_px and R_px (or similar) sometimes. We'll use them if present.
        has_area = ("area_px" in sub.columns)
        has_R = ("R_px" in sub.columns)

        def frame_passes(row) -> bool:
            ok = True
            if has_area:
                ok = ok and (float(row["area_px"]) >= float(args.amin_px))
            if has_R:
                ok = ok and (float(row["R_px"]) <= float(args.rnuc_max))
            return bool(ok)

        pass_idx = [i for i, r in sub.iterrows() if frame_passes(r)]
        if not pass_idx:
            rej += 1
            rejected_rows.append(
                dict(
                    track_id=tid,
                    reason="trackgate_no_frame_passes_area_rnuc",
                    n_obs_total=len(sub),
                    amin_px=args.amin_px,
                    rnuc_max=args.rnuc_max,
                    nuc_frame_i=np.nan,
                    nuc_time_ms=np.nan,
                    overlap_prev=np.nan,
                    overlap_note=np.nan,
                )
            )
            if args.progress_every > 0 and ((gi + 1) % args.progress_every == 0):
                print(f"[PROGRESS] {gi+1}/{len(track_ids)} | kept={kept} | rej={rej}", flush=True)
                save_ckpt(gi + 1)
            continue

        # Candidate nucleation start = first passing frame
        # We then require a stable run of L consecutive passing frames
        pass_frames = sub.loc[pass_idx, "frame_idx"].astype(int).tolist()
        pass_frames_set = set(pass_frames)

        nuc_frame = None
        for f in pass_frames:
            ok_run = True
            for k in range(args.L):
                if (f + k) not in pass_frames_set:
                    ok_run = False
                    break
            if ok_run:
                nuc_frame = int(f)
                break

        if nuc_frame is None:
            rej += 1
            rejected_rows.append(
                dict(
                    track_id=tid,
                    reason=f"no_stable_run_L{args.L}_after_gates",
                    n_obs_total=len(sub),
                    amin_px=args.amin_px,
                    rnuc_max=args.rnuc_max,
                    nuc_frame_i=np.nan,
                    nuc_time_ms=np.nan,
                    overlap_prev=np.nan,
                    overlap_note=np.nan,
                )
            )
            if args.progress_every > 0 and ((gi + 1) % args.progress_every == 0):
                print(f"[PROGRESS] {gi+1}/{len(track_ids)} | kept={kept} | rej={rej}", flush=True)
                save_ckpt(gi + 1)
            continue

        # Get track centroid at nuc_frame
        r_nuc = sub[sub["frame_idx"] == nuc_frame].iloc[0]
        cx_t = float(r_nuc["cx"])
        cy_t = float(r_nuc["cy"])

        # Apply offset mapping: track -> JSON coords
        dx, dy = (0.0, 0.0)
        if offsets_map:
            dx, dy = offsets_map.get(int(nuc_frame), (0.0, 0.0))
        cx_json = cx_t - dx
        cy_json = cy_t - dy

        # Load current frame detections and find best match (closest bbox center to track point in JSON coords)
        curr_dets = cache.load(int(nuc_frame))
        if len(curr_dets) == 0:
            rej += 1
            rejected_rows.append(
                dict(
                    track_id=tid,
                    reason="missing_or_empty_json_curr",
                    n_obs_total=len(sub),
                    amin_px=args.amin_px,
                    rnuc_max=args.rnuc_max,
                    nuc_frame_i=nuc_frame,
                    nuc_time_ms=(float(r_nuc["t_ms"]) if "t_ms" in r_nuc else np.nan),
                    overlap_prev=np.nan,
                    overlap_note="curr_empty",
                )
            )
            continue

        curr_dets.sort(key=lambda d: (d.cx - cx_json) ** 2 + (d.cy - cy_json) ** 2)
        curr_det = curr_dets[0]

        # Previous-frame overlap gate
        prev_frame = nuc_frame - 1
        if prev_frame < 0 or not cache.has_frame(prev_frame):
            if args.accept_empty_prev:
                overlap_prev = 0.0
                overlap_note = "prev_missing_assumed_0"
            else:
                rej += 1
                rejected_rows.append(
                    dict(
                        track_id=tid,
                        reason="prev_missing",
                        n_obs_total=len(sub),
                        amin_px=args.amin_px,
                        rnuc_max=args.rnuc_max,
                        nuc_frame_i=nuc_frame,
                        nuc_time_ms=(float(r_nuc["t_ms"]) if "t_ms" in r_nuc else np.nan),
                        overlap_prev=np.nan,
                        overlap_note="prev_missing",
                    )
                )
                continue
        else:
            prev_dets = cache.load(prev_frame)
            if len(prev_dets) == 0:
                if args.accept_empty_prev:
                    overlap_prev = 0.0
                    overlap_note = "prev_empty_assumed_0"
                else:
                    rej += 1
                    rejected_rows.append(
                        dict(
                            track_id=tid,
                            reason="prev_empty",
                            n_obs_total=len(sub),
                            amin_px=args.amin_px,
                            rnuc_max=args.rnuc_max,
                            nuc_frame_i=nuc_frame,
                            nuc_time_ms=(float(r_nuc["t_ms"]) if "t_ms" in r_nuc else np.nan),
                            overlap_prev=np.nan,
                            overlap_note="prev_empty",
                        )
                    )
                    continue
            else:
                # Use offset at prev_frame to map the same track point into prev JSON coords
                dxp, dyp = (dx, dy)
                if offsets_map:
                    dxp, dyp = offsets_map.get(int(prev_frame), (dx, dy))
                cx_prev_json = cx_t - dxp
                cy_prev_json = cy_t - dyp

                prev_candidates = preselect_prev_candidates(
                    prev_dets=prev_dets,
                    cx_json=cx_prev_json,
                    cy_json=cy_prev_json,
                    pad_px=float(args.prev_search_pad_px),
                    prev_max_candidates=int(args.prev_max_candidates),
                    k_closest=int(args.k_closest),
                )

                if len(prev_candidates) == 0:
                    if args.accept_no_candidate_prev:
                        overlap_prev = 0.0
                        overlap_note = "prev_nocand_assumed_0"
                    else:
                        rej += 1
                        rejected_rows.append(
                            dict(
                                track_id=tid,
                                reason="no_prev_candidates",
                                n_obs_total=len(sub),
                                amin_px=args.amin_px,
                                rnuc_max=args.rnuc_max,
                                nuc_frame_i=nuc_frame,
                                nuc_time_ms=(float(r_nuc["t_ms"]) if "t_ms" in r_nuc else np.nan),
                                overlap_prev=np.nan,
                                overlap_note="prev_nocand",
                            )
                        )
                        continue
                else:
                    ov, note = compute_overlap_prev(
                        prev_candidates=prev_candidates,
                        curr_det=curr_det,
                        overlap_prev_max=float(args.overlap_prev_max),
                        strict_overlap=bool(args.strict_overlap),
                        overlap_strategy=str(args.overlap_strategy),
                        overlap_unknown_policy=str(args.overlap_unknown_policy),
                        per_call_timeout_s=float(args.overlap_per_call_timeout_s),
                    )

                    if ov is None:
                        # Unknown overlap - apply policy
                        pol = str(args.overlap_unknown_policy).lower().strip()
                        if pol == "assume0":
                            overlap_prev = 0.0
                            overlap_note = f"{note}|assume0"
                        elif pol == "reject":
                            rej += 1
                            rejected_rows.append(
                                dict(
                                    track_id=tid,
                                    reason="overlap_unknown_reject",
                                    n_obs_total=len(sub),
                                    amin_px=args.amin_px,
                                    rnuc_max=args.rnuc_max,
                                    nuc_frame_i=nuc_frame,
                                    nuc_time_ms=(float(r_nuc["t_ms"]) if "t_ms" in r_nuc else np.nan),
                                    overlap_prev=np.nan,
                                    overlap_note=note,
                                )
                            )
                            continue
                        else:  # keep
                            overlap_prev = np.nan
                            overlap_note = note
                    else:
                        overlap_prev = float(ov)
                        overlap_note = note

                    # Strict overlap gate
                    if args.strict_overlap and (not np.isnan(overlap_prev)) and overlap_prev > float(args.overlap_prev_max):
                        rej += 1
                        rejected_rows.append(
                            dict(
                                track_id=tid,
                                reason=f"overlap_prev>{args.overlap_prev_max}",
                                n_obs_total=len(sub),
                                amin_px=args.amin_px,
                                rnuc_max=args.rnuc_max,
                                nuc_frame_i=nuc_frame,
                                nuc_time_ms=(float(r_nuc["t_ms"]) if "t_ms" in r_nuc else np.nan),
                                overlap_prev=overlap_prev,
                                overlap_note=overlap_note,
                            )
                        )
                        continue

        # Keep event
        kept += 1
        filtered_rows.append(
            dict(
                track_id=tid,
                nuc_frame_i=nuc_frame,
                nuc_time_ms=(float(r_nuc["t_ms"]) if "t_ms" in r_nuc else np.nan),
                n_obs_total=len(sub),
                amin_px=args.amin_px,
                rnuc_max=args.rnuc_max,
                overlap_prev=overlap_prev,
                overlap_note=overlap_note,
                dx_used=dx,
                dy_used=dy,
                has_shapely=HAS_SHAPELY,
            )
        )

        if args.progress_every > 0 and ((gi + 1) % args.progress_every == 0):
            print(f"[PROGRESS] {gi+1}/{len(track_ids)} | kept={kept} | rej={rej}", flush=True)
            save_ckpt(gi + 1)

    # final save
    pd.DataFrame(filtered_rows).to_csv(os.path.join(args.out_dir, "nucleation_events_filtered.csv"), index=False)
    pd.DataFrame(rejected_rows).to_csv(os.path.join(args.out_dir, "nucleation_events_rejected.csv"), index=False)
    with open(state_path, "w") as f:
        json.dump(
            {
                "next_index": end_i,
                "args": vars(args),
                "has_shapely": HAS_SHAPELY,
            },
            f,
            indent=2,
        )

    print(f"[OK] Wrote outputs to: {args.out_dir}", flush=True)
    print(f"[OK] kept={kept} | rejected={rej} | has_shapely={HAS_SHAPELY}", flush=True)
    print(f"[OK] State file: {state_path}", flush=True)


if __name__ == "__main__":
    main()