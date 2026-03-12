#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
stable_nucleation_rebuild_from_json_v5.py

FAST + robust stable nucleation rebuild with overlap gating.

Works with per-frame JSON files that are LISTS of COCO-style annotation dicts.
Handles empty frames [] with bounded lookahead (never infinite loops).

Big speed fix for overlap gating:
- Filter prev-frame candidates by bbox-center proximity (prev_search_pad_px)
- Hard cap number of prev candidates (prev_max_candidates)

Also supports resuming:
- start_track_idx: skip the first N tracks in group order
"""

import os
import re
import glob
import json
import math
import argparse
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

try:
    from pycocotools import mask as mask_utils
except Exception:
    raise SystemExit(
        "[ERR] pycocotools missing. Install in samcuda env:\n"
        "  conda install -c conda-forge pycocotools\n"
        "or\n"
        "  pip install pycocotools"
    )

# -------------------------
# Filename parsing helpers
# -------------------------
_FRAME_RE = re.compile(r"frame_(\d+)_t([0-9.]+)ms", re.IGNORECASE)

def parse_frame_and_time_from_name(fname: str) -> Tuple[Optional[int], Optional[float]]:
    m = _FRAME_RE.search(fname or "")
    if not m:
        return None, None
    return int(m.group(1)), float(m.group(2))

def find_json_for_frame(json_dir: str, frame_idx: int, t_ms: Optional[float] = None) -> Optional[str]:
    patt = os.path.join(json_dir, f"frame_{frame_idx:05d}_t*ms_idmapped.json")
    cands = sorted(glob.glob(patt))
    if not cands:
        return None
    if t_ms is None or len(cands) == 1:
        return cands[0]
    best = cands[0]
    best_dt = float("inf")
    for p in cands:
        _, tt = parse_frame_and_time_from_name(os.path.basename(p))
        if tt is None:
            continue
        dt = abs(tt - float(t_ms))
        if dt < best_dt:
            best_dt = dt
            best = p
    return best

# -------------------------
# JSON loading (LIST schema)
# -------------------------
def read_json_list(path: Optional[str]) -> List[Dict[str, Any]]:
    if path is None or (not os.path.exists(path)):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return []
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ("detections", "annotations"):
            if key in obj and isinstance(obj[key], list):
                return [x for x in obj[key] if isinstance(x, dict)]
    return []

# -------------------------
# RLE helpers
# -------------------------
def normalize_rle(seg: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(seg, dict):
        return None
    if "counts" not in seg or "size" not in seg:
        return None
    rle = dict(seg)
    c = rle["counts"]
    if isinstance(c, str):
        rle["counts"] = c.encode("utf-8")
    return rle

def overlap_frac_a_over_b(rle_a: Dict[str, Any], rle_b: Dict[str, Any]) -> float:
    inter = mask_utils.merge([rle_a, rle_b], intersect=True)
    ai = float(mask_utils.area(inter))
    aa = float(mask_utils.area(rle_a))
    if aa <= 0:
        return float("nan")
    return ai / aa

def overlap_frac_prev(cur_rle: Dict[str, Any], prev_rles: List[Dict[str, Any]]) -> float:
    best_val = 0.0
    got = False
    for prle in prev_rles:
        try:
            of = overlap_frac_a_over_b(cur_rle, prle)
        except Exception:
            continue
        if np.isfinite(of):
            got = True
            if of > best_val:
                best_val = float(of)
    return best_val if got else float("nan")

# -------------------------
# Matching helpers
# -------------------------
def bbox_center(bbox: Any) -> Tuple[float, float]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return float("nan"), float("nan")
    x, y, w, h = bbox[:4]
    return float(x + 0.5 * w), float(y + 0.5 * h)

def match_det_by_centroid(
    anns: List[Dict[str, Any]],
    cx: float,
    cy: float,
    max_dist: float
) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    best = None
    best_d = None
    for ann in anns:
        dx, dy = bbox_center(ann.get("bbox", None))
        if not (np.isfinite(dx) and np.isfinite(dy)):
            continue
        d = math.hypot(dx - cx, dy - cy)
        if d <= max_dist and (best is None or d < best_d):
            best = ann
            best_d = d
    return best, best_d

def get_first_usable_match(
    json_dir: str,
    start_frame: int,
    lookahead_k: int,
    cx: float,
    cy: float,
    max_dist: float
) -> Tuple[Optional[int], Optional[str], Optional[Dict[str, Any]], str]:
    for j in range(0, lookahead_k + 1):
        f = start_frame + j
        jp = find_json_for_frame(json_dir, f, None)
        anns = read_json_list(jp)
        if not anns:
            continue
        ann, _ = match_det_by_centroid(anns, cx, cy, max_dist)
        if ann is None:
            continue
        return f, jp, ann, ("ok" if j == 0 else f"lookahead+{j}")
    return None, None, None, "no_match_in_lookahead"

# -------------------------
# Tracks helpers
# -------------------------
def pick_column(df: pd.DataFrame, candidates: List[str], required_name: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(f"[ERR] tracks.csv missing required {required_name}. Tried: {candidates}")

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--tracks_csv", required=True)
    ap.add_argument("--json_dir", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--L", type=int, default=5)
    ap.add_argument("--amin_px", type=float, default=800.0)
    ap.add_argument("--rmin_px", type=float, default=0.0)
    ap.add_argument("--rnuc_max", type=float, default=60.0)

    ap.add_argument("--use_rmono_gate", action="store_true")
    ap.add_argument("--rmono_min", type=float, default=0.0)

    ap.add_argument("--overlap_prev_max", type=float, default=0.3)
    ap.add_argument("--prefer_centroid_match", action="store_true")  # CLI compatibility
    ap.add_argument("--lookahead_k", type=int, default=3)

    ap.add_argument("--max_dist_min", type=float, default=25.0)
    ap.add_argument("--max_dist_factor", type=float, default=2.0)

    ap.add_argument("--reject_if_overlap_nan", action="store_true")
    ap.add_argument("--progress_every", type=int, default=100)

    # Overlap speed controls
    ap.add_argument("--prev_search_pad_px", type=float, default=150.0,
                    help="Only compare overlap to prev-frame detections within +/-pad (x,y) of current bbox-center.")
    ap.add_argument("--prev_max_candidates", type=int, default=25,
                    help="After pad filtering, test at most this many prev-frame candidates for overlap.")

    # Resume controls
    ap.add_argument("--start_track_idx", type=int, default=0,
                    help="Skip the first N tracks (in grouped order). Useful to resume after stopping.")
    ap.add_argument("--max_tracks", type=int, default=-1,
                    help="Process at most this many tracks after start_track_idx (<=0 means no limit).")

    args = ap.parse_args()

    print(f"[OK] Reading tracks: {args.tracks_csv}", flush=True)
    tracks = pd.read_csv(args.tracks_csv)

    track_id_col = pick_column(tracks, ["track_id", "tid", "id"], "track_id")
    frame_col    = pick_column(tracks, ["frame_idx", "frame", "frame_i", "frame_id"], "frame index")
    cx_col       = pick_column(tracks, ["cx", "cx_px", "x", "x_px"], "cx")
    cy_col       = pick_column(tracks, ["cy", "cy_px", "y", "y_px"], "cy")
    area_col     = pick_column(tracks, ["area_px", "area"], "area_px")
    r_col        = pick_column(tracks, ["R_px", "r_px", "radius_px", "R"], "R_px")

    t_col = None
    for c in ("t_ms", "time_ms", "t", "time"):
        if c in tracks.columns:
            t_col = c
            break

    rmono_col = None
    for c in ("R_mono", "r_mono", "rmono"):
        if c in tracks.columns:
            rmono_col = c
            break

    tracks = tracks.sort_values([track_id_col, frame_col]).reset_index(drop=True)
    groups = list(tracks.groupby(track_id_col, sort=False))
    n_tracks_total = len(groups)

    print(f"[OK] Tracks: {n_tracks_total} unique track_id", flush=True)

    start = max(0, int(args.start_track_idx))
    end = n_tracks_total
    if args.max_tracks and int(args.max_tracks) > 0:
        end = min(end, start + int(args.max_tracks))

    if start >= end:
        raise SystemExit(f"[ERR] start_track_idx={start} >= end={end}. Nothing to do.")

    os.makedirs(args.out_dir, exist_ok=True)

    filtered: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    rejected_trackgate: List[Dict[str, Any]] = []

    # Process only slice [start:end]
    slice_groups = groups[start:end]
    n_slice = len(slice_groups)

    for local_i, (tid, df) in enumerate(slice_groups, start=1):
        global_i = start + local_i  # 1-based-ish progress indicator
        n_obs = len(df)

        if n_obs < args.L:
            rejected.append({"track_id": int(tid), "reason": f"too_short(n={n_obs}<L={args.L})"})
            if (local_i % args.progress_every) == 0 or local_i == n_slice:
                print(f"[PROGRESS] {local_i}/{n_slice} (global {global_i}/{n_tracks_total}) | "
                      f"kept={len(filtered)} | rej={len(rejected)} | rej_trackgate={len(rejected_trackgate)}",
                      flush=True)
            continue

        frames = df[frame_col].to_numpy(dtype=int)
        areas  = df[area_col].to_numpy(dtype=float)
        Rs     = df[r_col].to_numpy(dtype=float)
        cxs    = df[cx_col].to_numpy(dtype=float)
        cys    = df[cy_col].to_numpy(dtype=float)
        times  = df[t_col].to_numpy(dtype=float) if t_col else None
        rmono  = df[rmono_col].to_numpy(dtype=float) if (args.use_rmono_gate and rmono_col) else None

        ok = (areas >= args.amin_px) & (Rs >= args.rmin_px) & (Rs <= args.rnuc_max)
        if rmono is not None:
            ok = ok & (rmono >= args.rmono_min)

        nuc_i = None
        for i in range(0, n_obs - args.L + 1):
            if np.all(ok[i:i + args.L]):
                nuc_i = i
                break

        if nuc_i is None:
            rejected.append({
                "track_id": int(tid),
                "reason": f"no_stable_run_L{args.L}_after_gates",
                "n_obs_total": int(n_obs),
                "amin_px": float(args.amin_px),
                "rnuc_max": float(args.rnuc_max),
            })
            if (local_i % args.progress_every) == 0 or local_i == n_slice:
                print(f"[PROGRESS] {local_i}/{n_slice} (global {global_i}/{n_tracks_total}) | "
                      f"kept={len(filtered)} | rej={len(rejected)} | rej_trackgate={len(rejected_trackgate)}",
                      flush=True)
            continue

        nuc_frame = int(frames[nuc_i])
        nuc_time  = float(times[nuc_i]) if times is not None else float("nan")
        R_nuc     = float(Rs[nuc_i])
        R_max     = float(np.nanmax(Rs))

        max_dist = max(args.max_dist_min, args.max_dist_factor * max(R_nuc, 1.0))

        mask_frame, json_path, ann, match_note = get_first_usable_match(
            json_dir=args.json_dir,
            start_frame=nuc_frame,
            lookahead_k=args.lookahead_k,
            cx=float(cxs[nuc_i]),
            cy=float(cys[nuc_i]),
            max_dist=float(max_dist),
        )
        if ann is None or json_path is None or mask_frame is None:
            rejected.append({
                "track_id": int(tid),
                "reason": "no_mask_match",
                "nuc_frame_i": int(nuc_frame),
                "nuc_time_ms": nuc_time,
                "match_note": match_note,
                "lookahead_k": int(args.lookahead_k),
                "max_dist_px": float(max_dist),
            })
            if (local_i % args.progress_every) == 0 or local_i == n_slice:
                print(f"[PROGRESS] {local_i}/{n_slice} (global {global_i}/{n_tracks_total}) | "
                      f"kept={len(filtered)} | rej={len(rejected)} | rej_trackgate={len(rejected_trackgate)}",
                      flush=True)
            continue

        cur_rle = normalize_rle(ann.get("segmentation", None))
        if cur_rle is None:
            rejected.append({
                "track_id": int(tid),
                "reason": "no_rle_in_segmentation",
                "nuc_frame_i": int(nuc_frame),
                "mask_frame_i": int(mask_frame),
                "nuc_time_ms": nuc_time,
                "match_note": match_note,
            })
            if (local_i % args.progress_every) == 0 or local_i == n_slice:
                print(f"[PROGRESS] {local_i}/{n_slice} (global {global_i}/{n_tracks_total}) | "
                      f"kept={len(filtered)} | rej={len(rejected)} | rej_trackgate={len(rejected_trackgate)}",
                      flush=True)
            continue

        # -------- overlap gate --------
        if args.overlap_prev_max >= 0.999:
            overlap_prev = float("nan")
            overlap_note = "skipped_overlap_prev"
        else:
            prev_frame = int(mask_frame) - 1
            prev_path = find_json_for_frame(args.json_dir, prev_frame, None) if prev_frame >= 0 else None
            prev_anns = read_json_list(prev_path)

            overlap_prev = float("nan")
            overlap_note = "ok"

            if not prev_anns:
                overlap_note = "prev_frame_empty_or_missing"
            else:
                # SPEED: keep only prev detections near current bbox-center
                cur_cx, cur_cy = bbox_center(ann.get("bbox", None))
                pad = float(args.prev_search_pad_px)

                if np.isfinite(cur_cx) and np.isfinite(cur_cy):
                    nearby = []
                    for pa in prev_anns:
                        px, py = bbox_center(pa.get("bbox", None))
                        if not (np.isfinite(px) and np.isfinite(py)):
                            continue
                        if abs(px - cur_cx) <= pad and abs(py - cur_cy) <= pad:
                            nearby.append(pa)
                    prev_anns = nearby

                if not prev_anns:
                    overlap_note = "prev_frame_no_candidates_nearby"
                else:
                    # HARD CAP: prevent worst-case frames from exploding runtime
                    if len(prev_anns) > int(args.prev_max_candidates):
                        prev_anns = prev_anns[:int(args.prev_max_candidates)]

                    prev_rles = []
                    for pa in prev_anns:
                        prle = normalize_rle(pa.get("segmentation", None))
                        if prle is not None:
                            prev_rles.append(prle)

                    if not prev_rles:
                        overlap_note = "prev_frame_no_valid_rle"
                    else:
                        try:
                            overlap_prev = overlap_frac_prev(cur_rle, prev_rles)
                        except Exception as e:
                            overlap_prev = float("nan")
                            overlap_note = f"overlap_failed:{type(e).__name__}"

            if np.isnan(overlap_prev) and args.reject_if_overlap_nan:
                rejected_trackgate.append({
                    "track_id": int(tid),
                    "reason": "overlap_nan",
                    "nuc_frame_i": int(nuc_frame),
                    "mask_frame_i": int(mask_frame),
                    "match_note": match_note,
                    "overlap_note": overlap_note,
                })
                if (local_i % args.progress_every) == 0 or local_i == n_slice:
                    print(f"[PROGRESS] {local_i}/{n_slice} (global {global_i}/{n_tracks_total}) | "
                          f"kept={len(filtered)} | rej={len(rejected)} | rej_trackgate={len(rejected_trackgate)}",
                          flush=True)
                continue

            if np.isfinite(overlap_prev) and overlap_prev > args.overlap_prev_max:
                rejected_trackgate.append({
                    "track_id": int(tid),
                    "reason": f"overlap_prev>{args.overlap_prev_max}",
                    "nuc_frame_i": int(nuc_frame),
                    "mask_frame_i": int(mask_frame),
                    "match_note": match_note,
                    "overlap_prev": float(overlap_prev),
                    "overlap_note": overlap_note,
                })
                if (local_i % args.progress_every) == 0 or local_i == n_slice:
                    print(f"[PROGRESS] {local_i}/{n_slice} (global {global_i}/{n_tracks_total}) | "
                          f"kept={len(filtered)} | rej={len(rejected)} | rej_trackgate={len(rejected_trackgate)}",
                          flush=True)
                continue

        # -------- accept --------
        _, mt = parse_frame_and_time_from_name(os.path.basename(json_path))
        filtered.append({
            "track_id": int(tid),
            "nuc_time_ms": nuc_time,
            "nuc_frame_i": int(nuc_frame),
            "mask_frame_i": int(mask_frame),
            "mask_time_ms": float(mt) if mt is not None else float("nan"),
            "R_nuc_px": float(R_nuc),
            "R_max_px": float(R_max),
            "n_obs_total": int(n_obs),
            "match_note": match_note,
            "max_dist_px": float(max_dist),
            "overlap_prev": float(overlap_prev) if np.isfinite(overlap_prev) else float("nan"),
            "overlap_prev_max": float(args.overlap_prev_max),
            "overlap_note": overlap_note,
        })

        if (local_i % args.progress_every) == 0 or local_i == n_slice:
            print(f"[PROGRESS] {local_i}/{n_slice} (global {global_i}/{n_tracks_total}) | "
                  f"kept={len(filtered)} | rej={len(rejected)} | rej_trackgate={len(rejected_trackgate)}",
                  flush=True)

    # Always write outputs
    pd.DataFrame(filtered).to_csv(os.path.join(args.out_dir, "nucleation_events_filtered.csv"), index=False)
    pd.DataFrame(rejected).to_csv(os.path.join(args.out_dir, "nucleation_events_rejected.csv"), index=False)
    pd.DataFrame(rejected_trackgate).to_csv(os.path.join(args.out_dir, "nucleation_events_rejected_trackgate.csv"), index=False)

    print(f"[OK] Wrote outputs to: {args.out_dir}", flush=True)
    print(f"[OK] kept={len(filtered)} | rejected={len(rejected)} | rejected_trackgate={len(rejected_trackgate)}", flush=True)


if __name__ == "__main__":
    main()
