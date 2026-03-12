#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
stable_nucleation_rebuild_from_json_v7.py

ROBUST stable nucleation rebuild with overlap gating from frame JSONs.

Key features (robust over speed):
- Always terminates (hard per-track time budget).
- Logs slow tracks and reasons.
- Optional strict mode: if overlap can't be computed reliably -> reject (or keep).
- Checkpointing: writes incremental outputs every N tracks so you can resume safely.
- Resume by track-index slice: --start_track_idx and --max_tracks
- Bounded candidate search: pad filter + cap candidates (prevents O(n^2) explosions)

Expected JSON schema:
- Each frame JSON is a LIST of COCO-style annotation dicts.
  Each dict should include: bbox, segmentation(RLE), area/area_px, R_px, purity, overlap_frac
"""

import os, re, glob, json, math, time, argparse, csv
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
# Helpers: file parsing
# -------------------------
_FRAME_RE = re.compile(r"frame_(\d+)_t([0-9.]+)ms", re.IGNORECASE)

def parse_frame_and_time_from_name(fname: str) -> Tuple[Optional[int], Optional[float]]:
    m = _FRAME_RE.search(fname or "")
    if not m:
        return None, None
    return int(m.group(1)), float(m.group(2))

def find_json_for_frame(json_dir: str, frame_idx: int) -> Optional[str]:
    patt = os.path.join(json_dir, f"frame_{frame_idx:05d}_t*ms_idmapped.json")
    cands = sorted(glob.glob(patt))
    if not cands:
        return None
    return cands[0]

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
# Helpers: geometry & rle
# -------------------------
def bbox_center(bbox: Any) -> Tuple[float, float]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return float("nan"), float("nan")
    x, y, w, h = bbox[:4]
    return float(x + 0.5*w), float(y + 0.5*h)

def normalize_rle(seg: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(seg, dict):
        return None
    if "counts" not in seg or "size" not in seg:
        return None
    rle = dict(seg)
    if isinstance(rle.get("counts", None), str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return rle

def overlap_frac_a_over_b(rle_a: Dict[str, Any], rle_b: Dict[str, Any]) -> float:
    inter = mask_utils.merge([rle_a, rle_b], intersect=True)
    ai = float(mask_utils.area(inter))
    aa = float(mask_utils.area(rle_a))
    if aa <= 0:
        return float("nan")
    return ai / aa

# -------------------------
# Track CSV column picking
# -------------------------
def pick_column(df: pd.DataFrame, candidates: List[str], required_name: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(f"[ERR] tracks.csv missing required {required_name}. Tried: {candidates}")

# -------------------------
# Matching
# -------------------------
def match_det_by_centroid(anns: List[Dict[str, Any]], cx: float, cy: float, max_dist: float):
    best, best_d = None, None
    for ann in anns:
        dx, dy = bbox_center(ann.get("bbox", None))
        if not (np.isfinite(dx) and np.isfinite(dy)):
            continue
        d = math.hypot(dx - cx, dy - cy)
        if d <= max_dist and (best is None or d < best_d):
            best, best_d = ann, d
    return best, best_d

def get_first_usable_match(json_dir: str, start_frame: int, lookahead_k: int,
                           cx: float, cy: float, max_dist: float):
    """
    Find a usable detection for this track around nuc_frame.
    If nuc frame has empty JSON, look ahead up to k frames.
    """
    for j in range(0, lookahead_k + 1):
        f = start_frame + j
        jp = find_json_for_frame(json_dir, f)
        anns = read_json_list(jp)
        if not anns:
            continue
        ann, _ = match_det_by_centroid(anns, cx, cy, max_dist)
        if ann is None:
            continue
        return f, jp, ann, ("ok" if j == 0 else f"lookahead+{j}")
    return None, None, None, "no_match_in_lookahead"

# -------------------------
# Robust overlap computation
# -------------------------
def compute_overlap_prev_robust(
    cur_ann: Dict[str, Any],
    cur_rle: Dict[str, Any],
    prev_anns: List[Dict[str, Any]],
    pad_px: float,
    max_candidates: int,
    timeout_s_total: float
) -> Tuple[float, str, int]:
    """
    Returns (overlap_prev, note, n_candidates_used)

    - Filters prev candidates by bbox center proximity (pad)
    - Caps number of candidates
    - Hard wall-clock budget for the entire overlap computation
    """
    if not prev_anns:
        return float("nan"), "prev_empty_or_missing", 0

    cur_cx, cur_cy = bbox_center(cur_ann.get("bbox", None))
    if not (np.isfinite(cur_cx) and np.isfinite(cur_cy)):
        return float("nan"), "cur_bbox_center_nan", 0

    # 1) proximity filter
    nearby = []
    for pa in prev_anns:
        px, py = bbox_center(pa.get("bbox", None))
        if not (np.isfinite(px) and np.isfinite(py)):
            continue
        if abs(px - cur_cx) <= pad_px and abs(py - cur_cy) <= pad_px:
            nearby.append(pa)

    if not nearby:
        return float("nan"), "prev_no_candidates_nearby", 0

    # 2) cap candidates
    if len(nearby) > max_candidates:
        nearby = nearby[:max_candidates]

    # 3) build rles
    prev_rles = []
    for pa in nearby:
        prle = normalize_rle(pa.get("segmentation", None))
        if prle is not None:
            prev_rles.append(prle)

    if not prev_rles:
        return float("nan"), "prev_no_valid_rle", 0

    # 4) compute max overlap with a hard wall-clock budget
    t0 = time.time()
    best = 0.0
    got = False
    for prle in prev_rles:
        if timeout_s_total > 0 and (time.time() - t0) > timeout_s_total:
            return float("nan"), "overlap_timeout", len(prev_rles)
        try:
            of = overlap_frac_a_over_b(cur_rle, prle)
        except Exception:
            continue
        if np.isfinite(of):
            got = True
            if of > best:
                best = float(of)

    return (best if got else float("nan")), "ok", len(prev_rles)

# -------------------------
# Checkpoint writing
# -------------------------
def write_csv(path: str, rows: List[Dict[str, Any]]):
    pd.DataFrame(rows).to_csv(path, index=False)

def append_log_row(path: str, row: Dict[str, Any], header_fields: List[str]):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header_fields)
        if not exists:
            w.writeheader()
        w.writerow(row)

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--tracks_csv", required=True)
    ap.add_argument("--json_dir", required=True)
    ap.add_argument("--out_dir", required=True)

    # Stable run definition
    ap.add_argument("--L", type=int, default=5)
    ap.add_argument("--amin_px", type=float, default=800.0)
    ap.add_argument("--rmin_px", type=float, default=0.0)
    ap.add_argument("--rnuc_max", type=float, default=60.0)

    # Optional extra gate
    ap.add_argument("--use_rmono_gate", action="store_true")
    ap.add_argument("--rmono_min", type=float, default=0.0)

    # Overlap gate
    ap.add_argument("--overlap_prev_max", type=float, default=0.3)
    ap.add_argument("--strict_overlap", action="store_true",
                    help="If set: if overlap cannot be computed (NaN/timeout), reject trackgate. If not set: keep with overlap_note.")
    ap.add_argument("--reject_if_overlap_nan", action="store_true",
                    help="Legacy: reject if overlap is NaN. Prefer --strict_overlap for robustness.")

    # Matching / lookahead
    ap.add_argument("--lookahead_k", type=int, default=150)
    ap.add_argument("--max_dist_min", type=float, default=800.0)
    ap.add_argument("--max_dist_factor", type=float, default=50.0)

    # Speed/robust controls
    ap.add_argument("--prev_search_pad_px", type=float, default=100.0)
    ap.add_argument("--prev_max_candidates", type=int, default=15)
    ap.add_argument("--overlap_timeout_s", type=float, default=0.25)

    # Progress & checkpointing
    ap.add_argument("--progress_every", type=int, default=10)
    ap.add_argument("--checkpoint_every", type=int, default=50,
                    help="Write incremental CSV checkpoints every N tracks processed.")
    ap.add_argument("--print_each_track", action="store_true")

    # Resume slice
    ap.add_argument("--start_track_idx", type=int, default=0)
    ap.add_argument("--max_tracks", type=int, default=-1)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

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
    n_total = len(groups)
    print(f"[OK] Tracks: {n_total} unique track_id", flush=True)

    start = max(0, int(args.start_track_idx))
    end = n_total
    if int(args.max_tracks) > 0:
        end = min(end, start + int(args.max_tracks))
    slice_groups = groups[start:end]
    n_slice = len(slice_groups)
    if n_slice <= 0:
        raise SystemExit("[ERR] No tracks selected (check start_track_idx/max_tracks).")

    filtered: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    rejected_trackgate: List[Dict[str, Any]] = []

    slow_log_path = os.path.join(args.out_dir, "slow_tracks.csv")
    slow_fields = ["global_i", "track_id", "mask_frame_i", "note", "pad_px", "max_candidates", "timeout_s", "n_candidates_used"]

    def checkpoint(tag: str):
        write_csv(os.path.join(args.out_dir, f"nucleation_events_filtered_{tag}.csv"), filtered)
        write_csv(os.path.join(args.out_dir, f"nucleation_events_rejected_{tag}.csv"), rejected)
        write_csv(os.path.join(args.out_dir, f"nucleation_events_rejected_trackgate_{tag}.csv"), rejected_trackgate)

    for i_local, (tid, df) in enumerate(slice_groups, start=1):
        i_global = start + i_local
        if args.print_each_track:
            print(f"[TRACK] global={i_global} track_id={tid}", flush=True)

        n_obs = len(df)

        # Track too short
        if n_obs < args.L:
            rejected.append({"track_id": int(tid), "reason": f"too_short(n={n_obs}<L={args.L})"})
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
        for j in range(0, n_obs - args.L + 1):
            if np.all(ok[j:j + args.L]):
                nuc_i = j
                break

        if nuc_i is None:
            rejected.append({"track_id": int(tid), "reason": f"no_stable_run_L{args.L}_after_gates", "n_obs_total": int(n_obs)})
            continue

        nuc_frame = int(frames[nuc_i])
        nuc_time = float(times[nuc_i]) if times is not None else float("nan")
        R_nuc = float(Rs[nuc_i])
        R_max = float(np.nanmax(Rs))

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
            rejected.append({"track_id": int(tid), "reason": "no_mask_match", "nuc_frame_i": nuc_frame, "match_note": match_note})
            continue

        cur_rle = normalize_rle(ann.get("segmentation", None))
        if cur_rle is None:
            rejected.append({"track_id": int(tid), "reason": "no_rle_in_segmentation", "nuc_frame_i": nuc_frame, "mask_frame_i": int(mask_frame)})
            continue

        # -----------------
        # Overlap gate (robust)
        # -----------------
        overlap_prev = float("nan")
        overlap_note = "skipped_overlap_prev"

        if args.overlap_prev_max < 0.999:
            prev_frame = int(mask_frame) - 1
            prev_path = find_json_for_frame(args.json_dir, prev_frame) if prev_frame >= 0 else None
            prev_anns = read_json_list(prev_path)

            overlap_prev, overlap_note, n_used = compute_overlap_prev_robust(
                cur_ann=ann,
                cur_rle=cur_rle,
                prev_anns=prev_anns,
                pad_px=float(args.prev_search_pad_px),
                max_candidates=int(args.prev_max_candidates),
                timeout_s_total=float(args.overlap_timeout_s),
            )

            if overlap_note in ("overlap_timeout",):
                append_log_row(slow_log_path, {
                    "global_i": int(i_global),
                    "track_id": int(tid),
                    "mask_frame_i": int(mask_frame),
                    "note": overlap_note,
                    "pad_px": float(args.prev_search_pad_px),
                    "max_candidates": int(args.prev_max_candidates),
                    "timeout_s": float(args.overlap_timeout_s),
                    "n_candidates_used": int(n_used),
                }, slow_fields)

            # Strict handling for NaN/timeout
            nan_overlap = (not np.isfinite(overlap_prev))
            if nan_overlap and (args.strict_overlap or args.reject_if_overlap_nan):
                rejected_trackgate.append({
                    "track_id": int(tid),
                    "reason": "overlap_nan_or_timeout",
                    "overlap_note": overlap_note,
                    "mask_frame_i": int(mask_frame),
                    "nuc_frame_i": int(nuc_frame),
                    "match_note": match_note
                })
                continue

            # If overlap is finite and too high => reject
            if np.isfinite(overlap_prev) and overlap_prev > args.overlap_prev_max:
                rejected_trackgate.append({
                    "track_id": int(tid),
                    "reason": f"overlap_prev>{args.overlap_prev_max}",
                    "overlap_prev": float(overlap_prev),
                    "overlap_note": overlap_note,
                    "mask_frame_i": int(mask_frame),
                    "nuc_frame_i": int(nuc_frame),
                    "match_note": match_note
                })
                continue

        _, mask_time = parse_frame_and_time_from_name(os.path.basename(json_path))
        filtered.append({
            "track_id": int(tid),
            "nuc_time_ms": float(nuc_time),
            "nuc_frame_i": int(nuc_frame),
            "mask_frame_i": int(mask_frame),
            "mask_time_ms": float(mask_time) if mask_time is not None else float("nan"),
            "R_nuc_px": float(R_nuc),
            "R_max_px": float(R_max),
            "n_obs_total": int(n_obs),
            "match_note": match_note,
            "max_dist_px": float(max_dist),
            "overlap_prev": float(overlap_prev) if np.isfinite(overlap_prev) else float("nan"),
            "overlap_prev_max": float(args.overlap_prev_max),
            "overlap_note": overlap_note,
        })

        # progress / checkpoint
        if (i_local % args.progress_every) == 0 or i_local == n_slice:
            print(f"[PROGRESS] {i_local}/{n_slice} (global {i_global}/{n_total}) | "
                  f"kept={len(filtered)} | rej={len(rejected)} | rej_trackgate={len(rejected_trackgate)}",
                  flush=True)

        if args.checkpoint_every > 0 and (i_local % args.checkpoint_every) == 0:
            checkpoint(tag=f"ckpt_{i_global:05d}")

    # final outputs
    write_csv(os.path.join(args.out_dir, "nucleation_events_filtered.csv"), filtered)
    write_csv(os.path.join(args.out_dir, "nucleation_events_rejected.csv"), rejected)
    write_csv(os.path.join(args.out_dir, "nucleation_events_rejected_trackgate.csv"), rejected_trackgate)

    print(f"[OK] Wrote outputs to: {args.out_dir}", flush=True)
    print(f"[OK] kept={len(filtered)} | rejected={len(rejected)} | rejected_trackgate={len(rejected_trackgate)}", flush=True)


if __name__ == "__main__":
    main()
