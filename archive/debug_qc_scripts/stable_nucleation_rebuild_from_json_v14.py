# stable_nucleation_rebuild_from_json_v14.py
# Robust stable nucleation rebuild from frame JSONs + tracks, with:
# - auto per-frame offset estimation (tracks vs JSON bbox centers)
# - nuc object selection via centroid match (after offset)
# - overlap gate using MAX bbox-IoU over nearby prev objects (robust, avoids "fake zeros")
# - explicit overlap status (ok / prev_empty / no_candidate / missing_prev / missing_nuc)
# - checkpointing + resumable state

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------
# Utilities
# -------------------------

def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)

def find_frame_json(json_dir: str, frame_i: int) -> Optional[str]:
    # filenames look like frame_00018_t36.00ms_idmapped.json
    patt = os.path.join(json_dir, f"frame_{frame_i:05d}_*_idmapped.json")
    hits = glob.glob(patt)
    return hits[0] if hits else None

def load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        # list of coco-like annotations
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        # sometimes wraps list; try common keys
        for k in ("annotations", "anns", "objects"):
            if k in obj and isinstance(obj[k], list):
                return [x for x in obj[k] if isinstance(x, dict)]
    return []

def bbox_center(b: List[float]) -> Tuple[float, float]:
    x, y, w, h = b
    return (x + 0.5*w, y + 0.5*h)

def bbox_iou(b1: List[float], b2: List[float]) -> float:
    # [x,y,w,h]
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
    ua = w1*h1 + w2*h2 - inter
    return float(inter / ua) if ua > 0 else 0.0

def parse_float_in_parens(s: str) -> Optional[float]:
    # e.g. bbox_fallback_ok(0.534)
    m = re.search(r"\(([0-9.]+)\)", str(s))
    return float(m.group(1)) if m else None

@dataclass
class OffsetSeries:
    dx_by_frame: Dict[int, float]
    dy_by_frame: Dict[int, float]
    dx_med: float
    dy_med: float


# -------------------------
# Offset estimation
# -------------------------

def estimate_offsets(
    tracks: pd.DataFrame,
    json_dir: str,
    min_tracks: int = 20,
    smooth_window: int = 21,
    use_prev_if_empty: bool = True,
    out_csv: Optional[str] = None,
) -> OffsetSeries:
    """
    For each frame f, compare track centroids (cx,cy) to JSON bbox centers of SAME frame f
    and estimate (dx,dy) = median(track_c - json_bbox_c) using frame-wise medians.
    Uses a rolling median smoothing.
    """
    frames = sorted(tracks["frame_idx"].unique().tolist())
    dx = {}
    dy = {}
    last_dx = None
    last_dy = None

    for f in frames:
        sub = tracks[tracks["frame_idx"] == f]
        if len(sub) < min_tracks:
            continue

        jf = find_frame_json(json_dir, int(f))
        if not jf:
            continue
        anns = load_json_list(jf)
        if len(anns) == 0:
            if use_prev_if_empty and last_dx is not None and last_dy is not None:
                dx[int(f)] = last_dx
                dy[int(f)] = last_dy
            continue

        # frame-wise medians
        tcx = float(sub["cx"].median())
        tcy = float(sub["cy"].median())
        bx = np.median([bbox_center(a["bbox"])[0] for a in anns if "bbox" in a])
        by = np.median([bbox_center(a["bbox"])[1] for a in anns if "bbox" in a])

        dx_f = float(tcx - bx)
        dy_f = float(tcy - by)
        dx[int(f)] = dx_f
        dy[int(f)] = dy_f
        last_dx, last_dy = dx_f, dy_f

    if len(dx) == 0:
        # fallback: no offsets estimated
        return OffsetSeries({}, {}, 0.0, 0.0)

    # rolling median smoothing on frame index
    f_sorted = sorted(dx.keys())
    dx_arr = np.array([dx[f] for f in f_sorted], dtype=float)
    dy_arr = np.array([dy[f] for f in f_sorted], dtype=float)

    # simple rolling median
    w = max(3, int(smooth_window) | 1)  # odd
    half = w // 2

    def roll_med(arr: np.ndarray) -> np.ndarray:
        out = np.copy(arr)
        for i in range(len(arr)):
            lo = max(0, i - half)
            hi = min(len(arr), i + half + 1)
            out[i] = float(np.median(arr[lo:hi]))
        return out

    dx_sm = roll_med(dx_arr)
    dy_sm = roll_med(dy_arr)

    dx_by_frame = {f_sorted[i]: float(dx_sm[i]) for i in range(len(f_sorted))}
    dy_by_frame = {f_sorted[i]: float(dy_sm[i]) for i in range(len(f_sorted))}
    dx_med = float(np.median(dx_sm))
    dy_med = float(np.median(dy_sm))

    if out_csv:
        ensure_dir(os.path.dirname(out_csv))
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            wri = csv.writer(f)
            wri.writerow(["frame_idx", "dx", "dy"])
            for fr in f_sorted:
                wri.writerow([fr, dx_by_frame[fr], dy_by_frame[fr]])
            wri.writerow(["MEDIAN", dx_med, dy_med])

    return OffsetSeries(dx_by_frame, dy_by_frame, dx_med, dy_med)


def get_offset_for_frame(off: OffsetSeries, frame_i: int) -> Tuple[float, float]:
    if frame_i in off.dx_by_frame and frame_i in off.dy_by_frame:
        return off.dx_by_frame[frame_i], off.dy_by_frame[frame_i]
    return off.dx_med, off.dy_med


# -------------------------
# Core logic
# -------------------------

def pick_nuc_annotation_by_centroid(
    anns: List[Dict[str, Any]],
    target_cx: float,
    target_cy: float,
    max_dist: float,
) -> Optional[Dict[str, Any]]:
    if len(anns) == 0:
        return None
    best = None
    best_d = None
    for a in anns:
        if "bbox" not in a:
            continue
        bx, by = bbox_center(a["bbox"])
        d = math.hypot(bx - target_cx, by - target_cy)
        if best_d is None or d < best_d:
            best_d = d
            best = a
    if best is None:
        return None
    if best_d is not None and best_d > max_dist:
        return None
    return best

def prev_candidates_in_window(
    prev_anns: List[Dict[str, Any]],
    cx: float,
    cy: float,
    pad: float,
) -> List[Dict[str, Any]]:
    out = []
    x0, y0 = cx - pad, cy - pad
    x1, y1 = cx + pad, cy + pad
    for a in prev_anns:
        if "bbox" not in a:
            continue
        bx, by = bbox_center(a["bbox"])
        if (x0 <= bx <= x1) and (y0 <= by <= y1):
            out.append(a)
    return out

def max_iou_with_prev(
    nuc_bbox: List[float],
    prev_anns: List[Dict[str, Any]],
    cx: float,
    cy: float,
    pad: float,
    k_closest: int = 0,
) -> Tuple[Optional[float], str, int]:
    """
    Compute MAX bbox IoU between nuc_bbox and prev bbox candidates within window around (cx,cy).
    If k_closest>0, preselect k closest by centroid distance (still maxIoU among them).
    Returns (max_iou or None, note, n_candidates)
    """
    cands = prev_candidates_in_window(prev_anns, cx, cy, pad)
    if len(cands) == 0:
        return None, "prev_nocand", 0

    if k_closest and k_closest > 0 and len(cands) > k_closest:
        # preselect by distance
        dist = []
        for a in cands:
            bx, by = bbox_center(a["bbox"])
            dist.append((math.hypot(bx - cx, by - cy), a))
        dist.sort(key=lambda t: t[0])
        cands = [a for _, a in dist[:k_closest]]

    m = 0.0
    for a in cands:
        i = bbox_iou(nuc_bbox, a["bbox"])
        if i > m:
            m = i
    return float(m), f"prev_maxiou(n={len(cands)})", len(cands)


# -------------------------
# Main
# -------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--tracks_csv", required=True)
    p.add_argument("--json_dir", required=True)
    p.add_argument("--out_dir", required=True)

    p.add_argument("--L", type=int, default=5)
    p.add_argument("--amin_px", type=float, default=800.0)
    p.add_argument("--rnuc_max", type=float, default=60.0)

    p.add_argument("--lookahead_k", type=int, default=150)

    # matching / window
    p.add_argument("--max_dist_min", type=float, default=800.0)
    p.add_argument("--max_dist_factor", type=float, default=50.0)
    p.add_argument("--prev_search_pad_px", type=float, default=250.0)
    p.add_argument("--k_closest", type=int, default=0)

    # overlap gate
    p.add_argument("--overlap_prev_max", type=float, default=0.3)
    p.add_argument("--strict_overlap", action="store_true")
    p.add_argument("--overlap_unknown_policy", choices=["keep", "reject", "assume0"], default="keep")
    p.add_argument("--accept_empty_prev", action="store_true")
    p.add_argument("--accept_no_candidate_prev", action="store_true")

    # bbox fallback for overlap if prev exists but something goes wrong
    p.add_argument("--bbox_fallback_max", type=float, default=0.7)

    # offset
    p.add_argument("--auto_offset", action="store_true")
    p.add_argument("--offset_min_tracks", type=int, default=20)
    p.add_argument("--offset_smooth_window", type=int, default=21)
    p.add_argument("--offset_use_prev_if_empty", action="store_true")

    # bookkeeping
    p.add_argument("--progress_every", type=int, default=50)
    p.add_argument("--checkpoint_every", type=int, default=100)
    p.add_argument("--start_track_idx", type=int, default=0)
    p.add_argument("--max_tracks", type=int, default=0)

    return p

def main():
    args = build_argparser().parse_args()
    out_dir = args.out_dir
    ensure_dir(out_dir)

    print(f"[OK] Reading tracks: {args.tracks_csv}")
    tracks = pd.read_csv(args.tracks_csv)

    # normalize expected columns
    # we need: track_id, frame_idx, cx, cy, area_px, R_px
    required = ["track_id", "frame_idx", "cx", "cy"]
    for c in required:
        if c not in tracks.columns:
            raise ValueError(f"tracks.csv missing column: {c}")

    # try find area & radius columns
    area_col = "area_px" if "area_px" in tracks.columns else ("area" if "area" in tracks.columns else None)
    r_col = "R_px" if "R_px" in tracks.columns else ("R" if "R" in tracks.columns else None)

    if area_col is None:
        raise ValueError("tracks.csv missing area_px (or area) column needed for amin_px gate.")
    if r_col is None:
        # if no radius, approximate from area
        tracks["_R_px_est"] = np.sqrt(np.maximum(tracks[area_col].values, 0.0) / math.pi)
        r_col = "_R_px_est"

    # offsets
    offsets = OffsetSeries({}, {}, 0.0, 0.0)
    if args.auto_offset:
        print("[OK] Estimating per-frame offsets (tracks vs JSON bbox centers)...")
        offsets_csv = os.path.join(out_dir, "offsets_estimated.csv")
        offsets = estimate_offsets(
            tracks=tracks,
            json_dir=args.json_dir,
            min_tracks=args.offset_min_tracks,
            smooth_window=args.offset_smooth_window,
            use_prev_if_empty=args.offset_use_prev_if_empty,
            out_csv=offsets_csv,
        )
        print(f"[OK] Offset estimation done. median dx={offsets.dx_med:.2f}, dy={offsets.dy_med:.2f}. Saved: {offsets_csv}")

    # group tracks
    tids = sorted(tracks["track_id"].unique().tolist())
    print(f"[OK] Tracks: {len(tids)} unique track_id")

    # outputs
    kept_rows = []
    rej_rows = []
    rej_trackgate_rows = []

    # state for resumability
    state_path = os.path.join(out_dir, "state.json")
    state = {"done": 0, "kept": 0, "rejected": 0, "rejected_trackgate": 0, "last_track_id": None}
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    t0 = time.time()

    def write_outputs(suffix: str = ""):
        f1 = os.path.join(out_dir, f"nucleation_events_filtered{suffix}.csv")
        f2 = os.path.join(out_dir, f"nucleation_events_rejected{suffix}.csv")
        f3 = os.path.join(out_dir, f"nucleation_events_rejected_trackgate{suffix}.csv")
        pd.DataFrame(kept_rows).to_csv(f1, index=False)
        pd.DataFrame(rej_rows).to_csv(f2, index=False)
        pd.DataFrame(rej_trackgate_rows).to_csv(f3, index=False)

        state.update({
            "done": len(kept_rows) + len(rej_rows) + len(rej_trackgate_rows),
            "kept": len(kept_rows),
            "rejected": len(rej_rows),
            "rejected_trackgate": len(rej_trackgate_rows),
            "last_track_id": state["last_track_id"],
            "elapsed_s": time.time() - t0,
        })
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    # main loop
    start_idx = max(0, args.start_track_idx)
    max_tracks = args.max_tracks if args.max_tracks and args.max_tracks > 0 else None

    for gi, tid in enumerate(tids[start_idx:], start=start_idx):
        if max_tracks is not None and (gi - start_idx) >= max_tracks:
            break

        state["last_track_id"] = int(tid)

        tr = tracks[tracks["track_id"] == tid].sort_values("frame_idx")
        n_obs_total = int(len(tr))

        # gate by area/radius existence: find contiguous frames meeting amin & rnuc gate
        ok = (tr[area_col].values >= args.amin_px) & (tr[r_col].values <= args.rnuc_max)
        idxs = np.where(ok)[0]
        if len(idxs) == 0:
            rej_trackgate_rows.append({
                "track_id": int(tid),
                "reason": "trackgate_no_frame_passes_area_rnuc",
                "n_obs_total": n_obs_total,
                "amin_px": args.amin_px,
                "rnuc_max": args.rnuc_max,
            })
            continue

        # find earliest run of length >= L in idxs consecutive
        run_start = None
        run_len = 1
        best_start = None
        for j in range(1, len(idxs)):
            if idxs[j] == idxs[j-1] + 1:
                run_len += 1
            else:
                if run_len >= args.L:
                    best_start = idxs[j-run_len]
                    break
                run_len = 1
        if best_start is None:
            if run_len >= args.L:
                best_start = idxs[len(idxs)-run_len]
        if best_start is None:
            rej_rows.append({
                "track_id": int(tid),
                "reason": f"no_stable_run_L{args.L}_after_gates",
                "n_obs_total": n_obs_total,
                "amin_px": args.amin_px,
                "rnuc_max": args.rnuc_max,
            })
            continue

        # nuc frame is first frame of stable run
        nuc_frame_i = int(tr.iloc[best_start]["frame_idx"])
        nuc_time_ms = float(tr.iloc[best_start]["t_ms"]) if "t_ms" in tr.columns else float("nan")
        cx_t = float(tr.iloc[best_start]["cx"])
        cy_t = float(tr.iloc[best_start]["cy"])

        # adjust centroid by offset to align with JSON coordinates
        dx, dy = get_offset_for_frame(offsets, nuc_frame_i)
        cx = cx_t - dx
        cy = cy_t - dy

        # load nuc frame JSON (with lookahead)
        nuc_path = None
        nuc_anns = []
        used_frame = None
        for k in range(0, args.lookahead_k + 1):
            f_try = nuc_frame_i + k
            jp = find_frame_json(args.json_dir, f_try)
            if not jp:
                continue
            anns = load_json_list(jp)
            if len(anns) == 0:
                continue
            nuc_path = jp
            nuc_anns = anns
            used_frame = f_try
            break

        if nuc_path is None or len(nuc_anns) == 0:
            rej_rows.append({
                "track_id": int(tid),
                "reason": "no_usable_nuc_json_in_lookahead",
                "n_obs_total": n_obs_total,
                "nuc_frame_i": nuc_frame_i,
                "lookahead_k": args.lookahead_k,
            })
            continue

        # centroid match radius
        # use R at nuc frame to scale max dist
        R_here = float(tr.iloc[best_start][r_col])
        max_dist = max(args.max_dist_min, args.max_dist_factor * R_here)

        nuc_ann = pick_nuc_annotation_by_centroid(nuc_anns, cx, cy, max_dist=max_dist)
        if nuc_ann is None:
            rej_rows.append({
                "track_id": int(tid),
                "reason": "no_mask_match",
                "n_obs_total": n_obs_total,
                "nuc_frame_i": nuc_frame_i,
                "match_note": f"centroid_match_failed(max_dist={max_dist:.1f})",
            })
            continue

        nuc_bbox = nuc_ann["bbox"]
        nuc_area = float(nuc_ann.get("area_px", nuc_ann.get("area", float("nan"))))
        nuc_R = float(nuc_ann.get("R_px", float("nan")))

        # prev overlap computation using MAX IoU over nearby prev bboxes
        prev_frame = nuc_frame_i - 1
        prev_path = find_frame_json(args.json_dir, prev_frame)

        overlap_prev = float("nan")
        overlap_note = ""
        overlap_status = ""

        if prev_path is None:
            overlap_status = "missing_prev"
            overlap_note = "missing_prev_json"
        else:
            prev_anns = load_json_list(prev_path)
            if len(prev_anns) == 0:
                overlap_status = "prev_empty"
                overlap_note = "prev_empty"
            else:
                # adjust centroid for prev frame offsets too (so the window tracks JSON space)
                dxp, dyp = get_offset_for_frame(offsets, prev_frame)
                cx_prev = cx_t - dxp
                cy_prev = cy_t - dyp

                m_iou, note, n_c = max_iou_with_prev(
                    nuc_bbox=nuc_bbox,
                    prev_anns=prev_anns,
                    cx=cx_prev,
                    cy=cy_prev,
                    pad=args.prev_search_pad_px,
                    k_closest=args.k_closest,
                )
                if m_iou is None:
                    overlap_status = "no_candidate"
                    overlap_note = "prev_nocand"
                else:
                    overlap_status = "ok"
                    overlap_prev = float(m_iou)
                    overlap_note = f"{note}(max={overlap_prev:.3f})"

        # overlap unknown handling
        is_unknown = overlap_status in ("missing_prev", "prev_empty", "no_candidate")
        if is_unknown:
            if overlap_status == "prev_empty" and args.accept_empty_prev:
                overlap_prev = 0.0
                overlap_note = "prev_empty_assumed_0"
            elif overlap_status == "no_candidate" and args.accept_no_candidate_prev:
                overlap_prev = 0.0
                overlap_note = "prev_nocand_assumed_0"
            else:
                if args.overlap_unknown_policy == "assume0":
                    overlap_prev = 0.0
                    overlap_note = f"{overlap_status}_assumed_0"
                elif args.overlap_unknown_policy == "reject" and args.strict_overlap:
                    rej_trackgate_rows.append({
                        "track_id": int(tid),
                        "reason": f"unknown_prev_overlap({overlap_status})",
                        "n_obs_total": n_obs_total,
                        "nuc_frame_i": nuc_frame_i,
                        "overlap_note": overlap_note,
                    })
                    continue
                else:
                    # keep with NaN overlap_prev
                    pass

        # strict overlap gate
        if args.strict_overlap and (not math.isnan(overlap_prev)):
            if overlap_prev > args.overlap_prev_max:
                rej_trackgate_rows.append({
                    "track_id": int(tid),
                    "reason": f"overlap_prev>{args.overlap_prev_max}",
                    "n_obs_total": n_obs_total,
                    "nuc_frame_i": nuc_frame_i,
                    "overlap_prev": overlap_prev,
                    "overlap_note": overlap_note,
                })
                continue

        kept_rows.append({
            "track_id": int(tid),
            "n_obs_total": n_obs_total,
            "nuc_frame_i": nuc_frame_i,
            "nuc_time_ms": nuc_time_ms,
            "nuc_json_frame_used": int(used_frame) if used_frame is not None else nuc_frame_i,
            "cx_track": cx_t,
            "cy_track": cy_t,
            "cx_json": cx,
            "cy_json": cy,
            "area_nuc_px": nuc_area,
            "R_nuc_px": nuc_R,
            "overlap_prev": overlap_prev,
            "overlap_note": overlap_note,
            "amin_px": args.amin_px,
            "rnuc_max": args.rnuc_max,
            "L": args.L,
            "prev_search_pad_px": args.prev_search_pad_px,
            "k_closest": args.k_closest,
        })

        # progress + checkpoints
        done = len(kept_rows) + len(rej_rows) + len(rej_trackgate_rows)
        if args.progress_every and (done % args.progress_every == 0):
            print(f"[PROGRESS] {gi+1}/{len(tids)} (global {gi+1}/{len(tids)}) | kept={len(kept_rows)} | rej={len(rej_rows)} | rej_trackgate={len(rej_trackgate_rows)}")

        if args.checkpoint_every and (done % args.checkpoint_every == 0):
            write_outputs(suffix=f"_ckpt_{done:05d}")

    # final write
    write_outputs("")
    print(f"[OK] Wrote outputs to: {out_dir}")
    print(f"[OK] kept={len(kept_rows)} | rejected={len(rej_rows)} | rejected_trackgate={len(rej_trackgate_rows)}")
    print(f"[OK] State file: {state_path}")


if __name__ == "__main__":
    main()