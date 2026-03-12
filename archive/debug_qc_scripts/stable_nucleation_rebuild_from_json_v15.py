#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
stable_nucleation_rebuild_from_json_v15.py

Robust 2-pass nucleation rebuild:
  Pass 1 (fast): find stable nucleation candidates using centroid/bbox matching (requires auto_offset).
  Pass 2 (selective): compute overlap with previous-frame objects ONLY for found candidates.
    - Uses bbox IoU as prefilter, optional polygon IoU if shapely is available.
    - Has explicit caps (k closest, max candidates, timeouts) to prevent hangs.
  Outputs:
    nucleation_events_filtered.csv
    nucleation_events_rejected.csv
    offsets_estimated.csv
    state.json

Assumptions:
  - tracks.csv has: track_id, frame_idx, cx, cy (and optionally area_px or R_px; otherwise uses provided columns).
  - JSON files: frame_00023_t46.00ms_idmapped.json etc: list of dicts with keys:
      bbox [x,y,w,h], area_px (or area), R_px (optional), segmentation (optional), id (optional)
"""

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# --------------------------
# Optional: polygon IoU
# --------------------------
_HAS_SHAPELY = False
try:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    _HAS_SHAPELY = True
except Exception:
    _HAS_SHAPELY = False


# --------------------------
# Utilities
# --------------------------
def log(msg: str):
    print(msg, flush=True)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def bbox_center(b):
    x, y, w, h = b
    return (x + 0.5 * w, y + 0.5 * h)

def bbox_iou(b1, b2) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa1, ya1, xa2, ya2 = x1, y1, x1 + w1, y1 + h1
    xb1, yb1, xb2, yb2 = x2, y2, x2 + w2, y2 + h2
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (w1 * h1) + (w2 * h2) - inter
    return float(inter / union) if union > 0 else 0.0

def find_json_for_frame(json_dir: str, frame_idx: int) -> Optional[str]:
    # matches: frame_00023_..._idmapped.json
    pat = f"frame_{frame_idx:05d}_"
    for fn in os.listdir(json_dir):
        if fn.startswith(pat) and fn.endswith("_idmapped.json"):
            return os.path.join(json_dir, fn)
    return None

def load_json_list(path: str) -> List[dict]:
    with open(path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    # tolerate dict-wrapped formats
    if isinstance(obj, dict) and "annotations" in obj and isinstance(obj["annotations"], list):
        return obj["annotations"]
    return []

def get_area_px(a: dict) -> float:
    if "area_px" in a and a["area_px"] is not None:
        return float(a["area_px"])
    if "area" in a and a["area"] is not None:
        return float(a["area"])
    return float("nan")

def get_r_px(a: dict) -> float:
    if "R_px" in a and a["R_px"] is not None:
        return float(a["R_px"])
    # derive from area if possible: R = sqrt(area/pi)
    ap = get_area_px(a)
    if not np.isnan(ap) and ap > 0:
        return float(math.sqrt(ap / math.pi))
    return float("nan")


# --------------------------
# Offset estimation
# --------------------------
def estimate_offsets(
    tracks: pd.DataFrame,
    json_dir: str,
    min_tracks_per_frame: int,
    smooth_window: int,
    use_prev_if_empty: bool
) -> pd.DataFrame:
    """
    For each frame f, estimate dx,dy such that:
      (track_cx + dx, track_cy + dy) aligns to JSON bbox centers in frame f.
    We do robust median matching:
      dx ~ median(json_bx) - median(track_cx)
      dy ~ median(json_by) - median(track_cy)
    Only if both sides have enough points.

    Returns df with columns: frame_idx, dx, dy, n_tracks, n_json
    """
    frames = sorted(tracks["frame_idx"].unique().tolist())
    rows = []
    prev_dx, prev_dy = 0.0, 0.0
    for f in frames:
        sub = tracks[tracks["frame_idx"] == f]
        if len(sub) < min_tracks_per_frame:
            if use_prev_if_empty:
                rows.append((f, prev_dx, prev_dy, len(sub), 0, "reuse_prev_low_tracks"))
            else:
                rows.append((f, np.nan, np.nan, len(sub), 0, "skip_low_tracks"))
            continue

        jpath = find_json_for_frame(json_dir, f)
        if not jpath:
            if use_prev_if_empty:
                rows.append((f, prev_dx, prev_dy, len(sub), 0, "reuse_prev_missing_json"))
            else:
                rows.append((f, np.nan, np.nan, len(sub), 0, "skip_missing_json"))
            continue

        ann = load_json_list(jpath)
        if len(ann) < 1:
            if use_prev_if_empty:
                rows.append((f, prev_dx, prev_dy, len(sub), 0, "reuse_prev_empty_json"))
            else:
                rows.append((f, np.nan, np.nan, len(sub), 0, "skip_empty_json"))
            continue

        bx = []
        by = []
        for a in ann:
            if "bbox" in a and isinstance(a["bbox"], list) and len(a["bbox"]) == 4:
                cx, cy = bbox_center(a["bbox"])
                bx.append(cx)
                by.append(cy)
        if len(bx) < 1:
            if use_prev_if_empty:
                rows.append((f, prev_dx, prev_dy, len(sub), 0, "reuse_prev_no_bbox"))
            else:
                rows.append((f, np.nan, np.nan, len(sub), 0, "skip_no_bbox"))
            continue

        dx = float(np.median(bx) - np.median(sub["cx"].astype(float).values))
        dy = float(np.median(by) - np.median(sub["cy"].astype(float).values))
        prev_dx, prev_dy = dx, dy
        rows.append((f, dx, dy, len(sub), len(bx), "ok"))

    df = pd.DataFrame(rows, columns=["frame_idx", "dx", "dy", "n_tracks", "n_json", "note"])

    # smooth dx/dy
    if smooth_window and smooth_window > 1:
        w = int(smooth_window)
        df["dx_smooth"] = df["dx"].rolling(w, center=True, min_periods=max(3, w//3)).median()
        df["dy_smooth"] = df["dy"].rolling(w, center=True, min_periods=max(3, w//3)).median()
        # if smoothing produced NaN, fall back to raw
        df["dx_smooth"] = df["dx_smooth"].fillna(df["dx"])
        df["dy_smooth"] = df["dy_smooth"].fillna(df["dy"])
    else:
        df["dx_smooth"] = df["dx"]
        df["dy_smooth"] = df["dy"]

    return df


# --------------------------
# Candidate matching
# --------------------------
def candidate_radius(max_dist_min: float, max_dist_factor: float, r_px_at_frame: float) -> float:
    if np.isnan(r_px_at_frame):
        return float(max_dist_min)
    return float(max(max_dist_min, max_dist_factor * r_px_at_frame))

def choose_candidates_near(
    ann: List[dict],
    tx: float,
    ty: float,
    max_dist: float,
    pad_px: float,
    k_closest: int,
    max_candidates: int
) -> List[dict]:
    """
    Fast prefilter: bbox center within (max_dist + pad_px)
    Then pick k closest by center distance (or cap by max_candidates).
    """
    R = float(max_dist + pad_px)
    R2 = R * R
    cand = []
    for a in ann:
        b = a.get("bbox", None)
        if not (isinstance(b, list) and len(b) == 4):
            continue
        bx, by = bbox_center(b)
        dx, dy = bx - tx, by - ty
        d2 = dx*dx + dy*dy
        if d2 <= R2:
            cand.append((d2, a))
    cand.sort(key=lambda x: x[0])
    out = [a for _, a in cand[:max(max_candidates, k_closest)]]
    if k_closest and k_closest > 0:
        out = out[:k_closest] if len(out) > k_closest else out
    if max_candidates and len(out) > max_candidates:
        out = out[:max_candidates]
    return out


# --------------------------
# Overlap computations
# --------------------------
def segmentation_to_polygons(seg) -> List["Polygon"]:
    """
    COCO polygon format: segmentation = [ [x1,y1,x2,y2,...], [...], ... ]
    """
    polys = []
    if not _HAS_SHAPELY:
        return polys
    if not isinstance(seg, list):
        return polys
    # sometimes seg is list of lists; sometimes it is a single list
    chunks = seg if (len(seg) > 0 and isinstance(seg[0], list)) else [seg]
    for pts in chunks:
        if not isinstance(pts, list) or len(pts) < 6:
            continue
        xy = list(zip(pts[0::2], pts[1::2]))
        try:
            p = Polygon(xy)
            if p.is_valid and p.area > 0:
                polys.append(p)
        except Exception:
            continue
    return polys

def polygon_iou(a: dict, b: dict) -> Optional[float]:
    if not _HAS_SHAPELY:
        return None
    sa = a.get("segmentation", None)
    sb = b.get("segmentation", None)
    if sa is None or sb is None:
        return None
    pa = segmentation_to_polygons(sa)
    pb = segmentation_to_polygons(sb)
    if not pa or not pb:
        return None
    try:
        ua = unary_union(pa)
        ub = unary_union(pb)
        inter = ua.intersection(ub).area
        union = ua.union(ub).area
        return float(inter / union) if union > 0 else 0.0
    except Exception:
        return None

def max_prev_overlap(
    nuc_obj: dict,
    prev_candidates: List[dict],
    per_call_timeout_s: float,
    strategy: str = "bbox_then_poly"
) -> Tuple[float, str]:
    """
    Returns: (max_overlap, note)
    overlap is IoU-like [0..1]
    """
    t0 = time.time()
    maxv = 0.0
    tried = 0
    # Always do bbox IoU (cheap) first
    nb = nuc_obj.get("bbox", None)
    if not (isinstance(nb, list) and len(nb) == 4):
        return float("nan"), "nuc_missing_bbox"

    for p in prev_candidates:
        if per_call_timeout_s and (time.time() - t0) > per_call_timeout_s:
            return float("nan"), f"timeout_after_{tried}"
        pb = p.get("bbox", None)
        if not (isinstance(pb, list) and len(pb) == 4):
            continue
        tried += 1
        bi = bbox_iou(nb, pb)
        if bi > maxv:
            maxv = bi

    # Optional refine with polygon IoU (robust) if available
    if strategy in ("bbox_then_poly", "poly") and _HAS_SHAPELY:
        # refine only if bbox overlap suggests potential overlap (saves time)
        # if bbox IoU already ~0, polygon IoU will be 0 too.
        if maxv > 0.01 or strategy == "poly":
            maxpoly = 0.0
            refined = 0
            for p in prev_candidates:
                if per_call_timeout_s and (time.time() - t0) > per_call_timeout_s:
                    return float("nan"), f"timeout_poly_after_{refined}"
                pi = polygon_iou(nuc_obj, p)
                if pi is None:
                    continue
                refined += 1
                if pi > maxpoly:
                    maxpoly = pi
            return maxpoly, f"poly_maxiou(n={refined})(max={maxpoly:.3f})"

    return maxv, f"bbox_maxiou(n={tried})(max={maxv:.3f})"


# --------------------------
# Core algorithm
# --------------------------
@dataclass
class Event:
    track_id: int
    nuc_frame_i: Optional[int]
    nuc_time_ms: Optional[float]
    n_obs_total: int
    reason: str
    overlap_prev: Optional[float] = None
    overlap_note: Optional[str] = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks_csv", required=True)
    ap.add_argument("--json_dir", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--L", type=int, default=5)
    ap.add_argument("--amin_px", type=float, default=800.0)
    ap.add_argument("--rnuc_max", type=float, default=60.0)

    ap.add_argument("--lookahead_k", type=int, default=150)

    # matching distance
    ap.add_argument("--max_dist_min", type=float, default=800.0)
    ap.add_argument("--max_dist_factor", type=float, default=50.0)
    ap.add_argument("--prev_search_pad_px", type=float, default=250.0)
    ap.add_argument("--k_closest", type=int, default=5)
    ap.add_argument("--prev_max_candidates", type=int, default=50)

    # overlap gating
    ap.add_argument("--strict_overlap", action="store_true")
    ap.add_argument("--overlap_prev_max", type=float, default=0.3)
    ap.add_argument("--overlap_unknown_policy", choices=["reject", "keep"], default="reject")
    ap.add_argument("--overlap_strategy", choices=["bbox", "bbox_then_poly", "poly"], default="bbox_then_poly")
    ap.add_argument("--overlap_per_call_timeout_s", type=float, default=0.30)

    # offsets
    ap.add_argument("--auto_offset", action="store_true")
    ap.add_argument("--offset_min_tracks", type=int, default=20)
    ap.add_argument("--offset_smooth_window", type=int, default=21)
    ap.add_argument("--offset_use_prev_if_empty", action="store_true")

    # progress + checkpoints
    ap.add_argument("--progress_every", type=int, default=25)
    ap.add_argument("--checkpoint_every", type=int, default=100)

    args = ap.parse_args()

    ensure_dir(args.out_dir)

    log(f"[OK] Reading tracks: {args.tracks_csv}")
    tracks = pd.read_csv(args.tracks_csv)

    # normalize required columns
    # require: track_id, frame_idx, cx, cy
    required = ["track_id", "frame_idx", "cx", "cy"]
    for c in required:
        if c not in tracks.columns:
            raise ValueError(f"tracks.csv missing required column: {c}")

    # time column optional
    has_time = "time_ms" in tracks.columns

    # R column optional
    has_r = "R_px" in tracks.columns
    has_area = "area_px" in tracks.columns

    offsets = None
    if args.auto_offset:
        log("[OK] Estimating per-frame offsets (tracks vs JSON bbox centers)...")
        offsets = estimate_offsets(
            tracks=tracks,
            json_dir=args.json_dir,
            min_tracks_per_frame=args.offset_min_tracks,
            smooth_window=args.offset_smooth_window,
            use_prev_if_empty=args.offset_use_prev_if_empty,
        )
        offsets_path = os.path.join(args.out_dir, "offsets_estimated.csv")
        offsets.to_csv(offsets_path, index=False)
        mdx = float(np.median(offsets["dx_smooth"].dropna().values)) if offsets["dx_smooth"].notna().any() else float("nan")
        mdy = float(np.median(offsets["dy_smooth"].dropna().values)) if offsets["dy_smooth"].notna().any() else float("nan")
        log(f"[OK] Offset estimation done. median dx={mdx:.2f}, dy={mdy:.2f}. Saved: {offsets_path}")

        # quick dict for lookup
        off_map = {int(r.frame_idx): (float(r.dx_smooth), float(r.dy_smooth)) for r in offsets.itertuples()}
    else:
        off_map = {}

    track_ids = sorted(tracks["track_id"].unique().tolist())
    log(f"[OK] Tracks: {len(track_ids)} unique track_id")

    kept: List[Event] = []
    rej: List[Event] = []

    def write_outputs(suffix: str = ""):
        f1 = os.path.join(args.out_dir, f"nucleation_events_filtered{suffix}.csv")
        f2 = os.path.join(args.out_dir, f"nucleation_events_rejected{suffix}.csv")
        pd.DataFrame([asdict(e) for e in kept]).to_csv(f1, index=False)
        pd.DataFrame([asdict(e) for e in rej]).to_csv(f2, index=False)

        state = {
            "kept": len(kept),
            "rejected": len(rej),
            "args": vars(args),
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "has_shapely": _HAS_SHAPELY,
        }
        with open(os.path.join(args.out_dir, "state.json"), "w") as f:
            json.dump(state, f, indent=2)

    # group per track for speed
    tracks_g = tracks.groupby("track_id", sort=False)

    for i, tid in enumerate(track_ids, start=1):
        sub = tracks_g.get_group(tid).sort_values("frame_idx")
        n_obs = len(sub)

        # Pass 1: find first frame where (area>=amin and R<=rnuc_max) and stable run length >= L
        # If track lacks area/R, we skip gating and just use frames as-is (but keep reason).
        frames = sub["frame_idx"].astype(int).values
        cx = sub["cx"].astype(float).values
        cy = sub["cy"].astype(float).values

        if has_time:
            tm = sub["time_ms"].astype(float).values
        else:
            tm = np.full_like(cx, np.nan, dtype=float)

        if has_area:
            area = sub["area_px"].astype(float).values
        else:
            area = np.full_like(cx, np.nan, dtype=float)

        if has_r:
            rpx = sub["R_px"].astype(float).values
        else:
            # derive from area if present
            rpx = np.where(np.isfinite(area) & (area > 0), np.sqrt(area / math.pi), np.nan)

        gate_ok = np.ones(len(frames), dtype=bool)
        if np.isfinite(args.amin_px):
            gate_ok &= (np.isfinite(area) & (area >= args.amin_px)) if has_area else gate_ok
        if np.isfinite(args.rnuc_max):
            gate_ok &= (np.isfinite(rpx) & (rpx <= args.rnuc_max)) if np.isfinite(rpx).any() else gate_ok

        # find stable run: L consecutive frames with gate_ok True
        nuc_idx = None
        if args.L <= 1:
            idxs = np.where(gate_ok)[0]
            nuc_idx = int(idxs[0]) if len(idxs) else None
        else:
            run = 0
            for k in range(len(gate_ok)):
                run = run + 1 if gate_ok[k] else 0
                if run >= args.L:
                    nuc_idx = k - (args.L - 1)
                    break

        if nuc_idx is None:
            rej.append(Event(
                track_id=int(tid),
                nuc_frame_i=None,
                nuc_time_ms=None,
                n_obs_total=int(n_obs),
                reason=f"trackgate_no_frame_passes_area_rnuc" if (has_area or np.isfinite(rpx).any()) else "no_stable_run",
            ))
        else:
            nuc_frame = int(frames[nuc_idx])
            nuc_time = float(tm[nuc_idx]) if has_time and np.isfinite(tm[nuc_idx]) else None

            # Pass 2: overlap check with prev frame (strict_overlap only)
            overlap_prev = None
            overlap_note = None

            if args.strict_overlap:
                prev_frame = nuc_frame - 1
                prev_json = find_json_for_frame(args.json_dir, prev_frame)
                nuc_json = find_json_for_frame(args.json_dir, nuc_frame)

                # if missing prev/nuc JSON:
                if not prev_json:
                    if args.overlap_unknown_policy == "reject":
                        rej.append(Event(int(tid), nuc_frame, nuc_time, int(n_obs), "missing_prev_json",
                                         overlap_prev=None, overlap_note="missing_prev_json"))
                    else:
                        kept.append(Event(int(tid), nuc_frame, nuc_time, int(n_obs), "ok_keep_unknown",
                                          overlap_prev=None, overlap_note="missing_prev_json_keep"))
                elif not nuc_json:
                    if args.overlap_unknown_policy == "reject":
                        rej.append(Event(int(tid), nuc_frame, nuc_time, int(n_obs), "missing_nuc_json",
                                         overlap_prev=None, overlap_note="missing_nuc_json"))
                    else:
                        kept.append(Event(int(tid), nuc_frame, nuc_time, int(n_obs), "ok_keep_unknown",
                                          overlap_prev=None, overlap_note="missing_nuc_json_keep"))
                else:
                    prev_ann = load_json_list(prev_json)
                    nuc_ann = load_json_list(nuc_json)

                    # if prev frame empty:
                    if len(prev_ann) == 0:
                        kept.append(Event(int(tid), nuc_frame, nuc_time, int(n_obs), "ok",
                                          overlap_prev=0.0, overlap_note="prev_empty_assumed_0"))
                    else:
                        # match nuc object near track centroid (with offset)
                        tx = float(cx[nuc_idx])
                        ty = float(cy[nuc_idx])
                        if args.auto_offset and nuc_frame in off_map:
                            dx, dy = off_map[nuc_frame]
                            tx += dx
                            ty += dy

                        # pick nuc candidates from nuc frame near (tx,ty)
                        maxd = candidate_radius(args.max_dist_min, args.max_dist_factor, float(rpx[nuc_idx]) if np.isfinite(rpx[nuc_idx]) else float("nan"))
                        nuc_cand = choose_candidates_near(
                            ann=nuc_ann,
                            tx=tx, ty=ty,
                            max_dist=maxd,
                            pad_px=args.prev_search_pad_px,
                            k_closest=1,              # pick the single best nuc obj
                            max_candidates=10
                        )
                        if len(nuc_cand) == 0:
                            rej.append(Event(int(tid), nuc_frame, nuc_time, int(n_obs), "no_mask_match",
                                             overlap_prev=None, overlap_note="no_nuc_candidate_near_track"))
                        else:
                            nuc_obj = nuc_cand[0]

                            # find prev candidates near same transformed centroid (use prev frame offset too)
                            ptx, pty = tx, ty
                            if args.auto_offset and prev_frame in off_map:
                                dxp, dyp = off_map[prev_frame]
                                # tx,ty already includes nuc_frame offset; adjust to prev offset:
                                # simplest robust: rebase from original track coords
                                ptx = float(cx[nuc_idx]) + dxp
                                pty = float(cy[nuc_idx]) + dyp

                            prev_cand = choose_candidates_near(
                                ann=prev_ann,
                                tx=ptx, ty=pty,
                                max_dist=maxd,
                                pad_px=args.prev_search_pad_px,
                                k_closest=args.k_closest,
                                max_candidates=args.prev_max_candidates,
                            )

                            if len(prev_cand) == 0:
                                kept.append(Event(int(tid), nuc_frame, nuc_time, int(n_obs), "ok",
                                                  overlap_prev=0.0, overlap_note="prev_nocand_assumed_0"))
                            else:
                                ov, note = max_prev_overlap(
                                    nuc_obj=nuc_obj,
                                    prev_candidates=prev_cand,
                                    per_call_timeout_s=args.overlap_per_call_timeout_s,
                                    strategy=args.overlap_strategy
                                )
                                overlap_prev = None if (ov is None or np.isnan(ov)) else float(ov)
                                overlap_note = note

                                if overlap_prev is None:
                                    if args.overlap_unknown_policy == "reject":
                                        rej.append(Event(int(tid), nuc_frame, nuc_time, int(n_obs),
                                                         "overlap_unknown", overlap_prev=None, overlap_note=note))
                                    else:
                                        kept.append(Event(int(tid), nuc_frame, nuc_time, int(n_obs),
                                                          "ok_keep_unknown", overlap_prev=None, overlap_note=note))
                                else:
                                    if overlap_prev > args.overlap_prev_max:
                                        rej.append(Event(int(tid), nuc_frame, nuc_time, int(n_obs),
                                                         f"overlap_prev>{args.overlap_prev_max}",
                                                         overlap_prev=overlap_prev, overlap_note=note))
                                    else:
                                        kept.append(Event(int(tid), nuc_frame, nuc_time, int(n_obs),
                                                          "ok", overlap_prev=overlap_prev, overlap_note=note))
            else:
                kept.append(Event(int(tid), nuc_frame, nuc_time, int(n_obs), "ok_no_overlap_gate"))

        if (args.progress_every and (i % args.progress_every == 0)) or (i == len(track_ids)):
            log(f"[PROGRESS] {i}/{len(track_ids)} | kept={len(kept)} | rej={len(rej)}")

        if args.checkpoint_every and (i % args.checkpoint_every == 0):
            write_outputs(suffix=f"_ckpt_{i:05d}")

    write_outputs()
    log(f"[OK] Wrote outputs to: {args.out_dir}")
    log(f"[OK] kept={len(kept)} | rejected={len(rej)} | has_shapely={_HAS_SHAPELY}")


if __name__ == "__main__":
    main()