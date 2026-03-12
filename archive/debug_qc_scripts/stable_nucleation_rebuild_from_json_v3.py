#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stable nucleation rebuild + overlap gate (TEMPO/FAPI) — v4 (rewritten)

This version fixes the two issues you hit:
  1) Frame JSONs are LISTS of COCO-style annotations (not dicts).
  2) Many early frames are empty lists [] (2 bytes). Lookahead must be bounded
     and must not infinite-loop. We now use a strict for-loop for lookahead.

It also:
  - prints progress so you can see it's alive
  - ALWAYS writes output CSVs (even if empty) so you don't get "nothing happened"
  - supports centroid matching via bbox-center (recommended for TEMPO)
  - computes overlap_prev using pycocotools RLE ops when possible
  - optionally rejects NaN overlaps with --reject_if_overlap_nan

Inputs:
  --tracks_csv : tracks.csv from kinetics pipeline (must include track_id, frame_idx, cx, cy, area_px, R_px;
                 optionally t_ms and R_mono)
  --json_dir   : folder containing per-frame frame_00018_t36.00ms_idmapped.json (each is a LIST of anns)
  --out_dir    : output folder

Outputs (in out_dir):
  nucleation_events_filtered.csv
  nucleation_events_rejected.csv
  nucleation_events_rejected_trackgate.csv
"""

import os
import re
import glob
import json
import math
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

try:
    from pycocotools import mask as mask_utils
except Exception as e:
    raise SystemExit(
        "[ERR] pycocotools not available. Install it in your env: "
        "conda install -c conda-forge pycocotools  (or pip install pycocotools)"
    )

# -------------------------
# File naming helpers
# -------------------------

_FRAME_RE = re.compile(r"frame_(\d+)_t([0-9.]+)ms", re.IGNORECASE)

def parse_frame_from_name(fname: str) -> Tuple[Optional[int], Optional[float]]:
    """Parse frame_idx and time_ms from filenames like frame_00018_t36.00ms_idmapped.json"""
    m = _FRAME_RE.search(fname or "")
    if not m:
        return None, None
    return int(m.group(1)), float(m.group(2))

def find_json_path(json_dir: str, frame_idx: int, t_ms: Optional[float]) -> Optional[str]:
    """Prefer exact match when t_ms is known; otherwise fall back by glob on index."""
    if frame_idx is None:
        return None
    # exact
    if t_ms is not None:
        exact = os.path.join(json_dir, f"frame_{frame_idx:05d}_t{float(t_ms):.2f}ms_idmapped.json")
        if os.path.exists(exact):
            return exact
    patt = os.path.join(json_dir, f"frame_{frame_idx:05d}_t*ms_idmapped.json")
    cands = sorted(glob.glob(patt))
    if not cands:
        return None
    if t_ms is None or len(cands) == 1:
        return cands[0]
    # choose closest time
    best = cands[0]
    best_dt = float("inf")
    for p in cands:
        _, tt = parse_frame_from_name(os.path.basename(p))
        if tt is None:
            continue
        dt = abs(tt - float(t_ms))
        if dt < best_dt:
            best = p
            best_dt = dt
    return best

# -------------------------
# JSON + RLE helpers
# -------------------------

def load_frame_anns(json_path: str) -> List[Dict[str, Any]]:
    """
    Frame JSONs for this pipeline are LISTs of dict annotations (COCO-style).
    Empty frames are [].
    """
    if json_path is None or (not os.path.exists(json_path)):
        return []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return []
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    # tolerate older schema
    if isinstance(obj, dict):
        if "detections" in obj and isinstance(obj["detections"], list):
            return [x for x in obj["detections"] if isinstance(x, dict)]
        if "annotations" in obj and isinstance(obj["annotations"], list):
            return [x for x in obj["annotations"] if isinstance(x, dict)]
    return []

def _ensure_rle_counts_bytes(rle: Dict[str, Any]) -> Dict[str, Any]:
    """pycocotools wants rle['counts'] as bytes in some cases."""
    if not isinstance(rle, dict):
        raise ValueError("RLE must be dict")
    if "counts" not in rle or "size" not in rle:
        raise ValueError("RLE missing counts/size")
    out = dict(rle)
    c = out["counts"]
    if isinstance(c, str):
        out["counts"] = c.encode("utf-8")
    return out

def ann_to_rle(ann: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    segmentation can be:
      - RLE dict {'counts':..., 'size':[H,W]}
      - polygons list (rare here; we'd need image size; skip if so)
    """
    seg = ann.get("segmentation", None)
    if seg is None:
        return None
    if isinstance(seg, dict) and "counts" in seg and "size" in seg:
        try:
            return _ensure_rle_counts_bytes(seg)
        except Exception:
            return None
    # polygons: unsupported without image size
    return None

def overlap_frac_rle(rle_a: Dict[str, Any], rle_b: Dict[str, Any]) -> float:
    """
    Fraction of A overlapping B: area(intersect(A,B)) / area(A).
    """
    inter = mask_utils.merge([rle_a, rle_b], intersect=True)
    ai = float(mask_utils.area(inter))
    aa = float(mask_utils.area(rle_a))
    if aa <= 0:
        return float("nan")
    return ai / aa

# -------------------------
# Matching helpers
# -------------------------

def bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """COCO bbox = [x,y,w,h]"""
    if not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
        return float("nan"), float("nan")
    x, y, w, h = bbox[:4]
    return float(x + 0.5*w), float(y + 0.5*h)

def match_det_by_centroid(
    anns: List[Dict[str, Any]],
    cx: float,
    cy: float,
    max_dist: float
) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    """
    Return (best_ann, best_dist). Best = smallest distance to bbox-center.
    """
    best = None
    best_d = None
    for ann in anns:
        bx = ann.get("bbox", None)
        dx, dy = bbox_center(bx)
        if not (np.isfinite(dx) and np.isfinite(dy)):
            continue
        d = math.hypot(dx - cx, dy - cy)
        if d <= max_dist and (best is None or d < best_d):
            best = ann
            best_d = d
    return best, best_d

def get_first_usable_frame_and_match(
    json_dir: str,
    start_frame: int,
    start_t_ms: Optional[float],
    lookahead_k: int,
    cx: float,
    cy: float,
    max_dist: float
) -> Tuple[Optional[int], Optional[float], Optional[Dict[str, Any]], str]:
    """
    BOUNDED lookahead: tries start_frame, start_frame+1, ..., start_frame+k.
    Returns (frame_idx, time_ms, matched_ann, note)
    """
    for j in range(0, lookahead_k + 1):
        f = start_frame + j
        p = find_json_path(json_dir, f, None if j > 0 else start_t_ms)
        anns = load_frame_anns(p)
        if not anns:
            continue
        ann, _ = match_det_by_centroid(anns, cx, cy, max_dist)
        if ann is None:
            continue
        _, tt = parse_frame_from_name(os.path.basename(p)) if p else (None, None)
        note = "ok" if j == 0 else f"lookahead+{j}"
        return f, tt, ann, note
    return None, None, None, "no_match_in_lookahead"

# -------------------------
# I/O helpers
# -------------------------

def ensure_out_dir(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

def write_outputs(out_dir: str, filtered: List[Dict[str, Any]], rejected: List[Dict[str, Any]], rejected_trackgate: List[Dict[str, Any]]) -> None:
    ensure_out_dir(out_dir)
    pd.DataFrame(filtered).to_csv(os.path.join(out_dir, "nucleation_events_filtered.csv"), index=False)
    pd.DataFrame(rejected).to_csv(os.path.join(out_dir, "nucleation_events_rejected.csv"), index=False)
    pd.DataFrame(rejected_trackgate).to_csv(os.path.join(out_dir, "nucleation_events_rejected_trackgate.csv"), index=False)

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

    ap.add_argument("--use_rmono_gate", action="store_true")
    ap.add_argument("--rmono_min", type=float, default=0.0)

    ap.add_argument("--rnuc_max", type=float, default=60.0)

    ap.add_argument("--overlap_prev_max", type=float, default=0.3)
    ap.add_argument("--prefer_centroid_match", action="store_true")  # kept for CLI compatibility

    ap.add_argument("--lookahead_k", type=int, default=3)

    ap.add_argument("--max_dist_min", type=float, default=25.0)
    ap.add_argument("--max_dist_factor", type=float, default=2.0)

    ap.add_argument("--reject_if_overlap_nan", action="store_true")

    args = ap.parse_args()

    print(f"[OK] Reading tracks: {args.tracks_csv}", flush=True)
    tracks = pd.read_csv(args.tracks_csv)

    # Required columns
    if "track_id" not in tracks.columns:
        raise SystemExit("[ERR] tracks.csv missing required column: track_id")

    # Frame column
    frame_col = None
    for c in ["frame_idx", "frame", "frame_i", "frame_id"]:
        if c in tracks.columns:
            frame_col = c
            break
    if frame_col is None:
        raise SystemExit("[ERR] tracks.csv missing a frame column (expected one of: frame_idx, frame, frame_i, frame_id)")

    # Time column (optional)
    t_col = None
    for c in ["t_ms", "time_ms", "time", "t"]:
        if c in tracks.columns:
            t_col = c
            break

    # Centroid columns
    cx_col = None
    cy_col = None
    for c in ["cx", "cx_px", "x", "x_px"]:
        if c in tracks.columns:
            cx_col = c
            break
    for c in ["cy", "cy_px", "y", "y_px"]:
        if c in tracks.columns:
            cy_col = c
            break
    if cx_col is None or cy_col is None:
        raise SystemExit("[ERR] tracks.csv missing centroid columns (need cx/cy or cx_px/cy_px).")

    # Size columns
    area_col = None
    for c in ["area_px", "area"]:
        if c in tracks.columns:
            area_col = c
            break
    if area_col is None:
        raise SystemExit("[ERR] tracks.csv missing area column (need area_px).")

    r_col = None
    for c in ["R_px", "r_px", "radius_px", "R"]:
        if c in tracks.columns:
            r_col = c
            break
    if r_col is None:
        raise SystemExit("[ERR] tracks.csv missing radius column (need R_px).")

    rmono_col = None
    for c in ["R_mono", "r_mono", "rmono"]:
        if c in tracks.columns:
            rmono_col = c
            break

    # Sort
    tracks = tracks.sort_values(["track_id", frame_col]).reset_index(drop=True)
    g = tracks.groupby("track_id", sort=False)
    n_tracks = len(g)
    print(f"[OK] Tracks: {n_tracks} unique track_id", flush=True)

    filtered_rows: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []
    rejected_trackgate_rows: List[Dict[str, Any]] = []

    for idx, (tid, df) in enumerate(g, start=1):
        frames = df[frame_col].to_numpy()
        areas = df[area_col].to_numpy(dtype=float)
        Rs = df[r_col].to_numpy(dtype=float)
        cxs = df[cx_col].to_numpy(dtype=float)
        cys = df[cy_col].to_numpy(dtype=float)
        times = df[t_col].to_numpy(dtype=float) if t_col else None

        n_obs = len(df)
        if n_obs < args.L:
            rejected_rows.append({"track_id": int(tid), "reason": f"too_short(n={n_obs}<L={args.L})"})
            continue

        ok_size = (areas >= args.amin_px) & (Rs >= args.rmin_px) & (Rs <= args.rnuc_max)
        if args.use_rmono_gate and rmono_col is not None:
            rmono = df[rmono_col].to_numpy(dtype=float)
            ok_size = ok_size & (rmono >= args.rmono_min)

        nuc_i = None
        for i in range(0, n_obs - args.L + 1):
            if np.all(ok_size[i:i+args.L]):
                nuc_i = i
                break
        if nuc_i is None:
            rejected_rows.append({
                "track_id": int(tid),
                "reason": f"no_stable_run_L{args.L}_after_gates",
                "n_obs_total": int(n_obs),
                "amin_px": args.amin_px,
                "rnuc_max": args.rnuc_max
            })
            continue

        nuc_frame = int(frames[nuc_i])
        nuc_time = float(times[nuc_i]) if times is not None else float("nan")
        R_nuc = float(Rs[nuc_i])
        area_nuc = float(areas[nuc_i])
        R_max = float(np.nanmax(Rs))

        max_dist = max(args.max_dist_min, args.max_dist_factor * max(R_nuc, 1.0))
        start_t = nuc_time if np.isfinite(nuc_time) else None

        mf, mt, ann, match_note = get_first_usable_frame_and_match(
            json_dir=args.json_dir,
            start_frame=nuc_frame,
            start_t_ms=start_t,
            lookahead_k=args.lookahead_k,
            cx=float(cxs[nuc_i]),
            cy=float(cys[nuc_i]),
            max_dist=float(max_dist)
        )

        if ann is None:
            rejected_rows.append({
                "track_id": int(tid),
                "reason": "no_mask_match",
                "nuc_frame_i": nuc_frame,
                "nuc_time_ms": nuc_time,
                "match_note": match_note,
                "lookahead_k": int(args.lookahead_k)
            })
            continue

        rle = ann_to_rle(ann)
        if rle is None:
            rejected_rows.append({
                "track_id": int(tid),
                "reason": "no_rle_in_segmentation",
                "nuc_frame_i": nuc_frame,
                "nuc_time_ms": nuc_time,
                "match_note": match_note
            })
            continue

        overlap_prev = float("nan")
        overlap_note = "ok"
        prev_path = find_json_path(args.json_dir, mf - 1, None)
        prev_anns = load_frame_anns(prev_path)
        if not prev_anns:
            overlap_note = "prev_frame_empty_or_missing"
        else:
            best = 0.0
            got_any = False
            for pann in prev_anns:
                prle = ann_to_rle(pann)
                if prle is None:
                    continue
                try:
                    of = overlap_frac_rle(rle, prle)
                except Exception:
                    continue
                if np.isfinite(of):
                    got_any = True
                    if of > best:
                        best = float(of)
            if got_any:
                overlap_prev = best
            else:
                overlap_note = "prev_frame_no_usable_rle"

        if np.isnan(overlap_prev):
            if args.reject_if_overlap_nan:
                rejected_trackgate_rows.append({
                    "track_id": int(tid),
                    "reason": "overlap_nan",
                    "nuc_frame_i": nuc_frame,
                    "mask_frame_i": mf,
                    "match_note": match_note,
                    "overlap_note": overlap_note
                })
                continue
        else:
            if overlap_prev > args.overlap_prev_max:
                rejected_trackgate_rows.append({
                    "track_id": int(tid),
                    "reason": f"overlap_prev>{args.overlap_prev_max}",
                    "nuc_frame_i": nuc_frame,
                    "mask_frame_i": mf,
                    "match_note": match_note,
                    "overlap_prev": overlap_prev,
                    "overlap_note": overlap_note
                })
                continue

        filtered_rows.append({
            "track_id": int(tid),
            "nuc_time_ms": nuc_time,
            "nuc_frame_i": nuc_frame,
            "mask_frame_i": int(mf) if mf is not None else np.nan,
            "mask_time_ms": float(mt) if mt is not None else np.nan,
            "R_nuc_px": R_nuc,
            "area_nuc_px": area_nuc,
            "R_max_px": R_max,
            "n_obs_total": int(n_obs),
            "match_note": match_note,
            "max_dist_px": float(max_dist),
            "overlap_prev": overlap_prev,
            "overlap_prev_max": float(args.overlap_prev_max),
            "overlap_note": overlap_note
        })

        if idx % 100 == 0 or idx == n_tracks:
            print(
                f"[PROGRESS] {idx}/{n_tracks} tracks | kept={len(filtered_rows)} | "
                f"rej={len(rejected_rows)} | rej_trackgate={len(rejected_trackgate_rows)}",
                flush=True
            )

    # ALWAYS write outputs
    write_outputs(args.out_dir, filtered_rows, rejected_rows, rejected_trackgate_rows)
    print(f"[OK] Wrote outputs to: {args.out_dir}", flush=True)
    print(f"[OK] kept={len(filtered_rows)} | rejected={len(rejected_rows)} | rejected_trackgate={len(rejected_trackgate_rows)}", flush=True)


if __name__ == "__main__":
    main()
