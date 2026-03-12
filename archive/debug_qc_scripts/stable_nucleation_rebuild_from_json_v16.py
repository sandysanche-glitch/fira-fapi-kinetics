#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stable nucleation rebuild from per-frame JSON (COCO-ish annotations) + tracks.csv.

Key ideas (robust):
- Track-gate: find first frame in each track where area_px>=amin and R_px<=rnuc_max.
- Stability: require >=L consecutive observations after nucleation frame (in tracks).
- Previous-frame overlap gate: reject if overlap(prev, nuc_obj) > overlap_prev_max.
  - Robust & fast: preselect candidates by centroid distance (k_closest) inside a pad,
    then compute bbox IoU; only if bbox IoU is "close enough" compute polygon IoU.
- Auto-offset: estimate per-frame dx,dy between track centroids and JSON bbox centers,
  then map track (cx,cy)->JSON coords as (cx - dx, cy - dy).
- Checkpointing & resume.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional shapely
HAS_SHAPELY = False
try:
    from shapely.geometry import Polygon  # type: ignore
    HAS_SHAPELY = True
except Exception:
    HAS_SHAPELY = False


# ---------------------------
# Utilities
# ---------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def find_json_for_frame(json_dir: str, frame_i: int) -> Optional[str]:
    # expects: frame_00018_t36.00ms_idmapped.json
    pat = os.path.join(json_dir, f"frame_{frame_i:05d}_*_idmapped.json")
    hits = glob.glob(pat)
    if not hits:
        return None
    # If multiple, pick first (stable order)
    hits.sort()
    return hits[0]

def bbox_center(b: List[float]) -> Tuple[float, float]:
    x, y, w, h = b
    return (x + 0.5 * w, y + 0.5 * h)

def bbox_iou(b1: List[float], b2: List[float]) -> float:
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
    a_area = w1 * h1
    b_area = w2 * h2
    union = a_area + b_area - inter
    return float(inter / union) if union > 0 else 0.0

def coco_segmentation_to_polygon(seg: Any) -> Optional["Polygon"]:
    """
    seg can be:
    - list of lists: [[x1,y1,x2,y2,...], [...]] (COCO polygon(s))
    - RLE dict (not supported here)
    """
    if not HAS_SHAPELY:
        return None
    if isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], list):
        # Use the largest ring by area (common for multipolygons/holes encoded as separate polys)
        best_poly = None
        best_area = -1.0
        for ring in seg:
            if not isinstance(ring, list) or len(ring) < 6:
                continue
            pts = [(ring[i], ring[i + 1]) for i in range(0, len(ring) - 1, 2)]
            try:
                poly = Polygon(pts)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                a = float(poly.area)
                if a > best_area:
                    best_area = a
                    best_poly = poly
            except Exception:
                continue
        return best_poly
    # unsupported (RLE etc.)
    return None

def poly_iou_from_coco(seg_a: Any, seg_b: Any) -> Optional[float]:
    if not HAS_SHAPELY:
        return None
    pa = coco_segmentation_to_polygon(seg_a)
    pb = coco_segmentation_to_polygon(seg_b)
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


# ---------------------------
# Offset estimation
# ---------------------------

@dataclass
class OffsetFrame:
    frame_idx: int
    dx: float
    dy: float
    n_tracks: int
    n_json: int

def estimate_offsets(
    tracks: pd.DataFrame,
    json_dir: str,
    min_tracks: int = 20,
    smooth_window: int = 21,
    use_prev_if_empty: bool = True,
) -> pd.DataFrame:
    """
    Compute per-frame dx,dy where:
      dx = median(track_cx) - median(json_bbox_cx)
      dy = median(track_cy) - median(json_bbox_cy)

    To map track->json coords:
      cx_json = cx_track - dx
      cy_json = cy_track - dy
    """
    frames = sorted(tracks["frame_idx"].unique().tolist())
    rows: List[OffsetFrame] = []

    last_dx, last_dy = 0.0, 0.0
    for f in frames:
        sub = tracks[tracks["frame_idx"] == f]
        if len(sub) < min_tracks:
            if use_prev_if_empty:
                rows.append(OffsetFrame(f, last_dx, last_dy, int(len(sub)), 0))
            else:
                rows.append(OffsetFrame(f, float("nan"), float("nan"), int(len(sub)), 0))
            continue

        jfn = find_json_for_frame(json_dir, int(f))
        if jfn is None:
            if use_prev_if_empty:
                rows.append(OffsetFrame(f, last_dx, last_dy, int(len(sub)), 0))
            else:
                rows.append(OffsetFrame(f, float("nan"), float("nan"), int(len(sub)), 0))
            continue

        try:
            obj = json.load(open(jfn, "r"))
        except Exception:
            obj = []

        if not isinstance(obj, list) or len(obj) == 0:
            if use_prev_if_empty:
                rows.append(OffsetFrame(f, last_dx, last_dy, int(len(sub)), 0))
            else:
                rows.append(OffsetFrame(f, float("nan"), float("nan"), int(len(sub)), 0))
            continue

        # bbox centers
        bcx = []
        bcy = []
        for a in obj:
            if isinstance(a, dict) and "bbox" in a and isinstance(a["bbox"], list) and len(a["bbox"]) == 4:
                cx, cy = bbox_center(a["bbox"])
                bcx.append(cx)
                bcy.append(cy)

        if len(bcx) == 0:
            if use_prev_if_empty:
                rows.append(OffsetFrame(f, last_dx, last_dy, int(len(sub)), 0))
            else:
                rows.append(OffsetFrame(f, float("nan"), float("nan"), int(len(sub)), 0))
            continue

        dx = float(np.median(sub["cx"].astype(float))) - float(np.median(bcx))
        dy = float(np.median(sub["cy"].astype(float))) - float(np.median(bcy))
        last_dx, last_dy = dx, dy
        rows.append(OffsetFrame(f, dx, dy, int(len(sub)), int(len(bcx))))

    df = pd.DataFrame([r.__dict__ for r in rows]).sort_values("frame_idx").reset_index(drop=True)

    # Smooth dx,dy (rolling median), keeping NaNs as-is
    w = max(3, int(smooth_window))
    if w % 2 == 0:
        w += 1
    df["dx_s"] = df["dx"].rolling(window=w, center=True, min_periods=max(3, w // 3)).median()
    df["dy_s"] = df["dy"].rolling(window=w, center=True, min_periods=max(3, w // 3)).median()

    # Fill edges
    df["dx_s"] = df["dx_s"].fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    df["dy_s"] = df["dy_s"].fillna(method="ffill").fillna(method="bfill").fillna(0.0)

    return df[["frame_idx", "dx", "dy", "dx_s", "dy_s", "n_tracks", "n_json"]]


# ---------------------------
# Core processing
# ---------------------------

def load_json_objects(json_dir: str, frame_i: int) -> List[Dict[str, Any]]:
    fn = find_json_for_frame(json_dir, frame_i)
    if fn is None:
        return []
    try:
        obj = json.load(open(fn, "r"))
        if isinstance(obj, list):
            return [a for a in obj if isinstance(a, dict)]
        return []
    except Exception:
        return []

def choose_object_for_track_frame(
    objs: List[Dict[str, Any]],
    cx_json: float,
    cy_json: float,
    pad_px: float,
    k_closest: int,
) -> List[Dict[str, Any]]:
    """
    Filter objects whose bbox center is within pad_px (chebyshev-ish square gate),
    then return k closest by euclidean distance.
    """
    cand = []
    for a in objs:
        bb = a.get("bbox", None)
        if not (isinstance(bb, list) and len(bb) == 4):
            continue
        bx, by = bbox_center(bb)
        if abs(bx - cx_json) <= pad_px and abs(by - cy_json) <= pad_px:
            d = math.hypot(bx - cx_json, by - cy_json)
            cand.append((d, a))
    cand.sort(key=lambda t: t[0])
    return [a for _, a in cand[: max(1, int(k_closest))]]

def overlap_prev_gate(
    nuc_obj: Dict[str, Any],
    prev_objs: List[Dict[str, Any]],
    cx_json: float,
    cy_json: float,
    args: argparse.Namespace,
) -> Tuple[Optional[float], str, bool]:
    """
    Returns (overlap_value_or_None, overlap_note, pass_gate_bool)
    pass_gate_bool True means we PASS the overlap gate (i.e., overlap <= threshold or unknown kept).
    """
    # No prev frame masks => policy
    if len(prev_objs) == 0:
        if args.accept_empty_prev:
            return (0.0, "prev_empty_assumed_0", True)
        return (None, "prev_empty_reject", False)

    # Candidate prev objects near track location
    prev_cand = choose_object_for_track_frame(
        prev_objs, cx_json, cy_json, args.prev_search_pad_px, args.k_closest
    )
    if len(prev_cand) == 0:
        if args.accept_no_candidate_prev:
            return (0.0, "prev_nocand_assumed_0", True)
        return (None, "prev_nocand_reject", False)

    nuc_bbox = nuc_obj.get("bbox", None)
    nuc_seg = nuc_obj.get("segmentation", None)
    if not (isinstance(nuc_bbox, list) and len(nuc_bbox) == 4):
        # Can't compute overlap robustly
        if args.overlap_unknown_policy == "keep":
            return (None, "nuc_missing_bbox_keep", True)
        return (None, "nuc_missing_bbox_reject", False)

    # Strategy: bbox_then_poly
    # Step 1: compute bbox IoU for candidates and take max
    best_bbox_iou = 0.0
    best_prev = None
    for p in prev_cand:
        pb = p.get("bbox", None)
        if not (isinstance(pb, list) and len(pb) == 4):
            continue
        iou_b = bbox_iou(nuc_bbox, pb)
        if iou_b > best_bbox_iou:
            best_bbox_iou = iou_b
            best_prev = p

    # Quick accept if bbox IoU already small and strict_overlap is enabled
    # (If strict_overlap: we only care about rejecting high overlaps.)
    if best_bbox_iou <= args.overlap_prev_max and args.strict_overlap:
        return (best_bbox_iou, f"prev_bbox_iou(max={best_bbox_iou:.3f})", True)

    # If shapely is missing, bbox-only decision
    if not HAS_SHAPELY:
        if args.overlap_unknown_policy == "keep":
            # We *can* still decide based on bbox iou; it's conservative-ish.
            ok = (best_bbox_iou <= args.overlap_prev_max)
            note = f"bbox_only_no_shapely(max={best_bbox_iou:.3f})"
            return (best_bbox_iou, note, ok)
        else:
            return (None, "no_shapely_reject", False)

    # Step 2: polygon IoU (only if we have best_prev and it has segmentation)
    if best_prev is None:
        if args.overlap_unknown_policy == "keep":
            return (None, "prev_no_bbox_keep", True)
        return (None, "prev_no_bbox_reject", False)

    prev_seg = best_prev.get("segmentation", None)

    t0 = time.time()
    iou_p = None
    try:
        iou_p = poly_iou_from_coco(nuc_seg, prev_seg)
    except Exception:
        iou_p = None
    dt = time.time() - t0

    if iou_p is None:
        # Fallback to bbox IoU decision
        ok = (best_bbox_iou <= args.overlap_prev_max) if args.overlap_unknown_policy == "keep" else False
        return (best_bbox_iou, f"poly_fail_fallback_bbox(max={best_bbox_iou:.3f},dt={dt:.3f})", ok)

    ok = (iou_p <= args.overlap_prev_max)
    return (float(iou_p), f"bbox_then_poly(bbox={best_bbox_iou:.3f},poly={iou_p:.3f},dt={dt:.3f})", ok)


def parse_args() -> argparse.Namespace:
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
    ap.add_argument("--overlap_unknown_policy", choices=["keep", "reject"], default="keep")

    ap.add_argument("--prev_search_pad_px", type=float, default=250)
    ap.add_argument("--k_closest", type=int, default=15)
    ap.add_argument("--accept_empty_prev", action="store_true")
    ap.add_argument("--accept_no_candidate_prev", action="store_true")

    ap.add_argument("--auto_offset", action="store_true")
    ap.add_argument("--offset_min_tracks", type=int, default=20)
    ap.add_argument("--offset_smooth_window", type=int, default=21)
    ap.add_argument("--offset_use_prev_if_empty", action="store_true")
    ap.add_argument("--offsets_csv", default="")  # optional path to reuse offsets

    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--checkpoint_every", type=int, default=100)
    ap.add_argument("--progress_every", type=int, default=25)

    return ap.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dir(args.out_dir)

    # Load tracks
    print(f"[OK] Reading tracks: {args.tracks_csv}")
    tracks = pd.read_csv(args.tracks_csv)

    # Normalize expected column names (frame_idx vs frame_i)
    if "frame_idx" not in tracks.columns:
        if "frame_i" in tracks.columns:
            tracks = tracks.rename(columns={"frame_i": "frame_idx"})
        else:
            raise ValueError("tracks.csv missing frame_idx (or frame_i) column")

    required_cols = ["track_id", "frame_idx", "cx", "cy"]
    for c in required_cols:
        if c not in tracks.columns:
            raise ValueError(f"tracks.csv missing required column: {c}")

    # Optional cols
    if "area_px" not in tracks.columns:
        # Sometimes area is named "area"
        if "area" in tracks.columns:
            tracks["area_px"] = tracks["area"]
        else:
            tracks["area_px"] = np.nan

    if "R_px" not in tracks.columns:
        # Sometimes radius is named "R"
        if "R" in tracks.columns:
            tracks["R_px"] = tracks["R"]
        else:
            tracks["R_px"] = np.nan

    # Auto offsets
    offsets_df = None
    offsets_path = args.offsets_csv.strip() if args.offsets_csv else ""
    if args.auto_offset or offsets_path:
        if offsets_path and os.path.isfile(offsets_path):
            offsets_df = pd.read_csv(offsets_path)
            print(f"[OK] Loaded offsets from: {offsets_path}")
        else:
            print("[OK] Estimating per-frame offsets (tracks vs JSON bbox centers)...")
            offsets_df = estimate_offsets(
                tracks=tracks,
                json_dir=args.json_dir,
                min_tracks=args.offset_min_tracks,
                smooth_window=args.offset_smooth_window,
                use_prev_if_empty=args.offset_use_prev_if_empty,
            )
            out_off = os.path.join(args.out_dir, "offsets_estimated.csv")
            offsets_df.to_csv(out_off, index=False)
            med_dx = float(np.median(offsets_df["dx_s"].values))
            med_dy = float(np.median(offsets_df["dy_s"].values))
            print(f"[OK] Offset estimation done. median dx={med_dx:.2f}, dy={med_dy:.2f}. Saved: {out_off}")

    # Map frame->(dx_s,dy_s)
    offset_map: Dict[int, Tuple[float, float]] = {}
    if offsets_df is not None:
        for _, r in offsets_df.iterrows():
            offset_map[int(r["frame_idx"])] = (float(r["dx_s"]), float(r["dy_s"]))

    # Resume state
    state_path = os.path.join(args.out_dir, "state.json")
    start_idx = 0
    kept_rows: List[Dict[str, Any]] = []
    rej_rows: List[Dict[str, Any]] = []

    if args.resume and os.path.isfile(state_path):
        try:
            st = json.load(open(state_path, "r"))
            start_idx = int(st.get("next_track_index", 0))
            print(f"[OK] Resuming from state: {state_path} (next_track_index={start_idx})")
        except Exception:
            start_idx = 0

        # also try load existing outputs to continue appending
        filt_path = os.path.join(args.out_dir, "nucleation_events_filtered.csv")
        rej_path = os.path.join(args.out_dir, "nucleation_events_rejected.csv")
        if os.path.isfile(filt_path):
            kept_rows = pd.read_csv(filt_path).to_dict("records")
        if os.path.isfile(rej_path):
            rej_rows = pd.read_csv(rej_path).to_dict("records")

    # Process tracks
    track_ids = sorted(tracks["track_id"].unique().tolist())
    print(f"[OK] Tracks: {len(track_ids)} unique track_id")

    kept = 0
    rejected = 0

    for gi, tid in enumerate(track_ids[start_idx:], start=start_idx):
        tdf = tracks[tracks["track_id"] == tid].sort_values("frame_idx")
        # Gate: find first frame with area & rnuc
        gate = tdf[(tdf["area_px"].astype(float) >= args.amin_px) & (tdf["R_px"].astype(float) <= args.rnuc_max)]
        if len(gate) == 0:
            rejected += 1
            rej_rows.append({
                "track_id": tid,
                "reason": "trackgate_no_frame_passes_area_rnuc",
                "n_obs_total": int(len(tdf)),
                "amin_px": args.amin_px,
                "rnuc_max": args.rnuc_max,
                "L": args.L,
            })
            continue

        nuc_frame = int(gate.iloc[0]["frame_idx"])
        nuc_time_ms = float(gate.iloc[0].get("time_ms", np.nan))

        # Stability: require L observations after nuc_frame within lookahead window
        post = tdf[(tdf["frame_idx"] >= nuc_frame) & (tdf["frame_idx"] <= nuc_frame + args.lookahead_k)]
        if len(post) < args.L:
            rejected += 1
            rej_rows.append({
                "track_id": tid,
                "reason": f"too_short(n={int(len(post))}<L={args.L})",
                "n_obs_total": int(len(tdf)),
                "n_post": int(len(post)),
                "nuc_frame_i": nuc_frame,
                "nuc_time_ms": nuc_time_ms,
                "L": args.L,
            })
            continue

        # Need the nucleation-frame JSON object closest to track position in JSON coordinates
        # Map track coords -> json coords using offsets
        row_nuc = tdf[tdf["frame_idx"] == nuc_frame].iloc[0]
        cx = float(row_nuc["cx"])
        cy = float(row_nuc["cy"])
        dx, dy = offset_map.get(nuc_frame, (0.0, 0.0))
        cx_json = cx - dx
        cy_json = cy - dy

        nuc_objs = load_json_objects(args.json_dir, nuc_frame)
        nuc_cand = choose_object_for_track_frame(nuc_objs, cx_json, cy_json, args.prev_search_pad_px, max(1, args.k_closest))
        if len(nuc_cand) == 0:
            rejected += 1
            rej_rows.append({
                "track_id": tid,
                "reason": "no_mask_match_nuc_frame",
                "nuc_frame_i": nuc_frame,
                "nuc_time_ms": nuc_time_ms,
                "cx": cx, "cy": cy,
                "cx_json": cx_json, "cy_json": cy_json,
            })
            continue

        # pick closest nuc object by bbox-center distance
        best_nuc = None
        best_d = 1e18
        for a in nuc_cand:
            bb = a.get("bbox", None)
            if not (isinstance(bb, list) and len(bb) == 4):
                continue
            bx, by = bbox_center(bb)
            d = math.hypot(bx - cx_json, by - cy_json)
            if d < best_d:
                best_d = d
                best_nuc = a

        if best_nuc is None:
            rejected += 1
            rej_rows.append({
                "track_id": tid,
                "reason": "no_mask_match_nuc_frame",
                "nuc_frame_i": nuc_frame,
                "nuc_time_ms": nuc_time_ms,
            })
            continue

        # Previous-frame overlap gate
        prev_frame = nuc_frame - 1
        if prev_frame >= 0:
            prev_dx, prev_dy = offset_map.get(prev_frame, (dx, dy))
            cx_prev_json = cx - prev_dx
            cy_prev_json = cy - prev_dy
            prev_objs = load_json_objects(args.json_dir, prev_frame)

            ov, note, ok = overlap_prev_gate(best_nuc, prev_objs, cx_prev_json, cy_prev_json, args)
            if not ok:
                rejected += 1
                rej_rows.append({
                    "track_id": tid,
                    "reason": f"overlap_prev>{args.overlap_prev_max}",
                    "nuc_frame_i": nuc_frame,
                    "nuc_time_ms": nuc_time_ms,
                    "overlap_prev": ov,
                    "overlap_note": note,
                })
                continue
        else:
            ov, note = 0.0, "prev_frame_negative_assumed_0"

        # Keep
        kept += 1
        kept_rows.append({
            "track_id": tid,
            "nuc_frame_i": nuc_frame,
            "nuc_time_ms": nuc_time_ms,
            "cx": cx, "cy": cy,
            "cx_json": cx_json, "cy_json": cy_json,
            "overlap_prev": ov,
            "overlap_note": note,
            "L": args.L,
            "amin_px": args.amin_px,
            "rnuc_max": args.rnuc_max,
            "has_shapely": HAS_SHAPELY,
        })

        # Progress / checkpoint
        if (gi + 1) % args.progress_every == 0:
            print(f"[PROGRESS] {gi+1}/{len(track_ids)} | kept={kept} | rej={rejected}")

        if (gi + 1) % args.checkpoint_every == 0:
            pd.DataFrame(kept_rows).to_csv(os.path.join(args.out_dir, f"nucleation_events_filtered_ckpt_{gi+1:05d}.csv"), index=False)
            pd.DataFrame(rej_rows).to_csv(os.path.join(args.out_dir, f"nucleation_events_rejected_ckpt_{gi+1:05d}.csv"), index=False)
            json.dump({"next_track_index": gi + 1}, open(state_path, "w"), indent=2)

    # Final outputs
    filt_path = os.path.join(args.out_dir, "nucleation_events_filtered.csv")
    rej_path = os.path.join(args.out_dir, "nucleation_events_rejected.csv")
    pd.DataFrame(kept_rows).to_csv(filt_path, index=False)
    pd.DataFrame(rej_rows).to_csv(rej_path, index=False)
    json.dump({"next_track_index": len(track_ids)}, open(state_path, "w"), indent=2)

    print(f"[OK] Wrote outputs to: {args.out_dir}")
    print(f"[OK] kept={kept} | rejected={rejected} | has_shapely={HAS_SHAPELY}")
    print(f"[OK] State file: {state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())