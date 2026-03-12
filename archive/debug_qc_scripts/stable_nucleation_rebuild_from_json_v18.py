import os
import re
import json
import math
import time
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

# --- Optional shapely (polygon IOU) ---
HAS_SHAPELY = False
try:
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except Exception:
    HAS_SHAPELY = False


# -------------------------
# Helpers
# -------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def find_json_for_frame(json_dir: str, frame_idx: int) -> Optional[str]:
    # expects filenames like frame_00018_t36.00ms_idmapped.json
    pref = f"frame_{frame_idx:05d}_"
    for fn in os.listdir(json_dir):
        if fn.startswith(pref) and fn.endswith("_idmapped.json"):
            return os.path.join(json_dir, fn)
    return None

def bbox_center(b: List[float]) -> Tuple[float, float]:
    x, y, w, h = b
    return (x + 0.5*w, y + 0.5*h)

def bbox_iou(b1: List[float], b2: List[float]) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ax1, ay1, ax2, ay2 = x1, y1, x1+w1, y1+h1
    bx1, by1, bx2, by2 = x2, y2, x2+w2, y2+h2

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw*ih
    if inter <= 0:
        return 0.0
    a = w1*h1
    b = w2*h2
    return float(inter / (a + b - inter + 1e-9))

def _seg_to_polygons(seg: Any) -> List[List[Tuple[float, float]]]:
    """
    COCO-like polygon segmentation:
      seg = [ [x1,y1,x2,y2,...], [ ... ], ... ]
    Sometimes seg is a dict (RLE) - we won't decode here.
    """
    polys = []
    if isinstance(seg, list):
        # could be list of lists, or a single flat list
        if len(seg) == 0:
            return polys
        if isinstance(seg[0], (int, float)):
            seg = [seg]
        for poly in seg:
            if not isinstance(poly, list) or len(poly) < 6:
                continue
            pts = list(zip(poly[0::2], poly[1::2]))
            if len(pts) >= 3:
                polys.append(pts)
    return polys

def poly_iou_from_seg(seg1: Any, seg2: Any, timeout_s: float) -> Tuple[Optional[float], str]:
    """
    Returns (iou or None, note).
    None means "unknown/failed".
    """
    if not HAS_SHAPELY:
        return None, "no_shapely"

    t0 = time.time()
    polys1 = _seg_to_polygons(seg1)
    polys2 = _seg_to_polygons(seg2)
    if not polys1 or not polys2:
        return None, "no_poly_data"

    # Guard: timeout while building polygons too
    if time.time() - t0 > timeout_s:
        return None, "poly_timeout_build"

    def make_geom(polys):
        geoms = []
        for pts in polys:
            if time.time() - t0 > timeout_s:
                return None
            try:
                g = Polygon(pts)
                if not g.is_valid:
                    g = g.buffer(0)
                if g.area > 0:
                    geoms.append(g)
            except Exception:
                continue
        if not geoms:
            return None
        try:
            u = unary_union(geoms)
            return u
        except Exception:
            return None

    g1 = make_geom(polys1)
    if g1 is None:
        return None, "poly_fail_1"
    if time.time() - t0 > timeout_s:
        return None, "poly_timeout_1"

    g2 = make_geom(polys2)
    if g2 is None:
        return None, "poly_fail_2"
    if time.time() - t0 > timeout_s:
        return None, "poly_timeout_2"

    try:
        inter = g1.intersection(g2).area
        union = g1.union(g2).area
        if union <= 0:
            return 0.0, "poly_ok"
        return float(inter / union), "poly_ok"
    except Exception:
        return None, "poly_exception"


@dataclass
class OffsetSeries:
    dx_by_frame: Dict[int, float]
    dy_by_frame: Dict[int, float]
    dx_med: float
    dy_med: float

def estimate_offsets(tracks: pd.DataFrame, json_dir: str,
                     min_tracks: int = 20,
                     smooth_window: int = 21,
                     use_prev_if_empty: bool = True,
                     out_csv: Optional[str] = None) -> OffsetSeries:
    """
    Estimate per-frame (dx,dy) where:
        dx = median(track_cx) - median(json_bbox_cx)
        dy = median(track_cy) - median(json_bbox_cy)

    Then to map a track centroid into JSON coord system:
        x_json = x_track - dx(frame)
        y_json = y_track - dy(frame)
    """
    frames = sorted(tracks["frame_idx"].unique().tolist())
    rows = []
    last_dx, last_dy = 0.0, 0.0

    for f in frames:
        sub = tracks[tracks["frame_idx"] == f]
        if len(sub) < min_tracks:
            rows.append((f, np.nan, np.nan, "few_tracks"))
            continue

        jprev = find_json_for_frame(json_dir, f)
        if jprev is None:
            rows.append((f, np.nan, np.nan, "missing_json"))
            continue

        try:
            obj = json.load(open(jprev, "r"))
        except Exception:
            rows.append((f, np.nan, np.nan, "bad_json"))
            continue

        if not isinstance(obj, list) or len(obj) == 0:
            rows.append((f, np.nan, np.nan, "empty_json"))
            continue

        bcx = []
        bcy = []
        for a in obj:
            if "bbox" not in a:
                continue
            cxj, cyj = bbox_center(a["bbox"])
            bcx.append(cxj)
            bcy.append(cyj)

        if len(bcx) == 0:
            rows.append((f, np.nan, np.nan, "no_bbox"))
            continue

        dx = float(np.median(sub["cx"])) - float(np.median(bcx))
        dy = float(np.median(sub["cy"])) - float(np.median(bcy))
        rows.append((f, dx, dy, "ok"))

    df = pd.DataFrame(rows, columns=["frame_idx", "dx_raw", "dy_raw", "status"])

    # fill missing
    if use_prev_if_empty:
        df["dx_raw"] = df["dx_raw"].ffill().bfill().fillna(0.0)
        df["dy_raw"] = df["dy_raw"].ffill().bfill().fillna(0.0)
    else:
        df["dx_raw"] = df["dx_raw"].fillna(0.0)
        df["dy_raw"] = df["dy_raw"].fillna(0.0)

    # smooth
    w = max(3, int(smooth_window))
    if w % 2 == 0:
        w += 1
    df["dx_s"] = df["dx_raw"].rolling(window=w, center=True, min_periods=1).median()
    df["dy_s"] = df["dy_raw"].rolling(window=w, center=True, min_periods=1).median()

    dx_by = {int(r.frame_idx): float(r.dx_s) for r in df.itertuples(index=False)}
    dy_by = {int(r.frame_idx): float(r.dy_s) for r in df.itertuples(index=False)}
    dx_med = float(np.median(df["dx_s"].values))
    dy_med = float(np.median(df["dy_s"].values))

    if out_csv:
        df.to_csv(out_csv, index=False)

    return OffsetSeries(dx_by_frame=dx_by, dy_by_frame=dy_by, dx_med=dx_med, dy_med=dy_med)


# -------------------------
# Main logic
# -------------------------

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
    ap.add_argument("--overlap_unknown_policy", choices=["keep", "reject", "flag"], default="flag")

    ap.add_argument("--prev_search_pad_px", type=float, default=250)
    ap.add_argument("--prev_max_candidates", type=int, default=50)
    ap.add_argument("--k_closest", type=int, default=15)

    ap.add_argument("--accept_empty_prev", action="store_true")
    ap.add_argument("--accept_no_candidate_prev", action="store_true")

    ap.add_argument("--auto_offset", action="store_true")
    ap.add_argument("--offset_min_tracks", type=int, default=20)
    ap.add_argument("--offset_smooth_window", type=int, default=21)
    ap.add_argument("--offset_use_prev_if_empty", action="store_true")

    ap.add_argument("--overlap_poly_timeout_s", type=float, default=0.60)

    ap.add_argument("--progress_every", type=int, default=25)
    ap.add_argument("--checkpoint_every", type=int, default=100)

    args = ap.parse_args()

    ensure_dir(args.out_dir)

    print(f"[OK] Reading tracks: {args.tracks_csv}")
    tracks = pd.read_csv(args.tracks_csv)

    # column normalisation
    if "frame_i" in tracks.columns and "frame_idx" not in tracks.columns:
        tracks = tracks.rename(columns={"frame_i": "frame_idx"})
    required = ["track_id", "frame_idx", "cx", "cy"]
    for c in required:
        if c not in tracks.columns:
            raise SystemExit(f"[ERR] tracks.csv missing column: {c}. Found: {tracks.columns.tolist()}")

    # Optional per-frame offset model
    offsets = None
    if args.auto_offset:
        print("[OK] Estimating per-frame offsets (tracks vs JSON bbox centers)...")
        off_csv = os.path.join(args.out_dir, "offsets_estimated.csv")
        offsets = estimate_offsets(
            tracks=tracks,
            json_dir=args.json_dir,
            min_tracks=args.offset_min_tracks,
            smooth_window=args.offset_smooth_window,
            use_prev_if_empty=args.offset_use_prev_if_empty,
            out_csv=off_csv,
        )
        print(f"[OK] Offset estimation done. median dx={offsets.dx_med:.2f}, dy={offsets.dy_med:.2f}. Saved: {off_csv}")

    track_ids = sorted(tracks["track_id"].unique().tolist())
    print(f"[OK] Tracks: {len(track_ids)} unique track_id")

    kept_rows = []
    rej_rows = []
    rej_trackgate_rows = []

    def checkpoint(tag: str):
        f1 = os.path.join(args.out_dir, f"nucleation_events_filtered_{tag}.csv")
        f2 = os.path.join(args.out_dir, f"nucleation_events_rejected_{tag}.csv")
        f3 = os.path.join(args.out_dir, f"nucleation_events_rejected_trackgate_{tag}.csv")
        pd.DataFrame(kept_rows).to_csv(f1, index=False)
        pd.DataFrame(rej_rows).to_csv(f2, index=False)
        pd.DataFrame(rej_trackgate_rows).to_csv(f3, index=False)

    for gi, tid in enumerate(track_ids, start=1):
        tdf = tracks[tracks["track_id"] == tid].sort_values("frame_idx").reset_index(drop=True)

        # --- Track gate: find frames that pass area/rnuc thresholds
        # Your tracks.csv may have area/R columns with different names; try common ones:
        area_col = None
        for cand in ["area_px", "area", "A_px", "A"]:
            if cand in tdf.columns:
                area_col = cand
                break
        r_col = None
        for cand in ["R_px", "R", "radius_px", "r_px"]:
            if cand in tdf.columns:
                r_col = cand
                break

        if area_col is None or r_col is None:
            # If missing, we cannot do nucleation detection properly
            rej_trackgate_rows.append({
                "track_id": tid,
                "reason": "trackgate_missing_area_or_r",
                "area_col": area_col,
                "r_col": r_col,
            })
            continue

        pass_gate = tdf[(tdf[area_col] >= args.amin_px) & (tdf[r_col] <= args.rnuc_max)]
        if pass_gate.empty:
            rej_trackgate_rows.append({
                "track_id": tid,
                "reason": "trackgate_no_frame_passes_area_rnuc",
            })
            continue

        # earliest candidate nucleation frame among those
        first_gate_idx = int(pass_gate["frame_idx"].iloc[0])

        # define a lookahead window starting there
        window = tdf[(tdf["frame_idx"] >= first_gate_idx) & (tdf["frame_idx"] <= first_gate_idx + args.lookahead_k)]
        if window.empty:
            rej_rows.append({"track_id": tid, "reason": "empty_lookahead"})
            continue

        # need stable run length L under gates
        good = (window[area_col] >= args.amin_px) & (window[r_col] <= args.rnuc_max)
        # find first run of length L
        good_idx = window.loc[good, "frame_idx"].values
        if len(good_idx) < args.L:
            rej_rows.append({"track_id": tid, "reason": f"too_short(n={len(good_idx)}<L={args.L})"})
            continue

        # check consecutive run
        good_frames = sorted(good_idx.tolist())
        run_start = None
        run_len = 1
        for i in range(1, len(good_frames)):
            if good_frames[i] == good_frames[i-1] + 1:
                run_len += 1
            else:
                run_len = 1
            if run_len >= args.L:
                run_start = good_frames[i - args.L + 1]
                break
        if run_start is None:
            rej_rows.append({"track_id": tid, "reason": f"no_stable_run_L{args.L}_after_gates"})
            continue

        nuc_frame = int(run_start)
        nuc_row = tdf[tdf["frame_idx"] == nuc_frame]
        if nuc_row.empty:
            rej_rows.append({"track_id": tid, "reason": "nuc_row_missing"})
            continue
        nuc_row = nuc_row.iloc[0]

        # --- overlap gate with prev frame
        prev_frame = nuc_frame - 1
        prev_path = find_json_for_frame(args.json_dir, prev_frame)

        overlap_prev = np.nan
        overlap_note = ""
        overlap_used = "unknown"

        if prev_path is None:
            if args.accept_empty_prev:
                overlap_prev = 0.0
                overlap_note = "prev_missing_json_assumed_0"
                overlap_used = "assumed"
            else:
                overlap_note = "prev_missing_json"
        else:
            try:
                prev_obj = json.load(open(prev_path, "r"))
            except Exception:
                prev_obj = None

            if not isinstance(prev_obj, list) or prev_obj is None:
                if args.accept_empty_prev:
                    overlap_prev = 0.0
                    overlap_note = "prev_bad_json_assumed_0"
                    overlap_used = "assumed"
                else:
                    overlap_note = "prev_bad_json"
            elif len(prev_obj) == 0:
                if args.accept_empty_prev:
                    overlap_prev = 0.0
                    overlap_note = "prev_empty_assumed_0"
                    overlap_used = "assumed"
                else:
                    overlap_note = "prev_empty"
            else:
                # map track centroid into JSON coordinate system
                cx_t, cy_t = float(nuc_row["cx"]), float(nuc_row["cy"])
                dx = offsets.dx_by_frame.get(nuc_frame, offsets.dx_med) if offsets else 0.0
                dy = offsets.dy_by_frame.get(nuc_frame, offsets.dy_med) if offsets else 0.0
                cx = cx_t - dx
                cy = cy_t - dy

                # spatial candidate filter by bbox center in a padded square
                pad = float(args.prev_search_pad_px)
                cand = []
                for a in prev_obj:
                    if "bbox" not in a:
                        continue
                    bx, by = bbox_center(a["bbox"])
                    if (abs(bx - cx) <= pad) and (abs(by - cy) <= pad):
                        d = math.hypot(bx - cx, by - cy)
                        cand.append((d, a))
                cand.sort(key=lambda x: x[0])
                cand = cand[:max(1, int(args.prev_max_candidates))]

                if len(cand) == 0:
                    if args.accept_no_candidate_prev:
                        overlap_prev = 0.0
                        overlap_note = "prev_nocand_assumed_0"
                        overlap_used = "assumed"
                    else:
                        overlap_note = "prev_nocand"
                else:
                    # keep k closest for IOU checks
                    cand = cand[:max(1, int(args.k_closest))]
                    # bbox IOU precheck vs nuc bbox (need nuc bbox from nuc-frame json)
                    nuc_path = find_json_for_frame(args.json_dir, nuc_frame)
                    nuc_obj = None
                    if nuc_path:
                        try:
                            nuc_obj = json.load(open(nuc_path, "r"))
                        except Exception:
                            nuc_obj = None

                    # Find nuc mask by closest bbox to (cx,cy) in nuc frame
                    nuc_ann = None
                    if isinstance(nuc_obj, list) and len(nuc_obj) > 0:
                        best = None
                        for a in nuc_obj:
                            if "bbox" not in a:
                                continue
                            bx, by = bbox_center(a["bbox"])
                            d = math.hypot(bx - cx, by - cy)
                            if best is None or d < best[0]:
                                best = (d, a)
                        if best is not None:
                            nuc_ann = best[1]

                    if nuc_ann is None or "bbox" not in nuc_ann:
                        overlap_note = "no_nuc_mask_match"
                    else:
                        nuc_bbox = nuc_ann["bbox"]
                        # bbox stage
                        best_bbox_iou = 0.0
                        best_bbox_ann = None
                        for _, a in cand:
                            iou_b = bbox_iou(nuc_bbox, a["bbox"])
                            if iou_b > best_bbox_iou:
                                best_bbox_iou = iou_b
                                best_bbox_ann = a

                        # if bbox already clearly overlaps too much, we can reject early in strict mode
                        if args.strict_overlap and best_bbox_iou > args.overlap_prev_max and not HAS_SHAPELY:
                            overlap_prev = best_bbox_iou
                            overlap_note = f"prev_bbox_max(n={len(cand)})(max={best_bbox_iou:.3f})"
                            overlap_used = "bbox"
                        else:
                            # poly stage (if possible)
                            if HAS_SHAPELY and "segmentation" in nuc_ann and best_bbox_ann is not None and "segmentation" in best_bbox_ann:
                                iou_p, note = poly_iou_from_seg(nuc_ann["segmentation"], best_bbox_ann["segmentation"], args.overlap_poly_timeout_s)
                                if iou_p is None:
                                    # fallback to bbox value but mark clearly
                                    overlap_prev = best_bbox_iou
                                    overlap_note = f"prev_poly_fail->{note}_bbox({best_bbox_iou:.3f})"
                                    overlap_used = "bbox_fallback"
                                else:
                                    overlap_prev = iou_p
                                    overlap_note = f"prev_poly_max(n={len(cand)})(max={iou_p:.3f})"
                                    overlap_used = "poly"
                            else:
                                overlap_prev = best_bbox_iou
                                overlap_note = f"prev_bbox_max(n={len(cand)})(max={best_bbox_iou:.3f})"
                                overlap_used = "bbox"

        # Apply overlap policy
        if args.strict_overlap:
            if not np.isnan(overlap_prev):
                if overlap_prev > args.overlap_prev_max:
                    rej_trackgate_rows.append({
                        "track_id": tid,
                        "nuc_frame_i": nuc_frame,
                        "reason": f"overlap_prev>{args.overlap_prev_max}",
                        "overlap_prev": float(overlap_prev),
                        "overlap_note": overlap_note,
                        "overlap_used": overlap_used,
                    })
                    continue
            else:
                # unknown
                if args.overlap_unknown_policy == "reject":
                    rej_trackgate_rows.append({
                        "track_id": tid,
                        "nuc_frame_i": nuc_frame,
                        "reason": "overlap_unknown_reject",
                        "overlap_note": overlap_note,
                    })
                    continue
                elif args.overlap_unknown_policy == "flag":
                    overlap_note = overlap_note + "|unknown_overlap_kept"

        kept_rows.append({
            "track_id": tid,
            "nuc_frame_i": nuc_frame,
            "nuc_time_ms": nuc_row["t_ms"] if "t_ms" in tdf.columns else np.nan,
            "area_px": float(nuc_row[area_col]),
            "R_px": float(nuc_row[r_col]),
            "overlap_prev": float(overlap_prev) if not np.isnan(overlap_prev) else np.nan,
            "overlap_note": overlap_note,
            "overlap_used": overlap_used,
        })

        if args.progress_every and (gi % args.progress_every == 0 or gi == len(track_ids)):
            print(f"[PROGRESS] {gi}/{len(track_ids)} | kept={len(kept_rows)} | rej={len(rej_rows)} | rej_trackgate={len(rej_trackgate_rows)}")

        if args.checkpoint_every and (gi % args.checkpoint_every == 0):
            checkpoint(f"ckpt_{gi:05d}")

    # write final
    out_f = os.path.join(args.out_dir, "nucleation_events_filtered.csv")
    out_r = os.path.join(args.out_dir, "nucleation_events_rejected.csv")
    out_tg = os.path.join(args.out_dir, "nucleation_events_rejected_trackgate.csv")
    pd.DataFrame(kept_rows).to_csv(out_f, index=False)
    pd.DataFrame(rej_rows).to_csv(out_r, index=False)
    pd.DataFrame(rej_trackgate_rows).to_csv(out_tg, index=False)

    # state
    st = {
        "kept": len(kept_rows),
        "rejected": len(rej_rows),
        "rejected_trackgate": len(rej_trackgate_rows),
        "has_shapely": HAS_SHAPELY,
        "dx_med": float(offsets.dx_med) if offsets else 0.0,
        "dy_med": float(offsets.dy_med) if offsets else 0.0,
    }
    with open(os.path.join(args.out_dir, "state.json"), "w") as f:
        json.dump(st, f, indent=2)

    print(f"[OK] Wrote outputs to: {args.out_dir}")
    print(f"[OK] kept={len(kept_rows)} | rejected={len(rej_rows)} | rejected_trackgate={len(rej_trackgate_rows)} | has_shapely={HAS_SHAPELY}")


if __name__ == "__main__":
    main()