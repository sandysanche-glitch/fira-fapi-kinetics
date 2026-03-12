#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re, glob, json, math, time, argparse
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

def normalize_rle(seg: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(seg, dict):
        return None
    if "counts" not in seg or "size" not in seg:
        return None
    rle = dict(seg)
    if isinstance(rle.get("counts", None), str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return rle

def bbox_center(bbox: Any) -> Tuple[float, float]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return float("nan"), float("nan")
    x, y, w, h = bbox[:4]
    return float(x + 0.5 * w), float(y + 0.5 * h)

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

def pick_column(df: pd.DataFrame, candidates: List[str], required_name: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(f"[ERR] tracks.csv missing required {required_name}. Tried: {candidates}")

def overlap_frac_a_over_b(rle_a: Dict[str, Any], rle_b: Dict[str, Any]) -> float:
    inter = mask_utils.merge([rle_a, rle_b], intersect=True)
    ai = float(mask_utils.area(inter))
    aa = float(mask_utils.area(rle_a))
    if aa <= 0:
        return float("nan")
    return ai / aa

def overlap_frac_prev_with_timeout(cur_rle: Dict[str, Any], prev_rles: List[Dict[str, Any]], timeout_s: float):
    """
    Max overlap(cur, prev) with a soft timeout.
    If timeout is reached, returns (nan, "timeout").
    """
    t0 = time.time()
    best = 0.0
    got = False
    for k, prle in enumerate(prev_rles):
        if timeout_s > 0 and (time.time() - t0) > timeout_s:
            return float("nan"), "timeout"
        try:
            of = overlap_frac_a_over_b(cur_rle, prle)
        except Exception:
            continue
        if np.isfinite(of):
            got = True
            if of > best:
                best = float(of)
    return (best if got else float("nan")), "ok"

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
    ap.add_argument("--reject_if_overlap_nan", action="store_true")

    ap.add_argument("--lookahead_k", type=int, default=150)
    ap.add_argument("--max_dist_min", type=float, default=800.0)
    ap.add_argument("--max_dist_factor", type=float, default=50.0)

    ap.add_argument("--progress_every", type=int, default=50)
    ap.add_argument("--print_each_track", action="store_true")

    # Overlap speed controls
    ap.add_argument("--prev_search_pad_px", type=float, default=100.0)
    ap.add_argument("--prev_max_candidates", type=int, default=15)
    ap.add_argument("--overlap_timeout_s", type=float, default=0.25,
                    help="Soft timeout per-track overlap computation. If reached, overlap is set to NaN and note='timeout'.")

    # Resume / slice controls
    ap.add_argument("--start_track_idx", type=int, default=0)
    ap.add_argument("--max_tracks", type=int, default=-1)

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

    os.makedirs(args.out_dir, exist_ok=True)

    filtered, rejected, rejected_trackgate, slow = [], [], [], []

    for i_local, (tid, df) in enumerate(slice_groups, start=1):
        i_global = start + i_local
        if args.print_each_track:
            print(f"[TRACK] global={i_global} tid={tid}", flush=True)

        n_obs = len(df)
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
        if ann is None:
            rejected.append({"track_id": int(tid), "reason": "no_mask_match", "nuc_frame_i": nuc_frame, "match_note": match_note})
            continue

        cur_rle = normalize_rle(ann.get("segmentation", None))
        if cur_rle is None:
            rejected.append({"track_id": int(tid), "reason": "no_rle_in_segmentation", "nuc_frame_i": nuc_frame, "mask_frame_i": mask_frame})
            continue

        # Overlap gate
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
                # filter by proximity
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
                    # cap candidates
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
                        t0 = time.time()
                        overlap_prev, status = overlap_frac_prev_with_timeout(
                            cur_rle, prev_rles, float(args.overlap_timeout_s)
                        )
                        dt = time.time() - t0
                        if status == "timeout":
                            overlap_note = "timeout"
                            slow.append({
                                "track_id": int(tid),
                                "global_i": int(i_global),
                                "mask_frame_i": int(mask_frame),
                                "prev_candidates": int(len(prev_rles)),
                                "pad_px": float(pad),
                                "timeout_s": float(args.overlap_timeout_s),
                                "dt_s": float(dt),
                            })

            if np.isnan(overlap_prev) and args.reject_if_overlap_nan:
                rejected_trackgate.append({"track_id": int(tid), "reason": "overlap_nan", "overlap_note": overlap_note})
                continue

            if np.isfinite(overlap_prev) and overlap_prev > args.overlap_prev_max:
                rejected_trackgate.append({"track_id": int(tid), "reason": f"overlap_prev>{args.overlap_prev_max}",
                                           "overlap_prev": float(overlap_prev), "overlap_note": overlap_note})
                continue

        _, mt = parse_frame_and_time_from_name(os.path.basename(json_path or ""))
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

        if (i_local % args.progress_every) == 0 or i_local == n_slice:
            print(f"[PROGRESS] {i_local}/{n_slice} (global {i_global}/{n_total}) | kept={len(filtered)} | rej={len(rejected)} | rej_trackgate={len(rejected_trackgate)}",
                  flush=True)

    # Write outputs
    pd.DataFrame(filtered).to_csv(os.path.join(args.out_dir, "nucleation_events_filtered.csv"), index=False)
    pd.DataFrame(rejected).to_csv(os.path.join(args.out_dir, "nucleation_events_rejected.csv"), index=False)
    pd.DataFrame(rejected_trackgate).to_csv(os.path.join(args.out_dir, "nucleation_events_rejected_trackgate.csv"), index=False)
    pd.DataFrame(slow).to_csv(os.path.join(args.out_dir, "slow_tracks.csv"), index=False)

    print(f"[OK] Wrote outputs to: {args.out_dir}", flush=True)
    print(f"[OK] kept={len(filtered)} | rejected={len(rejected)} | rejected_trackgate={len(rejected_trackgate)} | slow_logged={len(slow)}",
          flush=True)

if __name__ == "__main__":
    main()
