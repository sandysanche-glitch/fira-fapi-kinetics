#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
stable_nucleation_rebuild_from_json_v12.py

v12 = v11 + robust prev-candidate selection:
- Take k closest prev-frame candidates (centroid distance) within pad box
- Compute bbox overlap fraction for each (intersection / cur_bbox_area)
- Choose candidate with MAX bbox overlap as "previous"
- Compute ONE mask-overlap for that chosen candidate
- If mask-overlap is killed/timeout -> bbox fallback gating using bbox_fallback_max (default 0.7)

Intended use: TEMPO where some overlaps are pathologically slow and prev association matters.

Outputs:
- nucleation_events_filtered.csv
- nucleation_events_rejected.csv
- nucleation_events_rejected_trackgate.csv
- slow_tracks.csv (timeout/killed info)
- state.json (resume)
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

import multiprocessing as mp


_FRAME_RE = re.compile(r"frame_(\d+)_t([0-9.]+)ms", re.IGNORECASE)


def parse_frame_and_time_from_name(fname: str) -> Tuple[Optional[int], Optional[float]]:
    m = _FRAME_RE.search(fname or "")
    if not m:
        return None, None
    return int(m.group(1)), float(m.group(2))


def find_json_for_frame(json_dir: str, frame_idx: int) -> Optional[str]:
    patt = os.path.join(json_dir, f"frame_{frame_idx:05d}_t*ms_idmapped.json")
    cands = sorted(glob.glob(patt))
    return cands[0] if cands else None


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


def bbox_overlap_frac(cur_bbox: Any, prev_bbox: Any) -> float:
    """
    Overlap fraction = intersection_area / cur_bbox_area
    """
    if (not isinstance(cur_bbox, (list, tuple))) or (not isinstance(prev_bbox, (list, tuple))):
        return float("nan")
    if len(cur_bbox) < 4 or len(prev_bbox) < 4:
        return float("nan")

    x1, y1, w1, h1 = map(float, cur_bbox[:4])
    x2, y2, w2, h2 = map(float, prev_bbox[:4])

    if w1 <= 0 or h1 <= 0:
        return float("nan")

    ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
    bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = w1 * h1
    return float(inter / area_a) if area_a > 0 else float("nan")


def pick_column(df: pd.DataFrame, candidates: List[str], required_name: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(f"[ERR] tracks.csv missing required {required_name}. Tried: {candidates}")


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
        jp = find_json_for_frame(json_dir, f)
        anns = read_json_list(jp)
        if not anns:
            continue
        ann, _ = match_det_by_centroid(anns, cx, cy, max_dist)
        if ann is None:
            continue
        return f, jp, ann, ("ok" if j == 0 else f"lookahead+{j}")
    return None, None, None, "no_match_in_lookahead"


# ---------- SAFE overlap in subprocess ----------
def _overlap_worker(q, cur_rle, prev_rle):
    try:
        inter = mask_utils.merge([cur_rle, prev_rle], intersect=True)
        ai = float(mask_utils.area(inter))
        aa = float(mask_utils.area(cur_rle))
        q.put(ai / aa if aa > 0 else float("nan"))
    except Exception:
        q.put(float("nan"))


def overlap_frac_safe_mp(cur_rle: Dict[str, Any],
                         prev_rle: Dict[str, Any],
                         per_call_timeout_s: float) -> Tuple[float, str]:
    ctx = mp.get_context("spawn")
    q = ctx.Queue(maxsize=1)
    p = ctx.Process(target=_overlap_worker, args=(q, cur_rle, prev_rle))
    p.daemon = True
    p.start()
    p.join(timeout=max(0.001, float(per_call_timeout_s)))

    if p.is_alive():
        p.terminate()
        p.join(timeout=1.0)
        return float("nan"), "overlap_killed_timeout"

    try:
        return float(q.get_nowait()), "ok"
    except Exception:
        return float("nan"), "overlap_no_value"


def select_prev_candidate_bboxpreselect(
    cur_ann: Dict[str, Any],
    prev_anns: List[Dict[str, Any]],
    pad_px: float,
    k_closest: int,
) -> Tuple[Optional[Dict[str, Any]], float, str, int]:
    """
    v12 selection:
    - filter candidates within pad box
    - take k closest by centroid distance
    - compute bbox overlap fraction for those k
    - pick candidate with max bbox overlap fraction
    returns: (chosen_ann, chosen_bbox_overlap, note, n_candidates_used_for_bbox)
    """
    if not prev_anns:
        return None, float("nan"), "prev_empty_or_missing", 0

    cur_bbox = cur_ann.get("bbox", None)
    cur_cx, cur_cy = bbox_center(cur_bbox)
    if not (np.isfinite(cur_cx) and np.isfinite(cur_cy)):
        return None, float("nan"), "cur_bbox_center_nan", 0

    cand = []
    for pa in prev_anns:
        px, py = bbox_center(pa.get("bbox", None))
        if not (np.isfinite(px) and np.isfinite(py)):
            continue
        if abs(px - cur_cx) <= pad_px and abs(py - cur_cy) <= pad_px:
            d = math.hypot(px - cur_cx, py - cur_cy)
            cand.append((d, pa))

    if not cand:
        return None, float("nan"), "prev_no_candidates_nearby", 0

    cand.sort(key=lambda x: x[0])
    cand = cand[:max(1, int(k_closest))]

    best_pa = None
    best_ov = -1.0
    for _, pa in cand:
        ov = bbox_overlap_frac(cur_bbox, pa.get("bbox", None))
        if np.isfinite(ov) and ov > best_ov:
            best_ov = float(ov)
            best_pa = pa

    if best_pa is None:
        return None, float("nan"), "prev_no_valid_bbox", len(cand)

    return best_pa, float(best_ov), f"bboxpreselect_k={len(cand)}", len(cand)


def write_csv(path: str, rows: List[Dict[str, Any]]):
    pd.DataFrame(rows).to_csv(path, index=False)


def append_log_row(path: str, row: Dict[str, Any], header_fields: List[str]):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header_fields)
        if not exists:
            w.writeheader()
        w.writerow(row)


def write_state(path: str, state: Dict[str, Any]):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def try_load_state(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--tracks_csv", required=True)
    ap.add_argument("--json_dir", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--L", type=int, default=5)
    ap.add_argument("--amin_px", type=float, default=800.0)
    ap.add_argument("--rmin_px", type=float, default=0.0)
    ap.add_argument("--rnuc_max", type=float, default=60.0)

    ap.add_argument("--overlap_prev_max", type=float, default=0.3)
    ap.add_argument("--strict_overlap", action="store_true")
    ap.add_argument("--reject_if_overlap_nan", action="store_true")

    ap.add_argument("--lookahead_k", type=int, default=150)
    ap.add_argument("--max_dist_min", type=float, default=800.0)
    ap.add_argument("--max_dist_factor", type=float, default=50.0)

    ap.add_argument("--prev_search_pad_px", type=float, default=250.0)

    # v12 selection parameters
    ap.add_argument("--k_closest", type=int, default=5)

    # overlap timeouts
    ap.add_argument("--overlap_timeout_s", type=float, default=3.0)
    ap.add_argument("--overlap_per_call_timeout_s", type=float, default=0.30)

    # Handling empty/none previous masks
    ap.add_argument("--accept_empty_prev", action="store_true")
    ap.add_argument("--empty_prev_overlap_value", type=float, default=0.0)
    ap.add_argument("--accept_no_candidate_prev", action="store_true")
    ap.add_argument("--no_candidate_overlap_value", type=float, default=0.0)

    # bbox fallback (separate threshold!)
    ap.add_argument("--bbox_fallback_on_timeout", action="store_true")
    ap.add_argument("--bbox_fallback_only_for_killed", action="store_true")
    ap.add_argument("--bbox_fallback_max", type=float, default=0.7)

    ap.add_argument("--progress_every", type=int, default=10)
    ap.add_argument("--checkpoint_every", type=int, default=25)

    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--state_path", default=None)

    ap.add_argument("--start_track_idx", type=int, default=0)
    ap.add_argument("--max_tracks", type=int, default=-1)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    state_path = args.state_path or os.path.join(args.out_dir, "state.json")

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

    tracks = tracks.sort_values([track_id_col, frame_col]).reset_index(drop=True)
    groups = list(tracks.groupby(track_id_col, sort=False))
    n_total = len(groups)
    print(f"[OK] Tracks: {n_total} unique track_id", flush=True)

    start = max(0, int(args.start_track_idx))
    if args.resume:
        st = try_load_state(state_path)
        if st and isinstance(st.get("next_start_track_idx", None), int):
            start = max(start, int(st["next_start_track_idx"]))
            print(f"[OK] Resume: start_track_idx -> {start} (from {state_path})", flush=True)

    end = n_total
    if int(args.max_tracks) > 0:
        end = min(end, start + int(args.max_tracks))

    slice_groups = groups[start:end]
    n_slice = len(slice_groups)
    if n_slice <= 0:
        raise SystemExit("[ERR] No tracks selected (check start_track_idx/max_tracks/resume).")

    filtered, rejected, rejected_trackgate = [], [], []

    slow_log_path = os.path.join(args.out_dir, "slow_tracks.csv")
    slow_fields = ["global_i","track_id","mask_frame_i","note","bbox_preselect_ov","k_used","per_call_timeout_s"]

    for i_local, (tid, df) in enumerate(slice_groups, start=1):
        i_global = start + i_local
        write_state(state_path, {
            "updated_unix_s": time.time(),
            "current_global_i": i_global,
            "current_track_id": int(tid),
            "next_start_track_idx": i_global - 1,
        })

        n_obs = len(df)
        if n_obs < args.L:
            rejected.append({"track_id": int(tid), "reason": f"too_short(n={n_obs}<L={args.L})"})
            write_state(state_path, {"updated_unix_s": time.time(), "next_start_track_idx": i_global})
            continue

        df = df.sort_values(frame_col)
        frames = df[frame_col].to_numpy(dtype=int)
        areas  = df[area_col].to_numpy(dtype=float)
        Rs     = df[r_col].to_numpy(dtype=float)
        cxs    = df[cx_col].to_numpy(dtype=float)
        cys    = df[cy_col].to_numpy(dtype=float)
        times  = df[t_col].to_numpy(dtype=float) if t_col else None

        ok = (areas >= args.amin_px) & (Rs >= args.rmin_px) & (Rs <= args.rnuc_max)

        nuc_i = None
        for j in range(0, n_obs - args.L + 1):
            if np.all(ok[j:j + args.L]):
                nuc_i = j
                break

        if nuc_i is None:
            rejected.append({"track_id": int(tid), "reason": f"no_stable_run_L{args.L}_after_gates", "n_obs_total": int(n_obs)})
            write_state(state_path, {"updated_unix_s": time.time(), "next_start_track_idx": i_global})
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
            rejected.append({"track_id": int(tid), "reason": "no_mask_match", "nuc_frame_i": nuc_frame, "match_note": match_note})
            write_state(state_path, {"updated_unix_s": time.time(), "next_start_track_idx": i_global})
            continue

        cur_rle = normalize_rle(ann.get("segmentation", None))
        if cur_rle is None:
            rejected.append({"track_id": int(tid), "reason": "no_rle_in_segmentation", "mask_frame_i": int(mask_frame)})
            write_state(state_path, {"updated_unix_s": time.time(), "next_start_track_idx": i_global})
            continue

        overlap_prev = float("nan")
        overlap_note = "skipped_overlap_prev"
        bbox_preselect_ov = float("nan")
        bbox_fallback = float("nan")
        k_used = 0

        if args.overlap_prev_max < 0.999:
            prev_frame = int(mask_frame) - 1
            prev_path = find_json_for_frame(args.json_dir, prev_frame) if prev_frame >= 0 else None
            prev_anns = read_json_list(prev_path)

            # v12: choose prev by bbox-preselect among kclosest
            chosen_prev, bbox_preselect_ov, sel_note, k_used = select_prev_candidate_bboxpreselect(
                cur_ann=ann,
                prev_anns=prev_anns,
                pad_px=float(args.prev_search_pad_px),
                k_closest=int(args.k_closest),
            )

            if sel_note == "prev_empty_or_missing" and args.accept_empty_prev:
                overlap_prev = float(args.empty_prev_overlap_value)
                overlap_note = f"prev_empty_assumed_{overlap_prev:g}"
                chosen_prev = None
            elif sel_note == "prev_no_candidates_nearby" and args.accept_no_candidate_prev:
                overlap_prev = float(args.no_candidate_overlap_value)
                overlap_note = f"prev_nocand_assumed_{overlap_prev:g}"
                chosen_prev = None
            elif chosen_prev is None:
                # can't compute overlap
                overlap_prev = float("nan")
                overlap_note = sel_note

            # if we have a chosen prev, attempt ONE mask-overlap
            if chosen_prev is not None:
                prev_rle = normalize_rle(chosen_prev.get("segmentation", None))
                if prev_rle is None:
                    overlap_prev = float("nan")
                    overlap_note = "prev_no_valid_rle"
                else:
                    t0 = time.time()
                    of, note = overlap_frac_safe_mp(cur_rle, prev_rle, per_call_timeout_s=float(args.overlap_per_call_timeout_s))
                    dt = time.time() - t0

                    if note == "overlap_killed_timeout":
                        overlap_prev = float("nan")
                        overlap_note = "overlap_killed_timeout"
                        append_log_row(slow_log_path, {
                            "global_i": int(i_global),
                            "track_id": int(tid),
                            "mask_frame_i": int(mask_frame),
                            "note": overlap_note,
                            "bbox_preselect_ov": float(bbox_preselect_ov) if np.isfinite(bbox_preselect_ov) else float("nan"),
                            "k_used": int(k_used),
                            "per_call_timeout_s": float(args.overlap_per_call_timeout_s),
                        }, slow_fields)
                    else:
                        overlap_prev = float(of)
                        overlap_note = f"ok_bboxpreselect(k={k_used},bboxov={bbox_preselect_ov:.3f},dt={dt:.3f}s)"

            # bbox fallback on timeout/NaN
            if (not np.isfinite(overlap_prev)) and args.bbox_fallback_on_timeout and chosen_prev is not None:
                only_for_killed = bool(args.bbox_fallback_only_for_killed)
                is_killed_or_timeout = ("killed" in overlap_note) or ("timeout" in overlap_note)
                if (not only_for_killed) or is_killed_or_timeout:
                    bbox_fallback = bbox_overlap_frac(ann.get("bbox", None), chosen_prev.get("bbox", None))
                    if np.isfinite(bbox_fallback):
                        if bbox_fallback <= float(args.bbox_fallback_max):
                            overlap_note = f"bbox_fallback_ok({bbox_fallback:.3f})"
                            # keep overlap_prev as NaN but we "accept" based on fallback
                        else:
                            rejected_trackgate.append({
                                "track_id": int(tid),
                                "reason": f"bbox_overlap_fallback>{args.bbox_fallback_max}",
                                "overlap_note": f"bbox_fallback_reject({bbox_fallback:.3f})",
                                "mask_frame_i": int(mask_frame),
                                "nuc_frame_i": int(nuc_frame),
                                "match_note": match_note,
                            })
                            continue

            # strict handling after fallback
            nan_overlap = (not np.isfinite(overlap_prev)) and (not overlap_note.startswith("bbox_fallback_ok"))
            if nan_overlap and (args.strict_overlap or args.reject_if_overlap_nan):
                rejected_trackgate.append({
                    "track_id": int(tid),
                    "reason": "overlap_nan_or_timeout",
                    "overlap_note": overlap_note,
                    "mask_frame_i": int(mask_frame),
                    "nuc_frame_i": int(nuc_frame),
                    "match_note": match_note,
                })
                continue

            # true overlap gate (only when overlap_prev finite)
            if np.isfinite(overlap_prev) and overlap_prev > float(args.overlap_prev_max):
                rejected_trackgate.append({
                    "track_id": int(tid),
                    "reason": f"overlap_prev>{args.overlap_prev_max}",
                    "overlap_note": overlap_note,
                    "mask_frame_i": int(mask_frame),
                    "nuc_frame_i": int(nuc_frame),
                    "match_note": match_note,
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
            "bbox_preselect_overlap": float(bbox_preselect_ov) if np.isfinite(bbox_preselect_ov) else float("nan"),
            "bbox_overlap_fallback": float(bbox_fallback) if np.isfinite(bbox_fallback) else float("nan"),
            "overlap_note": overlap_note,
            "k_used_prev": int(k_used),
            "bbox_fallback_max": float(args.bbox_fallback_max),
        })

        if (i_local % int(args.progress_every)) == 0 or i_local == n_slice:
            print(f"[PROGRESS] {i_local}/{n_slice} (global {i_global}/{n_total}) | "
                  f"kept={len(filtered)} | rej={len(rejected)} | rej_trackgate={len(rejected_trackgate)}",
                  flush=True)

        if int(args.checkpoint_every) > 0 and (i_local % int(args.checkpoint_every)) == 0:
            write_csv(os.path.join(args.out_dir, f"nucleation_events_filtered_ckpt_{i_global:05d}.csv"), filtered)
            write_csv(os.path.join(args.out_dir, f"nucleation_events_rejected_ckpt_{i_global:05d}.csv"), rejected)
            write_csv(os.path.join(args.out_dir, f"nucleation_events_rejected_trackgate_ckpt_{i_global:05d}.csv"), rejected_trackgate)

        write_state(state_path, {"updated_unix_s": time.time(), "next_start_track_idx": i_global})

    write_csv(os.path.join(args.out_dir, "nucleation_events_filtered.csv"), filtered)
    write_csv(os.path.join(args.out_dir, "nucleation_events_rejected.csv"), rejected)
    write_csv(os.path.join(args.out_dir, "nucleation_events_rejected_trackgate.csv"), rejected_trackgate)

    print(f"[OK] Wrote outputs to: {args.out_dir}", flush=True)
    print(f"[OK] kept={len(filtered)} | rejected={len(rejected)} | rejected_trackgate={len(rejected_trackgate)}", flush=True)
    print(f"[OK] State file: {state_path}", flush=True)


if __name__ == "__main__":
    main()
