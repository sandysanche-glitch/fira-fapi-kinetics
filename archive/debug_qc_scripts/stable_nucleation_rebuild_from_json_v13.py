#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
stable_nucleation_rebuild_from_json_v13.py

v13 = v12 + ROBUST coordinate alignment between tracks.csv and JSON masks

Key addition:
- Auto-estimate per-frame (dx, dy) offset between tracks centroids and JSON bbox-centers
- Smooth offsets with rolling median
- Apply corrected (cx, cy) when matching nuc-frame masks (including lookahead frames)

This fixes the observed issue:
- tracks.cy ~ 1200-1400 px higher than JSON bbox centers (median ~1262)
- which breaks prev-candidate search and overlap gating.

Outputs:
- nucleation_events_filtered.csv
- nucleation_events_rejected.csv
- nucleation_events_rejected_trackgate.csv
- slow_tracks.csv
- state.json
- offsets_estimated.csv
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


# ------------------ small utils ------------------
def parse_frame_and_time_from_name(fname: str) -> Tuple[Optional[int], Optional[float]]:
    m = _FRAME_RE.search(fname)
    if not m:
        return None, None
    try:
        return int(m.group(1)), float(m.group(2))
    except Exception:
        return None, None


def pick_column(df: pd.DataFrame, candidates: List[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"[ERR] Could not find {label} column. Tried: {candidates}. Available: {list(df.columns)}")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def write_csv(path: str, rows: List[Dict[str, Any]]):
    if rows is None:
        rows = []
    if len(rows) == 0:
        # still write header-less empty file? better to write 0-byte?
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    cols = sorted(set().union(*[r.keys() for r in rows]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def try_load_state(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_state(path: str, upd: Dict[str, Any]):
    st = try_load_state(path) or {}
    st.update(upd or {})
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)
    os.replace(tmp, path)


def append_log_row(path: str, row: Dict[str, Any], fieldnames: List[str]):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)


# ------------------ JSON reading & matching ------------------
def find_json_for_frame(json_dir: str, frame_i: int) -> Optional[str]:
    if frame_i < 0:
        return None
    pat = os.path.join(json_dir, f"frame_{frame_i:05d}_*idmapped.json")
    hits = glob.glob(pat)
    if not hits:
        return None
    hits.sort()
    return hits[0]


def read_json_list(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path or (not os.path.exists(path)):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            # keep only dict entries
            return [x for x in obj if isinstance(x, dict)]
        return []
    except Exception:
        return []


def bbox_center(bbox) -> Tuple[float, float]:
    if bbox is None or (not isinstance(bbox, (list, tuple))) or len(bbox) < 4:
        return float("nan"), float("nan")
    x, y, w, h = bbox[:4]
    try:
        return float(x) + 0.5 * float(w), float(y) + 0.5 * float(h)
    except Exception:
        return float("nan"), float("nan")


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


# ------------------ RLE normalize ------------------
def normalize_rle(seg) -> Optional[Dict[str, Any]]:
    if seg is None:
        return None
    if isinstance(seg, dict) and "counts" in seg and "size" in seg:
        # pycocotools expects counts bytes sometimes
        rle = dict(seg)
        if isinstance(rle.get("counts", None), str):
            rle["counts"] = rle["counts"].encode("utf-8")
        return rle
    return None


# ------------------ SAFE overlap in subprocess ------------------
def _overlap_worker(q, cur_rle, prev_rle):
    try:
        inter = mask_utils.merge([cur_rle, prev_rle], intersect=True)
        ai = float(mask_utils.area(inter))
        aa = float(mask_utils.area(cur_rle))
        q.put(ai / aa if aa > 0 else float("nan"))
    except Exception:
        q.put(float("nan"))


def overlap_frac_safe_mp(cur_rle, prev_rle, per_call_timeout_s: float):
    q = mp.Queue(maxsize=1)
    p = mp.Process(target=_overlap_worker, args=(q, cur_rle, prev_rle))
    p.daemon = True
    p.start()
    p.join(timeout=float(per_call_timeout_s))
    if p.is_alive():
        try:
            p.terminate()
        except Exception:
            pass
        try:
            p.join(timeout=0.05)
        except Exception:
            pass
        return float("nan"), "overlap_killed_timeout"
    try:
        v = q.get_nowait()
    except Exception:
        v = float("nan")
    return float(v), "ok"


# ------------------ bbox overlap fallback ------------------
def bbox_overlap_frac(cur_bbox, prev_bbox) -> float:
    """
    Intersection area / cur_bbox area
    """
    if cur_bbox is None or prev_bbox is None:
        return float("nan")
    try:
        x1, y1, w1, h1 = [float(x) for x in cur_bbox[:4]]
        x2, y2, w2, h2 = [float(x) for x in prev_bbox[:4]]
    except Exception:
        return float("nan")
    if w1 <= 0 or h1 <= 0:
        return float("nan")
    ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
    bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    return float(inter / (w1 * h1))


# ------------------ v12 prev selection ------------------
def select_prev_candidate_bboxpreselect(cur_ann: Dict[str, Any],
                                        prev_anns: List[Dict[str, Any]],
                                        pad_px: float,
                                        k_closest: int) -> Tuple[Optional[Dict[str, Any]], float, str, int]:
    """
    1) Filter prev anns to those whose bbox centers lie within expanded cur bbox (+pad)
    2) Take k closest to cur bbox center
    3) Compute bbox overlap frac for each (inter / cur area)
    4) Choose the one with MIN bbox overlap (conservative) among those k
    """
    if not prev_anns:
        return None, float("nan"), "prev_empty_or_missing", 0

    cur_bbox = cur_ann.get("bbox", None)
    if cur_bbox is None or (not isinstance(cur_bbox, (list, tuple))) or len(cur_bbox) < 4:
        return None, float("nan"), "cur_no_bbox", 0

    cx, cy = bbox_center(cur_bbox)
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return None, float("nan"), "cur_bbox_center_nan", 0

    x, y, w, h = [float(v) for v in cur_bbox[:4]]
    x1 = x - float(pad_px)
    y1 = y - float(pad_px)
    x2 = x + w + float(pad_px)
    y2 = y + h + float(pad_px)

    candidates = []
    for a in prev_anns:
        bx, by = bbox_center(a.get("bbox", None))
        if not (np.isfinite(bx) and np.isfinite(by)):
            continue
        if (bx >= x1) and (bx <= x2) and (by >= y1) and (by <= y2):
            d = math.hypot(bx - cx, by - cy)
            candidates.append((d, a))

    if not candidates:
        return None, float("nan"), "prev_no_candidates_nearby", 0

    candidates.sort(key=lambda t: t[0])
    k = max(1, int(k_closest))
    top = candidates[:k]
    # choose min bbox overlap among top-k
    best_ann, best_ov = None, None
    for _, a in top:
        ov = bbox_overlap_frac(cur_bbox, a.get("bbox", None))
        if not np.isfinite(ov):
            continue
        if best_ann is None or ov < best_ov:
            best_ann, best_ov = a, ov

    if best_ann is None:
        return None, float("nan"), "prev_candidates_no_bbox_overlap", len(top)

    return best_ann, float(best_ov), "ok_bboxpreselect", len(top)


# ------------------ NEW: offset estimation ------------------
def estimate_offsets(tracks: pd.DataFrame,
                     json_dir: str,
                     frame_col: str,
                     cx_col: str,
                     cy_col: str,
                     min_tracks_per_frame: int = 20,
                     frame_min: Optional[int] = None,
                     frame_max: Optional[int] = None,
                     use_prev_if_empty: bool = True) -> pd.DataFrame:
    """
    For each frame f:
      dx = median(tracks.cx@f) - median(json_bbox_center_x@f)
      dy = median(tracks.cy@f) - median(json_bbox_center_y@f)

    If JSON list is empty for frame f and use_prev_if_empty=True:
      use frame f-1 JSON to estimate (still store under f).
    """
    frames = sorted(tracks[frame_col].dropna().astype(int).unique().tolist())
    if frame_min is not None:
        frames = [f for f in frames if f >= int(frame_min)]
    if frame_max is not None:
        frames = [f for f in frames if f <= int(frame_max)]

    rows = []
    for f in frames:
        sub = tracks[tracks[frame_col].astype(int) == int(f)]
        if len(sub) < int(min_tracks_per_frame):
            continue

        jp = find_json_for_frame(json_dir, int(f))
        anns = read_json_list(jp)
        used_frame = int(f)
        if (not anns) and use_prev_if_empty:
            jp2 = find_json_for_frame(json_dir, int(f) - 1)
            anns = read_json_list(jp2)
            used_frame = int(f) - 1

        if not anns:
            continue

        bx = []
        by = []
        for a in anns:
            x, y = bbox_center(a.get("bbox", None))
            if np.isfinite(x) and np.isfinite(y):
                bx.append(x)
                by.append(y)
        if len(bx) < 5:
            continue

        tr_cx = float(np.median(sub[cx_col].astype(float).to_numpy()))
        tr_cy = float(np.median(sub[cy_col].astype(float).to_numpy()))
        js_cx = float(np.median(np.array(bx, dtype=float)))
        js_cy = float(np.median(np.array(by, dtype=float)))

        rows.append({
            "frame_i": int(f),
            "used_json_frame_i": int(used_frame),
            "tracks_cx_med": tr_cx,
            "tracks_cy_med": tr_cy,
            "json_bbox_cx_med": js_cx,
            "json_bbox_cy_med": js_cy,
            "dx_tracks_minus_json": tr_cx - js_cx,
            "dy_tracks_minus_json": tr_cy - js_cy,
            "n_tracks": int(len(sub)),
            "n_json": int(len(bx)),
        })

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    out = out.sort_values("frame_i").reset_index(drop=True)
    return out


def smooth_offsets(df_off: pd.DataFrame, smooth_window: int = 21) -> pd.DataFrame:
    """
    Rolling median smoothing on dx/dy over frame index order.
    """
    if df_off is None or len(df_off) == 0:
        return df_off
    w = max(1, int(smooth_window))
    df = df_off.copy()
    df["dx_smooth"] = df["dx_tracks_minus_json"].rolling(w, center=True, min_periods=max(3, w // 3)).median()
    df["dy_smooth"] = df["dy_tracks_minus_json"].rolling(w, center=True, min_periods=max(3, w // 3)).median()
    # fill edges with raw if needed
    df["dx_smooth"] = df["dx_smooth"].fillna(df["dx_tracks_minus_json"])
    df["dy_smooth"] = df["dy_smooth"].fillna(df["dy_tracks_minus_json"])
    return df


def offsets_to_dict(df_off: pd.DataFrame) -> Dict[int, Tuple[float, float]]:
    """
    frame_i -> (dx, dy) where dx/dy = tracks - json.
    To convert a track centroid into JSON coords:
      cx_json = cx_track - dx
      cy_json = cy_track - dy
    """
    out = {}
    if df_off is None or len(df_off) == 0:
        return out
    for _, r in df_off.iterrows():
        fi = int(r["frame_i"])
        dx = float(r.get("dx_smooth", r.get("dx_tracks_minus_json", 0.0)))
        dy = float(r.get("dy_smooth", r.get("dy_tracks_minus_json", 0.0)))
        out[fi] = (dx, dy)
    return out


def corrected_xy(frame_i: int, cx: float, cy: float, off: Dict[int, Tuple[float, float]]):
    if off and (int(frame_i) in off):
        dx, dy = off[int(frame_i)]
        return float(cx - dx), float(cy - dy)
    return float(cx), float(cy)


def get_first_usable_match(json_dir: str, start_frame: int, lookahead_k: int,
                           cx_track: float, cy_track: float, max_dist: float,
                           offsets: Dict[int, Tuple[float, float]]):
    """
    Same as v12 but applies per-frame offset correction to (cx, cy) before matching.
    """
    for j in range(0, lookahead_k + 1):
        f = start_frame + j
        jp = find_json_for_frame(json_dir, f)
        anns = read_json_list(jp)
        if not anns:
            continue
        cx, cy = corrected_xy(f, cx_track, cy_track, offsets)
        ann, _ = match_det_by_centroid(anns, cx, cy, max_dist)
        if ann is None:
            continue
        return f, jp, ann, ("ok" if j == 0 else f"lookahead+{j}")
    return None, None, None, "no_match_in_lookahead"


# ------------------ MAIN ------------------
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
    ap.add_argument("--k_closest", type=int, default=5)

    ap.add_argument("--overlap_timeout_s", type=float, default=3.0)
    ap.add_argument("--overlap_per_call_timeout_s", type=float, default=0.30)

    ap.add_argument("--accept_empty_prev", action="store_true")
    ap.add_argument("--empty_prev_overlap_value", type=float, default=0.0)
    ap.add_argument("--accept_no_candidate_prev", action="store_true")
    ap.add_argument("--no_candidate_overlap_value", type=float, default=0.0)

    ap.add_argument("--bbox_fallback_on_timeout", action="store_true")
    ap.add_argument("--bbox_fallback_only_for_killed", action="store_true")
    ap.add_argument("--bbox_fallback_max", type=float, default=0.7)

    # NEW offset controls
    ap.add_argument("--auto_offset", action="store_true", help="Estimate per-frame offsets between tracks and JSON (recommended for TEMPO).")
    ap.add_argument("--offset_min_tracks", type=int, default=20)
    ap.add_argument("--offset_frame_min", type=int, default=None)
    ap.add_argument("--offset_frame_max", type=int, default=None)
    ap.add_argument("--offset_smooth_window", type=int, default=21)
    ap.add_argument("--offset_use_prev_if_empty", action="store_true")
    ap.add_argument("--offset_manual_dx", type=float, default=None)
    ap.add_argument("--offset_manual_dy", type=float, default=None)

    ap.add_argument("--progress_every", type=int, default=10)
    ap.add_argument("--checkpoint_every", type=int, default=25)

    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--state_path", default=None)

    ap.add_argument("--start_track_idx", type=int, default=0)
    ap.add_argument("--max_tracks", type=int, default=-1)

    args = ap.parse_args()

    ensure_dir(args.out_dir)
    state_path = args.state_path or os.path.join(args.out_dir, "state.json")

    print(f"[OK] Reading tracks: {args.tracks_csv}", flush=True)
    tracks = pd.read_csv(args.tracks_csv)

    track_id_col = pick_column(tracks, ["track_id", "tid", "id"], "track_id")
    frame_col    = pick_column(tracks, ["frame_idx", "frame", "frame_i", "frame_id"], "frame index")
    cx_col       = pick_column(tracks, ["cx", "cx_px", "x", "x_px"], "cx")
    cy_col       = pick_column(tracks, ["cy", "cy_px", "y", "y_px"], "cy")
    area_col     = pick_column(tracks, ["area_px", "area", "A_px", "A"], "area")
    r_col        = pick_column(tracks, ["R_px", "R", "r_px", "r"], "radius")
    t_col        = tracks.columns.intersection(["time_ms", "t_ms", "time", "t"]).tolist()[0] if len(tracks.columns.intersection(["time_ms", "t_ms", "time", "t"])) > 0 else None

    # -------- offsets --------
    offsets = {}
    df_off = pd.DataFrame()

    if args.offset_manual_dx is not None or args.offset_manual_dy is not None:
        # global constant offset
        dx = float(args.offset_manual_dx) if args.offset_manual_dx is not None else 0.0
        dy = float(args.offset_manual_dy) if args.offset_manual_dy is not None else 0.0
        frames_unique = tracks[frame_col].dropna().astype(int).unique().tolist()
        offsets = {int(f): (dx, dy) for f in frames_unique}
        print(f"[OK] Using MANUAL constant offset: dx={dx:.3f}, dy={dy:.3f}", flush=True)

    elif args.auto_offset:
        print("[OK] Estimating per-frame offsets (tracks vs JSON bbox centers)...", flush=True)
        df_off = estimate_offsets(
            tracks=tracks,
            json_dir=args.json_dir,
            frame_col=frame_col,
            cx_col=cx_col,
            cy_col=cy_col,
            min_tracks_per_frame=int(args.offset_min_tracks),
            frame_min=args.offset_frame_min,
            frame_max=args.offset_frame_max,
            use_prev_if_empty=bool(args.offset_use_prev_if_empty),
        )
        if len(df_off) == 0:
            print("[WARN] Offset estimation produced 0 frames. Proceeding with NO correction.", flush=True)
        else:
            df_off = smooth_offsets(df_off, smooth_window=int(args.offset_smooth_window))
            offsets = offsets_to_dict(df_off)
            off_path = os.path.join(args.out_dir, "offsets_estimated.csv")
            df_off.to_csv(off_path, index=False)
            med_dx = float(np.median(df_off["dx_smooth"].to_numpy()))
            med_dy = float(np.median(df_off["dy_smooth"].to_numpy()))
            print(f"[OK] Offset estimation done. median dx={med_dx:.2f}, dy={med_dy:.2f}. Saved: {off_path}", flush=True)

    else:
        print("[OK] Offsets disabled (no correction).", flush=True)

    # -------- group tracks --------
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
            cx_track=float(cxs[nuc_i]),
            cy_track=float(cys[nuc_i]),
            max_dist=float(max_dist),
            offsets=offsets,
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
                overlap_prev = float("nan")
                overlap_note = sel_note

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
                        else:
                            rejected_trackgate.append({
                                "track_id": int(tid),
                                "reason": f"bbox_overlap_fallback>{args.bbox_fallback_max}",
                                "overlap_note": f"bbox_fallback_reject({bbox_fallback:.3f})",
                                "mask_frame_i": int(mask_frame),
                                "nuc_frame_i": int(nuc_frame),
                                "match_note": match_note,
                            })
                            write_state(state_path, {"updated_unix_s": time.time(), "next_start_track_idx": i_global})
                            continue

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
                write_state(state_path, {"updated_unix_s": time.time(), "next_start_track_idx": i_global})
                continue

            if np.isfinite(overlap_prev) and overlap_prev > float(args.overlap_prev_max):
                rejected_trackgate.append({
                    "track_id": int(tid),
                    "reason": f"overlap_prev>{args.overlap_prev_max}",
                    "overlap_note": overlap_note,
                    "mask_frame_i": int(mask_frame),
                    "nuc_frame_i": int(nuc_frame),
                    "match_note": match_note,
                })
                write_state(state_path, {"updated_unix_s": time.time(), "next_start_track_idx": i_global})
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