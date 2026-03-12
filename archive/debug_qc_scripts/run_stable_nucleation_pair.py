# stable_nucleation_rebuild_from_json.py
# Build "stable nucleation" events from tracks.csv + per-frame JSON masks,
# and reject late "split/ID-switch" artifacts using a mask-overlap gate.

import os
import re
import glob
import json
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

# --- requires pycocotools for robust COCO RLE decode ---
try:
    from pycocotools import mask as mask_utils
except Exception as e:
    mask_utils = None


def die(msg: str):
    raise SystemExit(f"[FATAL] {msg}")


def ensure_pycocotools():
    if mask_utils is None:
        die(
            "pycocotools is required for decoding COCO RLE.\n"
            "Install (conda env): pip install pycocotools\n"
            "If Windows build issues: pip install pycocotools-windows"
        )


def _as_int(x):
    if pd.isna(x):
        return None
    try:
        return int(x)
    except Exception:
        try:
            return int(str(x))
        except Exception:
            return None


def infer_columns(df: pd.DataFrame):
    """
    Unify FAPI vs TEMPO column conventions.
    FAPI has: frame, frame_id, time_ms, annotation_id, area_px, R_px
    TEMPO has: frame_idx, t_ms, det_id, area_px, R_px, R_mono (optional)
    """
    # frame index
    if "frame" in df.columns:
        frame_col = "frame"
    elif "frame_idx" in df.columns:
        frame_col = "frame_idx"
    else:
        die("tracks.csv must contain either 'frame' or 'frame_idx' column.")

    # time
    if "time_ms" in df.columns:
        time_col = "time_ms"
    elif "t_ms" in df.columns:
        time_col = "t_ms"
    else:
        die("tracks.csv must contain either 'time_ms' or 't_ms' column.")

    # detection/annotation id (for mask lookup inside JSON)
    det_col = None
    for c in ["annotation_id", "det_id", "ann_id", "id"]:
        if c in df.columns:
            det_col = c
            break
    if det_col is None:
        # You *can* still run without det_id lookup if you only use track-based gates,
        # but overlap-gate needs a per-detection mask id.
        die("tracks.csv must contain an annotation id column (annotation_id or det_id).")

    # frame_id string (useful for FAPI)
    frame_id_col = "frame_id" if "frame_id" in df.columns else None

    # required geometry
    if "area_px" not in df.columns or "R_px" not in df.columns:
        die("tracks.csv must contain 'area_px' and 'R_px' columns.")

    return frame_col, time_col, det_col, frame_id_col


def find_json_for_frame(
    json_dir: str,
    frame_idx: int,
    frame_id: Optional[str] = None,
) -> Optional[str]:
    """
    Try hard to locate the JSON corresponding to a frame.

    Typical patterns seen:
      frame_00065_t130.00ms.json
      frame_00065_t130.00ms_idmapped.json
      frame_00065_....json
    """
    if frame_id:
        # exact-ish match on frame_id prefix
        cand = glob.glob(os.path.join(json_dir, f"{frame_id}*.json"))
        if cand:
            return sorted(cand)[0]

    # fallback: match by zero-padded frame index
    # most of your files use 5 digits
    pref = f"frame_{int(frame_idx):05d}_"
    cand = glob.glob(os.path.join(json_dir, f"{pref}*.json"))
    if cand:
        return sorted(cand)[0]

    # last attempt: any json containing that frame index number
    cand = glob.glob(os.path.join(json_dir, f"*{int(frame_idx):05d}*.json"))
    if cand:
        return sorted(cand)[0]

    return None


def parse_json_list(path: str) -> List[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "annotations" in data:
        data = data["annotations"]
    if not isinstance(data, list):
        return []
    return data


def ann_id_from_obj(a: dict) -> Optional[int]:
    # try a few key conventions
    for k in ["annotation_id", "det_id", "id", "ann_id"]:
        if k in a:
            v = _as_int(a.get(k))
            if v is not None:
                return v
    return None


def ann_to_rle(a: dict):
    """
    Return (rle, h, w) usable by pycocotools.decode.
    Supports:
      - segmentation as RLE dict with 'counts' and 'size'
      - segmentation as polygons list (via frPyObjects)
    """
    seg = a.get("segmentation", None)
    if seg is None:
        return None, None, None

    # RLE dict
    if isinstance(seg, dict) and "counts" in seg:
        size = seg.get("size", None)
        if size and len(size) == 2:
            h, w = int(size[0]), int(size[1])
        else:
            # sometimes stored at top-level
            h = a.get("height", None)
            w = a.get("width", None)
            if h is None or w is None:
                return None, None, None
            h, w = int(h), int(w)
        return seg, h, w

    # polygons list (COCO style)
    if isinstance(seg, list):
        # need image size from top-level keys
        h = a.get("height", None)
        w = a.get("width", None)
        if h is None or w is None:
            return None, None, None
        h, w = int(h), int(w)
        rles = mask_utils.frPyObjects(seg, h, w)
        rle = mask_utils.merge(rles)
        return rle, h, w

    return None, None, None


def decode_mask(rle) -> np.ndarray:
    m = mask_utils.decode(rle)
    # decode can return HxWx1
    if m.ndim == 3:
        m = m[:, :, 0]
    return (m > 0).astype(np.uint8)


@dataclass
class NucEvent:
    track_id: int
    nuc_frame: int
    nuc_time_ms: float
    det_id: int
    area_nuc_px: float
    R_nuc_px: float
    overlap_prev: float
    rmono: Optional[float]


def find_stable_nucleation_events(
    tracks: pd.DataFrame,
    L: int,
    amin_px: float,
    rmin_px: float,
    use_rmono_gate: bool,
    rmono_min: float,
    rnuc_max: Optional[float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pure track-based "stable nucleation" definition:
      nucleation time = first frame where a detection is followed by >=L-1 consecutive frames,
      and all frames in that L-run satisfy area>=amin and R>=rmin (and optionally R_mono>=rmono_min).
    Returns:
      accepted_df (one row per track) and rejected_df with reasons.
    """
    frame_col, time_col, det_col, frame_id_col = infer_columns(tracks)

    # normalize types
    t = tracks.copy()
    t[frame_col] = t[frame_col].astype(int)
    t["track_id"] = t["track_id"].astype(int)
    t[time_col] = t[time_col].astype(float)
    t["area_px"] = t["area_px"].astype(float)
    t["R_px"] = t["R_px"].astype(float)
    t[det_col] = t[det_col].apply(_as_int)

    if use_rmono_gate and "R_mono" not in t.columns:
        # allow running, but gate can't be applied
        use_rmono_gate = False

    accepted: List[dict] = []
    rejected: List[dict] = []

    for tid, g in t.groupby("track_id", sort=True):
        g = g.sort_values(frame_col).reset_index(drop=True)

        frames = g[frame_col].to_numpy()
        areas = g["area_px"].to_numpy()
        Rs = g["R_px"].to_numpy()
        times = g[time_col].to_numpy()
        dets = g[det_col].to_numpy()

        rmono_arr = g["R_mono"].to_numpy() if ("R_mono" in g.columns) else None
        frame_id_arr = g[frame_id_col].to_numpy() if frame_id_col else None

        # quick sanity
        if len(g) < L:
            rejected.append({"track_id": tid, "reason": f"len<{L}"})
            continue

        # find first index i that starts a run of L consecutive frames
        found = False
        for i in range(0, len(g) - L + 1):
            run_frames = frames[i : i + L]
            # consecutive?
            if not np.all(run_frames == run_frames[0] + np.arange(L)):
                continue

            run_areas = areas[i : i + L]
            run_R = Rs[i : i + L]
            if np.any(run_areas < amin_px) or np.any(run_R < rmin_px):
                continue

            if use_rmono_gate and rmono_arr is not None:
                run_rmono = rmono_arr[i : i + L]
                if np.any(run_rmono < rmono_min):
                    continue

            # nucleation event at i
            nuc_frame = int(frames[i])
            nuc_time = float(times[i])
            det_id = dets[i]
            if det_id is None:
                rejected.append({"track_id": tid, "reason": "missing det_id at nuc"})
                found = True  # stop searching: track is "badly formed"
                break

            R_nuc = float(Rs[i])
            if rnuc_max is not None and R_nuc > rnuc_max:
                rejected.append({"track_id": tid, "reason": f"R_nuc>{rnuc_max}"})
                found = True
                break

            accepted.append(
                {
                    "track_id": tid,
                    "nuc_frame": nuc_frame,
                    "nuc_time_ms": nuc_time,
                    "det_id": int(det_id),
                    "area_nuc_px": float(areas[i]),
                    "R_nuc_px": R_nuc,
                    "R_mono_at_nuc": float(rmono_arr[i]) if (rmono_arr is not None) else np.nan,
                    "frame_id": str(frame_id_arr[i]) if (frame_id_arr is not None) else "",
                }
            )
            found = True
            break

        if not found:
            rejected.append({"track_id": tid, "reason": "no stable run found"})

    return pd.DataFrame(accepted), pd.DataFrame(rejected)


def apply_overlap_gate(
    events: pd.DataFrame,
    json_dir: str,
    overlap_prev_max: float,
    cache_union: Dict[int, np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reject events whose nucleation mask overlaps too much with the union mask
    of the *previous* frame (already-transformed region).

    overlap_prev = area(nuc_mask & union(prev))/area(nuc_mask)
    """
    ensure_pycocotools()

    acc = []
    rej = []

    for _, e in events.iterrows():
        nuc_frame = int(e["nuc_frame"])
        det_id = _as_int(e["det_id"])
        frame_id = str(e.get("frame_id", "")).strip() or None

        js_nuc = find_json_for_frame(json_dir, nuc_frame, frame_id)
        js_prev = find_json_for_frame(json_dir, nuc_frame - 1, None)

        if js_nuc is None or det_id is None:
            rej.append({**e.to_dict(), "reason": "missing nuc json or det_id", "overlap_prev": np.nan})
            continue

        anns = parse_json_list(js_nuc)
        target = None
        for a in anns:
            aid = ann_id_from_obj(a)
            if aid == det_id:
                target = a
                break

        if target is None:
            # if id didn't match, this usually means a mismatch between tracking ids and json ids
            rej.append({**e.to_dict(), "reason": "det_id not found in nuc json", "overlap_prev": np.nan})
            continue

        rle, h, w = ann_to_rle(target)
        if rle is None:
            rej.append({**e.to_dict(), "reason": "no segmentation in target ann", "overlap_prev": np.nan})
            continue

        nuc_mask = decode_mask(rle)
        nuc_area = float(nuc_mask.sum())
        if nuc_area <= 0:
            rej.append({**e.to_dict(), "reason": "empty nuc mask", "overlap_prev": np.nan})
            continue

        # union mask for previous frame
        if nuc_frame - 1 < 0 or js_prev is None:
            # no previous frame → treat as no overlap
            overlap = 0.0
        else:
            if (nuc_frame - 1) in cache_union:
                union_prev = cache_union[nuc_frame - 1]
            else:
                prev_anns = parse_json_list(js_prev)
                union_prev = np.zeros_like(nuc_mask, dtype=np.uint8)
                for a in prev_anns:
                    rle2, _, _ = ann_to_rle(a)
                    if rle2 is None:
                        continue
                    m2 = decode_mask(rle2)
                    # OR
                    union_prev |= (m2 > 0).astype(np.uint8)
                cache_union[nuc_frame - 1] = union_prev

            inter = float((nuc_mask & union_prev).sum())
            overlap = inter / nuc_area

        row = e.to_dict()
        row["overlap_prev"] = overlap

        if overlap > overlap_prev_max:
            row["reason"] = f"overlap_prev>{overlap_prev_max}"
            rej.append(row)
        else:
            acc.append(row)

    return pd.DataFrame(acc), pd.DataFrame(rej)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks_csv", required=True)
    ap.add_argument("--json_dir", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--L", type=int, default=5)
    ap.add_argument("--amin_px", type=float, default=800.0)
    ap.add_argument("--rmin_px", type=float, default=3.0)

    ap.add_argument("--use_rmono_gate", action="store_true")
    ap.add_argument("--rmono_min", type=float, default=0.6)
    ap.add_argument("--rnuc_max", type=float, default=None)

    # overlap artifact gate
    ap.add_argument("--overlap_prev_max", type=float, default=0.5)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[OK] Reading tracks:", args.tracks_csv)
    df = pd.read_csv(args.tracks_csv)

    # 1) stable nucleation from tracks
    events0, rej0 = find_stable_nucleation_events(
        df,
        L=args.L,
        amin_px=args.amin_px,
        rmin_px=args.rmin_px,
        use_rmono_gate=args.use_rmono_gate,
        rmono_min=args.rmono_min,
        rnuc_max=args.rnuc_max,
    )

    # 2) overlap gate from JSON masks
    cache_union: Dict[int, np.ndarray] = {}
    events, rej_overlap = apply_overlap_gate(
        events0,
        json_dir=args.json_dir,
        overlap_prev_max=args.overlap_prev_max,
        cache_union=cache_union,
    )

    # outputs
    out_events = os.path.join(args.out_dir, "nucleation_events_filtered.csv")
    out_rej = os.path.join(args.out_dir, "nucleation_events_rejected.csv")
    out_rej0 = os.path.join(args.out_dir, "nucleation_events_rejected_trackgate.csv")

    events.to_csv(out_events, index=False)
    rej_overlap.to_csv(out_rej, index=False)
    rej0.to_csv(out_rej0, index=False)

    print("\n[OK] Wrote:")
    print(" ", out_events)
    print(" ", out_rej)
    print(" ", out_rej0)

    # quick summary
    def med(x): return float(np.nanmedian(np.asarray(x, dtype=float))) if len(x) else np.nan

    print("\n=== Summary ===")
    print(f"accepted: {len(events)} / initial stable candidates: {len(events0)}")
    if len(events):
        print(f"R_nuc median: {med(events['R_nuc_px']) :.3g} px | overlap_prev median: {med(events['overlap_prev']) :.3g}")
        if "R_mono_at_nuc" in events.columns:
            print(f"R_mono@birth median: {med(events['R_mono_at_nuc']) :.3g}")

    if len(rej_overlap):
        print(f"rejected by overlap: {len(rej_overlap)} (overlap_prev_max={args.overlap_prev_max})")


if __name__ == "__main__":
    main()
