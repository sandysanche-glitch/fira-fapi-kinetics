#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pycocotools import mask as mask_utils


# ----------------------------
# Utilities: robust RLE decoding
# ----------------------------

def _normalize_coco_rle(seg: dict) -> dict:
    """
    Normalize COCO RLE segmentation to something pycocotools.decode accepts.
    Handles:
      - counts as str (common in JSON) -> bytes
      - counts as bytes/bytearray
      - counts as list[int] (uncompressed RLE) -> frPyObjects + merge
    """
    if not isinstance(seg, dict):
        raise ValueError(f"segmentation is not dict (type={type(seg)})")
    if "size" not in seg or "counts" not in seg:
        raise ValueError(f"segmentation dict missing size/counts keys: keys={list(seg.keys())}")

    size = seg["size"]
    counts = seg["counts"]

    if not (isinstance(size, (list, tuple)) and len(size) == 2):
        raise ValueError(f"bad size field: {size}")

    # JSON commonly stores compressed RLE counts as a python string
    if isinstance(counts, str):
        try:
            counts_b = counts.encode("ascii")
        except Exception:
            counts_b = counts.encode("latin-1", errors="ignore")
        return {"size": list(size), "counts": counts_b}

    if isinstance(counts, (bytes, bytearray)):
        return {"size": list(size), "counts": bytes(counts)}

    if isinstance(counts, list):
        rle = {"size": list(size), "counts": counts}
        rles = mask_utils.frPyObjects(rle, size[0], size[1])
        rle_m = mask_utils.merge(rles)
        return rle_m

    raise ValueError(f"Unsupported counts type: {type(counts)}")


def decode_mask(seg: dict) -> np.ndarray:
    """
    Returns boolean mask (H,W). Raises ValueError if cannot decode.
    """
    rle = _normalize_coco_rle(seg)
    m = mask_utils.decode(rle)
    if m.ndim == 3:
        m = m[:, :, 0]
    return (m > 0)


# ----------------------------
# JSON frame indexing helpers
# ----------------------------

_FRAME_RE = re.compile(r"frame_(\d+)_t([0-9.]+)ms", re.IGNORECASE)

def build_frame_index(json_dir: str) -> Dict[int, str]:
    """
    Build mapping: frame_idx -> filepath for files like:
      frame_00018_t36.00ms_idmapped.json
    Keeps the first match per frame_idx.
    """
    mapping = {}
    fs = sorted(glob.glob(os.path.join(json_dir, "frame_*_idmapped.json")))
    for f in fs:
        base = os.path.basename(f)
        m = _FRAME_RE.search(base)
        if not m:
            continue
        idx = int(m.group(1))
        if idx not in mapping:
            mapping[idx] = f
    return mapping


def find_json_for_frame(json_dir: str,
                        frame_idx: Optional[int] = None,
                        frame_id: Optional[str] = None) -> Optional[str]:
    """
    Try to locate the JSON file for a given frame.
    - If frame_id given (e.g. "frame_00065_t130.00ms"), use glob frame_id*_idmapped.json
    - Else if frame_idx given, use prefix match frame_{idx:05d}_*_idmapped.json
    """
    if frame_id:
        pat = os.path.join(json_dir, f"{frame_id}*_idmapped.json")
        hits = glob.glob(pat)
        if hits:
            return sorted(hits)[0]

    if frame_idx is not None:
        pat = os.path.join(json_dir, f"frame_{int(frame_idx):05d}_*_idmapped.json")
        hits = glob.glob(pat)
        if hits:
            return sorted(hits)[0]

    return None


def load_frame_json(path: str) -> List[dict]:
    """
    Load a frame JSON file. Returns list of annotations (can be empty).
    """
    try:
        with open(path, "r") as f:
            d = json.load(f)
        if isinstance(d, list):
            return d
        # if somehow wrapped, try annotations key
        if isinstance(d, dict) and "annotations" in d and isinstance(d["annotations"], list):
            return d["annotations"]
    except Exception:
        pass
    return []


# ----------------------------
# Matching track row -> JSON annotation
# ----------------------------

def ann_bbox_center(ann: dict) -> Optional[Tuple[float, float]]:
    bb = ann.get("bbox", None)
    if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
        return None
    x, y, w, h = map(float, bb)
    return (x + 0.5 * w, y + 0.5 * h)


def find_annotation_in_frame(anns: List[dict],
                             prefer_id: Optional[int],
                             cx: Optional[float],
                             cy: Optional[float],
                             max_dist_px: float = 10.0) -> Optional[dict]:
    """
    Find the nucleus annotation in a list of frame annotations.

    Priority:
      1) ID match: ann['id'] == prefer_id
      2) centroid match: nearest bbox center to (cx,cy) within max_dist_px
    """
    if prefer_id is not None:
        for a in anns:
            if "id" in a:
                try:
                    if int(a["id"]) == int(prefer_id):
                        return a
                except Exception:
                    pass

    if cx is None or cy is None:
        return None

    best = None
    best_d = None
    for a in anns:
        c = ann_bbox_center(a)
        if c is None:
            continue
        dx = c[0] - cx
        dy = c[1] - cy
        d = float(np.hypot(dx, dy))
        if best_d is None or d < best_d:
            best_d = d
            best = a

    if best is not None and best_d is not None and best_d <= max_dist_px:
        return best
    return None


# ----------------------------
# Stable nucleation event detection from tracks
# ----------------------------

@dataclass
class TrackColumns:
    track_id: str
    time_ms: str
    frame_idx: Optional[str]
    frame_id: Optional[str]
    det_id: Optional[str]
    cx: Optional[str]
    cy: Optional[str]
    area_px: str
    R_px: str
    R_mono: Optional[str]


def infer_tracks_schema(df: pd.DataFrame) -> TrackColumns:
    cols = set(df.columns)

    # common
    if "track_id" not in cols:
        raise KeyError("tracks.csv must contain 'track_id' column.")

    # time
    time_col = "time_ms" if "time_ms" in cols else ("t_ms" if "t_ms" in cols else None)
    if time_col is None:
        raise KeyError("tracks.csv must contain 'time_ms' (FAPI) or 't_ms' (TEMPO).")

    # frame identifiers
    frame_idx = "frame_idx" if "frame_idx" in cols else None
    frame_id  = "frame_id" if "frame_id" in cols else None
    if frame_idx is None and frame_id is None:
        # fallback: some have "frame"
        if "frame" in cols:
            # often this is string like frame_00010_t20.00ms
            frame_id = "frame"
        else:
            raise KeyError("tracks.csv must contain 'frame_idx' (TEMPO) or 'frame_id'/'frame' (FAPI).")

    # detection/annotation id
    det_id = None
    for c in ["det_id", "annotation_id", "id"]:
        if c in cols:
            det_id = c
            break

    # geometry
    cx = "cx" if "cx" in cols else None
    cy = "cy" if "cy" in cols else None

    # size
    if "area_px" not in cols:
        raise KeyError("tracks.csv must contain 'area_px'.")
    if "R_px" not in cols:
        raise KeyError("tracks.csv must contain 'R_px'.")

    rmono = "R_mono" if "R_mono" in cols else None

    return TrackColumns(
        track_id="track_id",
        time_ms=time_col,
        frame_idx=frame_idx,
        frame_id=frame_id,
        det_id=det_id,
        cx=cx,
        cy=cy,
        area_px="area_px",
        R_px="R_px",
        R_mono=rmono
    )


def find_stable_nucleation_events(df: pd.DataFrame,
                                 tc: TrackColumns,
                                 L: int,
                                 amin_px: float,
                                 rmin_px: float,
                                 use_rmono_gate: bool,
                                 rmono_min: float,
                                 rnuc_max: Optional[float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      events_df: one row per accepted stable nucleation (pre-overlap gate)
      rej_df: rejected by basic/stability gates (pre-overlap)
    """
    events = []
    rejected = []

    # ensure numeric
    d = df.copy()
    d[tc.time_ms] = pd.to_numeric(d[tc.time_ms], errors="coerce")
    d[tc.area_px] = pd.to_numeric(d[tc.area_px], errors="coerce")
    d[tc.R_px]    = pd.to_numeric(d[tc.R_px], errors="coerce")
    if tc.cx and tc.cy:
        d[tc.cx] = pd.to_numeric(d[tc.cx], errors="coerce")
        d[tc.cy] = pd.to_numeric(d[tc.cy], errors="coerce")
    if tc.det_id:
        # det id may be numeric or string; store raw and also numeric where possible
        pass
    if tc.R_mono:
        d[tc.R_mono] = pd.to_numeric(d[tc.R_mono], errors="coerce")

    for track_id, g in d.groupby(tc.track_id):
        g = g.sort_values(tc.time_ms).reset_index(drop=True)

        # candidate indices where size gate passes
        ok = (g[tc.area_px] >= amin_px) & (g[tc.R_px] >= rmin_px) & g[tc.time_ms].notna()
        idxs = np.where(ok.to_numpy())[0]
        if len(idxs) == 0:
            rejected.append({"track_id": int(track_id), "reason": "never passes size gate"})
            continue

        # find first index i such that i..i+L-1 all ok and consecutive in time order
        nuc_i = None
        for i in idxs:
            if i + L - 1 >= len(g):
                break
            if bool(ok.iloc[i:i+L].all()):
                nuc_i = int(i)
                break

        if nuc_i is None:
            rejected.append({"track_id": int(track_id), "reason": f"no length-{L} stable run after size gate"})
            continue

        row = g.loc[nuc_i]

        # rmono gate (if available)
        rmono_at = np.nan
        if use_rmono_gate:
            if tc.R_mono is None:
                rejected.append({"track_id": int(track_id), "reason": "use_rmono_gate set but tracks lacks R_mono"})
                continue
            rmono_at = float(row[tc.R_mono])
            if not np.isfinite(rmono_at) or rmono_at < rmono_min:
                rejected.append({"track_id": int(track_id), "reason": f"R_mono_at_nuc<{rmono_min}"})
                continue

        nuc_time = float(row[tc.time_ms])
        R_nuc = float(row[tc.R_px])
        area_nuc = float(row[tc.area_px])

        if rnuc_max is not None and np.isfinite(R_nuc) and R_nuc > rnuc_max:
            rejected.append({"track_id": int(track_id), "reason": f"R_nuc>{rnuc_max}"})
            continue

        # frame identifiers
        nuc_frame_idx = int(row[tc.frame_idx]) if tc.frame_idx and pd.notna(row[tc.frame_idx]) else None
        nuc_frame_id  = str(row[tc.frame_id]) if tc.frame_id and pd.notna(row[tc.frame_id]) else None

        # det/annotation id
        det_id = None
        if tc.det_id and pd.notna(row[tc.det_id]):
            try:
                det_id = int(row[tc.det_id])
            except Exception:
                # sometimes det_id is string; keep as raw string but we can still attempt int later
                try:
                    det_id = int(float(row[tc.det_id]))
                except Exception:
                    det_id = None

        cx = float(row[tc.cx]) if tc.cx and pd.notna(row[tc.cx]) else np.nan
        cy = float(row[tc.cy]) if tc.cy and pd.notna(row[tc.cy]) else np.nan

        events.append({
            "track_id": int(track_id),
            "nuc_frame": nuc_frame_idx if nuc_frame_idx is not None else np.nan,
            "nuc_time_ms": nuc_time,
            "det_id": det_id if det_id is not None else np.nan,
            "area_nuc_px": area_nuc,
            "R_nuc_px": R_nuc,
            "R_mono_at_nuc": rmono_at,
            "frame_id": nuc_frame_id if nuc_frame_id is not None else "",
            "cx": cx,
            "cy": cy,
        })

    return pd.DataFrame(events), pd.DataFrame(rejected)


# ----------------------------
# Overlap gate computation
# ----------------------------

def overlap_fraction(mask_nuc: np.ndarray, mask_prev: np.ndarray) -> float:
    inter = np.logical_and(mask_nuc, mask_prev).sum()
    denom = mask_nuc.sum()
    if denom <= 0:
        return 0.0
    return float(inter) / float(denom)


def compute_overlap_prev_for_event(event: dict,
                                   json_dir: str,
                                   max_decode_fail_keep: bool = True) -> Tuple[float, str]:
    """
    Compute overlap_prev for one nucleation event:
      max over all masks in previous frame of intersection(mask_nuc, mask_prev)/area(mask_nuc)

    Returns: (overlap_prev, note)
      overlap_prev may be np.nan if not computable
      note explains failures
    """
    nuc_frame_idx = None
    if "nuc_frame" in event and np.isfinite(event["nuc_frame"]):
        nuc_frame_idx = int(event["nuc_frame"])

    frame_id = event.get("frame_id", "") or None
    det_id = None
    if "det_id" in event and np.isfinite(event["det_id"]):
        det_id = int(event["det_id"])

    cx = event.get("cx", np.nan)
    cy = event.get("cy", np.nan)
    cx = None if not np.isfinite(cx) else float(cx)
    cy = None if not np.isfinite(cy) else float(cy)

    nuc_json = find_json_for_frame(json_dir, frame_idx=nuc_frame_idx, frame_id=frame_id)
    if nuc_json is None or not os.path.exists(nuc_json):
        return (np.nan, "missing nuc json")

    prev_json = None
    if nuc_frame_idx is not None:
        prev_json = find_json_for_frame(json_dir, frame_idx=nuc_frame_idx - 1, frame_id=None)
    else:
        # if we only have frame_id like frame_00065_t130.00ms, infer previous by index inside it
        if frame_id:
            m = _FRAME_RE.search(frame_id)
            if m:
                idx = int(m.group(1))
                prev_json = find_json_for_frame(json_dir, frame_idx=idx - 1, frame_id=None)

    if prev_json is None or not os.path.exists(prev_json):
        return (np.nan, "missing prev json")

    anns_nuc = load_frame_json(nuc_json)
    anns_prev = load_frame_json(prev_json)

    if len(anns_nuc) == 0:
        return (np.nan, "nuc frame empty json")
    if len(anns_prev) == 0:
        # prev empty means overlap is 0 by definition (no previous masks)
        return (0.0, "prev frame empty -> overlap_prev=0")

    # choose nucleus annotation
    max_dist = 10.0
    # allow a bit more if we know nucleus radius (tracks are px units)
    if "R_nuc_px" in event and np.isfinite(event["R_nuc_px"]):
        max_dist = max(10.0, 0.5 * float(event["R_nuc_px"]))

    a_nuc = find_annotation_in_frame(
        anns_nuc, prefer_id=det_id, cx=cx, cy=cy, max_dist_px=max_dist
    )
    if a_nuc is None:
        return (np.nan, "missing nuc det_id match")

    seg_nuc = a_nuc.get("segmentation", None)
    if seg_nuc is None:
        return (np.nan, "nuc ann missing segmentation")

    try:
        m_nuc = decode_mask(seg_nuc)
    except Exception as e:
        return (np.nan, f"bad nuc rle: {type(e).__name__}: {e}")

    if m_nuc.sum() <= 0:
        return (np.nan, "nuc mask empty")

    # compute maximum overlap with any previous mask
    best = 0.0
    bad_prev = 0

    for a in anns_prev:
        seg = a.get("segmentation", None)
        if seg is None:
            continue
        try:
            m_prev = decode_mask(seg)
        except Exception:
            bad_prev += 1
            continue

        ov = overlap_fraction(m_nuc, m_prev)
        if ov > best:
            best = ov
        # small speed: if already near 1, stop
        if best >= 0.999:
            break

    note = ""
    if bad_prev > 0:
        note = f"bad_prev_masks_skipped={bad_prev}"
    return (float(best), note)


def apply_overlap_gate(events_df: pd.DataFrame,
                       json_dir: str,
                       overlap_prev_max: Optional[float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adds overlap_prev to events.
    If overlap_prev_max is not None, rejects events with overlap_prev > overlap_prev_max.
    IMPORTANT: if overlap_prev is NaN (not computable), we KEEP the event but annotate it.
    """
    kept = []
    rejected = []

    for _, ev in events_df.iterrows():
        row = ev.to_dict()
        ov, note = compute_overlap_prev_for_event(row, json_dir)

        row["overlap_prev"] = ov
        row["overlap_note"] = note

        if overlap_prev_max is None:
            kept.append(row)
            continue

        # if overlap not computable -> keep (don’t nuke dataset due to decode issues)
        if not np.isfinite(ov):
            kept.append(row)
            continue

        if ov > overlap_prev_max:
            row["reason"] = f"overlap_prev>{overlap_prev_max}"
            rejected.append(row)
        else:
            kept.append(row)

    return pd.DataFrame(kept), pd.DataFrame(rejected)


# ----------------------------
# Main
# ----------------------------

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
    ap.add_argument("--overlap_prev_max", type=float, default=None)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[OK] Reading tracks:", args.tracks_csv)
    df = pd.read_csv(args.tracks_csv)
    tc = infer_tracks_schema(df)

    # stable nucleation
    events0, rej0 = find_stable_nucleation_events(
        df, tc,
        L=args.L,
        amin_px=args.amin_px,
        rmin_px=args.rmin_px,
        use_rmono_gate=args.use_rmono_gate,
        rmono_min=args.rmono_min,
        rnuc_max=args.rnuc_max
    )

    # overlap gate from JSON
    events1, rej_overlap = apply_overlap_gate(
        events0, args.json_dir, overlap_prev_max=args.overlap_prev_max
    )

    # Write outputs
    out_filtered = os.path.join(args.out_dir, "nucleation_events_filtered.csv")
    out_rej      = os.path.join(args.out_dir, "nucleation_events_rejected.csv")
    out_rej_tg   = os.path.join(args.out_dir, "nucleation_events_rejected_trackgate.csv")

    events1.to_csv(out_filtered, index=False)
    rej0.to_csv(out_rej, index=False)
    rej_overlap.to_csv(out_rej_tg, index=False)

    # Summary
    n0 = len(events0)
    n1 = len(events1)
    print("\n[OK] Wrote:")
    print(" ", out_filtered)
    print(" ", out_rej)
    print(" ", out_rej_tg)

    print("\n=== Summary ===")
    print(f"initial stable candidates: {n0}")
    print(f"accepted after overlap gate: {n1}")
    if args.overlap_prev_max is not None:
        print(f"rejected by overlap gate: {len(rej_overlap)} (overlap_prev_max={args.overlap_prev_max})")

    if n1 > 0:
        print(f"R_nuc median: {np.nanmedian(events1['R_nuc_px']):.2f} px")
        if "overlap_prev" in events1.columns:
            print(f"overlap_prev median (finite only): {np.nanmedian(events1['overlap_prev']):.3f}")
            frac_nan = float(np.mean(~np.isfinite(events1["overlap_prev"].to_numpy())))
            print(f"overlap_prev NaN fraction kept: {frac_nan:.3f}")

    # Extra: report how many overlap computations were not possible
    if n1 > 0 and "overlap_prev" in events1.columns:
        nan_reasons = events1.loc[~np.isfinite(events1["overlap_prev"]), "overlap_note"].value_counts().head(10)
        if len(nan_reasons) > 0:
            print("\nTop overlap_note for NaNs (kept):")
            print(nan_reasons.to_string())


if __name__ == "__main__":
    main()
