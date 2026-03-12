import os
import json
import time
import argparse
from functools import lru_cache

import numpy as np
import pandas as pd
from pycocotools import mask as maskUtils


# -----------------------------
# Helpers
# -----------------------------
def bbox_center_xy(b):
    x, y, w, h = b
    return (x + 0.5 * w, y + 0.5 * h)


def bbox_iou(b1, b2) -> float:
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
    union = (w1 * h1) + (w2 * h2) - inter
    return float(inter / (union + 1e-9))


def mask_iou_rle(rleA, rleB) -> float:
    # pycocotools expects list-of-RLEs; returns matrix
    return float(maskUtils.iou([rleA], [rleB], [0])[0][0])


def build_frame_index(json_dir: str) -> dict[int, str]:
    """
    Map frame_idx -> json filepath once.
    Avoid repeated os.listdir inside loops.
    """
    idx = {}
    for fn in os.listdir(json_dir):
        if not (fn.startswith("frame_") and fn.endswith("_idmapped.json")):
            continue
        # frame_00050_t100.00ms_idmapped.json
        parts = fn.split("_")
        if len(parts) < 2:
            continue
        try:
            frame_i = int(parts[1])
        except Exception:
            continue
        idx.setdefault(frame_i, os.path.join(json_dir, fn))
    return idx


@lru_cache(maxsize=128)
def load_json_cached(path: str):
    with open(path, "r") as f:
        return json.load(f)


def get_offset(offsets_df: pd.DataFrame, frame_idx: int) -> tuple[float, float]:
    """
    Uses dx_smooth/dy_smooth if available else dx/dy.
    If exact frame not present, uses nearest frame in offsets_df.
    """
    m = offsets_df[offsets_df["frame_idx"] == frame_idx]
    if len(m) == 0:
        j = (offsets_df["frame_idx"] - frame_idx).abs().argsort().iloc[0]
        row = offsets_df.iloc[j]
    else:
        row = m.iloc[0]

    if "dx_smooth" in offsets_df.columns:
        dx = float(row["dx_smooth"])
        dy = float(row["dy_smooth"])
    else:
        dx = float(row["dx"])
        dy = float(row["dy"])
    return dx, dy


def select_prev_candidates(prev_objs: list, cxj: float, cyj: float, pad: float, global_k: int):
    """
    Prefer ROI (pad box around cxj/cyj); fallback to global KNN.
    Returns (candidates, note).
    """
    if not prev_objs:
        return [], "prev_empty"

    # keep only valid anns
    anns = []
    for a in prev_objs:
        if not isinstance(a, dict):
            continue
        if "bbox" not in a or "segmentation" not in a:
            continue
        if not isinstance(a["bbox"], (list, tuple)) or len(a["bbox"]) != 4:
            continue
        if not isinstance(a["segmentation"], dict):
            continue
        anns.append(a)

    if not anns:
        return [], "prev_no_valid_ann"

    x0, x1 = cxj - pad, cxj + pad
    y0, y1 = cyj - pad, cyj + pad

    roi = []
    for a in anns:
        bx, by = bbox_center_xy(a["bbox"])
        if x0 <= bx <= x1 and y0 <= by <= y1:
            roi.append(a)

    if roi:
        roi.sort(key=lambda a: (bbox_center_xy(a["bbox"])[0] - cxj) ** 2 + (bbox_center_xy(a["bbox"])[1] - cyj) ** 2)
        return roi[: max(1, global_k)], f"roi(n={len(roi)})"

    anns.sort(key=lambda a: (bbox_center_xy(a["bbox"])[0] - cxj) ** 2 + (bbox_center_xy(a["bbox"])[1] - cyj) ** 2)
    cand = anns[: max(1, global_k)]
    return cand, f"global_knn(n={len(cand)})"


def ensure_parent_dir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v15_csv", required=True)
    ap.add_argument("--tracks_csv", required=True)
    ap.add_argument("--offsets_csv", required=True)
    ap.add_argument("--json_dir", required=True)
    ap.add_argument("--out_csv", required=True)

    ap.add_argument("--pad", type=float, default=400.0)
    ap.add_argument("--global_k", type=int, default=15)

    ap.add_argument("--progress_every", type=int, default=10)
    ap.add_argument("--checkpoint_every", type=int, default=10)

    ap.add_argument("--resume", action="store_true",
                    help="If out_csv exists, skip already processed (track_id,nuc_frame_i).")
    ap.add_argument("--use_prev_offset", action="store_true",
                    help="Use offset from prev_frame instead of nuc_frame (optional).")

    ap.add_argument("--max_events", type=int, default=0,
                    help="Debug: process only first N events (0 = all).")

    args = ap.parse_args()

    ensure_parent_dir(args.out_csv)

    print("[OK] Stage2 v4 starting...", flush=True)
    print("[OK] Inputs:", flush=True)
    print("  v15_csv    =", args.v15_csv, flush=True)
    print("  tracks_csv =", args.tracks_csv, flush=True)
    print("  offsets_csv=", args.offsets_csv, flush=True)
    print("  json_dir   =", args.json_dir, flush=True)
    print("  out_csv    =", args.out_csv, flush=True)

    # Load inputs
    ev = pd.read_csv(args.v15_csv)
    tracks = pd.read_csv(args.tracks_csv)
    offsets = pd.read_csv(args.offsets_csv)

    # Verify required columns
    for col in ["track_id", "nuc_frame_i"]:
        if col not in ev.columns:
            raise ValueError(f"v15_csv missing column: {col}")

    # Tracks must have track_id, frame_idx, cx, cy
    # (You previously confirmed it's frame_idx in your tracks.csv)
    for col in ["track_id", "frame_idx", "cx", "cy"]:
        if col not in tracks.columns:
            raise ValueError(f"tracks_csv missing column: {col}")

    if "frame_idx" not in offsets.columns:
        raise ValueError("offsets_csv missing column: frame_idx")

    print(f"[OK] Loaded ev={len(ev)} tracks={len(tracks)} offsets={len(offsets)}", flush=True)

    frame_index = build_frame_index(args.json_dir)
    print(f"[OK] JSON frames indexed: {len(frame_index)}", flush=True)
    if len(frame_index) == 0:
        raise ValueError("No frame_*.json files found in json_dir")

    # Fast lookup table for tracks at (track_id, frame_idx)
    tracks_key = tracks.set_index(["track_id", "frame_idx"])

    # Resume
    out_rows = []
    done = set()
    if args.resume and os.path.exists(args.out_csv):
        try:
            prev = pd.read_csv(args.out_csv)
            if "track_id" in prev.columns and "nuc_frame_i" in prev.columns:
                done = set(zip(prev["track_id"].astype(int), prev["nuc_frame_i"].astype(int)))
                out_rows = prev.to_dict("records")
                print(f"[OK] Resume enabled: loaded {len(out_rows)} rows; skipping {len(done)} events.", flush=True)
        except Exception as e:
            print(f"[WARN] Could not resume from existing out_csv: {e}", flush=True)

    # Ensure output exists immediately (so you can see it)
    out_cols = list(ev.columns) + [
        "mask_iou_prev_max", "bbox_iou_prev_max",
        "cand_note", "n_prev_cands",
        "dx_used", "dy_used",
        "status"
    ]
    if not os.path.exists(args.out_csv):
        pd.DataFrame(columns=out_cols).to_csv(args.out_csv, index=False)

    # Main loop
    total = len(ev)
    if args.max_events and args.max_events > 0:
        total = min(total, args.max_events)

    processed_now = 0
    for i in range(total):
        r = ev.iloc[i]
        track_id = int(r["track_id"])
        nuc_frame = int(r["nuc_frame_i"])
        key = (track_id, nuc_frame)
        if key in done:
            continue

        prev_frame = nuc_frame - 1
        base = {c: r[c] for c in ev.columns}
        base.update({
            "mask_iou_prev_max": np.nan,
            "bbox_iou_prev_max": np.nan,
            "cand_note": "",
            "n_prev_cands": 0,
            "dx_used": np.nan,
            "dy_used": np.nan,
            "status": "ok",
        })

        # Try compute
        try:
            # Track centroid at nuc_frame
            trow = tracks_key.loc[(track_id, nuc_frame)]
            cx_t, cy_t = float(trow["cx"]), float(trow["cy"])

            use_off_frame = prev_frame if args.use_prev_offset else nuc_frame
            dx, dy = get_offset(offsets, use_off_frame)
            base["dx_used"], base["dy_used"] = dx, dy

            # Convert track coords to JSON coords
            cx_j, cy_j = cx_t - dx, cy_t - dy

            nuc_path = frame_index.get(nuc_frame, None)
            prev_path = frame_index.get(prev_frame, None)

            if nuc_path is None:
                base["status"] = "missing_nuc_json"
                out_rows.append(base); done.add(key); continue
            if prev_path is None:
                # this is acceptable (e.g., nuc_frame=0), mark and keep going
                base["status"] = "missing_prev_json"
                out_rows.append(base); done.add(key); continue

            nuc_objs = load_json_cached(nuc_path)
            prev_objs = load_json_cached(prev_path)

            # If nuc frame has no masks, we cannot define a current mask => overlap undefined
            if not isinstance(nuc_objs, list) or len(nuc_objs) == 0:
                base["status"] = "empty_nuc_json"
                out_rows.append(base); done.add(key); continue

            # Filter valid nuc anns
            nuc_anns = []
            for a in nuc_objs:
                if not isinstance(a, dict):
                    continue
                if "bbox" not in a or "segmentation" not in a:
                    continue
                if not isinstance(a["bbox"], (list, tuple)) or len(a["bbox"]) != 4:
                    continue
                if not isinstance(a["segmentation"], dict):
                    continue
                nuc_anns.append(a)

            if not nuc_anns:
                base["status"] = "no_valid_nuc_ann"
                out_rows.append(base); done.add(key); continue

            # Choose nuc ann closest to (cx_j, cy_j)
            nuc_anns.sort(key=lambda a: (bbox_center_xy(a["bbox"])[0] - cx_j) ** 2 + (bbox_center_xy(a["bbox"])[1] - cy_j) ** 2)
            nuc_ann = nuc_anns[0]
            rle_curr = nuc_ann["segmentation"]
            b_curr = nuc_ann["bbox"]

            # Candidate prev anns (ROI then KNN)
            cands, cand_note = select_prev_candidates(prev_objs, cx_j, cy_j, args.pad, args.global_k)
            base["cand_note"] = cand_note
            base["n_prev_cands"] = int(len(cands))

            if len(cands) == 0:
                # This is also acceptable: means prev is empty or no valid anns
                base["status"] = "no_prev_candidates"
                out_rows.append(base); done.add(key); continue

            bbox_max = 0.0
            mask_max = 0.0
            for a in cands:
                bbox_max = max(bbox_max, bbox_iou(b_curr, a["bbox"]))
                try:
                    mask_max = max(mask_max, mask_iou_rle(rle_curr, a["segmentation"]))
                except Exception:
                    # If one ann has a bad RLE, skip it
                    pass

            base["bbox_iou_prev_max"] = float(bbox_max)
            base["mask_iou_prev_max"] = float(mask_max)

        except KeyError:
            base["status"] = "missing_track_row"
        except Exception as e:
            base["status"] = f"error:{type(e).__name__}"
        finally:
            out_rows.append(base)
            done.add(key)
            processed_now += 1

        # Progress + checkpoints
        if args.progress_every and (processed_now % args.progress_every == 0):
            print(f"[PROGRESS] processed_now={processed_now} / target={total}  last=({track_id},{nuc_frame}) status={base['status']}", flush=True)

        if args.checkpoint_every and (processed_now % args.checkpoint_every == 0):
            pd.DataFrame(out_rows, columns=out_cols).to_csv(args.out_csv, index=False)

    # Final write
    pd.DataFrame(out_rows, columns=out_cols).to_csv(args.out_csv, index=False)
    print(f"[OK] Done. Wrote: {args.out_csv}", flush=True)
    print(f"[OK] rows_total_written={len(out_rows)} processed_now={processed_now}", flush=True)


if __name__ == "__main__":
    main()