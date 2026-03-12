import os
import re
import json
import time
import argparse
from functools import lru_cache
from multiprocessing import Process, Queue, set_start_method

import numpy as np
import pandas as pd
from pycocotools import mask as maskUtils


# -----------------------------
# Geometry / IoU helpers
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


def ensure_rle(rle: dict):
    """
    Normalize an RLE dict for pycocotools:
      - counts can be bytes or str; ensure bytes
      - size must be [h, w]
    """
    if not isinstance(rle, dict):
        return None
    if "counts" not in rle or "size" not in rle:
        return None
    out = dict(rle)

    c = out.get("counts", None)
    if isinstance(c, str):
        out["counts"] = c.encode("ascii")

    s = out.get("size", None)
    if isinstance(s, (tuple, list)) and len(s) == 2:
        out["size"] = [int(s[0]), int(s[1])]
    else:
        return None
    return out


def _mask_iou_direct(rle_curr, rle_prev) -> float:
    # WARNING: can hang inside pycocotools for some malformed RLEs
    return float(maskUtils.iou([rle_curr], [rle_prev], [0])[0][0])


def _mask_iou_worker(q: Queue, rle_curr: dict, rle_list: list[dict]):
    """
    Runs in a child process so we can kill it if pycocotools hangs.
    Computes max IoU over rle_list.
    """
    try:
        m = 0.0
        for r in rle_list:
            v = float(maskUtils.iou([rle_curr], [r], [0])[0][0])
            if v > m:
                m = v
        q.put(("ok", m))
    except Exception as e:
        q.put(("err", f"{type(e).__name__}:{e}"))


def mask_iou_max_with_timeout(rle_curr: dict, rle_list: list[dict], timeout_s: float):
    """
    Returns (status, value_or_msg)
      status: 'ok', 'timeout', 'err'
    """
    if not rle_list:
        return "ok", 0.0

    q = Queue(maxsize=1)
    p = Process(target=_mask_iou_worker, args=(q, rle_curr, rle_list), daemon=True)
    p.start()
    p.join(timeout_s)

    if p.is_alive():
        p.terminate()
        p.join(0.2)
        return "timeout", None

    if q.empty():
        return "err", "empty_queue"
    return q.get()


# -----------------------------
# JSON indexing / caching
# -----------------------------
def build_frame_index(json_dir: str) -> dict[int, str]:
    """
    Map frame_idx -> json filepath.
    Supports filenames like:
      frame_00030_t60.00ms_idmapped.json
      frame_00030.json
      frame_00030_anything.json
    Uses the integer immediately after 'frame_'.
    """
    idx: dict[int, str] = {}
    pat = re.compile(r"^frame_(\d+)")
    for fn in os.listdir(json_dir):
        if not fn.lower().endswith(".json"):
            continue
        m = pat.match(fn)
        if not m:
            continue
        fi = int(m.group(1))
        full = os.path.join(json_dir, fn)
        # keep "best" if duplicates exist
        if fi not in idx or len(fn) > len(os.path.basename(idx[fi])):
            idx[fi] = full
    return idx


@lru_cache(maxsize=256)
def load_json_cached(path: str):
    with open(path, "r") as f:
        return json.load(f)


def ensure_parent_dir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# -----------------------------
# Offsets / candidate selection
# -----------------------------
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


def filter_valid_anns(objs) -> list:
    """
    Keep only dict anns with bbox=[x,y,w,h] and segmentation as RLE dict.
    """
    out = []
    if not isinstance(objs, list):
        return out
    for a in objs:
        if not isinstance(a, dict):
            continue
        b = a.get("bbox", None)
        seg = a.get("segmentation", None)
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            continue
        if not isinstance(seg, dict):
            continue
        seg2 = ensure_rle(seg)
        if seg2 is None:
            continue
        aa = dict(a)
        aa["segmentation"] = seg2
        out.append(aa)
    return out


def select_prev_candidates(prev_anns: list, cxj: float, cyj: float, pad: float, global_k: int):
    """
    Prefer ROI (pad box around cxj/cyj); fallback to global KNN.
    Returns (candidates, note).
    """
    if not prev_anns:
        return [], "prev_empty"

    x0, x1 = cxj - pad, cxj + pad
    y0, y1 = cyj - pad, cyj + pad

    roi = []
    for a in prev_anns:
        bx, by = bbox_center_xy(a["bbox"])
        if x0 <= bx <= x1 and y0 <= by <= y1:
            roi.append(a)

    def dist2(a):
        bx, by = bbox_center_xy(a["bbox"])
        return (bx - cxj) ** 2 + (by - cyj) ** 2

    if roi:
        roi.sort(key=dist2)
        return roi[: max(1, global_k)], f"roi(n={len(roi)})"

    prev_anns.sort(key=dist2)
    cand = prev_anns[: max(1, global_k)]
    return cand, f"global_knn(n={len(cand)})"


# -----------------------------
# Core per-event evaluation
# -----------------------------
def find_first_available_frame(frame_index: dict[int, str], start_frame: int, max_shift: int):
    for s in range(max_shift + 1):
        fi = start_frame + s
        p = frame_index.get(fi, None)
        if p is not None:
            return fi, p
    return None, None


def evaluate_event(
    track_id: int,
    nuc_frame_i: int,
    tracks_key,
    offsets: pd.DataFrame,
    frame_index: dict[int, str],
    pad: float,
    global_k: int,
    use_prev_offset: bool,
    max_shift: int,
    event_timeout_s: float,
    mask_iou_mode: str,
    mask_timeout_s: float,
):
    """
    Returns dict with computed fields.
    mask_iou_mode: 'off' | 'direct' | 'process'
    """
    t0 = time.time()
    out = {
        "eval_frame_i": np.nan,
        "shift_used": np.nan,
        "mask_iou_prev_max": np.nan,
        "bbox_iou_prev_max": np.nan,
        "cand_note": "",
        "n_prev_cands": 0,
        "dx_used": np.nan,
        "dy_used": np.nan,
        "status": "ok",
        "runtime_ms": np.nan,
    }

    # track row at nuc_frame_i
    try:
        trow = tracks_key.loc[(track_id, nuc_frame_i)]
    except KeyError:
        out["status"] = "missing_track_row"
        out["runtime_ms"] = int(1000 * (time.time() - t0))
        return out

    cx_t, cy_t = float(trow["cx"]), float(trow["cy"])

    # nuc frame path (with shift)
    eval_frame_i, nuc_path = find_first_available_frame(frame_index, nuc_frame_i, max_shift)
    if nuc_path is None:
        out["status"] = "missing_nuc_json"
        out["runtime_ms"] = int(1000 * (time.time() - t0))
        return out

    out["eval_frame_i"] = int(eval_frame_i)
    out["shift_used"] = int(eval_frame_i - nuc_frame_i)

    prev_frame_i = eval_frame_i - 1
    prev_path = frame_index.get(prev_frame_i, None)
    if prev_path is None:
        out["status"] = "missing_prev_json"
        out["runtime_ms"] = int(1000 * (time.time() - t0))
        return out

    off_frame = prev_frame_i if use_prev_offset else eval_frame_i
    dx, dy = get_offset(offsets, off_frame)
    out["dx_used"], out["dy_used"] = dx, dy

    cx_j, cy_j = cx_t - dx, cy_t - dy

    # load json
    try:
        nuc_objs = load_json_cached(nuc_path)
        prev_objs = load_json_cached(prev_path)
    except Exception:
        out["status"] = "error_json_load"
        out["runtime_ms"] = int(1000 * (time.time() - t0))
        return out

    nuc_anns = filter_valid_anns(nuc_objs)
    prev_anns = filter_valid_anns(prev_objs)

    if len(nuc_anns) == 0:
        out["status"] = "empty_nuc_json"
        out["runtime_ms"] = int(1000 * (time.time() - t0))
        return out
    if len(prev_anns) == 0:
        out["status"] = "empty_prev_json"
        out["runtime_ms"] = int(1000 * (time.time() - t0))
        return out

    # choose nuc ann closest to (cx_j, cy_j)
    def nuc_dist2(a):
        bx, by = bbox_center_xy(a["bbox"])
        return (bx - cx_j) ** 2 + (by - cy_j) ** 2

    nuc_anns.sort(key=nuc_dist2)
    nuc_ann = nuc_anns[0]
    rle_curr = nuc_ann["segmentation"]
    b_curr = nuc_ann["bbox"]

    # prev candidates
    cands, cand_note = select_prev_candidates(prev_anns, cx_j, cy_j, pad, global_k)
    out["cand_note"] = cand_note
    out["n_prev_cands"] = int(len(cands))
    if len(cands) == 0:
        out["status"] = "no_prev_candidates"
        out["runtime_ms"] = int(1000 * (time.time() - t0))
        return out

    # bbox IoU is always safe
    bbox_max = 0.0
    for a in cands:
        bbox_max = max(bbox_max, bbox_iou(b_curr, a["bbox"]))
    out["bbox_iou_prev_max"] = float(bbox_max)

    # optional mask IoU
    if mask_iou_mode == "off":
        out["mask_iou_prev_max"] = np.nan
        out["runtime_ms"] = int(1000 * (time.time() - t0))
        return out

    # Build list of prev RLEs
    prev_rles = [a["segmentation"] for a in cands]

    if mask_iou_mode == "direct":
        # WARNING: may hang for some RLEs
        mask_max = 0.0
        for rle_prev in prev_rles:
            if event_timeout_s > 0 and (time.time() - t0) > event_timeout_s:
                out["status"] = "timeout"
                out["mask_iou_prev_max"] = float(mask_max)
                out["runtime_ms"] = int(1000 * (time.time() - t0))
                return out
            try:
                mask_max = max(mask_max, _mask_iou_direct(rle_curr, rle_prev))
            except Exception:
                pass
        out["mask_iou_prev_max"] = float(mask_max)
        out["runtime_ms"] = int(1000 * (time.time() - t0))
        return out

    if mask_iou_mode == "process":
        # bounded: cannot hang indefinitely
        status, val = mask_iou_max_with_timeout(
            rle_curr=rle_curr,
            rle_list=prev_rles,
            timeout_s=max(0.1, float(mask_timeout_s)),
        )
        if status == "ok":
            out["mask_iou_prev_max"] = float(val)
        elif status == "timeout":
            out["status"] = "mask_iou_timeout"
            out["mask_iou_prev_max"] = np.nan
        else:
            out["status"] = f"mask_iou_error:{val}"
            out["mask_iou_prev_max"] = np.nan

        out["runtime_ms"] = int(1000 * (time.time() - t0))
        return out

    out["status"] = f"bad_mask_iou_mode:{mask_iou_mode}"
    out["runtime_ms"] = int(1000 * (time.time() - t0))
    return out


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

    ap.add_argument("--max_shift", type=int, default=0)
    ap.add_argument("--event_timeout_s", type=float, default=0.0)

    ap.add_argument("--mask_iou", choices=["off", "direct", "process"], default="off",
                    help="off=bbox-only (safe). direct=pycocotools inline (may hang). process=pycocotools in subprocess with timeout.")
    ap.add_argument("--mask_timeout_s", type=float, default=1.5,
                    help="Only used when --mask_iou process. Kills stuck mask IoU per-event.")

    ap.add_argument("--progress_every", type=int, default=10)
    ap.add_argument("--checkpoint_every", type=int, default=10)

    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--use_prev_offset", action="store_true")
    ap.add_argument("--max_events", type=int, default=0)

    args = ap.parse_args()

    ensure_parent_dir(args.out_csv)

    print("[OK] Stage2 v5.2 starting...", flush=True)
    print("[OK] Inputs:", flush=True)
    print("  v15_csv    =", args.v15_csv, flush=True)
    print("  tracks_csv =", args.tracks_csv, flush=True)
    print("  offsets_csv=", args.offsets_csv, flush=True)
    print("  json_dir   =", args.json_dir, flush=True)
    print("  out_csv    =", args.out_csv, flush=True)
    print("[OK] Params:", flush=True)
    print("  pad            =", args.pad, flush=True)
    print("  global_k       =", args.global_k, flush=True)
    print("  max_shift      =", args.max_shift, flush=True)
    print("  event_timeout_s=", args.event_timeout_s, flush=True)
    print("  mask_iou       =", args.mask_iou, flush=True)
    print("  mask_timeout_s =", args.mask_timeout_s, flush=True)
    print("  resume         =", args.resume, flush=True)

    ev = pd.read_csv(args.v15_csv)
    tracks = pd.read_csv(args.tracks_csv)
    offsets = pd.read_csv(args.offsets_csv)

    for col in ["track_id", "nuc_frame_i"]:
        if col not in ev.columns:
            raise ValueError(f"v15_csv missing column: {col}")

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

    tracks_key = tracks.set_index(["track_id", "frame_idx"])

    done = set()
    out_rows = []
    if args.resume and os.path.exists(args.out_csv):
        try:
            prev = pd.read_csv(args.out_csv)
            if "track_id" in prev.columns and "nuc_frame_i" in prev.columns:
                done = set(zip(prev["track_id"].astype(int), prev["nuc_frame_i"].astype(int)))
                out_rows = prev.to_dict("records")
                print(f"[OK] Resume enabled: loaded {len(out_rows)} rows; skipping {len(done)} events.", flush=True)
        except Exception as e:
            print(f"[WARN] Could not resume: {e}", flush=True)

    extra_cols = [
        "eval_frame_i", "shift_used",
        "mask_iou_prev_max", "bbox_iou_prev_max",
        "cand_note", "n_prev_cands",
        "dx_used", "dy_used",
        "status", "runtime_ms"
    ]
    out_cols = list(ev.columns) + extra_cols

    if not os.path.exists(args.out_csv):
        pd.DataFrame(columns=out_cols).to_csv(args.out_csv, index=False)

    total = len(ev)
    if args.max_events and args.max_events > 0:
        total = min(total, args.max_events)

    processed_now = 0
    t_global0 = time.time()

    for i in range(total):
        r = ev.iloc[i]
        track_id = int(r["track_id"])
        nuc_frame = int(r["nuc_frame_i"])
        key = (track_id, nuc_frame)
        if key in done:
            continue

        base = {c: r[c] for c in ev.columns}

        computed = evaluate_event(
            track_id=track_id,
            nuc_frame_i=nuc_frame,
            tracks_key=tracks_key,
            offsets=offsets,
            frame_index=frame_index,
            pad=float(args.pad),
            global_k=int(args.global_k),
            use_prev_offset=bool(args.use_prev_offset),
            max_shift=int(args.max_shift),
            event_timeout_s=float(args.event_timeout_s),
            mask_iou_mode=str(args.mask_iou),
            mask_timeout_s=float(args.mask_timeout_s),
        )

        row = dict(base)
        row.update(computed)

        out_rows.append(row)
        done.add(key)
        processed_now += 1

        if args.progress_every and (processed_now % args.progress_every == 0):
            elapsed = time.time() - t_global0
            print(
                f"[PROGRESS] processed_now={processed_now} / target={total} "
                f"last=({track_id},{nuc_frame}) status={row['status']} elapsed_s={elapsed:.1f}",
                flush=True
            )

        if args.checkpoint_every and (processed_now % args.checkpoint_every == 0):
            pd.DataFrame(out_rows, columns=out_cols).to_csv(args.out_csv, index=False)

    pd.DataFrame(out_rows, columns=out_cols).to_csv(args.out_csv, index=False)
    print(f"[OK] Done. Wrote: {args.out_csv}", flush=True)
    print(f"[OK] rows_total_written={len(out_rows)} processed_now={processed_now}", flush=True)


if __name__ == "__main__":
    # Needed on Windows for subprocess mask-iou mode
    try:
        set_start_method("spawn", force=True)
    except Exception:
        pass
    main()