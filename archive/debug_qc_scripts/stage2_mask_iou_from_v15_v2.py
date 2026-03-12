import os, json, argparse
import pandas as pd
import numpy as np
from functools import lru_cache
from pycocotools import mask as maskUtils

def bbox_center(b):
    x, y, w, h = b
    return (x + 0.5*w, y + 0.5*h)

def bbox_iou(b1, b2):
    x1,y1,w1,h1 = b1
    x2,y2,w2,h2 = b2
    ax1,ay1,ax2,ay2 = x1,y1,x1+w1,y1+h1
    bx1,by1,bx2,by2 = x2,y2,x2+w2,y2+h2
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw,ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw*ih
    if inter <= 0: return 0.0
    return float(inter / (w1*h1 + w2*h2 - inter + 1e-9))

def mask_iou(rleA, rleB):
    return float(maskUtils.iou([rleA], [rleB], [0])[0][0])

def build_frame_index(json_dir: str) -> dict[int, str]:
    idx = {}
    for fn in os.listdir(json_dir):
        if not fn.endswith("_idmapped.json"): 
            continue
        if not fn.startswith("frame_"):
            continue
        try:
            f = int(fn.split("_")[1])
        except Exception:
            continue
        idx.setdefault(f, os.path.join(json_dir, fn))
    return idx

@lru_cache(maxsize=64)
def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def get_offset(offsets: pd.DataFrame, frame_idx: int):
    m = offsets[offsets["frame_idx"] == frame_idx]
    if len(m):
        row = m.iloc[0]
    else:
        j = (offsets["frame_idx"] - frame_idx).abs().argsort().iloc[0]
        row = offsets.iloc[j]
    dx = float(row["dx_smooth"]) if "dx_smooth" in offsets.columns else float(row["dx"])
    dy = float(row["dy_smooth"]) if "dy_smooth" in offsets.columns else float(row["dy"])
    return dx, dy

def select_candidates(prev_objs, cxj, cyj, pad, global_k):
    if not prev_objs:
        return [], "prev_empty"
    prev_objs = [a for a in prev_objs if "bbox" in a and "segmentation" in a and isinstance(a["segmentation"], dict)]
    if not prev_objs:
        return [], "prev_no_valid_ann"

    x0,x1 = cxj-pad, cxj+pad
    y0,y1 = cyj-pad, cyj+pad
    roi = []
    for a in prev_objs:
        bx,by = bbox_center(a["bbox"])
        if x0 <= bx <= x1 and y0 <= by <= y1:
            roi.append(a)

    if roi:
        roi.sort(key=lambda a: (bbox_center(a["bbox"])[0]-cxj)**2 + (bbox_center(a["bbox"])[1]-cyj)**2)
        return roi, f"roi(n={len(roi)})"

    prev_objs.sort(key=lambda a: (bbox_center(a["bbox"])[0]-cxj)**2 + (bbox_center(a["bbox"])[1]-cyj)**2)
    cand = prev_objs[:max(1, global_k)]
    return cand, f"global_knn(n={len(cand)})"

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
    ap.add_argument("--checkpoint_every", type=int, default=25)
    ap.add_argument("--use_prev_offset", action="store_true")
    args = ap.parse_args()

    ev = pd.read_csv(args.v15_csv)
    tracks = pd.read_csv(args.tracks_csv)
    offsets = pd.read_csv(args.offsets_csv)

    frame_index = build_frame_index(args.json_dir)
    if not frame_index:
        raise SystemExit("[ERR] No JSON frames found in json_dir")

    tracks_key = tracks.set_index(["track_id", "frame_idx"])
    out_rows = []
    n = len(ev)

    for i, r in enumerate(ev.itertuples(index=False), start=1):
        track_id = int(getattr(r, "track_id"))
        nuc_frame = int(getattr(r, "nuc_frame_i"))
        prev_frame = nuc_frame - 1

        try:
            trow = tracks_key.loc[(track_id, nuc_frame)]
            cx_t, cy_t = float(trow["cx"]), float(trow["cy"])
        except Exception:
            out_rows.append({**{c: getattr(r, c) for c in ev.columns},
                             "mask_iou_prev_max": np.nan,
                             "bbox_iou_prev_max": np.nan,
                             "cand_note": "missing_track_row",
                             "n_prev_cands": 0})
            continue

        use_frame = prev_frame if args.use_prev_offset else nuc_frame
        dx, dy = get_offset(offsets, use_frame)
        cx_j, cy_j = cx_t - dx, cy_t - dy

        nuc_path = frame_index.get(nuc_frame)
        prev_path = frame_index.get(prev_frame)
        if nuc_path is None or prev_path is None:
            out_rows.append({**{c: getattr(r, c) for c in ev.columns},
                             "mask_iou_prev_max": np.nan,
                             "bbox_iou_prev_max": np.nan,
                             "cand_note": "missing_nuc_or_prev_json",
                             "n_prev_cands": 0,
                             "dx_used": dx, "dy_used": dy})
            continue

        nuc_objs = load_json(nuc_path)
        prev_objs = load_json(prev_path)

        nuc_objs = [a for a in nuc_objs if "bbox" in a and "segmentation" in a and isinstance(a["segmentation"], dict)]
        if not nuc_objs:
            out_rows.append({**{c: getattr(r, c) for c in ev.columns},
                             "mask_iou_prev_max": np.nan,
                             "bbox_iou_prev_max": np.nan,
                             "cand_note": "no_valid_nuc_ann",
                             "n_prev_cands": 0})
            continue

        nuc_objs.sort(key=lambda a: (bbox_center(a["bbox"])[0]-cx_j)**2 + (bbox_center(a["bbox"])[1]-cy_j)**2)
        nuc_ann = nuc_objs[0]
        rle_curr = nuc_ann["segmentation"]
        b_curr = nuc_ann["bbox"]

        cands, cand_note = select_candidates(prev_objs, cx_j, cy_j, args.pad, args.global_k)
        if not cands:
            out_rows.append({**{c: getattr(r, c) for c in ev.columns},
                             "mask_iou_prev_max": np.nan,
                             "bbox_iou_prev_max": np.nan,
                             "cand_note": cand_note,
                             "n_prev_cands": 0})
            continue

        bbox_max = 0.0
        mask_max = 0.0
        for a in cands:
            bbox_max = max(bbox_max, bbox_iou(b_curr, a["bbox"]))
            try:
                mask_max = max(mask_max, mask_iou(rle_curr, a["segmentation"]))
            except Exception:
                pass

        out_rows.append({**{c: getattr(r, c) for c in ev.columns},
                         "mask_iou_prev_max": float(mask_max),
                         "bbox_iou_prev_max": float(bbox_max),
                         "cand_note": cand_note,
                         "n_prev_cands": int(len(cands)),
                         "dx_used": dx, "dy_used": dy})

        if args.progress_every and (i % args.progress_every == 0):
            print(f"[PROGRESS] {i}/{n}", flush=True)

        if args.checkpoint_every and (i % args.checkpoint_every == 0):
            pd.DataFrame(out_rows).to_csv(args.out_csv, index=False)

    pd.DataFrame(out_rows).to_csv(args.out_csv, index=False)
    print(f"[OK] Wrote: {args.out_csv}")
    print(f"[OK] rows={len(out_rows)}")

if __name__ == "__main__":
    main()