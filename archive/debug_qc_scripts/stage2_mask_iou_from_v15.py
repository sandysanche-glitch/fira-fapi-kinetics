import os, json, math, argparse
import pandas as pd
import numpy as np

from pycocotools import mask as maskUtils


def find_json_for_frame(json_dir: str, frame_idx: int) -> str | None:
    pref = f"frame_{frame_idx:05d}_"
    hits = [f for f in os.listdir(json_dir) if f.startswith(pref) and f.endswith("_idmapped.json")]
    if not hits:
        return None
    hits.sort()
    return os.path.join(json_dir, hits[0])


def bbox_center(b):
    x, y, w, h = b
    return (x + 0.5 * w, y + 0.5 * h)


def bbox_iou(b1, b2):
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
    a = w1 * h1
    b = w2 * h2
    return float(inter / (a + b - inter + 1e-9))


def get_offset_row(offsets: pd.DataFrame, frame_idx: int):
    # exact frame if possible, else nearest
    m = offsets[offsets["frame_idx"] == frame_idx]
    if len(m):
        return m.iloc[0]
    j = (offsets["frame_idx"] - frame_idx).abs().argsort().iloc[0]
    return offsets.iloc[j]


def select_candidates(prev_objs, cxj, cyj, pad, global_k):
    if not prev_objs:
        return [], "prev_empty"

    x0, x1 = cxj - pad, cxj + pad
    y0, y1 = cyj - pad, cyj + pad

    roi = []
    for a in prev_objs:
        if "bbox" not in a:
            continue
        bx, by = bbox_center(a["bbox"])
        if (x0 <= bx <= x1) and (y0 <= by <= y1):
            roi.append(a)

    if roi:
        roi.sort(key=lambda a: (bbox_center(a["bbox"])[0] - cxj) ** 2 + (bbox_center(a["bbox"])[1] - cyj) ** 2)
        return roi, f"roi(n={len(roi)})"

    # fallback global KNN
    all_list = [a for a in prev_objs if "bbox" in a]
    all_list.sort(key=lambda a: (bbox_center(a["bbox"])[0] - cxj) ** 2 + (bbox_center(a["bbox"])[1] - cyj) ** 2)
    cand = all_list[: max(1, global_k)]
    return cand, f"global_knn(n={len(cand)})"


def mask_iou(rleA: dict, rleB: dict) -> float:
    # maskUtils.iou returns array shape (len(A), len(B))
    i = maskUtils.iou([rleA], [rleB], [0])
    return float(i[0][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v15_csv", required=True, help="v15 nucleation_events_filtered.csv")
    ap.add_argument("--tracks_csv", required=True, help="tracks.csv (must include track_id, frame_idx, cx, cy)")
    ap.add_argument("--offsets_csv", required=True, help="offsets_estimated.csv from v15 run (dx_smooth/dy_smooth)")
    ap.add_argument("--json_dir", required=True, help="json_patched_v2 folder")
    ap.add_argument("--out_csv", required=True)

    ap.add_argument("--pad", type=float, default=400.0)
    ap.add_argument("--global_k", type=int, default=15)
    ap.add_argument("--use_prev_offset", action="store_true",
                    help="Use offset from prev frame instead of nuc frame for coordinate mapping.")

    args = ap.parse_args()

    ev = pd.read_csv(args.v15_csv)
    tracks = pd.read_csv(args.tracks_csv)
    offsets = pd.read_csv(args.offsets_csv)

    # basic checks
    for c in ["track_id", "nuc_frame_i"]:
        if c not in ev.columns:
            raise SystemExit(f"[ERR] v15 csv missing column {c}")

    for c in ["track_id", "frame_idx", "cx", "cy"]:
        if c not in tracks.columns:
            raise SystemExit(f"[ERR] tracks.csv missing column {c}. Found: {tracks.columns.tolist()}")

    out_rows = []
    missing_tracks = 0
    missing_json = 0

    for r in ev.itertuples(index=False):
        track_id = int(getattr(r, "track_id"))
        nuc_frame = int(getattr(r, "nuc_frame_i"))
        prev_frame = nuc_frame - 1

        # lookup track centroid at nuc frame
        trow = tracks[(tracks["track_id"] == track_id) & (tracks["frame_idx"] == nuc_frame)]
        if len(trow) == 0:
            missing_tracks += 1
            out_rows.append({
                **{k: getattr(r, k) for k in ev.columns},
                "mask_iou_prev_max": np.nan,
                "bbox_iou_prev_max": np.nan,
                "cand_note": "missing_track_row",
                "n_prev_cands": 0,
            })
            continue
        trow = trow.iloc[0]
        cx_t, cy_t = float(trow["cx"]), float(trow["cy"])

        # offset (track ≈ json + offset  => json = track - offset)
        use_frame = prev_frame if args.use_prev_offset else nuc_frame
        orow = get_offset_row(offsets, use_frame)
        dx = float(orow["dx_smooth"]) if "dx_smooth" in orow else float(orow["dx"])
        dy = float(orow["dy_smooth"]) if "dy_smooth" in orow else float(orow["dy"])

        cx_j = cx_t - dx
        cy_j = cy_t - dy

        # load nuc json + pick nuc ann
        nuc_path = find_json_for_frame(args.json_dir, nuc_frame)
        prev_path = find_json_for_frame(args.json_dir, prev_frame)
        if nuc_path is None or prev_path is None:
            missing_json += 1
            out_rows.append({
                **{k: getattr(r, k) for k in ev.columns},
                "mask_iou_prev_max": np.nan,
                "bbox_iou_prev_max": np.nan,
                "cand_note": "missing_nuc_or_prev_json",
                "n_prev_cands": 0,
                "dx_used": dx,
                "dy_used": dy,
                "cx_track": cx_t,
                "cy_track": cy_t,
                "cx_json": cx_j,
                "cy_json": cy_j,
            })
            continue

        nuc_objs = json.load(open(nuc_path, "r"))
        prev_objs = json.load(open(prev_path, "r"))

        if not isinstance(nuc_objs, list) or len(nuc_objs) == 0:
            out_rows.append({
                **{k: getattr(r, k) for k in ev.columns},
                "mask_iou_prev_max": np.nan,
                "bbox_iou_prev_max": np.nan,
                "cand_note": "empty_nuc_json",
                "n_prev_cands": 0,
                "dx_used": dx,
                "dy_used": dy,
            })
            continue

        nuc_objs = [a for a in nuc_objs if "bbox" in a and "segmentation" in a and isinstance(a["segmentation"], dict)]
        if len(nuc_objs) == 0:
            out_rows.append({
                **{k: getattr(r, k) for k in ev.columns},
                "mask_iou_prev_max": np.nan,
                "bbox_iou_prev_max": np.nan,
                "cand_note": "no_valid_nuc_ann",
                "n_prev_cands": 0,
                "dx_used": dx,
                "dy_used": dy,
            })
            continue

        nuc_objs.sort(key=lambda a: (bbox_center(a["bbox"])[0] - cx_j) ** 2 + (bbox_center(a["bbox"])[1] - cy_j) ** 2)
        nuc_ann = nuc_objs[0]
        rle_curr = nuc_ann["segmentation"]
        b_curr = nuc_ann["bbox"]

        # candidates from prev frame
        prev_objs = [a for a in prev_objs if "bbox" in a and "segmentation" in a and isinstance(a["segmentation"], dict)]
        cands, cand_note = select_candidates(prev_objs, cx_j, cy_j, args.pad, args.global_k)

        if len(cands) == 0:
            out_rows.append({
                **{k: getattr(r, k) for k in ev.columns},
                "mask_iou_prev_max": np.nan,
                "bbox_iou_prev_max": np.nan,
                "cand_note": cand_note,
                "n_prev_cands": 0,
                "dx_used": dx,
                "dy_used": dy,
                "cx_track": cx_t,
                "cy_track": cy_t,
                "cx_json": cx_j,
                "cy_json": cy_j,
            })
            continue

        # compute max bbox IoU and max mask IoU
        bbox_max = 0.0
        mask_max = 0.0
        mask_max_id = None

        for a in cands:
            bbox_max = max(bbox_max, bbox_iou(b_curr, a["bbox"]))
            try:
                v = mask_iou(rle_curr, a["segmentation"])
                if v > mask_max:
                    mask_max = v
                    mask_max_id = a.get("id", None)
            except Exception:
                # if one fails, skip it
                continue

        out_rows.append({
            **{k: getattr(r, k) for k in ev.columns},
            "mask_iou_prev_max": float(mask_max),
            "mask_iou_prev_argmax_id": mask_max_id,
            "bbox_iou_prev_max": float(bbox_max),
            "cand_note": cand_note,
            "n_prev_cands": int(len(cands)),
            "dx_used": dx,
            "dy_used": dy,
            "cx_track": cx_t,
            "cy_track": cy_t,
            "cx_json": cx_j,
            "cy_json": cy_j,
            "nuc_json": os.path.basename(nuc_path),
            "prev_json": os.path.basename(prev_path),
        })

    out = pd.DataFrame(out_rows)
    out.to_csv(args.out_csv, index=False)

    print(f"[OK] Wrote: {args.out_csv}")
    print(f"[OK] rows={len(out)}  missing_tracks={missing_tracks}  missing_json={missing_json}")


if __name__ == "__main__":
    main()