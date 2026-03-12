import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from pycocotools import mask as mask_utils


def parse_frame_index(p: Path) -> int:
    # last integer in filename, e.g. FAPI_996.json -> 996
    nums = re.findall(r"(\d+)", p.stem)
    if not nums:
        raise ValueError(f"Cannot infer frame index from filename: {p.name}")
    return int(nums[-1])


def load_annotations_any(json_path: Path) -> List[dict]:
    with json_path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    if isinstance(d, dict):
        anns = d.get("annotations", [])
        return anns if isinstance(anns, list) else []
    if isinstance(d, list):
        return [x for x in d if isinstance(x, dict)]
    return []


def normalize_rle(rle: dict) -> dict:
    # Ensure counts are bytes for pycocotools
    r = dict(rle)
    c = r.get("counts", None)
    if isinstance(c, str):
        r["counts"] = c.encode("utf-8")
    return r


def ann_to_rle(ann: dict) -> dict | None:
    seg = ann.get("segmentation", None)
    if isinstance(seg, dict) and "size" in seg and "counts" in seg:
        return normalize_rle(seg)
    return None


def rle_area(rle: dict) -> float:
    return float(mask_utils.area(rle))


def rle_bbox_center(rle: dict) -> Tuple[float, float]:
    x, y, w, h = mask_utils.toBbox(rle)
    return float(x + w / 2.0), float(y + h / 2.0)


def iou_matrix(prev_rles: List[dict], curr_rles: List[dict]) -> np.ndarray:
    if len(prev_rles) == 0 or len(curr_rles) == 0:
        return np.zeros((len(prev_rles), len(curr_rles)), dtype=float)
    iscrowd = np.zeros((len(curr_rles),), dtype=np.uint8)
    mat = mask_utils.iou(prev_rles, curr_rles, iscrowd)  # (prev, curr)
    return np.asarray(mat, dtype=float)


def greedy_match_iou(iou: np.ndarray, thr: float) -> List[Tuple[int, int, float]]:
    matches = []
    if iou.size == 0:
        return matches
    work = iou.copy()
    while True:
        idx = np.unravel_index(np.nanargmax(work), work.shape)
        m = work[idx]
        if not np.isfinite(m) or m < thr:
            break
        ip, ic = int(idx[0]), int(idx[1])
        matches.append((ip, ic, float(m)))
        work[ip, :] = np.nan
        work[:, ic] = np.nan
    return matches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", required=True, help="Folder containing per-frame JSONs")
    ap.add_argument("--pattern", default="*.json", help="Which JSONs to use (glob), e.g. FAPI_*.json")
    ap.add_argument("--out_dir", default="sequence_tracks_out", help="Output folder")

    ap.add_argument("--start", type=int, default=None, help="First frame index to include (from filename)")
    ap.add_argument("--end", type=int, default=None, help="Last frame index to include (from filename)")
    ap.add_argument("--stride", type=int, default=1, help="Use every Nth frame")

    ap.add_argument("--iou_thr", type=float, default=0.20, help="IoU threshold for linking")
    ap.add_argument("--min_area_px", type=float, default=10.0, help="Drop tiny instances below this area (px)")
    ap.add_argument("--max_area_px", type=float, default=None, help="Optional: drop huge blobs above this area (px)")

    ap.add_argument("--frame_offset", type=int, default=0,
                    help="Add this to filename frame index before writing 'frame' (useful if your video starts later)")
    args = ap.parse_args()

    seq_dir = Path(args.seq_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # collect + filter
    jsons_all = sorted(seq_dir.glob(args.pattern), key=parse_frame_index)
    if not jsons_all:
        raise SystemExit(f"No JSON files found for pattern '{args.pattern}' in: {seq_dir}")

    selected = []
    for p in jsons_all:
        fr = parse_frame_index(p)
        if args.start is not None and fr < args.start:
            continue
        if args.end is not None and fr > args.end:
            continue
        selected.append(p)

    selected = selected[:: max(int(args.stride), 1)]
    if not selected:
        raise SystemExit("No JSONs remain after filtering start/end/stride.")

    print(f"[INFO] Using {len(selected)} JSON frames from {selected[0].name} .. {selected[-1].name}")

    next_track_id = 1
    prev_track_ids: List[int] = []
    prev_rles: List[dict] = []

    per_instance_rows = []
    nucleation_new_tracks = []

    for jf in selected:
        frame_from_name = parse_frame_index(jf)
        frame = int(frame_from_name + args.frame_offset)

        anns = load_annotations_any(jf)

        curr_rles = []
        curr_areas = []
        curr_centers = []
        curr_instance_ids = []

        inst_id = 0
        for ann in anns:
            rle = ann_to_rle(ann)
            if rle is None:
                continue
            area = rle_area(rle)
            if not np.isfinite(area) or area < args.min_area_px:
                continue
            if args.max_area_px is not None and area > float(args.max_area_px):
                continue

            cx, cy = rle_bbox_center(rle)

            inst_id += 1
            curr_instance_ids.append(inst_id)
            curr_rles.append(rle)
            curr_areas.append(float(area))
            curr_centers.append((float(cx), float(cy)))

        assigned_track_ids = [-1] * len(curr_rles)
        new_tracks_this_frame = 0

        if len(prev_rles) and len(curr_rles):
            iou = iou_matrix(prev_rles, curr_rles)
            matches = greedy_match_iou(iou, thr=args.iou_thr)
            for ip, ic, _ in matches:
                assigned_track_ids[ic] = prev_track_ids[ip]

        for i in range(len(curr_rles)):
            if assigned_track_ids[i] < 0:
                assigned_track_ids[i] = next_track_id
                new_tracks_this_frame += 1
                next_track_id += 1

        nucleation_new_tracks.append((frame, new_tracks_this_frame))

        for i, tid in enumerate(assigned_track_ids):
            area = curr_areas[i]
            cx, cy = curr_centers[i]
            per_instance_rows.append(
                {"frame": frame, "track_id": int(tid), "instance_id": int(curr_instance_ids[i]),
                 "cy": cy, "cx": cx, "area": area}
            )

        prev_track_ids = assigned_track_ids
        prev_rles = curr_rles

    per_instance = pd.DataFrame(per_instance_rows)
    if per_instance.empty:
        raise SystemExit("No instances produced. Check JSON format/min_area_px/pattern.")

    per_instance_csv = out_dir / "per_instance_tracks.csv"
    per_instance.to_csv(per_instance_csv, index=False)

    summary = (
        per_instance.groupby("track_id")
        .agg(
            nuc_frame=("frame", "min"),
            final_frame=("frame", "max"),
            final_area_px=("area", "last"),
            final_cx=("cx", "last"),
            final_cy=("cy", "last"),
            n_detections=("frame", "count"),
        )
        .reset_index()
        .sort_values("track_id")
    )
    summary_csv = out_dir / "per_track_summary.csv"
    summary.to_csv(summary_csv, index=False)

    nuc_hist = pd.DataFrame(nucleation_new_tracks, columns=["frame", "new_tracks"])
    nuc_hist_csv = out_dir / "nucleation_histogram.csv"
    nuc_hist.to_csv(nuc_hist_csv, index=False)

    print(f"[OK] Wrote:\n  {per_instance_csv}\n  {summary_csv}\n  {nuc_hist_csv}")
    print(f"[INFO] Tracks: {summary.shape[0]} | Frames used: {len(selected)}")


if __name__ == "__main__":
    main()
