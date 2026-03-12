# build_tracks_from_idmap_jsons_v2.py
import argparse
import csv
import glob
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple


def parse_frame_and_time(fname: str) -> Tuple[int, float]:
    """
    Expected filename like:
      frame_00083_t166.00ms_idmapped.json
      frame_00083_t166.00ms_idmap.json
    Returns (frame_idx, time_ms)
    """
    base = os.path.basename(fname)
    m = re.search(r"frame_(\d+)_t([0-9.]+)ms", base)
    if not m:
        raise ValueError(f"Cannot parse frame/time from filename: {base}")
    return int(m.group(1)), float(m.group(2))


def r_px_from_area(area_px: float) -> float:
    return math.sqrt(max(area_px, 0.0) / math.pi)


def load_annotations(obj: Any) -> List[Dict[str, Any]]:
    """
    Supports:
      - list of annotations
      - dict with key 'annotations'
    """
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        anns = obj.get("annotations", [])
        return anns if isinstance(anns, list) else []
    return []


def try_infer_size_from_annotations(anns: List[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
    """
    Tries to read H,W from annotation['segmentation']['size'] = [H,W]
    """
    for a in anns:
        seg = a.get("segmentation", {})
        size = seg.get("size", None)
        if isinstance(size, (list, tuple)) and len(size) == 2:
            try:
                H, W = int(size[0]), int(size[1])
                if H > 0 and W > 0:
                    return H, W
            except Exception:
                pass
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Build tracks.csv from per-frame *_idmapped.json using stable 'id' as track_id (no linking)."
    )
    ap.add_argument("--seq_dir", required=True, help="Folder containing per-frame JSONs")
    ap.add_argument("--pattern", default="frame_*_idmapped.json", help="Glob pattern within seq_dir")
    ap.add_argument("--out_csv", required=True, help="Output tracks CSV path")
    ap.add_argument("--min_area_px", type=float, default=800.0, help="Drop tiny regions below this area")
    ap.add_argument("--max_area_frac", type=float, default=0.90,
                    help="Drop regions above this fraction of image area (use 0.30 to remove giant blob, 0.90 to keep most).")
    ap.add_argument("--frame_offset", type=int, default=0,
                    help="Add this to parsed frame index before writing (useful if video starts later).")
    ap.add_argument("--img_h_px", type=int, default=None, help="Optional: force image height (px)")
    ap.add_argument("--img_w_px", type=int, default=None, help="Optional: force image width (px)")
    args = ap.parse_args()

    seq_glob = os.path.join(args.seq_dir, args.pattern)
    files = sorted(glob.glob(seq_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {seq_glob}")

    # Determine image size
    H = args.img_h_px
    W = args.img_w_px

    if H is None or W is None:
        inferred = None
        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
            anns = load_annotations(obj)
            inferred = try_infer_size_from_annotations(anns)
            if inferred is not None:
                H, W = inferred
                break

    if H is None or W is None:
        raise RuntimeError(
            "Could not infer image size from JSONs (early frames may be empty). "
            "Re-run with --img_h_px 3424 --img_w_px 4704 (or your real size)."
        )

    img_area = float(H * W)
    max_area_px = args.max_area_frac * img_area

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fieldnames = [
        "frame_id", "frame", "time_ms",
        "track_id", "annotation_id",
        "area_px", "cx", "cy",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        "R_px"
    ]

    rows_written = 0
    frames_empty = 0

    with open(args.out_csv, "w", newline="", encoding="utf-8") as fout:
        wcsv = csv.DictWriter(fout, fieldnames=fieldnames)
        wcsv.writeheader()

        for fp in files:
            frame_idx, time_ms = parse_frame_and_time(fp)
            frame_out = frame_idx + args.frame_offset
            frame_id = f"frame_{frame_out:05d}_t{time_ms:.2f}ms"

            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
            anns = load_annotations(obj)

            if not anns:
                frames_empty += 1
                continue

            for a in anns:
                area = float(a.get("area", 0.0))
                if area < args.min_area_px:
                    continue
                if area > max_area_px:
                    continue

                bbox = a.get("bbox", [0, 0, 0, 0])
                if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                    continue
                bx, by, bw, bh = [float(x) for x in bbox]
                cx = bx + bw / 2.0
                cy = by + bh / 2.0

                tid = a.get("id", None)
                if tid is None:
                    continue
                try:
                    track_id = int(tid)
                except Exception:
                    continue
                if track_id <= 0:
                    # convention: 0 = background
                    continue

                wcsv.writerow({
                    "frame_id": frame_id,
                    "frame": frame_out,
                    "time_ms": time_ms,
                    "track_id": track_id,
                    "annotation_id": track_id,
                    "area_px": area,
                    "cx": cx,
                    "cy": cy,
                    "bbox_x": bx,
                    "bbox_y": by,
                    "bbox_w": bw,
                    "bbox_h": bh,
                    "R_px": r_px_from_area(area),
                })
                rows_written += 1

    print(f"[OK] Wrote {rows_written} rows to: {args.out_csv}")
    print(f"[INFO] Image size: {H} x {W} (area={img_area:.0f}px), max_area_px={max_area_px:.0f}px")
    print(f"[INFO] Empty frames skipped: {frames_empty} / {len(files)}")


if __name__ == "__main__":
    main()
