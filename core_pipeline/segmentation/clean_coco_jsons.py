#!/usr/bin/env python3
import argparse
import glob
import json
import os
from typing import Any, Dict, List, Tuple, Optional

def _require_pycocotools():
    try:
        from pycocotools import mask as mask_utils  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "pycocotools is required for intersection gating with compressed RLE.\n"
            "Install in your env:\n"
            "  pip install pycocotools\n"
            "or (conda-forge):\n"
            "  conda install -c conda-forge pycocotools\n"
        ) from e
    return mask_utils


def infer_hw(ann_list: List[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
    """Infer (H, W) from COCO RLE segmentation size=[H,W]."""
    for ann in ann_list:
        seg = ann.get("segmentation", {})
        size = seg.get("size", None)
        if isinstance(size, (list, tuple)) and len(size) == 2:
            h, w = int(size[0]), int(size[1])
            if h > 0 and w > 0:
                return h, w
    return None


def build_union_rle(mask_utils, ann_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Union of all RLEs in a frame. Returns RLE dict or None if no valid RLEs."""
    rles = []
    for ann in ann_list:
        seg = ann.get("segmentation", None)
        if not isinstance(seg, dict):
            continue
        if "counts" not in seg or "size" not in seg:
            continue
        rles.append(seg)
    if not rles:
        return None
    # union: intersect=False
    return mask_utils.merge(rles, intersect=False)


def intersection_area(mask_utils, rle_a: Dict[str, Any], rle_b: Dict[str, Any]) -> float:
    """Area of intersection between two RLEs."""
    inter = mask_utils.merge([rle_a, rle_b], intersect=True)
    return float(mask_utils.area(inter))


def clean_one(
    ann_list: List[Dict[str, Any]],
    max_area_frac: float,
    force_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[List[Dict[str, Any]], int, int, int, int, float]:
    """
    Returns: (cleaned_list, kept, dropped, H, W, max_area_px)
    """
    if force_hw is not None:
        h, w = force_hw
    else:
        hw = infer_hw(ann_list)
        if hw is None:
            # No annotations -> cannot infer size; keep as-is
            return ann_list, len(ann_list), 0, -1, -1, float("inf")
        h, w = hw

    img_area = h * w
    max_area_px = max_area_frac * img_area

    kept_list = []
    dropped = 0
    for ann in ann_list:
        area = ann.get("area", None)
        if area is None:
            kept_list.append(ann)
            continue

        try:
            area = float(area)
        except Exception:
            kept_list.append(ann)
            continue

        if area > max_area_px:
            dropped += 1
        else:
            kept_list.append(ann)

    return kept_list, len(kept_list), dropped, h, w, max_area_px


def main():
    ap = argparse.ArgumentParser(
        description="Clean per-frame COCO JSONs: drop huge blobs and optionally drop annotations not intersecting a gate-foreground."
    )
    ap.add_argument("--in_glob", required=True, help='Input glob, e.g. "X:\\...\\frame_*.json"')
    ap.add_argument("--out_dir", required=True, help="Output folder for cleaned JSONs")
    ap.add_argument("--max_area_frac", type=float, default=0.30,
                    help="Drop annotations with area > max_area_frac*(W*H)")
    ap.add_argument("--img_w", type=int, default=None, help="Optional: force image width")
    ap.add_argument("--img_h", type=int, default=None, help="Optional: force image height")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    ap.add_argument("--quiet", action="store_true", help="Less printing")

    # New: gate filtering
    ap.add_argument("--gate_glob", default=None,
                    help='Optional: glob to original frame JSONs used to build a per-frame union foreground gate. '
                         'Must match filenames (basenames) with in_glob.')
    ap.add_argument("--min_intersection_px", type=float, default=1.0,
                    help="Drop annotation if intersection area with gate-foreground is < this value (px).")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    in_files = sorted(glob.glob(args.in_glob))
    if not in_files:
        raise SystemExit(f"[ERR] No files match: {args.in_glob}")

    # Optional gating: map basename -> gate file
    gate_map = {}
    mask_utils = None
    if args.gate_glob:
        gate_files = sorted(glob.glob(args.gate_glob))
        if not gate_files:
            raise SystemExit(f"[ERR] No gate files match: {args.gate_glob}")
        gate_map = {os.path.basename(p): p for p in gate_files}
        mask_utils = _require_pycocotools()

    force_hw = None
    if (args.img_w is not None) and (args.img_h is not None):
        force_hw = (args.img_h, args.img_w)

    total_kept = total_dropped = 0
    total_dropped_no_inter = 0
    total_dropped_huge = 0

    for fp in in_files:
        base = os.path.basename(fp)

        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise SystemExit(f"[ERR] Expected a LIST of annotations in {fp}, got: {type(data)}")

        # 1) drop huge blobs
        data1, kept1, dropped_huge, h, w, max_area_px = clean_one(
            data, args.max_area_frac, force_hw=force_hw
        )

        # 2) optional: gate by original union foreground
        dropped_no_inter = 0
        if args.gate_glob:
            gate_fp = gate_map.get(base, None)
            if gate_fp is None:
                raise SystemExit(f"[ERR] No matching gate JSON for {base}. "
                                 f"Gate files must share the same basename as input files.")
            with open(gate_fp, "r", encoding="utf-8") as f:
                gate_data = json.load(f)
            if not isinstance(gate_data, list):
                raise SystemExit(f"[ERR] Gate JSON {gate_fp} not a list.")

            gate_union = build_union_rle(mask_utils, gate_data)

            if gate_union is None:
                # No gate foreground => drop everything (or keep nothing). Here: drop everything (makes sense).
                data2 = []
                dropped_no_inter = len(data1)
            else:
                data2 = []
                for ann in data1:
                    seg = ann.get("segmentation", None)
                    if not isinstance(seg, dict) or "counts" not in seg or "size" not in seg:
                        # If no RLE, keep it (rare)
                        data2.append(ann)
                        continue
                    inter_a = intersection_area(mask_utils, seg, gate_union)
                    if inter_a < args.min_intersection_px:
                        dropped_no_inter += 1
                    else:
                        data2.append(ann)
        else:
            data2 = data1

        out_fp = os.path.join(args.out_dir, base)
        if (not args.overwrite) and os.path.exists(out_fp):
            raise SystemExit(f"[ERR] Output exists (use --overwrite): {out_fp}")

        with open(out_fp, "w", encoding="utf-8") as f:
            json.dump(data2, f)

        total_kept += len(data2)
        total_dropped += (dropped_huge + dropped_no_inter)
        total_dropped_huge += dropped_huge
        total_dropped_no_inter += dropped_no_inter

        if not args.quiet:
            hw_txt = f"(H,W)=({h},{w})" if (h > 0 and w > 0) else "(H,W)=unknown"
            gate_txt = f", dropped_no_inter={dropped_no_inter}" if args.gate_glob else ""
            print(f"[OK] {base}  kept={len(data2)}  dropped_huge={dropped_huge}{gate_txt}  max_area_px={max_area_px:.0f}  {hw_txt}")

    if not args.quiet:
        print(f"[DONE] Total kept={total_kept}, dropped_huge={total_dropped_huge}, dropped_no_inter={total_dropped_no_inter}")
        print(f"       out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
