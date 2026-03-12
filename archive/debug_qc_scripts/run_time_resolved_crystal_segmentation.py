#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_time_resolved_crystal_segmentation.py

Time-resolved crystal segmentation and kinetics extraction.

This script IS MEANT TO BE RUN DIRECTLY.

What’s new vs your current version
----------------------------------
- Adds CLI options to pass SAM overrides without editing any code
- Lets you version the SAM output root folder (e.g. sam_cuda_vith_16pps)
- Lets you choose which datasets to run (FAPI, FAPI_TEMPO)

Example
-------
# Run both datasets with ViT-H safe settings (already default in your wrapper)
python run_time_resolved_crystal_segmentation.py

# Override SAM settings globally from CLI
python run_time_resolved_crystal_segmentation.py ^
  --sam_overrides '{"points_per_side":24,"points_per_batch":16,"crop_n_layers":0,"downscale_factor":0.5}' ^
  --sam_root sam_cuda_vith_24pps

# Override only TEMPO (e.g. more conservative)
python run_time_resolved_crystal_segmentation.py ^
  --sam_overrides '{"points_per_side":24}' ^
  --sam_overrides_tempo '{"points_per_side":16,"points_per_batch":8}' ^
  --sam_root sam_cuda_vith_mixed
"""

# ==================================================
# Make project root importable
# ==================================================
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ==================================================
# Standard imports
# ==================================================
import argparse
import csv
import json
import cv2
import numpy as np

# ==================================================
# Your modules
# ==================================================
from sam.run_segment_anything import run_sam_on_folder

from src.crystal_segmentation import (
    filter_and_label_annotations_as_nuclei_and_cells,
    CRYSTAL_NUCLEUS_ANNOTATION,
    CRYSTAL_CELL_ANNOTATION,
)

from src import coco


# ==================================================
# Default PATHS (edit here only if you want)
# ==================================================
DEFAULT_ROOT = Path(r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics")


# ==================================================
# FILTER PARAMETERS (keep as you had)
# ==================================================
ANNOTATION_MIN_AREA = 100
ANNOTATION_MAX_AREA = None
CRYSTAL_NUCLEUS_MAX_AREA = 1500
CRYSTAL_NUCLEUS_COLOR_HSV_MAX_VALUE = 80
IOU_THRESHOLD = 0.5


# ==================================================
# SAM override validation
# ==================================================
_ALLOWED_SAM_KEYS = {
    "downscale_factor",
    "points_per_side",
    "points_per_batch",
    "crop_n_layers",
    "pred_iou_thresh",
    "stability_score_thresh",
    "stability_score_offset",
    "box_nms_thresh",
    "crop_nms_thresh",
    "crop_overlap_ratio",
    "crop_n_points_downscale_factor",
    "min_mask_region_area",
}


def _parse_json_dict(s: str | None) -> dict:
    if not s:
        return {}
    try:
        d = json.loads(s)
    except Exception as e:
        raise ValueError(f"Could not parse JSON dict: {s}") from e
    if not isinstance(d, dict):
        raise ValueError("SAM overrides must be a JSON object/dict, e.g. '{\"points_per_side\":24}'")
    # validate keys
    bad = [k for k in d.keys() if k not in _ALLOWED_SAM_KEYS]
    if bad:
        raise ValueError(
            f"Unknown SAM override keys: {bad}\n"
            f"Allowed keys: {sorted(_ALLOWED_SAM_KEYS)}"
        )
    return d


def _dataset_list(s: str) -> list[str]:
    items = [x.strip() for x in s.split(",") if x.strip()]
    ok = {"FAPI", "FAPI_TEMPO"}
    bad = [x for x in items if x not in ok]
    if bad:
        raise ValueError(f"Unknown dataset(s): {bad}. Allowed: {sorted(ok)}")
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default=str(DEFAULT_ROOT),
        help="Project root (contains frames/). Default is hard-coded Kinetics root.",
    )
    ap.add_argument(
        "--datasets",
        type=str,
        default="FAPI,FAPI_TEMPO",
        help="Comma-separated: FAPI,FAPI_TEMPO (default runs both).",
    )
    ap.add_argument(
        "--sam_root",
        type=str,
        default="sam",
        help="Output folder name under ROOT (default: sam). Use this to version outputs.",
    )
    ap.add_argument(
        "--sam_overrides",
        type=str,
        default="",
        help="JSON dict of SAM params applied to ALL datasets (e.g. '{\"points_per_side\":24}').",
    )
    ap.add_argument(
        "--sam_overrides_fapi",
        type=str,
        default="",
        help="JSON dict of SAM params applied only to FAPI (merged on top of --sam_overrides).",
    )
    ap.add_argument(
        "--sam_overrides_tempo",
        type=str,
        default="",
        help="JSON dict of SAM params applied only to FAPI_TEMPO (merged on top of --sam_overrides).",
    )

    args = ap.parse_args()

    ROOT = Path(args.root)
    FRAMES_ROOT = ROOT / "frames"
    SAM_ROOT = ROOT / args.sam_root

    datasets = _dataset_list(args.datasets)

    global_overrides = _parse_json_dict(args.sam_overrides)
    fapi_overrides = _parse_json_dict(args.sam_overrides_fapi)
    tempo_overrides = _parse_json_dict(args.sam_overrides_tempo)

    dataset_to_folder = {
        "FAPI": FRAMES_ROOT / "FAPI",
        "FAPI_TEMPO": FRAMES_ROOT / "FAPI_TEMPO",
    }

    print("=== run_time_resolved_crystal_segmentation ===")
    print("ROOT:", ROOT)
    print("FRAMES_ROOT:", FRAMES_ROOT)
    print("SAM_ROOT:", SAM_ROOT)
    print("Datasets:", datasets)
    print("SAM overrides (global):", global_overrides)
    print("SAM overrides (FAPI):", fapi_overrides)
    print("SAM overrides (FAPI_TEMPO):", tempo_overrides)

    for dataset_name in datasets:
        frame_folder = dataset_to_folder[dataset_name]
        print(f"\n=== Processing {dataset_name} ===")

        if not frame_folder.exists():
            raise FileNotFoundError(f"Frame folder not found: {frame_folder}")

        out_root = SAM_ROOT / dataset_name
        coco_rle_path = out_root / "coco_rle"
        coco_rle_path.mkdir(parents=True, exist_ok=True)

        # --------------------------------------------------
        # 1) RUN SAM (with overrides)
        # --------------------------------------------------
        overrides = dict(global_overrides)
        if dataset_name == "FAPI":
            overrides.update(fapi_overrides)
        elif dataset_name == "FAPI_TEMPO":
            overrides.update(tempo_overrides)

        if overrides:
            print(f"[{dataset_name}] SAM overrides applied:", overrides)

        run_sam_on_folder(
            image_folder_path=frame_folder,
            coco_rle_annotations_folder_path=coco_rle_path,
            **overrides,
        )

        # --------------------------------------------------
        # 2) WRITE KINETICS TABLE
        # --------------------------------------------------
        csv_path = out_root / "frame_kinetics.csv"
        with open(csv_path, "w", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([
                "frame_id",
                "time_ms",
                "annotation_id",
                "class",
                "area_px",
                "cx",
                "cy",
                "bbox_x",
                "bbox_y",
                "bbox_w",
                "bbox_h",
                "R_px",
            ])

            coco_files = sorted(coco_rle_path.glob("*.json"))
            if not coco_files:
                raise FileNotFoundError(f"No COCO RLE JSON files found in: {coco_rle_path}")

            for coco_file in coco_files:
                frame_id = coco_file.stem  # expects frame filenames like 000001.png -> 000001.json
                img_path = frame_folder / f"{frame_id}.png"
                if not img_path.exists():
                    # try common alternatives
                    for ext in (".jpg", ".jpeg", ".tif", ".tiff"):
                        alt = frame_folder / f"{frame_id}{ext}"
                        if alt.exists():
                            img_path = alt
                            break

                image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if image is None:
                    print(f"[WARN] Could not read image for {frame_id}: {img_path}")
                    continue

                # Extract time from filename if you have it; otherwise keep 0.
                # If you have a separate timestamp pipeline, you can replace this.
                time_ms = 0.0

                with open(coco_file, "r", encoding="utf-8") as f:
                    annotations = json.load(f)

                annotations = filter_and_label_annotations_as_nuclei_and_cells(
                    image=image,
                    annotations=annotations,
                    annotation_min_area=ANNOTATION_MIN_AREA,
                    annotation_max_area=ANNOTATION_MAX_AREA,
                    crystal_nucleus_max_area=CRYSTAL_NUCLEUS_MAX_AREA,
                    crystal_nucleus_color_hsv_max_value=CRYSTAL_NUCLEUS_COLOR_HSV_MAX_VALUE,
                    iou_threshold=IOU_THRESHOLD,
                    decode_coco_rle_masks=True,
                )

                # Write rows
                for ann in annotations:
                    ann_id = ann.get("id", None)
                    cls = ann.get("class", None)
                    area_px = ann.get("area", None)
                    bbox = ann.get("bbox", None)

                    # centroid from decoded mask if present; else bbox center
                    cx = ann.get("cx", None)
                    cy = ann.get("cy", None)
                    if cx is None or cy is None:
                        if bbox and len(bbox) == 4:
                            cx = float(bbox[0]) + 0.5 * float(bbox[2])
                            cy = float(bbox[1]) + 0.5 * float(bbox[3])
                        else:
                            cx, cy = (np.nan, np.nan)

                    # Equivalent radius (px)
                    if area_px is not None and float(area_px) > 0:
                        R_px = float(np.sqrt(float(area_px) / np.pi))
                    else:
                        R_px = np.nan

                    if bbox and len(bbox) == 4:
                        bx, by, bw, bh = bbox
                    else:
                        bx = by = bw = bh = np.nan

                    writer.writerow([
                        frame_id,
                        time_ms,
                        ann_id,
                        cls,
                        area_px,
                        cx,
                        cy,
                        bx,
                        by,
                        bw,
                        bh,
                        R_px,
                    ])

        print(f"[{dataset_name}] Wrote kinetics CSV: {csv_path}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
