import cv2
import numpy as np
from pathlib import Path
import csv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ==============================
# SETTINGS
# ==============================
ROOT = Path(r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics")
FRAMES_ROOT = ROOT / "frames"
MASKS_ROOT = ROOT / "sam_masks"
TABLES_ROOT = ROOT / "tables"

SAM_CHECKPOINT = r"F:\SAM\sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"

MIN_AREA_PX = 500        # remove tiny junk
MAX_AREA_FRAC = 0.3      # remove background blobs
BORDER_MARGIN = 5        # px

DATASETS = ["FAPI", "FAPI_TEMPO"]

# ==============================
# LOAD SAM
# ==============================
sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.90,
    min_mask_region_area=MIN_AREA_PX,
)

# ==============================
# PROCESSING
# ==============================
for dataset in DATASETS:
    frame_dir = FRAMES_ROOT / dataset
    out_mask_dir = MASKS_ROOT / dataset
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    TABLES_ROOT.mkdir(exist_ok=True)

    frames = sorted(frame_dir.glob("frame_*.png"))
    if len(frames) == 0:
        raise RuntimeError(f"No frames found for {dataset}")

    print(f"\nSAM zero-shot on {dataset}: {len(frames)} frames")

    csv_path = TABLES_ROOT / f"{dataset}_sam_measurements.csv"
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["frame_id", "time_ms",
                          "mask_id", "area_px", "cx", "cy"])

        for frame_idx, frame_path in enumerate(frames):
            img = cv2.imread(str(frame_path))
            h, w = img.shape[:2]

            # extract time
            time_ms = float(frame_path.stem.split("_t")[1].replace("ms", ""))

            masks = mask_generator.generate(img)

            mask_id = 0
            for m in masks:
                area = m["area"]
                if area < MIN_AREA_PX or area > MAX_AREA_FRAC * h * w:
                    continue

                seg = m["segmentation"]

                # reject border-touching masks
                ys, xs = np.where(seg)
                if (xs.min() < BORDER_MARGIN or xs.max() > w - BORDER_MARGIN or
                    ys.min() < BORDER_MARGIN or ys.max() > h - BORDER_MARGIN):
                    continue

                cy, cx = ys.mean(), xs.mean()

                # save mask
                mask_img = (seg.astype(np.uint8) * 255)
                mask_name = f"{frame_path.stem}_mask_{mask_id:03d}.png"
                cv2.imwrite(str(out_mask_dir / mask_name), mask_img)

                writer.writerow([
                    frame_idx, time_ms, mask_id,
                    area, cx, cy
                ])

                mask_id += 1

    print(f"✔ {dataset}: SAM masks extracted")
