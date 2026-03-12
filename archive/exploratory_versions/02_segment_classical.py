import cv2
import numpy as np
from pathlib import Path
import csv

# ==============================
# SETTINGS
# ==============================
ROOT = Path(r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics")
FRAMES_ROOT = ROOT / "frames"
MASKS_ROOT = ROOT / "masks"

DATASETS = ["FAPI", "FAPI_TEMPO"]

MIN_AREA_PX = 300      # ⚠️ tune later
BLUR_KSIZE = 5         # odd number

# ==============================
# PROCESSING
# ==============================
for dataset in DATASETS:
    frame_dir = FRAMES_ROOT / dataset
    mask_dir = MASKS_ROOT / dataset / "grains"
    mask_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(frame_dir.glob("frame_*.png"))
    if len(frames) == 0:
        raise RuntimeError(f"No frames found for {dataset}")

    print(f"\nSegmenting {dataset}: {len(frames)} frames")

    csv_path = ROOT / f"{dataset}_grain_measurements.csv"
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["frame_id", "time_ms", "grain_id",
                          "area_px", "cx", "cy"])

        for frame_idx, frame_path in enumerate(frames):
            img = cv2.imread(str(frame_path))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # ---- mild blur
            gray_blur = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)

            # ---- Otsu threshold (invert if needed)
            _, bw = cv2.threshold(
                gray_blur, 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            # ---- morphology cleanup
            kernel = np.ones((3, 3), np.uint8)
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

            # ---- connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                bw, connectivity=8
            )

            # extract time from filename
            name = frame_path.stem
            time_ms = float(name.split("_t")[1].replace("ms", ""))

            grain_id = 0
            for lab in range(1, num_labels):  # skip background
                area = stats[lab, cv2.CC_STAT_AREA]
                if area < MIN_AREA_PX:
                    continue

                cx, cy = centroids[lab]

                # save mask
                mask = (labels == lab).astype(np.uint8) * 255
                mask_name = f"{frame_path.stem}_grain_{grain_id:03d}.png"
                cv2.imwrite(str(mask_dir / mask_name), mask)

                writer.writerow([
                    frame_idx, time_ms, grain_id,
                    area, cx, cy
                ])

                grain_id += 1

    print(f"✔ {dataset}: segmentation complete")
