import cv2
import os
from pathlib import Path

# ==============================
# USER SETTINGS
# ==============================
FPS = 500  # ⚠️ VERIFY FROM CAMERA METADATA

DATASETS = {
    "FAPI_TEMPO": r"F:\Sandy_data\Sandy\12.11.2025\sequences\v4",
    "FAPI":       r"F:\Sandy_data\Sandy\12.11.2025\sequences\v5",
}

OUTPUT_ROOT = r"C:\Users\User\A3P_kinetics\frames"

# ==============================
# PROCESSING
# ==============================
for label, folder in DATASETS.items():
    folder = Path(folder)

    # Find video file
    videos = list(folder.glob("*.mp4")) + list(folder.glob("*.avi"))
    if len(videos) != 1:
        raise RuntimeError(
            f"{label}: expected exactly 1 video, found {len(videos)}"
        )

    video_path = videos[0]
    out_dir = Path(OUTPUT_ROOT) / label
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {label}")
    print(f"Video: {video_path}")
    print(f"Output: {out_dir}")

    # 👉 FORCE FFMPEG BACKEND
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_ms = frame_id * 1000 / FPS
        fname = f"frame_{frame_id:05d}_t{t_ms:.2f}ms.png"
        cv2.imwrite(str(out_dir / fname), frame)

        frame_id += 1

    cap.release()
    print(f"✔ {frame_id} frames extracted for {label}")

print("\nAll datasets processed successfully.")
