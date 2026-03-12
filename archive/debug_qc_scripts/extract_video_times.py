import cv2
import numpy as np
import pandas as pd

def extract_times(video_path, out_csv):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    times_ms = []
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        times_ms.append(cap.get(cv2.CAP_PROP_POS_MSEC))

    cap.release()

    t = np.array(times_ms, dtype=float) / 1000.0
    dt = np.diff(t)

    print("Video:", video_path)
    print("Frames:", len(t), "FPS(meta):", fps)
    if len(dt):
        print("dt median:", float(np.median(dt)), "min/max:", float(dt.min()), float(dt.max()), "std:", float(dt.std()))
        if (dt.std() > 1e-4) or np.any(dt <= 0):
            print("[WARN] Non-uniform timestamps detected. Prefer ffprobe if possible.")

    pd.DataFrame({"frame": np.arange(len(t)), "t_s": t}).to_csv(out_csv, index=False)
    print("[OK] wrote:", out_csv)

if __name__ == "__main__":
    extract_times(r"F:\Sandy_data\Sandy\12.11.2025\sequences\v4\FAPI-TEMPO.avi", "FAPI-TEMPO_frame_times.csv")
    extract_times(r"F:\Sandy_data\Sandy\12.11.2025\sequences\v5\FAPI.avi", "FAPI_frame_times.csv")
