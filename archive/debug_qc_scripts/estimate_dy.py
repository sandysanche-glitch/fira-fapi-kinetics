import pandas as pd, json, os
import numpy as np

TRACKS_CSV = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\out\FAPI_TEMPO\tracks.csv"
JSON_DIR   = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\sam_cuda_vith_clean\FAPI_TEMPO\rebuild_filtered_win60\json_patched_v2"

tracks = pd.read_csv(TRACKS_CSV)

d = []
used_frames = []

for f in range(20, 61):
    sub = tracks[tracks.frame_idx == f]
    if len(sub) < 5:
        continue

    prev = f - 1
    fn = [x for x in os.listdir(JSON_DIR) if x.startswith(f"frame_{prev:05d}_") and x.endswith("_idmapped.json")]
    if not fn:
        continue

    obj = json.load(open(os.path.join(JSON_DIR, fn[0]), "r"))
    if len(obj) == 0:
        continue

    cy = float(sub.cy.median())
    by = float(np.median([a["bbox"][1] + 0.5 * a["bbox"][3] for a in obj]))
    d.append(cy - by)
    used_frames.append(f)

d = np.array(d, dtype=float)

print("Nframes", len(d))
if len(d):
    print("dy_median", float(np.median(d)))
    print("dy_min   ", float(np.min(d)))
    print("dy_max   ", float(np.max(d)))
    print("frames_used:", used_frames[:10], "...", used_frames[-10:])
else:
    print("No usable frames found (empty JSONs or missing frames).")
