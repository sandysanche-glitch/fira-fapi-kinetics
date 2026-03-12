import pandas as pd, json, os, math

TR = 116
NUC_FRAME = 24
JSON_DIR = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\sam_cuda_vith_clean\FAPI_TEMPO\rebuild_filtered_win60\json_patched_v2"
TRACKS_CSV = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\out\FAPI_TEMPO\tracks.csv"

tracks = pd.read_csv(TRACKS_CSV)
row = tracks[(tracks.track_id == TR) & (tracks.frame_idx == NUC_FRAME)].iloc[0]
cx, cy = float(row.cx), float(row.cy)

prev = NUC_FRAME - 1
prev_fn = [x for x in os.listdir(JSON_DIR) if x.startswith(f"frame_{prev:05d}_") and x.endswith("_idmapped.json")]
if not prev_fn:
    raise FileNotFoundError(f"No JSON for prev frame {prev}")

obj = json.load(open(os.path.join(JSON_DIR, prev_fn[0]), "r"))

pts = []
for a in obj:
    x, y, w, h = a["bbox"]
    bx = x + 0.5 * w
    by = y + 0.5 * h
    d = math.hypot(bx - cx, by - cy)
    pts.append((d, bx, by, a.get("id", None), a.get("area_px", None)))

pts = sorted(pts)[:10]

print("track", TR, "nuc_frame", NUC_FRAME, "prev_frame", prev, "track_c", cx, cy)
print("10 closest prev bbox centers:")
for d, bx, by, i, ap in pts:
    print(f"d={d:.1f}  (bx,by)=({bx:.1f},{by:.1f}) id={i} area={ap}")
