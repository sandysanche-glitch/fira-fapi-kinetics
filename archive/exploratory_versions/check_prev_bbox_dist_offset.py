import os, json
import pandas as pd
import numpy as np

TR = 116
NUC_FRAME = 24
PREV_FRAME = NUC_FRAME - 1

tracks_csv  = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\out\FAPI_TEMPO\tracks.csv"
json_dir    = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\sam_cuda_vith_clean\FAPI_TEMPO\rebuild_filtered_win60\json_patched_v2"
offsets_csv = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\stable_v15_out_overlap03_poly_pad400\offsets_estimated.csv"

tracks = pd.read_csv(tracks_csv)
row = tracks[(tracks.track_id == TR) & (tracks.frame_idx == NUC_FRAME)].iloc[0]
cx_t, cy_t = float(row.cx), float(row.cy)

off = pd.read_csv(offsets_csv)

def get_offset(frame_idx: int):
    # pick exact frame if exists, else nearest
    if (off["frame_idx"] == frame_idx).any():
        r = off.loc[off["frame_idx"] == frame_idx].iloc[0]
    else:
        r = off.iloc[(off["frame_idx"] - frame_idx).abs().argsort()[:1]].iloc[0]
    dx = float(r["dx_smooth"]) if "dx_smooth" in r else float(r["dx"])
    dy = float(r["dy_smooth"]) if "dy_smooth" in r else float(r["dy"])
    return dx, dy, int(r["frame_idx"])

dx, dy, used_frame = get_offset(PREV_FRAME)

# load prev JSON
fn = [x for x in os.listdir(json_dir)
      if x.startswith(f"frame_{PREV_FRAME:05d}_") and x.endswith("_idmapped.json")]
assert fn, "prev JSON missing"
obj = json.load(open(os.path.join(json_dir, fn[0]), "r"))

def closest_list(cx_j, cy_j, topk=10):
    pts = []
    for a in obj:
        x, y, w, h = a["bbox"]
        bx = x + 0.5*w
        by = y + 0.5*h
        d = ((bx - cx_j)**2 + (by - cy_j)**2)**0.5
        pts.append((d, bx, by, a.get("id"), a.get("area_px")))
    pts.sort(key=lambda t: t[0])
    return pts[:topk]

# Two hypotheses for mapping track -> JSON coords
# H1: json = track - offset
cx_j1, cy_j1 = cx_t - dx, cy_t - dy
# H2: json = track + offset
cx_j2, cy_j2 = cx_t + dx, cy_t + dy

print("track (cx,cy) =", cx_t, cy_t)
print(f"using offset from frame {used_frame}: dx={dx:.3f}, dy={dy:.3f}")
print()

print("H1: json = track - offset  ->", (cx_j1, cy_j1))
for d,bx,by,i,ap in closest_list(cx_j1, cy_j1, 10):
    print(f"  d={d:8.1f}  (bx,by)=({bx:7.1f},{by:7.1f}) id={i} area={ap}")
print()

print("H2: json = track + offset  ->", (cx_j2, cy_j2))
for d,bx,by,i,ap in closest_list(cx_j2, cy_j2, 10):
    print(f"  d={d:8.1f}  (bx,by)=({bx:7.1f},{by:7.1f}) id={i} area={ap}")