import os, json, time
import pandas as pd

v15_csv = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\stable_v15_out_overlap03_poly_pad400\nucleation_events_filtered.csv"
json_dir = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\sam_cuda_vith_clean\FAPI_TEMPO\rebuild_filtered_win60\json_patched_v2"

ev = pd.read_csv(v15_csv)
track_id = int(ev.iloc[0]["track_id"])
nuc_frame = int(ev.iloc[0]["nuc_frame_i"])
prev_frame = nuc_frame - 1

def find_json(frame_idx: int):
    pref = f"frame_{frame_idx:05d}_"
    hits = [x for x in os.listdir(json_dir) if x.startswith(pref) and x.endswith("_idmapped.json")]
    return os.path.join(json_dir, hits[0]) if hits else None

print("first event: track_id", track_id, "nuc_frame", nuc_frame, "prev_frame", prev_frame)

for k in [nuc_frame, prev_frame]:
    p = find_json(k)
    print("frame", k, "json_path", p)
    if p is None:
        continue
    t0 = time.time()
    obj = json.load(open(p, "r"))
    print("  loaded len", len(obj), "secs", time.time() - t0)