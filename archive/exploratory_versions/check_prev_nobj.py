import pandas as pd, json, os

df = pd.read_csv(r"stable_v12_pad400\nucleation_events_filtered.csv")
df = df[df["overlap_note"]=="prev_nocand_assumed_0"].head(10)

jdir = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\sam_cuda_vith_clean\FAPI_TEMPO\rebuild_filtered_win60\json_patched_v2"

for _, r in df.iterrows():
    f = int(r["nuc_frame_i"])
    prev = f - 1
    prev_fn = [x for x in os.listdir(jdir)
               if x.startswith(f"frame_{prev:05d}_") and x.endswith("_idmapped.json")]
    if not prev_fn:
        print("track", r["track_id"], "prev frame", prev, "MISSING JSON")
        continue
    obj = json.load(open(os.path.join(jdir, prev_fn[0]), "r"))
    print("track", r["track_id"], "nuc", f, "prev", prev, "prev_nobj", len(obj))
