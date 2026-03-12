import glob, json, statistics

fs = sorted(glob.glob(
    r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\sam_cuda_vith_clean\FAPI\json_idmapped\frame_*_idmapped.json"
))
print("files", len(fs))

total_anns = 0
ids = set()
pur = []

for f in fs:
    with open(f, "r") as fh:
        d = json.load(fh)
    total_anns += len(d)
    for a in d:
        gid = int(a.get("id", 0))
        if gid > 0:
            ids.add(gid)
        pur.append(float(a.get("purity", 0.0)))

print("total_anns", total_anns)
print("unique_ids", len(ids))

if pur:
    pur_sorted = sorted(pur)
    print("purity_median", pur_sorted[len(pur_sorted)//2])
    print("purity_min", min(pur_sorted))
    print("purity_mean", statistics.mean(pur_sorted))
else:
    print("purity_median", None)
    print("purity_min", None)
    print("purity_mean", None)
