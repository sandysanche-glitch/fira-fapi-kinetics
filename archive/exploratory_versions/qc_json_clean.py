import glob, json

fs = sorted(glob.glob(r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\sam_cuda_vith_clean\FAPI\json_clean\frame_*.json"))
print("files", len(fs))

n0 = 0
maxa = 0.0
for f in fs:
    with open(f, "r") as fh:
        d = json.load(fh)
    if len(d) == 0:
        n0 += 1
    for a in d:
        maxa = max(maxa, float(a.get("area", 0.0)))

print("empty", n0, "max_area_px", maxa)
