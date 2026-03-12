import json
import os
import sys

p = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\sam_cuda_vith_clean\FAPI_TEMPO\rebuild_filtered_win60\json_filtered\frame_00010_t20.00ms_idmapped.json"

d = json.load(open(p, "r"))
print("file:", p)
print("n_anns:", len(d))
if not d:
    sys.exit(0)

a = d[0]
print("ann keys:", list(a.keys()))

seg = a.get("segmentation", None)
print("seg type:", type(seg))

if isinstance(seg, dict):
    print("seg keys:", list(seg.keys()))
    counts = seg.get("counts", None)
    size = seg.get("size", None)
    print("counts type:", type(counts))
    if isinstance(counts, str):
        print("counts str len:", len(counts))
    elif isinstance(counts, (bytes, bytearray)):
        print("counts bytes len:", len(counts))
    elif isinstance(counts, list):
        print("counts list len:", len(counts), "first10:", counts[:10])
    else:
        print("counts:", counts)
    print("size:", size)

elif isinstance(seg, list):
    print("seg list len:", len(seg))
    if len(seg) and isinstance(seg[0], list):
        print("first poly len:", len(seg[0]), "first10:", seg[0][:10])
    else:
        print("seg[0] type:", type(seg[0]) if len(seg) else None)

else:
    print("seg value:", seg)

# also show any alternate mask keys that might exist
for k in ["rle", "mask", "counts", "size", "segmentation_rle"]:
    if k in a:
        print(f"top-level {k}:", type(a[k]))
