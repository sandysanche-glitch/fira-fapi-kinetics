import glob, json, os, re

W, H = 4704, 3424

files = sorted(glob.glob(r"sam\segfreeze_v1_fapi_vs_tempo\FAPI\coco_rle\frame_*_t*ms.json"))
print("n_json:", len(files))

for f in files[:20]:
    with open(f, "r") as fh:
        data = json.load(fh)

    m = re.search(r"frame_(\d+)_t([0-9.]+)ms", os.path.basename(f))
    n = len(data)

    if n == 0:
        print(os.path.basename(f), "EMPTY")
        continue

    fracs = []
    for a in data:
        x, y, w, h = a["bbox"]
        fracs.append((w*h)/(W*H))

    print(os.path.basename(f),
          "n=", n,
          "bbox_frac[min,max]=",
          round(min(fracs),4),
          round(max(fracs),4))