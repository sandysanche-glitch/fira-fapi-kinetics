import glob, json, os, re

W, H = 4704, 3424
GIANT = 0.5
N_SHOW = 25

files = sorted(glob.glob(r"sam\segfreeze_v1_fapi_vs_tempo\FAPI_TEMPO\coco_rle\frame_*_t*ms.json"))
print("n_json:", len(files))

for f in files[:N_SHOW]:
    d = json.load(open(f, "r"))
    fracs = []
    for a in d:
        x, y, w, h = a["bbox"]
        fracs.append((w * h) / (W * H))

    fr_f = [x for x in fracs if x <= GIANT]
    base = os.path.basename(f)

    if not fr_f:
        print(base, "raw", len(fracs), "after", 0, "EMPTY")
    else:
        print(base, "raw", len(fracs), "after", len(fr_f),
              "min/max_after", round(min(fr_f), 6), round(max(fr_f), 6))