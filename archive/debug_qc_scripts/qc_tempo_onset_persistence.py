import glob, json, os, re

W, H = 4704, 3424
GIANT = 0.5

# tweak these:
Nmin = 5   # need at least Nmin masks
M = 5      # for at least M consecutive frames

files = sorted(glob.glob(r"sam\segfreeze_v1_fapi_vs_tempo\FAPI_TEMPO\coco_rle\frame_*_t*ms.json"))
ts = []
ns = []

for f in files:
    m = re.search(r"frame_(\d+)_t([0-9.]+)ms", os.path.basename(f))
    fr = int(m.group(1))
    t = float(m.group(2))

    d = json.load(open(f, "r"))

    n = 0
    for a in d:
        x, y, w, h = a["bbox"]
        if (w * h) / (W * H) <= GIANT:
            n += 1

    ts.append((fr, t))
    ns.append(n)

onset = None
for i in range(0, len(ns) - M + 1):
    if all(ns[j] >= Nmin for j in range(i, i + M)):
        onset = (ts[i][0], ts[i][1], ns[i])
        break

print(f"Nmin={Nmin}  M={M}  onset={onset}")