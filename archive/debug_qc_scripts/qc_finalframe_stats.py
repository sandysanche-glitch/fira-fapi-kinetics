import glob, json, os, re

W, H = 4704, 3424
GIANT = 0.5
MINF = 0.004   # <-- set from sweep

RUN_ROOT = r"sam\segfreeze_v1_fapi_vs_tempo"
DATASET = "FAPI_TEMPO"

files = sorted(glob.glob(rf"{RUN_ROOT}\{DATASET}\coco_rle\frame_*_t*ms.json"))
last = files[-1]

d = json.load(open(last, "r"))
fracs = []
for a in d:
    x,y,w,h = a["bbox"]
    frac = (w*h)/(W*H)
    if frac <= GIANT and frac >= MINF:
        fracs.append(frac)

print("LAST:", os.path.basename(last))
print("kept:", len(fracs))
if fracs:
    print("min/max kept bbox_frac:", min(fracs), max(fracs))