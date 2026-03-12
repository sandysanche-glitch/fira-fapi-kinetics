import glob, json, os, re
import matplotlib.pyplot as plt

W, H = 4704, 3424
GIANT = 0.5
MINF = 0.006
RUN_ROOT = r"sam\segfreeze_v1_fapi_vs_tempo"
DATASET = "FAPI_TEMPO"

files = sorted(glob.glob(rf"{RUN_ROOT}\{DATASET}\coco_rle\frame_*_t*ms.json"))

ts, ns = [], []
for f in files:
    m = re.search(r"frame_(\d+)_t([0-9.]+)ms", os.path.basename(f))
    fr = int(m.group(1)); t = float(m.group(2))
    d = json.load(open(f, "r"))
    n = 0
    for a in d:
        x,y,w,h = a["bbox"]
        frac = (w*h)/(W*H)
        if (frac <= GIANT) and (frac >= MINF):
            n += 1
    ts.append(t); ns.append(n)

# onset: Nmin=5, M=5
Nmin, M = 5, 5
t0 = None
for i in range(0, len(ns)-M+1):
    if all(ns[j] >= Nmin for j in range(i, i+M)):
        t0 = ts[i]
        break

tshift = [t - (t0 if t0 is not None else 0.0) for t in ts]

print("t0(ms):", t0)

plt.figure()
plt.plot(tshift, ns)
plt.xlabel("t - t0 (ms)")
plt.ylabel("n_masks (after filters)")
plt.title(f"{DATASET} | MINF={MINF} | GIANT={GIANT}")
plt.show()