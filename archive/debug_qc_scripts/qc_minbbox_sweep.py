import glob, json, os, re
from statistics import median

W, H = 4704, 3424
GIANT = 0.5

RUN_ROOT = r"sam\segfreeze_v1_fapi_vs_tempo"
DATASET = "FAPI"   # <-- changed from FAPI_TEMPO to FAPI

files = sorted(glob.glob(rf"{RUN_ROOT}\{DATASET}\coco_rle\frame_*_t*ms.json"))

def load_counts(min_frac):
    ts = []
    ns = []
    for f in files:
        m = re.search(r"frame_(\d+)_t([0-9.]+)ms", os.path.basename(f))
        fr = int(m.group(1)); t = float(m.group(2))
        d = json.load(open(f, "r"))

        n = 0
        for a in d:
            x, y, w, h = a["bbox"]
            frac = (w*h)/(W*H)
            if frac <= GIANT and frac >= min_frac:
                n += 1

        ts.append((fr, t))
        ns.append(n)
    return ts, ns

def onset_persistent(ts, ns, Nmin=5, M=5):
    for i in range(0, len(ns)-M+1):
        if all(ns[j] >= Nmin for j in range(i, i+M)):
            fr, t = ts[i]
            return fr, t, ns[i]
    return None

mins = [0.0, 0.001, 0.002, 0.004, 0.006, 0.008, 0.010]

print("=== QC min-bbox-frac sweep ===")
print("DATASET:", DATASET)
print("n_json:", len(files))
print("GIANT:", GIANT)
print()

for min_frac in mins:
    ts, ns = load_counts(min_frac)

    empty = sum(1 for x in ns if x == 0)
    med = median(ns)
    first = next(((ts[i][0], ts[i][1], ns[i]) for i in range(len(ns)) if ns[i] > 0), None)
    onset = onset_persistent(ts, ns, Nmin=5, M=5)

    # early window emptiness (0–10 ms)
    early = [ns[i] for i in range(len(ns)) if ts[i][1] <= 10.0]
    early_empty = sum(1 for x in early if x == 0)
    early_total = len(early)

    print(
        f"min_frac={min_frac:0.3f} | empty={empty:3d} | median={med:6.1f} | "
        f"first={first} | onset(Nmin=5,M=5)={onset} | early_empty={early_empty}/{early_total}"
    )