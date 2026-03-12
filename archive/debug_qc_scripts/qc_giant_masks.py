import glob, json, os
W,H = 4704,3424

def scan(pattern, name):
    files = sorted(glob.glob(pattern))
    if not files:
        print(name, "NO FILES:", pattern)
        return
    huge = 0
    nonempty = 0
    max_area = 0.0
    for f in files:
        data = json.load(open(f,"r"))
        if not data:
            continue
        nonempty += 1
        # SAM JSONs include "area"
        areas = [a.get("area", 0.0)/(W*H) for a in data]
        ma = max(areas) if areas else 0.0
        max_area = max(max_area, ma)
        if ma > 0.5:
            huge += 1
    print(f"{name}: n_json={len(files)} nonempty={nonempty} huge_frames(>0.5)={huge} huge_rate={huge/max(1,nonempty):.3f} max_area_frac_seen={max_area:.4f}")

scan(r"sam\segfreeze_v1_fapi_vs_tempo\FAPI\coco_rle\frame_*_t*ms.json", "FAPI")
scan(r"sam\segfreeze_v1_fapi_vs_tempo\FAPI_TEMPO\coco_rle\frame_*_t*ms.json", "FAPI_TEMPO")