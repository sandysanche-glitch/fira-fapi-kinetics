import glob
import json
import os

JSON_DIR = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\sam_cuda_vith_clean\FAPI_TEMPO\rebuild_filtered_win60\json_filtered"

def summarize_seg(seg):
    if isinstance(seg, dict):
        keys = list(seg.keys())
        counts = seg.get("counts", None)
        size = seg.get("size", None)
        out = {
            "seg_type": "dict",
            "seg_keys": keys,
            "size": size,
            "counts_type": str(type(counts)),
        }
        if isinstance(counts, str):
            out["counts_preview"] = counts[:80]
            out["counts_len"] = len(counts)
        elif isinstance(counts, (bytes, bytearray)):
            out["counts_preview"] = counts[:20]
            out["counts_len"] = len(counts)
        elif isinstance(counts, list):
            out["counts_preview"] = counts[:20]
            out["counts_len"] = len(counts)
        else:
            out["counts_preview"] = counts
        return out

    if isinstance(seg, list):
        return {"seg_type": "list", "len": len(seg), "first_type": str(type(seg[0])) if len(seg) else None}

    return {"seg_type": str(type(seg)), "value_preview": str(seg)[:120]}

fs = sorted(glob.glob(os.path.join(JSON_DIR, "frame_*_idmapped.json")))
print("json_dir:", JSON_DIR)
print("files:", len(fs))

picked = None
for f in fs:
    try:
        d = json.load(open(f, "r"))
    except Exception as e:
        print("[READ FAIL]", os.path.basename(f), "->", repr(e))
        continue
    if isinstance(d, list) and len(d) > 0:
        picked = (f, d)
        break

if picked is None:
    print("No non-empty JSONs found in folder.")
    raise SystemExit(0)

f, d = picked
print("\nFOUND non-empty:")
print("file:", f)
print("n_anns:", len(d))

a = d[0]
print("ann keys:", list(a.keys()))
seg = a.get("segmentation", None)
print("seg summary:", summarize_seg(seg))

# also show alt keys if present
for k in ["id", "track_id", "det_id", "area", "area_px", "purity"]:
    if k in a:
        print(f"{k}:", a.get(k))
