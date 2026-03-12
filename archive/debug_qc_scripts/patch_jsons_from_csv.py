import os, json, glob
import pandas as pd
from pathlib import Path

# ---- USER SETTINGS ----
JSON_DIR = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\sam_cuda_vith_clean\FAPI_TEMPO\rebuild_filtered_win60\json_filtered"
CSV_FILTERED = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\sam_cuda_vith_clean\FAPI_TEMPO\rebuild_filtered_win60\nucleation_events_filtered.csv"
OUT_DIR = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\sam_cuda_vith_clean\FAPI_TEMPO\rebuild_filtered_win60\json_patched"

# If your filenames are like "frame_000123.json" set this:
FRAME_PREFIX = "frame_"
FRAME_ZFILL = None  # e.g. 6 if you use frame_000123.json; leave None to auto-detect
# -----------------------

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_FILTERED)

# Build a per-frame lookup from the CSV.
# We’ll use nuc_frame as the key; if missing, we can fall back to nuc_time_ms -> frame conversion (only if you know fps).
if "nuc_frame" not in df.columns:
    raise ValueError("CSV has no nuc_frame column. Need nuc_frame to rebuild frame JSONs fast.")

# Keep only rows with a real nuc_frame
df2 = df.dropna(subset=["nuc_frame"]).copy()
df2["nuc_frame"] = df2["nuc_frame"].astype(int)

by_frame = {}
for _, r in df2.iterrows():
    f = int(r["nuc_frame"])
    by_frame.setdefault(f, []).append({
        "track_id": int(r["track_id"]) if "track_id" in df2.columns else None,
        "area_px": float(r["area_nuc_px"]) if "area_nuc_px" in df2.columns else None,
        "R_nuc_px": float(r["R_nuc_px"]) if "R_nuc_px" in df2.columns else None,
        "cx": float(r["cx_px"]) if "cx_px" in df2.columns else None,
        "cy": float(r["cy_px"]) if "cy_px" in df2.columns else None,
        "purity": float(r["purity"]) if "purity" in df2.columns else None,
        "time_ms": float(r["nuc_time_ms"]) if "nuc_time_ms" in df2.columns else None,
    })

def is_bad_json(path):
    # bad if missing, empty, invalid, or missing required structure
    if not os.path.exists(path):
        return True, "missing"
    if os.path.getsize(path) == 0:
        return True, "empty"
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        # Minimal structure checks (adjust to your pipeline expectations)
        if obj is None:
            return True, "null"
        if not isinstance(obj, dict):
            return True, "not_dict"
        # If your pipeline expects a 'detections' list:
        if "detections" not in obj or not isinstance(obj["detections"], list):
            return True, "missing_detections"
        return False, "ok"
    except Exception as e:
        return True, f"invalid_json: {e}"

def write_minimal_frame_json(frame_id, out_path):
    dets = by_frame.get(frame_id, [])
    # Minimal valid schema. Keep it simple but consistent.
    obj = {
        "frame_id": frame_id,
        "source": "patched_from_csv",
        "detections": dets
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# Collect JSON files in directory
json_files = sorted(glob.glob(os.path.join(JSON_DIR, "*.json")))

# If directory is empty, still patch frames referenced by CSV
if not json_files:
    frames = sorted(by_frame.keys())
    print(f"No JSONs found in {JSON_DIR}. Will generate {len(frames)} minimal JSONs from CSV.")
    for fr in frames:
        name = f"{FRAME_PREFIX}{str(fr).zfill(FRAME_ZFILL) if FRAME_ZFILL else fr}.json"
        write_minimal_frame_json(fr, os.path.join(OUT_DIR, name))
    print("Done.")
    raise SystemExit

patched = 0
copied = 0
bad_log = []

for p in json_files:
    fname = os.path.basename(p)
    # Try to parse frame number from filename
    # e.g. frame_123.json or frame_000123.json
    stem = Path(fname).stem
    frame_id = None
    if stem.startswith(FRAME_PREFIX):
        num = stem[len(FRAME_PREFIX):]
        if num.isdigit():
            frame_id = int(num)

    bad, reason = is_bad_json(p)
    out_p = os.path.join(OUT_DIR, fname)

    if bad:
        # If we can’t parse frame id, we still write something minimal to keep pipeline alive
        if frame_id is None:
            # Make a minimal JSON that won’t crash the loader
            obj = {"source": "patched_unknown_frame", "detections": []}
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2)
        else:
            write_minimal_frame_json(frame_id, out_p)
        patched += 1
        bad_log.append((fname, reason, frame_id))
    else:
        # Copy good json unchanged (fast)
        with open(p, "r", encoding="utf-8") as f:
            content = f.read()
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(content)
        copied += 1

print(f"Copied good JSONs: {copied}")
print(f"Patched bad JSONs: {patched}")

if bad_log:
    log_path = os.path.join(OUT_DIR, "_patch_log.csv")
    pd.DataFrame(bad_log, columns=["file","reason","frame_id"]).to_csv(log_path, index=False)
    print(f"Wrote patch log: {log_path}")
