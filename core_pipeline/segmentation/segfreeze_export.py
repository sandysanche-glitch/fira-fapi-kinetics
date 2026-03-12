# segfreeze_export.py
# Freeze + export counts + shift-time (dataset-specific t0) for SAM COCO-RLE json outputs.
#
# Outputs in RUN_ROOT:
#   counts_filtered.csv
#   counts_shifted.csv
#   segfreeze_manifest.json
#
# Usage (from Kinetics folder):
#   python segfreeze_export.py --run_root sam\segfreeze_v1_fapi_vs_tempo --datasets FAPI,FAPI_TEMPO --giant 0.5 --min_frac 0.006 --nmin 5 --m 5
#
# Notes:
# - Expects JSONs at: <run_root>\<DATASET>\coco_rle\frame_*_t*ms.json
# - JSON must contain entries with key "bbox" = [x,y,w,h] (SAM output you already have)

import argparse
import csv
import glob
import hashlib
import json
import os
import platform
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from statistics import median

# Set these to your full-frame resolution (used for bbox_frac)
W_DEFAULT, H_DEFAULT = 4704, 3424

FRAME_RE = re.compile(r"frame_(\d+)_t([0-9.]+)ms", re.IGNORECASE)

@dataclass
class RunParams:
    run_root: str
    datasets: list
    width: int
    height: int
    giant_bbox_frac: float
    min_bbox_frac: float
    onset_nmin: int
    onset_m: int

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def parse_frame_and_time_ms(filename: str):
    m = FRAME_RE.search(os.path.basename(filename))
    if not m:
        raise ValueError(f"Could not parse frame/time from: {filename}")
    fr = int(m.group(1))
    t = float(m.group(2))
    return fr, t

def load_filtered_counts(json_files, W, H, GIANT, MINF):
    """
    Returns list of dict rows:
      dataset, frame, t_ms, n_raw, n_kept, n_dropped_giant, n_dropped_small,
      max_bbox_frac_kept, sum_bbox_area_frac_kept
    and a helper vector for onset detection: (frame, t_ms, n_kept)
    """
    rows = []
    onset_vec = []  # (frame, t_ms, n_kept)
    for f in json_files:
        fr, t = parse_frame_and_time_ms(f)
        with open(f, "r", encoding="utf-8") as fh:
            d = json.load(fh)

        n_raw = len(d)
        n_kept = 0
        drop_giant = 0
        drop_small = 0
        max_frac_kept = 0.0
        sum_frac_kept = 0.0

        for a in d:
            x, y, w, h = a["bbox"]
            frac = (w * h) / (W * H)

            if frac > GIANT:
                drop_giant += 1
                continue
            if frac < MINF:
                drop_small += 1
                continue

            n_kept += 1
            sum_frac_kept += frac
            if frac > max_frac_kept:
                max_frac_kept = frac

        rows.append({
            "frame": fr,
            "t_ms": t,
            "n_raw": n_raw,
            "n_kept": n_kept,
            "n_dropped_giant": drop_giant,
            "n_dropped_small": drop_small,
            "max_bbox_frac_kept": max_frac_kept if n_kept > 0 else "",
            "sum_bbox_area_frac_kept": sum_frac_kept if n_kept > 0 else 0.0,
        })
        onset_vec.append((fr, t, n_kept))
    return rows, onset_vec

def find_onset_persistent(onset_vec, Nmin=5, M=5):
    """
    onset_vec: list of (frame, t_ms, n_kept)
    Returns (frame, t_ms, n_kept_at_onset) or None
    """
    ns = [x[2] for x in onset_vec]
    for i in range(0, len(ns) - M + 1):
        if all(ns[j] >= Nmin for j in range(i, i + M)):
            fr, t, n = onset_vec[i]
            return fr, t, n
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True, help=r"e.g. sam\segfreeze_v1_fapi_vs_tempo")
    ap.add_argument("--datasets", required=True, help="comma-separated, e.g. FAPI,FAPI_TEMPO")
    ap.add_argument("--width", type=int, default=W_DEFAULT)
    ap.add_argument("--height", type=int, default=H_DEFAULT)
    ap.add_argument("--giant", type=float, default=0.5, help="drop bbox_frac > giant")
    ap.add_argument("--min_frac", type=float, default=0.006, help="drop bbox_frac < min_frac")
    ap.add_argument("--nmin", type=int, default=5, help="onset: require >=nmin masks")
    ap.add_argument("--m", type=int, default=5, help="onset: for m consecutive frames")

    args = ap.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    run_root = args.run_root

    params = RunParams(
        run_root=run_root,
        datasets=datasets,
        width=args.width,
        height=args.height,
        giant_bbox_frac=args.giant,
        min_bbox_frac=args.min_frac,
        onset_nmin=args.nmin,
        onset_m=args.m,
    )

    # Collect per-dataset stats + file hashes for freezing
    manifest = {
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "script": os.path.basename(__file__) if "__file__" in globals() else "segfreeze_export.py",
        "python": sys.version.replace("\n", " "),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "params": asdict(params),
        "datasets": {},
    }

    filtered_rows_all = []   # for counts_filtered.csv
    shifted_rows_all = []    # for counts_shifted.csv

    for ds in datasets:
        json_glob = os.path.join(run_root, ds, "coco_rle", "frame_*_t*ms.json")
        files = sorted(glob.glob(json_glob))
        if not files:
            raise FileNotFoundError(f"No JSON files found for dataset '{ds}' at: {json_glob}")

        # Hash list (strict freeze proof)
        file_hashes = []
        for f in files:
            file_hashes.append({
                "file": os.path.normpath(f),
                "sha256": sha256_file(f),
                "size_bytes": os.path.getsize(f),
            })

        rows, onset_vec = load_filtered_counts(
            files, params.width, params.height, params.giant_bbox_frac, params.min_bbox_frac
        )
        onset = find_onset_persistent(onset_vec, Nmin=params.onset_nmin, M=params.onset_m)
        if onset is None:
            # Still write outputs but mark t0 as None
            t0 = None
        else:
            t0 = float(onset[1])

        ns_kept = [r["n_kept"] for r in rows]
        empty_after = sum(1 for n in ns_kept if n == 0)

        # Dataset manifest block
        manifest["datasets"][ds] = {
            "n_json": len(files),
            "t0_ms_persistent_onset": t0,
            "onset_rule": {"Nmin": params.onset_nmin, "M": params.onset_m},
            "empty_frames_after_filter": empty_after,
            "median_n_kept": float(median(ns_kept)),
            "first_nonzero_after_filter": next(
                ({"frame": r["frame"], "t_ms": r["t_ms"], "n_kept": r["n_kept"]} for r in rows if r["n_kept"] > 0),
                None
            ),
            "last_frame": {
                "frame": rows[-1]["frame"],
                "t_ms": rows[-1]["t_ms"],
                "n_raw": rows[-1]["n_raw"],
                "n_kept": rows[-1]["n_kept"],
                "n_dropped_giant": rows[-1]["n_dropped_giant"],
                "n_dropped_small": rows[-1]["n_dropped_small"],
            },
            "files": file_hashes,
        }

        # Build CSV rows
        for r in rows:
            out = {
                "dataset": ds,
                **r,
                "t0_ms": t0 if t0 is not None else "",
                "t_shifted_ms": (r["t_ms"] - t0) if t0 is not None else "",
            }
            filtered_rows_all.append(out)

            shifted_rows_all.append({
                "dataset": ds,
                "frame": r["frame"],
                "t_shifted_ms": (r["t_ms"] - t0) if t0 is not None else "",
                "t_ms": r["t_ms"],
                "t0_ms": t0 if t0 is not None else "",
                "n_kept": r["n_kept"],
                "sum_bbox_area_frac_kept": r["sum_bbox_area_frac_kept"],
                "max_bbox_frac_kept": r["max_bbox_frac_kept"],
            })

        print(f"[{ds}] n_json={len(files)}  t0_ms={t0}  empty_after={empty_after}  median_n_kept={median(ns_kept)}")

    # Write outputs
    os.makedirs(run_root, exist_ok=True)

    counts_filtered_path = os.path.join(run_root, "counts_filtered.csv")
    counts_shifted_path = os.path.join(run_root, "counts_shifted.csv")
    manifest_path = os.path.join(run_root, "segfreeze_manifest.json")

    # counts_filtered.csv (full detail)
    filtered_fields = [
        "dataset", "frame", "t_ms", "t0_ms", "t_shifted_ms",
        "n_raw", "n_kept", "n_dropped_giant", "n_dropped_small",
        "max_bbox_frac_kept", "sum_bbox_area_frac_kept",
    ]
    with open(counts_filtered_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=filtered_fields)
        w.writeheader()
        for r in filtered_rows_all:
            w.writerow({k: r.get(k, "") for k in filtered_fields})

    # counts_shifted.csv (lean table for kinetics/plots)
    shifted_fields = [
        "dataset", "frame", "t_shifted_ms", "t_ms", "t0_ms",
        "n_kept", "sum_bbox_area_frac_kept", "max_bbox_frac_kept",
    ]
    with open(counts_shifted_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=shifted_fields)
        w.writeheader()
        for r in shifted_rows_all:
            w.writerow({k: r.get(k, "") for k in shifted_fields})

    # segfreeze_manifest.json
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nWROTE:")
    print(" -", os.path.normpath(counts_filtered_path))
    print(" -", os.path.normpath(counts_shifted_path))
    print(" -", os.path.normpath(manifest_path))
    print("\nDone.")

if __name__ == "__main__":
    main()