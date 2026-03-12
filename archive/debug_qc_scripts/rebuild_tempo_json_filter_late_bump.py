import os
import re
import json
import math
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Requires: pycocotools
try:
    from pycocotools import mask as maskUtils
except Exception as e:
    raise ImportError(
        "pycocotools is required. Install with:\n"
        "  pip install pycocotools\n"
        "or in conda:\n"
        "  conda install -c conda-forge pycocotools\n"
    ) from e


# -----------------------------
# Helpers
# -----------------------------
TIME_RE = re.compile(r"_t([0-9]+(?:\.[0-9]+)?)ms", re.IGNORECASE)

def parse_time_ms_from_filename(fname: str) -> float:
    m = TIME_RE.search(fname)
    if not m:
        return np.nan
    return float(m.group(1))

def r_from_area(area_px: float) -> float:
    # effective radius from area
    return math.sqrt(max(area_px, 0.0) / math.pi)

def ensure_rle_counts_str(rle):
    """
    pycocotools expects 'counts' to be bytes or str.
    Some JSONs store counts as list[int]; maskUtils.frPyObjects can fix polygons,
    but for RLE list[int] we need to encode. Most SAM COCO RLE is already str.
    """
    if isinstance(rle, dict) and "counts" in rle:
        if isinstance(rle["counts"], list):
            # Convert uncompressed RLE (counts list) to compressed
            rle_c = maskUtils.frPyObjects(rle, rle["size"][0], rle["size"][1])
            return rle_c
    return rle

def rle_area(rle) -> float:
    rle = ensure_rle_counts_str(rle)
    return float(maskUtils.area(rle))

def rle_bbox(rle):
    rle = ensure_rle_counts_str(rle)
    bb = maskUtils.toBbox(rle)  # [x,y,w,h]
    return [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]

def rle_merge(rles):
    rles = [ensure_rle_counts_str(r) for r in rles]
    if len(rles) == 1:
        return rles[0]
    return maskUtils.merge(rles, intersect=False)

def choose_best_ann(anns, key=("purity", "overlap_frac", "area")):
    # pick highest purity, then overlap, then area
    def score(a):
        return (
            float(a.get(key[0], 0.0)),
            float(a.get(key[1], 0.0)),
            float(a.get(key[2], a.get("area_px", a.get("area", 0.0)))),
        )
    return max(anns, key=score)

def compute_hist_curves(event_times_ms, edges, bin_ms):
    counts, _ = np.histogram(event_times_ms, bins=edges)
    cumN = np.cumsum(counts)
    dn_dt = counts / (bin_ms / 1000.0)  # 1/s
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, cumN, dn_dt

def make_edges(times_ms, bin_ms, t_min=0.0):
    if len(times_ms) == 0:
        t_max = 0.0
    else:
        t_max = float(np.nanmax(times_ms))
    t_max = bin_ms * np.ceil(t_max / bin_ms)
    return np.arange(t_min, t_max + bin_ms, bin_ms, dtype=float)


# -----------------------------
# Main rebuild logic
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="TEMPO folder with frame_*.json (id-mapped from clean)")
    ap.add_argument("--out_dir", required=True, help="Where to write filtered JSONs + CSVs + plots")

    # Basic per-annotation gates
    ap.add_argument("--min_area_px", type=float, default=800.0)
    ap.add_argument("--min_overlap_frac", type=float, default=0.30)
    ap.add_argument("--min_purity", type=float, default=0.70)

    # Stable nucleation gate (ID must persist >= L consecutive frames)
    ap.add_argument("--L", type=int, default=5)

    # Late bump / physics gates
    ap.add_argument("--R_NUC_MAX", type=float, default=60.0, help="Reject nucleation events already big at birth")
    ap.add_argument("--RNuc_OVER_Rmax_MAX", type=float, default=0.50, help="Reject if R_nuc/R_max exceeds this (ID-switch artifact)")

    # Output curves
    ap.add_argument("--bin_ms", type=float, default=20.0)

    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_json = out_dir / "json_filtered"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.mkdir(parents=True, exist_ok=True)

    json_files = sorted(in_dir.glob("frame_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No files frame_*.json in: {in_dir}")

    print(f"[OK] Input JSONs: {len(json_files)} from {in_dir}")
    print(f"[OK] Output dir:  {out_dir}")

    # Pass 1: load + per-frame filter + group by id
    # We'll store per-frame candidates by (frame_index-ish, time_ms)
    frames = []  # list of dict: {file, time_ms, anns_filtered}
    id_presence = defaultdict(list)  # id -> list of (time_ms, frame_i, R_px, area_px)

    dropped_counts = defaultdict(int)
    kept_counts = defaultdict(int)

    for fi, jf in enumerate(json_files):
        t_ms = parse_time_ms_from_filename(jf.name)
        with open(jf, "r") as f:
            data = json.load(f)

        # Some pipelines store dict with "annotations"
        if isinstance(data, dict) and "annotations" in data:
            anns = data["annotations"]
        else:
            anns = data

        filtered = []
        for a in anns:
            gid = a.get("id", a.get("grain_id", a.get("final_id", None)))
            if gid is None:
                dropped_counts["missing_id"] += 1
                continue
            gid = int(gid)

            purity = float(a.get("purity", a.get("mask_purity", np.nan)))
            overlap = float(a.get("overlap_frac", a.get("overlap", np.nan)))

            # area / R
            area = float(a.get("area_px", a.get("area", np.nan)))
            if np.isnan(area):
                # try compute from RLE
                if "segmentation" in a:
                    try:
                        area = rle_area(a["segmentation"])
                    except Exception:
                        area = np.nan

            if np.isnan(area):
                dropped_counts["missing_area"] += 1
                continue

            if area < args.min_area_px:
                dropped_counts["min_area"] += 1
                continue

            # If purity/overlap are missing, keep but mark as 1.0 (optimistic).
            # In your idmap-fromclean folder they should exist.
            if np.isnan(purity): purity = 1.0
            if np.isnan(overlap): overlap = 1.0

            if overlap < args.min_overlap_frac:
                dropped_counts["min_overlap"] += 1
                continue
            if purity < args.min_purity:
                dropped_counts["min_purity"] += 1
                continue

            Rpx = float(a.get("R_px", a.get("R", np.nan)))
            if np.isnan(Rpx):
                Rpx = r_from_area(area)

            a2 = dict(a)
            a2["id"] = gid
            a2["purity"] = purity
            a2["overlap_frac"] = overlap
            a2["area_px"] = area
            a2["R_px"] = Rpx

            filtered.append(a2)
            kept_counts["anns"] += 1
            id_presence[gid].append((t_ms, fi, Rpx, area))

        frames.append({"file": jf, "time_ms": t_ms, "anns": filtered})

    print("[OK] Per-annotation filtering done.")
    print("  kept anns:", kept_counts["anns"])
    if dropped_counts:
        print("  dropped (top reasons):")
        for k, v in sorted(dropped_counts.items(), key=lambda x: -x[1])[:8]:
            print(f"   - {k}: {v}")

    # Pass 2: stable nucleation per ID (persistence >= L consecutive frames)
    # Build per-id sorted by frame order
    stable_events = []
    rejected_ids = []

    # Determine which IDs are valid after stability + physics gates
    kept_ids = set()

    for gid, obs in id_presence.items():
        obs_sorted = sorted(obs, key=lambda x: x[1])  # by fi
        frame_idxs = np.array([o[1] for o in obs_sorted], dtype=int)
        tms = np.array([o[0] for o in obs_sorted], dtype=float)
        Rpx = np.array([o[2] for o in obs_sorted], dtype=float)

        # Find first run of >= L consecutive frames in frame_idxs
        # consecutive meaning diff==1
        if len(frame_idxs) < args.L:
            rejected_ids.append((gid, "too_few_obs"))
            continue

        dif = np.diff(frame_idxs)
        # positions where run breaks
        run_start = 0
        found = False
        for i in range(len(dif) + 1):
            # end run when i==len(dif) or dif[i]!=1
            if i == len(dif) or dif[i] != 1:
                run_len = i - run_start + 1
                if run_len >= args.L:
                    # nucleation at run_start
                    nuc_idx = run_start
                    found = True
                    break
                run_start = i + 1

        if not found:
            rejected_ids.append((gid, "no_L_consecutive"))
            continue

        nuc_time = float(tms[nuc_idx])
        nuc_frame_i = int(frame_idxs[nuc_idx])

        # R_nuc: median over first L frames of that run (more robust)
        run_slice = slice(nuc_idx, nuc_idx + args.L)
        R_nuc = float(np.median(Rpx[run_slice]))

        # R_max: max over entire lifetime
        R_max = float(np.max(Rpx))
        if R_max <= 0:
            rejected_ids.append((gid, "Rmax_nonpos"))
            continue

        # Physics gates to remove late bump artifacts
        if R_nuc > args.R_NUC_MAX:
            rejected_ids.append((gid, f"R_nuc>{args.R_NUC_MAX}"))
            continue

        if (R_nuc / R_max) > args.RNuc_OVER_Rmax_MAX:
            rejected_ids.append((gid, f"R_nuc/R_max>{args.RNuc_OVER_Rmax_MAX}"))
            continue

        kept_ids.add(gid)
        stable_events.append({
            "id": gid,
            "nuc_time_ms": nuc_time,
            "nuc_frame_i": nuc_frame_i,
            "R_nuc_px": R_nuc,
            "R_max_px": R_max,
            "n_obs_total": int(len(obs_sorted)),
        })

    stable_events_df = pd.DataFrame(stable_events).sort_values("nuc_time_ms").reset_index(drop=True)
    rej_df = pd.DataFrame(rejected_ids, columns=["id", "reason"])

    print("\n=== Stable nucleation IDs ===")
    print(f"  kept IDs: {len(kept_ids)}")
    print(f"  rejected IDs: {len(rejected_ids)}")
    if len(rej_df):
        print("  rejected reasons (top):")
        print(rej_df["reason"].value_counts().head(10))

    # Pass 3: rebuild filtered JSONs per frame:
    # - keep only anns whose id is in kept_ids
    # - if multiple anns map to same id in same frame, merge their RLEs and compute new area/bbox
    wrote = 0
    for fr in frames:
        jf = fr["file"]
        t_ms = fr["time_ms"]
        anns = [a for a in fr["anns"] if int(a["id"]) in kept_ids]

        by_id = defaultdict(list)
        for a in anns:
            by_id[int(a["id"])].append(a)

        out_anns = []
        for gid, group in by_id.items():
            if len(group) == 1:
                out_anns.append(group[0])
            else:
                # merge segmentations (union), keep best metadata
                best = choose_best_ann(group)
                if "segmentation" in best and all("segmentation" in g for g in group):
                    try:
                        merged = rle_merge([g["segmentation"] for g in group])
                        best = dict(best)
                        best["segmentation"] = merged
                        best["area_px"] = rle_area(merged)
                        bb = rle_bbox(merged)
                        best["bbox_x"], best["bbox_y"], best["bbox_w"], best["bbox_h"] = bb
                        best["R_px"] = r_from_area(best["area_px"])
                    except Exception:
                        # fallback: just keep best
                        pass
                out_anns.append(best)

        out_path = out_json / jf.name
        with open(out_path, "w") as f:
            json.dump(out_anns, f)
        wrote += 1

    print(f"\n[OK] Wrote filtered JSONs: {wrote} -> {out_json}")

    # Pass 4: compute N(t) and dn/dt from stable nucleation events
    events_out = out_dir / "nucleation_events_filtered_TEMPO.csv"
    rej_out = out_dir / "nucleation_events_rejected_TEMPO.csv"
    stable_events_df.to_csv(events_out, index=False)
    rej_df.to_csv(rej_out, index=False)

    times = stable_events_df["nuc_time_ms"].to_numpy(dtype=float) if len(stable_events_df) else np.array([], dtype=float)
    edges = make_edges(times, args.bin_ms, t_min=0.0)
    centers, cumN, dn_dt = compute_hist_curves(times, edges, args.bin_ms)

    Nt_df = pd.DataFrame({"bin_center_ms": centers, "cum_n": cumN})
    dn_df = pd.DataFrame({"bin_center_ms": centers, "dn_dt_per_s": dn_dt, "n_nucleated": np.diff(np.r_[0, cumN])})

    Nt_out = out_dir / "N_t_filtered_TEMPO.csv"
    dn_out = out_dir / "dn_dt_filtered_TEMPO.csv"
    Nt_df.to_csv(Nt_out, index=False)
    dn_df.to_csv(dn_out, index=False)

    # Plot
    png_out = out_dir / "tempo_stable_nucleation_dn_dt.png"
    plt.figure()
    plt.plot(centers, dn_dt, marker="o")
    plt.xlabel("time (ms)")
    plt.ylabel("dn/dt (1/s)")
    plt.title(
        f"TEMPO stable nucleation (bin={args.bin_ms} ms)\n"
        f"min_area>={args.min_area_px}, overlap>={args.min_overlap_frac}, purity>={args.min_purity}, "
        f"L>={args.L}, R_nuc<={args.R_NUC_MAX}, R_nuc/R_max<={args.RNuc_OVER_Rmax_MAX}\n"
        f"kept IDs={len(kept_ids)}"
    )
    plt.tight_layout()
    plt.savefig(png_out, dpi=200)
    plt.close()

    # Methods text
    methods = out_dir / "tempo_rebuild_methods.txt"
    with open(methods, "w") as f:
        f.write(
            "TEMPO late-bump artifact suppression (JSON-level rebuild)\n"
            "------------------------------------------------------\n"
            f"Input masks: {in_dir}\n"
            f"Output masks: {out_json}\n\n"
            "Per-annotation filtering (applied to each frame):\n"
            f"  - area_px >= {args.min_area_px}\n"
            f"  - overlap_frac >= {args.min_overlap_frac}\n"
            f"  - purity >= {args.min_purity}\n\n"
            "Stable nucleation definition (per final-grain ID):\n"
            f"  - nucleation time is first frame where the ID is detected and persists for >= {args.L} consecutive frames\n"
            "  - R_nuc computed as median effective radius over first L consecutive detections\n"
            "  - R_max computed as maximum effective radius over the full track lifetime\n\n"
            "Physics / artifact gates (remove late split/ID-switch events):\n"
            f"  - R_nuc_px <= {args.R_NUC_MAX}  (reject already-large-at-birth)\n"
            f"  - R_nuc/R_max <= {args.RNuc_OVER_Rmax_MAX}  (reject nonphysical re-appearance / ID-switch)\n\n"
            f"Binning for dn/dt: bin_ms = {args.bin_ms}\n"
        )

    print("\n[OK] Wrote:")
    print("  ", events_out)
    print("  ", rej_out)
    print("  ", Nt_out)
    print("  ", dn_out)
    print("  ", png_out)
    print("  ", methods)


if __name__ == "__main__":
    main()
