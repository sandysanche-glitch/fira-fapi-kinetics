import sys
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

# Hungarian matching
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    raise ImportError("Please install scipy: pip install scipy")

# ==================================================
# PATHS
# ==================================================
ROOT = Path(r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics")
DATASETS = {
    "FAPI": {
        "frames": ROOT / "frames" / "FAPI",
        "coco_rle": ROOT / "sam" / "FAPI" / "coco_rle",
    },
    "FAPI_TEMPO": {
        "frames": ROOT / "frames" / "FAPI_TEMPO",
        "coco_rle": ROOT / "sam" / "FAPI_TEMPO" / "coco_rle",
    },
}
OUT_DIR = ROOT / "analysis_tracks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# TRACKING PARAMETERS (tune if needed)
# ==================================================
IOU_MATCH_MIN = 0.10          # minimum overlap to consider a match
DIST_MAX_PX = 120.0           # max centroid jump between frames
MAX_MISSES = 2                # allow short gaps (object missing for 1–2 frames)
AREA_MIN_PX = 200             # ignore tiny fragments

# ==================================================
# Helpers
# ==================================================
TIME_RE = re.compile(r"_t([0-9]+(?:\.[0-9]+)?)ms", re.IGNORECASE)

def time_from_stem(stem: str) -> float:
    """
    stem like: frame_00012_t24.00ms
    """
    m = TIME_RE.search(stem)
    if not m:
        raise ValueError(f"Cannot parse time from: {stem}")
    return float(m.group(1))

def rle_to_mask(rle_obj):
    """
    COCO RLE decode without pycocotools dependency.
    Assumes rle_obj is dict with 'counts' and 'size'.
    counts is either list (uncompressed) or string (compressed).
    Your pipeline writes standard COCO JSON; in practice this is usually compressed string.
    If you already have pycocotools, it’s faster — but we keep it pure python.
    """
    # If you have pycocotools available, use it:
    try:
        from pycocotools import mask as mask_util
        return mask_util.decode(rle_obj).astype(bool)
    except Exception:
        pass

    # Fallback: requires counts as list (uncompressed). If counts is string, ask user to install pycocotools.
    if isinstance(rle_obj.get("counts", None), str):
        raise RuntimeError(
            "RLE 'counts' is compressed string. Install pycocotools:\n"
            "  pip install pycocotools\n"
            "Then rerun."
        )

    counts = rle_obj["counts"]
    h, w = rle_obj["size"]
    arr = np.zeros(h * w, dtype=np.uint8)
    idx = 0
    val = 0
    for c in counts:
        if c > 0:
            arr[idx:idx + c] = val
        idx += c
        val = 1 - val
    return arr.reshape((w, h)).T.astype(bool)

def mask_area_centroid(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, (np.nan, np.nan)
    area = len(xs)
    return area, (float(xs.mean()), float(ys.mean()))

def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1, y1)

def iou_from_bboxes(b1, b2):
    # b = (x0, y0, x1, y1)
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    inter_w = max(0, xB - xA + 1)
    inter_h = max(0, yB - yA + 1)
    inter = inter_w * inter_h
    area1 = (b1[2]-b1[0]+1) * (b1[3]-b1[1]+1)
    area2 = (b2[2]-b2[0]+1) * (b2[3]-b2[1]+1)
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union

def dist(c1, c2):
    return float(np.hypot(c1[0]-c2[0], c1[1]-c2[1]))

# ==================================================
# Tracking core
# ==================================================
class Track:
    def __init__(self, track_id, t0):
        self.id = track_id
        self.t0 = t0
        self.last_t = t0
        self.misses = 0
        self.records = []  # list of dicts per frame
        self.last_centroid = None
        self.last_bbox = None

    def add(self, rec):
        self.records.append(rec)
        self.last_t = rec["time_ms"]
        self.misses = 0
        self.last_centroid = (rec["cx"], rec["cy"])
        self.last_bbox = rec["bbox"]

    def miss(self):
        self.misses += 1

def load_cells_for_frame(json_path: Path):
    """
    Load all masks in this frame and treat them as candidate 'cells'.
    If your JSON contains nuclei too, you can filter by size (AREA_MIN_PX).
    """
    with open(json_path, "r") as f:
        anns = json.load(f)

    dets = []
    for a in anns:
        # a["segmentation"] is COCO RLE
        seg = a.get("segmentation", None)
        if seg is None:
            continue

        mask = rle_to_mask(seg)
        area, (cx, cy) = mask_area_centroid(mask)
        if area < AREA_MIN_PX:
            continue
        bb = bbox_from_mask(mask)
        if bb is None:
            continue

        dets.append({
            "mask": mask,
            "area_px": int(area),
            "cx": cx,
            "cy": cy,
            "bbox": bb,
        })
    return dets

def track_dataset(label: str, frames_dir: Path, coco_rle_dir: Path):
    coco_files = sorted(coco_rle_dir.glob("*.json"))
    if len(coco_files) == 0:
        raise RuntimeError(f"No COCO JSONs found in {coco_rle_dir}")

    # Sort by time extracted from filename stem
    coco_files = sorted(coco_files, key=lambda p: time_from_stem(p.stem))

    tracks = []
    active = []
    next_track_id = 1

    for k, jf in enumerate(coco_files):
        t = time_from_stem(jf.stem)
        dets = load_cells_for_frame(jf)

        # If no active tracks, start new ones
        if len(active) == 0:
            for d in dets:
                tr = Track(next_track_id, t)
                next_track_id += 1
                tr.add({
                    "track_id": tr.id,
                    "time_ms": t,
                    "area_px": d["area_px"],
                    "cx": d["cx"],
                    "cy": d["cy"],
                    "bbox": d["bbox"],
                })
                active.append(tr)
            continue

        # Build cost matrix between active tracks and current detections
        # Cost uses (1 - IoU) plus centroid distance penalty.
        nT = len(active)
        nD = len(dets)
        cost = np.full((nT, nD), 1e6, dtype=float)

        for i, tr in enumerate(active):
            for j, d in enumerate(dets):
                # gating by distance
                if tr.last_centroid is not None:
                    dj = dist(tr.last_centroid, (d["cx"], d["cy"]))
                    if dj > DIST_MAX_PX:
                        continue
                else:
                    dj = 0.0

                # overlap proxy using bbox IoU (fast)
                iou = iou_from_bboxes(tr.last_bbox, d["bbox"]) if tr.last_bbox else 0.0

                # gating by overlap OR distance (allow match if close even if bbox IoU small)
                if iou < IOU_MATCH_MIN and dj > (0.5 * DIST_MAX_PX):
                    continue

                # cost: prefer high IoU and small distance
                cost[i, j] = (1.0 - iou) + 0.002 * dj

        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost)

        assigned_tracks = set()
        assigned_dets = set()

        # Apply matches below a threshold
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= 1e5:
                continue
            tr = active[r]
            d = dets[c]
            tr.add({
                "track_id": tr.id,
                "time_ms": t,
                "area_px": d["area_px"],
                "cx": d["cx"],
                "cy": d["cy"],
                "bbox": d["bbox"],
            })
            assigned_tracks.add(r)
            assigned_dets.add(c)

        # Unmatched tracks -> miss counter
        new_active = []
        for i, tr in enumerate(active):
            if i not in assigned_tracks:
                tr.miss()
            if tr.misses <= MAX_MISSES:
                new_active.append(tr)
            else:
                tracks.append(tr)
        active = new_active

        # Unmatched detections -> start new tracks (nucleation)
        for j, d in enumerate(dets):
            if j in assigned_dets:
                continue
            tr = Track(next_track_id, t)
            next_track_id += 1
            tr.add({
                "track_id": tr.id,
                "time_ms": t,
                "area_px": d["area_px"],
                "cx": d["cx"],
                "cy": d["cy"],
                "bbox": d["bbox"],
            })
            active.append(tr)

    # close remaining
    tracks.extend(active)

    # Convert to dataframe
    records = []
    for tr in tracks:
        for rec in tr.records:
            records.append(rec)

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(f"No tracks produced for {label}")

    # Compute R(t) and v(t) per track
    df = df.sort_values(["track_id", "time_ms"]).reset_index(drop=True)
    df["R_px"] = np.sqrt(df["area_px"] / np.pi)

    # Per-track derivative
    df["v_px_per_ms"] = np.nan
    for tid, g in df.groupby("track_id"):
        t = g["time_ms"].values
        R = g["R_px"].values
        v = np.zeros_like(R, dtype=float)
        if len(R) == 1:
            v[:] = np.nan
        else:
            v[0] = (R[1] - R[0]) / (t[1] - t[0])
            v[-1] = (R[-1] - R[-2]) / (t[-1] - t[-2])
            for i in range(1, len(R) - 1):
                v[i] = (R[i + 1] - R[i - 1]) / (t[i + 1] - t[i - 1])
        df.loc[g.index, "v_px_per_ms"] = v

    df["v_px_per_s"] = df["v_px_per_ms"] * 1000.0

    # Track birth times (nucleation times)
    t_birth = df.groupby("track_id")["time_ms"].min().rename("t_nuc_ms")
    df = df.merge(t_birth, on="track_id", how="left")

    return df


def make_plots(label: str, df: pd.DataFrame):
    import matplotlib.pyplot as plt

    # Plot a few trajectories
    plt.figure()
    for tid, g in df.groupby("track_id"):
        if len(g) < 5:
            continue
        plt.plot(g["time_ms"], g["R_px"], alpha=0.35)
        # only plot first ~30 tracks to keep readable
        if tid >= 30:
            break
    plt.xlabel("Time (ms)")
    plt.ylabel("R (px)")
    plt.title(f"{label}: sample spherulite trajectories R_i(t)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{label}_Ri_t.png", dpi=200)
    plt.close()

    # Mean growth curve
    gmean = df.groupby("time_ms").agg(R_mean=("R_px", "mean"), R_std=("R_px", "std")).reset_index()
    plt.figure()
    plt.plot(gmean["time_ms"], gmean["R_mean"])
    plt.xlabel("Time (ms)")
    plt.ylabel("Mean R (px)")
    plt.title(f"{label}: mean growth ⟨R(t)⟩")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{label}_Rmean_t.png", dpi=200)
    plt.close()

    # Nucleation times histogram
    tb = df[["track_id", "t_nuc_ms"]].drop_duplicates()
    plt.figure()
    plt.hist(tb["t_nuc_ms"], bins=20)
    plt.xlabel("Nucleation time (ms)")
    plt.ylabel("Count")
    plt.title(f"{label}: nucleation time distribution")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{label}_t_nuc_hist.png", dpi=200)
    plt.close()


# ==================================================
# RUN
# ==================================================
all_summaries = []

for label, cfg in DATASETS.items():
    print(f"\n=== TRACKING {label} ===")
    df_tracks = track_dataset(label, cfg["frames"], cfg["coco_rle"])

    out_csv = OUT_DIR / f"{label}_tracks.csv"
    df_tracks.drop(columns=["bbox"], errors="ignore").to_csv(out_csv, index=False)
    print(f"Saved tracks: {out_csv}")

    make_plots(label, df_tracks)

    # summary
    tb = df_tracks[["track_id", "t_nuc_ms"]].drop_duplicates()
    summary = {
        "dataset": label,
        "n_tracks": int(tb.shape[0]),
        "t_first_nuc_ms": float(tb["t_nuc_ms"].min()),
        "t_last_nuc_ms": float(tb["t_nuc_ms"].max()),
        "R_max_px": float(df_tracks["R_px"].max()),
        "v_max_px_per_s": float(df_tracks["v_px_per_s"].max()),
    }
    all_summaries.append(summary)

summary_df = pd.DataFrame(all_summaries)
summary_df.to_csv(OUT_DIR / "tracking_summary.csv", index=False)
print("\nDone. Outputs in:", OUT_DIR)
