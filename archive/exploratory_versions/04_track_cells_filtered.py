# 04_track_cells_filtered.py
# Tracks spherulites ("cells") across time from SAM COCO-RLE JSONs using
# Hungarian matching (IoU proxy via bbox overlap + centroid distance).
#
# Key improvements vs earlier version:
#  - Reject impossible "giant" masks (background / whole-frame artifacts)
#  - Better size filtering for real spherulites
#  - Track-length filtering to remove 1–few-frame fragments
#  - Cohort-aligned growth curve R(τ) where τ = t - t_nuc
#  - Outputs clean QC plots and CSVs
#
# Requirements:
#   pip install numpy pandas matplotlib scipy pycocotools tqdm
#
# Run:
#   python 04_track_cells_filtered.py

import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment
from pycocotools import mask as mask_util
from tqdm import tqdm


# ==================================================
# PATHS (EDIT ONLY IF YOUR ROOT CHANGES)
# ==================================================
ROOT = Path(r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics")

DATASETS = {
    "FAPI": ROOT / "sam" / "FAPI" / "coco_rle",
    "FAPI_TEMPO": ROOT / "sam" / "FAPI_TEMPO" / "coco_rle",
}

OUT_DIR = ROOT / "analysis_tracks_filtered"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# IMAGE SIZE (from your example: 2048 x 1490)
# If some frames differ slightly, we still compute per-mask H,W from decode.
# This is only used for a sanity cap on R.
# ==================================================
IMG_W = 2048
IMG_H = 1490
IMAGE_AREA_REF = IMG_W * IMG_H
R_IMAGE_MAX = float(np.sqrt(IMAGE_AREA_REF / np.pi))  # ~985 px


# ==================================================
# DETECTION FILTERS (MOST IMPORTANT)
# ==================================================
AREA_MIN_PX = 1500              # remove tiny fragments (tune 1200–5000)
AREA_MAX_FRAC = 0.25            # reject masks > 25% of image area (background)
R_MAX_FRAC = 0.95               # reject if R > 0.95 * R_image_max (extra safety)

MIN_TRACK_LEN_FRAMES = 10       # drop short fragment tracks

# ==================================================
# TRACKING PARAMETERS
# ==================================================
IOU_MIN = 0.05                  # bbox IoU proxy threshold
DIST_MAX_PX = 140.0             # max centroid jump allowed
MAX_MISSES = 2                  # tolerate small disappearances

LAMBDA_DIST = 0.002             # weight centroid distance in assignment cost

# Smoothing for derivatives
SMOOTH_WINDOW = 9               # must be odd
SMOOTH_POLY = 2


TIME_RE = re.compile(r"_t([0-9]+(?:\.[0-9]+)?)ms", re.IGNORECASE)


def time_ms_from_stem(stem: str) -> float:
    m = TIME_RE.search(stem)
    if not m:
        raise ValueError(f"Cannot parse time from filename stem: {stem}")
    return float(m.group(1))


def decode_rle(rle):
    m = mask_util.decode(rle)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(bool)


def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))  # x0,y0,x1,y1


def bbox_iou(b1, b2) -> float:
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    inter = interW * interH
    area1 = (b1[2] - b1[0] + 1) * (b1[3] - b1[1] + 1)
    area2 = (b2[2] - b2[0] + 1) * (b2[3] - b2[1] + 1)
    union = area1 + area2 - inter
    return 0.0 if union <= 0 else inter / union


def euclid(c1, c2) -> float:
    return float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))


def smooth_series(y: np.ndarray):
    if len(y) < SMOOTH_WINDOW:
        return y
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(y, window_length=SMOOTH_WINDOW, polyorder=SMOOTH_POLY)
    except Exception:
        return y


def compute_derivative(t, y):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    v = np.full_like(y, np.nan, dtype=float)
    if len(y) < 2:
        return v
    v[0] = (y[1] - y[0]) / (t[1] - t[0])
    v[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2])
    for i in range(1, len(y) - 1):
        v[i] = (y[i + 1] - y[i - 1]) / (t[i + 1] - t[i - 1])
    return v


def load_detections_cells(json_path: Path):
    """
    Load SAM detections from one frame and filter to plausible spherulite masks.
    """
    with open(json_path, "r") as f:
        anns = json.load(f)

    dets = []
    for a in anns:
        seg = a.get("segmentation", None)
        if seg is None:
            continue

        mask = decode_rle(seg)
        H, W = mask.shape
        image_area = H * W

        area = int(mask.sum())
        if area < AREA_MIN_PX:
            continue

        # reject background-like huge masks
        if area > AREA_MAX_FRAC * image_area:
            continue

        # extra safety: reject impossible radii (kills the ~2300 px bug)
        R = float(np.sqrt(area / np.pi))
        R_max_local = float(np.sqrt(image_area / np.pi))
        if R > R_MAX_FRAC * R_max_local:
            continue

        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue

        cx = float(xs.mean())
        cy = float(ys.mean())
        bb = bbox_from_mask(mask)
        if bb is None:
            continue

        dets.append({
            "area_px": area,
            "R_px": R,
            "cx": cx,
            "cy": cy,
            "bbox": bb,
        })
    return dets


class Track:
    def __init__(self, tid: int, t0: float):
        self.id = tid
        self.t0 = t0
        self.last_centroid = None
        self.last_bbox = None
        self.misses = 0
        self.records = []

    def add(self, t, det):
        self.records.append({
            "track_id": self.id,
            "time_ms": float(t),
            "area_px": int(det["area_px"]),
            "R_px": float(det["R_px"]),
            "cx": float(det["cx"]),
            "cy": float(det["cy"]),
        })
        self.last_centroid = (det["cx"], det["cy"])
        self.last_bbox = det["bbox"]
        self.misses = 0

    def miss(self):
        self.misses += 1


def hungarian_match(active_tracks, detections):
    nT = len(active_tracks)
    nD = len(detections)
    if nT == 0 or nD == 0:
        return []

    BIG = 1e6
    cost = np.full((nT, nD), BIG, dtype=float)

    for i, tr in enumerate(active_tracks):
        for j, d in enumerate(detections):
            dj = euclid(tr.last_centroid, (d["cx"], d["cy"])) if tr.last_centroid else 0.0
            if dj > DIST_MAX_PX:
                continue

            iou = bbox_iou(tr.last_bbox, d["bbox"]) if tr.last_bbox else 0.0
            if iou < IOU_MIN and dj > (0.5 * DIST_MAX_PX):
                continue

            cost[i, j] = (1.0 - iou) + LAMBDA_DIST * dj

    r, c = linear_sum_assignment(cost)
    pairs = []
    for rr, cc in zip(r, c):
        if cost[rr, cc] >= BIG * 0.5:
            continue
        pairs.append((rr, cc))
    return pairs


def track_dataset(label: str, coco_rle_dir: Path) -> pd.DataFrame:
    coco_files = sorted(coco_rle_dir.glob("*.json"))
    if not coco_files:
        raise RuntimeError(f"No JSONs found in {coco_rle_dir}")

    coco_files = sorted(coco_files, key=lambda p: time_ms_from_stem(p.stem))

    tracks_done = []
    active = []
    next_id = 1

    # quick stats
    det_counts = []

    for jf in tqdm(coco_files, desc=f"Tracking {label}", unit="frame"):
        t = time_ms_from_stem(jf.stem)
        dets = load_detections_cells(jf)
        det_counts.append(len(dets))

        # init
        if len(active) == 0:
            for d in dets:
                tr = Track(next_id, t)
                next_id += 1
                tr.add(t, d)
                active.append(tr)
            continue

        # match
        pairs = hungarian_match(active, dets)
        assigned_tracks = set()
        assigned_dets = set()

        for ti, di in pairs:
            active[ti].add(t, dets[di])
            assigned_tracks.add(ti)
            assigned_dets.add(di)

        # misses
        new_active = []
        for i, tr in enumerate(active):
            if i not in assigned_tracks:
                tr.miss()
            if tr.misses <= MAX_MISSES:
                new_active.append(tr)
            else:
                tracks_done.append(tr)
        active = new_active

        # births
        for j, d in enumerate(dets):
            if j in assigned_dets:
                continue
            tr = Track(next_id, t)
            next_id += 1
            tr.add(t, d)
            active.append(tr)

    tracks_done.extend(active)

    rows = []
    for tr in tracks_done:
        rows.extend(tr.records)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No tracks produced for {label}")

    df["dataset"] = label
    df = df.sort_values(["track_id", "time_ms"]).reset_index(drop=True)

    # add nucleation time, tau
    df["t_nuc_ms"] = df.groupby("track_id")["time_ms"].transform("min")
    df["tau_ms"] = df["time_ms"] - df["t_nuc_ms"]

    # Smooth R and compute v
    df["R_smooth_px"] = np.nan
    df["v_px_per_ms"] = np.nan

    for tid, g in df.groupby("track_id"):
        t = g["time_ms"].values
        R = g["R_px"].values
        R_s = smooth_series(R)
        v = compute_derivative(t, R_s)
        df.loc[g.index, "R_smooth_px"] = R_s
        df.loc[g.index, "v_px_per_ms"] = v

    df["v_px_per_s"] = df["v_px_per_ms"] * 1000.0

    # filter short tracks (fragment removal)
    sizes = df.groupby("track_id").size()
    keep_ids = sizes[sizes >= MIN_TRACK_LEN_FRAMES].index
    df = df[df["track_id"].isin(keep_ids)].copy()
    df.reset_index(drop=True, inplace=True)

    # save detection count stats
    stats = {
        "dataset": label,
        "n_frames": len(coco_files),
        "detections_min": int(np.min(det_counts)),
        "detections_median": float(np.median(det_counts)),
        "detections_max": int(np.max(det_counts)),
        "n_tracks_after_len_filter": int(df["track_id"].nunique()),
    }
    pd.DataFrame([stats]).to_csv(OUT_DIR / f"{label}_detcount_stats.csv", index=False)

    return df


def qc_plots(label: str, df: pd.DataFrame):
    # 1) sample trajectories R_i(t)
    plt.figure()
    shown = 0
    for tid, g in df.groupby("track_id"):
        if len(g) < MIN_TRACK_LEN_FRAMES:
            continue
        plt.plot(g["time_ms"], g["R_smooth_px"], alpha=0.35)
        shown += 1
        if shown >= 30:
            break
    plt.xlabel("Time (ms)")
    plt.ylabel("R_smooth (px)")
    plt.title(f"{label}: sample tracked spherulite trajectories")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{label}_tracks_Ri.png", dpi=200)
    plt.close()

    # 2) nucleation times histogram (track births)
    tb = df[["track_id", "t_nuc_ms"]].drop_duplicates().sort_values("t_nuc_ms")
    plt.figure()
    plt.hist(tb["t_nuc_ms"], bins=30)
    plt.xlabel("t_nuc (ms)")
    plt.ylabel("Count")
    plt.title(f"{label}: nucleation times (track births)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{label}_t_nuc_hist.png", dpi=200)
    plt.close()

    # 3) cohort growth: median R(τ)
    bin_ms = 5.0
    d2 = df.copy()
    d2["tau_bin_ms"] = (d2["tau_ms"] // bin_ms) * bin_ms
    cohort = d2.groupby("tau_bin_ms")["R_smooth_px"].median().reset_index()

    plt.figure()
    plt.plot(cohort["tau_bin_ms"], cohort["R_smooth_px"])
    plt.xlabel("τ = t - t_nuc (ms)")
    plt.ylabel("Median R(τ) (px)")
    plt.title(f"{label}: cohort growth (aligned by nucleation)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{label}_cohort_Rtau.png", dpi=200)
    plt.close()

    # 4) mean growth ⟨R(t)⟩ (population mean; can drop with new births)
    mean_t = df.groupby("time_ms")["R_smooth_px"].mean().reset_index()
    plt.figure()
    plt.plot(mean_t["time_ms"], mean_t["R_smooth_px"])
    plt.xlabel("Time (ms)")
    plt.ylabel("Mean R(t) (px)")
    plt.title(f"{label}: population mean ⟨R(t)⟩ (can drop with new births)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{label}_Rmean.png", dpi=200)
    plt.close()


def main():
    all_df = []
    for label, coco_dir in DATASETS.items():
        print(f"\n=== {label} ===")
        print("COCO RLE:", coco_dir)
        df = track_dataset(label, coco_dir)

        out_csv = OUT_DIR / f"{label}_tracks_cells.csv"
        df.to_csv(out_csv, index=False)
        print("Saved:", out_csv)

        nuc = df[["track_id", "t_nuc_ms"]].drop_duplicates().sort_values("t_nuc_ms")
        nuc.to_csv(OUT_DIR / f"{label}_nucleation_times_cells.csv", index=False)

        qc_plots(label, df)
        all_df.append(df)

    # combined summary
    comb = pd.concat(all_df, ignore_index=True)
    summary = []
    for label, g in comb.groupby("dataset"):
        tb = g[["track_id", "t_nuc_ms"]].drop_duplicates()
        summary.append({
            "dataset": label,
            "n_tracks": int(tb.shape[0]),
            "t_first_nuc_ms": float(tb["t_nuc_ms"].min()),
            "t_last_nuc_ms": float(tb["t_nuc_ms"].max()),
            "R_median_px": float(g["R_smooth_px"].median()),
            "R_p90_px": float(g["R_smooth_px"].quantile(0.90)),
            "v_median_px_per_s": float(g["v_px_per_s"].median(skipna=True)),
            "v_p90_px_per_s": float(g["v_px_per_s"].quantile(0.90)),
        })
    pd.DataFrame(summary).to_csv(OUT_DIR / "tracking_summary.csv", index=False)
    print("\nDone. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
