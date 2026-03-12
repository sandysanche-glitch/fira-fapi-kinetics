import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment
from pycocotools import mask as mask_util


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
AREA_MIN_PX = 500           # remove tiny fragments
IOU_MIN = 0.05              # allow weak overlap early
DIST_MAX_PX = 180.0         # max centroid jump between frames
MAX_MISSES = 2              # allow short disappearances
LAMBDA_DIST = 0.002         # weight centroid distance in cost

# Smoothing for derivatives
SMOOTH_WINDOW = 9           # must be odd
SMOOTH_POLY = 2


TIME_RE = re.compile(r"_t([0-9]+(?:\.[0-9]+)?)ms", re.IGNORECASE)


def time_ms_from_stem(stem: str) -> float:
    m = TIME_RE.search(stem)
    if not m:
        raise ValueError(f"Cannot parse time from filename stem: {stem}")
    return float(m.group(1))


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


def dist(c1, c2) -> float:
    return float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))


def decode_rle(rle):
    # mask_util.decode returns HxWx1 sometimes; squeeze it
    m = mask_util.decode(rle)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(bool)


def load_detections_cells(json_path: Path):
    """
    Load SAM detections from one frame.
    We treat "cells" as the large components (area >= AREA_MIN_PX).
    """
    with open(json_path, "r") as f:
        anns = json.load(f)

    dets = []
    for a in anns:
        seg = a.get("segmentation", None)
        if seg is None:
            continue

        mask = decode_rle(seg)
        area = int(mask.sum())
        if area < AREA_MIN_PX:
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
            "time_ms": t,
            "area_px": det["area_px"],
            "cx": det["cx"],
            "cy": det["cy"],
        })
        self.last_centroid = (det["cx"], det["cy"])
        self.last_bbox = det["bbox"]
        self.misses = 0

    def miss(self):
        self.misses += 1


def hungarian_match(active_tracks, detections):
    """
    Return list of (track_index, det_index) assignments.
    """
    nT = len(active_tracks)
    nD = len(detections)
    if nT == 0 or nD == 0:
        return []

    BIG = 1e6
    cost = np.full((nT, nD), BIG, dtype=float)

    for i, tr in enumerate(active_tracks):
        for j, d in enumerate(detections):
            # distance gate
            dj = dist(tr.last_centroid, (d["cx"], d["cy"])) if tr.last_centroid else 0.0
            if dj > DIST_MAX_PX:
                continue

            # overlap gate (bbox IoU proxy)
            iou = bbox_iou(tr.last_bbox, d["bbox"]) if tr.last_bbox else 0.0
            if iou < IOU_MIN and dj > (0.5 * DIST_MAX_PX):
                continue

            # cost combines overlap + distance
            cost[i, j] = (1.0 - iou) + LAMBDA_DIST * dj

    r, c = linear_sum_assignment(cost)

    pairs = []
    for rr, cc in zip(r, c):
        if cost[rr, cc] >= BIG * 0.5:
            continue
        pairs.append((rr, cc))
    return pairs


def smooth_series(y: np.ndarray):
    """
    Savitzky–Golay smoothing if possible; otherwise return y unchanged.
    """
    if len(y) < SMOOTH_WINDOW:
        return y
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(y, window_length=SMOOTH_WINDOW, polyorder=SMOOTH_POLY)
    except Exception:
        return y


def compute_derivative(t, y):
    """
    Central difference derivative dy/dt.
    """
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


def track_cells_dataset(label: str, coco_rle_dir: Path) -> pd.DataFrame:
    coco_files = sorted(coco_rle_dir.glob("*.json"))
    if not coco_files:
        raise RuntimeError(f"No JSONs in {coco_rle_dir}")

    coco_files = sorted(coco_files, key=lambda p: time_ms_from_stem(p.stem))

    tracks_done = []
    active = []
    next_id = 1

    for jf in coco_files:
        t = time_ms_from_stem(jf.stem)
        dets = load_detections_cells(jf)

        # init if nothing active
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

        # handle misses
        new_active = []
        for i, tr in enumerate(active):
            if i not in assigned_tracks:
                tr.miss()
            if tr.misses <= MAX_MISSES:
                new_active.append(tr)
            else:
                tracks_done.append(tr)
        active = new_active

        # create new tracks for unmatched detections (nucleation)
        for j, d in enumerate(dets):
            if j in assigned_dets:
                continue
            tr = Track(next_id, t)
            next_id += 1
            tr.add(t, d)
            active.append(tr)

    tracks_done.extend(active)

    # dataframe
    rows = []
    for tr in tracks_done:
        rows.extend(tr.records)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No tracks produced for {label}")

    df = df.sort_values(["track_id", "time_ms"]).reset_index(drop=True)

    # compute R, smoothed R, v
    df["R_px"] = np.sqrt(df["area_px"] / np.pi)

    # add nucleation time (birth)
    df["t_nuc_ms"] = df.groupby("track_id")["time_ms"].transform("min")

    # per-track smoothed derivatives
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
    return df


def qc_plots(label: str, df: pd.DataFrame):
    # plot sample trajectories
    plt.figure()
    n = 0
    for tid, g in df.groupby("track_id"):
        if len(g) < 6:
            continue
        plt.plot(g["time_ms"], g["R_smooth_px"], alpha=0.35)
        n += 1
        if n >= 30:
            break
    plt.xlabel("Time (ms)")
    plt.ylabel("R_smooth (px)")
    plt.title(f"{label}: sample tracked spherulite trajectories")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{label}_tracks_Ri.png", dpi=200)
    plt.close()

    # mean growth
    mean = df.groupby("time_ms").agg(R_mean=("R_smooth_px", "mean")).reset_index()
    plt.figure()
    plt.plot(mean["time_ms"], mean["R_mean"])
    plt.xlabel("Time (ms)")
    plt.ylabel("Mean R (px)")
    plt.title(f"{label}: mean growth ⟨R(t)⟩")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{label}_Rmean.png", dpi=200)
    plt.close()

    # nucleation time histogram
    tb = df[["track_id", "t_nuc_ms"]].drop_duplicates()
    plt.figure()
    plt.hist(tb["t_nuc_ms"], bins=20)
    plt.xlabel("t_nuc (ms)")
    plt.ylabel("Count")
    plt.title(f"{label}: nucleation times (track births)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{label}_t_nuc_hist.png", dpi=200)
    plt.close()


# ==================================================
# RUN
# ==================================================
summaries = []

for label, cfg in DATASETS.items():
    print(f"\n=== Tracking cells for {label} ===")
    df = track_cells_dataset(label, cfg["coco_rle"])

    out_csv = OUT_DIR / f"{label}_tracks_cells.csv"
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

    # nucleation times table
    nuc = df[["track_id", "t_nuc_ms"]].drop_duplicates().sort_values("t_nuc_ms")
    nuc.to_csv(OUT_DIR / f"{label}_nucleation_times_cells.csv", index=False)

    qc_plots(label, df)

    summaries.append({
        "dataset": label,
        "n_tracks": int(nuc.shape[0]),
        "t_first_nuc_ms": float(nuc["t_nuc_ms"].min()),
        "t_last_nuc_ms": float(nuc["t_nuc_ms"].max()),
        "R_max_px": float(df["R_smooth_px"].max()),
        "v_median_px_per_s": float(df["v_px_per_s"].median(skipna=True)),
        "v_p90_px_per_s": float(df["v_px_per_s"].quantile(0.9)),
    })

summary_df = pd.DataFrame(summaries)
summary_df.to_csv(OUT_DIR / "tracking_cells_summary.csv", index=False)
print("\nDone. Outputs in:", OUT_DIR)
