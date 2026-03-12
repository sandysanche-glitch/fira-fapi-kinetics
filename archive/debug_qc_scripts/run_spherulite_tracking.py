# run_spherulite_tracking.py
import os
import json
import math
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

from pycocotools import mask as mask_utils


# ----------------------------
# Helpers
# ----------------------------
def rle_to_mask(rle):
    """Decode COCO RLE to a uint8 mask."""
    m = mask_utils.decode(rle)
    # m can be HxW or HxWx1
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(np.uint8)


def mask_area(rle):
    return float(mask_utils.area(rle))


def mask_bbox(rle):
    # returns [x, y, w, h]
    bb = mask_utils.toBbox(rle).astype(float)
    return bb


def mask_centroid_from_mask(m):
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def effective_radius_from_area(area_px):
    return float(math.sqrt(area_px / math.pi))


def iou_rle(rle_a, rle_b):
    # pycocotools expects list
    iou = mask_utils.iou([rle_a], [rle_b], [0])[0][0]
    return float(iou)


def touches_border(bbox_xywh, W, H, margin=2):
    x, y, w, h = bbox_xywh
    x2 = x + w
    y2 = y + h
    return (x <= margin) or (y <= margin) or (x2 >= (W - margin)) or (y2 >= (H - margin))


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class Det:
    det_id: int
    rle: dict
    area: float
    bbox: np.ndarray  # [x,y,w,h]
    cx: float
    cy: float
    R: float


@dataclass
class TrackState:
    track_id: int
    last_det: Det
    last_frame: int
    missed: int = 0


# ----------------------------
# Core tracking
# ----------------------------
def build_detections_from_json(json_path, max_area_px=None, min_area_px=100,
                              max_area_frac=0.05, border_reject=True, border_margin=2):
    """
    Read one SAM json and return filtered detections.
    We infer H,W from the RLE size.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    ann = data.get("annotations", data)  # sometimes file is a list, sometimes dict
    if isinstance(ann, dict) and "annotations" in ann:
        ann = ann["annotations"]

    dets = []
    det_id = 0

    # infer image size from first RLE
    H = W = None
    for a in ann:
        seg = a.get("segmentation", None)
        if isinstance(seg, dict) and "size" in seg:
            H, W = seg["size"]
            break
    if H is None or W is None:
        # fallback: decode one mask
        for a in ann:
            seg = a.get("segmentation", None)
            if isinstance(seg, dict):
                m = rle_to_mask(seg)
                H, W = m.shape
                break

    img_area = float(H * W)
    if max_area_px is None:
        max_area_px = max_area_frac * img_area

    for a in ann:
        seg = a.get("segmentation", None)
        if not isinstance(seg, dict):
            continue

        area = mask_area(seg)
        if area < min_area_px or area > max_area_px:
            continue

        bb = mask_bbox(seg)

        if border_reject and touches_border(bb, W, H, margin=border_margin):
            continue

        # centroid needs decode (fast enough after filtering)
        m = rle_to_mask(seg)
        c = mask_centroid_from_mask(m)
        if c is None:
            continue
        cx, cy = c

        R = effective_radius_from_area(area)

        dets.append(Det(
            det_id=det_id,
            rle=seg,
            area=area,
            bbox=bb,
            cx=cx,
            cy=cy,
            R=R
        ))
        det_id += 1

    return dets, (H, W)


def bbox_center(bb):
    x, y, w, h = bb
    return np.array([x + 0.5*w, y + 0.5*h], dtype=float)


def centroid_dist(detA: Det, detB: Det):
    dx = detA.cx - detB.cx
    dy = detA.cy - detB.cy
    return float(math.sqrt(dx*dx + dy*dy))


def track_dataset(
    dataset_name: str,
    sam_root: Path,
    frame_dt_ms: float = 2.0,
    max_gap: int = 3,
    # filtering
    min_area_px: float = 150.0,
    max_area_frac: float = 0.03,   # <= 3% of image area (kills “whole-image” masks)
    border_reject: bool = True,
    border_margin: int = 2,
    # matching
    max_dist_px: float = 60.0,
    min_iou: float = 0.05,
    w_dist: float = 1.0,
    w_iou: float = 2.0,
    max_cost: float = 2.5,
):
    coco_dir = sam_root / dataset_name / "coco_rle"
    if not coco_dir.exists():
        raise FileNotFoundError(f"Missing coco_rle folder: {coco_dir}")

    json_files = sorted(coco_dir.glob("*.json"))
    if len(json_files) == 0:
        raise RuntimeError(f"No JSON files in {coco_dir}")

    out_dir = sam_root / dataset_name
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    tracks_rows = []

    active = []
    next_track_id = 1

    H = W = None

    for frame_idx, jp in enumerate(tqdm(json_files, desc=f"Tracking {dataset_name}", unit="frame")):
        t_ms = frame_idx * frame_dt_ms

        dets, (H, W) = build_detections_from_json(
            jp,
            min_area_px=min_area_px,
            max_area_frac=max_area_frac,
            border_reject=border_reject,
            border_margin=border_margin,
        )

        # --- Build cost matrix between active tracks and current detections
        if len(active) > 0 and len(dets) > 0:
            cost = np.full((len(active), len(dets)), fill_value=1e9, dtype=float)

            for i, tr in enumerate(active):
                for j, d in enumerate(dets):
                    dist = centroid_dist(tr.last_det, d)
                    if dist > max_dist_px:
                        continue

                    # Quick bbox overlap gate (avoid IoU calc when impossible)
                    # (optional but cheap)
                    bb1 = tr.last_det.bbox
                    bb2 = d.bbox
                    x1, y1, w1, h1 = bb1
                    x2, y2, w2, h2 = bb2
                    xa = max(x1, x2)
                    ya = max(y1, y2)
                    xb = min(x1+w1, x2+w2)
                    yb = min(y1+h1, y2+h2)
                    if (xb - xa) <= 0 or (yb - ya) <= 0:
                        # no bbox intersection -> IoU is basically 0
                        iou = 0.0
                    else:
                        iou = iou_rle(tr.last_det.rle, d.rle)

                    if iou < min_iou:
                        continue

                    dist_norm = dist / (math.sqrt(H*H + W*W) + 1e-9)
                    c = w_dist * dist_norm + w_iou * (1.0 - iou)
                    cost[i, j] = c

            row_ind, col_ind = linear_sum_assignment(cost)

            assigned_tracks = set()
            assigned_dets = set()

            # accept matches under max_cost
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= max_cost:
                    continue
                tr = active[r]
                d = dets[c]

                assigned_tracks.add(r)
                assigned_dets.add(c)

                tr.last_det = d
                tr.last_frame = frame_idx
                tr.missed = 0

                tracks_rows.append({
                    "dataset": dataset_name,
                    "track_id": tr.track_id,
                    "frame_idx": frame_idx,
                    "t_ms": t_ms,
                    "cx": d.cx,
                    "cy": d.cy,
                    "area_px": d.area,
                    "R_px": d.R,
                })

            # tracks not assigned -> missed++
            for i, tr in enumerate(active):
                if i not in assigned_tracks:
                    tr.missed += 1

            # remove dead tracks
            active = [tr for tr in active if tr.missed <= max_gap]

            # unmatched dets -> new tracks
            for j, d in enumerate(dets):
                if j in assigned_dets:
                    continue
                tr = TrackState(track_id=next_track_id, last_det=d, last_frame=frame_idx, missed=0)
                next_track_id += 1
                active.append(tr)

                tracks_rows.append({
                    "dataset": dataset_name,
                    "track_id": tr.track_id,
                    "frame_idx": frame_idx,
                    "t_ms": t_ms,
                    "cx": d.cx,
                    "cy": d.cy,
                    "area_px": d.area,
                    "R_px": d.R,
                })

        else:
            # no matching possible
            # all dets become new tracks; all active tracks get missed
            for tr in active:
                tr.missed += 1
            active = [tr for tr in active if tr.missed <= max_gap]

            for d in dets:
                tr = TrackState(track_id=next_track_id, last_det=d, last_frame=frame_idx, missed=0)
                next_track_id += 1
                active.append(tr)

                tracks_rows.append({
                    "dataset": dataset_name,
                    "track_id": tr.track_id,
                    "frame_idx": frame_idx,
                    "t_ms": t_ms,
                    "cx": d.cx,
                    "cy": d.cy,
                    "area_px": d.area,
                    "R_px": d.R,
                })

    tracks_df = pd.DataFrame(tracks_rows)
    tracks_csv = out_dir / "tracks.csv"
    tracks_df.to_csv(tracks_csv, index=False)

    # Track summary
    g = tracks_df.sort_values(["track_id", "t_ms"]).groupby("track_id")
    summary = g.agg(
        t_birth_ms=("t_ms", "min"),
        t_death_ms=("t_ms", "max"),
        n_points=("t_ms", "size"),
        R_birth_px=("R_px", "first"),
        R_last_px=("R_px", "last"),
    ).reset_index()

    # mean growth speed per track (simple slope from endpoints)
    dt_s = (summary.t_death_ms - summary.t_birth_ms) / 1000.0
    dR = summary.R_last_px - summary.R_birth_px
    summary["v_mean_px_per_s"] = np.where(dt_s > 0, dR / dt_s, np.nan)

    summary_csv = out_dir / "track_summary.csv"
    summary.to_csv(summary_csv, index=False)

    return tracks_df, summary, plots_dir


# ----------------------------
# Plotting
# ----------------------------
def make_plots(dataset_name, tracks_df, summary_df, plots_dir,
               min_track_len=30, exclude_R_gt=500.0):
    import matplotlib.pyplot as plt

    # Filter: remove absurd radii tracks and short tracks for “physics plots”
    good_tracks = summary_df[
        (summary_df.n_points >= min_track_len) &
        (summary_df.R_last_px <= exclude_R_gt) &
        (summary_df.R_birth_px <= exclude_R_gt)
    ].copy()
    good_ids = set(good_tracks.track_id.tolist())
    T = tracks_df[tracks_df.track_id.isin(good_ids)].copy()

    # 1) Population mean R(t)
    pop_mean = T.groupby("t_ms")["R_px"].mean()
    pop_median = T.groupby("t_ms")["R_px"].median()

    plt.figure()
    plt.plot(pop_mean.index, pop_mean.values)
    plt.title(f"{dataset_name}: population mean <R(t)> (filtered)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Mean R(t) (px)")
    plt.tight_layout()
    plt.savefig(plots_dir / f"{dataset_name}_Rmean.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(pop_median.index, pop_median.values)
    plt.title(f"{dataset_name}: population median R(t) (filtered)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Median R(t) (px)")
    plt.tight_layout()
    plt.savefig(plots_dir / f"{dataset_name}_Rmedian.png", dpi=150)
    plt.close()

    # 2) Nucleation times histogram (track births)
    plt.figure()
    plt.hist(good_tracks.t_birth_ms.values, bins=30)
    plt.title(f"{dataset_name}: nucleation times (track births)")
    plt.xlabel("t_nuc (ms)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(plots_dir / f"{dataset_name}_t_nuc_hist.png", dpi=150)
    plt.close()

    # 3) Sample trajectories
    plt.figure()
    sample_ids = list(good_ids)[:25]
    for tid in sample_ids:
        tt = T[T.track_id == tid].sort_values("t_ms")
        plt.plot(tt.t_ms.values, tt.R_px.values, alpha=0.8)
    plt.title(f"{dataset_name}: sample tracked spherulite trajectories")
    plt.xlabel("Time (ms)")
    plt.ylabel("R(t) (px)")
    plt.tight_layout()
    plt.savefig(plots_dir / f"{dataset_name}_tracks_Ri.png", dpi=150)
    plt.close()

    # 4) Cohort growth aligned by nucleation: tau = t - t_birth
    # Build tau grid at frame times
    rows = []
    for tid, grp in T.sort_values(["track_id", "t_ms"]).groupby("track_id"):
        tb = grp.t_ms.iloc[0]
        tau = grp.t_ms.values - tb
        for a, r in zip(tau, grp.R_px.values):
            rows.append((a, r))
    coh = pd.DataFrame(rows, columns=["tau_ms", "R_px"])
    coh_med = coh.groupby("tau_ms")["R_px"].median()
    coh_p25 = coh.groupby("tau_ms")["R_px"].quantile(0.25)
    coh_p75 = coh.groupby("tau_ms")["R_px"].quantile(0.75)
    coh_n = coh.groupby("tau_ms")["R_px"].size()

    plt.figure()
    plt.plot(coh_med.index, coh_med.values)
    plt.title(f"{dataset_name}: cohort growth (aligned by nucleation)")
    plt.xlabel("τ = t - t_nuc (ms)")
    plt.ylabel("Median R(τ) (px)")
    plt.tight_layout()
    plt.savefig(plots_dir / f"{dataset_name}_cohort_Rtau.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(coh_n.index, coh_n.values)
    plt.title(f"{dataset_name}: cohort contributors N(τ)")
    plt.xlabel("τ = t - t_nuc (ms)")
    plt.ylabel("Number of samples")
    plt.tight_layout()
    plt.savefig(plots_dir / f"{dataset_name}_cohort_Ntau.png", dpi=150)
    plt.close()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Project root (contains sam/)")
    ap.add_argument("--datasets", type=str, default="FAPI,FAPI_TEMPO", help="Comma list")
    ap.add_argument("--dt_ms", type=float, default=2.0, help="Frame timestep in ms")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    sam_root = root / "sam"

    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        print(f"\n=== Tracking {ds} ===")
        tracks_df, summary_df, plots_dir = track_dataset(
            dataset_name=ds,
            sam_root=sam_root,
            frame_dt_ms=args.dt_ms,
            # IMPORTANT filters to prevent “whole image” mask
            max_area_frac=0.03,
            min_area_px=150,
            border_reject=True,
            # matching
            max_dist_px=60.0,
            min_iou=0.05,
            w_dist=1.0,
            w_iou=2.0,
            max_cost=2.5,
            max_gap=3,
        )
        print(f"Saved: {sam_root/ds/'tracks.csv'}")
        print(f"Saved: {sam_root/ds/'track_summary.csv'}")

        make_plots(ds, tracks_df, summary_df, plots_dir,
                   min_track_len=30, exclude_R_gt=500.0)
        print(f"Plots in: {plots_dir}")


if __name__ == "__main__":
    main()
