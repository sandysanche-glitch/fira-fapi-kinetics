#!/usr/bin/env python3
"""
Time-resolved crystal segmentation + simple tracking from per-frame SAM JSONs.

Expected folder layout (typical):
  frames_root/
    FAPI/
      frame_00000_t0.00ms.png
      frame_00001_t2.00ms.png
      ...
    FAPI_TEMPO/
      ...

  sam_root/
    FAPI/
      coco_rle/
        frame_00000_t0.00ms.json     (common for SAM-exported masks)
        frame_00001_t2.00ms.json
        ...
    FAPI_TEMPO/
      coco_rle/
        ...

This script:
  - reads per-frame detections (bbox, area) from SAM json
  - filters by min/max area and max radius
  - tracks objects frame-to-frame with a greedy assignment (distance + IoU + ΔR gating)
  - optional monotonic enforcement per-track (cumulative max of R)
  - exports:
      out_root/<dataset>/tracks.csv
      out_root/<dataset>/track_summary.csv
      out_root/<dataset>/tracks_Ri.png
      out_root/<dataset>/Rmean.png
      out_root/<dataset>/t_nuc_hist.png
      out_root/<dataset>/cohort_Rtau.png

Run example (PowerShell/cmd; use full paths, NOT "..."):
  python run_time_resolved_crystal_segmentation_fast.py ^
    --frames_root "F:\\...\\frames" ^
    --sam_root "F:\\...\\sam" ^
    --datasets "FAPI,FAPI_TEMPO" ^
    --out_root "F:\\...\\out" ^
    --skip_sam ^
    --min_area_px 300 ^
    --max_area_frac 0.25 ^
    --max_R_px 800 ^
    --enforce_monotonic ^
    --min_cohort_count 8 ^
    --max_match_dist_px 80 ^
    --iou_weight 0.3 ^
    --max_delta_R_px 80
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Detection:
    det_id: int
    bbox: Tuple[float, float, float, float]  # x, y, w, h
    area_px: float
    score: float
    cx: float
    cy: float
    R_px: float


@dataclass
class TrackState:
    track_id: int
    last_t_ms: float
    last_frame_idx: int
    last_bbox: Tuple[float, float, float, float]
    last_cx: float
    last_cy: float
    last_R_px: float
    gap: int = 0  # how many consecutive frames not matched


# -----------------------------
# Helpers
# -----------------------------

_TIME_RE = re.compile(r"_t([0-9]+(?:\.[0-9]+)?)ms", re.IGNORECASE)

def parse_time_ms_from_name(name: str) -> Optional[float]:
    m = _TIME_RE.search(name)
    if not m:
        return None
    return float(m.group(1))


def bbox_center(b: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x, y, w, h = b
    return (x + 0.5 * w, y + 0.5 * h)


def bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (aw * ah) + (bw * bh) - inter
    return float(inter / union) if union > 0 else 0.0


def safe_mkdir(p: Path) -> None:
    # Windows cannot create "..." directories (trailing dots are special)
    s = str(p)
    if "...\\" in s or s.endswith("...") or s.endswith("...\\") or s.startswith("..."):
        raise ValueError(
            f'out_root looks like a placeholder: "{p}". '
            "Please pass a real path (Windows cannot create '...' folders)."
        )
    p.mkdir(parents=True, exist_ok=True)


def split_datasets_arg(s: str) -> List[str]:
    # Accept: "FAPI,FAPI_TEMPO" or "FAPI FAPI_TEMPO" or "FAPI, FAPI_TEMPO"
    s = s.strip().strip('"').strip("'")
    parts = re.split(r"[,\s]+", s)
    return [p for p in (x.strip() for x in parts) if p]


# -----------------------------
# SAM JSON loading
# -----------------------------

def load_annotation_list(json_path: Path) -> List[dict]:
    """
    Supports:
      - COCO style dict: { "annotations": [...] }
      - Direct list of annotations: [ {...}, {...} ]   (your SAM coco_rle exports)
    """
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        anns = data.get("annotations", [])
        if not isinstance(anns, list):
            raise ValueError(f"{json_path}: 'annotations' exists but is not a list.")
        return anns

    if isinstance(data, list):
        return data

    raise ValueError(f"{json_path}: unsupported JSON root type: {type(data)}")


def find_json_for_frame(img_path: Path, sam_dataset_dir: Path) -> Optional[Path]:
    """
    Tries common patterns:
      sam/<ds>/<name>.json
      sam/<ds>/coco_rle/<name>.json
      sam/<ds>/<name>.png.json
      sam/<ds>/coco_rle/<name>.png.json
    """
    stem = img_path.stem  # frame_00000_t0.00ms
    cands = [
        sam_dataset_dir / f"{stem}.json",
        sam_dataset_dir / "coco_rle" / f"{stem}.json",
        sam_dataset_dir / f"{img_path.name}.json",
        sam_dataset_dir / "coco_rle" / f"{img_path.name}.json",
    ]
    for c in cands:
        if c.exists():
            return c

    # As a last resort, search a bit deeper but keep it constrained
    for sub in ["coco_rle", "coco", "json", "pred", "preds"]:
        p = sam_dataset_dir / sub / f"{stem}.json"
        if p.exists():
            return p

    return None


def parse_frame_dets(
    json_path: Path,
    frame_idx: int,
    min_area_px: float,
    max_area_px: Optional[float],
    max_R_px: Optional[float],
) -> Tuple[List[Detection], Optional[Tuple[int, int]]]:
    """
    Returns (detections, (H,W) if inferable).
    """
    anns = load_annotation_list(json_path)

    dets: List[Detection] = []
    hw: Optional[Tuple[int, int]] = None

    for i, ann in enumerate(anns):
        if not isinstance(ann, dict):
            continue

        bbox = ann.get("bbox", None)
        if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue

        x, y, w, h = map(float, bbox)
        if w <= 0 or h <= 0:
            continue

        area = ann.get("area", None)
        if area is None:
            area = w * h
        area = float(area)

        score = float(ann.get("score", 1.0))

        # infer H,W from segmentation if present
        seg = ann.get("segmentation", None)
        if hw is None and isinstance(seg, dict):
            size = seg.get("size", None)
            if isinstance(size, (list, tuple)) and len(size) == 2:
                H, W = int(size[0]), int(size[1])
                if H > 0 and W > 0:
                    hw = (H, W)

        if area < float(min_area_px):
            continue
        if max_area_px is not None and area > float(max_area_px):
            continue

        R = math.sqrt(area / math.pi)

        if max_R_px is not None and R > float(max_R_px):
            continue

        cx, cy = bbox_center((x, y, w, h))

        dets.append(
            Detection(
                det_id=i,
                bbox=(x, y, w, h),
                area_px=area,
                score=score,
                cx=cx,
                cy=cy,
                R_px=float(R),
            )
        )

    return dets, hw


# -----------------------------
# Tracking
# -----------------------------

def greedy_match(
    tracks: List[TrackState],
    dets: List[Detection],
    max_match_dist_px: float,
    iou_weight: float,
    max_delta_R_px: Optional[float],
) -> Tuple[Dict[int, int], List[int], List[int]]:
    """
    Returns:
      match_t2d: track_index -> det_index
      unmatched_track_idxs
      unmatched_det_idxs
    """
    if not tracks or not dets:
        return {}, list(range(len(tracks))), list(range(len(dets)))

    candidates: List[Tuple[float, int, int]] = []
    for ti, tr in enumerate(tracks):
        for di, d in enumerate(dets):
            dx = d.cx - tr.last_cx
            dy = d.cy - tr.last_cy
            dist = math.hypot(dx, dy)
            if dist > max_match_dist_px:
                continue

            if max_delta_R_px is not None:
                if abs(d.R_px - tr.last_R_px) > max_delta_R_px:
                    continue

            iou = bbox_iou(tr.last_bbox, d.bbox)

            # cost: lower is better
            # normalize dist to [0,1] by max_match_dist
            dist_term = dist / max_match_dist_px
            iou_term = (1.0 - iou)  # 0 is best

            # blend
            cost = (1.0 - iou_weight) * dist_term + iou_weight * iou_term
            candidates.append((cost, ti, di))

    candidates.sort(key=lambda x: x[0])

    matched_tracks = set()
    matched_dets = set()
    match_t2d: Dict[int, int] = {}

    for cost, ti, di in candidates:
        if ti in matched_tracks or di in matched_dets:
            continue
        matched_tracks.add(ti)
        matched_dets.add(di)
        match_t2d[ti] = di

    unmatched_track_idxs = [i for i in range(len(tracks)) if i not in matched_tracks]
    unmatched_det_idxs = [i for i in range(len(dets)) if i not in matched_dets]
    return match_t2d, unmatched_track_idxs, unmatched_det_idxs


# -----------------------------
# Analysis + plotting
# -----------------------------

def enforce_monotonic_per_track(df_tracks: pd.DataFrame) -> pd.DataFrame:
    df = df_tracks.copy()
    df.sort_values(["dataset", "track_id", "t_ms"], inplace=True)
    df["R_mono"] = df.groupby(["dataset", "track_id"])["R_px"].cummax()
    return df


def plot_tracks(df_tracks: pd.DataFrame, dataset: str, out_png: Path, n_show: int = 25, seed: int = 0) -> None:
    rng = random.Random(seed)
    track_ids = sorted(df_tracks["track_id"].unique().tolist())
    if len(track_ids) > n_show:
        track_ids = rng.sample(track_ids, n_show)

    plt.figure(figsize=(10, 7))
    for tid in track_ids:
        sub = df_tracks[df_tracks["track_id"] == tid].sort_values("t_ms")
        plt.plot(sub["t_ms"].values, sub["R_mono"].values, linewidth=2)

    plt.title(f"{dataset}: sample tracked trajectories")
    plt.xlabel("Time (ms)")
    plt.ylabel("R_mono (px)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_mean_R(df_tracks: pd.DataFrame, dataset: str, out_png: Path) -> None:
    g = df_tracks.groupby("t_ms", as_index=False)["R_mono"].mean().sort_values("t_ms")

    plt.figure(figsize=(10, 6))
    plt.plot(g["t_ms"].values, g["R_mono"].values, linewidth=2)
    plt.title(f"{dataset}: population mean <R(t)> (mono)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Mean R(t) (px)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_tnuc_hist(track_summary: pd.DataFrame, dataset: str, out_png: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(track_summary["t_nuc_ms"].values, bins=25)
    plt.title(f"{dataset}: nucleation times")
    plt.xlabel("t_nuc (ms)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_cohort_Rtau(
    df_tracks: pd.DataFrame,
    track_summary: pd.DataFrame,
    dataset: str,
    out_png: Path,
    dt_ms: float,
    min_cohort_count: int,
) -> None:
    # align by nucleation time
    df = df_tracks.merge(track_summary[["track_id", "t_nuc_ms"]], on="track_id", how="left")
    df["tau_ms"] = df["t_ms"] - df["t_nuc_ms"]

    # bin to dt_ms grid (avoid float noise)
    df["tau_bin"] = (np.round(df["tau_ms"] / dt_ms) * dt_ms).astype(float)

    grp = df.groupby("tau_bin")["R_mono"]
    med = grp.median()
    cnt = grp.size()

    # keep only bins with enough tracks
    valid = cnt[cnt >= int(min_cohort_count)].index
    med = med.loc[valid].sort_index()
    cnt = cnt.loc[valid].sort_index()

    plt.figure(figsize=(11, 7))
    ax = plt.gca()
    ax.plot(med.index.values, med.values, linewidth=2)
    ax.set_title(f"{dataset}: cohort growth (aligned by nucleation)")
    ax.set_xlabel("τ = t - t_nuc (ms)")
    ax.set_ylabel("Median R(τ) (px)")

    ax2 = ax.twinx()
    ax2.plot(cnt.index.values, cnt.values, linestyle="--", linewidth=2)
    ax2.set_ylabel("Cohort count N(τ)")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# -----------------------------
# Main processing per dataset
# -----------------------------

def process_dataset(
    dataset: str,
    frames_root: Path,
    sam_root: Path,
    out_root: Path,
    skip_sam: bool,
    min_area_px: float,
    max_area_frac: float,
    max_R_px: float,
    enforce_monotonic: bool,
    min_cohort_count: int,
    max_match_dist_px: float,
    dt_ms: Optional[float],
    iou_weight: float,
    max_delta_R_px: Optional[float],
    max_gap_frames: int = 1,
    verbose: bool = True,
) -> None:
    frames_dir = frames_root / dataset
    sam_dir = sam_root / dataset

    if not frames_dir.exists():
        raise FileNotFoundError(f"{dataset}: frames dir not found: {frames_dir}")
    if not sam_dir.exists():
        raise FileNotFoundError(f"{dataset}: sam dir not found: {sam_dir}")

    img_paths = sorted(frames_dir.glob("*.png"))
    if not img_paths:
        raise FileNotFoundError(f"{dataset}: no .png frames found in {frames_dir}")

    # infer dt if not provided
    times = []
    for p in img_paths:
        t = parse_time_ms_from_name(p.name)
        if t is not None:
            times.append(t)
    inferred_dt = None
    if len(times) >= 2:
        times_sorted = np.sort(np.array(times, dtype=float))
        diffs = np.diff(times_sorted)
        diffs = diffs[diffs > 0]
        if len(diffs) > 0:
            inferred_dt = float(np.median(diffs))
    if dt_ms is None:
        dt_ms = inferred_dt if inferred_dt is not None else 1.0

    ds_out = out_root / dataset
    safe_mkdir(ds_out)

    tracks: List[TrackState] = []
    next_track_id = 0

    rows = []
    hw: Optional[Tuple[int, int]] = None

    for frame_idx, img_path in enumerate(img_paths):
        t_ms = parse_time_ms_from_name(img_path.name)
        if t_ms is None:
            t_ms = frame_idx * float(dt_ms)

        json_path = find_json_for_frame(img_path, sam_dir)
        if json_path is None:
            if skip_sam:
                raise FileNotFoundError(
                    f"{dataset}: Could not find json for frame {img_path.name}.\n"
                    f"Looked in: {sam_dir} and common subfolders (e.g. coco_rle).\n"
                    f"(--skip_sam set, so the script will not generate it.)"
                )
            else:
                raise RuntimeError(
                    f"{dataset}: SAM generation not implemented in this standalone script.\n"
                    f"Please generate per-frame jsons first, or run with --skip_sam."
                )

        dets, hw_frame = parse_frame_dets(
            json_path=json_path,
            frame_idx=frame_idx,
            min_area_px=min_area_px,
            max_area_px=(None if hw is None else max_area_frac * hw[0] * hw[1]),
            max_R_px=max_R_px,
        )
        if hw is None and hw_frame is not None:
            hw = hw_frame

        # if hw just became known, recompute max_area_px for this frame by re-filtering (optional)
        if hw is not None:
            max_area_px = max_area_frac * hw[0] * hw[1]
            dets = [d for d in dets if d.area_px <= max_area_px]

        if verbose and frame_idx % 50 == 0:
            print(f"{dataset}: frame {frame_idx:04d} t={t_ms:.2f}ms dets={len(dets)} tracks_active={len(tracks)}")

        # Match to active tracks
        match_t2d, unmatched_t, unmatched_d = greedy_match(
            tracks=tracks,
            dets=dets,
            max_match_dist_px=max_match_dist_px,
            iou_weight=iou_weight,
            max_delta_R_px=max_delta_R_px,
        )

        # Update matched tracks
        for ti, di in match_t2d.items():
            tr = tracks[ti]
            d = dets[di]
            tr.last_t_ms = float(t_ms)
            tr.last_frame_idx = frame_idx
            tr.last_bbox = d.bbox
            tr.last_cx = d.cx
            tr.last_cy = d.cy
            tr.last_R_px = d.R_px
            tr.gap = 0

            rows.append(
                dict(
                    dataset=dataset,
                    track_id=tr.track_id,
                    frame_idx=frame_idx,
                    t_ms=float(t_ms),
                    det_id=int(d.det_id),
                    cx=float(d.cx),
                    cy=float(d.cy),
                    bbox_x=float(d.bbox[0]),
                    bbox_y=float(d.bbox[1]),
                    bbox_w=float(d.bbox[2]),
                    bbox_h=float(d.bbox[3]),
                    area_px=float(d.area_px),
                    R_px=float(d.R_px),
                    score=float(d.score),
                )
            )

        # Increase gap for unmatched tracks; drop if too old
        kept_tracks: List[TrackState] = []
        for i, tr in enumerate(tracks):
            if i in match_t2d:
                kept_tracks.append(tr)
                continue
            tr.gap += 1
            if tr.gap <= max_gap_frames:
                kept_tracks.append(tr)

        tracks = kept_tracks

        # Create new tracks for unmatched detections
        for di in unmatched_d:
            d = dets[di]
            tr = TrackState(
                track_id=next_track_id,
                last_t_ms=float(t_ms),
                last_frame_idx=frame_idx,
                last_bbox=d.bbox,
                last_cx=d.cx,
                last_cy=d.cy,
                last_R_px=d.R_px,
                gap=0,
            )
            next_track_id += 1
            tracks.append(tr)

            rows.append(
                dict(
                    dataset=dataset,
                    track_id=tr.track_id,
                    frame_idx=frame_idx,
                    t_ms=float(t_ms),
                    det_id=int(d.det_id),
                    cx=float(d.cx),
                    cy=float(d.cy),
                    bbox_x=float(d.bbox[0]),
                    bbox_y=float(d.bbox[1]),
                    bbox_w=float(d.bbox[2]),
                    bbox_h=float(d.bbox[3]),
                    area_px=float(d.area_px),
                    R_px=float(d.R_px),
                    score=float(d.score),
                )
            )

    if not rows:
        raise RuntimeError(
            f"{dataset}: no usable frames had JSON detections.\n"
            f"Checked SAM dir: {sam_dir}\n"
            "Tip: ensure json names match frame names, and that filters aren't too strict."
        )

    df_tracks = pd.DataFrame(rows)
    df_tracks.sort_values(["track_id", "t_ms"], inplace=True)

    # Enforce monotonic if requested
    if enforce_monotonic:
        df_tracks = enforce_monotonic_per_track(df_tracks)
    else:
        df_tracks["R_mono"] = df_tracks["R_px"].astype(float)

    # Track summary
    g = df_tracks.groupby("track_id")
    track_summary = pd.DataFrame(
        {
            "track_id": g.size().index.astype(int),
            "n_points": g.size().values.astype(int),
            "t_nuc_ms": g["t_ms"].min().values.astype(float),
            "t_end_ms": g["t_ms"].max().values.astype(float),
            "R0_px": g["R_mono"].first().values.astype(float),
            "R_end_px": g["R_mono"].last().values.astype(float),
            "R_max_px": g["R_mono"].max().values.astype(float),
        }
    ).sort_values("track_id")

    # Save CSVs
    df_tracks_out = ds_out / "tracks.csv"
    summ_out = ds_out / "track_summary.csv"
    df_tracks.to_csv(df_tracks_out, index=False)
    track_summary.to_csv(summ_out, index=False)

    # Plots
    plot_tracks(df_tracks, dataset, ds_out / "tracks_Ri.png", n_show=25, seed=0)
    plot_mean_R(df_tracks, dataset, ds_out / "Rmean.png")
    plot_tnuc_hist(track_summary, dataset, ds_out / "t_nuc_hist.png")
    plot_cohort_Rtau(df_tracks, track_summary, dataset, ds_out / "cohort_Rtau.png", dt_ms=float(dt_ms), min_cohort_count=int(min_cohort_count))

    if verbose:
        print(f"{dataset}: wrote {df_tracks_out}")
        print(f"{dataset}: wrote {summ_out}")
        print(f"{dataset}: plots in {ds_out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_root", required=True, type=str, help="Root folder containing per-dataset frame PNGs.")
    ap.add_argument("--sam_root", required=True, type=str, help="Root folder containing per-dataset SAM JSONs.")
    ap.add_argument("--datasets", required=True, type=str, help='Comma/space separated, e.g. "FAPI,FAPI_TEMPO"')
    ap.add_argument("--out_root", default="out", type=str, help="Output root folder.")
    ap.add_argument("--skip_sam", action="store_true", help="Assume JSONs already exist; do not generate SAM outputs.")

    ap.add_argument("--min_area_px", default=300.0, type=float)
    ap.add_argument("--max_area_frac", default=0.25, type=float)
    ap.add_argument("--max_R_px", default=800.0, type=float)

    ap.add_argument("--enforce_monotonic", action="store_true")
    ap.add_argument("--min_cohort_count", default=8, type=int)

    ap.add_argument("--max_match_dist_px", default=80.0, type=float)
    ap.add_argument("--dt_ms", default=None, type=float, help="Frame time step; if omitted, inferred from filenames.")
    ap.add_argument("--iou_weight", default=0.3, type=float, help="Weight on IoU term in matching cost (0..1).")
    ap.add_argument("--max_delta_R_px", default=None, type=float, help="Optional gate: |ΔR| must be <= this to match.")
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    frames_root = Path(args.frames_root)
    sam_root = Path(args.sam_root)
    out_root = Path(args.out_root)
    safe_mkdir(out_root)

    datasets = split_datasets_arg(args.datasets)
    if not datasets:
        raise ValueError("No datasets parsed from --datasets")

    for ds in datasets:
        process_dataset(
            dataset=ds,
            frames_root=frames_root,
            sam_root=sam_root,
            out_root=out_root,
            skip_sam=bool(args.skip_sam),
            min_area_px=float(args.min_area_px),
            max_area_frac=float(args.max_area_frac),
            max_R_px=float(args.max_R_px),
            enforce_monotonic=bool(args.enforce_monotonic),
            min_cohort_count=int(args.min_cohort_count),
            max_match_dist_px=float(args.max_match_dist_px),
            dt_ms=(None if args.dt_ms is None else float(args.dt_ms)),
            iou_weight=float(args.iou_weight),
            max_delta_R_px=(None if args.max_delta_R_px is None else float(args.max_delta_R_px)),
            verbose=bool(args.verbose),
        )


if __name__ == "__main__":
    main()
