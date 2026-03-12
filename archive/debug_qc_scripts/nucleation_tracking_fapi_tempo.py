# nucleation_tracking_fapi_tempo.py
# Loads PNG frames, segments into instances, tracks across frames, and saves per-grain data + nucleation histogram.

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image

try:
    from scipy import ndimage as ndi
except ImportError as e:
    raise ImportError(
        "This script needs scipy for connected-components. Install with:\n"
        "  pip install scipy\n"
    ) from e


# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class Config:
    # You can point these either directly at the frames folder
    # OR at the parent folder (it will auto-detect subfolders that contain PNGs).
    FAPI_FRAMES_DIR: Path = Path(r"F:\Sandy_data\Sandy\12.11.2025\sequences\v5\FAPI_files")
    FAPITEMPO_FRAMES_DIR: Path = Path(r"F:\Sandy_data\Sandy\12.11.2025\sequences\v4\FAPI-TEMPO_files")

    OUT_DIR_FAPI: Path = Path(r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\comparative datasets\FAPI")
    OUT_DIR_FAPITEMPO: Path = Path(r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\comparative datasets\FAPI_TEMPO")

    # Frame loading
    EXTENSIONS: Tuple[str, ...] = (".png",)

    # Segmentation settings (simple baseline)
    # If your images are bright grains on dark background, this works okay.
    # If inverted, set INVERT=True.
    INVERT: bool = False
    MIN_AREA_PX: int = 8

    # Tracking settings
    MAX_LINK_DIST_PX: float = 25.0  # max centroid distance for linking an object to previous frame


# ----------------------------
# Utilities
# ----------------------------

_NUM_RE = re.compile(r"(\d+)")


def natural_key(p: Path):
    s = p.name
    return [int(t) if t.isdigit() else t.lower() for t in _NUM_RE.split(s)]


def find_pngs_in_dir(folder: Path, exts: Tuple[str, ...], recursive: bool) -> List[Path]:
    globber = folder.rglob if recursive else folder.glob
    files: List[Path] = []
    for ext in exts:
        files.extend(globber(f"*{ext}"))
    return sorted(files, key=natural_key)


def auto_detect_frames_dir(frames_dir: Path, exts: Tuple[str, ...]) -> Path:
    """
    If frames_dir contains no PNGs directly, search immediate subfolders for PNGs and choose the best one.
    """
    direct = find_pngs_in_dir(frames_dir, exts=exts, recursive=False)
    if direct:
        return frames_dir

    if not frames_dir.exists():
        raise RuntimeError(f"Frames directory does not exist: {frames_dir}")

    # Look one level down for subfolders containing images
    subfolders = [p for p in frames_dir.iterdir() if p.is_dir()]
    best_folder = None
    best_count = 0

    for sub in subfolders:
        sub_files = find_pngs_in_dir(sub, exts=exts, recursive=False)
        if len(sub_files) > best_count:
            best_count = len(sub_files)
            best_folder = sub

    if best_folder is None or best_count == 0:
        # as a last resort, try recursive in the whole tree
        rec = find_pngs_in_dir(frames_dir, exts=exts, recursive=True)
        if rec:
            return frames_dir  # we can load recursively later if you want
        raise RuntimeError(
            f"No image frames found in {frames_dir}\n"
            f"Subfolders checked: {[p.name for p in subfolders]}"
        )

    print(f"[Loader] No PNGs in {frames_dir}. Using subfolder: {best_folder} ({best_count} PNGs)")
    return best_folder


def load_frames_from_dir(frames_dir: Path, exts: Tuple[str, ...]) -> List[Path]:
    frames_dir = Path(frames_dir)
    frames_dir = auto_detect_frames_dir(frames_dir, exts)

    files = find_pngs_in_dir(frames_dir, exts=exts, recursive=False)
    if not files:
        raise RuntimeError(f"No image frames found in {frames_dir}")

    return files


def read_grayscale(path: Path) -> np.ndarray:
    # Load as grayscale float32 [0..1]
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def otsu_threshold(img: np.ndarray) -> float:
    """
    Pure-numpy Otsu threshold for grayscale [0..1].
    """
    x = np.clip((img * 255.0).round().astype(np.uint8), 0, 255)
    hist = np.bincount(x.ravel(), minlength=256).astype(np.float64)
    hist /= hist.sum()

    omega = np.cumsum(hist)
    mu = np.cumsum(hist * np.arange(256))
    mu_t = mu[-1]

    # between-class variance
    denom = omega * (1.0 - omega)
    denom[denom == 0] = np.nan
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom

    k = int(np.nanargmax(sigma_b2))
    return k / 255.0


# ----------------------------
# Segmentation (baseline)
# ----------------------------

def segment_instances(img: np.ndarray, invert: bool, min_area: int) -> np.ndarray:
    """
    Returns labeled instance mask (0=background, 1..N=instances).

    Replace this function with your ML segmentation if you have one.
    For example, if your model already outputs instance IDs, just return that.
    """
    thr = otsu_threshold(img)
    if invert:
        bw = img < thr
    else:
        bw = img > thr

    # Clean up a bit (optional)
    bw = ndi.binary_opening(bw, iterations=1)

    labeled, n = ndi.label(bw)

    if n == 0:
        return labeled.astype(np.int32)

    # Filter by area
    sizes = np.bincount(labeled.ravel())
    keep = np.zeros_like(sizes, dtype=bool)
    keep[0] = False
    keep[sizes >= min_area] = True

    labeled2 = labeled.copy()
    labeled2[~keep[labeled2]] = 0

    # Re-label to 1..K consecutive
    labeled2, _ = ndi.label(labeled2 > 0)
    return labeled2.astype(np.int32)


def instances_to_props(lbl: np.ndarray) -> pd.DataFrame:
    """
    Extract centroid + area for each instance id in lbl.
    """
    ids = np.unique(lbl)
    ids = ids[ids != 0]
    if len(ids) == 0:
        return pd.DataFrame(columns=["instance_id", "cy", "cx", "area"])

    areas = ndi.sum(np.ones_like(lbl, dtype=np.float32), labels=lbl, index=ids)
    cy = ndi.center_of_mass(np.ones_like(lbl, dtype=np.float32), labels=lbl, index=ids)
    # center_of_mass returns list of tuples (y,x)
    cy = np.array([c[0] for c in cy], dtype=np.float32)
    cx = np.array([c[1] for c in ndi.center_of_mass(np.ones_like(lbl, dtype=np.float32), labels=lbl, index=ids)], dtype=np.float32)

    return pd.DataFrame(
        {
            "instance_id": ids.astype(np.int32),
            "cy": cy,
            "cx": cx,
            "area": np.array(areas, dtype=np.float32),
        }
    )


# ----------------------------
# Tracking
# ----------------------------

def track_by_nearest_centroid(
    props_per_frame: List[pd.DataFrame],
    max_dist: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Simple frame-to-frame centroid linking.

    Returns:
      - per_instance_tracks: rows for each instance observation with track_id
      - per_track_summary: one row per track with birth/death frames, lifetime, etc.
      - nucleation_hist: number of NEW tracks born per frame
    """
    next_track_id = 1
    active: Dict[int, Tuple[float, float]] = {}  # track_id -> (cy, cx)

    rows = []
    births = []

    for t, df in enumerate(props_per_frame):
        df = df.copy()
        if df.empty:
            births.append({"frame": t, "new_tracks": 0})
            active = {}  # everything dies if nothing detected; adjust if you prefer persistence
            continue

        # If no active tracks, every instance is a new track
        if not active:
            track_ids = []
            for _, r in df.iterrows():
                tid = next_track_id
                next_track_id += 1
                track_ids.append(tid)
                active[tid] = (float(r.cy), float(r.cx))
            df["track_id"] = track_ids
            births.append({"frame": t, "new_tracks": len(track_ids)})
        else:
            # Assign each detection to nearest active track under max_dist
            det = df[["cy", "cx"]].to_numpy(dtype=np.float32)
            act_ids = np.array(list(active.keys()), dtype=np.int32)
            act_xy = np.array([active[i] for i in act_ids], dtype=np.float32)

            # Distance matrix: [n_det, n_act]
            d2 = ((det[:, None, :] - act_xy[None, :, :]) ** 2).sum(axis=2)
            d = np.sqrt(d2)

            # Greedy matching: repeatedly take closest pair
            used_det = set()
            used_act = set()
            assigned_track = [-1] * len(df)

            # Flatten indices sorted by distance
            flat = np.argsort(d.ravel())
            n_act = d.shape[1]

            for idx in flat:
                i = int(idx // n_act)  # det index
                j = int(idx % n_act)   # act index
                if i in used_det or j in used_act:
                    continue
                if d[i, j] > max_dist:
                    break
                used_det.add(i)
                used_act.add(j)
                assigned_track[i] = int(act_ids[j])

            # Unassigned detections become new tracks
            new_count = 0
            for i in range(len(assigned_track)):
                if assigned_track[i] == -1:
                    assigned_track[i] = next_track_id
                    next_track_id += 1
                    new_count += 1

            df["track_id"] = assigned_track
            births.append({"frame": t, "new_tracks": new_count})

            # Update active tracks to new positions (drop unmatched act tracks)
            new_active: Dict[int, Tuple[float, float]] = {}
            for _, r in df.iterrows():
                new_active[int(r.track_id)] = (float(r.cy), float(r.cx))
            active = new_active

        # Save per-instance observations
        df["frame"] = t
        rows.append(df[["frame", "track_id", "instance_id", "cy", "cx", "area"]])

    per_instance = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["frame", "track_id", "instance_id", "cy", "cx", "area"]
    )

    # Per-track summary
    if per_instance.empty:
        per_track = pd.DataFrame(columns=[
            "track_id", "birth_frame", "death_frame", "lifetime_frames",
            "mean_area", "max_area"
        ])
    else:
        g = per_instance.groupby("track_id", as_index=False)
        per_track = g.agg(
            birth_frame=("frame", "min"),
            death_frame=("frame", "max"),
            lifetime_frames=("frame", lambda x: int(x.max() - x.min() + 1)),
            mean_area=("area", "mean"),
            max_area=("area", "max"),
        )

    nucleation_hist = pd.DataFrame(births)
    return per_instance, per_track, nucleation_hist


# ----------------------------
# Pipeline
# ----------------------------

def process_experiment(frames_dir: Path, out_dir: Path, label: str, cfg: Config) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = load_frames_from_dir(Path(frames_dir), exts=cfg.EXTENSIONS)
    print(f"[{label}] Loaded {len(frame_paths)} frames from {Path(frames_dir)}")

    props_per_frame: List[pd.DataFrame] = []

    for i, p in enumerate(frame_paths, start=1):
        img = read_grayscale(p)

        lbl = segment_instances(img, invert=cfg.INVERT, min_area=cfg.MIN_AREA_PX)
        props = instances_to_props(lbl)

        props_per_frame.append(props)

        # progress log every 10 frames like your output
        if i % 10 == 0 or i == len(frame_paths):
            print(f"[{label}] Segmented frame {i}/{len(frame_paths)} ({len(props)} instances)")

    per_instance, per_track, nucleation_hist = track_by_nearest_centroid(
        props_per_frame, max_dist=cfg.MAX_LINK_DIST_PX
    )

    print(f"[{label}] Tracking produced {per_track.shape[0]} tracks.")

    per_instance_path = out_dir / "per_instance_tracks.csv"
    per_track_path = out_dir / "per_track_summary.csv"
    nucleation_path = out_dir / "nucleation_histogram.csv"

    per_instance.to_csv(per_instance_path, index=False)
    per_track.to_csv(per_track_path, index=False)
    nucleation_hist.to_csv(nucleation_path, index=False)

    print(f"[{label}] Saved per-track + nucleation data to: {out_dir}")


def main():
    cfg = Config()

    # Run both experiments, but don’t crash the whole run if one is missing.
    try:
        process_experiment(cfg.FAPI_FRAMES_DIR, cfg.OUT_DIR_FAPI, label="FAPI", cfg=cfg)
    except Exception as e:
        print(f"[FAPI] ERROR: {e}", file=sys.stderr)

    try:
        process_experiment(cfg.FAPITEMPO_FRAMES_DIR, cfg.OUT_DIR_FAPITEMPO, label="FAPI_TEMPO", cfg=cfg)
    except Exception as e:
        print(f"[FAPI_TEMPO] ERROR: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
