import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


# ---------------------------
# File/frame utilities
# ---------------------------

def natural_key(p: Path):
    m = re.findall(r"(\d+)", p.stem)
    return (int(m[-1]) if m else -1, p.name)


def list_frames(frames_dir: Path, pattern: str) -> List[Path]:
    frames = sorted(frames_dir.glob(pattern), key=natural_key)
    if not frames:
        raise FileNotFoundError(f"No frames found in {frames_dir} with pattern {pattern}")
    return frames


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return img


# ---------------------------
# ROI building (ROBUST)
# ---------------------------

def largest_component_mask(bin_img_255: np.ndarray) -> np.ndarray:
    """bin_img_255 is uint8 {0,255}"""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img_255, connectivity=8)
    if num <= 1:
        return (bin_img_255 > 0).astype(np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    return (labels == idx).astype(np.uint8)


def build_roi_mask(first_gray: np.ndarray) -> np.ndarray:
    """
    Robust ROI mask:
    - Try multiple percentile thresholds
    - Morph close
    - Keep largest component
    - If still bad, fallback to full frame
    """
    g = first_gray.astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 3)

    H, W = g.shape
    full = np.ones((H, W), dtype=np.uint8)

    for perc in [1, 2, 5, 10, 15, 20, 30]:
        thr = np.percentile(g, perc)
        m = (g > thr).astype(np.uint8) * 255
        m = cv2.morphologyEx(
            m, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)),
            iterations=1
        )
        m = (largest_component_mask(m) > 0).astype(np.uint8)

        n = int(np.sum(m > 0))
        if n > 0.05 * H * W:
            return m

    return full


def safe_fill_value(gray: np.ndarray, roi: np.ndarray) -> int:
    inside = gray[roi > 0]
    if inside.size:
        v = float(np.median(inside))
        if np.isfinite(v):
            return int(v)
    v2 = float(np.median(gray))
    return int(v2) if np.isfinite(v2) else 0


# ---------------------------
# Segmentation helpers
# ---------------------------

def otsu_binary(gray: np.ndarray, roi: np.ndarray, polarity: str, blur_sigma: float = 2.0) -> np.ndarray:
    """
    Otsu threshold within ROI. polarity:
      - "dark": foreground becomes darker than background -> threshold on inverted
      - "bright": foreground becomes brighter -> threshold directly
    Returns uint8 mask {0,1}
    """
    g = gray.copy()
    if blur_sigma and blur_sigma > 0:
        g = cv2.GaussianBlur(g, (0, 0), blur_sigma)

    fill = safe_fill_value(g, roi)
    g2 = g.copy()
    g2[roi == 0] = fill

    if polarity == "dark":
        g2 = 255 - g2

    _, th = cv2.threshold(g2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = (th > 0).astype(np.uint8)
    th[roi == 0] = 0
    return th


def detect_nuclei(gray: np.ndarray,
                  roi: np.ndarray,
                  min_area: int,
                  max_area: int,
                  blackhat_ksize: int = 31) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect nuclei as small dark blobs using blackhat + Otsu + area filtering.
    Returns:
      - centroids Nx2 (cx,cy)
      - areas N
    """
    g = gray.copy()
    g[roi == 0] = safe_fill_value(g, roi)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (blackhat_ksize, blackhat_ksize))
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k)
    bh = cv2.GaussianBlur(bh, (0, 0), 1.2)

    _, bw = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw[roi == 0] = 0
    bw = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1
    )

    n, labels, stats, cent = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n <= 1:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=int)

    keep_c, keep_a = [], []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < min_area or a > max_area:
            continue
        cx, cy = cent[i]
        keep_c.append((float(cx), float(cy)))
        keep_a.append(a)

    if not keep_c:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=int)

    return np.array(keep_c, dtype=float), np.array(keep_a, dtype=int)


# ---------------------------
# Tracking
# ---------------------------

@dataclass
class Track:
    tid: int
    first_frame: int
    first_time_s: float
    last_frame: int
    last_time_s: float
    cx: float
    cy: float
    detections: int
    confirmed: bool
    confirm_frame: Optional[int] = None
    confirm_time_s: Optional[float] = None


def match_detections_to_tracks(det_xy: np.ndarray,
                               tracks: Dict[int, Track],
                               max_dist_px: float) -> Tuple[Dict[int, int], List[int]]:
    """
    Greedy nearest-neighbor matching: det_index -> track_id.
    Returns matches and unmatched detection indices.
    """
    if det_xy.shape[0] == 0:
        return {}, []
    if len(tracks) == 0:
        return {}, list(range(det_xy.shape[0]))

    track_ids = list(tracks.keys())
    track_xy = np.array([(tracks[tid].cx, tracks[tid].cy) for tid in track_ids], dtype=float)

    # FIX: compute true Euclidean distance matrix (N_det x N_tracks)
    diff = det_xy[:, None, :] - track_xy[None, :, :]           # (N, M, 2)
    d = np.sqrt((diff * diff).sum(axis=2))                     # (N, M)

    pairs = np.argwhere(d <= max_dist_px)                      # (K,2): det_i, track_j
    if pairs.size == 0:
        return {}, list(range(det_xy.shape[0]))

    pairs = sorted(
        [(int(i), int(j), float(d[int(i), int(j)])) for i, j in pairs],
        key=lambda x: x[2]
    )

    matches = {}
    used_tracks = set()
    used_dets = set()

    for di, tj, _dist in pairs:
        if di in used_dets:
            continue
        tid = track_ids[tj]
        if tid in used_tracks:
            continue
        matches[di] = tid
        used_dets.add(di)
        used_tracks.add(tid)

    unmatched = [i for i in range(det_xy.shape[0]) if i not in used_dets]
    return matches, unmatched


# ---------------------------
# Kinetics
# ---------------------------

def robust_growth_proxy_dt(cryst_mask01: np.ndarray, centers_xy: np.ndarray) -> np.ndarray:
    """
    Radius proxy (px): distanceTransform of crystallized mask.
    For each center, r_px = DT at that point (distance to nearest non-crystal pixel).
    """
    fg = (cryst_mask01 > 0).astype(np.uint8) * 255
    dt = cv2.distanceTransform(fg, cv2.DIST_L2, 5)

    r = np.zeros((centers_xy.shape[0],), dtype=float)
    H, W = dt.shape
    for i, (cx, cy) in enumerate(centers_xy):
        x = int(np.clip(round(cx), 0, W - 1))
        y = int(np.clip(round(cy), 0, H - 1))
        r[i] = float(dt[y, x])
    return r


def compute_dn_dt(confirmed_tracks: List[Track], roi_area_um2: float, bin_s: float) -> pd.DataFrame:
    if not confirmed_tracks or roi_area_um2 <= 0:
        return pd.DataFrame(columns=["bin_start_s", "bin_center_s", "new_tracks", "dn_dt_per_um2_s"])

    nuc_times = np.array([t.confirm_time_s for t in confirmed_tracks if t.confirm_time_s is not None], dtype=float)
    if nuc_times.size == 0:
        return pd.DataFrame(columns=["bin_start_s", "bin_center_s", "new_tracks", "dn_dt_per_um2_s"])

    t0 = float(np.min(nuc_times))
    t1 = float(np.max(nuc_times))
    edges = np.arange(np.floor(t0 / bin_s) * bin_s, np.ceil(t1 / bin_s) * bin_s + bin_s, bin_s)
    if len(edges) < 2:
        edges = np.array([t0, t0 + bin_s], dtype=float)

    counts, _ = np.histogram(nuc_times, bins=edges)
    bin_starts = edges[:-1]
    bin_centers = bin_starts + 0.5 * bin_s
    dn_dt = counts / (roi_area_um2 * bin_s)

    return pd.DataFrame({
        "bin_start_s": bin_starts,
        "bin_center_s": bin_centers,
        "new_tracks": counts,
        "dn_dt_per_um2_s": dn_dt
    })


def compute_growth_rate_hist(df_inst: pd.DataFrame, bin_s: float) -> pd.DataFrame:
    if df_inst.empty:
        return pd.DataFrame(columns=["bin_center_s", "dr_dt_median_um_s", "n_samples"])

    df = df_inst.sort_values(["track_id", "t_s"]).copy()
    df["dr_um"] = df.groupby("track_id")["r_um"].diff()
    df["dt_s"] = df.groupby("track_id")["t_s"].diff()
    df["dr_dt_um_s"] = df["dr_um"] / df["dt_s"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["dr_dt_um_s"])

    if df.empty:
        return pd.DataFrame(columns=["bin_center_s", "dr_dt_median_um_s", "n_samples"])

    t0 = float(df["t_s"].min())
    t1 = float(df["t_s"].max())
    edges = np.arange(np.floor(t0 / bin_s) * bin_s, np.ceil(t1 / bin_s) * bin_s + bin_s, bin_s)
    if len(edges) < 2:
        edges = np.array([t0, t0 + bin_s], dtype=float)

    df["bin"] = np.digitize(df["t_s"].values, edges) - 1

    out = []
    for b in sorted(df["bin"].unique()):
        if b < 0 or b >= len(edges) - 1:
            continue
        chunk = df[df["bin"] == b]
        out.append({
            "bin_center_s": float(edges[b] + 0.5 * bin_s),
            "dr_dt_median_um_s": float(np.median(chunk["dr_dt_um_s"].values)),
            "n_samples": int(len(chunk))
        })
    return pd.DataFrame(out)


# ---------------------------
# Dataset runner
# ---------------------------

def run_dataset(label: str,
                frames_dir: Path,
                pattern: str,
                fps: float,
                px_per_um: float,
                out_dir: Path,
                bin_s: float,
                min_nuc_area_px: int,
                max_nuc_area_px: int,
                confirm_n: int,
                max_link_dist_px: float,
                cryst_polarity: str,
                nuclei_blackhat_ksize: int,
                debug_every: int = 25) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = list_frames(frames_dir, pattern)
    first = read_gray(frames[0])
    H, W = first.shape

    roi = build_roi_mask(first)
    roi_px = int(np.sum(roi > 0))
    if roi_px == 0:
        roi[:] = 1
        roi_px = int(np.sum(roi > 0))

    um_per_px = 1.0 / float(px_per_um)
    roi_area_um2 = float(roi_px) * (um_per_px ** 2)

    print(f"[INFO] {label}: frames={len(frames)} size={W}x{H} ROI_px={roi_px} ROI_area={roi_area_um2:.3e} um^2 (px_per_um={px_per_um})")

    tracks: Dict[int, Track] = {}
    next_tid = 1
    rows_inst = []

    for fi, fp in enumerate(frames):
        t_s = fi / float(fps)
        gray = read_gray(fp)

        gray_roi = gray.copy()
        gray_roi[roi == 0] = safe_fill_value(gray, roi)

        # crystallized mask for growth proxy
        cryst01 = otsu_binary(gray_roi, roi, polarity=cryst_polarity, blur_sigma=2.0)
        cryst01 = cv2.morphologyEx(cryst01.astype(np.uint8) * 255, cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
        cryst01 = (cryst01 > 0).astype(np.uint8)

        # nuclei detection
        det_xy, _ = detect_nuclei(
            gray_roi, roi,
            min_area=min_nuc_area_px,
            max_area=max_nuc_area_px,
            blackhat_ksize=nuclei_blackhat_ksize
        )

        matches, unmatched = match_detections_to_tracks(det_xy, tracks, max_dist_px=max_link_dist_px)

        # update matched
        for di, tid in matches.items():
            tr = tracks[tid]
            tr.last_frame = fi
            tr.last_time_s = t_s
            tr.cx = float(det_xy[di, 0])
            tr.cy = float(det_xy[di, 1])
            tr.detections += 1
            if (not tr.confirmed) and (tr.detections >= confirm_n):
                tr.confirmed = True
                tr.confirm_frame = tr.first_frame
                tr.confirm_time_s = tr.first_time_s

        # new tracks
        for di in unmatched:
            cx, cy = float(det_xy[di, 0]), float(det_xy[di, 1])
            confirmed = (confirm_n <= 1)
            tracks[next_tid] = Track(
                tid=next_tid,
                first_frame=fi,
                first_time_s=t_s,
                last_frame=fi,
                last_time_s=t_s,
                cx=cx,
                cy=cy,
                detections=1,
                confirmed=confirmed,
                confirm_frame=fi if confirmed else None,
                confirm_time_s=t_s if confirmed else None
            )
            next_tid += 1

        # growth proxy for confirmed tracks only
        confirmed_now = [tr for tr in tracks.values() if tr.confirmed]
        if confirmed_now:
            centers = np.array([(tr.cx, tr.cy) for tr in confirmed_now], dtype=float)
            r_px = robust_growth_proxy_dt(cryst01, centers)
            r_um = r_px * um_per_px
            area_um2 = np.pi * (r_um ** 2)

            for tr, rp, ru, au in zip(confirmed_now, r_px, r_um, area_um2):
                rows_inst.append({
                    "frame": fi,
                    "t_s": t_s,
                    "track_id": tr.tid,
                    "cx": tr.cx,
                    "cy": tr.cy,
                    "r_px": float(rp),
                    "r_um": float(ru),
                    "area_um2_est": float(au),
                })

        if debug_every and (fi % debug_every == 0):
            n_conf = sum(1 for tr in tracks.values() if tr.confirmed)
            print(f"[INFO] {label}: processed {fi}/{len(frames)} | tracks={len(tracks)} confirmed={n_conf}")

    df_inst = pd.DataFrame(rows_inst)
    df_inst.to_csv(out_dir / "per_instance_tracks.csv", index=False)

    confirmed_tracks = [tr for tr in tracks.values() if tr.confirmed]
    dn_dt_df = compute_dn_dt(confirmed_tracks, roi_area_um2=roi_area_um2, bin_s=bin_s)
    dn_dt_df.to_csv(out_dir / "nucleation_histogram.csv", index=False)

    growth_df = compute_growth_rate_hist(df_inst, bin_s=bin_s)
    growth_df.to_csv(out_dir / "growth_rate_histogram.csv", index=False)

    # summary
    summ_rows = []
    if not df_inst.empty:
        last = df_inst.sort_values(["track_id", "t_s"]).groupby("track_id").tail(1)
        last_map = {int(r.track_id): r for r in last.itertuples(index=False)}
        for tr in confirmed_tracks:
            lr = last_map.get(tr.tid, None)
            summ_rows.append({
                "track_id": tr.tid,
                "nuc_frame": tr.confirm_frame,
                "nuc_time_s": tr.confirm_time_s,
                "final_frame": tr.last_frame,
                "final_time_s": tr.last_time_s,
                "final_cx": tr.cx,
                "final_cy": tr.cy,
                "final_r_um": float(lr.r_um) if lr is not None else np.nan,
                "final_area_um2_est": float(lr.area_um2_est) if lr is not None else np.nan,
                "n_detections": tr.detections
            })

    df_sum = pd.DataFrame(summ_rows)
    df_sum.to_csv(out_dir / "per_track_summary.csv", index=False)

    return dn_dt_df, growth_df, df_sum


# ---------------------------
# Plotting
# ---------------------------

def plot_compare_dn_dt(dn1: pd.DataFrame, dn2: pd.DataFrame, label1: str, label2: str, out_png: Path):
    plt.figure()
    if not dn1.empty:
        plt.plot(dn1["bin_center_s"], dn1["dn_dt_per_um2_s"], label=label1)
    if not dn2.empty:
        plt.plot(dn2["bin_center_s"], dn2["dn_dt_per_um2_s"], label=label2)
    plt.xlabel("Time (s)")
    plt.ylabel("dn/dt (1/µm²/s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_compare_growth(g1: pd.DataFrame, g2: pd.DataFrame, label1: str, label2: str, out_png: Path):
    plt.figure()
    if not g1.empty:
        plt.plot(g1["bin_center_s"], g1["dr_dt_median_um_s"], label=label1)
    if not g2.empty:
        plt.plot(g2["bin_center_s"], g2["dr_dt_median_um_s"], label=label2)
    plt.xlabel("Time (s)")
    plt.ylabel("median dr/dt (µm/s) [proxy]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fapi_dir", required=True)
    ap.add_argument("--tempo_dir", required=True)
    ap.add_argument("--pattern", default="*.png")

    ap.add_argument("--fps_fapi", type=float, default=200.0)
    ap.add_argument("--fps_tempo", type=float, default=125.0)

    ap.add_argument("--px_per_um_gray", type=float, default=1.936641)

    ap.add_argument("--bin_s", type=float, default=1.0)
    ap.add_argument("--out_dir", default="kinetics_um_out")

    ap.add_argument("--min_nuc_area_px", type=int, default=6)
    ap.add_argument("--max_nuc_area_px", type=int, default=300)
    ap.add_argument("--confirm_n", type=int, default=3)
    ap.add_argument("--max_link_dist_px", type=float, default=12.0)
    ap.add_argument("--nuclei_blackhat_ksize", type=int, default=31)

    ap.add_argument("--cryst_polarity", choices=["dark", "bright"], default="dark")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dn_fapi, g_fapi, _ = run_dataset(
        label="FAPI",
        frames_dir=Path(args.fapi_dir),
        pattern=args.pattern,
        fps=args.fps_fapi,
        px_per_um=args.px_per_um_gray,
        out_dir=out_dir / "FAPI",
        bin_s=args.bin_s,
        min_nuc_area_px=args.min_nuc_area_px,
        max_nuc_area_px=args.max_nuc_area_px,
        confirm_n=args.confirm_n,
        max_link_dist_px=args.max_link_dist_px,
        cryst_polarity=args.cryst_polarity,
        nuclei_blackhat_ksize=args.nuclei_blackhat_ksize,
        debug_every=25
    )

    dn_t, g_t, _ = run_dataset(
        label="FAPI-TEMPO",
        frames_dir=Path(args.tempo_dir),
        pattern=args.pattern,
        fps=args.fps_tempo,
        px_per_um=args.px_per_um_gray,
        out_dir=out_dir / "FAPI-TEMPO",
        bin_s=args.bin_s,
        min_nuc_area_px=args.min_nuc_area_px,
        max_nuc_area_px=args.max_nuc_area_px,
        confirm_n=args.confirm_n,
        max_link_dist_px=args.max_link_dist_px,
        cryst_polarity=args.cryst_polarity,
        nuclei_blackhat_ksize=args.nuclei_blackhat_ksize,
        debug_every=25
    )

    plot_compare_dn_dt(dn_fapi, dn_t, "FAPI", "FAPI-TEMPO", out_dir / "dn_dt_compare_um.png")
    plot_compare_growth(g_fapi, g_t, "FAPI", "FAPI-TEMPO", out_dir / "growth_compare_um.png")

    print(f"[OK] outputs: {out_dir}")
    print(f" - {out_dir / 'dn_dt_compare_um.png'}")
    print(f" - {out_dir / 'growth_compare_um.png'}")
    print("Per-dataset CSVs in:")
    print(f" - {out_dir / 'FAPI'}")
    print(f" - {out_dir / 'FAPI-TEMPO'}")


if __name__ == "__main__":
    main()
