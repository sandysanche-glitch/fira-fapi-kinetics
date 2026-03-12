import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_frame_index(p: Path) -> int:
    nums = re.findall(r"(\d+)", p.stem)
    if not nums:
        raise ValueError(f"Cannot infer frame index from filename: {p.name}")
    return int(nums[-1])


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Cannot read: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def build_crystallized_mask(frame: np.ndarray, bg: np.ndarray, polarity: str,
                            thr_mode: str, diff_thr: float | None,
                            open_ks: int, close_ks: int) -> np.ndarray:
    if polarity == "dark":
        diff = (bg.astype(np.int16) - frame.astype(np.int16))
    else:
        diff = (frame.astype(np.int16) - bg.astype(np.int16))
    diff = np.clip(diff, 0, 255).astype(np.uint8)

    if thr_mode == "otsu":
        _, bw = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        if diff_thr is None:
            raise ValueError("diff_thr must be set if thr_mode=fixed")
        _, bw = cv2.threshold(diff, int(diff_thr), 255, cv2.THRESH_BINARY)

    if open_ks > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ks, open_ks))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
    if close_ks > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)

    return bw > 0


def detect_seeds_from_mask(cryst: np.ndarray,
                           min_peak_dist_px: int = 25,
                           min_peak_value_px: float = 20.0,
                           max_seeds: int = 2000) -> np.ndarray:
    """
    Detect seed points from the final crystallized mask using distance transform peaks.
    Returns centers_xy as float array (N,2) = (cx, cy).
    """
    u8 = (cryst.astype(np.uint8) * 255)
    dist = cv2.distanceTransform(u8, cv2.DIST_L2, 5).astype(np.float32)

    # local maxima via dilation (non-max suppression)
    k = max(3, int(min_peak_dist_px) | 1)  # odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dist_dil = cv2.dilate(dist, kernel)
    peaks = (dist == dist_dil) & (dist >= float(min_peak_value_px))

    ys, xs = np.where(peaks)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=float)

    vals = dist[ys, xs]
    order = np.argsort(vals)[::-1]
    if len(order) > max_seeds:
        order = order[:max_seeds]

    xs = xs[order].astype(float)
    ys = ys[order].astype(float)
    return np.stack([xs, ys], axis=1)


def make_voronoi_labels(h: int, w: int, centers_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Uses distanceTransformWithLabels by placing 0-valued seed pixels.
    Returns (labels, seed_comp_labels_at_centers).
    """
    seed = np.ones((h, w), np.uint8) * 255
    for (cx, cy) in centers_xy:
        x = int(round(cx))
        y = int(round(cy))
        if 0 <= x < w and 0 <= y < h:
            seed[y, x] = 0

    _, labels = cv2.distanceTransformWithLabels(
        seed, distanceType=cv2.DIST_L2, maskSize=5, labelType=cv2.DIST_LABEL_CCOMP
    )
    labels = labels.astype(np.int32)

    comp_labels = []
    for (cx, cy) in centers_xy:
        x = int(round(cx)); y = int(round(cy))
        comp_labels.append(labels[y, x] if (0 <= x < w and 0 <= y < h) else 0)
    return labels, np.array(comp_labels, dtype=np.int32)


def central_diff_vec(R: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    """
    R: (F,T) radii
    returns v: (F,T) with central differences
    """
    v = np.full_like(R, np.nan, dtype=np.float32)
    if R.shape[0] < 3:
        return v
    dt = (t_s[2:] - t_s[:-2]).astype(np.float32)  # (F-2,)
    v[1:-1, :] = (R[2:, :] - R[:-2, :]) / dt[:, None]
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--pattern", default="*.png")
    ap.add_argument("--frame_times_csv", default=None)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--out_dir", default="FAPI_png_tracks_out")

    ap.add_argument("--bg_n", type=int, default=5)
    ap.add_argument("--polarity", choices=["auto", "dark", "bright"], default="auto")
    ap.add_argument("--thr_mode", choices=["otsu", "fixed"], default="otsu")
    ap.add_argument("--diff_thr", type=float, default=None)
    ap.add_argument("--open_ks", type=int, default=3)
    ap.add_argument("--close_ks", type=int, default=5)

    # seed detection controls
    ap.add_argument("--min_peak_dist_px", type=int, default=35)
    ap.add_argument("--min_peak_value_px", type=float, default=18.0)
    ap.add_argument("--max_seeds", type=int, default=1500)

    ap.add_argument("--nuc_area_thr_px", type=float, default=50.0)

    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = list(frames_dir.glob(args.pattern))
    if not frames:
        raise SystemExit(f"No frames found in {frames_dir} with pattern {args.pattern}")

    # try frame indices from filenames; else 0..N-1
    try:
        frames = sorted(frames, key=parse_frame_index)
        frame_idx = np.array([parse_frame_index(p) for p in frames], dtype=int)
    except Exception:
        frames = sorted(frames)
        frame_idx = np.arange(len(frames), dtype=int)

    nF = len(frames)
    f0 = read_gray(frames[0])
    h, w = f0.shape
    print(f"[INFO] Frames: {nF} | Size: {w}x{h}")

    # time axis
    if args.frame_times_csv:
        ft = pd.read_csv(args.frame_times_csv)
        if not {"frame", "t_s"}.issubset(ft.columns):
            raise SystemExit("frame_times_csv must contain frame,t_s")
        tmap = dict(zip(ft["frame"].astype(int).values, ft["t_s"].astype(float).values))
        t_s = np.array([tmap.get(int(fr), np.nan) for fr in frame_idx], dtype=float)
        if np.any(~np.isfinite(t_s)):
            raise SystemExit("Some PNG frames missing in frame_times_csv (frame indices don’t match). Use --fps instead.")
        t_s = t_s - float(np.min(t_s))
    else:
        if args.fps is None:
            raise SystemExit("Provide either --frame_times_csv or --fps")
        t_s = (frame_idx - frame_idx.min()).astype(float) / float(args.fps)

    # background
    bg_n = max(1, min(args.bg_n, nF))
    bg_stack = np.stack([read_gray(frames[i]) for i in range(bg_n)], axis=0)
    bg = np.median(bg_stack, axis=0).astype(np.uint8)

    # polarity
    polarity = args.polarity
    if polarity == "auto":
        last = read_gray(frames[-1])
        polarity = "dark" if last.mean() < bg.mean() else "bright"
    print(f"[INFO] Polarity: {polarity} | Threshold: {args.thr_mode}")

    # crystallized mask of final frame => seeds
    last_img = read_gray(frames[-1])
    cryst_last = build_crystallized_mask(
        last_img, bg, polarity=polarity,
        thr_mode=args.thr_mode, diff_thr=args.diff_thr,
        open_ks=args.open_ks, close_ks=args.close_ks
    )

    centers_xy = detect_seeds_from_mask(
        cryst_last,
        min_peak_dist_px=args.min_peak_dist_px,
        min_peak_value_px=args.min_peak_value_px,
        max_seeds=args.max_seeds
    )

    if centers_xy.shape[0] == 0:
        raise SystemExit("No seeds detected. Try lowering --min_peak_value_px or --min_peak_dist_px.")

    nT = centers_xy.shape[0]
    print(f"[INFO] Detected nuclei seeds: {nT}")

    # assign track ids 1..nT
    track_ids = np.arange(1, nT + 1, dtype=int)

    # Voronoi labels from seeds
    labels, comp_labels = make_voronoi_labels(h, w, centers_xy)
    max_lab = int(labels.max())

    lab_to_tid = np.zeros((max_lab + 1,), dtype=np.int32)
    for cl, tid in zip(comp_labels, track_ids):
        if cl > 0:
            lab_to_tid[cl] = tid

    tid_to_col = {tid: i for i, tid in enumerate(track_ids)}

    # area matrix
    A = np.zeros((nF, nT), dtype=np.float32)

    # per-frame crystallized area within Voronoi cell
    for i, fp in enumerate(frames):
        img = read_gray(fp)
        cryst = build_crystallized_mask(
            img, bg, polarity=polarity,
            thr_mode=args.thr_mode, diff_thr=args.diff_thr,
            open_ks=args.open_ks, close_ks=args.close_ks
        )

        lab = labels[cryst]
        if lab.size:
            counts = np.bincount(lab.ravel(), minlength=max_lab + 1)
            nz = np.nonzero(counts)[0]
            for cl in nz:
                tid = int(lab_to_tid[cl])
                if tid == 0:
                    continue
                A[i, tid_to_col[tid]] = float(counts[cl])

        if (i % 50) == 0:
            print(f"[INFO] processed frame {i}/{nF}")

    R = np.sqrt(np.maximum(A, 0.0) / np.pi).astype(np.float32)
    v = central_diff_vec(R, t_s)

    # write per_instance_tracks.csv incrementally (no huge np.repeat)
    out_csv = out_dir / "per_instance_tracks.csv"
    header_written = False
    for i in range(nF):
        df = pd.DataFrame({
            "frame": np.full(nT, int(frame_idx[i]), dtype=int),
            "t_s": np.full(nT, float(t_s[i]), dtype=float),
            "track_id": track_ids.astype(int),
            "cx": centers_xy[:, 0].astype(float),
            "cy": centers_xy[:, 1].astype(float),
            "area_px2": A[i, :].astype(float),
            "R_px": R[i, :].astype(float),
            "v_px_per_s": v[i, :].astype(float),
        })
        df.to_csv(out_csv, index=False, mode="w" if not header_written else "a",
                  header=not header_written)
        header_written = True

    # summary + nucleation hist
    thr = float(args.nuc_area_thr_px)
    mask = A >= thr
    has = mask.any(axis=0)
    first = np.where(has, mask.argmax(axis=0), -1)
    last = np.where(has, (nF - 1 - mask[::-1, :].argmax(axis=0)), -1)
    n_det = mask.sum(axis=0).astype(int)

    nuc_frame = np.where(first >= 0, frame_idx[first], frame_idx[0]).astype(int)
    final_frame = np.where(last >= 0, frame_idx[last], frame_idx[-1]).astype(int)
    final_area = np.where(last >= 0, A[last, np.arange(nT)], A[-1, :]).astype(float)

    summary = pd.DataFrame({
        "track_id": track_ids.astype(int),
        "nuc_frame": nuc_frame,
        "final_frame": final_frame,
        "final_area_px": final_area,
        "final_cx": centers_xy[:, 0].astype(float),
        "final_cy": centers_xy[:, 1].astype(float),
        "n_detections": n_det,
    })
    summary.to_csv(out_dir / "per_track_summary.csv", index=False)

    vc = pd.Series(nuc_frame).value_counts().sort_index()
    nuc_hist = pd.DataFrame({"frame": vc.index.astype(int), "new_tracks": vc.values.astype(int)})
    nuc_hist.to_csv(out_dir / "nucleation_histogram.csv", index=False)

    # save seeds too (for sanity-check)
    pd.DataFrame({"track_id": track_ids, "cx": centers_xy[:, 0], "cy": centers_xy[:, 1]}).to_csv(
        out_dir / "detected_nuclei_centers.csv", index=False
    )

    print("[OK] Wrote:")
    print(" ", out_csv)
    print(" ", out_dir / "per_track_summary.csv")
    print(" ", out_dir / "nucleation_histogram.csv")
    print(" ", out_dir / "detected_nuclei_centers.csv")


if __name__ == "__main__":
    main()
