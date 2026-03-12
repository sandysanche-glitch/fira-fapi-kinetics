#!/usr/bin/env python3
# stable_nucleation_curves_lookahead.py

import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Helpers
# -------------------------
def safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def infer_cols(df: pd.DataFrame, dataset_label: str):
    """
    Try to infer columns for:
      track_id, time_ms, frame_idx, area_px, R_px
    Supports your two formats noted in your methods file. :contentReference[oaicite:1]{index=1}
    """
    # track
    if "track_id" not in df.columns:
        raise KeyError(f"[{dataset_label}] Missing track_id column")

    # area
    if "area_px" not in df.columns:
        raise KeyError(f"[{dataset_label}] Missing area_px column")

    # radius
    if "R_px" not in df.columns:
        df = df.copy()
        df["R_px"] = np.sqrt(np.maximum(df["area_px"].astype(float), 0.0) / np.pi)

    # time
    if "time_ms" in df.columns:
        time_col = "time_ms"
    elif "t_ms" in df.columns:
        time_col = "t_ms"
    else:
        # try parse from frame_id strings like frame_00065_t130.00ms
        if "frame_id" in df.columns:
            df = df.copy()
            def parse_t(s):
                m = re.search(r"_t([0-9.]+)ms", str(s))
                return float(m.group(1)) if m else np.nan
            df["time_ms"] = df["frame_id"].map(parse_t)
            time_col = "time_ms"
        else:
            raise KeyError(f"[{dataset_label}] Missing time_ms/t_ms and cannot parse from frame_id")

    # frame index
    if "frame" in df.columns:
        frame_col = "frame"
    elif "frame_idx" in df.columns:
        frame_col = "frame_idx"
    elif "frame_id" in df.columns:
        # parse frame number from frame_00065...
        df = df.copy()
        def parse_f(s):
            m = re.search(r"frame_(\d+)", str(s))
            return int(m.group(1)) if m else np.nan
        df["frame_idx"] = df["frame_id"].map(parse_f)
        frame_col = "frame_idx"
    else:
        raise KeyError(f"[{dataset_label}] Missing frame/frame_idx/frame_id")

    return df, frame_col, time_col

def has_run_of_L(frames: np.ndarray, ok: np.ndarray, L: int) -> int | None:
    """
    frames: sorted integer frame indices
    ok: boolean per row (size thresholds met)
    Return index (position in arrays) of the first frame of the first valid run, else None.
    Run requires consecutive frames with no gaps (frame[t+i] = frame[t] + i).
    """
    n = len(frames)
    if n < L:
        return None
    for start in range(0, n - L + 1):
        if not np.all(ok[start:start+L]):
            continue
        f0 = frames[start]
        if np.all(frames[start:start+L] == (f0 + np.arange(L))):
            return start
    return None

def find_stable_tracks_and_times(df: pd.DataFrame, frame_col: str, time_col: str,
                                 L: int, amin_px: float, rmin_px: float,
                                 nuc_time_mode: str):
    """
    Returns:
      events_df: one row per retained track with nucleation time (ms)
      rejected_df: one row per rejected track with reason
    """
    rows_keep = []
    rows_rej = []

    for tid, g in df.groupby("track_id"):
        g = g.sort_values(frame_col)
        frames = g[frame_col].astype(int).to_numpy()
        times = g[time_col].astype(float).to_numpy()
        area = g["area_px"].astype(float).to_numpy()
        rpx  = g["R_px"].astype(float).to_numpy()

        ok = (area >= amin_px) & (rpx >= rmin_px)
        run_start = has_run_of_L(frames, ok, L)

        if run_start is None:
            rows_rej.append({
                "track_id": int(tid),
                "reason": "no_L_consecutive_frames_meeting_size",
                "n_points": int(len(g)),
                "t_first_ms": float(np.nanmin(times)),
                "t_last_ms": float(np.nanmax(times)),
                "Amax_px": float(np.nanmax(area)),
                "Rmax_px": float(np.nanmax(rpx)),
            })
            continue

        # Track is VALIDATED by size+stability.
        # Now define nucleation time:
        if nuc_time_mode == "first":
            t_nuc = float(times[0])  # first appearance
            f_nuc = int(frames[0])
        elif nuc_time_mode == "first_stable":
            t_nuc = float(times[run_start])  # threshold-crossing (old behavior)
            f_nuc = int(frames[run_start])
        else:
            raise ValueError("nuc_time_mode must be 'first' or 'first_stable'")

        rows_keep.append({
            "track_id": int(tid),
            "nuc_time_ms": t_nuc,
            "nuc_frame": f_nuc,
            "first_time_ms": float(times[0]),
            "first_frame": int(frames[0]),
            "stable_start_time_ms": float(times[run_start]),
            "stable_start_frame": int(frames[run_start]),
            "n_points": int(len(g)),
            "Amax_px": float(np.nanmax(area)),
            "Rmax_px": float(np.nanmax(rpx)),
        })

    events_df = pd.DataFrame(rows_keep).sort_values("nuc_time_ms")
    rej_df = pd.DataFrame(rows_rej).sort_values(["reason","track_id"])
    return events_df, rej_df

def bin_events(nuc_times_ms: np.ndarray, bin_ms: float, tmax_ms: float):
    edges = np.arange(0.0, tmax_ms + bin_ms + 1e-9, bin_ms)
    counts, _ = np.histogram(nuc_times_ms, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dn_dt_per_s = (counts / bin_ms) * 1000.0
    cum_n = np.cumsum(counts).astype(int)
    out = pd.DataFrame({
        "bin_center_ms": centers,
        "n_nucleated": counts.astype(int),
        "dn_dt_per_s": dn_dt_per_s,
        "cum_n": cum_n
    })
    return out

def bootstrap_ci(events_df: pd.DataFrame, bin_ms: float, tmax_ms: float, B: int, seed: int):
    rng = np.random.default_rng(seed)
    times = events_df["nuc_time_ms"].to_numpy(dtype=float)
    n = len(times)
    if n == 0:
        return pd.DataFrame(columns=["bin_center_ms","dn_dt_lo","dn_dt_hi","N_lo","N_hi"])
    edges = np.arange(0.0, tmax_ms + bin_ms + 1e-9, bin_ms)
    centers = 0.5 * (edges[:-1] + edges[1:])

    dn_mat = np.zeros((B, len(centers)), dtype=float)
    N_mat  = np.zeros((B, len(centers)), dtype=float)

    for b in range(B):
        sample = rng.choice(times, size=n, replace=True)
        counts, _ = np.histogram(sample, bins=edges)
        dn_mat[b] = (counts / bin_ms) * 1000.0
        N_mat[b]  = np.cumsum(counts)

    lo = 2.5
    hi = 97.5
    out = pd.DataFrame({
        "bin_center_ms": centers,
        "dn_dt_lo": np.percentile(dn_mat, lo, axis=0),
        "dn_dt_hi": np.percentile(dn_mat, hi, axis=0),
        "N_lo": np.percentile(N_mat, lo, axis=0),
        "N_hi": np.percentile(N_mat, hi, axis=0),
    })
    return out

def plot_overlay(d1, ci1, label1, d2, ci2, label2, out_png, title, ycol, ylo, yhi, normalize=False):
    plt.figure()
    if normalize:
        # normalize by final cum_n
        n1 = max(d1["cum_n"].max(), 1)
        n2 = max(d2["cum_n"].max(), 1)
        y1 = d1[ycol] / n1
        y2 = d2[ycol] / n2
        plt.plot(d1["bin_center_ms"], y1, marker="o", label=f"{label1} (norm)")
        plt.plot(d2["bin_center_ms"], y2, marker="o", label=f"{label2} (norm)")
        if ci1 is not None and len(ci1):
            plt.fill_between(ci1["bin_center_ms"], ci1[ylo]/n1, ci1[yhi]/n1, alpha=0.2)
        if ci2 is not None and len(ci2):
            plt.fill_between(ci2["bin_center_ms"], ci2[ylo]/n2, ci2[yhi]/n2, alpha=0.2)
        plt.ylabel(f"{ycol} / Nfinal")
    else:
        plt.plot(d1["bin_center_ms"], d1[ycol], marker="o", label=label1)
        plt.plot(d2["bin_center_ms"], d2[ycol], marker="o", label=label2)
        if ci1 is not None and len(ci1):
            plt.fill_between(ci1["bin_center_ms"], ci1[ylo], ci1[yhi], alpha=0.2)
        if ci2 is not None and len(ci2):
            plt.fill_between(ci2["bin_center_ms"], ci2[ylo], ci2[yhi], alpha=0.2)
        plt.ylabel(ycol)

    plt.xlabel("time (ms)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fapi_tracks", required=True)
    ap.add_argument("--tempo_tracks", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--L", type=int, default=5)
    ap.add_argument("--amin_px", type=float, default=800.0)
    ap.add_argument("--rmin_px", type=float, default=3.0)
    ap.add_argument("--bin_ms", type=float, default=20.0)
    ap.add_argument("--bootstrap_B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--nuc_time_mode", choices=["first","first_stable"], default="first",
                    help="first = first appearance (recommended); first_stable = threshold-crossing time")
    args = ap.parse_args()
    safe_mkdir(args.out_dir)

    print("[OK] Reading:")
    print("  FAPI :", args.fapi_tracks)
    print("  TEMPO:", args.tempo_tracks)

    fapi = pd.read_csv(args.fapi_tracks)
    tempo = pd.read_csv(args.tempo_tracks)

    fapi, f_frame, f_time = infer_cols(fapi, "FAPI")
    tempo, t_frame, t_time = infer_cols(tempo, "FAPI_TEMPO")

    # Stable track validation + nucleation time definition
    f_events, f_rej = find_stable_tracks_and_times(
        fapi, f_frame, f_time, args.L, args.amin_px, args.rmin_px, args.nuc_time_mode
    )
    t_events, t_rej = find_stable_tracks_and_times(
        tempo, t_frame, t_time, args.L, args.amin_px, args.rmin_px, args.nuc_time_mode
    )

    # Write events
    f_events.to_csv(os.path.join(args.out_dir, "nucleation_events_filtered_FAPI.csv"), index=False)
    t_events.to_csv(os.path.join(args.out_dir, "nucleation_events_filtered_FAPI_TEMPO.csv"), index=False)
    f_rej.to_csv(os.path.join(args.out_dir, "nucleation_events_rejected_FAPI.csv"), index=False)
    t_rej.to_csv(os.path.join(args.out_dir, "nucleation_events_rejected_FAPI_TEMPO.csv"), index=False)

    # Bin to dn/dt and N(t)
    tmax = float(max(
        np.nanmax(fapi[f_time].astype(float).to_numpy()),
        np.nanmax(tempo[t_time].astype(float).to_numpy()),
    ))

    f_curve = bin_events(f_events["nuc_time_ms"].to_numpy(dtype=float), args.bin_ms, tmax)
    t_curve = bin_events(t_events["nuc_time_ms"].to_numpy(dtype=float), args.bin_ms, tmax)

    f_curve.to_csv(os.path.join(args.out_dir, "dn_dt_filtered_FAPI.csv"), index=False)
    t_curve.to_csv(os.path.join(args.out_dir, "dn_dt_filtered_FAPI_TEMPO.csv"), index=False)

    f_N = f_curve[["bin_center_ms","cum_n"]].copy()
    t_N = t_curve[["bin_center_ms","cum_n"]].copy()
    f_N.to_csv(os.path.join(args.out_dir, "N_t_filtered_FAPI.csv"), index=False)
    t_N.to_csv(os.path.join(args.out_dir, "N_t_filtered_FAPI_TEMPO.csv"), index=False)

    # Bootstrap CI
    f_ci = bootstrap_ci(f_events, args.bin_ms, tmax, args.bootstrap_B, args.seed)
    t_ci = bootstrap_ci(t_events, args.bin_ms, tmax, args.bootstrap_B, args.seed+1)
    out_ci = f_ci.merge(t_ci, on="bin_center_ms", how="outer", suffixes=("_FAPI","_TEMPO")).sort_values("bin_center_ms")
    out_ci.to_csv(os.path.join(args.out_dir, "bootstrap_CI_FAPI_and_TEMPO.csv"), index=False)

    # Plots
    plot_overlay(
        f_curve, f_ci, "FAPI",
        t_curve, t_ci, "FAPI-TEMPO",
        os.path.join(args.out_dir, "overlay_dn_dt_filtered.png"),
        title=f"Stable nucleation rate (mode={args.nuc_time_mode}, L={args.L}, Amin={args.amin_px:g}, Rmin={args.rmin_px:g})",
        ycol="dn_dt_per_s", ylo="dn_dt_lo", yhi="dn_dt_hi",
        normalize=False
    )
    plot_overlay(
        f_curve, f_ci, "FAPI",
        t_curve, t_ci, "FAPI-TEMPO",
        os.path.join(args.out_dir, "overlay_dn_dt_filtered_normalized.png"),
        title=f"Stable nucleation rate / Nfinal (mode={args.nuc_time_mode})",
        ycol="dn_dt_per_s", ylo="dn_dt_lo", yhi="dn_dt_hi",
        normalize=True
    )
    plot_overlay(
        f_curve, f_ci, "FAPI",
        t_curve, t_ci, "FAPI-TEMPO",
        os.path.join(args.out_dir, "overlay_N_t_filtered.png"),
        title=f"Cumulative nucleation N(t) (mode={args.nuc_time_mode})",
        ycol="cum_n", ylo="N_lo", yhi="N_hi",
        normalize=False
    )
    plot_overlay(
        f_curve, f_ci, "FAPI",
        t_curve, t_ci, "FAPI-TEMPO",
        os.path.join(args.out_dir, "overlay_N_t_filtered_normalized.png"),
        title=f"Cumulative nucleation N(t)/Nfinal (mode={args.nuc_time_mode})",
        ycol="cum_n", ylo="N_lo", yhi="N_hi",
        normalize=True
    )

    # Methods text
    methods = f"""Stable nucleation curve (look-ahead option)
------------------------------------------------
Validated tracks: keep only tracks that contain at least one run of >=L consecutive frames (no frame gaps)
where each detection satisfies area_px >= Amin AND R_px >= Rmin. (L={args.L}, Amin={args.amin_px:g}px, Rmin={args.rmin_px:g}px)
Nucleation time definition (mode={args.nuc_time_mode}):
  - first: nucleation time = first appearance time of tracks that eventually become stable (recommended to avoid threshold-crossing spikes)
  - first_stable: nucleation time = first frame of the first stable run (this can create late bumps if many grains cross size thresholds together)
Binning: bin_ms={args.bin_ms:g} ms; dn/dt computed as counts/bin_ms * 1000 (1/s).
Uncertainty: bootstrap resampling of tracks/events (B={args.bootstrap_B}, seed={args.seed}) to obtain 95% CI.
"""
    with open(os.path.join(args.out_dir, "stable_nucleation_methods.txt"), "w", encoding="utf-8") as f:
        f.write(methods)

    print("\n=== Summary ===")
    print(f"Mode={args.nuc_time_mode}")
    print(f"FAPI retained n={len(f_events)} | rejected n={len(f_rej)}")
    print(f"TEMPO retained n={len(t_events)} | rejected n={len(t_rej)}")
    print("[OK] Wrote outputs to:", args.out_dir)

if __name__ == "__main__":
    main()
