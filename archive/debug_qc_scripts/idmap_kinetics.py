#!/usr/bin/env python3
"""
ID-map-native kinetics from tracks.csv (no retracking).

Reads tracks.csv (one row per grain per frame), computes:
- nucleation time per grain = first appearance time_ms
- active grain count vs time
- nucleation bins + dn/dt
- tau fit per grain: R(t)=Rinf*(1-exp(-dt/tau))
- early growth slope per grain: linear fit of R vs dt over [0, early_window_ms]
- growth rate vs absolute time: median of per-step dR/dt (raw + clipped)

Outputs CSVs + PNG plots in out_dir.
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def exp_growth(dt_ms, Rinf, tau_ms):
    # dt_ms >= 0
    return Rinf * (1.0 - np.exp(-dt_ms / tau_ms))


def r2_score(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks_csv", required=True, help="tracks.csv from build_tracks_from_idmap_jsons.py")
    ap.add_argument("--out_dir", required=True, help="output folder")
    ap.add_argument("--dt_ms", type=float, default=2.0)
    ap.add_argument("--px_um", type=float, default=0.065)

    ap.add_argument("--min_track_frames", type=int, default=10)
    ap.add_argument("--nuc_bin_ms", type=float, default=20.0)

    ap.add_argument("--fit_tau_max_ms", type=float, default=300.0)
    ap.add_argument("--fit_min_points", type=int, default=8)

    # NEW: early growth window for per-grain slope
    ap.add_argument("--early_window_ms", type=float, default=60.0)

    ap.add_argument("--make_plots", action="store_true", help="write PNG plots")
    return ap.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    df = pd.read_csv(args.tracks_csv)
    if df.empty:
        raise RuntimeError("tracks_csv is empty.")

    # Expected columns (from your builder): time_ms, frame, track_id, area_px, R_px
    required = {"time_ms", "frame", "track_id", "area_px", "R_px"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"tracks_csv missing columns: {sorted(missing)}")

    # Basic cleaning / typing
    df["time_ms"] = df["time_ms"].astype(float)
    df["frame"] = df["frame"].astype(int)
    df["track_id"] = df["track_id"].astype(int)
    df["area_px"] = df["area_px"].astype(float)
    df["R_px"] = df["R_px"].astype(float)

    df["R_um"] = df["R_px"] * float(args.px_um)

    # ---- Active tracks vs time ----
    active = (
        df.groupby(["frame", "time_ms"])["track_id"]
          .nunique()
          .reset_index(name="n_active")
          .sort_values("time_ms")
    )
    active.to_csv(Path(args.out_dir) / "active_tracks.csv", index=False)

    if args.make_plots:
        plt.figure()
        plt.plot(active["time_ms"], active["n_active"])
        plt.xlabel("time (ms)")
        plt.ylabel("active grains")
        plt.title("Active grains vs time")
        plt.tight_layout()
        plt.savefig(Path(args.out_dir) / "active_tracks.png", dpi=200)
        plt.close()

    # ---- Per-grain nucleation time (first appearance) ----
    g = df.groupby("track_id", as_index=False).agg(
        t0_ms=("time_ms", "min"),
        first_frame=("frame", "min"),
        n_frames=("frame", "count"),
        Rmax_um=("R_um", "max"),
        Amax_px=("area_px", "max"),
    )
    # Keep only tracks with enough frames
    g = g[g["n_frames"] >= int(args.min_track_frames)].copy()

    # If you want to include short-lived grains, lower min_track_frames.
    # We'll filter df accordingly for the rest:
    keep_ids = set(g["track_id"].tolist())
    dfk = df[df["track_id"].isin(keep_ids)].copy()

    # ---- Nucleation bins + dn/dt ----
    t_min = float(dfk["time_ms"].min())
    t_max = float(dfk["time_ms"].max())
    bin_w = float(args.nuc_bin_ms)

    # bins aligned to 0..max for easier comparison
    edges = np.arange(0.0, t_max + bin_w + 1e-9, bin_w)
    counts, _ = np.histogram(g["t0_ms"].values, bins=edges)

    bin_left = edges[:-1]
    bin_right = edges[1:]
    bin_center = 0.5 * (bin_left + bin_right)

    nuc = pd.DataFrame({
        "bin_left_ms": bin_left,
        "bin_right_ms": bin_right,
        "bin_center_ms": bin_center,
        "n_nucleated": counts.astype(int),
        "dn_dt_per_ms": counts / bin_w,
        "dn_dt_per_s": (counts / bin_w) * 1000.0,
        "cum_n": np.cumsum(counts).astype(int),
    })
    nuc.to_csv(Path(args.out_dir) / "nucleation_bins.csv", index=False)

    if args.make_plots:
        plt.figure()
        plt.plot(nuc["bin_center_ms"], nuc["dn_dt_per_s"], marker="o")
        plt.xlabel("time (ms)")
        plt.ylabel("dn/dt (1/s)")
        plt.title(f"Nucleation rate (bin={bin_w:g} ms)")
        plt.tight_layout()
        plt.savefig(Path(args.out_dir) / "nucleation_rate.png", dpi=200)
        plt.close()

    # Also export a simple dn/dt CSV for your draft (same info, lighter)
    nuc_out = nuc[["bin_center_ms", "n_nucleated", "dn_dt_per_s", "cum_n"]].copy()
    nuc_out.to_csv(Path(args.out_dir) / "dn_dt_nucleation.csv", index=False)
    if args.make_plots:
        plt.figure()
        plt.plot(nuc_out["bin_center_ms"], nuc_out["dn_dt_per_s"], marker="o")
        plt.xlabel("time (ms)")
        plt.ylabel("dn/dt (1/s)")
        plt.title("dn/dt nucleation")
        plt.tight_layout()
        plt.savefig(Path(args.out_dir) / "dn_dt_nucleation.png", dpi=200)
        plt.close()

    # ---- Tau fits per grain ----
    fit_rows = []
    tau_max = float(args.fit_tau_max_ms)

    for tid, dfi in dfk.groupby("track_id"):
        dfi = dfi.sort_values("time_ms")
        t0 = float(dfi["time_ms"].min())
        dfi["dt_ms"] = dfi["time_ms"] - t0

        # Fit window
        dff = dfi[(dfi["dt_ms"] >= 0.0) & (dfi["dt_ms"] <= tau_max)].copy()
        if len(dff) < int(args.fit_min_points):
            continue

        x = dff["dt_ms"].values.astype(float)
        y = dff["R_um"].values.astype(float)
        y = np.maximum(y, 0.0)

        # initial guesses
        Rinf0 = float(np.nanmax(y))
        if not np.isfinite(Rinf0) or Rinf0 <= 0:
            continue
        tau0 = max(5.0, tau_max / 3.0)

        try:
            popt, pcov = curve_fit(
                exp_growth,
                x, y,
                p0=[Rinf0, tau0],
                bounds=([0.0, 1e-3], [np.inf, np.inf]),
                maxfev=20000,
            )
            Rinf, tau = float(popt[0]), float(popt[1])
            yhat = exp_growth(x, Rinf, tau)
            r2 = r2_score(y, yhat)

            fit_rows.append({
                "track_id": int(tid),
                "t0_ms": t0,
                "n_points": int(len(dff)),
                "Rinf_um": Rinf,
                "tau_ms": tau,
                "r2": r2,
                "Rmax_um": float(np.nanmax(dfi["R_um"])),
            })
        except Exception:
            continue

    tau_df = pd.DataFrame(fit_rows).sort_values("tau_ms")
    tau_df.to_csv(Path(args.out_dir) / "tau_fits.csv", index=False)

    if args.make_plots and not tau_df.empty:
        plt.figure()
        plt.hist(tau_df["tau_ms"].values, bins=30)
        plt.xlabel("tau (ms)")
        plt.ylabel("count")
        plt.title("Tau distribution")
        plt.tight_layout()
        plt.savefig(Path(args.out_dir) / "tau_hist.png", dpi=200)
        plt.close()

    # ---- Early growth slope per grain (NEW) ----
    win = float(args.early_window_ms)
    slope_rows = []

    for tid, dfi in dfk.groupby("track_id"):
        dfi = dfi.sort_values("time_ms")
        t0 = float(dfi["time_ms"].min())
        dt = (dfi["time_ms"] - t0).values.astype(float)
        R = dfi["R_um"].values.astype(float)

        m = (dt >= 0.0) & (dt <= win)
        if np.sum(m) < 3:
            continue

        x = dt[m]
        y = R[m]

        # linear fit y = a*x + b
        A = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        yhat = a * x + b
        r2 = r2_score(y, yhat)

        slope_rows.append({
            "track_id": int(tid),
            "t0_ms": t0,
            "n_points": int(np.sum(m)),
            "early_window_ms": win,
            "slope_um_per_ms": float(a),
            "slope_um_per_s": float(a * 1000.0),
            "r2": float(r2),
        })

    slopes = pd.DataFrame(slope_rows)
    slopes.to_csv(Path(args.out_dir) / "per_grain_growth_slopes.csv", index=False)

    if args.make_plots and not slopes.empty:
        plt.figure()
        plt.hist(slopes["slope_um_per_s"].values, bins=40)
        plt.xlabel("early growth slope (µm/s)")
        plt.ylabel("count")
        plt.title(f"Per-grain early growth slopes (0–{win:g} ms)")
        plt.tight_layout()
        plt.savefig(Path(args.out_dir) / "per_grain_growth_slope_hist.png", dpi=200)
        plt.close()

    # ---- Growth rate vs absolute time (median of per-step dR/dt) ----
    # Compute per-step slopes (between consecutive frames) per grain
    step_rows = []
    for tid, dfi in dfk.groupby("track_id"):
        dfi = dfi.sort_values("time_ms")
        t = dfi["time_ms"].values.astype(float)
        R = dfi["R_um"].values.astype(float)

        if len(t) < 2:
            continue

        dt = np.diff(t)  # ms
        dR = np.diff(R)  # um
        # avoid divide by 0
        good = dt > 0
        if not np.any(good):
            continue

        rate_um_per_s = (dR[good] / dt[good]) * 1000.0
        t_mid = 0.5 * (t[1:][good] + t[:-1][good])

        for tm, rr in zip(t_mid, rate_um_per_s):
            step_rows.append((tm, rr))

    if step_rows:
        steps = pd.DataFrame(step_rows, columns=["time_ms", "rate_um_per_s_raw"])
        # Bin rates in the same nuc_bin_ms by default for stability
        edges_r = np.arange(0.0, t_max + bin_w + 1e-9, bin_w)
        steps["bin"] = np.digitize(steps["time_ms"], edges_r) - 1
        steps = steps[(steps["bin"] >= 0) & (steps["bin"] < len(edges_r) - 1)].copy()

        out_rows = []
        for b, dfi in steps.groupby("bin"):
            tm = 0.5 * (edges_r[b] + edges_r[b + 1])
            rates = dfi["rate_um_per_s_raw"].values.astype(float)
            med_raw = float(np.nanmedian(rates)) if len(rates) else np.nan
            # clipped median (helps when tiny segmentation jitter produces negative dR)
            med_clip = float(np.nanmedian(np.clip(rates, 0, None))) if len(rates) else np.nan
            out_rows.append((tm, len(rates), med_raw, med_clip))

        gr = pd.DataFrame(out_rows, columns=["time_ms", "n_steps", "median_um_per_s_raw", "median_um_per_s_clipped"])
        gr.to_csv(Path(args.out_dir) / "growth_rate_vs_time.csv", index=False)

        if args.make_plots:
            plt.figure()
            plt.plot(gr["time_ms"], gr["median_um_per_s_raw"], marker="o", label="median (raw)")
            plt.plot(gr["time_ms"], gr["median_um_per_s_clipped"], marker="o", label="median (clipped≥0)")
            plt.xlabel("time (ms)")
            plt.ylabel("growth rate (µm/s)")
            plt.title(f"Growth rate vs time (binned {bin_w:g} ms)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(Path(args.out_dir) / "growth_rate_vs_time.png", dpi=200)
            plt.close()
    else:
        # still write an empty file so pipelines don't crash
        pd.DataFrame(columns=["time_ms", "n_steps", "median_um_per_s_raw", "median_um_per_s_clipped"]).to_csv(
            Path(args.out_dir) / "growth_rate_vs_time.csv", index=False
        )

    # ---- Track summary (nice for quick sanity) ----
    # Provide one-row-per-grain summary including tau + slope if available
    summary = g.copy()

    if not tau_df.empty:
        summary = summary.merge(
            tau_df[["track_id", "tau_ms", "Rinf_um", "r2"]],
            on="track_id", how="left", suffixes=("", "_tau")
        )
    if not slopes.empty:
        summary = summary.merge(
            slopes[["track_id", "slope_um_per_s", "r2"]].rename(columns={"r2": "r2_slope"}),
            on="track_id", how="left"
        )

    summary = summary.sort_values(["t0_ms", "track_id"])
    summary.to_csv(Path(args.out_dir) / "track_summary.csv", index=False)

    print(f"[DONE] out_dir={args.out_dir}")
    print(f"       grains_kept={len(g)}  (min_track_frames={args.min_track_frames})")
    print(f"       time_range_ms=[{t_min:.2f}, {t_max:.2f}]")
    print(f"       early_window_ms={win:g}")
    print(f"       tau_fits={len(tau_df)}")


if __name__ == "__main__":
    main()
