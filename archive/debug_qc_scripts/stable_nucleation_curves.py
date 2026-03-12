# stable_nucleation_curves.py
# Compare "stable nucleation" curves between FAPI and FAPI-TEMPO
#
# Usage (Windows cmd):
#   cd /d "F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics"
#   python stable_nucleation_curves.py ^
#     --fapi_tracks  "F:\...\sam_cuda_vith_clean\FAPI\tracks.csv" ^
#     --tempo_tracks "F:\...\out\FAPI_TEMPO\tracks.csv" ^
#     --out_dir      "F:\...\sam_cuda_vith_clean\FAPI\stable_nucleation_compare_win60" ^
#     --L 5 --amin_px 800 --rmin_px 3 --bin_ms 20 --bootstrap_B 1000
#
# Notes:
# - "stable nucleation" event for a track is the FIRST frame where the track passes
#   area>=Amin and R>=Rmin for L consecutive frames.
# - This suppresses late bumps caused by short-lived artifacts / track fragmentation.

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------

FRAME_ID_RE = re.compile(r"frame_(\d+)_t([0-9.]+)ms", re.IGNORECASE)

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def parse_frame_number(row: pd.Series) -> Optional[int]:
    """Prefer numeric 'frame' column; else parse from string frame_id."""
    if "frame" in row and pd.notna(row["frame"]):
        try:
            return int(row["frame"])
        except Exception:
            pass
    if "frame_id" in row and isinstance(row["frame_id"], str):
        m = FRAME_ID_RE.search(row["frame_id"])
        if m:
            return int(m.group(1))
    return None

def parse_time_ms(row: pd.Series) -> Optional[float]:
    """Prefer numeric 'time_ms'; else parse from frame_id pattern."""
    if "time_ms" in row and pd.notna(row["time_ms"]):
        try:
            return float(row["time_ms"])
        except Exception:
            pass
    if "frame_id" in row and isinstance(row["frame_id"], str):
        m = FRAME_ID_RE.search(row["frame_id"])
        if m:
            return float(m.group(2))
    return None

def first_run_start(mask: np.ndarray, L: int) -> Optional[int]:
    """
    Return start index of first run of >=L True values in boolean array.
    """
    if L <= 1:
        idx = np.where(mask)[0]
        return int(idx[0]) if len(idx) else None

    # run length encoding style
    x = mask.astype(np.int32)
    # cumulative sum of consecutive truths
    consec = np.zeros_like(x)
    c = 0
    for i, v in enumerate(x):
        if v == 1:
            c += 1
        else:
            c = 0
        consec[i] = c
    hits = np.where(consec >= L)[0]
    if len(hits) == 0:
        return None
    end_i = int(hits[0])
    start_i = end_i - L + 1
    return start_i

def make_bins(times_ms: np.ndarray, bin_ms: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create bin edges and centers covering the full time range.
    """
    if len(times_ms) == 0:
        edges = np.array([0.0, bin_ms], dtype=float)
        centers = np.array([bin_ms / 2.0], dtype=float)
        return edges, centers

    tmin = 0.0
    tmax = float(np.nanmax(times_ms))
    nbins = int(np.ceil((tmax - tmin) / bin_ms))
    nbins = max(nbins, 1)
    edges = np.linspace(tmin, tmin + nbins * bin_ms, nbins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return edges, centers


# -----------------------------
# Core extraction
# -----------------------------

@dataclass
class NucleationResult:
    events: pd.DataFrame
    rejected: pd.DataFrame

def find_stable_nucleation_events(
    tracks: pd.DataFrame,
    L: int,
    amin_px: float,
    rmin_px: float,
) -> NucleationResult:
    """
    For each track_id, determine stable nucleation time using L-frame persistence
    on (area_px>=amin_px) & (R_px>=rmin_px).
    """
    required_cols = {"track_id", "area_px"}
    missing = [c for c in required_cols if c not in tracks.columns]
    if missing:
        raise KeyError(f"tracks.csv missing required columns: {missing}")

    # We expect frame/time columns, but we can derive from frame_id if present.
    if "frame" not in tracks.columns and "frame_id" not in tracks.columns:
        raise KeyError("tracks.csv must contain either 'frame' or 'frame_id' column.")
    if "time_ms" not in tracks.columns and "frame_id" not in tracks.columns:
        raise KeyError("tracks.csv must contain either 'time_ms' or 'frame_id' column.")
    if "R_px" not in tracks.columns:
        raise KeyError("tracks.csv missing 'R_px' column (needed for rmin_px gate).")

    # Build numeric frame/time if needed
    if "frame" not in tracks.columns or tracks["frame"].dtype == object:
        tracks = tracks.copy()
        tracks["_frame_num"] = tracks.apply(parse_frame_number, axis=1)
    else:
        tracks = tracks.copy()
        tracks["_frame_num"] = tracks["frame"].astype(int)

    if "time_ms" not in tracks.columns or tracks["time_ms"].dtype == object:
        tracks["_time_ms_num"] = tracks.apply(parse_time_ms, axis=1)
    else:
        tracks["_time_ms_num"] = tracks["time_ms"].astype(float)

    # Drop rows where we cannot recover ordering/time
    tracks = tracks.dropna(subset=["_frame_num", "_time_ms_num"]).copy()
    tracks["_frame_num"] = tracks["_frame_num"].astype(int)
    tracks["_time_ms_num"] = tracks["_time_ms_num"].astype(float)

    events_rows = []
    rejected_rows = []

    for tid, g in tracks.groupby("track_id"):
        g = g.sort_values(["_frame_num", "_time_ms_num"]).reset_index(drop=True)

        ok = (g["area_px"].astype(float) >= float(amin_px)) & (g["R_px"].astype(float) >= float(rmin_px))
        start = first_run_start(ok.to_numpy(), L=L)

        if start is None:
            rejected_rows.append({
                "track_id": int(tid),
                "reason": "no_stable_run",
                "n_points": int(len(g)),
                "min_area_px": float(g["area_px"].min()),
                "max_area_px": float(g["area_px"].max()),
                "min_R_px": float(g["R_px"].min()),
                "max_R_px": float(g["R_px"].max()),
                "first_time_ms": float(g["_time_ms_num"].iloc[0]),
                "last_time_ms": float(g["_time_ms_num"].iloc[-1]),
            })
            continue

        row0 = g.iloc[start]
        events_rows.append({
            "track_id": int(tid),
            "nuc_frame": int(row0["_frame_num"]),
            "nuc_time_ms": float(row0["_time_ms_num"]),
            "nuc_area_px": float(row0["area_px"]),
            "nuc_R_px": float(row0["R_px"]),
            "n_points": int(len(g)),
            "first_time_ms": float(g["_time_ms_num"].iloc[0]),
            "last_time_ms": float(g["_time_ms_num"].iloc[-1]),
            "max_area_px": float(g["area_px"].max()),
            "max_R_px": float(g["R_px"].max()),
        })

    events = pd.DataFrame(events_rows).sort_values("nuc_time_ms").reset_index(drop=True)
    rejected = pd.DataFrame(rejected_rows).sort_values("track_id").reset_index(drop=True)
    return NucleationResult(events=events, rejected=rejected)


# -----------------------------
# Curve building + bootstrap
# -----------------------------

def build_curves_from_events(events: pd.DataFrame, bin_ms: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    From nucleation events -> N(t) and dn/dt in bins.
    """
    times = events["nuc_time_ms"].to_numpy(dtype=float) if len(events) else np.array([], dtype=float)
    edges, centers = make_bins(times, bin_ms=bin_ms)

    counts, _ = np.histogram(times, bins=edges)
    cum = np.cumsum(counts)

    dn_dt = counts / (bin_ms / 1000.0)  # per second

    dfN = pd.DataFrame({
        "bin_center_ms": centers,
        "cum_n": cum,
        "n_nucleated": counts,
    })
    dfdn = pd.DataFrame({
        "bin_center_ms": centers,
        "n_nucleated": counts,
        "dn_dt_per_s": dn_dt,
        "cum_n": cum,
    })
    return dfN, dfdn

def bootstrap_ci(events: pd.DataFrame, bin_ms: float, B: int, seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bootstrap CI for N(t) and dn/dt.
    Returns dataframes with median + CI bands.
    """
    rng = np.random.default_rng(seed)
    n = len(events)

    baseN, basedn = build_curves_from_events(events, bin_ms)
    centers = baseN["bin_center_ms"].to_numpy()

    # Storage: (B, T)
    T = len(centers)
    boot_cum = np.zeros((B, T), dtype=float)
    boot_dn = np.zeros((B, T), dtype=float)

    if n == 0:
        # Return zeros with trivial CI
        outN = baseN.copy()
        outN["cum_n_med"] = outN["cum_n"]
        outN["cum_n_lo"] = outN["cum_n"]
        outN["cum_n_hi"] = outN["cum_n"]
        outdn = basedn.copy()
        outdn["dn_dt_med"] = outdn["dn_dt_per_s"]
        outdn["dn_dt_lo"] = outdn["dn_dt_per_s"]
        outdn["dn_dt_hi"] = outdn["dn_dt_per_s"]
        return outN, outdn

    for b in range(B):
        sample_idx = rng.integers(0, n, size=n)
        samp = events.iloc[sample_idx].reset_index(drop=True)
        dfN_b, dfdn_b = build_curves_from_events(samp, bin_ms)
        boot_cum[b, :] = dfN_b["cum_n"].to_numpy()
        boot_dn[b, :] = dfdn_b["dn_dt_per_s"].to_numpy()

    def q(a, p):  # quantile along axis 0
        return np.quantile(a, p, axis=0)

    outN = baseN.copy()
    outN["cum_n_med"] = q(boot_cum, 0.50)
    outN["cum_n_lo"]  = q(boot_cum, 0.025)
    outN["cum_n_hi"]  = q(boot_cum, 0.975)

    outdn = basedn.copy()
    outdn["dn_dt_med"] = q(boot_dn, 0.50)
    outdn["dn_dt_lo"]  = q(boot_dn, 0.025)
    outdn["dn_dt_hi"]  = q(boot_dn, 0.975)

    return outN, outdn


# -----------------------------
# Plotting
# -----------------------------

def plot_overlay_N(out_png: str, fapiN: pd.DataFrame, tempoN: pd.DataFrame) -> None:
    plt.figure()
    # FAPI
    plt.plot(fapiN["bin_center_ms"], fapiN["cum_n_med"], marker="o", label="FAPI")
    plt.fill_between(fapiN["bin_center_ms"], fapiN["cum_n_lo"], fapiN["cum_n_hi"], alpha=0.25)
    # TEMPO
    plt.plot(tempoN["bin_center_ms"], tempoN["cum_n_med"], marker="o", label="FAPI-TEMPO")
    plt.fill_between(tempoN["bin_center_ms"], tempoN["cum_n_lo"], tempoN["cum_n_hi"], alpha=0.25)

    plt.xlabel("time (ms)")
    plt.ylabel("cumulative nucleated grains")
    plt.title("Stable nucleation N(t) (bootstrap 95% CI)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_overlay_dn(out_png: str, fapidn: pd.DataFrame, tempodn: pd.DataFrame, bin_ms: float) -> None:
    plt.figure()
    plt.plot(fapidn["bin_center_ms"], fapidn["dn_dt_med"], marker="o", label="FAPI")
    plt.fill_between(fapidn["bin_center_ms"], fapidn["dn_dt_lo"], fapidn["dn_dt_hi"], alpha=0.25)

    plt.plot(tempodn["bin_center_ms"], tempodn["dn_dt_med"], marker="o", label="FAPI-TEMPO")
    plt.fill_between(tempodn["bin_center_ms"], tempodn["dn_dt_lo"], tempodn["dn_dt_hi"], alpha=0.25)

    plt.xlabel("time (ms)")
    plt.ylabel("dn/dt (1/s)")
    plt.title(f"Stable nucleation rate (bin={int(bin_ms)} ms, bootstrap 95% CI)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# -----------------------------
# Main
# -----------------------------

def write_methods(path: str, L: int, amin_px: float, rmin_px: float, bin_ms: float, B: int) -> None:
    txt = f"""Stable nucleation curve extraction (FAPI vs FAPI-TEMPO)

Input:
- tracks.csv per dataset (per-frame per-track measurements).

Stable nucleation event definition:
For each track_id, a nucleation event is assigned at the first frame where the track
satisfies BOTH:
  area_px >= {amin_px:g} px
  R_px    >= {rmin_px:g} px
for at least L = {L} consecutive frames.
The nucleation time t_nuc is the time_ms at the first frame of that first stable run.

This persistence+threshold rule suppresses short-lived segmentation artifacts and
track fragmentation that can create spurious late-time 'nucleation bumps'.

Binning:
Nucleation times are binned into uniform bins of width {bin_ms:g} ms.
dn/dt is computed as (counts per bin) / ({bin_ms:g} ms in seconds).

Uncertainty:
Bootstrap (B={B}) over tracks (resampling nucleation events with replacement) is used
to estimate 95% confidence intervals for:
- cumulative nucleation N(t)
- nucleation rate dn/dt
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fapi_tracks", required=True, help="Path to FAPI tracks.csv")
    ap.add_argument("--tempo_tracks", required=True, help="Path to FAPI-TEMPO tracks.csv")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--L", type=int, default=5)
    ap.add_argument("--amin_px", type=float, default=800.0)
    ap.add_argument("--rmin_px", type=float, default=3.0)
    ap.add_argument("--bin_ms", type=float, default=20.0)
    ap.add_argument("--bootstrap_B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    print("[OK] Reading:")
    print("  FAPI :", args.fapi_tracks)
    print("  TEMPO:", args.tempo_tracks)

    fapi = pd.read_csv(args.fapi_tracks)
    tempo = pd.read_csv(args.tempo_tracks)

    # Extract stable nucleation events
    fapi_res = find_stable_nucleation_events(fapi, args.L, args.amin_px, args.rmin_px)
    tempo_res = find_stable_nucleation_events(tempo, args.L, args.amin_px, args.rmin_px)

    # Write events + rejected
    fapi_events_path = os.path.join(args.out_dir, "nucleation_events_filtered_FAPI.csv")
    tempo_events_path = os.path.join(args.out_dir, "nucleation_events_filtered_FAPI_TEMPO.csv")
    fapi_rej_path = os.path.join(args.out_dir, "nucleation_events_rejected_FAPI.csv")
    tempo_rej_path = os.path.join(args.out_dir, "nucleation_events_rejected_FAPI_TEMPO.csv")

    fapi_res.events.to_csv(fapi_events_path, index=False)
    tempo_res.events.to_csv(tempo_events_path, index=False)
    fapi_res.rejected.to_csv(fapi_rej_path, index=False)
    tempo_res.rejected.to_csv(tempo_rej_path, index=False)

    # Curves + CI
    fapiN, fapidn = bootstrap_ci(fapi_res.events, args.bin_ms, args.bootstrap_B, seed=args.seed)
    tempoN, tempodn = bootstrap_ci(tempo_res.events, args.bin_ms, args.bootstrap_B, seed=args.seed + 1)

    fapiN.to_csv(os.path.join(args.out_dir, "N_t_filtered_FAPI.csv"), index=False)
    tempoN.to_csv(os.path.join(args.out_dir, "N_t_filtered_FAPI_TEMPO.csv"), index=False)
    fapidn.to_csv(os.path.join(args.out_dir, "dn_dt_filtered_FAPI.csv"), index=False)
    tempodn.to_csv(os.path.join(args.out_dir, "dn_dt_filtered_FAPI_TEMPO.csv"), index=False)

    # Plots
    plot_overlay_N(os.path.join(args.out_dir, "overlay_N_t_filtered.png"), fapiN, tempoN)
    plot_overlay_dn(os.path.join(args.out_dir, "overlay_dn_dt_filtered.png"), fapidn, tempodn, args.bin_ms)

    # Methods text
    write_methods(
        os.path.join(args.out_dir, "stable_nucleation_methods.txt"),
        args.L, args.amin_px, args.rmin_px, args.bin_ms, args.bootstrap_B
    )

    # Summary
    def summ(name, ev, rej):
        if len(ev) == 0:
            print(f"{name}: events=0 (rejected={len(rej)})")
            return
        print(
            f"{name}: events={len(ev)} (rejected={len(rej)}) | "
            f"t_nuc median={np.median(ev.nuc_time_ms):.2f} ms | "
            f"t_nuc IQR=({np.quantile(ev.nuc_time_ms,0.25):.2f}, {np.quantile(ev.nuc_time_ms,0.75):.2f})"
        )

    print("\n=== Stable nucleation summary ===")
    summ("FAPI", fapi_res.events, fapi_res.rejected)
    summ("FAPI-TEMPO", tempo_res.events, tempo_res.rejected)

    print("\n[OK] Wrote outputs to:")
    print(" ", args.out_dir)

if __name__ == "__main__":
    main()
