import os
import math
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# FAPI vs FAPI-TEMPO comparison (win60, calibrated)
# - robust file detection
# - tau fallback for FAPI-TEMPO track_summary.csv:
#     tau_ms = t_end_ms - t_nuc_ms
# - growth parsing supports:
#     median_um_per_s_raw / median_um_per_s_clipped
#     and other common schemas
# - calibration fixed to 0.065 um/px for both
# ============================================================

# -------------------------
# USER PATHS (EDIT IF NEEDED)
# -------------------------
BASE_KINETICS = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics"

# FAPI matched kinetics (artifact-filter symmetric result you built)
FAPI_BASE = (
    r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics"
    r"\out\FAPI\retrack_cuda_vith\matched_kinetics_for_compare"
)

# FAPI-TEMPO artifact-filtered kinetics (tau=0.3)
FAPI_TEMPO_BASE = (
    r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics"
    r"\stable_v15_out_overlap03_poly_pad400"
)

# Extra sources (active tracks / tau / growth)
FAPI_EXTRA_DIRS = [
    r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\out\FAPI\retrack_cuda_vith",
    r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\sam_cuda_vith_clean\FAPI",
    r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\sam_cuda_vith_clean\FAPI\kinetics",  # <- growth here
]

FAPI_TEMPO_EXTRA_DIRS = [
    r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\out\FAPI_TEMPO",
    r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\sam_cuda_vith_clean\FAPI_TEMPO",
    r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\out\FAPI_TEMPO\idmap_kinetics_win60",  # <- growth here
]

OUT_DIR = (
    r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics"
    r"\out\stable_nucleation_compare_final\compare_FAPI_vs_FAPI_TEMPO_win60_calibrated"
)

# Calibration (same for both)
UM_PER_PX = 0.065

# Optional plot limits / knobs
TAU_HIST_MAX_MS = 5000.0
EVENT_HIST_BIN_MS = 20.0

# -------------------------
# Helpers
# -------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def maybe_read_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        if path and os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        pass
    return None


def find_existing_file(dirs: List[str], candidates: List[str]) -> Optional[str]:
    for d in dirs:
        if not d:
            continue
        for name in candidates:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
    return None


def pick_first_existing(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    if required:
        raise KeyError(f"Could not find any of columns {candidates}. Found: {list(df.columns)}")
    return None


def ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def save_csv(df: pd.DataFrame, out_path: str) -> None:
    df.to_csv(out_path, index=False)


def safe_label_dataset(name: str) -> str:
    return name.replace("/", "_")


# -------------------------
# Standardizers
# -------------------------
def standardize_rate(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    out = df.copy()

    t_col = pick_first_existing(out, ["t_center_s", "t_s", "t_center_ms", "time_ms"])
    r_col = pick_first_existing(out, ["dNdt_per_s", "dn_dt", "rate", "dndt"])

    out = ensure_numeric(out, [t_col, r_col])

    t = pd.to_numeric(out[t_col], errors="coerce")
    if "ms" in t_col.lower():
        t_s = t / 1000.0
        t_ms = t
    else:
        t_s = t
        t_ms = t * 1000.0

    rate = pd.to_numeric(out[r_col], errors="coerce")

    std = pd.DataFrame({
        "time_s": t_s,
        "time_ms": t_ms,
        "dn_dt_per_s": rate,
        "dataset": dataset_name
    }).dropna().sort_values("time_ms").reset_index(drop=True)

    return std


def standardize_Nt(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    out = df.copy()

    t_col = pick_first_existing(out, ["t_s", "t_ms", "time_ms"])
    n_col = pick_first_existing(out, ["N", "n", "cum_nucleation", "cumulative_nucleations"])

    out = ensure_numeric(out, [t_col, n_col])

    t = pd.to_numeric(out[t_col], errors="coerce")
    if "ms" in t_col.lower():
        t_s = t / 1000.0
        t_ms = t
    else:
        t_s = t
        t_ms = t * 1000.0

    N = pd.to_numeric(out[n_col], errors="coerce")

    std = pd.DataFrame({
        "time_s": t_s,
        "time_ms": t_ms,
        "N": N,
        "dataset": dataset_name
    }).dropna().sort_values("time_ms").reset_index(drop=True)

    # Ensure monotonic non-decreasing if tiny glitches exist
    std["N"] = std["N"].cummax()

    return std


def standardize_active(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    out = df.copy()

    # time
    t_col = pick_first_existing(out, ["time_ms", "t_ms", "time", "frame", "frame_idx"])
    # active count
    a_col = pick_first_existing(out, ["active_tracks", "active_grains", "active", "n_active"])

    out = ensure_numeric(out, [t_col, a_col])

    t = pd.to_numeric(out[t_col], errors="coerce")
    if "frame" in t_col.lower():
        # assume 2 ms/frame if frame index is provided
        t_ms = t * 2.0
    else:
        t_ms = t

    std = pd.DataFrame({
        "time_ms": t_ms,
        "active_count": pd.to_numeric(out[a_col], errors="coerce"),
        "dataset": dataset_name
    }).dropna().sort_values("time_ms").reset_index(drop=True)

    return std


def standardize_growth(df: pd.DataFrame, dataset_name: str, um_per_px: float) -> pd.DataFrame:
    """
    Supports multiple schemas, including:
    - time_ms, median_um_per_s_raw, median_um_per_s_clipped
    - time_ms, median_um_per_s
    - time_ms, growth_um_per_s
    - time_ms, median_px_per_s (converted)
    """
    out = df.copy()

    t_col = pick_first_existing(out, ["time_ms", "t_ms", "time", "t_center_ms", "t_center_s"], required=False)
    if t_col is None:
        raise KeyError(f"Could not find time column in growth CSV. Found: {list(out.columns)}")

    out = ensure_numeric(out, [t_col])

    t = pd.to_numeric(out[t_col], errors="coerce")
    if "s" in t_col.lower() and "ms" not in t_col.lower():
        t_ms = t * 1000.0
        t_s = t
    else:
        t_ms = t
        t_s = t / 1000.0

    # Preferred columns (your current issue)
    raw_um_col = pick_first_existing(out, ["median_um_per_s_raw"], required=False)
    clip_um_col = pick_first_existing(out, ["median_um_per_s_clipped"], required=False)

    if raw_um_col is not None or clip_um_col is not None:
        std = pd.DataFrame({
            "time_ms": t_ms,
            "time_s": t_s,
            "growth_um_per_s_raw": pd.to_numeric(out[raw_um_col], errors="coerce") if raw_um_col else np.nan,
            "growth_um_per_s_clipped": pd.to_numeric(out[clip_um_col], errors="coerce") if clip_um_col else np.nan,
            "dataset": dataset_name
        }).dropna(subset=["time_ms"]).sort_values("time_ms").reset_index(drop=True)
        return std

    # Generic single-column growth
    growth_um_candidates = [
        "median_um_per_s", "median_growth_um_per_s", "growth_um_per_s", "median_rate_um_per_s"
    ]
    growth_px_candidates = [
        "median_px_per_s", "median_growth_px_per_s", "growth_px_per_s", "median", "growth_rate", "rate"
    ]

    g_um_col = pick_first_existing(out, growth_um_candidates, required=False)
    g_px_col = pick_first_existing(out, growth_px_candidates, required=False)

    if g_um_col is not None:
        g_um = pd.to_numeric(out[g_um_col], errors="coerce")
    elif g_px_col is not None:
        g_um = pd.to_numeric(out[g_px_col], errors="coerce") * um_per_px
    else:
        raise KeyError(
            f"Could not find growth column. Found: {list(out.columns)}"
        )

    std = pd.DataFrame({
        "time_ms": t_ms,
        "time_s": t_s,
        "growth_um_per_s_raw": g_um,
        "growth_um_per_s_clipped": np.where(pd.notna(g_um), np.maximum(g_um, 0), np.nan),
        "dataset": dataset_name
    }).dropna(subset=["time_ms"]).sort_values("time_ms").reset_index(drop=True)

    return std


def standardize_tau(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    out = df.copy()

    # Primary tau columns
    tau_candidates = ["tau_ms", "tau", "tau_fit_ms", "tau_fit", "tau_value_ms"]
    tau_col = pick_first_existing(out, tau_candidates, required=False)

    if tau_col is not None:
        out = ensure_numeric(out, [tau_col])
        tau_vals = pd.to_numeric(out[tau_col], errors="coerce").dropna()

        # heuristic: if likely seconds and column not labeled ms, convert to ms
        if len(tau_vals) > 0:
            med = float(np.nanmedian(tau_vals.values))
            if med < 5.0 and "ms" not in tau_col.lower():
                tau_vals = tau_vals * 1000.0

        std = pd.DataFrame({
            "tau_ms": tau_vals.values,
            "tau_source": tau_col,
            "dataset": dataset_name
        })
        return std.reset_index(drop=True)

    # Fallback for track_summary.csv (FAPI-TEMPO):
    # tau_ms = t_end_ms - t_nuc_ms
    tnuc_col = pick_first_existing(out, ["t_nuc_ms", "tnuc_ms", "nuc_time_ms"], required=False)
    tend_col = pick_first_existing(out, ["t_end_ms", "tlast_ms", "t_stop_ms", "t_final_ms"], required=False)

    if tnuc_col is not None and tend_col is not None:
        out = ensure_numeric(out, [tnuc_col, tend_col])
        tau_vals = pd.to_numeric(out[tend_col], errors="coerce") - pd.to_numeric(out[tnuc_col], errors="coerce")
        tau_vals = tau_vals.replace([np.inf, -np.inf], np.nan)
        tau_vals = tau_vals[(tau_vals > 0) & pd.notna(tau_vals)]

        std = pd.DataFrame({
            "tau_ms": tau_vals.values,
            "tau_source": f"{tend_col}-{tnuc_col}",
            "dataset": dataset_name
        })
        return std.reset_index(drop=True)

    raise KeyError(
        f"Could not find tau columns or derivable tau columns. Found: {list(df.columns)}"
    )


def standardize_events(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    out = df.copy()
    t_col = pick_first_existing(out, ["t_ms", "nuc_time_ms", "time_ms", "t_s"], required=False)
    if t_col is None:
        raise KeyError(f"No event time column found. Found: {list(out.columns)}")

    out = ensure_numeric(out, [t_col])

    t = pd.to_numeric(out[t_col], errors="coerce")
    if t_col.endswith("_s") or t_col == "t_s":
        t_ms = t * 1000.0
    else:
        t_ms = t

    std = pd.DataFrame({
        "event_time_ms": t_ms,
        "dataset": dataset_name
    }).dropna().sort_values("event_time_ms").reset_index(drop=True)

    return std


# -------------------------
# Dataset loading
# -------------------------
def load_dataset_bundle(base_dir: str, dataset_name: str, extra_dirs: List[str]) -> Tuple[
    pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, str]
]:
    """
    Returns:
        rate_df, Nt_df, active_df, growth_df, tau_df, manifest
    """
    manifest: Dict[str, str] = {}
    search_dirs = [base_dir] + extra_dirs

    # Required compare curves
    rate_csv = find_existing_file([base_dir], ["kinetics_tau0p3_rate.csv"])
    Nt_csv = find_existing_file([base_dir], ["kinetics_tau0p3_Nt.csv"])

    if rate_csv is None:
        raise FileNotFoundError(f"Missing kinetics_tau0p3_rate.csv in {base_dir}")
    if Nt_csv is None:
        raise FileNotFoundError(f"Missing kinetics_tau0p3_Nt.csv in {base_dir}")

    manifest["rate_csv"] = rate_csv
    manifest["Nt_csv"] = Nt_csv

    # Events
    event_candidates = [
        f"kinetics_tau0p3_events_{dataset_name}.csv",
        "kinetics_tau0p3_events.csv",
        "nucleation_events_filtered.csv",
    ]
    events_csv = find_existing_file(search_dirs, event_candidates)
    if events_csv:
        manifest["events_csv"] = events_csv

    # Active tracks/grains
    active_candidates = [
        f"{dataset_name}_active_tracks.csv",
        "active_tracks.csv",
        "active_grains.csv",
        "FAPI_active_tracks.csv",
    ]
    active_csv = find_existing_file(search_dirs, active_candidates)
    if active_csv:
        manifest["active_csv"] = active_csv

    # Growth
    growth_candidates = [
        "growth_rate_vs_time.csv",
        f"{dataset_name}_growth_rate_vs_time.csv",
        "growth_per_frame_tau0p3.csv",
        "growth_vs_time.csv",
        "growth_rate.csv",
    ]
    growth_csv = find_existing_file(search_dirs, growth_candidates)
    if growth_csv:
        manifest["growth_csv"] = growth_csv

    # Tau
    tau_candidates = [
        f"{dataset_name}_tau_fits.csv",
        "tau_fits.csv",
        "track_summary.csv",   # FAPI-TEMPO fallback derivation
    ]
    tau_csv = find_existing_file(search_dirs, tau_candidates)
    if tau_csv:
        manifest["tau_csv"] = tau_csv

    # Read + standardize required
    raw_rate = pd.read_csv(rate_csv)
    raw_Nt = pd.read_csv(Nt_csv)
    rate = standardize_rate(raw_rate, dataset_name)
    Nt = standardize_Nt(raw_Nt, dataset_name)

    # Optional
    active = None
    growth = None
    tau = None

    if active_csv:
        try:
            active = standardize_active(pd.read_csv(active_csv), dataset_name)
        except Exception as e:
            print(f"[WARN] Could not standardize active table for {dataset_name}: {e}")

    if growth_csv:
        try:
            growth = standardize_growth(pd.read_csv(growth_csv), dataset_name, UM_PER_PX)
        except Exception as e:
            print(f"[WARN] Could not standardize growth table for {dataset_name}: {e}")

    if tau_csv:
        try:
            tau = standardize_tau(pd.read_csv(tau_csv), dataset_name)
        except Exception as e:
            print(f"[WARN] Could not standardize tau table for {dataset_name}: {e}")

    # Optional events standardized only if present
    if events_csv:
        try:
            events = standardize_events(pd.read_csv(events_csv), dataset_name)
            manifest["events_count"] = str(len(events))
        except Exception as e:
            print(f"[WARN] Could not standardize events table for {dataset_name}: {e}")

    return rate, Nt, active, growth, tau, manifest


# -------------------------
# Metrics / exports
# -------------------------
def summarize_dataset(rate: pd.DataFrame, Nt: pd.DataFrame, events_df: Optional[pd.DataFrame], dataset_name: str) -> Dict[str, object]:
    row: Dict[str, object] = {"dataset": dataset_name}

    if not Nt.empty:
        row["N_final"] = int(pd.to_numeric(Nt["N"], errors="coerce").dropna().iloc[-1])
        row["onset_ms"] = float(Nt["time_ms"].iloc[0])
        row["t_last_ms"] = float(Nt["time_ms"].iloc[-1])

        Nf = row["N_final"]
        if Nf > 0:
            for frac, key in [(0.1, "t10_ms"), (0.5, "t50_ms"), (0.9, "t90_ms")]:
                target = math.ceil(frac * Nf)
                hit = Nt.loc[pd.to_numeric(Nt["N"], errors="coerce") >= target]
                row[key] = float(hit["time_ms"].iloc[0]) if not hit.empty else np.nan
        else:
            row["t10_ms"] = row["t50_ms"] = row["t90_ms"] = np.nan
    else:
        row["N_final"] = row["onset_ms"] = row["t_last_ms"] = np.nan
        row["t10_ms"] = row["t50_ms"] = row["t90_ms"] = np.nan

    if not rate.empty:
        idx = pd.to_numeric(rate["dn_dt_per_s"], errors="coerce").idxmax()
        row["peak_dNdt_per_s"] = float(rate.loc[idx, "dn_dt_per_s"])
        row["t_peak_ms"] = float(rate.loc[idx, "time_ms"])
    else:
        row["peak_dNdt_per_s"] = np.nan
        row["t_peak_ms"] = np.nan

    if events_df is not None:
        row["n_events"] = int(len(events_df))
    return row


def align_curves_on_grid(df: pd.DataFrame, time_col: str, value_col: str, dataset_col: str = "dataset") -> pd.DataFrame:
    """
    Outer join by rounded ms times for simple overlay CSV export.
    """
    out_parts = []
    for ds, g in df.groupby(dataset_col):
        h = g[[time_col, value_col]].copy().dropna().sort_values(time_col)
        h = h.drop_duplicates(subset=[time_col], keep="last")
        h = h.rename(columns={value_col: f"{value_col}_{ds}"})
        out_parts.append(h)

    if not out_parts:
        return pd.DataFrame()

    merged = out_parts[0]
    for nxt in out_parts[1:]:
        merged = pd.merge(merged, nxt, on=time_col, how="outer")
    return merged.sort_values(time_col).reset_index(drop=True)


def build_event_hist_from_events(events: pd.DataFrame, dataset_name: str, bin_ms: float = 20.0) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame(columns=["time_ms", "dn_dt_per_s", "dataset"])

    e = events["event_time_ms"].dropna().to_numpy()
    if len(e) == 0:
        return pd.DataFrame(columns=["time_ms", "dn_dt_per_s", "dataset"])

    tmin = 0.0
    tmax = float(np.nanmax(e))
    edges = np.arange(tmin, tmax + bin_ms + 1e-9, bin_ms)
    if len(edges) < 2:
        edges = np.array([0.0, bin_ms])

    counts, edges = np.histogram(e, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rate = counts / (bin_ms / 1000.0)

    return pd.DataFrame({
        "time_ms": centers,
        "dn_dt_per_s": rate,
        "count_in_bin": counts,
        "dataset": dataset_name
    })


# -------------------------
# Plotters
# -------------------------
def base_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
    })


def plot_dn_dt(rate_all: pd.DataFrame, out_png: str):
    plt.figure(figsize=(8.5, 5.5))
    for ds, g in rate_all.groupby("dataset"):
        g = g.sort_values("time_ms")
        n_label = ""
        plt.plot(g["time_ms"], g["dn_dt_per_s"], marker="o", label=f"{ds}{n_label}")
    plt.xlabel("time (ms)")
    plt.ylabel("dn/dt (1/s)")
    plt.title("Nucleation rate dN/dt")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_Nt(Nt_all: pd.DataFrame, out_png: str):
    plt.figure(figsize=(8.5, 5.5))
    for ds, g in Nt_all.groupby("dataset"):
        g = g.sort_values("time_ms")
        plt.step(g["time_ms"], g["N"], where="post", label=ds)
    plt.xlabel("time (ms)")
    plt.ylabel("cumulative nucleation N(t)")
    plt.title("Cumulative nucleation N(t)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_Nt_normalized(Nt_all: pd.DataFrame, out_png: str):
    plt.figure(figsize=(8.5, 5.5))
    for ds, g in Nt_all.groupby("dataset"):
        g = g.sort_values("time_ms").copy()
        nmax = pd.to_numeric(g["N"], errors="coerce").max()
        if pd.isna(nmax) or nmax <= 0:
            continue
        g["N_norm"] = pd.to_numeric(g["N"], errors="coerce") / nmax
        plt.step(g["time_ms"], g["N_norm"], where="post", label=ds)
    plt.xlabel("time (ms)")
    plt.ylabel("N(t) / N_final")
    plt.title("Normalized cumulative nucleation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_active(active_all: Optional[pd.DataFrame], out_png: str):
    if active_all is None or active_all.empty:
        return
    plt.figure(figsize=(8.5, 5.5))
    for ds, g in active_all.groupby("dataset"):
        g = g.sort_values("time_ms")
        plt.plot(g["time_ms"], g["active_count"], marker="o", label=ds)
    plt.xlabel("time (ms)")
    plt.ylabel("active grains")
    plt.title("Active grains vs time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_growth(growth_all: Optional[pd.DataFrame], out_png: str):
    if growth_all is None or growth_all.empty:
        return

    plt.figure(figsize=(8.5, 5.5))
    for ds, g in growth_all.groupby("dataset"):
        g = g.sort_values("time_ms")
        # prefer clipped if available and not all-NaN; else raw
        use_col = None
        if "growth_um_per_s_clipped" in g.columns and g["growth_um_per_s_clipped"].notna().any():
            use_col = "growth_um_per_s_clipped"
        elif "growth_um_per_s_raw" in g.columns and g["growth_um_per_s_raw"].notna().any():
            use_col = "growth_um_per_s_raw"

        if use_col is None:
            continue

        plt.plot(g["time_ms"], g[use_col], marker="o", label=ds)

    plt.xlabel("time (ms)")
    plt.ylabel("growth rate (µm/s)")
    plt.title(f"Median growth rate vs time (calibration = {UM_PER_PX:.3f} µm/px)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_tau_hist(tau_all: Optional[pd.DataFrame], out_png: str):
    if tau_all is None or tau_all.empty:
        return

    plt.figure(figsize=(8.5, 5.5))
    for ds, g in tau_all.groupby("dataset"):
        vals = pd.to_numeric(g["tau_ms"], errors="coerce")
        vals = vals[(vals > 0) & (vals <= TAU_HIST_MAX_MS)].dropna()
        if len(vals) == 0:
            continue
        plt.hist(vals, bins=40, alpha=0.45, label=ds)
    plt.xlabel("tau (ms)")
    plt.ylabel("count")
    plt.title(f"Tau distribution (tau<={int(TAU_HIST_MAX_MS)} ms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_event_hist(events_hist_all: Optional[pd.DataFrame], out_png: str):
    if events_hist_all is None or events_hist_all.empty:
        return
    plt.figure(figsize=(8.5, 5.5))
    for ds, g in events_hist_all.groupby("dataset"):
        g = g.sort_values("time_ms")
        plt.plot(g["time_ms"], g["count_in_bin"], marker="o", label=ds)
    plt.xlabel("time (ms)")
    plt.ylabel(f"events / {EVENT_HIST_BIN_MS:.0f} ms bin")
    plt.title(f"Event histogram (bin={EVENT_HIST_BIN_MS:.0f} ms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_event_hist_rate(events_hist_all: Optional[pd.DataFrame], out_png: str):
    if events_hist_all is None or events_hist_all.empty:
        return
    plt.figure(figsize=(8.5, 5.5))
    for ds, g in events_hist_all.groupby("dataset"):
        g = g.sort_values("time_ms")
        plt.plot(g["time_ms"], g["dn_dt_per_s"], marker="o", label=ds)
    plt.xlabel("time (ms)")
    plt.ylabel("dn/dt (1/s)")
    plt.title(f"Event-derived nucleation rate (bin={EVENT_HIST_BIN_MS:.0f} ms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# -------------------------
# Main
# -------------------------
def main():
    base_style()
    ensure_dir(OUT_DIR)

    print("[INFO] Loading datasets...")

    # Load FAPI
    rate_fapi, Nt_fapi, active_fapi, growth_fapi, tau_fapi, man_fapi = load_dataset_bundle(
        FAPI_BASE, "FAPI", FAPI_EXTRA_DIRS
    )

    # Load FAPI-TEMPO
    rate_tempo, Nt_tempo, active_tempo, growth_tempo, tau_tempo, man_tempo = load_dataset_bundle(
        FAPI_TEMPO_BASE, "FAPI-TEMPO", FAPI_TEMPO_EXTRA_DIRS
    )

    # Print manifest summary
    print(f"[INFO] FAPI base       = {FAPI_BASE}")
    print(f"[INFO] FAPI-TEMPO base = {FAPI_TEMPO_BASE}")
    print("[INFO] Files used:")
    print("  [FAPI]")
    for k, v in man_fapi.items():
        if k == "events_count":
            continue
        print(f"    {k}: {v}")
    print("  [FAPI-TEMPO]")
    for k, v in man_tempo.items():
        if k == "events_count":
            continue
        print(f"    {k}: {v}")

    # Optional events for hist plots / metrics
    events_fapi = None
    events_tempo = None
    if "events_csv" in man_fapi and os.path.exists(man_fapi["events_csv"]):
        try:
            events_fapi = standardize_events(pd.read_csv(man_fapi["events_csv"]), "FAPI")
        except Exception as e:
            print(f"[WARN] Failed to load/standardize FAPI events: {e}")
    if "events_csv" in man_tempo and os.path.exists(man_tempo["events_csv"]):
        try:
            events_tempo = standardize_events(pd.read_csv(man_tempo["events_csv"]), "FAPI-TEMPO")
        except Exception as e:
            print(f"[WARN] Failed to load/standardize FAPI-TEMPO events: {e}")

    # Combine overlays
    rate_all = pd.concat([rate_fapi, rate_tempo], ignore_index=True)
    Nt_all = pd.concat([Nt_fapi, Nt_tempo], ignore_index=True)

    active_all = None
    if active_fapi is not None or active_tempo is not None:
        active_parts = [x for x in [active_fapi, active_tempo] if x is not None and not x.empty]
        if active_parts:
            active_all = pd.concat(active_parts, ignore_index=True)

    growth_all = None
    if growth_fapi is not None or growth_tempo is not None:
        growth_parts = [x for x in [growth_fapi, growth_tempo] if x is not None and not x.empty]
        if growth_parts:
            growth_all = pd.concat(growth_parts, ignore_index=True)

    tau_all = None
    if tau_fapi is not None or tau_tempo is not None:
        tau_parts = [x for x in [tau_fapi, tau_tempo] if x is not None and not x.empty]
        if tau_parts:
            tau_all = pd.concat(tau_parts, ignore_index=True)

    events_hist_all = None
    ev_parts = []
    if events_fapi is not None and not events_fapi.empty:
        ev_parts.append(build_event_hist_from_events(events_fapi, "FAPI", EVENT_HIST_BIN_MS))
    if events_tempo is not None and not events_tempo.empty:
        ev_parts.append(build_event_hist_from_events(events_tempo, "FAPI-TEMPO", EVENT_HIST_BIN_MS))
    if ev_parts:
        events_hist_all = pd.concat(ev_parts, ignore_index=True)

    # -------------------------
    # Save overlay CSVs
    # -------------------------
    save_csv(rate_all.sort_values(["dataset", "time_ms"]), os.path.join(OUT_DIR, "dn_dt_overlay.csv"))
    save_csv(Nt_all.sort_values(["dataset", "time_ms"]), os.path.join(OUT_DIR, "Nt_overlay.csv"))

    if active_all is not None and not active_all.empty:
        save_csv(active_all.sort_values(["dataset", "time_ms"]), os.path.join(OUT_DIR, "active_overlay.csv"))
    if growth_all is not None and not growth_all.empty:
        save_csv(growth_all.sort_values(["dataset", "time_ms"]), os.path.join(OUT_DIR, "growth_overlay.csv"))
    if tau_all is not None and not tau_all.empty:
        save_csv(tau_all.sort_values(["dataset", "tau_ms"]), os.path.join(OUT_DIR, "tau_overlay.csv"))
    if events_hist_all is not None and not events_hist_all.empty:
        save_csv(events_hist_all.sort_values(["dataset", "time_ms"]), os.path.join(OUT_DIR, "events_overlay.csv"))

    # aligned overlays
    aligned_dn = align_curves_on_grid(rate_all, "time_ms", "dn_dt_per_s")
    save_csv(aligned_dn, os.path.join(OUT_DIR, "aligned_dn_dt_overlay.csv"))

    aligned_N = align_curves_on_grid(Nt_all, "time_ms", "N")
    save_csv(aligned_N, os.path.join(OUT_DIR, "aligned_Nt_overlay.csv"))

    # -------------------------
    # Summary metrics
    # -------------------------
    metrics_rows = []
    metrics_rows.append(summarize_dataset(rate_fapi, Nt_fapi, events_fapi, "FAPI"))
    metrics_rows.append(summarize_dataset(rate_tempo, Nt_tempo, events_tempo, "FAPI-TEMPO"))
    summary_df = pd.DataFrame(metrics_rows)
    save_csv(summary_df, os.path.join(OUT_DIR, "compare_summary_metrics_FAPI_vs_FAPI-TEMPO.csv"))

    # input manifest
    manifest_rows = []
    for k, v in man_fapi.items():
        manifest_rows.append({"dataset": "FAPI", "key": k, "value": v})
    for k, v in man_tempo.items():
        manifest_rows.append({"dataset": "FAPI-TEMPO", "key": k, "value": v})
    manifest_df = pd.DataFrame(manifest_rows)
    save_csv(manifest_df, os.path.join(OUT_DIR, "compare_input_manifest.csv"))

    # also export per-dataset standardized curves
    save_csv(rate_fapi, os.path.join(OUT_DIR, "rate_curve_FAPI.csv"))
    save_csv(rate_tempo, os.path.join(OUT_DIR, "rate_curve_FAPI-TEMPO.csv"))
    save_csv(Nt_fapi, os.path.join(OUT_DIR, "Nt_curve_FAPI.csv"))
    save_csv(Nt_tempo, os.path.join(OUT_DIR, "Nt_curve_FAPI-TEMPO.csv"))

    # -------------------------
    # Figures
    # -------------------------
    plot_dn_dt(rate_all, os.path.join(OUT_DIR, "compare_dn_dt_FAPI_vs_FAPI-TEMPO.png"))
    plot_Nt(Nt_all, os.path.join(OUT_DIR, "compare_Nt_FAPI_vs_FAPI-TEMPO.png"))
    plot_Nt_normalized(Nt_all, os.path.join(OUT_DIR, "compare_Nt_normalized_FAPI_vs_FAPI-TEMPO.png"))

    if active_all is not None and not active_all.empty:
        plot_active(active_all, os.path.join(OUT_DIR, "compare_active_grains_FAPI_vs_FAPI-TEMPO.png"))
    else:
        print("[WARN] No active-track data; skipping active plot.")

    if growth_all is not None and not growth_all.empty:
        plot_growth(growth_all, os.path.join(OUT_DIR, "compare_growth_rate_FAPI_vs_FAPI-TEMPO.png"))
    else:
        print("[WARN] No growth-rate data; skipping growth plot.")

    if tau_all is not None and not tau_all.empty:
        plot_tau_hist(tau_all, os.path.join(OUT_DIR, "compare_tau_hist_FAPI_vs_FAPI-TEMPO.png"))
    else:
        print("[WARN] No tau data; skipping tau histogram.")

    if events_hist_all is not None and not events_hist_all.empty:
        plot_event_hist(events_hist_all, os.path.join(OUT_DIR, "compare_event_hist_FAPI_vs_FAPI-TEMPO.png"))
        plot_event_hist_rate(events_hist_all, os.path.join(OUT_DIR, "compare_event_hist_rate_FAPI_vs_FAPI-TEMPO.png"))
    else:
        print("[WARN] No event files; skipping event hist plots.")

    # -------------------------
    # Final log
    # -------------------------
    print("[OK] Wrote outputs to:")
    print(f"  {OUT_DIR}")
    print("[OK] Main figures (if data available):")
    print("  - compare_dn_dt_FAPI_vs_FAPI-TEMPO.png")
    print("  - compare_Nt_FAPI_vs_FAPI-TEMPO.png")
    print("  - compare_Nt_normalized_FAPI_vs_FAPI-TEMPO.png")
    print("  - compare_active_grains_FAPI_vs_FAPI-TEMPO.png")
    print("  - compare_growth_rate_FAPI_vs_FAPI-TEMPO.png")
    print("  - compare_tau_hist_FAPI_vs_FAPI-TEMPO.png")
    print("  - compare_event_hist_FAPI_vs_FAPI-TEMPO.png")
    print("  - compare_event_hist_rate_FAPI_vs_FAPI-TEMPO.png")
    print("[OK] Tables:")
    print("  - compare_input_manifest.csv")
    print("  - compare_summary_metrics_FAPI_vs_FAPI-TEMPO.csv")
    print("  - dn_dt_overlay.csv")
    print("  - Nt_overlay.csv")
    print("  - active_overlay.csv")
    print("  - growth_overlay.csv")
    print("  - tau_overlay.csv")
    print("  - events_overlay.csv")
    print("  - aligned_dn_dt_overlay.csv")
    print("  - aligned_Nt_overlay.csv")


if __name__ == "__main__":
    main()