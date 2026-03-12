import os
import pandas as pd
import numpy as np

# ============================================================
# Build matched FAPI kinetics files from retrack_cuda_vith outputs
# Produces:
#   kinetics_tau0p3_events_FAPI.csv
#   kinetics_tau0p3_Nt.csv
#   kinetics_tau0p3_rate.csv
#
# NOTE: "tau0p3" here is a matched filename convention for plotting.
# It does NOT imply bbox-IoU tau filtering unless you applied that to FAPI.
# ============================================================

# -----------------------------
# EDIT PATHS
# -----------------------------
base_retrack = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\out\FAPI\retrack_cuda_vith"

# Output folder (easy to point compare script to)
out_dir = os.path.join(base_retrack, "matched_kinetics_for_compare")
os.makedirs(out_dir, exist_ok=True)

# Inputs
bins_csv = os.path.join(base_retrack, "FAPI_nucleation_bins.csv")
track_summary_csv = os.path.join(base_retrack, "FAPI_track_summary.csv")

# -----------------------------
# Helpers
# -----------------------------
def ensure_exists(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")

def find_col(df, preferred_names):
    for c in preferred_names:
        if c in df.columns:
            return c
    return None

# -----------------------------
# Load nucleation bins -> Nt + rate
# -----------------------------
ensure_exists(bins_csv, "FAPI_nucleation_bins.csv")
bins = pd.read_csv(bins_csv)

# Expected columns in your file:
# bin_start_ms, bin_end_ms, bin_center_ms, nucleations,
# nucleation_rate_per_ms, cumulative_nucleations
required_bins = ["bin_center_ms", "nucleations", "nucleation_rate_per_ms", "cumulative_nucleations"]
missing_bins = [c for c in required_bins if c not in bins.columns]
if missing_bins:
    raise ValueError(f"Bins CSV missing columns {missing_bins}. Found: {list(bins.columns)}")

bins["bin_center_ms"] = pd.to_numeric(bins["bin_center_ms"], errors="coerce")
bins["nucleations"] = pd.to_numeric(bins["nucleations"], errors="coerce")
bins["nucleation_rate_per_ms"] = pd.to_numeric(bins["nucleation_rate_per_ms"], errors="coerce")
bins["cumulative_nucleations"] = pd.to_numeric(bins["cumulative_nucleations"], errors="coerce")

bins = bins.dropna(subset=["bin_center_ms"]).sort_values("bin_center_ms").reset_index(drop=True)

# Matched N(t)
Nt = pd.DataFrame({
    "t_ms": bins["bin_center_ms"].astype(float),
    "t_s": bins["bin_center_ms"].astype(float) / 1000.0,
    "N": bins["cumulative_nucleations"].fillna(method="ffill").fillna(0).astype(int)
})

# Matched dN/dt (convert per ms -> per s)
rate = pd.DataFrame({
    "t_center_ms": bins["bin_center_ms"].astype(float),
    "t_center_s": bins["bin_center_ms"].astype(float) / 1000.0,
    "dNdt_per_s": bins["nucleation_rate_per_ms"].astype(float) * 1000.0
})

# -----------------------------
# Build event list from track summary (births)
# -----------------------------
events = None
if os.path.exists(track_summary_csv):
    ts = pd.read_csv(track_summary_csv)

    track_col = find_col(ts, ["track_id"])
    birth_frame_col = find_col(ts, ["birth_frame", "birth_frame_i", "birth_idx"])
    birth_time_col = find_col(ts, ["birth_time_ms", "t_birth_ms"])

    if track_col is not None and birth_time_col is not None:
        ev = ts[[track_col, birth_time_col] + ([birth_frame_col] if birth_frame_col else [])].copy()
        rename_map = {track_col: "track_id", birth_time_col: "t_ms"}
        if birth_frame_col:
            rename_map[birth_frame_col] = "nuc_frame_i"
        ev = ev.rename(columns=rename_map)

        ev["track_id"] = pd.to_numeric(ev["track_id"], errors="coerce")
        ev["t_ms"] = pd.to_numeric(ev["t_ms"], errors="coerce")
        if "nuc_frame_i" in ev.columns:
            ev["nuc_frame_i"] = pd.to_numeric(ev["nuc_frame_i"], errors="coerce")
        else:
            # infer frame index at 2 ms/frame
            ev["nuc_frame_i"] = (ev["t_ms"] / 2.0).round()

        ev = ev.dropna(subset=["track_id", "t_ms", "nuc_frame_i"]).copy()
        ev["track_id"] = ev["track_id"].astype(int)
        ev["nuc_frame_i"] = ev["nuc_frame_i"].astype(int)
        ev["t_s"] = ev["t_ms"] / 1000.0

        # one event per track
        ev = (ev.sort_values(["track_id", "t_ms"])
                .drop_duplicates(subset=["track_id"], keep="first")
                .sort_values(["t_ms", "track_id"])
                .reset_index(drop=True))

        events = ev[["track_id", "nuc_frame_i", "t_ms", "t_s"]].copy()

# Fallback: reconstruct pseudo-events from bins only (if track_summary missing)
if events is None:
    # This preserves kinetics but not per-track identities.
    # We create synthetic track ids, one event per counted nucleation at bin center.
    rows = []
    tid = 0
    for _, r in bins.iterrows():
        n = int(r["nucleations"]) if pd.notna(r["nucleations"]) else 0
        t_ms = float(r["bin_center_ms"])
        frame_i = int(round(t_ms / 2.0))
        for _ in range(max(0, n)):
            rows.append({"track_id": tid, "nuc_frame_i": frame_i, "t_ms": t_ms, "t_s": t_ms / 1000.0})
            tid += 1
    events = pd.DataFrame(rows, columns=["track_id", "nuc_frame_i", "t_ms", "t_s"])

# -----------------------------
# Write outputs
# -----------------------------
events_out = os.path.join(out_dir, "kinetics_tau0p3_events_FAPI.csv")
Nt_out = os.path.join(out_dir, "kinetics_tau0p3_Nt.csv")
rate_out = os.path.join(out_dir, "kinetics_tau0p3_rate.csv")

events.to_csv(events_out, index=False)
Nt.to_csv(Nt_out, index=False)
rate.to_csv(rate_out, index=False)

print("[OK] Wrote:")
print(" ", events_out)
print(" ", Nt_out)
print(" ", rate_out)

print(f"[SUMMARY] events={len(events)}")
if len(events):
    print(f"[SUMMARY] onset={events['t_ms'].min():.1f} ms, last={events['t_ms'].max():.1f} ms")
if len(rate):
    p = rate.loc[rate['dNdt_per_s'].idxmax()]
    print(f"[SUMMARY] peak dN/dt={p['dNdt_per_s']:.1f}/s at t={p['t_center_ms']:.1f} ms")