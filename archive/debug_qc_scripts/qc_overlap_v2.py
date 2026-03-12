import pandas as pd
import numpy as np

# ---- edit path if needed ----
p = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\out\stable_nucleation_pair_overlapgate_sharedbins\FAPI_TEMPO_rebuild\nucleation_events_rejected_trackgate.csv"

df = pd.read_csv(p)
print("rows", len(df))
print("cols:", list(df.columns))

def pick_col(candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# common variants seen across your pipelines
time_col = pick_col([
    "nuc_time_ms", "t_ms", "time_ms", "time", "nuc_t_ms", "bin_center_ms"
])
overlap_col = pick_col([
    "overlap_prev", "overlap_prev_frac", "overlap_prev_max", "overlap_with_prev",
    "prev_overlap", "overlap"
])
r_col = pick_col([
    "R_nuc_px", "R_birth_px", "R0_px", "R_first_px", "R_px_birth", "R_px",
])
area_col = pick_col([
    "A_nuc_px", "A_birth_px", "area_nuc_px", "area_px_birth", "area_px"
])

print("\n[auto-detected]")
print("  time_col   =", time_col)
print("  overlap_col=", overlap_col)
print("  r_col      =", r_col)
print("  area_col   =", area_col)

# If we didn't find some, don't crash—just skip that part.
use_cols = [c for c in [time_col, overlap_col, r_col, area_col] if c is not None]
if not use_cols:
    raise SystemExit("Could not detect any useful numeric columns. Check the printed column list.")

# make sure numeric
for c in use_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

print("\n[describe]")
print(df[use_cols].describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]))

# overlap vs time bin
if (time_col is not None) and (overlap_col is not None):
    bin_ms = 50
    df2 = df[[time_col, overlap_col]].dropna().copy()
    df2["bin"] = (df2[time_col] // bin_ms) * bin_ms
    print(f"\n[overlap median by time bin ({bin_ms} ms)]")
    print(df2.groupby("bin")[overlap_col].median().head(30))

    # helpful “how extreme is it?”
    for thr in [0.5, 0.8, 0.95]:
        frac = (df2[overlap_col] > thr).mean()
        print(f"frac {overlap_col} > {thr}: {frac:.3f}")

else:
    print("\n(skip) Could not compute overlap-by-time-bin because time_col or overlap_col was not found.")
