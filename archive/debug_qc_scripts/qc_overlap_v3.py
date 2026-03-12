import pandas as pd
import numpy as np

p = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\out\stable_nucleation_pair_overlapgate_sharedbins\FAPI_TEMPO_rebuild\nucleation_events_rejected.csv"

df = pd.read_csv(p)
print("rows", len(df))
print("cols:", list(df.columns))

# expected columns from our rebuild:
# nuc_time_ms, overlap_prev, R_nuc_px, etc.
for c in ["nuc_time_ms", "overlap_prev", "R_nuc_px", "area_nuc_px", "reason"]:
    if c in df.columns:
        print(f"\n[{c}] head:")
        print(df[c].head())

if all(c in df.columns for c in ["nuc_time_ms", "overlap_prev", "R_nuc_px"]):
    print("\n[describe]")
    print(df[["nuc_time_ms","overlap_prev","R_nuc_px"]].describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]))

    print("\n[overlap median by time bin (50 ms)]")
    df2 = df[["nuc_time_ms","overlap_prev"]].dropna().copy()
    df2["bin"] = (df2["nuc_time_ms"] // 50) * 50
    print(df2.groupby("bin")["overlap_prev"].median().head(30))

    for thr in [0.5, 0.8, 0.95]:
        print(f"frac overlap_prev>{thr}:", (df2["overlap_prev"] > thr).mean())
else:
    print("\nMissing expected columns; paste the 'cols' list above.")
