import pandas as pd

p = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\out\stable_nucleation_pair_overlapgate_sharedbins\FAPI_TEMPO_rebuild\nucleation_events_rejected_trackgate.csv"
df = pd.read_csv(p)

print("rows", len(df))
print(df[["nuc_time_ms", "overlap_prev", "R_nuc_px"]].describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]))

print("\noverlap_prev median by time bin (50 ms):")
df["bin"] = (df["nuc_time_ms"] // 50) * 50
print(df.groupby("bin")["overlap_prev"].median().head(20))
