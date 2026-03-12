# save as: make_filtered_tau_and_overlay_v2.py
# run: python make_filtered_tau_and_overlay_v2.py
#
# Creates:
#   - tau_filtered_FAPI.csv
#   - tau_filtered_FAPI_TEMPO.csv
#   - overlay_tau_hist_filtered.png
#   - tau_gating_methods.txt
#
# Adds an extra PHYSICS gate:
#   Rinf_um / Rmax_um <= RINF_OVER_RMAX_MAX
# which removes non-saturating / extrapolation-driven tau outliers.

import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------- INPUT PATHS (edit if needed) ----------
FAPI_DIR  = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\sam_cuda_vith_clean\FAPI\kinetics"
TEMPO_DIR = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\out\FAPI_TEMPO\idmap_kinetics_win60"

# Prefer tau_fits.csv if present; otherwise fall back to track_summary.csv
FAPI_FILE_CANDIDATES  = ["tau_fits.csv", "track_summary.csv"]
TEMPO_FILE_CANDIDATES = ["tau_fits.csv", "track_summary.csv"]

# Output folder (created under ...\sam_cuda_vith_clean\FAPI\ )
OUT_DIR = os.path.join(os.path.dirname(FAPI_DIR), "tau_filtered_compare_win60_v2")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- GATING (defaults) ----------
R2_MIN = 0.40
MIN_POINTS = 20
TAU_MAX_MS = 5000

# Optional: set a minimum tau to avoid sub-frame tau (set to None to disable)
TAU_MIN_MS = None  # e.g. 6 if dt=2 ms and you want tau>=3 frames

# Physics gate to reject non-saturating extrapolations:
RINF_OVER_RMAX_MAX = 1.5   # <=2.0 also reasonable; 1.5 is stricter

# ---------- HELPERS ----------
def find_existing(dirpath, candidates):
    for fn in candidates:
        p = os.path.join(dirpath, fn)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of {candidates} found in: {dirpath}")

def load_tau_table(path):
    df = pd.read_csv(path)

    # Normalize column names across possible tables
    rename = {}
    if "R2" in df.columns and "r2" not in df.columns:
        rename["R2"] = "r2"
    if "tau" in df.columns and "tau_ms" not in df.columns:
        rename["tau"] = "tau_ms"
    if "n_frames" in df.columns and "n_points" not in df.columns:
        rename["n_frames"] = "n_points"
    if "frames" in df.columns and "n_points" not in df.columns:
        rename["frames"] = "n_points"
    df = df.rename(columns=rename)

    # Ensure required columns exist
    if "track_id" not in df.columns or "tau_ms" not in df.columns:
        raise ValueError(
            f"{os.path.basename(path)} must contain at least track_id and tau_ms. "
            f"Columns found: {list(df.columns)}"
        )

    # Create missing optional columns so gating fails gracefully (rather than crashing)
    if "r2" not in df.columns:
        df["r2"] = float("nan")
    if "n_points" not in df.columns:
        df["n_points"] = float("nan")

    # If present in tau_fits.csv, keep these for physics gate
    for col in ["Rinf_um", "Rmax_um"]:
        if col not in df.columns:
            df[col] = float("nan")

    return df

def apply_gates(df):
    out = df.copy()

    # Coerce numeric columns
    for c in ["tau_ms", "r2", "n_points", "Rinf_um", "Rmax_um"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    mask = pd.Series(True, index=out.index)

    # Basic validity
    mask &= out["tau_ms"].notna()
    mask &= out["r2"].notna()
    mask &= out["n_points"].notna()

    # Primary gates
    mask &= out["tau_ms"] <= TAU_MAX_MS
    mask &= out["r2"] >= R2_MIN
    mask &= out["n_points"] >= MIN_POINTS

    if TAU_MIN_MS is not None:
        mask &= out["tau_ms"] >= TAU_MIN_MS

    # Physics gate: require reasonable saturation extrapolation if columns are available
    # If Rinf_um/Rmax_um are missing (NaN), the mask will reject those rows (good: no silent misuse)
    ratio = out["Rinf_um"] / out["Rmax_um"]
    out["Rinf_over_Rmax"] = ratio
    mask &= ratio.notna()
    mask &= ratio > 0
    mask &= ratio <= RINF_OVER_RMAX_MAX

    return out.loc[mask].copy()

def summarize(df, label):
    if len(df) == 0:
        return f"{label}: no rows after filtering."
    q = df["tau_ms"].quantile([0.25, 0.5, 0.75]).to_dict()
    return (
        f"{label}: n={len(df)} | "
        f"tau_ms median={q[0.5]:.3g}, IQR=({q[0.25]:.3g}, {q[0.75]:.3g}) | "
        f"r2 median={df['r2'].median():.3g} | "
        f"n_points median={df['n_points'].median():.3g} | "
        f"Rinf/Rmax median={df['Rinf_over_Rmax'].median():.3g}"
    )

# ---------- LOAD ----------
fapi_path  = find_existing(FAPI_DIR,  FAPI_FILE_CANDIDATES)
tempo_path = find_existing(TEMPO_DIR, TEMPO_FILE_CANDIDATES)

df_fapi_raw  = load_tau_table(fapi_path)
df_tempo_raw = load_tau_table(tempo_path)

df_fapi_filt  = apply_gates(df_fapi_raw)
df_tempo_filt = apply_gates(df_tempo_raw)

# ---------- WRITE FILTERED CSVs ----------
out_fapi_csv  = os.path.join(OUT_DIR, "tau_filtered_FAPI.csv")
out_tempo_csv = os.path.join(OUT_DIR, "tau_filtered_FAPI_TEMPO.csv")
df_fapi_filt.to_csv(out_fapi_csv, index=False)
df_tempo_filt.to_csv(out_tempo_csv, index=False)

# ---------- OVERLAY HISTOGRAM ----------
plt.figure()
plt.hist(df_fapi_filt["tau_ms"],  bins=30, alpha=0.6, label="FAPI")
plt.hist(df_tempo_filt["tau_ms"], bins=30, alpha=0.6, label="FAPI-TEMPO")
plt.xlabel("tau (ms)")
plt.ylabel("count")

title = (
    f"Tau distribution (filtered: r2>={R2_MIN}, n>={MIN_POINTS}, "
    f"tau<={TAU_MAX_MS} ms, Rinf/Rmax<={RINF_OVER_RMAX_MAX}"
)
if TAU_MIN_MS is not None:
    title += f", tau>={TAU_MIN_MS} ms"
title += ")"
plt.title(title)
plt.legend()
plt.tight_layout()

hist_path = os.path.join(OUT_DIR, "overlay_tau_hist_filtered.png")
plt.savefig(hist_path, dpi=200)
plt.close()

# ---------- METHODS TEXT ----------
methods = f"""Tau-fit filtering (“gating”) for robust comparison
------------------------------------------------------------
We fitted each grain’s radius–time trajectory with the same τ-model used in the kinetics pipeline.
Because τ becomes unstable when growth does not approach saturation (or when fit quality is poor),
we applied objective gates before comparing τ distributions between datasets.

Filtering criteria (applied identically to both datasets):
  1) Fit-quality threshold: R² ≥ {R2_MIN}
  2) Minimum number of post-nucleation timepoints: n_points ≥ {MIN_POINTS}
  3) Outlier exclusion for non-saturating fits: τ ≤ {TAU_MAX_MS} ms
  4) Saturation plausibility gate: R∞/Rmax ≤ {RINF_OVER_RMAX_MAX}
"""
if TAU_MIN_MS is not None:
    methods += f"  5) Resolution-floor exclusion: τ ≥ {TAU_MIN_MS} ms\n"

methods += f"""
Outputs:
  - {out_fapi_csv}
  - {out_tempo_csv}
  - {hist_path}

Summary after filtering:
  - {summarize(df_fapi_filt, "FAPI")}
  - {summarize(df_tempo_filt, "FAPI-TEMPO")}
"""

methods_path = os.path.join(OUT_DIR, "tau_gating_methods.txt")
with open(methods_path, "w", encoding="utf-8") as f:
    f.write(methods)

print("[OK] Inputs:")
print("  FAPI :", fapi_path)
print("  TEMPO:", tempo_path)
print()
print("[OK] Wrote:")
print("  -", out_fapi_csv)
print("  -", out_tempo_csv)
print("  -", hist_path)
print("  -", methods_path)
print()
print("=== Summary ===")
print(summarize(df_fapi_filt, "FAPI"))
print(summarize(df_tempo_filt, "FAPI-TEMPO"))
