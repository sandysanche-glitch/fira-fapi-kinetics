import os
import numpy as np
import pandas as pd

# ----------------------------
# INPUTS (edit if needed)
# ----------------------------
FAPI_TAU = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\sam_cuda_vith_clean\FAPI\kinetics\tau_fits.csv"
TEMPO_TAU = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\out\FAPI_TEMPO\idmap_kinetics_win60\tau_fits.csv"

OUT_DIR = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\sam_cuda_vith_clean\FAPI\tau_filtered_compare_win60_v2"
OUT_CSV = os.path.join(OUT_DIR, "tau_gate_sweep_summary.csv")

# ----------------------------
# BASE GATES (match v2)
# ----------------------------
TAU_MAX_MS = 5000.0
NPOINTS_MIN = 20
R2_SWEEP = [0.3, 0.4, 0.5, 0.6]
RINF_OVER_RMAX_SWEEP = [1.2, 1.5, 2.0]

# If your tau_fits has different column names, map them here:
# expected: tau_ms, r2, n_points, Rinf_over_Rmax  (or Rinf_over_Rmax derived from Rinf_px/Rmax_px)
COL_TAU = "tau_ms"
COL_R2 = "r2"
COL_N = "n_points"

# We'll accept any of these as the "Rinf/Rmax" column
RATIO_COL_CANDIDATES = ["Rinf_over_Rmax", "Rinf_Rmax", "Rinf_over_Rmax_fit", "Rinf_over_Rmax_ratio", "Rinf_div_Rmax"]

def infer_ratio_column(df: pd.DataFrame) -> str:
    for c in RATIO_COL_CANDIDATES:
        if c in df.columns:
            return c
    # Try to build from Rinf and Rmax if present
    if "Rinf_px" in df.columns and "Rmax_px" in df.columns:
        df["Rinf_over_Rmax"] = df["Rinf_px"] / df["Rmax_px"].replace(0, np.nan)
        return "Rinf_over_Rmax"
    if "R_inf_px" in df.columns and "R_max_px" in df.columns:
        df["Rinf_over_Rmax"] = df["R_inf_px"] / df["R_max_px"].replace(0, np.nan)
        return "Rinf_over_Rmax"
    raise KeyError(
        "Could not find or derive Rinf/Rmax. "
        "Looked for columns: %s, or (Rinf_px & Rmax_px)." % RATIO_COL_CANDIDATES
    )

def describe_tau(df: pd.DataFrame):
    if len(df) == 0:
        return dict(n=0, med=np.nan, q25=np.nan, q75=np.nan, r2med=np.nan, nmed=np.nan, rrmed=np.nan)
    return dict(
        n=int(len(df)),
        med=float(np.nanmedian(df[COL_TAU])),
        q25=float(np.nanpercentile(df[COL_TAU], 25)),
        q75=float(np.nanpercentile(df[COL_TAU], 75)),
        r2med=float(np.nanmedian(df[COL_R2])) if COL_R2 in df.columns else np.nan,
        nmed=float(np.nanmedian(df[COL_N])) if COL_N in df.columns else np.nan,
        rrmed=float(np.nanmedian(df["_RINF_OVER_RMAX"]))  # internal
    )

def apply_gates(df: pd.DataFrame, r2_min: float, rr_max: float):
    m = np.ones(len(df), dtype=bool)

    # basic numeric sanity
    for c in [COL_TAU, COL_R2, COL_N]:
        if c not in df.columns:
            raise KeyError(f"Missing required column '{c}' in tau_fits.csv. Found: {list(df.columns)}")

    m &= np.isfinite(df[COL_TAU].to_numpy())
    m &= np.isfinite(df[COL_R2].to_numpy())
    m &= np.isfinite(df[COL_N].to_numpy())
    m &= np.isfinite(df["_RINF_OVER_RMAX"].to_numpy())

    # base gates
    m &= (df[COL_TAU] > 0) & (df[COL_TAU] <= TAU_MAX_MS)
    m &= (df[COL_N] >= NPOINTS_MIN)

    # sweep gates
    m &= (df[COL_R2] >= r2_min)
    m &= (df["_RINF_OVER_RMAX"] <= rr_max)

    return df.loc[m].copy()

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    fapi = pd.read_csv(FAPI_TAU)
    tempo = pd.read_csv(TEMPO_TAU)

    # Identify/compute ratio column, store into a unified internal name
    fapi_ratio_col = infer_ratio_column(fapi)
    tempo_ratio_col = infer_ratio_column(tempo)
    fapi["_RINF_OVER_RMAX"] = fapi[fapi_ratio_col].astype(float)
    tempo["_RINF_OVER_RMAX"] = tempo[tempo_ratio_col].astype(float)

    rows = []
    for r2_min in R2_SWEEP:
        for rr_max in RINF_OVER_RMAX_SWEEP:
            f_filt = apply_gates(fapi, r2_min=r2_min, rr_max=rr_max)
            t_filt = apply_gates(tempo, r2_min=r2_min, rr_max=rr_max)

            fd = describe_tau(f_filt)
            td = describe_tau(t_filt)

            row = dict(
                r2_min=r2_min,
                Rinf_over_Rmax_max=rr_max,
                # FAPI
                FAPI_n=fd["n"],
                FAPI_tau_med_ms=fd["med"],
                FAPI_tau_q25_ms=fd["q25"],
                FAPI_tau_q75_ms=fd["q75"],
                FAPI_r2_med=fd["r2med"],
                FAPI_npoints_med=fd["nmed"],
                FAPI_RinfRmax_med=fd["rrmed"],
                # TEMPO
                TEMPO_n=td["n"],
                TEMPO_tau_med_ms=td["med"],
                TEMPO_tau_q25_ms=td["q25"],
                TEMPO_tau_q75_ms=td["q75"],
                TEMPO_r2_med=td["r2med"],
                TEMPO_npoints_med=td["nmed"],
                TEMPO_RinfRmax_med=td["rrmed"],
                # effect size (ratio)
                tau_ratio_FAPI_over_TEMPO=(fd["med"] / td["med"]) if np.isfinite(fd["med"]) and np.isfinite(td["med"]) and td["med"] > 0 else np.nan,
            )
            rows.append(row)

    out = pd.DataFrame(rows)

    # Pretty console print (compact)
    show_cols = [
        "r2_min", "Rinf_over_Rmax_max",
        "FAPI_n", "FAPI_tau_med_ms",
        "TEMPO_n", "TEMPO_tau_med_ms",
        "tau_ratio_FAPI_over_TEMPO"
    ]
    print("\n=== Gate-sensitivity sweep (base: tau<=%.0f ms, n_points>=%d) ===" % (TAU_MAX_MS, NPOINTS_MIN))
    print(out[show_cols].to_string(index=False, justify="center", float_format=lambda x: f"{x:0.3g}"))

    out.to_csv(OUT_CSV, index=False)
    print(f"\n[OK] Wrote sweep summary CSV:\n  {OUT_CSV}")

if __name__ == "__main__":
    main()
