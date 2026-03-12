import os
import re
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
# BASE GATES
# ----------------------------
TAU_MAX_MS = 5000.0
NPOINTS_MIN = 20

# Sweep settings (you can change these lists)
R2_SWEEP = [0.3, 0.4, 0.5, 0.6]
RINF_OVER_RMAX_SWEEP = [1.2, 1.5, 2.0]   # includes your physics gate 1.5

# Expected core columns
COL_TAU = "tau_ms"
COL_R2 = "r2"
COL_N = "n_points"

# ----------------------------
# Helpers
# ----------------------------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def find_tau_col(df):
    # Usually tau_ms exists; otherwise try variants
    for c in df.columns:
        if _norm(c) in ["taums", "tau", "tau_ms"]:
            return c
    raise KeyError(f"Could not find tau column. Columns: {list(df.columns)}")

def find_r2_col(df):
    for c in df.columns:
        if _norm(c) in ["r2", "rsquared", "r_sq", "rsq"]:
            return c
    raise KeyError(f"Could not find r2 column. Columns: {list(df.columns)}")

def find_npoints_col(df):
    for c in df.columns:
        if _norm(c) in ["npoints", "n_point", "n_points", "nobs", "n"]:
            return c
    raise KeyError(f"Could not find n_points column. Columns: {list(df.columns)}")

def infer_ratio_column(df: pd.DataFrame) -> str:
    """
    Try:
      1) direct ratio columns (many name variants)
      2) compute from Rinf & Rmax columns (any unit)
      3) heuristic substring matching
    """
    # 1) direct ratio columns
    direct_candidates = []
    for c in df.columns:
        cn = _norm(c)
        if "rinf" in cn and ("rmax" in cn or "rmax" in cn) and ("over" in cn or "div" in cn or "ratio" in cn):
            direct_candidates.append(c)
    # also accept common short names
    for c in df.columns:
        cn = _norm(c)
        if cn in ["rinfoverrmax", "rinfrmax", "rinfdivrmax", "rinf_rmax", "rinf_over_rmax"]:
            direct_candidates.append(c)

    if direct_candidates:
        # pick first
        return direct_candidates[0]

    # 2) compute from separate Rinf and Rmax cols
    # Try common explicit names first
    rinf_name_priority = [
        "Rinf_px", "R_inf_px", "Rinf_um", "R_inf_um", "Rinf", "R_inf", "R_inf_fit", "Rinf_fit",
        "Rinf_fit_px", "R_inf_fit_px", "Rinf_fit_um", "R_inf_fit_um"
    ]
    rmax_name_priority = [
        "Rmax_px", "R_max_px", "Rmax_um", "R_max_um", "Rmax", "R_max",
        "Rmax_px_last", "R_max_px_last", "Rmax_um_last", "R_max_um_last"
    ]

    def pick_existing(names):
        for n in names:
            if n in df.columns:
                return n
        return None

    rinf_col = pick_existing(rinf_name_priority)
    rmax_col = pick_existing(rmax_name_priority)

    # 3) heuristic search: any col containing rinf / r_inf and any containing rmax / r_max
    if rinf_col is None:
        for c in df.columns:
            cn = _norm(c)
            if cn.startswith("rinf") or "rinf" in cn or "rinf" in cn or cn.startswith("rinf"):
                if "over" not in cn and "ratio" not in cn and "div" not in cn:
                    rinf_col = c
                    break

    if rmax_col is None:
        for c in df.columns:
            cn = _norm(c)
            if cn.startswith("rmax") or "rmax" in cn or cn.startswith("rmax"):
                if "over" not in cn and "ratio" not in cn and "div" not in cn:
                    rmax_col = c
                    break

    if (rinf_col is not None) and (rmax_col is not None):
        df["_RINF_OVER_RMAX_COMPUTED"] = pd.to_numeric(df[rinf_col], errors="coerce") / pd.to_numeric(df[rmax_col], errors="coerce").replace(0, np.nan)
        return "_RINF_OVER_RMAX_COMPUTED"

    # If we reach here: fail loudly with columns shown
    raise KeyError(
        "Could not find or derive Rinf/Rmax.\n"
        f"Columns are:\n{list(df.columns)}\n\n"
        "Fix: ensure tau_fits.csv contains either a direct ratio column (e.g. Rinf_over_Rmax)\n"
        "or both Rinf and Rmax columns (any unit)."
    )

def apply_gates(df: pd.DataFrame, tau_col, r2_col, n_col, rr_col, r2_min: float, rr_max: float):
    x = df.copy()
    # numeric coercion
    for c in [tau_col, r2_col, n_col, rr_col]:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    m = np.isfinite(x[tau_col]) & np.isfinite(x[r2_col]) & np.isfinite(x[n_col]) & np.isfinite(x[rr_col])
    m &= (x[tau_col] > 0) & (x[tau_col] <= TAU_MAX_MS)
    m &= (x[n_col] >= NPOINTS_MIN)
    m &= (x[r2_col] >= r2_min)
    m &= (x[rr_col] <= rr_max)
    return x.loc[m].copy()

def summarize(df: pd.DataFrame, tau_col, r2_col, n_col, rr_col):
    if len(df) == 0:
        return dict(n=0, tau_med=np.nan, tau_q25=np.nan, tau_q75=np.nan,
                    r2_med=np.nan, n_med=np.nan, rr_med=np.nan)
    return dict(
        n=int(len(df)),
        tau_med=float(np.nanmedian(df[tau_col])),
        tau_q25=float(np.nanpercentile(df[tau_col], 25)),
        tau_q75=float(np.nanpercentile(df[tau_col], 75)),
        r2_med=float(np.nanmedian(df[r2_col])),
        n_med=float(np.nanmedian(df[n_col])),
        rr_med=float(np.nanmedian(df[rr_col])),
    )

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    fapi = pd.read_csv(FAPI_TAU)
    tempo = pd.read_csv(TEMPO_TAU)

    # robust column detection
    tau_f = find_tau_col(fapi);    tau_t = find_tau_col(tempo)
    r2_f  = find_r2_col(fapi);     r2_t  = find_r2_col(tempo)
    n_f   = find_npoints_col(fapi); n_t  = find_npoints_col(tempo)

    rr_f = infer_ratio_column(fapi)
    rr_t = infer_ratio_column(tempo)

    rows = []
    for r2_min in R2_SWEEP:
        for rr_max in RINF_OVER_RMAX_SWEEP:
            ff = apply_gates(fapi,  tau_f, r2_f, n_f, rr_f, r2_min, rr_max)
            tt = apply_gates(tempo, tau_t, r2_t, n_t, rr_t, r2_min, rr_max)

            sf = summarize(ff, tau_f, r2_f, n_f, rr_f)
            st = summarize(tt, tau_t, r2_t, n_t, rr_t)

            rows.append(dict(
                r2_min=r2_min,
                Rinf_over_Rmax_max=rr_max,
                FAPI_n=sf["n"],
                FAPI_tau_med_ms=sf["tau_med"],
                FAPI_tau_q25_ms=sf["tau_q25"],
                FAPI_tau_q75_ms=sf["tau_q75"],
                FAPI_r2_med=sf["r2_med"],
                FAPI_npoints_med=sf["n_med"],
                FAPI_RinfRmax_med=sf["rr_med"],
                TEMPO_n=st["n"],
                TEMPO_tau_med_ms=st["tau_med"],
                TEMPO_tau_q25_ms=st["tau_q25"],
                TEMPO_tau_q75_ms=st["tau_q75"],
                TEMPO_r2_med=st["r2_med"],
                TEMPO_npoints_med=st["n_med"],
                TEMPO_RinfRmax_med=st["rr_med"],
                tau_ratio_FAPI_over_TEMPO=(sf["tau_med"]/st["tau_med"]) if np.isfinite(sf["tau_med"]) and np.isfinite(st["tau_med"]) and st["tau_med"]>0 else np.nan
            ))

    out = pd.DataFrame(rows)

    # Compact print
    show = out[[
        "r2_min", "Rinf_over_Rmax_max",
        "FAPI_n", "FAPI_tau_med_ms",
        "TEMPO_n", "TEMPO_tau_med_ms",
        "tau_ratio_FAPI_over_TEMPO"
    ]].copy()

    print(f"\n=== Gate sensitivity sweep (base: tau<= {TAU_MAX_MS:.0f} ms, n_points>= {NPOINTS_MIN}) ===")
    print(show.to_string(index=False, float_format=lambda x: f"{x:0.3g}"))

    out.to_csv(OUT_CSV, index=False)
    print(f"\n[OK] Wrote:\n  {OUT_CSV}")
    print("\nNote: Rinf/Rmax column used:")
    print(f"  FAPI : {rr_f}")
    print(f"  TEMPO: {rr_t}")

if __name__ == "__main__":
    main()
