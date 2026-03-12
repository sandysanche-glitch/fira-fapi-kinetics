# nucleation_growth_decoupled_with_jmak.py
# Assumes TOTAL_MS = 600 ms, nucleation = 0–60 ms, growth = 60–600 ms (duration 540 ms)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==== USER SETTINGS ==========================================================
STS_PATH = Path(r"D:\SWITCHdrive\Institution\Sts_grain morphology_ML\Sts_metrics.txt")
TOTAL_MS = 600.0

NUCLEATION_MS_START = 0.0
NUCLEATION_MS_END   = 60.0                 # 0–60 ms for nucleation
GROWTH_MS_START     = NUCLEATION_MS_END    # 60–600 ms for growth
GROWTH_MS_END       = TOTAL_MS
GROWTH_MS_DURATION  = GROWTH_MS_END - GROWTH_MS_START  # 540 ms

BIN_MS   = 10.0     # bin width for dn/dt and growth stats (ms)
GRID_MS  = 1.0      # export grid for JMAK overlays (ms)
# ============================================================================

def load_sts_metrics(path: Path) -> pd.DataFrame:
    """
    Load Sts_metrics.txt and return tidy dataframe: [SampleID, Dataset, Area_um2]
    Robust to slightly different column names/positions.
    """
    df = pd.read_csv(path, sep=r"\t", engine="python")
    cols = list(df.columns)

    def pick(patterns):
        out = {}
        for key, pats in patterns.items():
            found = None
            for c in cols:
                c_norm = c.replace(" ", "").lower()
                for p in pats:
                    if p in c_norm:
                        found = c
                        break
                if found: break
            out[key] = found
        return out

    # Left block (often FAPI)
    left = pick({
        "file":["file_name"],
        "dataset":["dataset"],
        "area":["fapi_area_", "area_um2", "area"]
    })
    # Right block (often FAPI-TEMPO)
    right = pick({
        "file":["file_name.1","filename.1"],
        "dataset":["dataset.1"],
        "area":["fapitempo_area_", "fapi tempo_area_", "area_um2.1", "area.1"]
    })

    parts = []
    if left["dataset"] and left["area"]:
        parts.append(pd.DataFrame({
            "SampleID": df[left["file"]] if left["file"] else None,
            "Dataset": df[left["dataset"]],
            "Area_um2": pd.to_numeric(df[left["area"]], errors="coerce"),
        }))
    if right["dataset"] and right["area"]:
        parts.append(pd.DataFrame({
            "SampleID": df[right["file"]] if right["file"] else None,
            "Dataset": df[right["dataset"]],
            "Area_um2": pd.to_numeric(df[right["area"]], errors="coerce"),
        }))

    if parts:
        tidy = pd.concat(parts, ignore_index=True)
    else:
        tidy = pd.DataFrame(columns=["SampleID","Dataset","Area_um2"])

    tidy = tidy.dropna(subset=["Dataset","Area_um2"])
    tidy["Dataset"] = (
        tidy["Dataset"].astype(str).str.strip()
        .replace({"FAPI TEMPO":"FAPI-TEMPO","FAPI_TEMPO":"FAPI-TEMPO","FAPI TEMPO_":"FAPI-TEMPO"})
    )
    return tidy

def map_to_time(df: pd.DataFrame, total_ms: float) -> pd.DataFrame:
    """
    Rank areas (largest = earliest) and map to pseudo-time in [0, total_ms].
    Also compute radius = sqrt(area/pi).
    """
    w = df.copy()
    w["area_rank_desc"] = w["Area_um2"].rank(method="average", ascending=False, pct=True)
    w["t_ms"] = (1.0 - w["area_rank_desc"]) * total_ms
    w["radius_um"] = np.sqrt(w["Area_um2"] / np.pi)
    return w

def estimate_dn_dt(t_ms: np.ndarray, bin_ms: float, total_ms: float):
    if t_ms.size == 0:
        return np.array([]), np.array([])
    edges = np.arange(0, max(total_ms, float(np.nanmax(t_ms))) + bin_ms, bin_ms)
    hist, edges = np.histogram(t_ms, bins=edges)
    centers = 0.5*(edges[:-1] + edges[1:])
    dn_dt = hist / bin_ms  # counts per ms
    if dn_dt.size >= 3:
        dn_dt = pd.Series(dn_dt).rolling(3, center=True, min_periods=1).mean().to_numpy()
    return centers, dn_dt

def estimate_growth_rate(t_ms: np.ndarray, r_um: np.ndarray, bin_ms: float, total_ms: float):
    if t_ms.size == 0:
        return np.array([]), np.array([]), np.array([])
    edges = np.arange(0, max(total_ms, float(np.nanmax(t_ms))) + bin_ms, bin_ms)
    idx = np.digitize(t_ms, edges) - 1
    med_r, centers = [], []
    for b in range(len(edges)-1):
        sel = r_um[idx==b]
        if sel.size:
            med_r.append(np.median(sel))
            centers.append(0.5*(edges[b]+edges[b+1]))
    med_r = np.asarray(med_r, float)
    centers = np.asarray(centers, float)
    if centers.size >= 2:
        drdt = np.gradient(med_r, centers)
        if drdt.size >= 3:
            drdt = pd.Series(drdt).rolling(3, center=True, min_periods=1).mean().to_numpy()
    else:
        drdt = np.zeros_like(med_r)
    return centers, med_r, drdt

def cumulative_area_fraction(df_time: pd.DataFrame):
    d = df_time.sort_values("t_ms").copy()
    total_area = d["Area_um2"].sum()
    if total_area <= 0 or len(d)==0:
        return np.array([]), np.array([])
    d["cumA"] = d["Area_um2"].cumsum()
    d["X"] = d["cumA"] / total_area
    return d["t_ms"].to_numpy(), d["X"].to_numpy()

def resample_to_grid(t, y, t_grid):
    if t.size == 0:
        return np.full_like(t_grid, np.nan, dtype=float)
    order = np.argsort(t)
    t_sorted = t[order]
    y_sorted = y[order]
    s = pd.Series(y_sorted, index=t_sorted)
    s = s[~s.index.duplicated(keep="last")]
    return np.interp(t_grid, s.index.to_numpy(), s.to_numpy(), left=s.iloc[0], right=s.iloc[-1])

def jmak_func(t, k, n):
    t = np.maximum(t, 0.0)
    n = max(n, 1e-6)
    return 1.0 - np.exp(- (k * t)**n)

def fit_jmak(t, X, total_ms):
    # Simple grid-search + local refine (no SciPy dependency)
    mask = (X > 1e-6) & (X < 1-1e-6) & (t > 0)
    t_fit = t[mask]; X_fit = X[mask]
    if t_fit.size < 10:
        return 1/total_ms, 2.0  # fallback

    k_grid = np.logspace(np.log10(1/(10*total_ms)), np.log10(10.0/total_ms), 60)
    n_grid = np.linspace(1.0, 4.0, 31)
    best = (np.inf, k_grid[0], n_grid[0])

    for n in n_grid:
        t_pow = (t_fit**n)
        for k in k_grid:
            X_hat = 1.0 - np.exp(-(k**n) * t_pow)
            err = np.nanmean((X_hat - X_fit)**2)
            if err < best[0]:
                best = (err, k, n)

    # local refine
    k0, n0 = best[1], best[2]
    k_ref = np.logspace(np.log10(k0/3), np.log10(k0*3), 40)
    n_ref = np.linspace(max(0.5, n0-0.5), n0+0.5, 25)
    best2 = best
    for n in n_ref:
        t_pow = (t_fit**n)
        for k in k_ref:
            X_hat = 1.0 - np.exp(-(k**n) * t_pow)
            err = np.nanmean((X_hat - X_fit)**2)
            if err < best2[0]:
                best2 = (err, k, n)
    return best2[1], best2[2]

def main():
    if not STS_PATH.exists():
        raise FileNotFoundError(f"Cannot find {STS_PATH}. Edit STS_PATH at top of script.")

    # Record the assumed windows
    pd.DataFrame({
        "TOTAL_MS":[TOTAL_MS],
        "NUCLEATION_START_MS":[NUCLEATION_MS_START],
        "NUCLEATION_END_MS":[NUCLEATION_MS_END],
        "GROWTH_START_MS":[GROWTH_MS_START],
        "GROWTH_END_MS":[GROWTH_MS_END],
        "GROWTH_DURATION_MS":[GROWTH_MS_DURATION],
    }).to_csv("assumed_windows.csv", index=False)

    tidy = load_sts_metrics(STS_PATH)

    # Split datasets
    fapi = tidy[ tidy["Dataset"].str.contains(r"\bFAPI\b", case=False, regex=True) &
                 ~tidy["Dataset"].str.contains("TEMPO", case=False) ].copy()
    ftempo = tidy[ tidy["Dataset"].str.contains("FAPI-TEMPO", case=False) ].copy()

    fapi_t   = map_to_time(fapi, TOTAL_MS)
    ftempo_t = map_to_time(ftempo, TOTAL_MS)

    # ---------- NUCLEATION (dn/dt) ----------
    c_f,  dn_f  = estimate_dn_dt(fapi_t["t_ms"].to_numpy(),   BIN_MS, TOTAL_MS)
    c_t,  dn_t  = estimate_dn_dt(ftempo_t["t_ms"].to_numpy(), BIN_MS, TOTAL_MS)

    plt.figure(figsize=(7,4.5))
    if c_f.size: plt.plot(c_f, dn_f, label="FAPI")
    if c_t.size: plt.plot(c_t, dn_t, label="FAPI-TEMPO")
    # Shade nucleation window: 0–60 ms
    plt.axvspan(NUCLEATION_MS_START, NUCLEATION_MS_END, color="0.9", alpha=0.5,
                label=f"Nucleation 0–{int(NUCLEATION_MS_END)} ms")
    plt.xlabel("Pseudo-time (ms)"); plt.ylabel("Estimated dn/dt (counts/ms)")
    plt.title("Nucleation rate proxy (decoupled)")
    plt.legend(); plt.tight_layout()
    plt.savefig("decoupled_dn_dt.png", dpi=160); plt.close()

    dn_dt_df = pd.concat([
        pd.DataFrame({"t_ms": c_f, "dn_dt_counts_per_ms": dn_f, "Dataset":"FAPI"}),
        pd.DataFrame({"t_ms": c_t, "dn_dt_counts_per_ms": dn_t, "Dataset":"FAPI-TEMPO"})
    ], ignore_index=True)
    dn_dt_df.to_csv("decoupled_dn_dt_proxy.csv", index=False)

    # ---------- GROWTH (median radius & dr/dt) ----------
    cg_f, mr_f, dr_f = estimate_growth_rate(fapi_t["t_ms"].to_numpy(),   fapi_t["radius_um"].to_numpy(),   BIN_MS, TOTAL_MS)
    cg_t, mr_t, dr_t = estimate_growth_rate(ftempo_t["t_ms"].to_numpy(), ftempo_t["radius_um"].to_numpy(), BIN_MS, TOTAL_MS)

    plt.figure(figsize=(7,4.5))
    if cg_f.size: plt.plot(cg_f, dr_f, label="FAPI")
    if cg_t.size: plt.plot(cg_t, dr_t, label="FAPI-TEMPO")
    # Mark nucleation end (growth region begins)
    plt.axvline(NUCLEATION_MS_END, linestyle="--", alpha=0.8,
                label=f"Growth window: {int(GROWTH_MS_START)}–{int(GROWTH_MS_END)} ms")
    plt.xlabel("Pseudo-time (ms)"); plt.ylabel("d⟨radius⟩/dt (μm/ms)")
    plt.title("Growth rate proxy (decoupled)")
    plt.legend(); plt.tight_layout()
    plt.savefig("decoupled_growth_rate.png", dpi=160); plt.close()

    gr_df = pd.concat([
        pd.DataFrame({"t_ms": cg_f, "median_radius_um": mr_f, "drdt_um_per_ms": dr_f, "Dataset":"FAPI"}),
        pd.DataFrame({"t_ms": cg_t, "median_radius_um": mr_t, "drdt_um_per_ms": dr_t, "Dataset":"FAPI-TEMPO"})
    ], ignore_index=True)
    gr_df.to_csv("decoupled_growth_proxy.csv", index=False)

    # ---------- JMAK OVERLAYS ON X(t) ----------
    def X_of_t(df_time):
        t, X = cumulative_area_fraction(df_time)
        grid = np.arange(0.0, TOTAL_MS+GRID_MS, GRID_MS)
        Xg = resample_to_grid(t, X, grid)
        k, n = fit_jmak(grid, Xg, TOTAL_MS)
        Xfit = jmak_func(grid, k, n)
        return grid, Xg, Xfit, k, n

    tgf, Xf, Xf_fit, kf, nf = X_of_t(fapi_t)
    tgt, Xt, Xt_fit, kt, nt = X_of_t(ftempo_t)

    # FAPI plot
    plt.figure(figsize=(7,4.5))
    plt.plot(tgf, Xf, label="FAPI X(t) (area fraction)", alpha=0.9)
    plt.plot(tgf, Xf_fit, "--", label=f"JMAK fit (k={kf:.3e}, n={nf:.2f})")
    plt.axvspan(NUCLEATION_MS_START, NUCLEATION_MS_END, color="0.9", alpha=0.4,
                label=f"Nucleation 0–{int(NUCLEATION_MS_END)} ms")
    plt.axvspan(GROWTH_MS_START, GROWTH_MS_END, color="0.95", alpha=0.25,
                label=f"Growth {int(GROWTH_MS_START)}–{int(GROWTH_MS_END)} ms")
    plt.xlabel("Pseudo-time (ms)"); plt.ylabel("Transformed fraction X(t)")
    plt.ylim(0,1.05); plt.title("FAPI: X(t) with JMAK overlay")
    plt.legend(); plt.tight_layout()
    plt.savefig("FAPI_X_t_JMAK.png", dpi=160); plt.close()

    # FAPI-TEMPO plot
    plt.figure(figsize=(7,4.5))
    plt.plot(tgt, Xt, label="FAPI-TEMPO X(t) (area fraction)", alpha=0.9)
    plt.plot(tgt, Xt_fit, "--", label=f"JMAK fit (k={kt:.3e}, n={nt:.2f})")
    plt.axvspan(NUCLEATION_MS_START, NUCLEATION_MS_END, color="0.9", alpha=0.4,
                label=f"Nucleation 0–{int(NUCLEATION_MS_END)} ms")
    plt.axvspan(GROWTH_MS_START, GROWTH_MS_END, color="0.95", alpha=0.25,
                label=f"Growth {int(GROWTH_MS_START)}–{int(GROWTH_MS_END)} ms")
    plt.xlabel("Pseudo-time (ms)"); plt.ylabel("Transformed fraction X(t)")
    plt.ylim(0,1.05); plt.title("FAPI-TEMPO: X(t) with JMAK overlay")
    plt.legend(); plt.tight_layout()
    plt.savefig("FAPI_TEMPO_X_t_JMAK.png", dpi=160); plt.close()

    # CSV exports for overlays
    pd.DataFrame({
        "t_ms": tgf, "X_area_frac": Xf, "X_JMAK": Xf_fit,
        "k_fit": [kf]*len(tgf), "n_fit": [nf]*len(tgf)
    }).to_csv("fapi_X_t_overlay.csv", index=False)

    pd.DataFrame({
        "t_ms": tgt, "X_area_frac": Xt, "X_JMAK": Xt_fit,
        "k_fit": [kt]*len(tgt), "n_fit": [nt]*len(tgt)
    }).to_csv("fapi_tempo_X_t_overlay.csv", index=False)

    print("Saved:")
    print(" - assumed_windows.csv")
    print(" - decoupled_dn_dt.png")
    print(" - decoupled_growth_rate.png")
    print(" - FAPI_X_t_JMAK.png")
    print(" - FAPI_TEMPO_X_t_JMAK.png")
    print(" - decoupled_dn_dt_proxy.csv")
    print(" - decoupled_growth_proxy.csv")
    print(" - fapi_X_t_overlay.csv")
    print(" - fapi_tempo_X_t_overlay.csv")
    print(f"JMAK fits | FAPI: k={kf:.4e}, n={nf:.2f} | FAPI-TEMPO: k={kt:.4e}, n={nt:.2f}")

if __name__ == "__main__":
    main()
