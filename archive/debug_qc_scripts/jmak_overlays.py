# jmak_overlays.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---- settings ----
STS_PATH = Path("Sts_metrics.txt")  # path to your table
TOTAL_MS = 600.0                    # assumed total solidification window
GRID_MS  = 1.0                      # export grid resolution for X(t)
# ------------------

def load_sts_metrics(path: Path) -> pd.DataFrame:
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

    # Left: FAPI-like
    left_map = pick({
        "file":["file_name"],
        "dataset":["dataset"],
        "area":["fapi_area_"],
    })
    # Right: FAPI-TEMPO-like
    right_map = pick({
        "file":["file_name.1","filename.1"],
        "dataset":["dataset.1"],
        "area":["fapitempo_area_","fapi tempo_area_"],
    })

    parts = []
    if left_map["dataset"] and left_map["area"]:
        parts.append(pd.DataFrame({
            "SampleID": df[left_map["file"]] if left_map["file"] else None,
            "Dataset": df[left_map["dataset"]],
            "Area_um2": pd.to_numeric(df[left_map["area"]], errors="coerce"),
        }))
    if right_map["dataset"] and right_map["area"]:
        parts.append(pd.DataFrame({
            "SampleID": df[right_map["file"]] if right_map["file"] else None,
            "Dataset": df[right_map["dataset"]],
            "Area_um2": pd.to_numeric(df[right_map["area"]], errors="coerce"),
        }))

    tidy = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["SampleID","Dataset","Area_um2"])
    tidy = tidy.dropna(subset=["Dataset","Area_um2"])
    tidy["Dataset"] = tidy["Dataset"].astype(str).str.strip().replace({
        "FAPI TEMPO":"FAPI-TEMPO",
        "FAPI_TEMPO":"FAPI-TEMPO",
        "FAPI TEMPO_":"FAPI-TEMPO"
    })
    return tidy

def map_to_time(df: pd.DataFrame, total_ms: float) -> pd.DataFrame:
    w = df.copy()
    # Largest areas -> earliest times; map ranks to [0,total_ms]
    w["area_rank_desc"] = w["Area_um2"].rank(method="average", ascending=False, pct=True)
    w["t_ms"] = (1.0 - w["area_rank_desc"]) * total_ms
    return w

def cumulative_area_fraction(df_time: pd.DataFrame):
    d = df_time.sort_values("t_ms").copy()
    S = d["Area_um2"].sum()
    if S <= 0 or len(d)==0:
        return np.array([]), np.array([])
    d["cumA"] = d["Area_um2"].cumsum()
    d["X"] = d["cumA"] / S
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
    # Fit (k,n) via grid-search + local refine (no SciPy required)
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

def run_one(df, label, total_ms, grid_ms):
    t, X = cumulative_area_fraction(df)
    t_grid = np.arange(0.0, total_ms+grid_ms, grid_ms)
    X_grid = resample_to_grid(t, X, t_grid)
    k, n = fit_jmak(t_grid, X_grid, total_ms)
    X_fit = jmak_func(t_grid, k, n)
    # Plot
    plt.figure(figsize=(7,4.5))
    plt.plot(t_grid, X_grid, label=f"{label} X(t) (area fraction)", alpha=0.85)
    plt.plot(t_grid, X_fit,  "--", label=f"JMAK fit {label} (k={k:.3e}, n={n:.2f})")
    plt.xlabel("Pseudo-time (ms)")
    plt.ylabel("Transformed fraction X(t)")
    plt.ylim(0, 1.05)
    plt.title(f"{label}: X(t) with JMAK overlay")
    plt.legend()
    plt.tight_layout()
    out_png = f"{label.replace('-','_')}_X_t_JMAK.png"
    plt.savefig(out_png, dpi=160)
    plt.close()
    # CSV export
    out_csv = f"{label.lower().replace('-','_')}_X_t_overlay.csv"
    pd.DataFrame({"t_ms": t_grid, "X_area_frac": X_grid, "X_JMAK": X_fit,
                  "k_fit": k, "n_fit": n}).to_csv(out_csv, index=False)
    return out_png, out_csv, (k, n)

def main():
    df = load_sts_metrics(STS_PATH)
    fapi = df[ df["Dataset"].str.contains(r"\bFAPI\b", case=False, regex=True) &
               ~df["Dataset"].str.contains("TEMPO", case=False) ].copy()
    ftempo = df[ df["Dataset"].str.contains("FAPI-TEMPO", case=False) ].copy()

    fapi_t = map_to_time(fapi, TOTAL_MS)
    ftmp_t = map_to_time(ftempo, TOTAL_MS)

    fapi_png, fapi_csv, (kf, nf) = run_one(fapi_t, "FAPI", TOTAL_MS, GRID_MS)
    ftmp_png, ftmp_csv, (kt, nt) = run_one(ftmp_t, "FAPI-TEMPO", TOTAL_MS, GRID_MS)

    print("Saved files:")
    print(" ", fapi_png)
    print(" ", ftmp_png)
    print(" ", fapi_csv)
    print(" ", ftmp_csv)
    print(f"FAPI   fit: k={kf:.4e}, n={nf:.2f}")
    print(f"TEMPO  fit: k={kt:.4e}, n={nt:.2f}")

if __name__ == "__main__":
    assert STS_PATH.exists(), f"Cannot find {STS_PATH}"
    main()
