import pandas as pd
import numpy as np
import math
from pathlib import Path
import io, zipfile

TOTAL_MS = 600.0
SRC = Path("Sts_metrics.txt")  # put this script next to your file

# ---------- I/O + parsing ----------
def load_metrics_txt(path: Path) -> pd.DataFrame:
    """
    File has 12 columns (two 6-col blocks). We split left/right and stack rows.
    Columns per block: file_name, Dataset, Area_um, Perimeter_um, CD, Entropy
    """
    df = pd.read_csv(path, sep=r"\t", engine="python")
    L = df.columns[:6]; R = df.columns[6:]
    left = df[list(L)].copy(); right = df[list(R)].copy()
    left.columns = ["file_name","Dataset","Area_um","Perimeter_um","CD","Entropy"]
    right.columns = ["file_name","Dataset","Area_um","Perimeter_um","CD","Entropy"]
    out = pd.concat([left, right], ignore_index=True)
    out = out.dropna(subset=["file_name","Dataset","Area_um"])
    out["Dataset"] = out["Dataset"].astype(str).str.strip()
    # SampleID from filename (adjust if your naming differs)
    fn = out["file_name"].astype(str).str.replace("\\\\","/", regex=True).str.split("/").str[-1]
    out["SampleID"] = fn.str.split("_").str[0]
    return out

def areas_to_times_ms(area: np.ndarray, t_total_ms: float=600.0) -> np.ndarray:
    """Largest area => earliest time; rank areas (desc) onto [0, t_total]."""
    order = np.argsort(-area)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(area)+1)
    return t_total_ms * (ranks - 0.5) / len(area)

# ---------- distributions for I(t) ----------
def fit_lognormal(t_ms: np.ndarray, eps: float=1e-9):
    x = np.log(np.clip(t_ms, eps, None))
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1)) if len(x) > 1 else 1e-6
    return {"mu": mu, "sigma": sigma}

def loglik_lognormal(t, mu, sigma, eps: float=1e-12):
    t = np.asarray(t); x = np.clip(t, eps, None)
    return float(np.sum(-np.log(x * sigma * math.sqrt(2*math.pi)) - ((np.log(x)-mu)**2)/(2*sigma**2)))

def cdf_lognormal(t, mu, sigma, eps: float=1e-12):
    from math import erf, sqrt
    t = np.asarray(t); x = np.clip(t, eps, None)
    z = (np.log(x)-mu)/(sigma*sqrt(2))
    return 0.5*(1 + np.vectorize(erf)(z))

def fit_gamma(t_ms: np.ndarray, eps: float=1e-9):
    x = np.clip(t_ms, eps, None)
    m = float(np.mean(x))
    v = float(np.var(x, ddof=1)) if len(x) > 1 else (m*m)
    if v <= 0 or m <= 0:
        return {"k": 1.0, "theta": max(eps, m)}
    return {"k": (m*m)/v, "theta": v/m}

def loglik_gamma(t, k, theta, eps: float=1e-12):
    t = np.asarray(t); x = np.clip(t, eps, None)
    return float(np.sum((k-1)*np.log(x) - x/theta - math.lgamma(k) - k*np.log(theta)))

def cdf_gamma_on_grid(t, k, theta, eps: float=1e-12):
    # Numerical CDF at 1-ms resolution
    x = np.clip(np.asarray(t), eps, None)
    grid = np.arange(0, max(np.max(x), 1)+1, 1.0); grid[0] = eps
    pdf = np.exp((k-1)*np.log(grid) - grid/theta - math.lgamma(k) - k*np.log(theta))
    cdf = np.cumsum((pdf[:-1] + pdf[1:]) * 0.5 * np.diff(grid))
    cdf = np.concatenate([[0.0], cdf])
    return np.interp(x, grid, np.clip(cdf, 0, 1))

def aic_bic(loglik, k_params, n):
    aic = 2*k_params - 2*loglik
    bic = k_params*np.log(n) - 2*loglik
    return aic, bic

# ---------- Avrami overlay ----------
def fit_jmak_from_cdf(t, X):
    """
    Fit n, k from ln[-ln(1-X)] = n ln t + n ln k, using mid-range points.
    """
    mask = (X > 1e-4) & (X < 1-1e-3) & (t > 0)
    if mask.sum() < 3:
        return {"n": np.nan, "k": np.nan}
    y = np.log(-np.log(1 - X[mask])); x = np.log(t[mask])
    A = np.vstack([x, np.ones_like(x)]).T
    (n, ln_kn), *_ = np.linalg.lstsq(A, y, rcond=None)
    k = float(np.exp(ln_kn / n)) if n != 0 else np.nan
    return {"n": float(n), "k": k}

def main():
    if not SRC.exists():
        raise FileNotFoundError("Sts_metrics.txt not found next to this script.")
    df = load_metrics_txt(SRC)
    df = df[df["Dataset"].isin(["FAPI", "FAPI-TEMPO"])].copy()

    t_grid = np.arange(0, TOTAL_MS+1, 1.0)
    rows = []
    overlays = []  # (Dataset, SampleID, DataFrame)

    for (ds, sid), g in df.groupby(["Dataset","SampleID"]):
        if g["Area_um"].count() < 10:
            continue

        # Map areas -> times
        t = areas_to_times_ms(g["Area_um"].to_numpy(), TOTAL_MS)

        # Fit models
        ln_par = fit_lognormal(t)
        gm_par = fit_gamma(t)

        # Likelihoods + BIC selection
        ll_ln = loglik_lognormal(t, ln_par["mu"], ln_par["sigma"])
        ll_gm = loglik_gamma(t, gm_par["k"], gm_par["theta"])
        aic_ln, bic_ln = aic_bic(ll_ln, 2, len(t))
        aic_gm, bic_gm = aic_bic(ll_gm, 2, len(t))
        best = "lognormal" if bic_ln <= bic_gm else "gamma"

        # Evaluate CDFs on 1-ms grid
        ln_cdf = cdf_lognormal(t_grid+1e-12, ln_par["mu"], ln_par["sigma"])
        gm_cdf = cdf_gamma_on_grid(t_grid+1e-12, gm_par["k"], gm_par["theta"])
        X_best = ln_cdf if best == "lognormal" else gm_cdf

        # Nucleation window (5–95%) and growth time
        t5 = float(np.interp(0.05, X_best, t_grid))
        t95 = float(np.interp(0.95, X_best, t_grid))
        tn = max(0.0, t95 - t5)
        tg = max(0.0, TOTAL_MS - t95)

        # JMAK overlay
        jmak = fit_jmak_from_cdf(t_grid, X_best)
        if np.isnan(jmak["n"]) or np.isnan(jmak["k"]):
            X_jmak = np.full_like(t_grid, np.nan, dtype=float)
        else:
            X_jmak = 1 - np.exp(-(jmak["k"]*np.clip(t_grid, 0, None))**jmak["n"])

        rows.append({
            "Dataset": ds, "SampleID": sid, "n_points": len(t),
            "best_model": best,
            "lognormal_mu": ln_par["mu"], "lognormal_sigma": ln_par["sigma"],
            "gamma_k": gm_par["k"], "gamma_theta": gm_par["theta"],
            "AIC_lognormal": aic_ln, "BIC_lognormal": bic_ln,
            "AIC_gamma": aic_gm, "BIC_gamma": bic_gm,
            "tn_start_ms": t5, "tn_end_ms": t95, "tn_ms": tn, "tg_ms": tg,
            "JMAK_n": jmak["n"], "JMAK_k": jmak["k"]
        })

        odf = pd.DataFrame({
            "time_ms": t_grid,
            "X_best": X_best,
            "X_JMAK": X_jmak
        })
        overlays.append((ds, sid, odf))

    # --- Exports ---
    out_dir = SRC.parent

    summary = pd.DataFrame(rows).sort_values(["Dataset","SampleID"]).reset_index(drop=True)
    summary.to_csv(out_dir/"per_sample_nucleation_summary.csv", index=False)

    # Zip all overlays to keep fs tidy
    with zipfile.ZipFile(out_dir/"per_dataset_Xt_overlays_1ms.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for ds, sid, odf in overlays:
            buf = io.StringIO()
            odf.to_csv(buf, index=False)
            zf.writestr(f"{ds}_{sid}_Xt_1ms.csv", buf.getvalue())

    # Excel summary (small, just one sheet)
    try:
        with pd.ExcelWriter(out_dir/"nucleation_per_sample.xlsx", engine="openpyxl") as w:
            summary.to_excel(w, sheet_name="Summary", index=False)
    except Exception as e:
        print("Excel write failed (openpyxl missing?). CSVs were written. Error:", e)

    print("DONE\n  -", (out_dir/"per_sample_nucleation_summary.csv").name,
          "\n  -", (out_dir/"per_dataset_Xt_overlays_1ms.zip").name,
          "\n  -", (out_dir/"nucleation_per_sample.xlsx").name)

if __name__ == "__main__":
    main()
