import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from scipy.stats import lognorm, gamma
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

def safe_num(x, default=np.nan):
    return pd.to_numeric(x, errors="coerce").fillna(default).to_numpy(dtype=float)

def rank_to_t0_ms(area_px: np.ndarray, t_win_ms: float = 60.0) -> np.ndarray:
    n = len(area_px)
    if n <= 1:
        return np.full_like(area_px, fill_value=t_win_ms * 0.5, dtype=float)
    order = np.argsort(area_px)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)
    q = ranks / max(n - 1, 1)
    return t_win_ms * q

def area_px_to_radius_um(area_px: np.ndarray, px_per_um: float) -> np.ndarray:
    R_px = np.sqrt(np.maximum(area_px, 0.0) / math.pi)
    return R_px / px_per_um

def eff_growth_rate_um_per_ms(R_um: np.ndarray, dt_ms: np.ndarray,
                              circularity: np.ndarray = None,
                              defect_frac: np.ndarray = None,
                              alpha: float = 0.0, beta: float = 0.0):
    C = np.clip(circularity, 1e-6, 1.0) if circularity is not None else np.ones_like(R_um)
    phi = np.clip(defect_frac, 0.0, 1.0) if defect_frac is not None else np.zeros_like(R_um)
    f = np.exp(-alpha * (1.0 - C)) * np.exp(-beta * phi)
    denom = np.maximum(dt_ms, 1e-9)
    v_unpen = R_um / denom
    v_eff   = v_unpen * f
    return v_eff, v_unpen, f

def effective_area_um2(area_eff_px: float, px_per_um: float) -> float:
    um_per_px = 1.0 / px_per_um
    return float(area_eff_px) * (um_per_px ** 2)

def fit_dn_dt(t0_ms: np.ndarray, area_um2: float, grid_ms: np.ndarray):
    tpos = t0_ms[(t0_ms > 0) & np.isfinite(t0_ms)]
    if len(tpos) < 5 or not HAVE_SCIPY:
        bins = np.arange(0, grid_ms.max() + 1.0, 1.0)
        hist, edges = np.histogram(t0_ms, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        dn_dt = hist / max(1.0, area_um2) * 1e6  # per (ms·mm^2)
        df = pd.DataFrame({"t_ms": centers,
                           "dn_dt_per_ms_mm2": dn_dt,
                           "model": ["hist"] * len(centers)})
        return df, {"model": "hist"}

    with np.errstate(all='ignore'):
        ln_params = lognorm.fit(tpos, floc=0)   # (s, 0, scale)
        s, loc, scale = ln_params
        ln_pdf = lognorm.pdf(grid_ms, s, loc=loc, scale=scale)

        gm_params = gamma.fit(tpos, floc=0)     # (k, 0, theta)
        k, loc_g, theta = gm_params
        gm_pdf = gamma.pdf(grid_ms, k, loc=loc_g, scale=theta)

    area_mm2 = area_um2 / 1e6
    N = len(t0_ms)
    ln_dn = (N * ln_pdf) / max(area_mm2, 1e-12)
    gm_dn = (N * gm_pdf) / max(area_mm2, 1e-12)

    def score(y, lam, kpar):
        sse = np.sum((y - lam)**2)
        return 2*kpar + len(y)*math.log(max(sse, 1e-24)), kpar*math.log(len(y)) + len(y)*math.log(max(sse, 1e-24))

    bins = np.arange(0, grid_ms.max() + 1.0, 1.0)
    hist, edges = np.histogram(t0_ms, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist_dn = hist / max(area_um2, 1.0) * 1e6

    ln_on = np.interp(centers, grid_ms, ln_dn)
    gm_on = np.interp(centers, grid_ms, gm_dn)
    aic_ln, bic_ln = score(hist_dn, ln_on, kpar=2)
    aic_gm, bic_gm = score(hist_dn, gm_on, kpar=2)

    if (bic_gm < bic_ln):
        params = {"model": "gamma", "k": k, "theta": theta, "AIC": aic_gm, "BIC": bic_gm}
        series = gm_dn; best = "gamma"
    else:
        params = {"model": "lognormal", "s": s, "scale": scale, "AIC": aic_ln, "BIC": bic_ln}
        series = ln_dn; best = "lognormal"

    df = pd.DataFrame({"t_ms": grid_ms,
                       "dn_dt_per_ms_mm2": series,
                       "model": [best] * len(grid_ms)})
    return df, params

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects_csv", required=True)
    ap.add_argument("--label",       required=True)
    ap.add_argument("--outdir",      required=True)
    ap.add_argument("--area_eff_px", type=float, required=True)
    ap.add_argument("--px_per_um",   type=float, default=2.20014)
    ap.add_argument("--use_penalties", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--beta",  type=float, default=0.0)
    ap.add_argument("--n_avrami", type=float, default=2.5)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.objects_csv)

    if "area_px" not in df.columns:
        raise ValueError("Required column 'area_px' not found.")

    area_px = safe_num(df["area_px"])
    C = safe_num(df["circularity"]) if "circularity" in df.columns else np.full_like(area_px, 1.0)
    phi = safe_num(df["defect_frac"]) if "defect_frac" in df.columns else np.zeros_like(area_px)

    t0_ms = rank_to_t0_ms(area_px, t_win_ms=60.0)
    dt_ms = np.clip(600.0 - t0_ms, 1e-9, None)
    R_um  = area_px_to_radius_um(area_px, px_per_um=args.px_per_um)

    alpha = args.alpha if args.use_penalties else 0.0
    beta  = args.beta  if args.use_penalties else 0.0
    v_eff_um_ms, v_unp_um_ms, pf = eff_growth_rate_um_per_ms(R_um, dt_ms, C, phi, alpha, beta)
    v0_um_ms = float(np.nanmedian(v_unp_um_ms))

    area_um2 = effective_area_um2(args.area_eff_px, args.px_per_um)
    grid_ms = np.arange(0.0, 60.0 + 1.0, 1.0)
    dndt_df, dn_meta = fit_dn_dt(t0_ms, area_um2, grid_ms)

    # X(t) in ms and seconds
    t_ms = np.arange(0.0, 600.0 + 1.0, 1.0)
    t_s  = t_ms / 1000.0
    X_pred = np.zeros_like(t_ms)
    for i, t in enumerate(t_ms):
        tau = np.clip(t - t0_ms, 0.0, None)
        r_um = np.maximum(0.0, v_eff_um_ms * tau)
        A_um2 = math.pi * (r_um ** 2)
        X_pred[i] = np.nansum(A_um2) / max(area_um2, 1e-12)

    n = float(args.n_avrami)
    y = np.clip(X_pred, 0.0, 0.999999)
    t_pow = (t_ms ** n)
    valid = (t_pow > 0) & (y > 0) & (y < 0.99)
    if np.any(valid):
        K_vals = -np.log(1.0 - y[valid]) / t_pow[valid]
        K_fit = float(np.nanmedian(K_vals[np.isfinite(K_vals)])) if np.any(np.isfinite(K_vals)) else 0.0
    else:
        K_fit = 0.0
    X_A = 1.0 - np.exp(-K_fit * (t_ms ** n))

    # Exports
    label = args.label
    out_obj = pd.DataFrame({
        "label": label,
        "area_px": area_px,
        "circularity": C,
        "defect_frac": phi,
        "t0_ms": t0_ms,
        "dt_ms": dt_ms,
        "R_um": R_um,
        "v_unpen_um_per_ms": v_unp_um_ms,
        "v_eff_um_per_ms": v_eff_um_ms,
        "v_unpen_um_per_s": v_unp_um_ms * 1000.0,
        "v_eff_um_per_s":   v_eff_um_ms   * 1000.0,
        "penalty_factor": pf
    })
    out_obj.to_csv(outdir / f"{label}_SI_metrics.csv", index=False)

    dndt_df.to_csv(outdir / f"{label}_nucleation_dn_dt.csv", index=False)

    pd.DataFrame({
        "t_ms": t_ms,
        "t_s":  t_s,
        "X_pred": X_pred,
        "X_Avrami": X_A
    }).to_csv(outdir / f"{label}_X_pred_Avrami.csv", index=False)

    pd.DataFrame([{
        "label": label,
        "px_per_um": args.px_per_um,
        "area_eff_px": args.area_eff_px,
        "area_eff_um2": area_um2,
        "use_penalties": bool(args.use_penalties),
        "alpha": alpha,
        "beta": beta,
        "v0_unpen_median_um_per_ms": v0_um_ms,
        "dn_dt_model": dn_meta.get("model", "hist"),
        "dn_dt_params": str({k:v for k,v in dn_meta.items() if k not in ("model",)})
    }]).to_csv(outdir / f"{label}_growth_summary.csv", index=False)

    # Plots
    try:
        import matplotlib.pyplot as plt

        # dn/dt (ms)
        plt.figure()
        plt.plot(dndt_df["t_ms"], dndt_df["dn_dt_per_ms_mm2"])
        plt.xlabel("t (ms)")
        plt.ylabel("dn/dt  [events / (ms·mm²)]")
        plt.title(f"{label}: nucleation density rate")
        plt.tight_layout()
        plt.savefig(outdir / f"{label}_dn_dt_ms.png", dpi=200)
        plt.close()

        # Growth-rate histogram
        plt.figure()
        vals = out_obj["v_eff_um_per_s"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals) > 0:
            plt.hist(vals, bins=60)
        plt.xlabel("growth rate (µm/s) [effective]")
        plt.ylabel("count")
        plt.title(f"{label}: growth-rate distribution")
        plt.tight_layout()
        plt.savefig(outdir / f"{label}_growth_hist.png", dpi=200)
        plt.close()

        # X(t) — ms axis
        plt.figure()
        plt.plot(t_ms, X_pred, label="X_pred (extended)")
        plt.plot(t_ms, X_A,  label=f"Avrami (n={n:.2f}, K={K_fit:.3g})")
        plt.xlabel("t (ms)")
        plt.ylabel("X(t) (fraction)")
        plt.legend(); plt.title(f"{label}: X(t) vs Avrami")
        plt.tight_layout()
        plt.savefig(outdir / f"{label}_X_overlay_ms.png", dpi=200)
        plt.close()

        # X(t) — seconds axis
        plt.figure()
        plt.plot(t_s, X_pred, label="X_pred (extended)")
        plt.plot(t_s, X_A,  label=f"Avrami (n={n:.2f}, K={K_fit:.3g})")
        plt.xlabel("t (s)")
        plt.ylabel("X(t) (fraction)")
        plt.legend(); plt.title(f"{label}: X(t) vs Avrami (seconds)")
        plt.tight_layout()
        plt.savefig(outdir / f"{label}_X_overlay_s.png", dpi=200)
        plt.close()
    except Exception as e:
        print("[WARN] Plotting skipped:", e)

    print(f"[OK] Wrote SI exports in: {outdir}")

if __name__ == "__main__":
    main()
