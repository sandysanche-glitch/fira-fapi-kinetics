import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd

# Try to import scipy for distribution fits; fall back to histogram-only if missing.
try:
    from scipy.stats import lognorm, gamma
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ---------- helpers ----------

def safe_num(x, default=np.nan):
    return pd.to_numeric(x, errors="coerce").fillna(default).to_numpy(dtype=float)

def rank_to_t0_ms(area_px: np.ndarray, t_win_ms: float = 60.0) -> np.ndarray:
    """Map final area rank to nucleation time in [0, t_win_ms]."""
    n = len(area_px)
    if n <= 1:
        return np.full_like(area_px, fill_value=t_win_ms * 0.5, dtype=float)
    order = np.argsort(area_px)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)
    q = ranks / max(n - 1, 1)
    return t_win_ms * q

def px_to_um(px: float, px_per_um: float) -> float:
    return px / px_per_um

def area_px_to_radius_um(area_px: np.ndarray, px_per_um: float) -> np.ndarray:
    # R_px = sqrt(A/pi); R_um = R_px / px_per_um
    R_px = np.sqrt(np.maximum(area_px, 0.0) / math.pi)
    return R_px / px_per_um

def eff_growth_rate_um_per_ms(R_um: np.ndarray, dt_ms: np.ndarray,
                              circularity: np.ndarray = None,
                              defect_frac: np.ndarray = None,
                              alpha: float = 0.0, beta: float = 0.0):
    """Compute per-object growth rates. Two outputs:
       v_eff (with penalties), v_unpenalized (without penalties).
    """
    # Penalties are time-constant here (snapshot)
    C = np.clip(circularity, 1e-6, 1.0) if circularity is not None else np.ones_like(R_um)
    phi = np.clip(defect_frac, 0.0, 1.0) if defect_frac is not None else np.zeros_like(R_um)
    f = np.exp(-alpha * (1.0 - C)) * np.exp(-beta * phi)
    denom = np.maximum(dt_ms, 1e-9)
    v_unpen = R_um / denom
    v_eff   = (R_um / denom) * f
    return v_eff, v_unpen, f

def effective_area_um2(area_eff_px: float, px_per_um: float) -> float:
    # (um_per_px) = 1/px_per_um; area scale = (um_per_px)^2
    um_per_px = 1.0 / px_per_um
    return float(area_eff_px) * (um_per_px ** 2)

def fit_dn_dt(t0_ms: np.ndarray, area_um2: float, grid_ms: np.ndarray):
    """
    Fit dn/dt by two models (lognormal, gamma) if SciPy available.
    Returns dataframe with columns: t_ms, dn_dt_per_ms_mm2, model
    Units: events / (ms * mm^2)
    """
    tpos = t0_ms[(t0_ms > 0) & np.isfinite(t0_ms)]
    if len(tpos) < 5 or not HAVE_SCIPY:
        # Fallback: histogram estimate only
        bins = np.arange(0, grid_ms.max() + 1.0, 1.0)
        hist, edges = np.histogram(t0_ms, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        # density per ms per um^2 -> convert to per ms per mm^2 (1e6 um^2 per mm^2)
        dn_dt = hist / max(1.0, area_um2) * 1e6
        df = pd.DataFrame({"t_ms": centers,
                           "dn_dt_per_ms_mm2": dn_dt,
                           "model": ["hist"] * len(centers)})
        return df, {"model": "hist"}
    # Fit lognormal on milliseconds
    # SciPy's lognorm(s, loc, scale) with pdf = 1/(s*(t-loc)) * phi( ln((t-loc)/scale)/s )
    # Constrain loc>=0
    with np.errstate(all='ignore'):
        # Lognormal
        ln_params = lognorm.fit(tpos, floc=0)  # (s, loc=0, scale)
        s, loc, scale = ln_params
        ln_pdf = lognorm.pdf(grid_ms, s, loc=loc, scale=scale)
        # Gamma
        gm_params = gamma.fit(tpos, floc=0)    # (k, loc=0, theta)
        k, loc_g, theta = gm_params
        gm_pdf = gamma.pdf(grid_ms, k, loc=loc_g, scale=theta)

    # Convert pdf (1/ms) to density per ms per mm^2 -> normalize by area (um^2 -> mm^2)
    # The number of events is len(tpos). Intensity ≈ N * pdf / area_mm2
    area_mm2 = area_um2 / 1e6
    N = len(t0_ms)
    ln_dn = (N * ln_pdf) / max(area_mm2, 1e-12)
    gm_dn = (N * gm_pdf) / max(area_mm2, 1e-12)

    # AIC/BIC for comparison
    def aic(y, lam):
        # y ~ Poisson-ish around expected intensity? Use SSE proxy on normalized histogram
        # To avoid overkill, use least-squares surrogate
        sse = np.sum((y - lam)**2)
        kpar = 2  # (s, scale) with loc=0; rough
        return 2*kpar + len(y)*math.log(max(sse, 1e-24))
    def bic(y, lam):
        sse = np.sum((y - lam)**2)
        kpar = 2
        return kpar*math.log(len(y)) + len(y)*math.log(max(sse, 1e-24))

    # Build a crude target from histogram for selection
    bins = np.arange(0, grid_ms.max() + 1.0, 1.0)
    hist, edges = np.histogram(t0_ms, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist_dn = hist / max(area_um2, 1.0) * 1e6  # per ms per mm^2

    # Interp model on centers
    ln_on_centers = np.interp(centers, grid_ms, ln_dn)
    gm_on_centers = np.interp(centers, grid_ms, gm_dn)

    aic_ln, bic_ln = aic(hist_dn, ln_on_centers), bic(hist_dn, ln_on_centers)
    aic_gm, bic_gm = aic(hist_dn, gm_on_centers), bic(hist_dn, gm_on_centers)

    if (bic_gm < bic_ln) or (math.isfinite(bic_gm) and not math.isfinite(bic_ln)):
        best = "gamma"
        series = gm_dn
        params = {"model": "gamma", "k": k, "theta": theta, "AIC": aic_gm, "BIC": bic_gm}
    else:
        best = "lognormal"
        series = ln_dn
        params = {"model": "lognormal", "s": s, "scale": scale, "AIC": aic_ln, "BIC": bic_ln}

    df = pd.DataFrame({"t_ms": grid_ms,
                       "dn_dt_per_ms_mm2": series,
                       "model": [best] * len(grid_ms)})
    return df, params

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects_csv", required=True, help="objects_*_aug.csv produced by augmenter")
    ap.add_argument("--label",       required=True, help="Label for outputs (e.g., FAPI)")
    ap.add_argument("--outdir",      required=True, help="Output folder")
    ap.add_argument("--area_eff_px", type=float, required=True, help="Effective observed area in pixels (FOV or convex hull)")
    ap.add_argument("--px_per_um",   type=float, default=2.20014, help="Pixels per micron (default 2.20014 px/um)")
    ap.add_argument("--use_penalties", action="store_true", help="Apply circularity/defect penalties to growth rates")
    ap.add_argument("--alpha", type=float, default=0.0, help="Penalty weight for circularity (if --use_penalties)")
    ap.add_argument("--beta",  type=float, default=0.0, help="Penalty weight for defect fraction (if --use_penalties)")
    ap.add_argument("--n_avrami", type=float, default=2.5, help="Fixed Avrami exponent for overlay fit")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.objects_csv)
    # Required minimal columns
    for col in ("area_px",):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {args.objects_csv}")

    area_px = safe_num(df["area_px"])
    C = safe_num(df["circularity"]) if "circularity" in df.columns else np.full_like(area_px, 1.0)
    phi = safe_num(df["defect_frac"]) if "defect_frac" in df.columns else np.zeros_like(area_px)

    # Nucleation times (rank-based) in 0..60 ms
    t0_ms = rank_to_t0_ms(area_px, t_win_ms=60.0)
    dt_ms = np.clip(600.0 - t0_ms, 1e-9, None)

    # Radii in microns
    R_um = area_px_to_radius_um(area_px, px_per_um=args.px_per_um)

    # Growth rates (per-object). If penalties disabled, alpha=beta=0 effectively.
    alpha = args.alpha if args.use_penalties else 0.0
    beta  = args.beta  if args.use_penalties else 0.0
    v_eff_um_per_ms, v_unpen_um_per_ms, penalty_factor = eff_growth_rate_um_per_ms(R_um, dt_ms, C, phi, alpha, beta)

    # Aggregate v0 estimate (unpenalized median)
    v0_um_per_ms = float(np.nanmedian(v_unpen_um_per_ms))

    # Nucleation dn/dt per (ms·mm^2)
    area_um2 = effective_area_um2(args.area_eff_px, args.px_per_um)
    grid = np.arange(0.0, 60.0 + 1.0, 1.0)  # 1-ms grid
    dn_dt_df, dn_meta = fit_dn_dt(t0_ms, area_um2, grid)

    # Avrami overlay against X_pred(t) (optional quick construction)
    # Build X_pred from discs (extended fraction, ignoring impingement): sum(pi R(t)^2)/A_eff
    # With linear growth r_i(t)=v_eff*(t-t0)+, capped at t>=t0
    t_grid = np.arange(0.0, 600.0 + 1.0, 1.0)
    X_pred = np.zeros_like(t_grid)
    for idx, t in enumerate(t_grid):
        tau = np.clip(t - t0_ms, 0.0, None)
        r_um = np.maximum(0.0, v_eff_um_per_ms * tau)
        A_um2 = math.pi * (r_um ** 2)
        X_pred[idx] = np.nansum(A_um2) / max(area_um2, 1e-12)

    # Fit K in X_Avrami = 1 - exp( -K t^n ) to match X_pred (least squares)
    n = float(args.n_avrami)
    # Only use the growth window where X grows (t>=some small threshold)
    y = np.clip(X_pred, 0.0, 0.999999)
    t_pow = (t_grid ** n)
    # Avoid t=0 (division by 0); solve for K = -ln(1-y)/t^n, take median over valid points
    valid = (t_pow > 0) & (y > 0) & (y < 0.99)
    if np.any(valid):
        K_vals = -np.log(1.0 - y[valid]) / t_pow[valid]
        K_fit = float(np.nanmedian(K_vals[np.isfinite(K_vals)])) if np.any(np.isfinite(K_vals)) else 0.0
    else:
        K_fit = 0.0
    X_A = 1.0 - np.exp(-K_fit * (t_grid ** n))

    # ---------- exports ----------
    label = args.label
    # Per-object metrics
    out_obj = pd.DataFrame({
        "label": label,
        "area_px": area_px,
        "circularity": C,
        "defect_frac": phi,
        "t0_ms": t0_ms,
        "dt_ms": dt_ms,
        "R_um": R_um,
        "v_unpen_um_per_ms": v_unpen_um_per_ms,
        "v_eff_um_per_ms": v_eff_um_per_ms,
        "v_unpen_um_per_s": v_unpen_um_per_ms * 1000.0,
        "v_eff_um_per_s":   v_eff_um_per_ms   * 1000.0,
        "penalty_factor": penalty_factor
    })
    out_obj.to_csv(outdir / f"{label}_SI_metrics.csv", index=False)

    # dn/dt
    dn_dt_df.to_csv(outdir / f"{label}_nucleation_dn_dt.csv", index=False)

    # X_pred and Avrami overlay
    pd.DataFrame({
        "t_ms": t_grid,
        "X_pred": X_pred,
        "X_Avrami": X_A
    }).to_csv(outdir / f"{label}_X_pred_Avrami.csv", index=False)

    # Summary
    meta_rows = [{
        "label": label,
        "px_per_um": args.px_per_um,
        "area_eff_px": args.area_eff_px,
        "area_eff_um2": area_um2,
        "use_penalties": bool(args.use_penalties),
        "alpha": alpha,
        "beta": beta,
        "v0_unpen_median_um_per_ms": v0_um_per_ms,
        "dn_dt_model": dn_meta.get("model", "hist"),
        "dn_dt_params": str({k:v for k,v in dn_meta.items() if k not in ("model",)})
    }]
    pd.DataFrame(meta_rows).to_csv(outdir / f"{label}_growth_summary.csv", index=False)

    # Quick plots (matplotlib, no style/colors)
    try:
        import matplotlib.pyplot as plt

        # dn/dt
        plt.figure()
        plt.plot(dn_dt_df["t_ms"], dn_dt_df["dn_dt_per_ms_mm2"])
        plt.xlabel("t (ms)")
        plt.ylabel("dn/dt  [events / (ms·mm²)]")
        plt.title(f"{label}: nucleation density rate")
        plt.tight_layout()
        plt.savefig(outdir / f"{label}_dn_dt.png", dpi=200)
        plt.close()

        # Growth-rate hist (effective)
        plt.figure()
        valid_v = out_obj["v_eff_um_per_s"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid_v) > 0:
            plt.hist(valid_v, bins=60)
        plt.xlabel("growth rate (µm/s)  [effective]")
        plt.ylabel("count")
        plt.title(f"{label}: growth-rate distribution")
        plt.tight_layout()
        plt.savefig(outdir / f"{label}_growth_hist.png", dpi=200)
        plt.close()

        # X(t)
        plt.figure()
        plt.plot(t_grid, X_pred, label="X_pred (extended)")
        plt.plot(t_grid, X_A,    label=f"Avrami (n={n:.2f}, K={K_fit:.3g})")
        plt.xlabel("t (ms)")
        plt.ylabel("transformed fraction X(t)  [extended]")
        plt.legend()
        plt.title(f"{label}: X(t) vs Avrami overlay")
        plt.tight_layout()
        plt.savefig(outdir / f"{label}_X_overlay.png", dpi=200)
        plt.close()
    except Exception as e:
        print("[WARN] Plotting skipped:", e)

    print(f"[OK] Wrote SI exports in: {outdir}")

if __name__ == "__main__":
    main()
