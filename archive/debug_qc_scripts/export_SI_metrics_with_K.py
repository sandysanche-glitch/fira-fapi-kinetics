#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_SI_metrics_with_K.py

Convert comparative outputs to SI units and fit Avrami K.

Inputs (from your previous pipeline):
- objects_csv: per-object table (area_px, optionally perim_px, circ, defect_frac, t0_ms, centroid/nucleus coords)
- dndt_csv:    nucleation dn/dt on a time grid (ms, arbitrary counts)
- (optional) x_pred_csv: X_pred(t) bulk fraction vs time to fit K

Outputs:
- <label>_growth_SI.csv        : per-object SI metrics (radius_um, v_um_s, etc.)
- <label>_nucleation_SI.csv    : J(t) = dn/dt per s per µm^2
- <label>_growth_summary.csv   : v0, alpha, beta, medians (and Avrami K if x_pred provided)

Run (FAPI example):
  python export_SI_metrics_with_K.py ^
    --objects_csv "D:\...\comparative_outputs_any\objects_FAPI.csv" ^
    --dndt_csv    "D:\...\comparative_outputs_any\dn_dt_FAPI.csv" ^
    --label       FAPI ^
    --outdir      "D:\...\comparative_outputs_any\outputs_SI" ^
    --area_eff_px 2500000 ^
    --x_pred_csv  "D:\...\comparative_outputs_any\X_pred_both.csv" ^
    --x_label     FAPI ^
    --n_avrami    2.5 ^
    --bootstrap   300 ^
    --use_penalties

Note: If x_pred_csv is omitted, K is skipped (still exports growth + J(t) in SI).
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- helpers ----------

def ensure_cols(df: pd.DataFrame, cols, fill=np.nan):
    missing = [c for c in cols if c not in df.columns]
    for c in missing:
        df[c] = fill
    return df

def compute_circularity(area_px, perim_px):
    area = np.asarray(area_px, float)
    perim = np.asarray(perim_px, float)
    ok = (area > 0) & (perim > 0)
    circ = np.full_like(area, np.nan, dtype=float)
    circ[ok] = (4.0*np.pi*area[ok])/(perim[ok]**2)
    # Clamp to (0,1]
    circ = np.clip(circ, 0.0, 1.0)
    return circ

def fallback_ray_px(df):
    """
    If ray_px is missing/NaN, estimate from area: r = sqrt(area/pi)
    If centroid + nucleus present, prefer that for missing rows.
    """
    ray = df.get("ray_px", pd.Series(np.nan, index=df.index)).astype(float).to_numpy()
    # centroid/nucleus option
    if all(c in df.columns for c in ["centroid_x","centroid_y","nuc_x","nuc_y"]):
        cx = pd.to_numeric(df["centroid_x"], errors="coerce").to_numpy()
        cy = pd.to_numeric(df["centroid_y"], errors="coerce").to_numpy()
        nx = pd.to_numeric(df["nuc_x"], errors="coerce").to_numpy()
        ny = pd.to_numeric(df["nuc_y"], errors="coerce").to_numpy()
        ray_cent = np.sqrt((cx-nx)**2 + (cy-ny)**2)
        use = ~np.isfinite(ray)
        ray[use & np.isfinite(ray_cent)] = ray_cent[use & np.isfinite(ray_cent)]
    # area-based fallback
    if "area_px" in df.columns:
        area = pd.to_numeric(df["area_px"], errors="coerce").to_numpy()
        ray_area = np.sqrt(np.maximum(area,0.0)/np.pi)
        use2 = ~np.isfinite(ray)
        ray[use2 & np.isfinite(ray_area)] = ray_area[use2 & np.isfinite(ray_area)]
    return ray

def compute_rank_t0_ms(area_px, t_window_ms=60.0):
    """Rank-to-time mapping in [0, t_window] based on ascending area."""
    a = np.asarray(area_px, float)
    order = np.argsort(a, kind="mergesort")  # stable
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(a))
    N = max(1, len(a)-1)
    q = ranks / float(N)
    return t_window_ms * q

def penalties(C, phi, alpha, beta):
    C = np.clip(np.asarray(C, float), 0, 1)
    phi = np.clip(np.asarray(phi, float), 0, 1)
    return np.exp(-alpha*(1.0 - C)) * np.exp(-beta*phi)

def calibrate_v0_alpha_beta(df, use_penalties: bool, alpha0=0.5, beta0=1.0):
    """
    Estimate base v0 (µm/s) and optionally refine alpha,beta
    using a simple robust statistic from dt = (ray_um) / (v0 * f).
    Here we target median dt ≈ median(600 - t0_ms).
    """
    px_per_um = df.attrs.get("px_per_um", 2.20014)
    ray_px = fallback_ray_px(df)
    ray_um = ray_px / px_per_um
    # t0
    if "t0_ms" in df.columns and df["t0_ms"].notna().any():
        t0_ms = pd.to_numeric(df["t0_ms"], errors="coerce").fillna(60.0).to_numpy()
    else:
        t0_ms = compute_rank_t0_ms(df["area_px"])
    dt_ms = np.clip(600.0 - t0_ms, 1.0, None)
    dt_s  = dt_ms/1000.0

    # morphology
    if "circ" in df.columns and df["circ"].notna().any():
        C = pd.to_numeric(df["circ"], errors="coerce").to_numpy()
    else:
        # try perim to compute circ
        if "perim_px" in df.columns and df["perim_px"].notna().any():
            C = compute_circularity(df["area_px"], df["perim_px"])
        else:
            C = np.full_like(dt_s, 1.0)  # neutral
    if "defect_frac" in df.columns and df["defect_frac"].notna().any():
        phi = pd.to_numeric(df["defect_frac"], errors="coerce").fillna(0.0).to_numpy()
    else:
        phi = np.zeros_like(dt_s)

    alpha, beta = (alpha0, beta0) if use_penalties else (0.0, 0.0)
    f = penalties(C, phi, alpha, beta)

    # v0 from median: ray_um ≈ v0 * f * dt_s
    num = np.nanmedian(ray_um)
    den = np.nanmedian(f*dt_s)
    v0 = float(num/den) if (np.isfinite(num) and np.isfinite(den) and den>0) else 0.0
    return v0, alpha, beta

def t0_from_backcalc(df, v0, alpha, beta):
    px_per_um = df.attrs.get("px_per_um", 2.20014)
    ray_um = fallback_ray_px(df)/px_per_um
    # morphology
    if "circ" in df.columns and df["circ"].notna().any():
        C = pd.to_numeric(df["circ"], errors="coerce").to_numpy()
    else:
        if "perim_px" in df.columns and df["perim_px"].notna().any():
            C = compute_circularity(df["area_px"], df["perim_px"])
        else:
            C = np.ones(len(df), float)
    if "defect_frac" in df.columns and df["defect_frac"].notna().any():
        phi = pd.to_numeric(df["defect_frac"], errors="coerce").fillna(0.0).to_numpy()
    else:
        phi = np.zeros(len(df), float)

    f = penalties(C, phi, alpha, beta)
    dt_s = ray_um / np.maximum(v0*f, 1e-12)
    t0_ms = 600.0 - 1000.0*dt_s
    return np.clip(t0_ms, 0.0, 60.0)

def fit_K_avrami_least_squares(t, X, n, n_boot=0, seed=0):
    """Fit K in X=1-exp(-K t^n) by LS through origin on y=-ln(1-X)=K t^n."""
    t = np.asarray(t, float)
    X = np.asarray(X, float)
    m = (t>0) & (X>0) & (X<0.99)
    if not np.any(m):
        return 0.0, (np.nan, np.nan), 0
    x = t[m]**n
    y = -np.log(1.0 - X[m])
    den = np.dot(x, x)
    K = float(max(0.0, np.dot(x, y)/den)) if den>0 else 0.0
    ci = (np.nan, np.nan)
    used = int(m.sum())
    if n_boot and used>5:
        rng = np.random.default_rng(seed)
        Ks = []
        for _ in range(n_boot):
            idx = rng.integers(0, used, used)
            xb, yb = x[idx], y[idx]
            d = np.dot(xb, xb)
            k = max(0.0, np.dot(xb, yb)/d) if d>0 else 0.0
            Ks.append(k)
        lo, hi = np.nanpercentile(Ks, [2.5, 97.5])
        ci = (float(lo), float(hi))
    return K, ci, used

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects_csv", required=True)
    ap.add_argument("--dndt_csv",    required=True)
    ap.add_argument("--label",       required=True)
    ap.add_argument("--outdir",      required=True)
    ap.add_argument("--area_eff_px", type=float, required=True,
                    help="Effective observation area in pixels (for per-area J(t)).")
    ap.add_argument("--px_per_um",   type=float, default=2.20014,
                    help="Pixels per micron (default: 2.20014 px/µm).")
    ap.add_argument("--use_penalties", action="store_true", help="Apply circ/defect penalties in growth.")
    ap.add_argument("--alpha0", type=float, default=0.5)
    ap.add_argument("--beta0",  type=float, default=1.0)
    ap.add_argument("--n_avrami", type=float, default=2.5)
    ap.add_argument("--x_pred_csv", default=None,
                    help="Optional CSV with columns: t_ms, X_pred_<label>. Fits K if provided.")
    ap.add_argument("--x_label", default=None, help="Column suffix for X_pred in x_pred_csv (e.g., FAPI).")
    ap.add_argument("--bootstrap", type=int, default=300, help="Bootstrap resamples for K CI.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- load objects and prepare ---
    obj = pd.read_csv(args.objects_csv)
    obj.attrs["px_per_um"] = args.px_per_um

    # Ensure needed columns (fill if missing)
    obj = ensure_cols(obj, ["area_px"], np.nan)
    obj = ensure_cols(obj, ["perim_px","circ","defect_frac","t0_ms",
                            "centroid_x","centroid_y","nuc_x","nuc_y","ray_px"], np.nan)

    # circularity if missing but perim present
    if obj["circ"].isna().all() and obj["perim_px"].notna().any():
        obj["circ"] = compute_circularity(obj["area_px"], obj["perim_px"])
    # defect frac default 0
    obj["defect_frac"] = pd.to_numeric(obj["defect_frac"], errors="coerce").fillna(0.0)

    # t0: use provided if non-empty; else rank mapping
    if obj["t0_ms"].notna().any():
        t0_ms = pd.to_numeric(obj["t0_ms"], errors="coerce").fillna(60.0).to_numpy()
    else:
        t0_ms = compute_rank_t0_ms(obj["area_px"])

    # Calibrate v0, alpha, beta
    v0, alpha, beta = calibrate_v0_alpha_beta(obj, use_penalties=args.use_penalties,
                                              alpha0=args.alpha0, beta0=args.beta0)

    # Per-object SI metrics
    px_per_um = args.px_per_um
    area_px = pd.to_numeric(obj["area_px"], errors="coerce").to_numpy()
    radius_px = np.sqrt(np.maximum(area_px,0.0)/np.pi)
    radius_um = radius_px / px_per_um

    C = pd.to_numeric(obj["circ"], errors="coerce").fillna(1.0).to_numpy()
    phi = pd.to_numeric(obj["defect_frac"], errors="coerce").fillna(0.0).to_numpy()
    f = penalties(C, phi, alpha if args.use_penalties else 0.0, beta if args.use_penalties else 0.0)

    dt_ms = np.clip(600.0 - t0_ms, 1.0, None)
    v_um_s = radius_um / (dt_ms/1000.0)               # effective observed average
    v0_um_s = v_um_s / np.maximum(f, 1e-12)           # base (de-penalized) estimate per object

    growth_out = pd.DataFrame({
        "label":     args.label,
        "area_px":   area_px,
        "radius_um": radius_um,
        "t0_ms":     t0_ms,
        "dt_ms":     dt_ms,
        "circ":      C,
        "defect_frac": phi,
        "f_penalty": f,
        "v_um_s":    v_um_s,
        "v0_um_s_est": v0_um_s,
    })
    growth_out.to_csv(outdir/f"{args.label}_growth_SI.csv", index=False)

    # Summary
    v0_median = float(np.nanmedian(v0_um_s))
    v_eff_median = float(np.nanmedian(v_um_s))
    summary = {
        "label": args.label,
        "px_per_um": px_per_um,
        "area_eff_px": args.area_eff_px,
        "area_eff_um2": args.area_eff_px/(px_per_um**2),
        "use_penalties": int(args.use_penalties),
        "alpha": alpha if args.use_penalties else 0.0,
        "beta":  beta if args.use_penalties else 0.0,
        "v0_um_s_median": v0_median,
        "v_eff_um_s_median": v_eff_median
    }

    # --- nucleation SI: convert dn/dt to J(t) ---
    dndt = pd.read_csv(args.dndt_csv)
    # Expect at least columns: t_ms, dn_dt
    # If column names differ, try to infer
    tcol = "t_ms"
    if tcol not in dndt.columns:
        # try 't' or first col
        if "t" in dndt.columns: tcol = "t"
        else: tcol = dndt.columns[0]
    ycol = "dn_dt"
    if ycol not in dndt.columns:
        # try label-specific
        cand = [c for c in dndt.columns if "dn_dt" in c]
        if cand: ycol = cand[0]
        else: ycol = dndt.columns[1] if len(dndt.columns)>1 else dndt.columns[0]

    t_ms = pd.to_numeric(dndt[tcol], errors="coerce").to_numpy()
    dn_dt_counts_per_ms = pd.to_numeric(dndt[ycol], errors="coerce").fillna(0.0).to_numpy()

    # Convert to per-area per-second:
    # J(t) [events s^-1 µm^-2] = (dn/dt [events/ms over FOV]) * (1000 ms/s) / (A_eff [µm^2])
    A_eff_um2 = summary["area_eff_um2"]
    J_t = (dn_dt_counts_per_ms * 1000.0) / max(A_eff_um2, 1e-12)

    nuc_out = pd.DataFrame({
        "label": args.label,
        "t_ms": t_ms,
        "t_s":  t_ms/1000.0,
        "dn_dt_counts_per_ms": dn_dt_counts_per_ms,
        "J_t_events_per_s_um2": J_t
    })
    nuc_out.to_csv(outdir/f"{args.label}_nucleation_SI.csv", index=False)

    # --- optional: fit Avrami K if X_pred is available ---
    if args.x_pred_csv:
        Xall = pd.read_csv(args.x_pred_csv)
        # Expect columns: t_ms, X_pred_<label>
        xcol = f"X_pred_{args.label}"
        if xcol not in Xall.columns:
            # try case-insensitive
            m = [c for c in Xall.columns if c.lower()==xcol.lower()]
            if m: xcol = m[0]
            else:
                # try generic 'X_pred' or second col
                if "X_pred" in Xall.columns:
                    xcol = "X_pred"
                else:
                    # last resort: pick the second numeric col
                    nums = [c for c in Xall.columns if c!=tcol and np.issubdtype(Xall[c].dtype, np.number)]
                    if nums: xcol = nums[0]
                    else: xcol = Xall.columns[1] if len(Xall.columns)>1 else Xall.columns[0]

        # Align by t_ms
        # Coerce both to numeric
        Xt = pd.to_numeric(Xall.get("t_ms", Xall.columns[0]), errors="coerce").to_numpy()
        Xv = pd.to_numeric(Xall[xcol], errors="coerce").to_numpy()

        # fit in ms-units
        n = float(args.n_avrami)
        K_ms, (Klo_ms, Khi_ms), used_ms = fit_K_avrami_least_squares(Xt, Xv, n, n_boot=args.bootstrap, seed=42)
        # also report in seconds (units change)
        K_s, (Klo_s, Khi_s), used_s = fit_K_avrami_least_squares(Xt/1000.0, Xv, n, n_boot=args.bootstrap, seed=42)

        summary.update({
            "n_avrami": n,
            "K_fit_ms_units": K_ms, "K_ci_ms_lo": Klo_ms, "K_ci_ms_hi": Khi_ms, "K_points_used": used_ms,
            "K_fit_s_units":  K_s,  "K_ci_s_lo":  Klo_s,  "K_ci_s_hi":  Khi_s,  "K_points_used_s": used_s
        })

    # Save summary
    pd.DataFrame([summary]).to_csv(outdir/f"{args.label}_growth_summary.csv", index=False)

    print(f"[OK] Wrote:\n  {outdir / (args.label + '_growth_SI.csv')}\n  {outdir / (args.label + '_nucleation_SI.csv')}")
    if args.x_pred_csv:
        print(f"  {outdir / (args.label + '_growth_summary.csv')} (includes Avrami K)")

if __name__ == "__main__":
    main()
