#!/usr/bin/env python
"""
compare_SI_from_json_generalKinetics.py

Directly read FAPI / FAPI-TEMPO JSON detection files (Mask R-CNN-style),
compute SI metrics, and fit kinetic models to the reconstructed bulk
transformed fraction X_pred(t):

  1) Ideal Avrami (fixed n):
        X_ideal(t) = X_inf * (1 - exp(-K * t^n))

  2) Shifted Avrami (still fixed n, but with an incubation time):
        X_shift(t) = X_inf_s * (1 - exp(-K_s * (t - t_shift)^n))  for t > t_shift
                     0                                         for t <= t_shift

n is fixed (typically 2.5); K, K_s, X_inf, X_inf_s and t_shift are fitted.

Defaults:
    FAPI JSONs:
        D:\\SWITCHdrive\\Institution\\Sts_grain morphology_ML\\comparative datasets\\FAPI
    FAPI-TEMPO JSONs:
        D:\\SWITCHdrive\\Institution\\Sts_grain morphology_ML\\comparative datasets\\FAPI-TEMPO

Outputs (in --out):
    combined_dn_dt.csv
    combined_X_pred_models.csv
    combined_growth_hist.csv
    combined_kinetic_fits.csv

    dn_dt_both_ms.png
    X_overlay_both_ms_ideal.png
    X_overlay_both_ms_shifted.png
    X_overlay_both_s_ideal.png
    X_overlay_both_s_shifted.png
    growth_hist_both.png
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# ---- requires pycocotools to decode RLE segmentation ----
try:
    from pycocotools import mask as maskUtils
except Exception as e:
    raise ImportError(
        "pycocotools is required for this script.\n"
        "Install with:  pip install pycocotools  (or pycocotools-windows on Windows)\n"
        f"Original error: {e}"
    )

# ---- SciPy (for dn/dt and shifted-Avrami fit) ----
try:
    from scipy.stats import lognorm, gamma
    from scipy.optimize import minimize
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# ---------- helpers (adapted from export_SI_metrics_rank_v2.py) ----------

def rank_to_t0_ms(area_px: np.ndarray, t_win_ms: float = 60.0) -> np.ndarray:
    """
    Map ranked areas onto [0, t_win_ms] to assign a nucleation time t0_ms.
    Larger final area => earlier nucleation.
    """
    n = len(area_px)
    if n <= 1:
        return np.full_like(area_px, fill_value=t_win_ms * 0.5, dtype=float)
    order = np.argsort(area_px)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)
    q = ranks / max(n - 1, 1)
    return t_win_ms * q


def area_px_to_radius_um(area_px: np.ndarray, px_per_um: float) -> np.ndarray:
    """
    Convert area in pixels to effective radius in microns, assuming disks.

    Example: 2.20014 px = 1 µm → px_per_um = 2.20014.
    """
    R_px = np.sqrt(np.maximum(area_px, 0.0) / math.pi)
    return R_px / px_per_um


def eff_growth_rate_um_per_ms(R_um: np.ndarray, dt_ms: np.ndarray,
                              circularity: np.ndarray = None,
                              defect_frac: np.ndarray = None,
                              alpha: float = 0.0, beta: float = 0.0):
    """
    Effective growth rate with optional penalties (circularity, defect).

    For now circularity and defect_frac are trivial (1, 0), but the framework
    allows future inclusion of real metrics.
    """
    C = np.clip(circularity, 1e-6, 1.0) if circularity is not None else np.ones_like(R_um)
    phi = np.clip(defect_frac, 0.0, 1.0) if defect_frac is not None else np.zeros_like(R_um)
    f = np.exp(-alpha * (1.0 - C)) * np.exp(-beta * phi)
    denom = np.maximum(dt_ms, 1e-9)
    v_unpen = R_um / denom
    v_eff = v_unpen * f
    return v_eff, v_unpen, f


def effective_area_um2(area_eff_px: float, px_per_um: float) -> float:
    """
    Convert effective area in pixels to µm².
    """
    um_per_px = 1.0 / px_per_um
    return float(area_eff_px) * (um_per_px ** 2)


# ---- dn/dt model ----

def fit_dn_dt(t0_ms: np.ndarray, area_um2: float, grid_ms: np.ndarray):
    """
    Fit dn/dt as either histogram, lognormal, or gamma model over time.
    Returns (df, meta).
    """
    tpos = t0_ms[(t0_ms > 0) & np.isfinite(t0_ms)]
    if len(tpos) < 5 or not HAVE_SCIPY:
        # fallback: histogram-only model
        bins = np.arange(0, grid_ms.max() + 1.0, 1.0)
        hist, edges = np.histogram(t0_ms, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        dn_dt = hist / max(1.0, area_um2) * 1e6  # per (ms·mm^2)
        df = pd.DataFrame({"t_ms": centers,
                           "dn_dt_per_ms_mm2": dn_dt,
                           "model": ["hist"] * len(centers)})
        return df, {"model": "hist"}

    # fit lognormal & gamma
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

    # compare on histogram support
    def score(y, lam, kpar):
        sse = np.sum((y - lam) ** 2)
        return 2 * kpar + len(y) * math.log(max(sse, 1e-24)), \
               kpar * math.log(len(y)) + len(y) * math.log(max(sse, 1e-24))

    bins = np.arange(0, grid_ms.max() + 1.0, 1.0)
    hist, edges = np.histogram(t0_ms, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist_dn = hist / max(area_um2, 1.0) * 1e6

    ln_on = np.interp(centers, grid_ms, ln_dn)
    gm_on = np.interp(centers, grid_ms, gm_dn)
    aic_ln, bic_ln = score(hist_dn, ln_on, kpar=2)
    aic_gm, bic_gm = score(hist_dn, gm_on, kpar=2)

    if bic_gm < bic_ln:
        params = {"model": "gamma", "k": k, "theta": theta, "AIC": aic_gm, "BIC": bic_gm}
        series = gm_dn
        best = "gamma"
    else:
        params = {"model": "lognormal", "s": s, "scale": scale, "AIC": aic_ln, "BIC": bic_ln}
        series = ln_dn
        best = "lognormal"

    df = pd.DataFrame({"t_ms": grid_ms,
                       "dn_dt_per_ms_mm2": series,
                       "model": [best] * len(grid_ms)})
    return df, params


# ---- Ideal Avrami fit (fixed n) in Avrami coordinates ----

def fit_K_Xinf_from_X_avrami_coords(t_ms: np.ndarray, X_pred: np.ndarray, n: float,
                                    x_min: float = 0.05, x_max: float = 0.95):
    """
    Fit K and X_inf in Avrami:

        X_ideal(t) = X_inf * (1 - exp(-K * t^n))

    with fixed n, by linear regression in Avrami coordinates:

        Y = ln[-ln(1 - X/X_inf)] = ln K + n ln t

    We:
        - take X_inf = max(X_pred)
        - restrict to x_min <= X/X_inf <= x_max
        - compute ln K = mean(Y - n ln t)

    Time is in ms → K has units of ms^-n.
    Returns:
        X_inf_fit, K_fit
    """
    t_ms = np.asarray(t_ms, dtype=float)
    X_pred = np.asarray(X_pred, dtype=float)

    X_inf = float(np.nanmax(X_pred))
    if X_inf <= 0:
        return 1.0, 0.0

    y_rel = np.clip(X_pred / X_inf, 1e-9, 1.0 - 1e-9)
    valid = (t_ms > 0) & (y_rel >= x_min) & (y_rel <= x_max)
    t = t_ms[valid]
    yv = y_rel[valid]

    if t.size < 5:
        return X_inf, 0.0

    with np.errstate(all='ignore'):
        Y = np.log(-np.log(1.0 - yv))   # ln[-ln(1 - X/X_inf)]
        Xlog = np.log(t)                # ln t

    good = np.isfinite(Y) & np.isfinite(Xlog)
    Y = Y[good]
    Xlog = Xlog[good]

    if Y.size < 3:
        return X_inf, 0.0

    lnK = float(np.mean(Y - n * Xlog))
    K_fit = float(np.exp(lnK))
    return X_inf, K_fit


# ---- Shifted Avrami fit (fixed n) by least squares ----

def fit_shifted_avrami(t_ms: np.ndarray, X_pred: np.ndarray, n: float,
                       x_min: float = 0.05, x_max: float = 0.95):
    """
    Fit shifted Avrami:

        X_shift(t) = X_inf_s * (1 - exp(-K_s * (t - t_shift)^n)) for t > t_shift
                     0                                          for t <= t_shift

    n is fixed. We fit X_inf_s, K_s, t_shift by least squares vs X_pred(t),
    restricting to the region where X_pred / max(X_pred) is between x_min and x_max.

    Returns:
        X_inf_shift, K_shift, t_shift_ms
    """
    t_ms = np.asarray(t_ms, dtype=float)
    X_pred = np.asarray(X_pred, dtype=float)

    if not HAVE_SCIPY:
        # fallback: no shifted fit
        X_inf_data = float(np.nanmax(X_pred))
        return X_inf_data, 0.0, 0.0

    X_inf_data = float(np.nanmax(X_pred))
    if X_inf_data <= 0:
        return 1.0, 0.0, 0.0

    y_rel = np.clip(X_pred / X_inf_data, 0.0, 1.0)
    valid = (t_ms > 0) & (y_rel >= x_min) & (y_rel <= x_max)
    t = t_ms[valid]
    y = X_pred[valid]

    if t.size < 5:
        return X_inf_data, 0.0, 0.0

    # Initial guess from ideal Avrami (no shift)
    X_inf0, K0 = fit_K_Xinf_from_X_avrami_coords(t_ms, X_pred, n, x_min, x_max)
    if K0 <= 0:
        K0 = 1e-6
    if X_inf0 <= 0:
        X_inf0 = X_inf_data

    logK0 = math.log(K0)
    logX0 = math.log(X_inf0)
    t_shift0 = 0.0

    t_max = float(np.nanmax(t_ms))

    def sse(params):
        logK, logXinf, t_shift = params
        K = math.exp(logK)
        X_inf = math.exp(logXinf)
        # ensure non-negative shift and not too large
        if t_shift < 0 or t_shift >= t_max:
            return 1e30
        tau = np.clip(t - t_shift, 0.0, None)
        X_model = X_inf * (1.0 - np.exp(-K * (tau ** n)))
        return float(np.sum((X_model - y) ** 2))

    bounds = [
        (logK0 - 5, logK0 + 5),
        (math.log(max(X_inf_data * 0.5, 1e-6)), math.log(X_inf_data * 2.0)),
        (0.0, t_max * 0.8),
    ]
    x0 = [logK0, logX0, t_shift0]

    res = minimize(sse, x0=x0, bounds=bounds, method="L-BFGS-B")
    if not res.success:
        return X_inf0, K0, 0.0

    logK_opt, logX_opt, t_shift_opt = res.x
    K_opt = float(math.exp(logK_opt))
    Xinf_opt = float(math.exp(logX_opt))
    return Xinf_opt, K_opt, float(t_shift_opt)


# ---------- JSON -> area_px ----------

def collect_areas_from_json_folder(folder: Path, min_score: float = 0.0) -> np.ndarray:
    """
    Parse all *.json files in `folder` and return an array of areas in pixels
    from the RLE segmentation (pycocotools).
    Assumes top-level JSON structure is a list of objects with:
        - 'segmentation': {'size': [H, W], 'counts': <RLE string>}
        - 'score' (optional)
    """
    all_areas = []

    json_files = sorted(folder.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {folder}")

    print(f"[INFO] Scanning {len(json_files)} JSON(s) in {folder} ...")

    for jf in json_files:
        with jf.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"[WARN] {jf} is not a list; skipping.")
            continue

        for obj in data:
            score = obj.get("score", 1.0)
            if score < min_score:
                continue
            seg = obj.get("segmentation", None)
            if seg is None:
                continue

            try:
                rle = {
                    "size": seg["size"],
                    "counts": seg["counts"]
                }
                area = float(maskUtils.area(rle))
                if area > 0:
                    all_areas.append(area)
            except Exception as e:
                print(f"[WARN] Failed to compute area in {jf}: {e}")

    if not all_areas:
        raise ValueError(f"No valid objects found in {folder} (after score filter).")

    areas = np.asarray(all_areas, dtype=float)
    print(f"[INFO] Collected {len(areas)} objects from {folder}")
    return areas


# ---------- main per-dataset SI computation ----------

def compute_si_from_areas(area_px: np.ndarray,
                          label: str,
                          px_per_um: float,
                          area_eff_px: float,
                          n_avrami: float,
                          x_min_fit: float = 0.05,
                          x_max_fit: float = 0.95,
                          use_penalties: bool = False,
                          alpha: float = 0.0,
                          beta: float = 0.0,
                          site_saturated_X: bool = False):
    """
    Compute SI metrics from area_px only (rank-based nucleation, growth rates, X(t)).
    Returns:
        out_obj_df: per-object metrics
        dndt_df: dn/dt vs t
        X_df: X_pred, ideal Avrami and shifted Avrami vs time
        params: dict with fitted kinetic parameters
    """

    # Penalties: C=1, phi=0 for now, but kept for future extension
    C = np.ones_like(area_px, dtype=float)
    phi = np.zeros_like(area_px, dtype=float)

    # nucleation times based on ranks (for dn/dt and growth-rate estimation)
    t0_ms = rank_to_t0_ms(area_px, t_win_ms=60.0)

    # duration available for growth (based on real t0)
    dt_ms_v = np.clip(600.0 - t0_ms, 1e-9, None)
    R_um = area_px_to_radius_um(area_px, px_per_um=px_per_um)

    alpha_eff = alpha if use_penalties else 0.0
    beta_eff = beta if use_penalties else 0.0
    v_eff_um_ms, v_unp_um_ms, pf = eff_growth_rate_um_per_ms(
        R_um, dt_ms_v, C, phi, alpha=alpha_eff, beta=beta_eff
    )

    area_um2 = effective_area_um2(area_eff_px, px_per_um=px_per_um)

    # dn/dt uses the actual t0_ms distribution
    grid_ms = np.arange(0.0, 60.0 + 1.0, 1.0)
    dndt_df, dn_meta = fit_dn_dt(t0_ms, area_um2, grid_ms)

    # X(t) reconstruction: optionally site-saturated test
    if site_saturated_X:
        t0_for_X = np.zeros_like(t0_ms)
    else:
        t0_for_X = t0_ms

    t_ms = np.arange(0.0, 600.0 + 1.0, 1.0)
    t_s = t_ms / 1000.0
    X_pred = np.zeros_like(t_ms, dtype=float)
    for i, t in enumerate(t_ms):
        tau = np.clip(t - t0_for_X, 0.0, None)
        r_um = np.maximum(0.0, v_eff_um_ms * tau)
        A_um2 = math.pi * (r_um ** 2)
        X_pred[i] = np.nansum(A_um2) / max(area_um2, 1e-12)

    # Fit ideal Avrami (fixed n) in Avrami coordinates
    n = float(n_avrami)
    Xinf_ideal, K_ideal = fit_K_Xinf_from_X_avrami_coords(
        t_ms, X_pred, n=n, x_min=x_min_fit, x_max=x_max_fit
    )
    X_ideal = Xinf_ideal * (1.0 - np.exp(-K_ideal * (t_ms ** n)))

    # Fit shifted Avrami (fixed n) by least squares
    Xinf_shift, K_shift, t_shift = fit_shifted_avrami(
        t_ms, X_pred, n=n, x_min=x_min_fit, x_max=x_max_fit
    )
    tau_shift = np.clip(t_ms - t_shift, 0.0, None)
    X_shift = Xinf_shift * (1.0 - np.exp(-K_shift * (tau_shift ** n)))

    # Per-object metrics
    out_obj_df = pd.DataFrame({
        "label": label,
        "area_px": area_px,
        "circularity": C,
        "defect_frac": phi,
        "t0_ms": t0_ms,
        "dt_ms_for_v": dt_ms_v,
        "R_um": R_um,
        "v_unpen_um_per_ms": v_unp_um_ms,
        "v_eff_um_per_ms": v_eff_um_ms,
        "v_unpen_um_per_s": v_unp_um_ms * 1000.0,
        "v_eff_um_per_s": v_eff_um_ms * 1000.0,
        "penalty_factor": pf,
    })

    X_df = pd.DataFrame({
        "t_ms": t_ms,
        "t_s": t_s,
        "X_pred": X_pred,
        "X_Avrami_ideal": X_ideal,
        "X_Avrami_shifted": X_shift,
    })

    params = {
        "n_avrami": n,
        "X_inf_ideal": Xinf_ideal,
        "K_ideal_ms_units": K_ideal,
        "X_inf_shifted": Xinf_shift,
        "K_shifted_ms_units": K_shift,
        "t_shift_ms": t_shift,
        "site_saturated_X": bool(site_saturated_X),
    }

    return out_obj_df, dndt_df, X_df, params


# ---------- main script ----------

def main():
    ap = argparse.ArgumentParser(
        description="Compare FAPI vs FAPI-TEMPO from JSON RLE masks, fitting ideal and shifted Avrami kinetics to X_pred(t)."
    )
    ap.add_argument(
        "--fapi_dir",
        default=r"D:\SWITCHdrive\Institution\Sts_grain morphology_ML\comparative datasets\FAPI",
        help="Folder with FAPI JSON detection files",
    )
    ap.add_argument(
        "--fapitempo_dir",
        default=r"D:\SWITCHdrive\Institution\Sts_grain morphology_ML\comparative datasets\FAPI-TEMPO",
        help="Folder with FAPI-TEMPO JSON detection files",
    )
    ap.add_argument(
        "--out",
        default="combined_out",
        help="Output folder for combined CSVs and plots",
    )
    ap.add_argument(
        "--labelA",
        default="FAPI",
        help="Label for dataset A (FAPI)",
    )
    ap.add_argument(
        "--labelB",
        default="FAPI-TEMPO",
        help="Label for dataset B (FAPI-TEMPO)",
    )
    ap.add_argument(
        "--px_per_um",
        type=float,
        default=2.20014,
        help="Pixels per micron (2.20014 px = 1 µm by default)",
    )
    ap.add_argument(
        "--area_eff_px",
        type=float,
        default=None,
        help="Effective analyzed area in pixels. "
             "If not set, estimated as full image area from first JSON in FAPI.",
    )
    ap.add_argument(
        "--n_avrami",
        type=float,
        default=2.5,
        help="Fixed Avrami exponent n used for both datasets (default: 2.5).",
    )
    ap.add_argument(
        "--x_min_fit",
        type=float,
        default=0.05,
        help="Lower X/X_inf threshold for kinetic fits (e.g. 0.05 = 5% transformed).",
    )
    ap.add_argument(
        "--x_max_fit",
        type=float,
        default=0.95,
        help="Upper X/X_inf threshold for kinetic fits (e.g. 0.95 = 95% transformed).",
    )
    ap.add_argument(
        "--min_score",
        type=float,
        default=0.0,
        help="Minimum detection score to include an object (default: 0.0)",
    )
    ap.add_argument(
        "--use_penalties",
        action="store_true",
        help="Enable penalties based on circularity/defect_frac (currently placeholders: C=1, phi=0).",
    )
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument(
        "--site_saturated_X",
        action="store_true",
        help="If set, reconstruct X_pred(t) assuming all grains start at t=0 "
             "(site-saturated test for X), while dn/dt still uses real t0.",
    )

    args = ap.parse_args()

    fapi_dir = Path(args.fapi_dir)
    fapitempo_dir = Path(args.fapitempo_dir)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    labelA = args.labelA
    labelB = args.labelB

    # Collect areas from JSON for both datasets
    areaA = collect_areas_from_json_folder(fapi_dir, min_score=args.min_score)
    areaB = collect_areas_from_json_folder(fapitempo_dir, min_score=args.min_score)

    # Estimate effective area if not given: full frame from first JSON in FAPI
    if args.area_eff_px is None:
        first_json = sorted(fapi_dir.glob("*.json"))[0]
        with first_json.open("r", encoding="utf-8") as f:
            data0 = json.load(f)
        if not data0:
            raise ValueError(f"First JSON file {first_json} is empty.")
        seg0 = data0[0]["segmentation"]
        H, W = seg0["size"]
        area_eff_px = float(H * W)
        print(f"[INFO] Estimated area_eff_px = {area_eff_px:.1f} from image size {H}x{W}")
    else:
        area_eff_px = float(args.area_eff_px)
        print(f"[INFO] Using user-specified area_eff_px = {area_eff_px:.1f}")

    # Compute SI metrics and kinetic fits for each dataset
    outA, dnA, XA, parA = compute_si_from_areas(
        areaA, labelA,
        px_per_um=args.px_per_um,
        area_eff_px=area_eff_px,
        n_avrami=args.n_avrami,
        x_min_fit=args.x_min_fit,
        x_max_fit=args.x_max_fit,
        use_penalties=args.use_penalties,
        alpha=args.alpha,
        beta=args.beta,
        site_saturated_X=args.site_saturated_X,
    )
    outB, dnB, XB, parB = compute_si_from_areas(
        areaB, labelB,
        px_per_um=args.px_per_um,
        area_eff_px=area_eff_px,
        n_avrami=args.n_avrami,
        x_min_fit=args.x_min_fit,
        x_max_fit=args.x_max_fit,
        use_penalties=args.use_penalties,
        alpha=args.alpha,
        beta=args.beta,
        site_saturated_X=args.site_saturated_X,
    )

    # ---------- Combined CSVs ----------

    # 1) Combined dn/dt
    dn_comb = pd.merge(
        dnA[["t_ms", "dn_dt_per_ms_mm2"]].rename(
            columns={"dn_dt_per_ms_mm2": f"dn_dt_per_ms_mm2_{labelA}"}),
        dnB[["t_ms", "dn_dt_per_ms_mm2"]].rename(
            columns={"dn_dt_per_ms_mm2": f"dn_dt_per_ms_mm2_{labelB}"}),
        on="t_ms",
        how="outer",
        sort=True,
    )
    dn_comb.to_csv(outdir / "combined_dn_dt.csv", index=False)

    # 2) Combined X_pred + kinetic models
    X_comb = pd.DataFrame({
        "t_ms": XA["t_ms"],
        "t_s": XA["t_s"],
        f"X_pred_{labelA}": XA["X_pred"],
        f"X_AvramiIdeal_{labelA}": XA["X_Avrami_ideal"],
        f"X_AvramiShifted_{labelA}": XA["X_Avrami_shifted"],
    })
    X_comb = pd.merge(
        X_comb,
        XB[["t_ms", "X_pred", "X_Avrami_ideal", "X_Avrami_shifted"]].rename(
            columns={
                "X_pred": f"X_pred_{labelB}",
                "X_Avrami_ideal": f"X_AvramiIdeal_{labelB}",
                "X_Avrami_shifted": f"X_AvramiShifted_{labelB}",
            }
        ),
        on="t_ms",
        how="outer",
        sort=True,
    )
    X_comb["t_s"] = X_comb["t_ms"].to_numpy(dtype=float) / 1000.0
    X_comb.to_csv(outdir / "combined_X_pred_models.csv", index=False)

    # 3) Combined growth-rate histogram
    vA = outA.get("v_eff_um_per_s", pd.Series([], dtype=float))
    vB = outB.get("v_eff_um_per_s", pd.Series([], dtype=float))

    vA = pd.to_numeric(vA, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    vB = pd.to_numeric(vB, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)

    if vA.size > 0 or vB.size > 0:
        if vA.size > 0 and vB.size > 0:
            v_all = np.concatenate([vA, vB])
        else:
            v_all = vA if vA.size > 0 else vB

        v_min, v_max = float(np.nanmin(v_all)), float(np.nanmax(v_all))
        if v_min == v_max:
            v_min, v_max = 0.0, (v_max * 1.1 if v_max > 0 else 1.0)
        bins = np.linspace(v_min, v_max, 61)

        def hist_df(v, label):
            if v.size == 0:
                return pd.DataFrame(columns=["label", "bin_left", "bin_right", "count"])
            counts, edges = np.histogram(v, bins=bins)
            return pd.DataFrame({
                "label": label,
                "bin_left": edges[:-1],
                "bin_right": edges[1:],
                "count": counts,
            })

        histA = hist_df(vA, labelA)
        histB = hist_df(vB, labelB)
        hist_comb = pd.concat([histA, histB], ignore_index=True)
        hist_comb.to_csv(outdir / "combined_growth_hist.csv", index=False)
    else:
        hist_comb = pd.DataFrame(columns=["label", "bin_left", "bin_right", "count"])
        hist_comb.to_csv(outdir / "combined_growth_hist.csv", index=False)

    # 4) Combined kinetic fits parameters
    parA_row = {"label": labelA}
    parA_row.update(parA)
    parB_row = {"label": labelB}
    parB_row.update(parB)
    fits_df = pd.DataFrame([parA_row, parB_row])
    fits_df.to_csv(outdir / "combined_kinetic_fits.csv", index=False)

    # ---------- Plots ----------

    try:
        import matplotlib.pyplot as plt

        n = args.n_avrami

        # X overlay (ms): ideal Avrami
        plt.figure()
        plt.plot(X_comb["t_ms"], X_comb[f"X_pred_{labelA}"], label=f"{labelA} X_pred")
        plt.plot(X_comb["t_ms"], X_comb[f"X_AvramiIdeal_{labelA}"],
                 linestyle="--",
                 label=f"{labelA} ideal Avrami (n={n:.2f})")
        plt.plot(X_comb["t_ms"], X_comb[f"X_pred_{labelB}"], label=f"{labelB} X_pred")
        plt.plot(X_comb["t_ms"], X_comb[f"X_AvramiIdeal_{labelB}"],
                 linestyle="--",
                 label=f"{labelB} ideal Avrami (n={n:.2f})")
        plt.xlabel("t (ms)")
        plt.ylabel("X(t) (a.u.)")
        plt.title("Bulk transformed fraction: X_pred vs ideal Avrami")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "X_overlay_both_ms_ideal.png", dpi=200)
        plt.close()

        # X overlay (ms): shifted Avrami
        plt.figure()
        plt.plot(X_comb["t_ms"], X_comb[f"X_pred_{labelA}"], label=f"{labelA} X_pred")
        plt.plot(X_comb["t_ms"], X_comb[f"X_AvramiShifted_{labelA}"],
                 linestyle="--",
                 label=f"{labelA} shifted Avrami (n={n:.2f})")
        plt.plot(X_comb["t_ms"], X_comb[f"X_pred_{labelB}"], label=f"{labelB} X_pred")
        plt.plot(X_comb["t_ms"], X_comb[f"X_AvramiShifted_{labelB}"],
                 linestyle="--",
                 label=f"{labelB} shifted Avrami (n={n:.2f})")
        plt.xlabel("t (ms)")
        plt.ylabel("X(t) (a.u.)")
        plt.title("Bulk transformed fraction: X_pred vs shifted Avrami")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "X_overlay_both_ms_shifted.png", dpi=200)
        plt.close()

        # X overlay (s): ideal Avrami
        plt.figure()
        plt.plot(X_comb["t_s"], X_comb[f"X_pred_{labelA}"], label=f"{labelA} X_pred")
        plt.plot(X_comb["t_s"], X_comb[f"X_AvramiIdeal_{labelA}"],
                 linestyle="--",
                 label=f"{labelA} ideal Avrami (n={n:.2f})")
        plt.plot(X_comb["t_s"], X_comb[f"X_pred_{labelB}"], label=f"{labelB} X_pred")
        plt.plot(X_comb["t_s"], X_comb[f"X_AvramiIdeal_{labelB}"],
                 linestyle="--",
                 label=f"{labelB} ideal Avrami (n={n:.2f})")
        plt.xlabel("t (s)")
        plt.ylabel("X(t) (a.u.)")
        plt.title("Bulk transformed fraction: X_pred vs ideal Avrami (seconds)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "X_overlay_both_s_ideal.png", dpi=200)
        plt.close()

        # X overlay (s): shifted Avrami
        plt.figure()
        plt.plot(X_comb["t_s"], X_comb[f"X_pred_{labelA}"], label=f"{labelA} X_pred")
        plt.plot(X_comb["t_s"], X_comb[f"X_AvramiShifted_{labelA}"],
                 linestyle="--",
                 label=f"{labelA} shifted Avrami (n={n:.2f})")
        plt.plot(X_comb["t_s"], X_comb[f"X_pred_{labelB}"], label=f"{labelB} X_pred")
        plt.plot(X_comb["t_s"], X_comb[f"X_AvramiShifted_{labelB}"],
                 linestyle="--",
                 label=f"{labelB} shifted Avrami (n={n:.2f})")
        plt.xlabel("t (s)")
        plt.ylabel("X(t) (a.u.)")
        plt.title("Bulk transformed fraction: X_pred vs shifted Avrami (seconds)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "X_overlay_both_s_shifted.png", dpi=200)
        plt.close()

        # dn/dt combined
        plt.figure()
        if f"dn_dt_per_ms_mm2_{labelA}" in dn_comb:
            plt.plot(dn_comb["t_ms"], dn_comb[f"dn_dt_per_ms_mm2_{labelA}"], label=labelA)
        if f"dn_dt_per_ms_mm2_{labelB}" in dn_comb:
            plt.plot(dn_comb["t_ms"], dn_comb[f"dn_dt_per_ms_mm2_{labelB}"], label=labelB)
        plt.xlabel("t (ms)")
        plt.ylabel("dn/dt  [events / (ms·mm²)]")
        plt.title("Nucleation density rate dn/dt — both datasets")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "dn_dt_both_ms.png", dpi=200)
        plt.close()

        # Growth-rate histogram
        if vA.size > 0 or vB.size > 0:
            if vA.size > 0 and vB.size > 0:
                v_all = np.concatenate([vA, vB])
            else:
                v_all = vA if vA.size > 0 else vB
            v_min, v_max = float(np.nanmin(v_all)), float(np.nanmax(v_all))
            if v_min == v_max:
                v_min, v_max = 0.0, (v_max * 1.1 if v_max > 0 else 1.0)
            bins = np.linspace(v_min, v_max, 61)

            plt.figure()
            if vA.size > 0:
                plt.hist(vA, bins=bins, histtype="step", label=labelA)
            if vB.size > 0:
                plt.hist(vB, bins=bins, histtype="step", label=labelB)
            plt.xlabel("growth rate (µm/s) [effective]")
            plt.ylabel("count")
            plt.title("Growth-rate distribution — both datasets")
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir / "growth_hist_both.png", dpi=200)
            plt.close()

    except Exception as e:
        print("[WARN] Plotting skipped:", e)

    print(f"[OK] Wrote combined CSVs and plots to: {outdir}")


if __name__ == "__main__":
    main()
