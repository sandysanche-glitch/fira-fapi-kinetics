#!/usr/bin/env python
"""
compare_SI_from_json_penalties.py

Directly read FAPI / FAPI-TEMPO JSON detection files (Mask R-CNN-style),
compute SI metrics, and include REAL penalties for each grain based on:

  - circularity C = 4πA / P^2  (from grain mask)
  - defect fraction φ = A_defect_in_grain / A_grain  (from overlap with defect masks)

These penalties modify the effective growth rate:

    v_eff = v_unpen * exp( -alpha * (1 - C) ) * exp( -beta * φ )

Mean-field X_pred(t) (no spatial impingement):

  - rank-based nucleation times t0_i,
  - linear radial growth,
  - X_pred = sum(π r_i(t)^2) / A_eff.

We then fit:

  1) Ideal Avrami (fixed n):
        X_ideal(t) = X_inf * (1 - exp(-K * t^n))

  2) Shifted Avrami (fixed n):
        X_shift(t) = X_inf_s * (1 - exp(-K_s * (t - t_shift)^n))  for t > t_shift
                     0                                           for t <= t_shift

Defaults:
    FAPI JSONs:
        D:\\SWITCHdrive\\Institution\\Sts_grain morphology_ML\\comparative datasets\\FAPI
    FAPI-TEMPO JSONs:
        D:\\SWITCHdrive\\Institution\\Sts_grain morphology_ML\\comparative datasets\\FAPI-TEMPO

Grains and defects are distinguished by category_id:
    grain_cat_id  (default: 2)
    defect_cat_id (default: 3)

Outputs (in --out):
    combined_dn_dt.csv
    combined_X_pred_models.csv
    combined_growth_hist.csv
    combined_kinetic_fits.csv
    per_grain_metrics_{label}.csv  (grain-level C, φ, v, etc.)

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
    from scipy.ndimage import binary_erosion
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# ---------- helpers (rank, growth, area conversions) ----------

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
    Effective growth rate with penalties (circularity, defect).

    v_unpen = R_um / dt_ms
    v_eff   = v_unpen * exp(-alpha * (1 - C)) * exp(-beta * φ)
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


# ---------- dn/dt model ----------

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


# ---------- Avrami fits (ideal + shifted, fixed n) ----------

def fit_K_Xinf_from_X_avrami_coords(t_ms: np.ndarray, X_pred: np.ndarray, n: float,
                                    x_min: float = 0.05, x_max: float = 0.95):
    """
    Fit K and X_inf in ideal Avrami:

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


# ---------- JSON → per-grain metrics (area, C, φ) ----------

def compute_perimeter_from_mask(mask: np.ndarray) -> float:
    """
    Approximate perimeter from binary mask using morphological erosion.

    perimeter ≈ number of boundary pixels (mask & ~eroded(mask))
    """
    if not HAVE_SCIPY:
        # crude fallback: count pixels that have a 0 neighbor in 4-connectivity
        h, w = mask.shape
        inner = np.ones_like(mask, dtype=bool)
        inner[1:, :] &= mask[1:, :] & mask[:-1, :]
        inner[:-1, :] &= mask[:-1, :] & mask[1:, :]
        inner[:, 1:] &= mask[:, 1:] & mask[:, :-1]
        inner[:, :-1] &= mask[:, :-1] & mask[:, 1:]
        boundary = mask & ~inner
    else:
        inner = binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
        boundary = mask & ~inner
    return float(np.count_nonzero(boundary))


def collect_grain_metrics_from_json_folder(
    folder: Path,
    grain_cat_id: int = 2,
    defect_cat_id: int = 3,
    min_score: float = 0.0,
):
    """
    Parse all JSON files in `folder` and compute per-grain metrics:

        - area_px
        - centroid (cx_px, cy_px)
        - perimeter_px
        - circularity C = 4πA / P^2
        - defect fraction φ = A_defect_in_grain / A_grain

    Grains are objects with category_id == grain_cat_id.
    Defects are objects with category_id == defect_cat_id.

    Returns:
        area_px, C, phi, H, W
    """
    all_area = []
    all_C = []
    all_phi = []
    all_cx = []
    all_cy = []

    H_global, W_global = None, None

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

        grains_local = []
        defects_masks = []

        H_local, W_local = None, None

        # First pass: gather masks and basic info
        for obj in data:
            score = obj.get("score", 1.0)
            if score < min_score:
                continue
            seg = obj.get("segmentation", None)
            if seg is None:
                continue

            rle = {
                "size": seg["size"],
                "counts": seg["counts"],
            }
            mask = maskUtils.decode(rle).astype(bool)
            H, W = seg["size"]

            if H_local is None:
                H_local, W_local = H, W
            else:
                if H != H_local or W != W_local:
                    print(f"[WARN] Inconsistent frame size in {jf}: {H}x{W} vs {H_local}x{W_local}")

            cat_id = obj.get("category_id", None)

            if cat_id == grain_cat_id:
                area = float(np.count_nonzero(mask))
                if area <= 0:
                    continue
                # centroid
                ys, xs = np.nonzero(mask)
                cx = float(xs.mean())
                cy = float(ys.mean())
                # perimeter
                P = compute_perimeter_from_mask(mask)
                if P <= 0:
                    C = 1.0
                else:
                    C = float(4.0 * math.pi * area / (P ** 2))
                C = max(0.0, min(1.0, C))
                grains_local.append({
                    "mask": mask,
                    "area": area,
                    "cx": cx,
                    "cy": cy,
                    "C": C,
                })
            elif cat_id == defect_cat_id:
                defects_masks.append(mask)

        if not grains_local:
            continue

        # Union of all defect masks in this image
        if defects_masks:
            union_defects = np.zeros_like(grains_local[0]["mask"], dtype=bool)
            for dm in defects_masks:
                union_defects |= dm
        else:
            union_defects = None

        # For each grain: compute defect fraction φ from overlap
        for g in grains_local:
            mask_g = g["mask"]
            area_g = g["area"]
            if union_defects is not None:
                inter = np.logical_and(mask_g, union_defects)
                defect_area = float(np.count_nonzero(inter))
                phi = defect_area / max(area_g, 1e-9)
            else:
                phi = 0.0

            all_area.append(area_g)
            all_C.append(g["C"])
            all_phi.append(phi)
            all_cx.append(g["cx"])
            all_cy.append(g["cy"])

        # global frame size
        if H_global is None:
            H_global, W_global = H_local, W_local
        else:
            if H_local != H_global or W_local != W_global:
                print(f"[WARN] Inconsistent global frame sizes in folder {folder}")

    if not all_area:
        raise ValueError(f"No valid grain objects found in {folder} (after score and category filters).")

    area_arr = np.asarray(all_area, dtype=float)
    C_arr = np.asarray(all_C, dtype=float)
    phi_arr = np.asarray(all_phi, dtype=float)
    cx_arr = np.asarray(all_cx, dtype=float)
    cy_arr = np.asarray(all_cy, dtype=float)

    print(f"[INFO] Collected {len(area_arr)} grains from {folder}")
    print(f"[INFO] Global frame size: {H_global} x {W_global} px")

    return area_arr, C_arr, phi_arr, cx_arr, cy_arr, H_global, W_global


# ---------- per-dataset SI + kinetics with penalties ----------

def compute_si_with_penalties(
    area_px: np.ndarray,
    C: np.ndarray,
    phi: np.ndarray,
    label: str,
    px_per_um: float,
    area_eff_px: float,
    n_avrami: float,
    x_min_fit: float = 0.05,
    x_max_fit: float = 0.95,
    t_max_ms: float = 600.0,
    use_penalties: bool = True,
    alpha: float = 1.0,
    beta: float = 1.0,
):
    """
    Compute SI metrics with real penalties (C, φ) and Avrami fits.

    Returns:
        out_obj_df: per-grain metrics
        dn_dt_df:   dn/dt vs t
        X_df:       t, X_pred, X_ideal, X_shifted
        params:     dict of kinetic fit parameters
    """

    # nucleation times (for dn/dt and growth)
    t0_ms = rank_to_t0_ms(area_px, t_win_ms=60.0)
    dt_ms_v = np.clip(t_max_ms - t0_ms, 1e-9, None)
    R_um = area_px_to_radius_um(area_px, px_per_um=px_per_um)

    alpha_eff = alpha if use_penalties else 0.0
    beta_eff = beta if use_penalties else 0.0
    v_eff_um_ms, v_unp_um_ms, pf = eff_growth_rate_um_per_ms(
        R_um, dt_ms_v, circularity=C, defect_frac=phi, alpha=alpha_eff, beta=beta_eff
    )

    area_um2 = effective_area_um2(area_eff_px, px_per_um=px_per_um)

    # dn/dt vs time
    dn_grid_ms = np.arange(0.0, 60.0 + 1.0, 1.0)
    dn_dt_df, dn_meta = fit_dn_dt(t0_ms, area_um2, dn_grid_ms)

    # Reconstruct X_pred(t) (mean-field, no spatial impingement)
    t_ms = np.arange(0.0, t_max_ms + 1.0, 1.0)
    t_s = t_ms / 1000.0
    X_pred = np.zeros_like(t_ms, dtype=float)
    for i, t in enumerate(t_ms):
        tau = np.clip(t - t0_ms, 0.0, None)
        r_um_t = np.maximum(0.0, v_eff_um_ms * tau)
        A_um2_t = math.pi * (r_um_t ** 2)
        X_pred[i] = np.nansum(A_um2_t) / max(area_um2, 1e-12)

    # Fit ideal Avrami
    n = float(n_avrami)
    Xinf_ideal, K_ideal = fit_K_Xinf_from_X_avrami_coords(
        t_ms, X_pred, n=n, x_min=x_min_fit, x_max=x_max_fit
    )
    X_ideal = Xinf_ideal * (1.0 - np.exp(-K_ideal * (t_ms ** n)))

    # Fit shifted Avrami
    Xinf_shift, K_shift, t_shift = fit_shifted_avrami(
        t_ms, X_pred, n=n, x_min=x_min_fit, x_max=x_max_fit
    )
    tau_shift = np.clip(t_ms - t_shift, 0.0, None)
    X_shift = Xinf_shift * (1.0 - np.exp(-K_shift * (tau_shift ** n)))

    # Per-grain metrics
    out_obj_df = pd.DataFrame({
        "label": label,
        "area_px": area_px,
        "circularity_C": C,
        "defect_frac_phi": phi,
        "t0_ms": t0_ms,
        "dt_ms_for_v": dt_ms_v,
        "R_um_final": R_um,
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
        "use_penalties": bool(use_penalties),
        "alpha": alpha_eff,
        "beta": beta_eff,
    }

    return out_obj_df, dn_dt_df, X_df, params


# ---------- main script ----------

def main():
    ap = argparse.ArgumentParser(
        description="Compare FAPI vs FAPI-TEMPO from JSON, with real penalties from circularity and defect fraction."
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
        default="combined_out_penalties",
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
        "--grain_cat_id",
        type=int,
        default=2,
        help="category_id for grains (default: 2)",
    )
    ap.add_argument(
        "--defect_cat_id",
        type=int,
        default=3,
        help="category_id for defects (default: 3)",
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
        help="Effective analyzed area in pixels. If not set, use full frame area.",
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
        "--no_penalties",
        action="store_true",
        help="If set, ignore C and φ (v_eff = v_unpen).",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Weight for circularity penalty exp(-alpha * (1 - C)).",
    )
    ap.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Weight for defect penalty exp(-beta * φ).",
    )

    args = ap.parse_args()

    fapi_dir = Path(args.fapi_dir)
    fapitempo_dir = Path(args.fapitempo_dir)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    labelA = args.labelA
    labelB = args.labelB

    # Collect grain metrics (area, C, φ) from JSON for both datasets
    areaA, CA, phiA, cxA, cyA, HA, WA = collect_grain_metrics_from_json_folder(
        fapi_dir,
        grain_cat_id=args.grain_cat_id,
        defect_cat_id=args.defect_cat_id,
        min_score=args.min_score,
    )
    areaB, CB, phiB, cxB, cyB, HB, WB = collect_grain_metrics_from_json_folder(
        fapitempo_dir,
        grain_cat_id=args.grain_cat_id,
        defect_cat_id=args.defect_cat_id,
        min_score=args.min_score,
    )

    # Effective area: full frame or user-specified
    if args.area_eff_px is None:
        area_eff_px_A = float(HA * WA)
        area_eff_px_B = float(HB * WB)
        print(f"[INFO] Using full frame area as A_eff: "
              f"{area_eff_px_A:.1f} px^2 (A), {area_eff_px_B:.1f} px^2 (B)")
    else:
        area_eff_px_A = float(args.area_eff_px)
        area_eff_px_B = float(args.area_eff_px)
        print(f"[INFO] Using user-specified area_eff_px = {area_eff_px_A:.1f} for both datasets.")

    use_pen = not args.no_penalties

    # Compute SI + kinetics with penalties for each dataset
    outA, dnA, XA, parA = compute_si_with_penalties(
        areaA, CA, phiA, labelA,
        px_per_um=args.px_per_um,
        area_eff_px=area_eff_px_A,
        n_avrami=args.n_avrami,
        x_min_fit=args.x_min_fit,
        x_max_fit=args.x_max_fit,
        t_max_ms=600.0,
        use_penalties=use_pen,
        alpha=args.alpha,
        beta=args.beta,
    )
    outB, dnB, XB, parB = compute_si_with_penalties(
        areaB, CB, phiB, labelB,
        px_per_um=args.px_per_um,
        area_eff_px=area_eff_px_B,
        n_avrami=args.n_avrami,
        x_min_fit=args.x_min_fit,
        x_max_fit=args.x_max_fit,
        t_max_ms=600.0,
        use_penalties=use_pen,
        alpha=args.alpha,
        beta=args.beta,
    )

    # Save per-grain metrics for each dataset
    outA.to_csv(outdir / f"per_grain_metrics_{labelA}.csv", index=False)
    outB.to_csv(outdir / f"per_grain_metrics_{labelB}.csv", index=False)

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

    # 2) Combined X_pred + models
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
        plt.title("Bulk transformed fraction: X_pred vs ideal Avrami (with penalties)")
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
        plt.title("Bulk transformed fraction: X_pred vs shifted Avrami (with penalties)")
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
        plt.title("Bulk transformed fraction: X_pred vs ideal Avrami (seconds, with penalties)")
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
        plt.title("Bulk transformed fraction: X_pred vs shifted Avrami (seconds, with penalties)")
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
        plt.ylabel("dn/dt [events / (ms·mm²)]")
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
            plt.title("Growth-rate distribution — both datasets (with penalties)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir / "growth_hist_both.png", dpi=200)
            plt.close()

    except Exception as e:
        print("[WARN] Plotting skipped:", e)

    print(f"[OK] Wrote CSVs and plots with penalties to: {outdir}")


if __name__ == "__main__":
    main()
