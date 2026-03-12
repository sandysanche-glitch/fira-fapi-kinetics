#!/usr/bin/env python
"""
compare_SI_from_json_spatialImpingement.py

Directly read FAPI / FAPI-TEMPO JSON detection files (Mask R-CNN-style),
compute SI metrics, and compare:

  1) Mean-field bulk transformed fraction X_pred_mf(t)
     - disks grow independently, X = sum(pi r_i^2) / A_eff

  2) Spatial bulk transformed fraction X_pred_spatial(t)
     - disks grow from their actual centers on a pixel grid
     - X = union area of all disks / frame area
     - explicitly includes spatial impingement / overlap

The nucleation-time distribution dn/dt is unchanged by spatial modeling,
since it depends only on t0_i, not on positions.

Defaults:
    FAPI JSONs:
        D:\\SWITCHdrive\\Institution\\Sts_grain morphology_ML\\comparative datasets\\FAPI
    FAPI-TEMPO JSONs:
        D:\\SWITCHdrive\\Institution\\Sts_grain morphology_ML\\comparative datasets\\FAPI-TEMPO

Outputs (in --out):
    combined_dn_dt.csv
    combined_X_pred_spatial.csv
    combined_growth_hist.csv

    dn_dt_both_ms.png
    X_overlay_both_ms_spatial_vs_mf.png
    X_overlay_both_s_spatial_vs_mf.png
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

# ---- SciPy (for dn/dt only) ----
try:
    from scipy.stats import lognorm, gamma
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# ---------- helpers (adapted from previous SI scripts) ----------

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


# ---------- JSON -> area + centers ----------

def collect_areas_and_centers_from_json_folder(folder: Path, min_score: float = 0.0):
    """
    Parse all *.json files in `folder` and return:

        area_px: [N] areas in pixels (via RLE)
        cx_px:   [N] x-center in pixels (bbox center)
        cy_px:   [N] y-center in pixels (bbox center)
        H, W:    frame size in pixels (from segmentation size)

    Assumes:
        - top-level JSON is a list of objects for each file
        - each object has:
              'segmentation': {'size': [H, W], 'counts': ...}
              'bbox': [x_min, y_min, width, height]
              'score' (optional)
    """
    all_areas = []
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

        for obj in data:
            score = obj.get("score", 1.0)
            if score < min_score:
                continue
            seg = obj.get("segmentation", None)
            bbox = obj.get("bbox", None)
            if seg is None or bbox is None:
                continue

            try:
                rle = {
                    "size": seg["size"],
                    "counts": seg["counts"]
                }
                area = float(maskUtils.area(rle))
                if area <= 0:
                    continue

                H, W = seg["size"]
                if H_global is None:
                    H_global, W_global = H, W
                else:
                    if H != H_global or W != W_global:
                        print(f"[WARN] Inconsistent frame size in {jf}: "
                              f"{H}x{W} vs {H_global}x{W_global}")

                x_min, y_min, width, height = bbox
                cx = x_min + 0.5 * width
                cy = y_min + 0.5 * height

                all_areas.append(area)
                all_cx.append(cx)
                all_cy.append(cy)
            except Exception as e:
                print(f"[WARN] Failed to process object in {jf}: {e}")

    if not all_areas:
        raise ValueError(f"No valid objects found in {folder} (after score filter).")

    areas = np.asarray(all_areas, dtype=float)
    cx_arr = np.asarray(all_cx, dtype=float)
    cy_arr = np.asarray(all_cy, dtype=float)

    print(f"[INFO] Collected {len(areas)} objects from {folder}")
    print(f"[INFO] Frame size used for spatial simulation: {H_global} x {W_global} px")

    return areas, cx_arr, cy_arr, H_global, W_global


# ---------- spatial union helper ----------

def compute_spatial_union_fraction(
    t_ms_grid: np.ndarray,
    t0_ms: np.ndarray,
    v_eff_um_ms: np.ndarray,
    cx_px: np.ndarray,
    cy_px: np.ndarray,
    H: int,
    W: int,
    px_per_um: float,
):
    """
    Compute X_pred_spatial(t): union area fraction of disks on an HxW grid.

    For each time t in t_ms_grid:
        - radius of grain i in µm: r_um_i(t) = v_eff_um_ms[i] * max(t - t0[i], 0)
        - in pixels: R_px_i(t) = r_um_i(t) * px_per_um
        - draw disk at center (cx_i, cy_i) with radius R_px_i onto a boolean grid
        - X_spatial(t) = (# True pixels) / (H * W)

    NOTE: This can be computationally heavy for many grains and fine time steps.
          Time grid should not be too dense (e.g. dt_ms = 5 or 10).
    """
    t_ms_grid = np.asarray(t_ms_grid, dtype=float)
    t0_ms = np.asarray(t0_ms, dtype=float)
    v_eff_um_ms = np.asarray(v_eff_um_ms, dtype=float)
    cx_px = np.asarray(cx_px, dtype=float)
    cy_px = np.asarray(cy_px, dtype=float)

    n_t = len(t_ms_grid)
    N = len(t0_ms)

    X_spatial = np.zeros(n_t, dtype=float)

    um_per_px = 1.0 / px_per_um

    for it, t in enumerate(t_ms_grid):
        # grow radii
        tau = np.clip(t - t0_ms, 0.0, None)
        r_um = np.maximum(0.0, v_eff_um_ms * tau)
        R_px = r_um * px_per_um

        # only grains with radius > 0
        active = R_px > 0.0
        if not np.any(active):
            X_spatial[it] = 0.0
            continue

        R_px_active = R_px[active]
        cx_active = cx_px[active]
        cy_active = cy_px[active]

        mask = np.zeros((H, W), dtype=bool)

        for R, cx, cy in zip(R_px_active, cx_active, cy_active):
            if R <= 0:
                continue
            r_int = int(math.ceil(R))
            if r_int <= 0:
                continue

            x0 = int(max(0, math.floor(cx - r_int)))
            x1 = int(min(W, math.ceil(cx + r_int + 1)))
            y0 = int(max(0, math.floor(cy - r_int)))
            y1 = int(min(H, math.ceil(cy + r_int + 1)))

            if x0 >= x1 or y0 >= y1:
                continue

            yy, xx = np.ogrid[y0:y1, x0:x1]
            dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
            disk = dist2 <= (R ** 2)

            mask[y0:y1, x0:x1] |= disk

        union_px = np.count_nonzero(mask)
        X_spatial[it] = union_px / float(H * W)

    return X_spatial


# ---------- main per-dataset SI computation ----------

def compute_si_with_spatial(
    area_px: np.ndarray,
    cx_px: np.ndarray,
    cy_px: np.ndarray,
    H: int,
    W: int,
    label: str,
    px_per_um: float,
    area_eff_px: float,
    dt_ms_full: float = 1.0,
    dt_ms_spatial: float = 5.0,
    t_max_ms: float = 600.0,
    use_penalties: bool = False,
    alpha: float = 0.0,
    beta: float = 0.0,
):
    """
    Compute dn/dt, mean-field X_pred_mf(t), and spatial X_pred_spatial(t).

    Returns:
        out_obj_df: per-object metrics
        dn_dt_df:   dn/dt vs time
        X_df:       t, X_pred_mf, X_pred_spatial
    """

    # Penalties: placeholders
    C = np.ones_like(area_px, dtype=float)
    phi = np.zeros_like(area_px, dtype=float)

    # nucleation times by rank (for dn/dt and growth)
    t0_ms = rank_to_t0_ms(area_px, t_win_ms=60.0)
    dt_ms_for_v = np.clip(t_max_ms - t0_ms, 1e-9, None)
    R_um_final = area_px_to_radius_um(area_px, px_per_um=px_per_um)

    alpha_eff = alpha if use_penalties else 0.0
    beta_eff = beta if use_penalties else 0.0
    v_eff_um_ms, v_unp_um_ms, pf = eff_growth_rate_um_per_ms(
        R_um_final, dt_ms_for_v, C, phi, alpha=alpha_eff, beta=beta_eff
    )

    # dn/dt (nucleation kinetics) from t0_ms, unchanged by spatial modeling
    area_um2 = effective_area_um2(area_eff_px, px_per_um=px_per_um)
    dn_grid_ms = np.arange(0.0, 60.0 + 1.0, 1.0)
    dn_dt_df, dn_meta = fit_dn_dt(t0_ms, area_um2, dn_grid_ms)

    # Mean-field X_pred_mf(t) at fine resolution dt_ms_full (e.g. 1 ms)
    t_full_ms = np.arange(0.0, t_max_ms + dt_ms_full, dt_ms_full)
    t_s_full = t_full_ms / 1000.0
    X_mf = np.zeros_like(t_full_ms, dtype=float)
    for i, t in enumerate(t_full_ms):
        tau = np.clip(t - t0_ms, 0.0, None)
        r_um = np.maximum(0.0, v_eff_um_ms * tau)
        A_um2 = math.pi * (r_um ** 2)
        X_mf[i] = np.nansum(A_um2) / max(area_um2, 1e-12)

    # Spatial X_pred_spatial(t) at coarser resolution dt_ms_spatial
    t_spatial_ms = np.arange(0.0, t_max_ms + dt_ms_spatial, dt_ms_spatial)
    X_spatial = compute_spatial_union_fraction(
        t_spatial_ms, t0_ms, v_eff_um_ms, cx_px, cy_px, H, W, px_per_um
    )

    # Interpolate X_spatial onto the same fine grid, for easier comparison & plotting
    X_spatial_interp = np.interp(t_full_ms, t_spatial_ms, X_spatial)

    out_obj_df = pd.DataFrame({
        "label": label,
        "area_px": area_px,
        "cx_px": cx_px,
        "cy_px": cy_px,
        "t0_ms": t0_ms,
        "dt_ms_for_v": dt_ms_for_v,
        "R_um_final": R_um_final,
        "v_unpen_um_per_ms": v_unp_um_ms,
        "v_eff_um_per_ms": v_eff_um_ms,
        "v_unpen_um_per_s": v_unp_um_ms * 1000.0,
        "v_eff_um_per_s": v_eff_um_ms * 1000.0,
        "penalty_factor": pf,
    })

    X_df = pd.DataFrame({
        "t_ms": t_full_ms,
        "t_s": t_s_full,
        "X_pred_mf": X_mf,
        "X_pred_spatial": X_spatial_interp,
    })

    return out_obj_df, dn_dt_df, X_df


# ---------- main script ----------

def main():
    ap = argparse.ArgumentParser(
        description="Compare FAPI vs FAPI-TEMPO from JSON masks with spatial impingement vs mean-field X(t)."
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
        default="combined_out_spatial",
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
        help="Effective analyzed area in pixels for mean-field X(t). "
             "If not set, estimated as full image area from the frame size.",
    )
    ap.add_argument(
        "--min_score",
        type=float,
        default=0.0,
        help="Minimum detection score to include an object (default: 0.0)",
    )
    ap.add_argument(
        "--dt_ms_full",
        type=float,
        default=1.0,
        help="Time step (ms) for mean-field X_pred_mf(t) (default: 1 ms).",
    )
    ap.add_argument(
        "--dt_ms_spatial",
        type=float,
        default=5.0,
        help="Time step (ms) for spatial X_pred_spatial(t) simulation (default: 5 ms).",
    )
    ap.add_argument(
        "--t_max_ms",
        type=float,
        default=600.0,
        help="Maximum time (ms) for X(t) simulation (default: 600 ms).",
    )
    ap.add_argument(
        "--use_penalties",
        action="store_true",
        help="Enable penalties based on circularity/defect_frac (currently placeholders: C=1, phi=0).",
    )
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--beta", type=float, default=0.0)

    args = ap.parse_args()

    fapi_dir = Path(args.fapi_dir)
    fapitempo_dir = Path(args.fapitempo_dir)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    labelA = args.labelA
    labelB = args.labelB

    # Collect areas and centers from JSON for both datasets
    areaA, cxA, cyA, HA, WA = collect_areas_and_centers_from_json_folder(
        fapi_dir, min_score=args.min_score
    )
    areaB, cxB, cyB, HB, WB = collect_areas_and_centers_from_json_folder(
        fapitempo_dir, min_score=args.min_score
    )

    # For simplicity, use full frame area for both, unless user overrides area_eff_px
    if args.area_eff_px is None:
        area_eff_px_A = float(HA * WA)
        area_eff_px_B = float(HB * WB)
        print(f"[INFO] Using full frame area for mean-field X(t): "
              f"{area_eff_px_A:.1f} px^2 (A), {area_eff_px_B:.1f} px^2 (B)")
    else:
        area_eff_px_A = float(args.area_eff_px)
        area_eff_px_B = float(args.area_eff_px)
        print(f"[INFO] Using user-specified area_eff_px = {area_eff_px_A:.1f} for both datasets.")

    # Compute SI metrics + mean-field & spatial X(t) for each dataset
    outA, dnA, XA = compute_si_with_spatial(
        areaA, cxA, cyA, HA, WA, labelA,
        px_per_um=args.px_per_um,
        area_eff_px=area_eff_px_A,
        dt_ms_full=args.dt_ms_full,
        dt_ms_spatial=args.dt_ms_spatial,
        t_max_ms=args.t_max_ms,
        use_penalties=args.use_penalties,
        alpha=args.alpha,
        beta=args.beta,
    )

    outB, dnB, XB = compute_si_with_spatial(
        areaB, cxB, cyB, HB, WB, labelB,
        px_per_um=args.px_per_um,
        area_eff_px=area_eff_px_B,
        dt_ms_full=args.dt_ms_full,
        dt_ms_spatial=args.dt_ms_spatial,
        t_max_ms=args.t_max_ms,
        use_penalties=args.use_penalties,
        alpha=args.alpha,
        beta=args.beta,
    )

    # ---------- Combined CSVs ----------

    # 1) Combined dn/dt (unchanged by spatial modeling)
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

    # 2) Combined X_pred (mean-field vs spatial) for both datasets
    # Assume same time grid for A and B (dt_ms_full, t_max_ms)
    X_comb = pd.DataFrame({
        "t_ms": XA["t_ms"],
        "t_s": XA["t_s"],
        f"X_pred_mf_{labelA}": XA["X_pred_mf"],
        f"X_pred_spatial_{labelA}": XA["X_pred_spatial"],
    })
    X_comb = pd.merge(
        X_comb,
        XB[["t_ms", "X_pred_mf", "X_pred_spatial"]].rename(
            columns={
                "X_pred_mf": f"X_pred_mf_{labelB}",
                "X_pred_spatial": f"X_pred_spatial_{labelB}",
            }
        ),
        on="t_ms",
        how="outer",
        sort=True,
    )
    X_comb["t_s"] = X_comb["t_ms"].to_numpy(dtype=float) / 1000.0
    X_comb.to_csv(outdir / "combined_X_pred_spatial.csv", index=False)

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

    # ---------- Plots ----------

    try:
        import matplotlib.pyplot as plt

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

        # X overlay (ms): mean-field vs spatial for both datasets
        plt.figure()
        plt.plot(X_comb["t_ms"], X_comb[f"X_pred_mf_{labelA}"], label=f"{labelA} mean-field")
        plt.plot(X_comb["t_ms"], X_comb[f"X_pred_spatial_{labelA}"],
                 linestyle="--", label=f"{labelA} spatial union")
        plt.plot(X_comb["t_ms"], X_comb[f"X_pred_mf_{labelB}"], label=f"{labelB} mean-field")
        plt.plot(X_comb["t_ms"], X_comb[f"X_pred_spatial_{labelB}"],
                 linestyle="--", label=f"{labelB} spatial union")
        plt.xlabel("t (ms)")
        plt.ylabel("X(t) (fraction of frame)")
        plt.title("Bulk transformed fraction: mean-field vs spatial impingement")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "X_overlay_both_ms_spatial_vs_mf.png", dpi=200)
        plt.close()

        # X overlay (s): same in seconds
        plt.figure()
        plt.plot(X_comb["t_s"], X_comb[f"X_pred_mf_{labelA}"], label=f"{labelA} mean-field")
        plt.plot(X_comb["t_s"], X_comb[f"X_pred_spatial_{labelA}"],
                 linestyle="--", label=f"{labelA} spatial union")
        plt.plot(X_comb["t_s"], X_comb[f"X_pred_mf_{labelB}"], label=f"{labelB} mean-field")
        plt.plot(X_comb["t_s"], X_comb[f"X_pred_spatial_{labelB}"],
                 linestyle="--", label=f"{labelB} spatial union")
        plt.xlabel("t (s)")
        plt.ylabel("X(t) (fraction of frame)")
        plt.title("Bulk transformed fraction: mean-field vs spatial impingement (seconds)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "X_overlay_both_s_spatial_vs_mf.png", dpi=200)
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
