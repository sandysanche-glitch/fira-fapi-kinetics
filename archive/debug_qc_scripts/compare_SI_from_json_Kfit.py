#!/usr/bin/env python
"""
compare_SI_from_json_Kfit.py

Directly read FAPI / FAPI-TEMPO JSON detection files (Mask R-CNN-style),
compute SI metrics, fit Avrami K for both (least squares fit to X_pred),
and export combined CSVs and plots.

Defaults:
    FAPI JSONs:
        D:\\SWITCHdrive\\Institution\\Sts_grain morphology_ML\\comparative datasets\\FAPI
    FAPI-TEMPO JSONs:
        D:\\SWITCHdrive\\Institution\\Sts_grain morphology_ML\\comparative datasets\\FAPI-TEMPO

Outputs (in --out):
    combined_dn_dt.csv
    combined_X_pred_Avrami.csv
    combined_growth_hist.csv
    combined_K_fits.csv

    dn_dt_both_ms.png
    X_overlay_both_ms.png
    X_overlay_both_s.png
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

# ---- SciPy (for dn/dt and K fitting) ----
try:
    from scipy.stats import lognorm, gamma
    from scipy.optimize import minimize_scalar
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# ---------- helpers (mostly adapted from export_SI_metrics_rank_v2.py) ----------

def safe_num(x, default=np.nan):
    return pd.to_numeric(x, errors="coerce").fillna(default).to_numpy(dtype=float)


def rank_to_t0_ms(area_px: np.ndarray, t_win_ms: float = 60.0) -> np.ndarray:
    """
    Map ranked areas onto [0, t_win_ms] to assign a nucleation time t0_ms.
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
    2.20014 px = 1 µm → px_per_um = 2.20014.
    """
    R_px = np.sqrt(np.maximum(area_px, 0.0) / math.pi)
    return R_px / px_per_um


def eff_growth_rate_um_per_ms(R_um: np.ndarray, dt_ms: np.ndarray,
                              circularity: np.ndarray = None,
                              defect_frac: np.ndarray = None,
                              alpha: float = 0.0, beta: float = 0.0):
    """
    Effective growth rate with optional penalties (circularity, defect).
    Here circularity and defect_frac are trivial (1, 0) unless you plug in real values.
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


# ---- dn/dt model (same as in export_SI_metrics_rank_v2.py) ----

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


# ---- improved K fitting: least-squares to X_pred ----

def fit_K_from_X(t_ms: np.ndarray, X_pred: np.ndarray, n: float) -> float:
    """
    Fit K in Avrami law: X_A(t) = 1 - exp(-K * t^n)
    by minimizing squared error between X_pred and the model.

    Uses SciPy's minimize_scalar if available, otherwise log-spaced grid search.
    Time is in ms → K has units of ms^-n.
    """
    t_ms = np.asarray(t_ms, dtype=float)
    X_pred = np.asarray(X_pred, dtype=float)

    y = np.clip(X_pred, 0.0, 0.999999)
    valid = (t_ms > 0) & (y > 0) & (y < 0.99)
    t = t_ms[valid]
    yv = y[valid]

    if t.size < 5:
        return 0.0

    # quick analytic estimate (original method) as initial guess
    t_pow = t ** n
    K_vals = -np.log(1.0 - yv) / t_pow
    K_vals = K_vals[np.isfinite(K_vals) & (K_vals > 0)]
    if K_vals.size == 0:
        return 0.0
    K0 = float(np.nanmedian(K_vals))

    def sse(K):
        if K <= 0:
            return 1e30
        X_model = 1.0 - np.exp(-K * (t ** n))
        return float(np.sum((X_model - yv) ** 2))

    # SciPy path
    if HAVE_SCIPY:
        lo = max(K0 / 100.0, 1e-12)
        hi = max(K0 * 100.0, lo * 10)

        res = minimize_scalar(sse, bounds=(lo, hi), method="bounded")
        if res.success and res.x > 0:
            return float(res.x)
        return K0

    # fallback: coarse log grid search
    K_min = max(K0 / 100.0, 1e-12)
    K_max = K0 * 100.0
    Ks = np.logspace(np.log10(K_min), np.log10(K_max), 200)
    errors = [sse(K) for K in Ks]
    best_idx = int(np.argmin(errors))
    return float(Ks[best_idx])


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
                          use_penalties: bool = False,
                          alpha: float = 0.0,
                          beta: float = 0.0):
    """
    Mirror the logic of export_SI_metrics_rank_v2.py but starting from area_px only.
    Returns:
        out_obj_df: per-object metrics
        dndt_df: dn/dt vs t
        X_df: X_pred and Avrami vs time
        K_fit: fitted Avrami K
    """

    # Optional penalties disabled by default; with only area we use C=1, phi=0
    C = np.ones_like(area_px, dtype=float)
    phi = np.zeros_like(area_px, dtype=float)

    t0_ms = rank_to_t0_ms(area_px, t_win_ms=60.0)
    dt_ms = np.clip(600.0 - t0_ms, 1e-9, None)
    R_um = area_px_to_radius_um(area_px, px_per_um=px_per_um)

    alpha_eff = alpha if use_penalties else 0.0
    beta_eff = beta if use_penalties else 0.0
    v_eff_um_ms, v_unp_um_ms, pf = eff_growth_rate_um_per_ms(
        R_um, dt_ms, C, phi, alpha=alpha_eff, beta=beta_eff
    )

    area_um2 = effective_area_um2(area_eff_px, px_per_um=px_per_um)

    # dn/dt
    grid_ms = np.arange(0.0, 60.0 + 1.0, 1.0)
    dndt_df, dn_meta = fit_dn_dt(t0_ms, area_um2, grid_ms)

    # X(t)
    t_ms = np.arange(0.0, 600.0 + 1.0, 1.0)
    t_s = t_ms / 1000.0
    X_pred = np.zeros_like(t_ms, dtype=float)
    for i, t in enumerate(t_ms):
        tau = np.clip(t - t0_ms, 0.0, None)
        r_um = np.maximum(0.0, v_eff_um_ms * tau)
        A_um2 = math.pi * (r_um ** 2)
        X_pred[i] = np.nansum(A_um2) / max(area_um2, 1e-12)

    n = float(n_avrami)
    K_fit = fit_K_from_X(t_ms, X_pred, n=n)
    X_A = 1.0 - np.exp(-K_fit * (t_ms ** n))

    # Per-object metrics
    out_obj_df = pd.DataFrame({
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
        "v_eff_um_per_s": v_eff_um_ms * 1000.0,
        "penalty_factor": pf,
    })

    X_df = pd.DataFrame({
        "t_ms": t_ms,
        "t_s": t_s,
        "X_pred": X_pred,
        "X_Avrami": X_A,
    })

    return out_obj_df, dndt_df, X_df, K_fit


# ---------- main script ----------

def main():
    ap = argparse.ArgumentParser(description="Compare FAPI vs FAPI-TEMPO directly from JSON RLE masks, Avrami K least-squares fit.")
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
        help="Fixed Avrami exponent n used for both datasets",
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
        help="Enable penalties based on circularity/defect_frac (here they are trivial 1 and 0).",
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

    # Compute SI metrics for each dataset
    outA, dnA, XA, K_A = compute_si_from_areas(
        areaA, labelA, px_per_um=args.px_per_um, area_eff_px=area_eff_px,
        n_avrami=args.n_avrami,
        use_penalties=args.use_penalties,
        alpha=args.alpha, beta=args.beta
    )
    outB, dnB, XB, K_B = compute_si_from_areas(
        areaB, labelB, px_per_um=args.px_per_um, area_eff_px=area_eff_px,
        n_avrami=args.n_avrami,
        use_penalties=args.use_penalties,
        alpha=args.alpha, beta=args.beta
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

    # 2) Combined X_pred + Avrami
    X_comb = pd.DataFrame({
        "t_ms": XA["t_ms"],
        "t_s": XA["t_s"],
        f"X_pred_{labelA}": XA["X_pred"],
        f"X_Avrami_{labelA}": XA["X_Avrami"],
    })
    X_comb = pd.merge(
        X_comb,
        XB[["t_ms", "X_pred", "X_Avrami"]].rename(
            columns={
                "X_pred": f"X_pred_{labelB}",
                "X_Avrami": f"X_Avrami_{labelB}",
            }
        ),
        on="t_ms",
        how="outer",
        sort=True,
    )
    X_comb["t_s"] = X_comb["t_ms"].to_numpy(dtype=float) / 1000.0
    X_comb.to_csv(outdir / "combined_X_pred_Avrami.csv", index=False)

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

    # 4) Combined K fits
    pd.DataFrame([
        {"label": labelA, "n_avrami": args.n_avrami, "K_fit_ms_units": K_A},
        {"label": labelB, "n_avrami": args.n_avrami, "K_fit_ms_units": K_B},
    ]).to_csv(outdir / "combined_K_fits.csv", index=False)

    # ---------- Plots ----------

    try:
        import matplotlib.pyplot as plt

        n = args.n_avrami

        # X overlay (ms)
        plt.figure()
        plt.plot(X_comb["t_ms"], X_comb[f"X_pred_{labelA}"], label=f"{labelA} X_pred")
        plt.plot(X_comb["t_ms"], X_comb[f"X_Avrami_{labelA}"],
                 linestyle="--", label=f"{labelA} Avrami (n={n:.2f}, K={K_A:.3g})")
        plt.plot(X_comb["t_ms"], X_comb[f"X_pred_{labelB}"], label=f"{labelB} X_pred")
        plt.plot(X_comb["t_ms"], X_comb[f"X_Avrami_{labelB}"],
                 linestyle="--", label=f"{labelB} Avrami (n={n:.2f}, K={K_B:.3g})")
        plt.xlabel("t (ms)")
        plt.ylabel("X(t) (fraction)")
        plt.title("X(t) vs Avrami — both datasets (ms)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "X_overlay_both_ms.png", dpi=200)
        plt.close()

        # X overlay (s)
        plt.figure()
        plt.plot(X_comb["t_s"], X_comb[f"X_pred_{labelA}"], label=f"{labelA} X_pred")
        plt.plot(X_comb["t_s"], X_comb[f"X_Avrami_{labelA}"],
                 linestyle="--", label=f"{labelA} Avrami (n={n:.2f}, K={K_A:.3g})")
        plt.plot(X_comb["t_s"], X_comb[f"X_pred_{labelB}"], label=f"{labelB} X_pred")
        plt.plot(X_comb["t_s"], X_comb[f"X_Avrami_{labelB}"],
                 linestyle="--", label=f"{labelB} Avrami (n={n:.2f}, K={K_B:.3g})")
        plt.xlabel("t (s)")
        plt.ylabel("X(t) (fraction)")
        plt.title("X(t) vs Avrami — both datasets (seconds)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "X_overlay_both_s.png", dpi=200)
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
