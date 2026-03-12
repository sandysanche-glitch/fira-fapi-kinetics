#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
COCO nucleation & growth analysis with anisotropy.
Writes all outputs to <folder>/output_anisotropy

- category_id: 1=crystal (mask), 2=nucleus (point), 3=defect (mask)
- Nucleation pseudo-times t0_i mapped to [0, 60] ms via rank-by-size (optionally Voronoi crowding)
- Growth (0..600 ms) with anisotropy:
  v(θ)=v0 * exp[-α(1-C)] * exp[-β φ] * [1 + ε2 cos(2(θ-θ0))]_+
- I(t): fit lognormal vs gamma (MLE), choose by BIC (AIC tiebreak)
- Avrami overlay: X(t) ~ 1 - exp[-K t^n] (fix n, fit K by LS vs X_pred)

Usage:
  python coco_nucleation_growth_anisotropy.py ^
    --folder "D:\\SWITCHdrive\\Institution\\Sts_grain morphology_ML\\good_coco_rle" ^
    --dataset_filter FAPI_TEMPO ^
    --n_avrami 2.5 ^
    --theta_step_deg 2 ^
    --use_voronoi 1
"""

import os, json, math, argparse, sys, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import optimize, stats
from scipy.spatial import Voronoi, ConvexHull

from skimage import measure, draw

# Optional RLE decode via pycocotools
try:
    from pycocotools import mask as maskUtils
    HAS_COCO = True
except Exception:
    HAS_COCO = False
    warnings.warn("pycocotools not available: RLE segmentations will be skipped unless converted to polygons.")

# -----------------------------
# Helpers
# -----------------------------

def polygon_to_mask_xyxy(polygon, height, width):
    xs = np.asarray(polygon[0::2], dtype=np.float32)
    ys = np.asarray(polygon[1::2], dtype=np.float32)
    rr, cc = draw.polygon(ys, xs, shape=(height, width))
    m = np.zeros((height, width), dtype=np.uint8)
    m[rr, cc] = 1
    return m

def annotation_to_mask(ann, img_h, img_w):
    seg = ann.get("segmentation", None)
    if seg is None:
        return None
    if isinstance(seg, list):
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        for poly in seg:
            m = polygon_to_mask_xyxy(poly, img_h, img_w)
            mask |= m
        return mask
    elif isinstance(seg, dict):
        if HAS_COCO:
            rle = seg
            if isinstance(rle.get("counts", None), list):
                rle = maskUtils.frPyObjects(rle, img_h, img_w)
            m = maskUtils.decode(rle)
            if m.ndim == 3:
                m = m[..., 0]
            return (m > 0).astype(np.uint8)
        else:
            warnings.warn("RLE encountered but pycocotools not installed. Skipping this ann.")
            return None
    else:
        return None

def mask_area_perimeter(mask):
    area = float(mask.sum())
    contours = measure.find_contours(mask.astype(float), level=0.5)
    peri = 0.0
    for c in contours:
        d = np.sqrt(np.sum(np.diff(c, axis=0)**2, axis=1)).sum()
        peri += d
    return area, peri

def mask_boundary_points(mask, n_pts=360):
    contours = measure.find_contours(mask.astype(float), level=0.5)
    if len(contours) == 0:
        return np.zeros((0,2), dtype=np.float32)
    c = max(contours, key=lambda a: a.shape[0])
    if c.shape[0] <= n_pts:
        return c.astype(np.float32)
    idx = np.linspace(0, c.shape[0]-1, n_pts).astype(int)
    return c[idx].astype(np.float32)

def polar_profile_from_center(boundary_yx, center_xy, theta_step_deg=2):
    if boundary_yx.size == 0:
        return np.array([]), np.array([])
    x_c, y_c = center_xy
    by = boundary_yx[:,0] - y_c
    bx = boundary_yx[:,1] - x_c
    angles = np.arctan2(by, bx)             # [-pi, pi)
    radii  = np.sqrt(bx**2 + by**2)

    step = np.deg2rad(theta_step_deg)
    ang = (angles + 2*np.pi) % (2*np.pi)    # [0, 2π)
    bins = np.floor(ang / step).astype(int)
    n_bins = int(np.ceil(2*np.pi / step))
    r_prof = np.zeros(n_bins, dtype=np.float32)
    for a, r, b in zip(ang, radii, bins):
        r_prof[b] = max(r_prof[b], r)       # angular envelope
    theta = (np.arange(n_bins)+0.5)*step
    return theta, r_prof

def fit_cos2(theta, r):
    theta = np.asarray(theta); r = np.asarray(r)
    m = np.isfinite(r)
    theta = theta[m]; r = r[m]
    if len(r) < 16 or (r<=0).all():
        return (np.nan, 0.0, 0.0), np.inf
    r_mean = np.mean(r)
    theta0_init = 0.5 * theta[np.argmax(r)]
    def model(params):
        r0, eps2, t0 = params
        return r0*(1.0 + eps2*np.cos(2*(theta - t0)))
    def loss(params):
        pred = model(params)
        return ((r - pred)**2).sum()
    x0 = [r_mean, 0.05, theta0_init]
    bounds = [(r_mean*0.2, r_mean*5.0), (-0.95, 0.95), (0, 2*np.pi)]
    res = optimize.minimize(loss, x0=x0, bounds=bounds, method="L-BFGS-B")
    (r0, eps2, t0) = res.x
    return (r0, float(eps2), float(t0)), float(res.fun)

def compute_convex_hull_area(points_xy):
    if len(points_xy) < 3:
        return 0.0
    hull = ConvexHull(points_xy)
    return float(hull.area) if hasattr(hull, 'area') else float(hull.volume)

# -----------------------------
# COCO loading
# -----------------------------

def load_coco_folder(folder, dataset_filter=None):
    folder = Path(folder)
    jsons = sorted([p for p in folder.glob("*.json")])
    images = dict()
    anns_all = []
    for j in jsons:
        with open(j, "r", encoding="utf-8") as f:
            data = json.load(f)
        dataset_name = j.stem.split(".")[0]
        if isinstance(data, dict) and "annotations" in data:
            imgs = {im["id"]: im for im in data.get("images", [])}
            for k, im in imgs.items():
                im["_dataset"] = dataset_name
                images[k] = dict(
                    file_name=im.get("file_name", str(k)),
                    width=im.get("width", None),
                    height=im.get("height", None),
                    dataset=dataset_name
                )
            for a in data["annotations"]:
                a = dict(a); a["_dataset"] = dataset_name
                anns_all.append(a)
        elif isinstance(data, list):
            for a in data:
                a = dict(a); a["_dataset"] = dataset_name
                anns_all.append(a)
        else:
            warnings.warn(f"Unrecognized JSON structure in {j}")

    for a in anns_all:
        if "image_id" not in a or a.get("image_id", None) is None:
            seg = a.get("segmentation", None)
            if isinstance(seg, dict) and "size" in seg:
                h, w = seg["size"]
                key = (a["_dataset"], w, h)
                image_id = hash(key) & 0x7FFFFFFF
                a["image_id"] = image_id
                if image_id not in images:
                    images[image_id] = dict(
                        file_name=str(image_id),
                        width=w, height=h,
                        dataset=a["_dataset"]
                    )

    if dataset_filter:
        anns_all = [a for a in anns_all if a.get("_dataset", None) and dataset_filter in a["_dataset"]]
        images = {k:v for k,v in images.items() if v.get("dataset") and dataset_filter in v["dataset"]}

    return images, anns_all

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Folder with COCO JSONs")
    ap.add_argument("--dataset_filter", default="", help="Filter JSONs by substring (e.g., FAPI_TEMPO)")
    ap.add_argument("--n_avrami", type=float, default=2.5, help="Fixed Avrami exponent n")
    ap.add_argument("--theta_step_deg", type=float, default=2.0, help="Angular sampling step for anisotropy")
    ap.add_argument("--use_voronoi", type=int, default=0, help="1=use Voronoi crowding correction for t0 ranks")
    ap.add_argument("--alpha_init", type=float, default=1.0, help="Init α")
    ap.add_argument("--beta_init", type=float, default=1.0, help="Init β")
    args = ap.parse_args()

    folder = Path(args.folder)
    out = folder / "output_anisotropy"     # <— fixed output folder
    out.mkdir(parents=True, exist_ok=True)

    images, anns = load_coco_folder(folder, dataset_filter=args.dataset_filter if args.dataset_filter else None)
    if len(anns) == 0:
        print("No annotations found after filtering. Exiting.")
        return

    anns_by_img = defaultdict(list)
    for a in anns:
        anns_by_img[a["image_id"]].append(a)

    rows = []
    boundary_points_global = []
    # nuclei storage not needed for hull but good to keep
    for img_id, alist in anns_by_img.items():
        imeta = images.get(img_id, dict(width=None, height=None, dataset=""))
        W = imeta.get("width", None); H = imeta.get("height", None)

        crystals = [a for a in alist if a.get("category_id", None) == 1]
        nuclei   = [a for a in alist if a.get("category_id", None) == 2]
        defects  = [a for a in alist if a.get("category_id", None) == 3]

        nucleus_points = []
        for a in nuclei:
            xy = None
            if "keypoints" in a and isinstance(a["keypoints"], list) and len(a["keypoints"]) >= 2:
                xy = (float(a["keypoints"][0]), float(a["keypoints"][1]))
            elif "point" in a and isinstance(a["point"], (list,tuple)) and len(a["point"]) >= 2:
                xy = (float(a["point"][0]), float(a["point"][1]))
            elif "bbox" in a:
                x,y,w,h = a["bbox"]
                xy = (x + w/2.0, y + h/2.0)
            if xy is not None:
                nucleus_points.append(xy)

        defect_mask = None
        for d in defects:
            mask = annotation_to_mask(d, H, W) if (H and W) else None
            if mask is None: continue
            defect_mask = mask if defect_mask is None else (defect_mask | mask)

        for c in crystals:
            mask = annotation_to_mask(c, H, W) if (H and W) else None
            if mask is None: continue
            area, peri = mask_area_perimeter(mask)
            if area <= 0 or not np.isfinite(area): continue
            C = (4.0 * math.pi * area) / (peri**2 + 1e-12)
            phi = 0.0
            if defect_mask is not None:
                overlap = (defect_mask & mask).sum()
                phi = float(overlap) / (float(area) + 1e-12)

            yx = np.column_stack(np.nonzero(mask))
            cy, cx = yx.mean(axis=0)
            nuc_xy = (cx, cy)
            if len(nucleus_points) > 0:
                pts = np.array(nucleus_points, dtype=np.float32)
                d2 = (pts[:,0]-cx)**2 + (pts[:,1]-cy)**2
                j = np.argmin(d2)
                nuc_xy = (float(pts[j,0]), float(pts[j,1]))

            boundary = mask_boundary_points(mask, n_pts=360)
            if boundary.size > 0:
                boundary_points_global.append(boundary[:, ::-1])  # (x,y)

            theta, rtheta = polar_profile_from_center(boundary, nuc_xy, theta_step_deg=args.theta_step_deg)
            (r0, eps2, theta0), rss = fit_cos2(theta, rtheta)

            R_final = math.sqrt(max(area, 0.0) / math.pi)
            rows.append(dict(
                image_id=img_id,
                dataset=imeta.get("dataset",""),
                area=area, perimeter=peri, circularity=C,
                defect_frac=phi,
                nuc_x=nuc_xy[0], nuc_y=nuc_xy[1],
                R_final=R_final,
                r0=r0, epsilon2=eps2, theta0=theta0
            ))

    df = pd.DataFrame(rows)
    if df.empty:
        print("No valid crystal masks decoded. Exiting.")
        return

    # --- t0 rank mapping (0..60 ms) ---
    if args.use_voronoi and len(df) >= 3:
        pts = df[["nuc_x","nuc_y"]].to_numpy()
        try:
            V = Voronoi(pts)
            cell_area = np.zeros(len(pts), dtype=np.float32)
            for i, r in enumerate(V.point_region):
                verts = V.regions[r]
                if not verts or -1 in verts:
                    cell_area[i] = np.nan
                else:
                    poly = np.array([V.vertices[v] for v in verts])
                    x = poly[:,0]; y=poly[:,1]
                    a = 0.5*np.abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))
                    cell_area[i] = a
            finite = np.isfinite(cell_area)
            if finite.any():
                med_area = np.nanmedian(cell_area[finite])
                cell_area[~finite] = med_area
            else:
                cell_area[:] = 1.0
            score = df["R_final"].to_numpy() * (cell_area / np.nanmedian(cell_area))
            order = np.argsort(score)
        except Exception:
            warnings.warn("Voronoi failed; falling back to size-only ranking.")
            order = np.argsort(df["R_final"].to_numpy())
    else:
        order = np.argsort(df["R_final"].to_numpy())

    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))
    q = ranks / max(1, len(order)-1)
    df["t0_ms"] = 60.0 * q

    # --- I(t) fit (lognormal vs gamma) ---
    t_nuc = df["t0_ms"].to_numpy()
    t_nuc = t_nuc[(t_nuc > 0) & (t_nuc <= 60)]

    def nll_lognormal(params):
        mu, sigma = params
        if sigma <= 1e-6: return np.inf
        ll = stats.lognorm(s=sigma, scale=np.exp(mu)).logpdf(t_nuc + 1e-9)
        return -np.sum(ll)
    mu0 = np.log(np.median(t_nuc + 1e-9)) if len(t_nuc)>0 else -1.0
    sig0 = 0.5
    res_ln = optimize.minimize(nll_lognormal, x0=[mu0, sig0], method="L-BFGS-B",
                               bounds=[(mu0-5, mu0+5),(1e-3,3)])
    mu_hat, sigma_hat = res_ln.x
    k_ln = 2
    AIC_ln = 2*k_ln + 2*res_ln.fun
    BIC_ln = k_ln*np.log(max(1,len(t_nuc))) + 2*res_ln.fun

    def nll_gamma(params):
        k, theta = params
        if k<=1e-6 or theta<=1e-9: return np.inf
        ll = stats.gamma(a=k, scale=theta).logpdf(t_nuc)
        return -np.sum(ll)
    res_ga = optimize.minimize(nll_gamma, x0=[2.0, np.mean(t_nuc)/2.0 if len(t_nuc)>0 else 10.0],
                               method="L-BFGS-B", bounds=[(1e-3, 10.0),(1e-6, 200.0)])
    k_hat, theta_hat = res_ga.x
    k_ga = 2
    AIC_ga = 2*k_ga + 2*res_ga.fun
    BIC_ga = k_ga*np.log(max(1,len(t_nuc))) + 2*res_ga.fun

    model_I = "gamma" if (BIC_ga < BIC_ln or (BIC_ga==BIC_ln and AIC_ga<=AIC_ln)) else "lognormal"
    t_grid = np.arange(0.5, 60.0, 1.0)
    if model_I == "gamma":
        I_grid = stats.gamma(a=k_hat, scale=theta_hat).pdf(t_grid)
        I_params = {"k": float(k_hat), "theta": float(theta_hat)}
        AIC, BIC = float(AIC_ga), float(BIC_ga)
    else:
        I_grid = stats.lognorm(s=sigma_hat, scale=np.exp(mu_hat)).pdf(t_grid)
        I_params = {"mu": float(mu_hat), "sigma": float(sigma_hat)}
        AIC, BIC = float(AIC_ln), float(BIC_ln)

    # --- Growth with anisotropy: solve v0, α, β by final area fit ---
    Ci = df["circularity"].to_numpy()
    phi = df["defect_frac"].to_numpy()
    eps2 = df["epsilon2"].to_numpy()
    theta0_arr = df["theta0"].to_numpy()  # kept for completeness
    Rf = df["R_final"].to_numpy()
    t0_ms = df["t0_ms"].to_numpy()
    dt_ms = np.maximum(600.0 - t0_ms, 1e-6)

    thetas = np.linspace(0, 2*np.pi, int(np.ceil(360.0 / max(0.5, args.theta_step_deg))), endpoint=False)

    def angular_sq_mean(e2):
        arr = 1.0 + e2*np.cos(2*thetas)
        arr = np.maximum(arr, 0.0)
        return float(np.mean(arr**2))

    M_ang = np.array([angular_sq_mean(e) for e in eps2], dtype=np.float64)
    A_obs = (np.pi * Rf**2)

    def resid_valpha_beta(params):
        v0, alpha, beta = params
        if v0 <= 0 or alpha < 0 or beta < 0:
            return 1e12
        G = np.exp(-alpha*(1.0 - Ci)) * np.exp(-beta*phi) * (dt_ms/1000.0)
        A_pred = np.pi * (v0**2) * (G**2) * M_ang
        return np.sum((A_pred - A_obs)**2)

    x0 = [np.median(Rf/(dt_ms/1000.0+1e-9)), args.alpha_init, args.beta_init]
    bounds = [(1e-6, 1e3), (0.0, 10.0), (0.0, 10.0)]
    res_growth = optimize.minimize(resid_valpha_beta, x0=x0, bounds=bounds, method="L-BFGS-B")
    v0_hat, alpha_hat, beta_hat = res_growth.x

    # time-resolved X_pred(t) using compact analytic area form
    t_ms_full = np.arange(0.0, 600.0+1e-6, 1.0)
    A_pred_curves = []
    for idx in range(len(df)):
        e2 = eps2[idx]; t0i = t0_ms[idx]
        Ci_i = Ci[idx]; phi_i = phi[idx]
        base = v0_hat * np.exp(-alpha_hat*(1.0 - Ci_i)) * np.exp(-beta_hat*phi_i)
        mean_ang_sq = np.mean(np.maximum(1.0 + e2*np.cos(2*(thetas - 0.0)), 0.0)**2)  # theta0 drops in the mean
        A_t = np.zeros_like(t_ms_full, dtype=np.float64)
        grow_mask = t_ms_full >= t0i
        dt_s = (t_ms_full[grow_mask] - t0i)/1000.0
        A_t[grow_mask] = np.pi * (base**2) * (dt_s**2) * mean_ang_sq
        A_pred_curves.append(A_t)
    A_pred_curves = np.vstack(A_pred_curves)

    # effective FOV area via convex hull of all boundary points
    all_pts = []
    for img_id, alist in anns_by_img.items():
        # collect crystal boundaries again for hull (cheap vs tracking)
        pass
    # We can approximate A_eff with sum of final areas * 1.25 if hull points aren’t available
    A_eff = float(np.sum(df["area"].to_numpy())) * 1.25

    X_pred = np.clip(A_pred_curves.sum(axis=0) / (A_eff + 1e-9), 0.0, 1.0)

    # Avrami fit (n fixed)
    n_av = float(args.n_avrami)
    def X_avrami(t_ms, K):
        t = t_ms/1000.0
        return 1.0 - np.exp(-(K*(t**n_av)))
    def lsq_loss(K):
        if K <= 0: return 1e12
        return np.sum((X_avrami(t_ms_full, K) - X_pred)**2)
    resK = optimize.minimize_scalar(lsq_loss, bounds=(1e-6, 1e3), method="bounded")
    K_hat = float(resK.x)

    # --- Save outputs to <folder>/output_anisotropy ---
    df_params = df.copy()
    df_params["v0_hat"] = v0_hat
    df_params["alpha_hat"] = alpha_hat
    df_params["beta_hat"] = beta_hat
    df_params.to_csv(out/"per_object_metrics_anisotropy.csv", index=False)

    pd.DataFrame({"t_ms": t_grid, "I_t": I_grid, **I_params}).to_csv(out/"nucleation_I_t.csv", index=False)
    pd.DataFrame({"t_ms": t_ms_full, "X_pred": X_pred}).to_csv(out/"X_pred.csv", index=False)
    pd.DataFrame({"t_ms": t_ms_full, "X_avrami": X_avrami(t_ms_full, K_hat)}).to_csv(out/"X_avrami.csv", index=False)

    # Growth curves (chunked)
    N = A_pred_curves.shape[0]; chunk = 500
    for s in range(0, N, chunk):
        e = min(N, s+chunk)
        df_chunk = pd.DataFrame(A_pred_curves[s:e, :], columns=[f"t{int(t)}ms" for t in t_ms_full])
        df_chunk.insert(0, "idx", np.arange(s, e))
        df_chunk.to_csv(out/f"growth_curves_chunk_{s}_{e-1}.csv", index=False)

    # Plots
    plt.figure(figsize=(6,4))
    plt.plot(t_grid, I_grid, lw=2)
    plt.title(f"I(t): {model_I}  params={I_params}  AIC={AIC:.1f}  BIC={BIC:.1f}")
    plt.xlabel("t (ms, nucleation window)"); plt.ylabel("I(t) (pdf)")
    plt.tight_layout(); plt.savefig(out/"I_t.png", dpi=180); plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(t_ms_full, X_pred, label="X_pred", lw=2)
    plt.plot(t_ms_full, X_avrami(t_ms_full, K_hat), "--", label=f"Avrami n={n_av}, K={K_hat:.3g}", lw=2)
    plt.ylim(0,1.05); plt.xlabel("t (ms)"); plt.ylabel("Transformed fraction X(t)")
    plt.legend(); plt.tight_layout(); plt.savefig(out/"X_overlay.png", dpi=180); plt.close()

    plt.figure(figsize=(6,4))
    plt.hist(df["epsilon2"].replace([np.inf,-np.inf], np.nan).dropna(), bins=40)
    plt.xlabel("ε2 (anisotropy amplitude)"); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(out/"epsilon2_hist.png", dpi=180); plt.close()

    with open(out/"summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Crystals analyzed: {len(df)}\n")
        f.write(f"I(t) model: {model_I}\n")
        f.write(f"I(t) params: {dict(I_params)}\n")
        f.write(f"AIC={AIC:.3f}  BIC={BIC:.3f}\n")
        f.write(f"Avrami: n={n_av}, K={K_hat:.6g}\n")
        f.write(f"v0={v0_hat:.6g}, alpha={alpha_hat:.6g}, beta={beta_hat:.6g}\n")
        f.write(f"Outputs: {out}\n")

    print("=== Done with anisotropy model ===")
    print(f"Crystals: {len(df)}")
    print(f"I(t): {model_I}  params={I_params}  AIC={AIC:.1f}  BIC={BIC:.1f}")
    print(f"Avrami: n={n_av}, K={K_hat:.3g}")
    print(f"v0={v0_hat:.3g}, alpha={alpha_hat:.3g}, beta={beta_hat:.3g}")
    print(f"Outputs in: {out}")

if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
