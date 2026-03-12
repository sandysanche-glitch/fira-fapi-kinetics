# -*- coding: utf-8 -*-
"""
COCO nucleation/growth analysis with inferred nucleation points
- category_id: 1=crystal mask, 2=nucleus point (optional), 3=defect mask (optional)
- If no nucleus points are present, infer per-crystal nucleation point as the distance-transform argmax.
- Nucleation t0 derived from adjusted radial extent from nucleation point (larger/rounder/cleaner -> earlier).
- Growth calibrated to reach final radius by 600 ms with morphology penalties.
- I(t) fit (lognormal vs gamma) on 0..60 ms, 1-ms grid (BIC/AIC).
- Avrami overlay X_Avrami(t)=1-exp(-K t^n) (fix n, fit K) vs X_pred(t).
"""

import os, json, glob, argparse, warnings
import numpy as np
import pandas as pd

from skimage import measure
from skimage.draw import line as sk_line
from scipy.ndimage import distance_transform_edt
from scipy import optimize, stats
import matplotlib.pyplot as plt

# COCO RLE decoder
try:
    from pycocotools import mask as cocomask
    HAS_COCO = True
except Exception:
    HAS_COCO = False
    warnings.warn("pycocotools not found. If masks are RLE only, please `pip install pycocotools`.")

MS_END = 600.0   # total solidification window (ms)
MS_NUC = 60.0    # nucleation window (ms)
EPS    = 1e-9

# ------------------ I/O ------------------

def load_instances(json_path):
    """
    Return list of dicts with fields:
      'category_id', 'mask' (np.bool_) for crystals/defects, 'point' for nucleus (x,y), 'file'
    Accepts either:
      (a) simple list of dicts with 'segmentation':{'size':[H,W],'counts':<rle>}
      (b) COCO dict {images, annotations, categories}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = []
    base = os.path.dirname(json_path)

    # Simple list format (like your FAPI_TEMPO_*.json)
    if isinstance(data, list):
        for det in data:
            cat = det.get("category_id", 1)
            out = {"category_id": cat, "file": json_path}

            # nucleus point (if present)
            if cat == 2:
                if "keypoints" in det and isinstance(det["keypoints"], (list, tuple)) and len(det["keypoints"]) >= 2:
                    out["point"] = (float(det["keypoints"][0]), float(det["keypoints"][1]))
                    items.append(out); continue
                if "bbox" in det and "segmentation" not in det:
                    x,y,w,h = det["bbox"]
                    out["point"] = (float(x + w/2), float(y + h/2))
                    items.append(out); continue

            # masks
            mask = None
            if "segmentation" in det and isinstance(det["segmentation"], dict) and "counts" in det["segmentation"]:
                if not HAS_COCO:
                    raise RuntimeError("RLE present but pycocotools not installed.")
                rle = {"counts": det["segmentation"]["counts"], "size": det["segmentation"]["size"]}
                mask = cocomask.decode(rle).astype(bool)
            elif "mask_name" in det:
                import imageio.v2 as imageio
                p = os.path.join(base, det["mask_name"])
                if os.path.isfile(p):
                    arr = imageio.imread(p); mask = (arr > 0)
            elif "bbox" in det and "segmentation" not in det:
                x,y,w,h = det["bbox"]
                H = int(max(y+h+5, 1)); W = int(max(x+w+5, 1))
                mask = np.zeros((H, W), dtype=bool)
                mask[int(y):int(y+h), int(x):int(x+w)] = True

            if mask is not None:
                out["mask"] = mask
                items.append(out)
        return items

    # COCO dict format
    images = {im["id"]: im for im in data.get("images", [])}
    anns = data.get("annotations", [])

    from skimage.draw import polygon

    for ann in anns:
        cat = ann.get("category_id", 1)
        out = {"category_id": cat, "file": json_path}

        if cat == 2:
            if "keypoints" in ann and isinstance(ann["keypoints"], (list, tuple)) and len(ann["keypoints"]) >= 2:
                out["point"] = (float(ann["keypoints"][0]), float(ann["keypoints"][1]))
                items.append(out); continue
            if "bbox" in ann:
                x,y,w,h = ann["bbox"]
                out["point"] = (float(x + w/2), float(y + h/2))
                items.append(out); continue
            continue

        mask = None
        if "segmentation" in ann and isinstance(ann["segmentation"], dict) and "counts" in ann["segmentation"]:
            if not HAS_COCO:
                raise RuntimeError("RLE present but pycocotools not installed.")
            rle = {"counts": ann["segmentation"]["counts"], "size": ann["segmentation"]["size"]}
            mask = cocomask.decode(rle).astype(bool)
        elif "segmentation" in ann and isinstance(ann["segmentation"], list):
            im = images.get(ann["image_id"])
            if im:
                H, W = im["height"], im["width"]
                poly = ann["segmentation"][0]
                xs = np.array(poly[0::2]); ys = np.array(poly[1::2])
                rr, cc = polygon(ys, xs, (H, W))
                mask = np.zeros((H, W), dtype=bool); mask[rr, cc] = True
        elif "mask_name" in ann:
            import imageio.v2 as imageio
            p = os.path.join(base, ann["mask_name"])
            if os.path.isfile(p):
                arr = imageio.imread(p); mask = (arr > 0)

        if mask is not None:
            out["mask"] = mask
            items.append(out)

    return items

# ------------------ geometry ------------------

def mask_area_perimeter(mask: np.ndarray):
    A = float(mask.sum())
    per = measure.perimeter(mask, neighbourhood=8)
    return A, per

def circularity(A, P):
    if P <= 0: return 0.0
    return float(4.0*np.pi*A / (P*P))

def defect_fraction(crystal_mask, defect_mask):
    if defect_mask is None:
        return 0.0
    inter = (crystal_mask & defect_mask).sum()
    total = crystal_mask.sum() + EPS
    return float(inter / total)

def find_nucleation_points(items):
    return [tuple(it["point"]) for it in items if it.get("category_id", 1) == 2 and "point" in it]

def infer_center_by_dist_transform(mask: np.ndarray):
    """If no explicit nucleus is available, take argmax of distance_transform_edt(mask)."""
    if not mask.any():
        return (0.0, 0.0)
    dt = distance_transform_edt(mask)
    y0, x0 = np.unravel_index(np.argmax(dt), dt.shape)
    return (float(x0), float(y0))

def nuclei_to_crystals_assignment(crystals, nuclei_pts):
    """Assign origin to each crystal: nearest explicit nucleus or DT-argmax if none."""
    from skimage.measure import regionprops
    pts = []
    has_nuclei = len(nuclei_pts) > 0

    for c in crystals:
        mask = c["mask"]
        if has_nuclei:
            props = regionprops(mask.astype(np.uint8))
            if props:
                cy, cx = props[0].centroid
            else:
                ys, xs = np.nonzero(mask)
                cy, cx = ys.mean(), xs.mean()
            d2 = [(cx - px)**2 + (cy - py)**2 for (px, py) in nuclei_pts]
            j = int(np.argmin(d2))
            pts.append((float(nuclei_pts[j][0]), float(nuclei_pts[j][1])))
        else:
            pts.append(infer_center_by_dist_transform(mask))
    return pts

def radial_extent_from_point(mask: np.ndarray, x0: float, y0: float, num_angles: int = 72):
    """Cast rays from (x0,y0) to boundary; return median and max radii in pixels."""
    H, W = mask.shape
    thetas = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
    radii = []

    # ensure (x0,y0) inside; if not, snap to nearest mask pixel
    if not (0 <= int(y0) < H and 0 <= int(x0) < W and mask[int(y0), int(x0)]):
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return 0.0, 0.0
        d2 = (xs - x0)**2 + (ys - y0)**2
        k = int(np.argmin(d2)); x0, y0 = float(xs[k]), float(ys[k])

    for th in thetas:
        x1 = x0 + 2*max(H,W)*np.cos(th)
        y1 = y0 + 2*max(H,W)*np.sin(th)
        rr, cc = sk_line(int(round(y0)), int(round(x0)), int(round(y1)), int(round(x1)))
        prev = None
        for (r, c) in zip(rr, cc):
            if not (0 <= r < H and 0 <= c < W): break
            if not mask[r, c]: break
            prev = (r, c)
        if prev is None: 
            continue
        ry, rx = prev
        radii.append(np.hypot(rx - x0, ry - y0))

    if len(radii) == 0:
        return 0.0, 0.0
    radii = np.asarray(radii, float)
    return float(np.median(radii)), float(np.max(radii))

# ------------------ stats/models ------------------

def fit_nucleation_pdf(t0_ms):
    """Fit lognormal and gamma on (0,60] ms; return selected model on 1-ms grid + params + AIC/BIC."""
    t = np.asarray(t0_ms, float)
    t = t[(t > 0) & (t <= MS_NUC)]
    if len(t) < 3:
        return None

    # Lognormal MLE (loc=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s, loc, scale = stats.lognorm.fit(t, floc=0)
    ll_logn = np.sum(stats.lognorm.logpdf(t, s, loc=0, scale=scale))
    k_logn = 2  # s, scale
    aic_logn = 2*k_logn - 2*ll_logn
    bic_logn = k_logn*np.log(len(t)) - 2*ll_logn

    # Gamma MLE (loc=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k_hat, loc_g, theta_hat = stats.gamma.fit(t, floc=0)
    ll_gamma = np.sum(stats.gamma.logpdf(t, k_hat, loc=0, scale=theta_hat))
    k_gamma = 2  # k, theta
    aic_gamma = 2*k_gamma - 2*ll_gamma
    bic_gamma = k_gamma*np.log(len(t)) - 2*ll_gamma

    grid = np.arange(1, int(MS_NUC)+1, dtype=float)  # 1..60 ms

    if bic_logn <= bic_gamma:
        I_grid = stats.lognorm.pdf(grid, s, loc=0, scale=scale)
        sel = {"model": "lognormal", "params": {"mu": float(np.log(scale)), "sigma": float(s)}, "AIC": float(aic_logn), "BIC": float(bic_logn)}
    else:
        I_grid = stats.gamma.pdf(grid, k_hat, loc=0, scale=theta_hat)
        sel = {"model": "gamma", "params": {"k": float(k_hat), "theta": float(theta_hat)}, "AIC": float(aic_gamma), "BIC": float(bic_gamma)}

    I_grid = I_grid / (I_grid.sum() + EPS)  # normalize for display
    nuc_df = pd.DataFrame({"t_ms": grid, "I_t": I_grid})
    return nuc_df, sel

def fit_avrami_K(t_ms, X_pred, n_fixed):
    """Least-squares K fit in X_Av = 1 - exp(-K t^n) against X_pred(t). t is in ms."""
    t = np.asarray(t_ms, float)
    X = np.asarray(X_pred, float)
    m = min(len(t), len(X))
    t = t[:m]; X = X[:m]

    def model(K):
        return 1.0 - np.exp(-K * np.power(t/1000.0, n_fixed))

    def loss(K):
        y = model(K)
        return np.sum((y - X)**2)

    res = optimize.minimize_scalar(loss, bounds=(1e-9, 1e3), method="bounded")
    K_fit = float(res.x)
    return K_fit, model(K_fit), t

# ------------------ main ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Folder with *.json COCO/RLE files")
    ap.add_argument("--out", required=True, help="Output folder")
    ap.add_argument("--dataset_filter", default="", help="Only files starting with this prefix (e.g. FAPI_TEMPO)")
    ap.add_argument("--n_avrami", type=float, default=2.5, help="Fixed Avrami exponent n")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "plots"), exist_ok=True)

    jsons = sorted(glob.glob(os.path.join(args.folder, "*.json")))
    if args.dataset_filter:
        jsons = [p for p in jsons if os.path.basename(p).startswith(args.dataset_filter)]
    if not jsons:
        raise RuntimeError("No JSON files found for the given filter/folder.")

    all_crystals, all_nuclei = [], []
    defects_union = None

    for jp in jsons:
        items = load_instances(jp)
        for it in items:
            cat = it.get("category_id", 1)
            if cat == 1 and "mask" in it:
                all_crystals.append({"mask": it["mask"], "file": jp})
            elif cat == 2 and "point" in it:
                all_nuclei.append(tuple(it["point"]))
            elif cat == 3 and "mask" in it:
                m = it["mask"].astype(bool)
                defects_union = m if defects_union is None else (defects_union | m)

    if not all_crystals:
        raise RuntimeError("No crystal masks (category_id==1) found.")

    # assign nucleation point to each crystal (explicit nucleus if present; else DT center)
    nuc_points = nuclei_to_crystals_assignment(all_crystals, all_nuclei)

    # per-crystal metrics & radial extents
    rows = []
    for i, c in enumerate(all_crystals):
        mask = c["mask"]
        A, P = mask_area_perimeter(mask)
        C = circularity(A, P)
        phi = defect_fraction(mask, defects_union)
        x0, y0 = nuc_points[i]
        r_med, r_max = radial_extent_from_point(mask, x0, y0, num_angles=72)

        rows.append({
            "crystal_id": i,
            "A_final_px2": A,
            "P_px": P,
            "C_circularity": C,
            "phi_defect": phi,
            "x0": x0, "y0": y0,
            "r_med_px": r_med,
            "r_max_px": r_max
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No usable crystals after metric computation.")

    # adjusted extent (bigger, rounder, cleaner -> earlier nucleation)
    alpha_init, beta_init = 1.0, 1.0
    extent_adj = df["r_med_px"] * np.exp(-alpha_init*(1.0 - df["C_circularity"])) * np.exp(-beta_init*df["phi_defect"])
    order = extent_adj.rank(method="average", ascending=False)  # largest => earliest
    q = (order - 1.0) / (len(df) - 1.0 + EPS)
    df["t0_ms"] = MS_NUC * q

    # calibrate v0 from median of Ri / (Δt_i * penalties)
    R_target = np.sqrt(np.maximum(df["A_final_px2"].values, 0.0) / np.pi)
    penalties = np.exp(-alpha_init*(1.0 - df["C_circularity"].values)) * np.exp(-beta_init*df["phi_defect"].values)
    delta_t = np.maximum(MS_END - df["t0_ms"].values, 1.0)
    v0 = np.median(R_target / (delta_t * penalties + EPS))
    alpha, beta = alpha_init, beta_init

    # per-object predicted curves (0..600 ms)
    grid = np.arange(0, int(MS_END)+1, dtype=float)
    recs = []
    for i, row in df.iterrows():
        v_i = v0 * np.exp(-alpha*(1.0 - row["C_circularity"])) * np.exp(-beta*row["phi_defect"])
        r_t = v_i * np.maximum(grid - row["t0_ms"], 0.0)
        A_pred = np.pi * (r_t**2)
        A_pred = np.minimum(A_pred, row["A_final_px2"])
        for t, A in zip(grid, A_pred):
            recs.append({"crystal_id": row["crystal_id"], "t_ms": t, "A_pred_px2": A})
    df_curves = pd.DataFrame(recs)

    # X_pred(t): normalize by total final area so X_pred(600) ~ 1
    A_tot = df["A_final_px2"].sum() + EPS
    X_pred = df_curves.groupby("t_ms")["A_pred_px2"].sum().reset_index()
    X_pred["X_pred"] = X_pred["A_pred_px2"] / A_tot

    # I(t) fit
    nuc_fit = fit_nucleation_pdf(df["t0_ms"].values)
    if nuc_fit is not None:
        nuc_df, sel = nuc_fit
        nuc_df.to_csv(os.path.join(args.out, "nucleation_I_t.csv"), index=False)
        pd.DataFrame([{**{"model": sel["model"]}, **sel["params"], **{"AIC": sel["AIC"], "BIC": sel["BIC"]}}]) \
          .to_csv(os.path.join(args.out, "model_selection.csv"), index=False)
    else:
        nuc_df, sel = None, None
        warnings.warn("Not enough nucleation samples to fit I(t).")

    # Avrami K fit (n fixed)
    K_fit, X_av, t_av = fit_avrami_K(X_pred["t_ms"].values, X_pred["X_pred"].values, args.n_avrami)
    xa = pd.DataFrame({"t_ms": t_av,
                       "X_pred": np.interp(t_av, X_pred["t_ms"].values, X_pred["X_pred"].values),
                       "X_avrami": X_av})
    xa.to_csv(os.path.join(args.out, "X_pred_vs_Avrami.csv"), index=False)

    # per-object curves & metrics
    df_curves.to_csv(os.path.join(args.out, "per_object_growth_curves.csv"), index=False)
    df.to_csv(os.path.join(args.out, "per_object_metrics.csv"), index=False)

    # ---- plots ----
    if nuc_df is not None:
        plt.figure(figsize=(6,4))
        plt.plot(nuc_df["t_ms"], nuc_df["I_t"], lw=2)
        ttl = f"I(t) fit: {sel['model']}  params={sel['params']}  BIC={sel['BIC']:.1f}"
        plt.title(ttl)
        plt.xlabel("t0 (ms)")
        plt.ylabel("I(t) (normalized)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "plots", "I_t_fit.png"), dpi=200)
        plt.close()

    plt.figure(figsize=(7,5))
    step = max(1, len(df)//20)  # ~5% sampling for clarity
    for cid, g in df_curves.groupby("crystal_id"):
        if cid % step == 0:
            plt.plot(g["t_ms"], g["A_pred_px2"], alpha=0.5)
    plt.xlabel("t (ms)")
    plt.ylabel("Predicted area (px²)")
    plt.title("Per-object predicted growth (subset)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "plots", "growth_overlays.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(X_pred["t_ms"], X_pred["X_pred"], label="X_pred (Σ A_pred / Σ A_final)", lw=2)
    plt.plot(t_av, X_av, label=f"Avrami (n={args.n_avrami}, K={K_fit:.3g})", lw=2, ls="--")
    plt.xlabel("t (ms)")
    plt.ylabel("Transformed fraction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "plots", "X_pred_Avrami.png"), dpi=200)
    plt.close()

    # summary
    print("\n=== Done ===")
    print(f"Crystals: {len(df)}  | Nuclei points present: {0 if nuc_df is None else 'see model_selection.csv'}")
    if sel is not None:
        print(f"I(t): {sel['model']}  params={sel['params']}  AIC={sel['AIC']:.1f}  BIC={sel['BIC']:.1f}")
    print(f"Avrami: n={args.n_avrami}, K={K_fit:.3g}")
    print(f"Outputs in: {args.out}")

if __name__ == "__main__":
    main()
