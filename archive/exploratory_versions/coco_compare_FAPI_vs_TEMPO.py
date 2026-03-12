# coco_compare_FAPI_vs_TEMPO.py
# Compare FAPI vs FAPI-TEMPO nucleation (dn/dt) and growth (X(t)) from single-snapshot COCO JSONs.
#
# Usage (Windows; carets for multiline):
#   python coco_compare_FAPI_vs_TEMPO.py ^
#     --folders "D:\...\comparative datasets\FAPI" "D:\...\comparative datasets\FAPI-TEMPO" ^
#     --labels  FAPI FAPI-TEMPO ^
#     --n_avrami 2.5 ^
#     --bootstrap 300 ^
#     --calibrate_penalties
#
# If --out is not provided, output goes to:
#   <parent of first folder>\comparative_outputs_FAPI_vs_TEMPO
#
# Requirements: numpy, pandas, matplotlib, scipy, scikit-image
# COCO categories: 1=crystal(mask), 2=nucleus(point), 3=defect(mask)

import os, json, math, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize
from skimage.draw import polygon2mask
from skimage.measure import find_contours

# -------------------------
# Utilities
# -------------------------
def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def as_iter(obj):
    return obj if isinstance(obj, (list, tuple)) else [obj]

def infer_hw_from_poly(seg):
    xs, ys = [], []
    for poly in as_iter(seg):
        arr = np.asarray(poly, dtype=float).reshape(-1, 2)
        xs.append(arr[:,0]); ys.append(arr[:,1])
    x = np.concatenate(xs) if xs else np.array([0.0])
    y = np.concatenate(ys) if ys else np.array([0.0])
    w = int(np.clip(np.ceil(x.max())+4, 8, 16384))
    h = int(np.clip(np.ceil(y.max())+4, 8, 16384))
    return h, w

def mask_from_annotation(ann, H=None, W=None):
    """
    Returns (mask, H, W)
    Supports:
      - segmentation as RLE dict {'counts':..., 'size':[H,W]} with uncompressed counts (list)
      - segmentation as list of polygon arrays
    If H/W missing, infers from segmentation.
    """
    seg = ann.get("segmentation", None)
    if seg is None:
        return None, H, W
    # RLE dict (uncompressed)
    if isinstance(seg, dict) and "size" in seg and "counts" in seg:
        Hs, Ws = int(seg["size"][0]), int(seg["size"][1])
        if H is None or W is None:
            H, W = Hs, Ws
        counts = seg["counts"]
        if isinstance(counts, list):
            arr = np.zeros(H*W, dtype=np.uint8)
            idx = 0
            val = 0
            for run in counts:
                run = int(run)
                if idx+run > arr.size:
                    run = max(0, arr.size-idx)
                if val == 1:
                    arr[idx:idx+run] = 1
                idx += run
                val = 1 - val
            m = arr.reshape((H, W), order="F")  # COCO RLE is column-major
            return m, H, W
        else:
            raise RuntimeError("Compressed RLE encountered. Install pycocotools to decode.")
    # Polygons
    if isinstance(seg, list):
        if H is None or W is None:
            H, W = infer_hw_from_poly(seg)
        m = np.zeros((H, W), dtype=np.uint8)
        for poly in seg:
            pts = np.asarray(poly, dtype=float).reshape(-1, 2)
            rr_mask = polygon2mask((H, W), np.fliplr(pts))  # expects (row=y, col=x)
            m |= rr_mask.astype(np.uint8)
        return m, H, W
    return None, H, W

def area_perimeter_from_mask(m):
    if m is None:
        return np.nan, np.nan
    A = float(m.sum())
    if A <= 0:
        return 0.0, 0.0
    per = 0.0
    for c in find_contours(m, 0.5):
        d = np.diff(c, axis=0)
        per += float(np.sum(np.sqrt((d**2).sum(1))))
    return A, per

def centroid_from_mask(m):
    if m is None or m.sum() == 0:
        return (np.nan, np.nan)
    ys, xs = np.nonzero(m)
    return (float(xs.mean()), float(ys.mean()))

def nearest_point_to_mask(m, x, y):
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return (np.nan, np.nan, np.inf)
    d2 = (xs - x)**2 + (ys - y)**2
    idx = int(np.argmin(d2))
    return float(xs[idx]), float(ys[idx]), float(np.sqrt(d2[idx]))

def circularity(area, per):
    if per <= 0 or area <= 0:
        return 0.0
    return float(4*math.pi*area/(per**2))

# -------------------------
# Loading a folder of COCO JSONs
# -------------------------
def load_coco_many(folder: Path, dataset_label: str, verbose=True):
    """Return tidy DataFrame with one row per crystal (merged with nucleus/defect info when possible)."""
    jsons = sorted([p for p in folder.rglob("*.json")])
    if verbose:
        print(f'Analyzing: {folder}  [{dataset_label}]  JSONs found: {len(jsons)}')
    rows = []

    for jp in jsons:
        try:
            data = load_json(jp)
        except Exception as e:
            print(f"[SKIP] {jp.name}: JSON load error: {e}")
            continue

        if isinstance(data, dict) and "annotations" in data:
            anns = data.get("annotations", [])
            images = {im["id"]: im for im in data.get("images", [])} if "images" in data else {}
        elif isinstance(data, list):
            anns = data
            images = {}
        else:
            anns = data.get("annotations", []) if isinstance(data, dict) else []
            if not anns:
                print(f"[SKIP] {jp.name}: No annotations found.")
                continue
            images = {}

        # First pass: collect nuclei & defects per image
        nuclei_pts = {}
        defects_masks = {}
        size_cache = {}

        for ann in anns:
            cat = ann.get("category_id", None)
            if cat not in (1,2,3):
                continue
            image_id = ann.get("image_id", None)
            seg = ann.get("segmentation", None)
            # infer H,W
            H, W = None, None
            if image_id in images:
                H = images[image_id].get("height", None)
                W = images[image_id].get("width", None)
            if isinstance(seg, dict) and "size" in seg:
                H = H or int(seg["size"][0]); W = W or int(seg["size"][1])
            elif isinstance(seg, list) and (H is None or W is None):
                H, W = infer_hw_from_poly(seg)
            if image_id is not None and (H is not None) and (W is not None):
                size_cache[image_id] = (int(H), int(W))

            if cat == 2:
                # nuclei points
                pt = ann.get("keypoints", None)
                if pt and len(pt) >= 2:
                    nx, ny = float(pt[0]), float(pt[1])
                else:
                    if isinstance(seg, list) and len(seg)>0:
                        xs, ys = [], []
                        for poly in seg:
                            arr = np.asarray(poly, dtype=float).reshape(-1, 2)
                            xs.append(arr[:,0]); ys.append(arr[:,1])
                        nx, ny = float(np.mean(np.concatenate(xs))), float(np.mean(np.concatenate(ys)))
                    else:
                        nx, ny = np.nan, np.nan
                nuclei_pts.setdefault(image_id, []).append((nx, ny))
            elif cat == 3:
                # defects mask union per image
                try:
                    m, HH, WW = mask_from_annotation(ann, H, W)
                except RuntimeError:
                    continue
                if m is not None:
                    defects_masks.setdefault(image_id, []).append(m)

        # union defects per image
        defects_union = {}
        for img_id, masks in defects_masks.items():
            H, W = size_cache.get(img_id, (None, None))
            if (H is None or W is None) and masks:
                H, W = masks[0].shape
            mu = np.zeros((H, W), dtype=np.uint8)
            for m in masks:
                mu |= m.astype(np.uint8)
            defects_union[img_id] = mu

        # Second pass: crystals
        for ann in anns:
            if ann.get("category_id") != 1:
                continue
            image_id = ann.get("image_id", None)
            H, W = size_cache.get(image_id, (None, None))
            seg = ann.get("segmentation", None)
            try:
                m, HH, WW = mask_from_annotation(ann, H, W)
            except RuntimeError:
                continue
            if m is None:
                continue

            A, P = area_perimeter_from_mask(m)
            C = circularity(A, P)
            cx, cy = centroid_from_mask(m)

            # nearest nucleus to this mask centroid
            nx, ny, ndist = (np.nan, np.nan, np.nan)
            if image_id in nuclei_pts and len(nuclei_pts[image_id])>0:
                pts = np.array(nuclei_pts[image_id], dtype=float)
                d2 = (pts[:,0]-cx)**2 + (pts[:,1]-cy)**2
                j = int(np.argmin(d2))
                nx, ny = float(pts[j,0]), float(pts[j,1])
                if not np.isnan(nx):
                    _, _, ndist = nearest_point_to_mask(m, nx, ny)

            # defect fraction by overlap
            phi = 0.0
            if image_id in defects_union:
                mu = defects_union[image_id]
                inter = (mu & m).sum()
                phi = float(inter) / float(A+1e-6)

            rows.append({
                "dataset": dataset_label,
                "file": str(jp),
                "image_id": image_id,
                "area_px": float(A),
                "perimeter_px": float(P),
                "circularity": float(C),
                "centroid_x": float(cx), "centroid_y": float(cy),
                "nuc_x": float(nx), "nuc_y": float(ny), "nuc_dist_px": float(ndist),
                "defect_frac": float(phi),
                "H": int(HH) if HH else None, "W": int(WW) if WW else None,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"[SUMMARY {dataset_label}] crystals=0, nuclei=0, defects=?, rows_kept=0, skips=0")
        return df
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["area_px"])
    crystals = (df["area_px"]>0).sum()
    nuclei   = np.isfinite(df["nuc_x"]).sum()
    defects  = (df["defect_frac"]>0).sum()
    print(f"[SUMMARY {dataset_label}] crystals={crystals}, nuclei={nuclei}, defects={defects}, rows_kept={len(df)}, skips=0")
    return df

# -------------------------
# Nucleation time & model fit
# -------------------------
def rank_to_t0_ms(areas_px):
    a = np.asarray(areas_px, float)
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    q = ranks / max(1, len(a)-1)
    return 60.0 * q  # 0..60 ms

def fit_logN(t):
    t = np.asarray(t, float)
    t = t[(t>0)&(t<=60)&np.isfinite(t)]
    if len(t)<5:
        return None
    l = np.log(t)
    mu = float(l.mean()); sigma = float(l.std(ddof=1))
    grid = np.arange(0,61,1,dtype=float)
    pdf = np.zeros_like(grid, dtype=float)
    gpos = grid>0
    pdf[gpos] = stats.lognorm.pdf(grid[gpos], s=sigma, scale=np.exp(mu))
    ll = np.sum(stats.lognorm.logpdf(t, s=sigma, scale=np.exp(mu)))
    k = 2; n = len(t)
    aic = 2*k - 2*ll
    bic = k*np.log(n) - 2*ll
    return {"model":"lognormal","grid_t":grid,"pdf":pdf,"mu":mu,"sigma":sigma,"AIC":aic,"BIC":bic}

def fit_gamma(t):
    t = np.asarray(t, float)
    t = t[(t>0)&(t<=60)&np.isfinite(t)]
    if len(t)<5:
        return None
    k, loc, theta = stats.gamma.fit(t, floc=0)
    grid = np.arange(0,61,1,dtype=float)
    pdf = stats.gamma.pdf(grid, a=k, loc=0, scale=theta)
    ll = np.sum(stats.gamma.logpdf(t, a=k, loc=0, scale=theta))
    n = len(t); p=2
    aic = 2*p - 2*ll
    bic = p*np.log(n) - 2*ll
    return {"model":"gamma","grid_t":grid,"pdf":pdf,"k":k,"theta":theta,"AIC":aic,"BIC":bic}

def pick_I_model(t0_ms):
    g = fit_gamma(t0_ms)
    l = fit_logN(t0_ms)
    if g is None and l is None:
        return None
    if g is None: return l
    if l is None: return g
    return g if g["BIC"] < l["BIC"] else l

# -------------------------
# Growth model & Avrami
# -------------------------
def calibrate_v0_alpha_beta(df, use_penalties=True):
    A = np.asarray(df["area_px"], float)
    R = np.sqrt(A/np.pi)
    C  = np.clip(np.asarray(df["circularity"], float), 0, 1)
    phi= np.clip(np.asarray(df["defect_frac"], float), 0, 1)
    t0 = np.asarray(df["t0_ms"], float)
    dT = 600.0 - t0
    dT = np.clip(dT, 1.0, None)

    if use_penalties:
        alphas = [0.0, 0.5, 1.0, 1.5]
        betas  = [0.0, 0.5, 1.0, 1.5]
        best = None
        for a in alphas:
            for b in betas:
                f = np.exp(-a*(1.0 - C)) * np.exp(-b*phi)
                v0s = (R / dT) / (f + 1e-9)
                v0 = np.median(v0s[np.isfinite(v0s) & (v0s>0)])
                Rp = v0 * f * dT
                err = np.nanmean((Rp - R)**2)
                if (best is None) or (err < best[0]):
                    best = (err, v0, a, b)
        _, v0, a_opt, b_opt = best
        return v0, a_opt, b_opt
    else:
        v0 = np.median(R / dT)
        return v0, 0.0, 0.0

def build_growth_curves(df, v0, alpha, beta):
    A = np.asarray(df["area_px"], float)
    R = np.sqrt(A/np.pi)
    C  = np.clip(np.asarray(df["circularity"], float), 0, 1)
    phi= np.clip(np.asarray(df["defect_frac"], float), 0, 1)
    t0 = np.asarray(df["t0_ms"], float)

    f = np.exp(-alpha*(1.0 - C)) * np.exp(-beta*phi)
    t = np.arange(0, 601, 1, dtype=float)
    A_sum = np.zeros_like(t, dtype=float)
    for Ri, t0i, fi in zip(R, t0, f):
        ri = np.maximum(0.0, (t - t0i)) * (v0 * fi)
        Ai = np.pi * (ri**2)
        A_sum += Ai
    A_final = A.sum()
    denom = max(A_final, 1.0)
    X_pred = np.clip(A_sum / denom, 0.0, 1.0)
    return t, X_pred

def fit_avrami_K(t_ms, X_pred, n_fixed):
    def loss(K):
        y = 1.0 - np.exp(-K*(t_ms**n_fixed))
        return np.nanmean((y - X_pred)**2)
    res = optimize.minimize(lambda z: loss(z[0]), x0=np.array([1e-3]), bounds=[(1e-9, 1e3)])
    return float(res.x[0])

# -------------------------
# Plots & CSVs
# -------------------------
def plot_dn_dt_compare(resA, labelA, resB, labelB, outpng, bandA=None, bandB=None):
    plt.figure()
    if bandA:
        plt.fill_between(bandA["grid"], bandA["lo"], bandA["hi"], alpha=0.2, label=f"{labelA} 95% CI")
    if bandB:
        plt.fill_between(bandB["grid"], bandB["lo"], bandB["hi"], alpha=0.2, label=f"{labelB} 95% CI")
    plt.plot(resA["grid_t"], resA["pdf"], label=f"{labelA} dn/dt ({resA['model']})")
    plt.plot(resB["grid_t"], resB["pdf"], label=f"{labelB} dn/dt ({resB['model']})")
    plt.xlabel("t (ms)")
    plt.ylabel("dn/dt (a.u.)")
    plt.title("Nucleation rate density dn/dt (0–60 ms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpng, dpi=200)
    plt.close()

def plot_X_compare(t, X_A, X_B, n, K_A, K_B, labelA, labelB, outpng):
    plt.figure()
    plt.plot(t, X_A, label=f"X_pred {labelA}")
    plt.plot(t, 1.0 - np.exp(-K_A*(t**n)), "--", label=f"Avrami {labelA} (n={n:.2f}, K={K_A:.3g})")
    plt.plot(t, X_B, label=f"X_pred {labelB}")
    plt.plot(t, 1.0 - np.exp(-K_B*(t**n)), "--", label=f"Avrami {labelB} (n={n:.2f}, K={K_B:.3g})")
    plt.xlabel("t (ms)")
    plt.ylabel("X(t) (fraction)")
    plt.ylim(0, 1.05)
    plt.title("X(t) with Avrami overlays")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpng, dpi=200)
    plt.close()

def bootstrap_dn_dt(t0_ms, n_boot=0, grid=np.arange(0,61,1,dtype=float)):
    if n_boot <= 0:
        return None
    rng = np.random.default_rng(42)
    t0 = np.asarray(t0_ms, float)
    t0 = t0[(t0>0)&(t0<=60)&np.isfinite(t0)]
    if len(t0) < 10:
        return None
    curves = []
    for _ in range(n_boot):
        tb = rng.choice(t0, size=len(t0), replace=True)
        res = pick_I_model(tb)
        if res is None:
            continue
        curves.append(res["pdf"])
    if not curves:
        return None
    C = np.vstack(curves)
    lo = np.percentile(C, 2.5, axis=0)
    hi = np.percentile(C, 97.5, axis=0)
    return {"grid": grid, "lo": lo, "hi": hi}

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Compare nucleation/growth between two COCO JSON folders.")
    ap.add_argument("--folders", nargs=2, required=True, help="Two folders with COCO JSONs.")
    ap.add_argument("--labels",  nargs=2, required=True, help="Labels for the two datasets (e.g., FAPI FAPI-TEMPO).")
    ap.add_argument("--out", default=None, help="Output folder (optional).")
    ap.add_argument("--n_avrami", type=float, default=2.5, help="Fixed Avrami exponent n.")
    ap.add_argument("--bootstrap", type=int, default=0, help="Bootstrap reps for dn/dt bands (0=off).")
    ap.add_argument("--calibrate_penalties", action="store_true", help="Grid search alpha,beta (else alpha=beta=0).")
    args = ap.parse_args()

    folderA = Path(args.folders[0]).resolve()
    folderB = Path(args.folders[1]).resolve()
    labelA, labelB = args.labels

    # Default output folder if not provided
    if args.out is None or len(args.out.strip()) == 0:
        outdir = folderA.parent / "comparative_outputs_FAPI_vs_TEMPO"
    else:
        outdir = Path(args.out).resolve()
    safe_mkdir(outdir)

    # Load
    dfA = load_coco_many(folderA, labelA)
    dfB = load_coco_many(folderB, labelB)

    if dfA.empty or dfB.empty:
        print("One or both datasets are empty. Aborting comparison after logging.")
        return

    # Nucleation times via rank->time (0..60 ms)
    dfA = dfA.copy(); dfB = dfB.copy()
    dfA["t0_ms"] = rank_to_t0_ms(dfA["area_px"].to_numpy())
    dfB["t0_ms"] = rank_to_t0_ms(dfB["area_px"].to_numpy())

    # Fit dn/dt I(t)
    resA = pick_I_model(dfA["t0_ms"].to_numpy())
    resB = pick_I_model(dfB["t0_ms"].to_numpy())
    if (resA is None) or (resB is None):
        print("[WARN] Could not fit dn/dt in one of the datasets.")
        return

    # Bootstrap bands optional
    bandA = bootstrap_dn_dt(dfA["t0_ms"].to_numpy(), n_boot=args.bootstrap)
    bandB = bootstrap_dn_dt(dfB["t0_ms"].to_numpy(), n_boot=args.bootstrap)

    # Save dn/dt CSV
    pd.DataFrame({
        "t_ms": resA["grid_t"],
        f"dn_dt_{labelA}": resA["pdf"],
        f"dn_dt_{labelB}": resB["pdf"],
    }).to_csv(outdir / "dn_dt_compare.csv", index=False)

    # Plot dn/dt
    plot_dn_dt_compare(resA, labelA, resB, labelB, outpng=str(outdir / "dn_dt_compare.png"),
                       bandA=bandA, bandB=bandB)

    # Growth calibration (v0, alpha, beta)
    v0A, alphaA, betaA = calibrate_v0_alpha_beta(dfA, use_penalties=args.calibrate_penalties)
    v0B, alphaB, betaB = calibrate_v0_alpha_beta(dfB, use_penalties=args.calibrate_penalties)

    # Build X(t) and fit Avrami K
    t, X_A = build_growth_curves(dfA, v0A, alphaA, betaA)
    _, X_B = build_growth_curves(dfB, v0B, alphaB, betaB)
    K_A = fit_avrami_K(t, X_A, args.n_avrami)
    K_B = fit_avrami_K(t, X_B, args.n_avrami)

    # Save X overlays CSV
    pd.DataFrame({
        "t_ms": t,
        f"X_pred_{labelA}": X_A,
        f"X_pred_{labelB}": X_B,
        f"X_Avrami_{labelA}": 1.0 - np.exp(-K_A*(t**args.n_avrami)),
        f"X_Avrami_{labelB}": 1.0 - np.exp(-K_B*(t**args.n_avrami)),
    }).to_csv(outdir / "X_overlays.csv", index=False)

    # Plot X overlays
    plot_X_compare(t, X_A, X_B, args.n_avrami, K_A, K_B, labelA, labelB,
                   outpng=str(outdir / "X_overlays.png"))

    # Save parameters summary
    rows = []
    def params_row(label, res, v0, a, b, K):
        d = {"dataset": label, "v0_px_per_ms": v0, "alpha": a, "beta": b, "K": K, "I_model": res["model"]}
        if res["model"] == "gamma":
            d.update({"k": res["k"], "theta": res["theta"], "AIC": res["AIC"], "BIC": res["BIC"]})
        else:
            d.update({"mu": res["mu"], "sigma": res["sigma"], "AIC": res["AIC"], "BIC": res["BIC"]})
        return d
    rows.append(params_row(labelA, resA, v0A, alphaA, betaA, K_A))
    rows.append(params_row(labelB, resB, v0B, alphaB, betaB, K_B))
    pd.DataFrame(rows).to_csv(outdir / "fit_parameters_summary.csv", index=False)

    # Export per-object tables
    keep_cols = ["dataset","file","image_id","area_px","perimeter_px","circularity",
                 "defect_frac","centroid_x","centroid_y","nuc_x","nuc_y","nuc_dist_px","t0_ms"]
    dfA[keep_cols].to_csv(outdir / f"objects_{labelA}.csv", index=False)
    dfB[keep_cols].to_csv(outdir / f"objects_{labelB}.csv", index=False)

    print("=== Done ===")
    print(f"Output folder: {outdir}")
    print(f"dn/dt CSV + plot   : {outdir/'dn_dt_compare.csv'}, {outdir/'dn_dt_compare.png'}")
    print(f"X overlays CSV+plot: {outdir/'X_overlays.csv'}, {outdir/'X_overlays.png'}")
    print(f"Params summary     : {outdir/'fit_parameters_summary.csv'}")
    print(f"Objects (A,B)      : {outdir/f'objects_{labelA}.csv'}, {outdir/f'objects_{labelB}.csv'}")

if __name__ == "__main__":
    main()
