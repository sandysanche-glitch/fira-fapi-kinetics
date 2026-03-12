#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two COCO-style datasets (e.g., FAPI vs FAPI-TEMPO):
 - robust category detection with overrides
 - segmentation heuristic to treat unknown categories with masks as crystals
 - rank-to-time nucleation over 0–60 ms
 - growth with circularity/defect penalties (alpha, beta)
 - v0 calibration so median crystal reaches final size by 600 ms
 - I(t) (gamma vs lognormal) via method-of-moments + AIC/BIC
 - Avrami overlay: X(t)=1-exp[-K t^n] with fixed n (user) and fitted K
 - Exports CSVs and comparison plots for both datasets
"""

import os, sys, json, math, glob, argparse, warnings, textwrap, itertools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional decoders (used if present)
try:
    from pycocotools import mask as maskUtils
    HAVE_COCO = True
except Exception:
    HAVE_COCO = False

try:
    from skimage.morphology import binary_erosion
    HAVE_SKIMAGE = True
except Exception:
    HAVE_SKIMAGE = False

# -------------------------- small geometry helpers --------------------------

def _polygon_area_and_perimeter(seg):
    """Return (area, perimeter) from COCO polygon list-of-lists."""
    if not isinstance(seg, list) or len(seg) == 0:
        return (np.nan, np.nan)
    A = 0.0
    P = 0.0
    for poly in seg:
        if len(poly) < 6:
            continue
        pts = np.asarray(poly, dtype=float).reshape(-1, 2)
        x, y = pts[:,0], pts[:,1]
        # Shoelace area
        A += 0.5 * abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))
        # Perimeter
        P += np.sum(np.sqrt(np.sum(np.diff(np.vstack([pts, pts[0]]), axis=0)**2, axis=1)))
    if A <= 0:
        return (np.nan, np.nan)
    return (float(A), float(P) if P>0 else np.nan)

def _bbox_area_and_perimeter(bbox):
    """bbox=[x,y,w,h] -> (area, perimeter approx)"""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return (np.nan, np.nan)
    w = max(float(bbox[2]), 0.0)
    h = max(float(bbox[3]), 0.0)
    A = w*h
    if A <= 0:
        return (np.nan, np.nan)
    P = 2.0*(w+h) if w>0 and h>0 else np.nan
    return (A, P)

def _rle_area_and_perimeter(seg_rle):
    """
    Decode RLE to mask and estimate area/perimeter if pycocotools is present.
    Otherwise return (nan, nan).
    """
    if not HAVE_COCO:
        return (np.nan, np.nan)
    try:
        rle = seg_rle
        if isinstance(seg_rle, dict) and 'counts' in seg_rle and isinstance(seg_rle['counts'], list):
            rle = maskUtils.frPyObjects([seg_rle], seg_rle['size'][0], seg_rle['size'][1])[0]
        m = maskUtils.decode(rle)  # HxW uint8
        A = float(m.sum())
        if A <= 0:
            return (np.nan, np.nan)
        if HAVE_SKIMAGE:
            edges = m ^ binary_erosion(m)
            P = float(edges.sum())
        else:
            r = math.sqrt(A/np.pi)
            P = float(2*np.pi*r)
        return (A, P)
    except Exception:
        return (np.nan, np.nan)

def _centroid_from_segmentation(seg, bbox=None):
    """Compute centroid from polygon or RLE (if possible), else from bbox center."""
    # polygon
    if isinstance(seg, list) and len(seg) > 0:
        xs, ys, A = [], [], 0.0
        cx = cy = 0.0
        for poly in seg:
            pts = np.asarray(poly, dtype=float).reshape(-1,2)
            x, y = pts[:,0], pts[:,1]
            cross = x*np.roll(y,-1) - y*np.roll(x,-1)
            polyA = 0.5*cross.sum()
            if abs(polyA) < 1e-12:
                continue
            A += polyA
            cx += ((x + np.roll(x,-1))*cross).sum()
            cy += ((y + np.roll(y,-1))*cross).sum()
        if abs(A) > 1e-12:
            cx = cx/(6*A)
            cy = cy/(6*A)
            return (float(cx), float(cy))
    # RLE
    if isinstance(seg, dict) and 'counts' in seg and HAVE_COCO:
        try:
            m = maskUtils.decode(seg)
            ys, xs = np.nonzero(m)
            if xs.size > 0:
                return (float(xs.mean()), float(ys.mean()))
        except Exception:
            pass
    # bbox fallback
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x, y, w, h = bbox
        return (float(x + w/2.0), float(y + h/2.0))
    return (np.nan, np.nan)

# ----------------------- category detection / overrides ----------------------

def _detect_category_ids(data, override):
    """
    Returns dict with lists {crystal:[...], nucleus:[...], defect:[...]}.
    'override' may provide explicit lists; if present, they take precedence.
    Otherwise, try name-based guesses; if no categories table or no matches, return empty lists.
    """
    if override and any(override.get(k) for k in ("crystal","nucleus","defect")):
        return {
            "crystal": override.get("crystal") or [],
            "nucleus": override.get("nucleus") or [],
            "defect" : override.get("defect")  or [],
        }

    cats = []
    if isinstance(data, dict):
        cats = data.get("categories", [])
    id2name = {c.get("id"): str(c.get("name","")).lower() for c in cats}
    def like(nm, keys): return any(k in nm for k in keys)

    crystal_ids = [cid for cid,nm in id2name.items() if like(nm, ["crystal","grain","spherulite","domain","object","mask"])]
    nucleus_ids = [cid for cid,nm in id2name.items() if like(nm, ["nucleus","nuclei","seed","center","centre","point","kp","keypoint"])]
    defect_ids  = [cid for cid,nm in id2name.items() if like(nm, ["defect","void","crack","hole","inclusion","artifact"])]

    if not crystal_ids and not nucleus_ids and not defect_ids:
        return {"crystal":[], "nucleus":[], "defect":[]}
    return {"crystal": crystal_ids, "nucleus": nucleus_ids, "defect": defect_ids}

# --------------------------- COCO loader (robust) ----------------------------

def load_coco_many(json_files, dataset_label,
                   crystal_ids=None, nucleus_ids=None, defect_ids=None,
                   seg_as_crystals=False):
    """
    Returns (DataFrame, log) with rows for crystals only:
      ['dataset','file','obj_id','category_id','area','perimeter',
       'defect_area','circularity','phi','cx','cy','has_nucleus','nearest_nucleus_dx','nearest_nucleus_dy','nearest_nucleus_dist']
    """
    rows = []
    skips = []

    for jp in json_files:
        try:
            data = json.load(open(jp, "r", encoding="utf-8"))
        except Exception as e:
            skips.append((str(jp), f"json_error:{e}"))
            continue

        if isinstance(data, dict):
            override = {"crystal": crystal_ids, "nucleus": nucleus_ids, "defect": defect_ids}
            cmap = _detect_category_ids(data, override)
            images = {im.get("id"):im for im in data.get("images", [])}
            anns = data.get("annotations", [])
        elif isinstance(data, list):
            cmap = {"crystal": crystal_ids or [], "nucleus": nucleus_ids or [], "defect": defect_ids or []}
            images = {}
            anns = data
        else:
            skips.append((str(jp), "bad_root"))
            continue

        # Collect nuclei coords
        nuclei_xy = []
        for a in anns:
            cid = a.get("category_id")
            if cid in (cmap["nucleus"] or []):
                # try centroid from seg; else keypoints; else bbox center
                seg = a.get("segmentation")
                bbox = a.get("bbox")
                cx, cy = _centroid_from_segmentation(seg, bbox)
                if (not np.isfinite(cx)) or (not np.isfinite(cy)):
                    kps = a.get("keypoints")
                    if isinstance(kps, list) and len(kps) >= 2:
                        cx, cy = float(kps[0]), float(kps[1])
                if np.isfinite(cx) and np.isfinite(cy):
                    nuclei_xy.append((cx,cy))

        # Collect defect masks (for overlap if we can decode)
        defect_items = []
        for a in anns:
            cid = a.get("category_id")
            if cid in (cmap["defect"] or []):
                defect_items.append(a)

        # Collect crystals (with heuristic)
        crystal_found = 0
        for a in anns:
            cid   = a.get("category_id")
            seg   = a.get("segmentation")
            bbox  = a.get("bbox")
            objid = a.get("id", None)

            # get area/perimeter best we can
            A = P = np.nan

            # polygon
            if isinstance(seg, list) and seg:
                A, P = _polygon_area_and_perimeter(seg)

            # RLE
            if (not np.isfinite(A) or A<=0) and isinstance(seg, dict) and 'counts' in seg:
                Ar, Pr = _rle_area_and_perimeter(seg)
                if np.isfinite(Ar) and Ar>0:
                    A, P = Ar, Pr

            # bbox fallback
            if (not np.isfinite(A) or A<=0) and isinstance(bbox,(list,tuple)):
                Ab, Pb = _bbox_area_and_perimeter(bbox)
                if np.isfinite(Ab) and Ab>0:
                    A, P = Ab, (Pb if np.isfinite(Pb) and Pb>0 else 2*np.pi*math.sqrt(Ab/np.pi))

            # decide type
            is_nucleus = cid in (cmap["nucleus"] or [])
            is_defect  = cid in (cmap["defect"]  or [])
            is_crystal = cid in (cmap["crystal"] or [])

            # segmentation heuristic
            if not is_crystal and not is_nucleus and not is_defect and np.isfinite(A) and A>0:
                is_crystal = True
            if seg_as_crystals and np.isfinite(A) and A>0 and not is_nucleus and not is_defect:
                is_crystal = True

            if not is_crystal:
                continue

            if not (np.isfinite(A) and A>0):
                continue

            # centroid
            cx, cy = _centroid_from_segmentation(seg, bbox)

            # circularity
            circ = np.nan
            if np.isfinite(P) and P>0:
                circ = float(4.0*np.pi*A/(P*P))
                circ = max(0.0, min(1.0, circ))

            # defect overlap (needs masks); fallback 0
            dA = 0.0
            if HAVE_COCO and isinstance(seg, dict) and 'counts' in seg:
                try:
                    m_cr = maskUtils.decode(seg).astype(np.uint8)
                    for d in defect_items:
                        s2 = d.get("segmentation")
                        if isinstance(s2, dict) and 'counts' in s2:
                            m_df = maskUtils.decode(s2).astype(np.uint8)
                            inter = (m_cr & m_df).sum()
                            dA += float(inter)
                except Exception:
                    pass

            # nearest nucleus
            has_nuc = False
            dx = dy = dist = np.nan
            if nuclei_xy and np.isfinite(cx) and np.isfinite(cy):
                pts = np.asarray(nuclei_xy, dtype=float)
                dxy = pts - np.array([cx,cy])
                d2 = np.sum(dxy*dxy, axis=1)
                j = int(np.argmin(d2))
                dx, dy = float(dxy[j,0]), float(dxy[j,1])
                dist   = float(np.sqrt(d2[j]))
                has_nuc = True

            rows.append({
                "dataset": dataset_label,
                "file": str(jp),
                "obj_id": objid,
                "category_id": cid,
                "area": float(A),
                "perimeter": float(P) if np.isfinite(P) else np.nan,
                "defect_area": float(dA),
                "circularity": float(circ) if np.isfinite(circ) else np.nan,
                "phi": float(dA/A) if (A>0) else 0.0,
                "cx": float(cx) if np.isfinite(cx) else np.nan,
                "cy": float(cy) if np.isfinite(cy) else np.nan,
                "has_nucleus": bool(has_nuc),
                "nearest_nucleus_dx": float(dx) if np.isfinite(dx) else np.nan,
                "nearest_nucleus_dy": float(dy) if np.isfinite(dy) else np.nan,
                "nearest_nucleus_dist": float(dist) if np.isfinite(dist) else np.nan,
            })
            crystal_found += 1

        if crystal_found == 0:
            # last resort: promote ANY segmentation (area>0) to crystal unless explicitly nucleus/defect
            for a in anns:
                cid = a.get("category_id")
                if cid in (cmap["nucleus"] or []) or cid in (cmap["defect"] or []):
                    continue
                seg = a.get("segmentation")
                bbox = a.get("bbox")
                A = P = np.nan
                if isinstance(seg, list) and seg:
                    A, P = _polygon_area_and_perimeter(seg)
                if (not np.isfinite(A) or A<=0) and isinstance(seg, dict) and 'counts' in seg:
                    Ar, Pr = _rle_area_and_perimeter(seg)
                    if np.isfinite(Ar) and Ar>0:
                        A, P = Ar, Pr
                if (not np.isfinite(A) or A<=0) and isinstance(bbox,(list,tuple)):
                    Ab, Pb = _bbox_area_and_perimeter(bbox)
                    if np.isfinite(Ab) and Ab>0:
                        A, P = Ab, (Pb if np.isfinite(Pb) and Pb>0 else 2*np.pi*math.sqrt(Ab/np.pi))
                if np.isfinite(A) and A>0:
                    cx, cy = _centroid_from_segmentation(seg, bbox)
                    circ = np.nan
                    if np.isfinite(P) and P>0:
                        circ = float(4.0*np.pi*A/(P*P)); circ = max(0.0, min(1.0, circ))
                    rows.append({
                        "dataset": dataset_label, "file": str(jp), "obj_id": a.get("id"),
                        "category_id": cid, "area": float(A),
                        "perimeter": float(P) if np.isfinite(P) else np.nan,
                        "defect_area": 0.0, "circularity": float(circ) if np.isfinite(circ) else np.nan,
                        "phi": 0.0, "cx": float(cx) if np.isfinite(cx) else np.nan,
                        "cy": float(cy) if np.isfinite(cy) else np.nan, "has_nucleus": False,
                        "nearest_nucleus_dx": np.nan, "nearest_nucleus_dy": np.nan, "nearest_nucleus_dist": np.nan
                    })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["area"]).query("area > 0")
        # Fill missing fields
        if "circularity" not in df.columns: df["circularity"] = np.nan
        if "phi" not in df.columns: df["phi"] = 0.0
        df["phi"] = df["phi"].fillna(0.0).clip(lower=0.0)
        df["circularity"] = df["circularity"].fillna(0.8).clip(lower=0.0, upper=1.0)

    return df, skips

# -------------------- nucleation (rank->time) + I(t) fit ---------------------

def rank_to_t0(df, t_nuc_ms=60.0):
    """Assign nucleation times 0..t_nuc_ms by ascending final area rank."""
    A = df["area"].to_numpy()
    order = np.argsort(A)
    N = len(A)
    q = np.empty(N, dtype=float)
    if N == 1:
        q[order] = 1.0
    elif N > 1:
        ranks = np.arange(N, dtype=float)
        q[order] = ranks/(N-1)
    t0 = q * float(t_nuc_ms)
    df = df.copy()
    df["t0_ms"] = t0
    return df

def fit_lognormal_moments(tms):
    t = np.asarray(tms, dtype=float)
    t = t[(t>0) & np.isfinite(t)]
    if len(t) < 2:
        return None
    m = np.mean(np.log(t)); s2 = np.var(np.log(t), ddof=1)
    if s2 <= 0: s2 = 1e-9
    return {"mu": float(m), "sigma": float(np.sqrt(s2))}

def fit_gamma_moments(tms):
    t = np.asarray(tms, dtype=float)
    t = t[(t>0) & np.isfinite(t)]
    if len(t) < 2: return None
    m = float(np.mean(t)); v = float(np.var(t, ddof=1))
    if m <= 0 or v <= 0: return None
    k = m*m / v
    th = v / m
    return {"k": float(k), "theta": float(th)}

def aic_bic_from_pdf(pdf_vals, t):
    """
    Simple pseudo-likelihood: sum log(pdf(t_i)) where pdf_vals are evaluated at t_i.
    AIC = 2k - 2lnL ; BIC = k ln n - 2lnL . k = #params
    """
    eps = 1e-12
    pdf_vals = np.maximum(pdf_vals, eps)
    lnL = float(np.sum(np.log(pdf_vals)))
    n = len(pdf_vals)
    return lnL, n

def lognormal_pdf(t, mu, sigma):
    t = np.asarray(t, dtype=float)
    y = np.zeros_like(t, dtype=float)
    mask = t > 0
    z = (np.log(t[mask]) - mu)/sigma
    y[mask] = (1.0/(t[mask]*sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*z*z)
    return y

def gamma_pdf(t, k, theta):
    t = np.asarray(t, dtype=float)
    y = np.zeros_like(t, dtype=float)
    mask = t > 0
    # use log form to avoid overflow
    lgamma = math.lgamma(k)
    y[mask] = np.exp((k-1.0)*np.log(t[mask]) - (t[mask]/theta) - lgamma - k*np.log(theta))
    return y

def pick_I_model(t0_ms):
    t = np.asarray(t0_ms, dtype=float)
    t = t[(t>0) & (t<=60.0) & np.isfinite(t)]
    if len(t) < 3:
        return {"model":"none","params":{}, "grid_t": np.arange(0,61,1,dtype=float), "pdf": np.zeros(61)}
    lg = fit_lognormal_moments(t)
    gm = fit_gamma_moments(t)

    grid = np.arange(0, 61, 1, dtype=float)
    # evaluate only at actual samples for likelihood
    if lg is not None:
        pdf_t = lognormal_pdf(t, lg["mu"], lg["sigma"])
        lnL, n = aic_bic_from_pdf(pdf_t, t)
        kpar = 2
        AIC_l = 2*kpar - 2*lnL
        BIC_l = kpar*np.log(n) - 2*lnL
        pdf_grid_l = lognormal_pdf(grid, lg["mu"], lg["sigma"])
    else:
        AIC_l = BIC_l = np.inf; pdf_grid_l = np.zeros_like(grid)

    if gm is not None:
        pdf_t = gamma_pdf(t, gm["k"], gm["theta"])
        lnL, n = aic_bic_from_pdf(pdf_t, t)
        kpar = 2
        AIC_g = 2*kpar - 2*lnL
        BIC_g = kpar*np.log(n) - 2*lnL
        pdf_grid_g = gamma_pdf(grid, gm["k"], gm["theta"])
    else:
        AIC_g = BIC_g = np.inf; pdf_grid_g = np.zeros_like(grid)

    # choose by BIC, AIC tie-break
    if BIC_g < BIC_l or (BIC_g == BIC_l and AIC_g <= AIC_l):
        return {"model":"gamma", "params": gm, "grid_t": grid, "pdf": pdf_grid_g, "AIC":AIC_g, "BIC":BIC_g}
    else:
        return {"model":"lognormal", "params": lg, "grid_t": grid, "pdf": pdf_grid_l, "AIC":AIC_l, "BIC":BIC_l}

# ----------------------- growth + Avrami (with penalties) --------------------

def calibrate_v0(df, alpha, beta, total_ms=600.0):
    """Choose v0 so median crystal reaches its final radius within available time."""
    A = df["area"].to_numpy(float)
    R = np.sqrt(A/np.pi)
    t0 = df["t0_ms"].to_numpy(float)
    dt = np.maximum(0.0, total_ms - t0)
    C  = df["circularity"].to_numpy(float)
    phi= df["phi"].to_numpy(float)

    fc = np.exp(-alpha*(1.0 - C))
    fd = np.exp(-beta*phi)
    denom = dt * fc * fd
    mask = (denom > 0)
    if not np.any(mask):
        return 1.0
    v0_est = R[mask]/denom[mask]
    return float(np.median(v0_est))

def build_A_pred(df, v0, alpha, beta, ms_grid=np.arange(0,601,1,dtype=float)):
    """Per-object area-vs-time predictions using radial growth with penalties."""
    out = []
    for i, row in df.iterrows():
        A = float(row["area"])
        Rf = math.sqrt(A/np.pi)
        t0 = float(row["t0_ms"])
        C  = float(row["circularity"])
        phi= float(row["phi"])
        fc = math.exp(-alpha*(1.0 - C))
        fd = math.exp(-beta*phi)
        v  = v0*fc*fd
        # r(t)=v*(t-t0)+, capped so area does not exceed final area?
        # We allow it to reach exactly Rf at t=600 if v0 calibrated
        r = np.maximum(0.0, v*(ms_grid - t0))
        A_pred = np.pi * np.minimum(r, Rf)**2
        out.append(A_pred)
    M = np.stack(out, axis=0) if out else np.zeros((0,len(ms_grid)))
    return ms_grid, M

def avrami_fit_K(X, t, n):
    """Fit K by LS: minimize sum (X - (1-exp(-K t^n)))^2 over K>0. Simple 1D grid + local refine."""
    t = np.asarray(t, dtype=float)
    X = np.clip(np.asarray(X, dtype=float), 0.0, 0.999999)
    tn = t**n
    # quick closed-ish form: logistic-like; do grid search
    K_grid = np.logspace(-8, 2, 200)
    def err(K): return np.sum((X - (1.0 - np.exp(-K*tn)))**2)
    errs = np.array([err(K) for K in K_grid])
    j = int(np.argmin(errs))
    K0 = K_grid[j]
    # small local refine by scalar steps
    K = K0
    for step in [0.5, 0.2, 0.1, 0.05]:
        for d in [-1, 1]:
            K1 = K*(10**(d*step))
            if K1<=0: continue
            if err(K1) < err(K): K = K1
    return float(K)

# ------------------------------ plotting helpers -----------------------------

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_csv(df, path):
    df.to_csv(path, index=False)

def plot_I_compare(resA, labelA, resB, labelB, outpng):
    plt.figure()
    plt.plot(resA["grid_t"], resA["pdf"], label=f"{labelA} {resA['model']}")
    plt.plot(resB["grid_t"], resB["pdf"], label=f"{labelB} {resB['model']}")
    plt.xlabel("t0 (ms)")
    plt.ylabel("I(t) (a.u., pdf)")
    plt.title("Nucleation-rate density (rank→time)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpng, dpi=200)
    plt.close()

def plot_X_compare(t, X_A, X_B, n, K_A, K_B, outpng):
    plt.figure()
    plt.plot(t, X_A, label="X_pred "+r"("+ "A"+")")
    plt.plot(t, 1.0 - np.exp(-K_A*(t**n)), "--", label=f"Avrami A (n={n:.2f}, K={K_A:.3g})")
    plt.plot(t, X_B, label="X_pred "+r"("+ "B"+")")
    plt.plot(t, 1.0 - np.exp(-K_B*(t**n)), "--", label=f"Avrami B (n={n:.2f}, K={K_B:.3g})")
    plt.xlabel("t (ms)")
    plt.ylabel("X(t) (fraction)")
    plt.ylim(0, 1.05)
    plt.title("X(t) vs Avrami overlays")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpng, dpi=200)
    plt.close()

# ------------------------------- main routine --------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Compare two COCO datasets (FAPI vs FAPI-TEMPO) with nucleation & growth modeling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--folders", nargs="+", required=True,
                    help="Two folders with COCO JSONs (first => label[0], second => label[1]).")
    ap.add_argument("--labels", nargs="+", required=True,
                    help="Two dataset labels, e.g. FAPI FAPI-TEMPO.")
    ap.add_argument("--out", required=True, help="Output folder.")
    ap.add_argument("--n_avrami", type=float, default=2.5, help="Fixed Avrami exponent n.")
    ap.add_argument("--treat_all_segmentations_as_crystals", action="store_true",
                    help="Heuristic: any segmentation with area>0 becomes a crystal unless nucleus/defect.")
    ap.add_argument("--crystal_ids", nargs="*", type=int, default=None,
                    help="Explicit category_id(s) for crystals.")
    ap.add_argument("--nucleus_ids", nargs="*", type=int, default=None,
                    help="Explicit category_id(s) for nuclei.")
    ap.add_argument("--defect_ids", nargs="*", type=int, default=None,
                    help="Explicit category_id(s) for defects.")
    ap.add_argument("--calibrate_penalties", action="store_true",
                    help="Calibrate alpha,beta by small grid to improve X(t) match.")
    args = ap.parse_args()

    if len(args.folders) != 2 or len(args.labels) != 2:
        print("Please pass exactly two folders and two labels.", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.out); ensure_dir(outdir)

    datasets = []
    for folder, label in zip(args.folders, args.labels):
        folder = Path(folder)
        jsons = sorted(glob.glob(str(folder / "**" / "*.json"), recursive=True))
        print(f"Analyzing: {folder}  [{label}]  JSONs found: {len(jsons)}")
        if not jsons:
            print(f"[WARN] Empty folder: {folder}")
            datasets.append((label, pd.DataFrame()))
            continue

        df, skips = load_coco_many(
            jsons, dataset_label=label,
            crystal_ids=args.crystal_ids,
            nucleus_ids=args.nucleus_ids,
            defect_ids=args.defect_ids,
            seg_as_crystals=args.treat_all_segmentations_as_crystals
        )

        if not df.empty:
            n_cr = len(df)
            n_nuc = int(df["has_nucleus"].sum())
            n_def = int((df["defect_area"]>0).sum())
            print(f"[SUMMARY {label}] crystals={n_cr}, nuclei_linked={n_nuc}, defects_overlap>0={n_def}, rows_kept={len(df)}, skips={len(skips)}")
        else:
            print(f"[WARN] Dataset empty after loading/cleaning: {folder}. See skips log if present.")
        datasets.append((label, df))

    # Abort if one/both empty
    (labelA, dfA), (labelB, dfB) = datasets
    if dfA.empty or dfB.empty:
        print("One or both datasets are empty. Aborting comparison after logging.")
        # dump minimal CSVs to help debug
        if not dfA.empty:
            dfA.to_csv(outdir/f"{labelA}_tidy.csv", index=False)
        if not dfB.empty:
            dfB.to_csv(outdir/f"{labelB}_tidy.csv", index=False)
        sys.exit(0)

    # Rank->t0
    dfA = rank_to_t0(dfA, 60.0)
    dfB = rank_to_t0(dfB, 60.0)

    # Calibrate alpha, beta (optional small grid). Keep small to avoid expense.
    alpha_list = [0.0, 0.5, 1.0]
    beta_list  = [0.0, 0.5, 1.0]
    if not args.calibrate_penalties:
        alpha_list = [0.5]
        beta_list  = [0.5]

    def find_best_ab(df, label):
        best = (1e18, 0.0, 0.0, 0.0)  # (err, alpha, beta, K)
        t = np.arange(0, 601, 1, dtype=float)
        # Build an observed X(t) proxy from final areas: we don't have frames; use predicted sums only for fitting K
        # We'll choose alpha,beta,v0 to minimize LS to a self-consistent X_pred then fit K for display.
        for a in alpha_list:
            for b in beta_list:
                v0 = calibrate_v0(df, a, b, total_ms=600.0)
                tt, M = build_A_pred(df, v0, a, b, ms_grid=t)
                X_pred = np.sum(M, axis=0)
                # effective area: convex-hull-like proxy -> sum(final area)*1.2 to avoid >1 issues
                A_eff = max(1.0, float(np.sum(df["area"])) * 1.2)
                X = np.clip(X_pred / A_eff, 0.0, 0.999999)
                K = avrami_fit_K(X, t, args.n_avrami)
                # LS error vs its own Avrami (smallest mismatch → smoother)
                X_av = 1.0 - np.exp(-K*(t**args.n_avrami))
                err = float(np.sum((X - X_av)**2))
                if err < best[0]:
                    best = (err, a, b, K)
        _, a, b, K = best
        return a, b, K

    alphaA, betaA, K_A = find_best_ab(dfA, labelA)
    alphaB, betaB, K_B = find_best_ab(dfB, labelB)

    v0A = calibrate_v0(dfA, alphaA, betaA, total_ms=600.0)
    v0B = calibrate_v0(dfB, alphaB, betaB, total_ms=600.0)

    # Predictions + X(t)
    t = np.arange(0, 601, 1, dtype=float)
    _, MA = build_A_pred(dfA, v0A, alphaA, betaA, ms_grid=t)
    _, MB = build_A_pred(dfB, v0B, alphaB, betaB, ms_grid=t)
    AeffA = max(1.0, float(np.sum(dfA["area"])) * 1.2)
    AeffB = max(1.0, float(np.sum(dfB["area"])) * 1.2)
    XA = np.clip(np.sum(MA, axis=0)/AeffA, 0.0, 0.999999)
    XB = np.clip(np.sum(MB, axis=0)/AeffB, 0.0, 0.999999)
    # Refine K on actual X
    K_A = avrami_fit_K(XA, t, args.n_avrami)
    K_B = avrami_fit_K(XB, t, args.n_avrami)

    # I(t) models
    resA = pick_I_model(dfA["t0_ms"].to_numpy())
    resB = pick_I_model(dfB["t0_ms"].to_numpy())

    # -------------------- exports --------------------
    ensure_dir(outdir)
    dfA.to_csv(outdir/f"{labelA}_tidy.csv", index=False)
    dfB.to_csv(outdir/f"{labelB}_tidy.csv", index=False)

    pd.DataFrame({
        "t_ms": t,
        "X_pred_"+labelA: XA,
        "X_pred_"+labelB: XB,
        "X_Avrami_"+labelA: 1.0 - np.exp(-K_A*(t**args.n_avrami)),
        "X_Avrami_"+labelB: 1.0 - np.exp(-K_B*(t**args.n_avrami)),
    }).to_csv(outdir/"X_overlays.csv", index=False)

    pd.DataFrame({
        "t0_ms": resA["grid_t"],
        f"I_{labelA}": resA["pdf"],
        f"I_{labelB}": resB["pdf"],
    }).to_csv(outdir/"I_t_compare.csv", index=False)

    pd.DataFrame({
        "dataset":[labelA,labelB],
        "alpha":[alphaA, alphaB],
        "beta":[betaA, betaB],
        "v0":[v0A, v0B],
        "Avrami_n":[args.n_avrami, args.n_avrami],
        "Avrami_K":[K_A, K_B],
        "I_model":[resA["model"], resB["model"]],
        "I_params":[json.dumps(resA.get("params",{})), json.dumps(resB.get("params",{}))],
        "I_AIC":[resA.get("AIC",np.nan), resB.get("AIC",np.nan)],
        "I_BIC":[resA.get("BIC",np.nan), resB.get("BIC",np.nan)],
    }).to_csv(outdir/"model_parameters.csv", index=False)

    # plots
    plot_I_compare(resA, labelA, resB, labelB, outpng=str(outdir/"I_t_compare.png"))
    plot_X_compare(t, XA, XB, args.n_avrami, K_A, K_B, outpng=str(outdir/"X_overlays.png"))

    print("=== Done ===")
    print(f"Outputs in: {outdir}")

if __name__ == "__main__":
    main()
