#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rank-to-time nucleation + morphology-penalized growth for FAPI (no frames).
category_id: 1=crystal (mask), 2=nucleus (point), 3=defect (mask/attr)

Outputs:
- nucleation_I_t.csv (lognormal/gamma + selected)
- per_object_predicted_area.csv (1-ms grid, 0-600 ms)
- X_t_pred_vs_Avrami.csv (optional, needs A_eff)
- figures: I_t.png, growth_overlays.png, X_vs_Avrami.png

Tune ALPHA, BETA; switch USE_CONVEX_HULL_FOR_AEFF if desired.
"""

import json, math, os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.special import gammaln

# ---------------- CONFIG ----------------
JSON_FILES = ["FAPI_0.json","FAPI_0 1.json","FAPI_0 2.json","FAPI_0 3.json"]
INPUT_DIR = "."
OUT_DIR = "./fapi_ranktime_out"

NUC_WINDOW_MS = 60.0
TOTAL_MS = 600.0
DT_MS = 1.0

# Morphology penalties (constant per object here)
ALPHA = 1.0  # penalty weight for (1 - circularity)
BETA  = 1.0  # penalty weight for defect fraction

# Avrami overlay (optional)
USE_AVRAMI = True
AVRAMI_N = 2.5

# A_eff estimation for bulk X(t)
USE_CONVEX_HULL_FOR_AEFF = True
USER_AEFF = None   # set to a known constant area if you prefer; else None

# -------------- Geometry helpers --------------
def polygon_area(poly):
    def _area(coords):
        xs = np.array(coords[0::2], dtype=float)
        ys = np.array(coords[1::2], dtype=float)
        if xs.size < 3: return 0.0
        xs2 = np.r_[xs, xs[0]]; ys2 = np.r_[ys, ys[0]]
        return 0.5 * abs(np.sum(xs2[:-1]*ys2[1:] - xs2[1:]*ys2[:-1]))
    if isinstance(poly, list) and poly and isinstance(poly[0], list):
        return sum(_area(p) for p in poly if len(p)>=6)
    return _area(poly) if isinstance(poly, list) and len(poly)>=6 else 0.0

def polygon_perimeter(poly):
    def _peri(coords):
        xs = np.array(coords[0::2], dtype=float)
        ys = np.array(coords[1::2], dtype=float)
        if xs.size < 2: return 0.0
        xs2 = np.r_[xs, xs[0]]; ys2 = np.r_[ys, ys[0]]
        dx = np.diff(xs2); dy = np.diff(ys2)
        return float(np.sum(np.sqrt(dx*dx+dy*dy)))
    if isinstance(poly, list) and poly and isinstance(poly[0], list):
        return sum(_peri(p) for p in poly if len(p)>=6)
    return _peri(poly) if isinstance(poly, list) and len(poly)>=6 else 0.0

def circularity(A, P, eps=1e-9):
    return 4.0*math.pi*A/(P*P + eps) if (A>0 and P>0) else 0.0

# COCO RLE utilities (size=[h,w], counts as str or list)
def _rle_decode_counts(counts):
    if isinstance(counts, list):
        return counts
    arr=[]; p=0; m=0; s=counts
    while p < len(s):
        x=0; k=0; more=True
        while more:
            c = ord(s[p]) - 48; p+=1
            x |= (c & 0x1f) << (5*k)
            more = (c & 0x20)!=0; k+=1
            if not more and (c & 0x10): m=-1
        if m==-1: x=-x
        arr.append(x); m=0
    return arr

def rle_to_area(rle, size):
    counts = _rle_decode_counts(rle)
    h, w = int(size[0]), int(size[1])
    mask = np.zeros(h*w, dtype=np.uint8)
    idx=0; val=0
    for c in counts:
        if val==1: mask[idx:idx+c]=1
        idx += c; val ^= 1
    M = mask.reshape((h,w), order='F')
    A = float(M.sum())
    # perimeter is noisy without pix size; skip precision here
    # (we’ll fallback to polygon/bbox for P if available)
    return A

# -------------- JSON load --------------
def load_any_json(paths):
    anns=[]
    for p in paths:
        with open(p,"r") as f:
            data=json.load(f)
        if isinstance(data, dict) and "annotations" in data:
            anns.extend(data["annotations"])
        elif isinstance(data, list):
            anns.extend(data)
        else:
            raise ValueError(f"Unsupported JSON structure in {p}")
    return anns

def area_perimeter_from_ann(a):
    # Prefer polygons; else RLE if size present; else bbox fallback
    seg = a.get("segmentation", None)
    if isinstance(seg, list) and len(seg)>0:
        A = polygon_area(seg); P = polygon_perimeter(seg)
        return A, P, seg
    if isinstance(seg, dict) and "counts" in seg and "size" in seg and seg["size"]:
        A = rle_to_area(seg["counts"], seg["size"])
        # perimeter unknown here without decoding mask edges fully; rough bbox fallback for P:
        bb = a.get("bbox", None)
        if bb:
            _,_,w,h = bb
            P = float(2*w + 2*h)
        else:
            P = 0.0
        return A, P, None
    if "bbox" in a and a["bbox"]:
        x,y,w,h = a["bbox"]
        return float(max(w,0)*max(h,0)), float(2*max(w,0)+2*max(h,0)), None
    return 0.0, 0.0, None

# -------------- Nucleation PDFs --------------
def ln_pdf(t, m, s):
    return (1.0/(t*s*np.sqrt(2*np.pi))) * np.exp(- (np.log(t)-m)**2 / (2*s*s))

def gm_pdf(t, k, th):
    return np.exp((k-1)*np.log(t) - t/th - k*np.log(th) - gammaln(k))

def fit_lognormal(times):
    y=np.log(times); m=float(y.mean()); s=float(y.std(ddof=1)) if len(y)>1 else 1e-6
    return {"model":"lognormal","m":m,"s":max(s,1e-6)}

def fit_gamma(times):
    t=np.array(times,dtype=float)
    mean, var = t.mean(), t.var(ddof=1) if len(t)>1 else (t.mean(), max(1e-6,(t.mean()/2)**2))
    k0=max(mean*mean/var,1e-3); th0=max(var/mean,1e-6)
    def nll(p):
        k,th=p
        if k<=0 or th<=0: return np.inf
        return -( (k-1)*np.log(t) - t/th - k*np.log(th) - gammaln(k) ).sum()
    res=minimize(nll, x0=[k0,th0], method="Nelder-Mead")
    k,th=res.x
    return {"model":"gamma","k":max(k,1e-6),"theta":max(th,1e-6)}

def aic_bic(times, fit):
    t=np.array(times, dtype=float)
    if fit["model"]=="lognormal":
        ll = np.log(ln_pdf(t, fit["m"], fit["s"]) + 1e-300).sum(); k=2
    else:
        ll = np.log(gm_pdf(t, fit["k"], fit["theta"]) + 1e-300).sum(); k=2
    n=len(t); AIC=2*k - 2*ll; BIC=k*np.log(n) - 2*ll
    return AIC, BIC

# -------------- Main --------------
def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    paths=[str(Path(INPUT_DIR,f)) for f in JSON_FILES if Path(INPUT_DIR,f).exists()]
    if not paths: raise FileNotFoundError("No FAPI JSON files found.")

    anns=load_any_json(paths)

    # collect crystals (cat 1) and defects (cat 3, optional)
    crystals=[a for a in anns if a.get("category_id")==1]
    defects =[a for a in anns if a.get("category_id")==3]

    if not crystals:
        raise ValueError("No crystals (category_id=1) found.")

    # Per-crystal features
    rows=[]
    hull_points=[]
    for a in crystals:
        A,P,seg = area_perimeter_from_ann(a)
        C = circularity(A,P)
        # defect fraction proxy (if explicit 'defect_area' is present)
        dfrac = float(a.get("defect_area", 0.0))/(A+1e-9) if A>0 else 0.0
        oid = a.get("id", id(a))
        rows.append({"object_id":oid,"A_final":A,"P":P,"C":C,"defect_frac":dfrac})
        # collect polygon vertices for convex hull (optional)
        if isinstance(seg, list):
            polys = seg if isinstance(seg[0], list) else [seg]
            for p in polys:
                xs = p[0::2]; ys=p[1::2]
                hull_points.extend(zip(xs,ys))
    df=pd.DataFrame(rows).sort_values("A_final").reset_index(drop=True)

    # Rank -> nucleation times in [0, 60] ms
    n=len(df)
    if n==1:
        df["q"]=0.0
    else:
        df["q"]=np.linspace(0.0,1.0,n)
    df["t0_ms"]=NUC_WINDOW_MS*df["q"]

    # Morphology penalties
    f_circ = np.exp(-ALPHA*(1.0 - df["C"].clip(0,1).to_numpy()))
    f_def  = np.exp(-BETA * df["defect_frac"].clip(0,1).to_numpy())
    penalties = f_circ * f_def

    # Final radii & available time
    R = np.sqrt(np.maximum(df["A_final"].to_numpy(),0.0)/math.pi)
    dt = np.maximum(TOTAL_MS - df["t0_ms"].to_numpy(), 1e-6)

    # Base v0 so median object exactly reaches final radius
    v0_candidates = R/(dt*penalties + 1e-9)
    v0 = np.median(v0_candidates)

    # Build per-object curves on 1 ms grid
    t_grid = np.arange(0.0, TOTAL_MS+DT_MS, DT_MS)
    wide = {"t_ms": t_grid}
    for oid, A_final, t0, pen in zip(df["object_id"], df["A_final"], df["t0_ms"], penalties):
        v = v0 * pen
        r = np.maximum(0.0, v*(t_grid - t0))
        A_pred = math.pi * r*r
        # clip to A_final so curves end exactly at snapshot value
        A_pred = np.minimum(A_pred, A_final)
        wide[f"A_obj_{oid}"] = A_pred
    wide_df = pd.DataFrame(wide)
    wide_df.to_csv(Path(OUT_DIR, "per_object_predicted_area.csv"), index=False)

    # Plot per-object overlays
    plt.figure(figsize=(8,5))
    for col in wide_df.columns:
        if col.startswith("A_obj_"):
            plt.plot(wide_df["t_ms"], wide_df[col], alpha=0.25)
    plt.xlabel("Time (ms)"); plt.ylabel("Area (px²)")
    plt.title("FAPI: morphology-penalized growth (rank-time model)")
    plt.tight_layout(); plt.savefig(Path(OUT_DIR,"growth_overlays.png"), dpi=200); plt.close()

    # Nucleation I(t) from rank-based t0
    t0 = df["t0_ms"].to_numpy()
    if (t0>0).sum()>=5:
        t_fit = np.clip(t0, 1e-3, NUC_WINDOW_MS)  # avoid zero for log
        ln = fit_lognormal(t_fit); gm = fit_gamma(t_fit)
        lnA, lnB = aic_bic(t_fit, ln); gmA, gmB = aic_bic(t_fit, gm)
        chosen = "lognormal" if (lnB < gmB or (abs(lnB-gmB)<1e-12 and lnA<=gmA)) else "gamma"
        t_dense = np.arange(1.0, NUC_WINDOW_MS+1.0, 1.0)
        I_ln = ln_pdf(t_dense, ln["m"], ln["s"])
        I_gm = gm_pdf(t_dense, gm["k"], gm["theta"])
        I_sel = I_ln if chosen=="lognormal" else I_gm
        pd.DataFrame({"t_ms":t_dense,"I_lognormal":I_ln,"I_gamma":I_gm,"I_selected":I_sel}).to_csv(
            Path(OUT_DIR,"nucleation_I_t.csv"), index=False)
        pd.DataFrame([
            {"model":"lognormal", **ln, "AIC":lnA, "BIC":lnB},
            {"model":"gamma", **gm, "AIC":gmA, "BIC":gmB},
            {"selected":chosen}
        ]).to_csv(Path(OUT_DIR,"nucleation_fit_params.csv"), index=False)

        plt.figure(figsize=(7,4))
        plt.plot(t_dense, I_ln, label="Lognormal")
        plt.plot(t_dense, I_gm, label="Gamma")
        plt.plot(t_dense, I_sel, "--", label=f"Selected ({chosen})")
        plt.xlabel("Time (ms)"); plt.ylabel("I(t) (arb.)"); plt.title("FAPI nucleation I(t) (rank-based)")
        plt.legend(); plt.tight_layout(); plt.savefig(Path(OUT_DIR,"I_t.png"), dpi=200); plt.close()
    else:
        with open(Path(OUT_DIR,"nucleation_fit_params.csv"),"w") as f:
            f.write("Not enough rank-based nucleation samples (need >=5).\n")

    # Optional bulk X(t)
    if USE_AVRAMI:
        # A_eff via convex hull (or user)
        A_eff = None
        if USER_AEFF is not None:
            A_eff = float(USER_AEFF)
        elif USE_CONVEX_HULL_FOR_AEFF and len(hull_points)>=3:
            pts = np.array(hull_points, dtype=float)
            try:
                hull = ConvexHull(pts)
                A_eff = float(hull.area if hasattr(hull,"area") else 0.0)
                # scipy ConvexHull.area is perimeter in 2D; we need polygon area:
                # rebuild polygon via hull.vertices:
                poly_idx = hull.vertices
                poly_xy = pts[poly_idx]
                x = poly_xy[:,0]; y = poly_xy[:,1]
                x2 = np.r_[x, x[0]]; y2 = np.r_[y, y[0]]
                A_eff = 0.5*abs(np.sum(x2[:-1]*y2[1:] - x2[1:]*y2[:-1]))
            except Exception:
                A_eff = None

        if A_eff and A_eff>0:
            X_pred = wide_df.drop(columns=["t_ms"]).sum(axis=1).to_numpy()/A_eff
            X_pred = np.clip(X_pred, 0.0, 0.999999)
            t_s = wide_df["t_ms"].to_numpy()/1000.0
            # Fit K for fixed n
            def av_loss(K):
                return np.sum(( (1.0 - np.exp(-(K*(t_s**AVRAMI_N)))) - X_pred )**2)
            K_grid = np.logspace(-8, 4, 121)
            losses = [av_loss(K) for K in K_grid]
            K_fit = K_grid[int(np.argmin(losses))]
            X_av = 1.0 - np.exp(-(K_fit*(t_s**AVRAMI_N)))
            pd.DataFrame({"t_ms":wide_df["t_ms"],"X_pred":X_pred,"X_Avrami":X_av}).to_csv(
                Path(OUT_DIR,"X_t_pred_vs_Avrami.csv"), index=False)

            plt.figure(figsize=(7,4))
            plt.plot(wide_df["t_ms"], X_pred, label="Pred (sum areas / A_eff)")
            plt.plot(wide_df["t_ms"], X_av, "--", label=f"Avrami (n={AVRAMI_N}, K={K_fit:.3g})")
            plt.xlabel("Time (ms)"); plt.ylabel("X(t)"); plt.title("FAPI: X(t) vs Avrami (rank-time model)")
            plt.legend(); plt.tight_layout(); plt.savefig(Path(OUT_DIR,"X_vs_Avrami.png"), dpi=200); plt.close()
        else:
            with open(Path(OUT_DIR,"X_t_pred_vs_Avrami.csv"),"w") as f:
                f.write("Skipped bulk X(t): A_eff unavailable (no hull or USER_AEFF not set).\n")

    # Save per-object table (times & factors) for traceability
    df_out = df.copy()
    df_out["R_final"] = np.sqrt(np.maximum(df_out["A_final"],0.0)/math.pi)
    df_out["dt_ms"] = np.maximum(TOTAL_MS - df_out["t0_ms"], 1e-6)
    df_out["penalty"] = penalties
    df_out["v0_median"] = v0
    df_out.to_csv(Path(OUT_DIR,"per_object_summary.csv"), index=False)

    print(f"Done. Objects: {len(df)}  Outputs -> {OUT_DIR}")

if __name__ == "__main__":
    main()
