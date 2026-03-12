#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Decoupled nucleation & growth analysis for FAPI (COCO JSONs that may be either:
- full COCO dicts with "images"/"annotations", or
- plain lists of annotation dicts (no "images" key))

category_id: 1=crystals, 2=nucleus, 3=defects

Outputs (./fapi_outputs/):
  - nucleation_times.csv, nucleation_I_t.csv, nucleation_fit_params.csv
  - per_object_timeseries.csv
  - X_t_observed_vs_avrami.csv
  - Plots: I_t.png, growth_overlays.png, avrami_overlay.png
"""

import json
import math
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import gammaln

# ------------------------- CONFIG -------------------------
JSON_FILES = [
    "FAPI_0.json",
    "FAPI_0 1.json",
    "FAPI_0 2.json",
    "FAPI_0 3.json",
]
INPUT_DIR = "."                 # folder with the JSONs
OUT_DIR = "./fapi_outputs"

# Timing assumptions
FRAME_DT_MS = 10.0              # ms per frame (edit if your fps differs)
TOTAL_SOLID_MS = 600.0          # solidification horizon
NUCLEATION_WINDOW_MS = 60.0     # nucleation window

# Pixel-size (optional). If None, keep areas in pixels.
PIXEL_SIZE_UM = None            # e.g., 0.5  (um per pixel). None -> pixels

# Avrami exponent n (fix; we fit K only)
AVRAMI_N = 2.5

# Threshold for first appearance (as % of per-object max within nucleation window)
APPEAR_FRAC = 0.05

# ---------------------------------------------------------

def ensure_outdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

# --------- COCO RLE decoding (pure-Python fallback) ---------
def _rle_decode_counts(counts):
    # If it's already a list (uncompressed), return it.
    if isinstance(counts, list):
        return counts

    # Otherwise treat as compressed COCO RLE string.
    # Adapted from COCO API encoding notes.
    arr = []
    p = 0
    m = 0
    s = counts
    while p < len(s):
        x = 0
        k = 0
        more = True
        while more:
            c = ord(s[p]) - 48
            p += 1
            x |= (c & 0x1f) << (5 * k)
            more = (c & 0x20) != 0
            k += 1
            if not more and (c & 0x10):
                m = -1
        if m == -1:
            x = -x
        arr.append(x)
        m = 0
    return arr

def rle_to_mask(rle, h, w):
    counts = _rle_decode_counts(rle)
    mask = np.zeros(h * w, dtype=np.uint8)
    idx = 0
    val = 0
    for c in counts:
        if c < 0:
            raise ValueError("Negative run-length in RLE.")
        if val == 1:
            mask[idx:idx + c] = 1
        idx += c
        val ^= 1
    return mask.reshape((h, w), order='F')  # COCO uses column-major

# --------- Robust loader (dict OR list) ---------
def load_coco_jsons(files):
    """
    Returns:
      images: dict[image_id] -> {"id":..., "height":H, "width":W}
      anns:   list of annotation dicts (must include image_id, segmentation.size, segmentation.counts, category_id, id)
    """
    images = {}
    anns = []

    for fp in files:
        with open(fp, "r") as f:
            data = json.load(f)

        if isinstance(data, dict) and ("annotations" in data):
            # Standard COCO dict
            for im in data.get("images", []):
                images[im["id"]] = {"id": im["id"], "height": im["height"], "width": im["width"]}
            anns.extend(data.get("annotations", []))

        elif isinstance(data, list):
            # Plain list of annotations
            anns.extend(data)
        else:
            # Unknown structure -> try to coerce
            raise ValueError(f"Unsupported JSON structure in {fp}")

    # If no images dict was provided, infer from annotations via segmentation.size
    if not images:
        inferred = {}
        for a in anns:
            im_id = a.get("image_id")
            seg = a.get("segmentation", {})
            size = seg.get("size", None)
            if im_id is None or size is None or len(size) != 2:
                # We need size to decode masks; fail clearly
                raise ValueError("Annotation missing 'image_id' or 'segmentation.size=[h,w]' needed to infer image metadata.")
            h, w = int(size[0]), int(size[1])
            if im_id not in inferred:
                inferred[im_id] = {"id": im_id, "height": h, "width": w}
            else:
                # sanity: if multiple anns disagree on size, keep the first
                pass
        images = inferred

    return images, anns

def area_perimeter_from_mask(mask, pixel_size_um=None):
    A = float(mask.sum())
    # Perimeter via 4-neighborhood edge count (fast proxy)
    border = 0
    border += np.sum(mask ^ np.pad(mask[1:, :], ((0,1),(0,0)), constant_values=0))
    border += np.sum(mask ^ np.pad(mask[:-1, :], ((1,0),(0,0)), constant_values=0))
    border += np.sum(mask ^ np.pad(mask[:, 1:], ((0,0),(0,1)), constant_values=0))
    border += np.sum(mask ^ np.pad(mask[:, :-1], ((0,0),(1,0)), constant_values=0))
    P = float(border) / 2.0
    if pixel_size_um is not None:
        A = A * (pixel_size_um ** 2)
        P = P * pixel_size_um
    return A, P

def circularity(A, P, eps=1e-9):
    return 4.0 * math.pi * A / (P * P + eps)

# -------------------------- NUCLEATION FITS ---------------------------
def lognormal_pdf(t, m, s):
    return (1.0 / (t * s * np.sqrt(2*np.pi))) * np.exp(-(np.log(t) - m)**2 / (2*s*s))

def gamma_pdf(t, k, theta):
    return np.exp((k-1)*np.log(t) - t/theta - k*np.log(theta) - gammaln(k))

def fit_lognormal_mle(times):
    y = np.log(times)
    m = np.mean(y)
    s = np.std(y, ddof=1)
    s = max(s, 1e-6)
    return {"model":"lognormal","m":m,"s":s}

def fit_gamma_mle(times):
    t = np.array(times)
    mean = t.mean()
    var = t.var(ddof=1)
    k0 = max(mean*mean/var, 1e-3)
    theta0 = max(var/mean, 1e-6)

    def nll(params):
        k, theta = params
        if k <= 0 or theta <= 0: return np.inf
        return -( (k-1)*np.log(t) - t/theta - k*np.log(theta) - gammaln(k) ).sum()

    res = minimize(nll, x0=[k0, theta0], method="Nelder-Mead")
    k, theta = res.x
    return {"model":"gamma","k":max(k,1e-6),"theta":max(theta,1e-6)}

def aic_bic(times, fit):
    t = np.array(times)
    if fit["model"]=="lognormal":
        m, s = fit["m"], fit["s"]
        ll = np.log(lognormal_pdf(t, m, s)+1e-300).sum()
        k = 2
    else:
        k, th = fit["k"], fit["theta"]
        ll = np.log(gamma_pdf(t, k, th)+1e-300).sum()
        k = 2
    n = len(t)
    AIC = 2*k - 2*ll
    BIC = k*np.log(n) - 2*ll
    return AIC, BIC

# ------------------------ MAIN PIPELINE -------------------------------
def main():
    ensure_outdir(OUT_DIR)

    # Resolve paths
    json_paths = [str(Path(INPUT_DIR)/fn) for fn in JSON_FILES if Path(INPUT_DIR, fn).exists()]
    if not json_paths:
        raise FileNotFoundError("No FAPI JSON files found in INPUT_DIR.")

    images, anns = load_coco_jsons(json_paths)

    # Build per-frame lists for each category
    # Sort images by ID as time order (works if image_id is monotonically increasing with frame index)
    image_ids_sorted = sorted(images.keys())
    id_to_frame = {im_id:i for i,im_id in enumerate(image_ids_sorted)}
    # Use size from any ann to set h,w if images absent; otherwise take from images dict
    any_im = image_ids_sorted[0]
    h = images[any_im]["height"]
    w = images[any_im]["width"]

    # Field-of-view (for bulk fraction)
    if PIXEL_SIZE_UM is None:
        A_FOV = float(h*w)  # pixels
    else:
        A_FOV = float(h*w) * (PIXEL_SIZE_UM**2)  # μm²

    # Split annotations by category
    by_cat = defaultdict(list)
    for a in anns:
        by_cat[a.get("category_id")].append(a)

    nuclei = by_cat.get(2, [])
    crystals = by_cat.get(1, [])
    defects = by_cat.get(3, [])

    # Decoder for each annotation
    def decode_ann(ann):
        seg = ann["segmentation"]
        size = seg["size"]  # [h,w]
        rle = seg["counts"]
        m = rle_to_mask(rle, size[0], size[1])
        return m

    # Group annotations per (frame -> list of (id, mask, cat))
    frames = defaultdict(list)
    for a in anns:
        im_id = a["image_id"]
        m = decode_ann(a)
        frames[im_id].append((a["id"], m, a["category_id"]))

    # Times for each frame
    frame_times_ms = {im_id: id_to_frame[im_id]*FRAME_DT_MS for im_id in image_ids_sorted}

    # ---------- NUCLEATION: appearance times from nuclei masks ----------
    nucleus_first_time = []
    seen_nuclei = set()
    for im_id in image_ids_sorted:
        t = frame_times_ms[im_id]
        if t > NUCLEATION_WINDOW_MS:
            break
        for (aid, mask, cat) in frames[im_id]:
            if cat != 2: 
                continue
            if aid in seen_nuclei:
                continue
            if mask.sum() > 0:
                nucleus_first_time.append(t)
                seen_nuclei.add(aid)

    nuc_df = pd.DataFrame({"t_ms": nucleus_first_time})
    nuc_df.to_csv(Path(OUT_DIR, "nucleation_times.csv"), index=False)

    # Fit I(t) on 1-ms grid in 0–60 ms
    t_grid = np.arange(1.0, NUCLEATION_WINDOW_MS+1.0, 1.0)
    if len(nucleus_first_time) >= 5:
        # Lognormal
        ln_fit = fit_lognormal_mle(np.array(nucleus_first_time)+1e-6)
        ln_AIC, ln_BIC = aic_bic(np.array(nucleus_first_time)+1e-6, ln_fit)
        I_logn = lognormal_pdf(t_grid, ln_fit["m"], ln_fit["s"])

        # Gamma
        gm_fit = fit_gamma_mle(np.array(nucleus_first_time)+1e-6)
        gm_AIC, gm_BIC = aic_bic(np.array(nucleus_first_time)+1e-6, gm_fit)
        I_gamm = gamma_pdf(t_grid, gm_fit["k"], gm_fit["theta"])

        # Select by BIC (AIC tiebreaker)
        sel = "lognormal"
        if gm_BIC < ln_BIC - 1e-9 or (abs(gm_BIC - ln_BIC) < 1e-9 and gm_AIC < ln_AIC):
            sel = "gamma"

        I_df = pd.DataFrame({
            "t_ms": t_grid,
            "I_lognormal": I_logn,
            "I_gamma": I_gamm,
            "I_selected": I_logn if sel=="lognormal" else I_gamm
        })
        I_df.to_csv(Path(OUT_DIR, "nucleation_I_t.csv"), index=False)

        pd.DataFrame([
            {"model":"lognormal", **ln_fit, "AIC":ln_AIC, "BIC":ln_BIC},
            {"model":"gamma", **gm_fit, "AIC":gm_AIC, "BIC":gm_BIC},
            {"selected": sel}
        ]).to_csv(Path(OUT_DIR, "nucleation_fit_params.csv"), index=False)
        # Plot I(t)
        plt.figure(figsize=(7,4))
        plt.plot(t_grid, I_logn, label="Lognormal")
        plt.plot(t_grid, I_gamm, label="Gamma")
        plt.plot(t_grid, I_df["I_selected"], "--", label=f"Selected ({sel})")
        plt.xlabel("Time (ms)")
        plt.ylabel("Nucleation rate density I(t) (arb.)")
        plt.title("FAPI nucleation I(t)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(OUT_DIR, "I_t.png"), dpi=200)
        plt.close()
    else:
        with open(Path(OUT_DIR, "nucleation_fit_params.csv"), "w") as f:
            f.write("Not enough nucleation samples for fitting (need >=5).\n")

    # ---------- GROWTH: per-object trajectories (category 1) ----------
    # Precompute union-of-defects per frame
    defect_masks_by_frame = {}
    for im_id in image_ids_sorted:
        masks = [m for (_aid,m,cat) in frames[im_id] if cat==3]
        if masks:
            acc = np.zeros((h,w), dtype=bool)
            for m in masks:
                acc |= (m>0)
            defect_masks_by_frame[im_id] = acc
        else:
            defect_masks_by_frame[im_id] = np.zeros((h,w), dtype=bool)

    per_rows = []
    areas_tmp = defaultdict(list)   # id -> list of (t_ms, area)
    perims_tmp = defaultdict(list)  # id -> list of (t_ms, perim)
    defects_tmp = defaultdict(list) # id -> list of (t_ms, defect area overlap)

    crystal_ids = set()
    for im_id in image_ids_sorted:
        t = frame_times_ms[im_id]
        dmask = defect_masks_by_frame[im_id]
        for (aid, mask, cat) in frames[im_id]:
            if cat != 1:
                continue
            crystal_ids.add(aid)
            A, P = area_perimeter_from_mask(mask, pixel_size_um=PIXEL_SIZE_UM)
            dA = float((mask.astype(bool) & dmask).sum())
            if PIXEL_SIZE_UM is not None:
                dA *= (PIXEL_SIZE_UM**2)
            areas_tmp[aid].append((t, A))
            perims_tmp[aid].append((t, P))
            defects_tmp[aid].append((t, dA))

    for aid in crystal_ids:
        ts = np.array([x[0] for x in areas_tmp[aid]])
        As = np.array([x[1] for x in areas_tmp[aid]])
        Ps = np.array([x[1] for x in perims_tmp[aid]])
        Ds = np.array([x[1] for x in defects_tmp[aid]])

        idx = np.argsort(ts)
        ts = ts[idx]; As = As[idx]; Ps = Ps[idx]; Ds = Ds[idx]

        Cs = np.array([circularity(a, p) for a,p in zip(As, Ps)])
        phi = Ds / (As + 1e-9)

        # appearance time: first t with area >= 5% of local max (within window if present)
        win = ts <= NUCLEATION_WINDOW_MS
        Amax_win = As[win].max() if win.any() else 0.0
        thresh = APPEAR_FRAC * max(Amax_win, As.max())
        appear_idx = np.where(As >= max(thresh, 1e-9))[0]
        t0 = ts[appear_idx[0]] if appear_idx.size>0 else ts[0]

        for t, a, c, dfrac in zip(ts, As, Cs, phi):
            per_rows.append({
                "object_id": aid,
                "t_ms": t,
                "area": a,
                "circularity": c,
                "defect_fraction": dfrac,
                "t0_ms": t0
            })

    per_df = pd.DataFrame(per_rows)
    per_df.to_csv(Path(OUT_DIR, "per_object_timeseries.csv"), index=False)

    # ---------- Bulk transformed fraction X(t) + Avrami ----------
    times_obs = []
    X_obs = []
    for im_id in image_ids_sorted:
        t = id_to_frame[im_id]*FRAME_DT_MS
        if t > TOTAL_SOLID_MS:
            continue
        A_sum = 0.0
        for (aid, mask, cat) in frames[im_id]:
            if cat != 1: continue
            A_sum += float(mask.sum())
        if PIXEL_SIZE_UM is not None:
            A_sum *= (PIXEL_SIZE_UM**2)
        times_obs.append(t)
        X_obs.append(A_sum / A_FOV)

    t_arr = np.array(times_obs)/1000.0
    X_arr = np.clip(np.array(X_obs), 0, 0.999999)

    def avrami_loss(K):
        return np.sum(( (1.0 - np.exp(-(K*(t_arr**AVRAMI_N)))) - X_arr )**2)

    K_grid = np.logspace(-6, 6, 121)
    losses = [avrami_loss(K) for K in K_grid]
    K_fit = K_grid[int(np.argmin(losses))]
    X_fit = 1.0 - np.exp(-(K_fit*(t_arr**AVRAMI_N)))

    avrami_df = pd.DataFrame({
        "t_ms": times_obs,
        "X_obs": X_arr,
        "X_avrami": X_fit
    })
    avrami_df.to_csv(Path(OUT_DIR, "X_t_observed_vs_avrami.csv"), index=False)

    # ---------- Plots ----------
    # I(t)
    if Path(OUT_DIR, "nucleation_I_t.csv").exists():
        I = pd.read_csv(Path(OUT_DIR, "nucleation_I_t.csv"))
        plt.figure(figsize=(7,4))
        if "I_lognormal" in I: plt.plot(I["t_ms"], I["I_lognormal"], label="Lognormal")
        if "I_gamma" in I:     plt.plot(I["t_ms"], I["I_gamma"], label="Gamma")
        if "I_selected" in I:  plt.plot(I["t_ms"], I["I_selected"], "--", label="Selected")
        plt.xlabel("Time (ms)"); plt.ylabel("I(t) (arb.)")
        plt.title("FAPI nucleation I(t)")
        plt.legend(); plt.tight_layout()
        plt.savefig(Path(OUT_DIR, "I_t.png"), dpi=200); plt.close()

    # Per-object growth overlays
    plt.figure(figsize=(8,5))
    if not per_df.empty:
        for oid, g in per_df.groupby("object_id"):
            g = g.sort_values("t_ms")
            plt.plot(g["t_ms"], g["area"], alpha=0.25)
    plt.xlabel("Time (ms)")
    plt.ylabel("Area" + (" (µm²)" if PIXEL_SIZE_UM else " (px²)"))
    plt.title("FAPI: per-object area growth")
    plt.tight_layout()
    plt.savefig(Path(OUT_DIR, "growth_overlays.png"), dpi=200)
    plt.close()

    # Avrami overlay
    plt.figure(figsize=(7,4))
    plt.plot(times_obs, X_arr, 'o', ms=3, label="Observed (sum areas / FOV)")
    plt.plot(times_obs, X_fit, '-', label=f"Avrami fit (n={AVRAMI_N:.2f}, K={K_fit:.3g})")
    plt.xlabel("Time (ms)")
    plt.ylabel("Transformed fraction X(t)")
    plt.title("FAPI: Avrami overlay (isothermal)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(OUT_DIR, "avrami_overlay.png"), dpi=200)
    plt.close()

    print(f"Done. Parsed {len(images)} frames, {len(anns)} annotations.")
    print(f"Outputs written to: {OUT_DIR}")

if __name__ == "__main__":
    main()
