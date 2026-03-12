import os, json, math, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize

# Optional: pycocotools for compressed RLE if present
try:
    import pycocotools.mask as mask_utils
    HAS_PYCOCO = True
except Exception:
    HAS_PYCOCO = False

# ---------- Utilities ----------
def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def as_iter(x):
    return x if isinstance(x, (list, tuple)) else [x]

def name_matches(name, targets_lower):
    if not name: return False
    nl = str(name).strip().lower()
    return any(t in nl for t in targets_lower)

def eq_circle_perimeter_from_area(A):
    # perimeter of a circle with area A
    if A <= 0: return 0.0
    r = math.sqrt(A / math.pi)
    return 2.0 * math.pi * r

def circularity(area, per):
    if area <= 0 or per <= 0:
        return 0.0
    return float(4.0 * math.pi * area / (per ** 2))

def centroid_from_bbox(b):
    # [x,y,w,h]
    return float(b[0] + b[2] / 2.0), float(b[1] + b[3] / 2.0)

# ---------- Mask decoders (polygon & RLE) ----------
def area_perimeter_from_segmentation(seg, size_hint=None):
    """
    Returns (area_px, perimeter_px, decoded_mask_or_None)
    Handles:
      - polygon (list of lists)
      - uncompressed RLE dict {'size':[H,W], 'counts':[...]}
      - compressed RLE dict {'size':[H,W], 'counts':'...'} -> needs pycocotools
    If decoding mask isn’t possible (e.g. missing pycocotools), returns numeric area from seg if available,
    perimeter approximated from equivalent circle, and mask=None.
    """
    # Polygon
    if isinstance(seg, list):
        # polygon area via shoelace; perimeter as polyline length
        total_area = 0.0
        total_per = 0.0
        for poly in seg:
            pts = np.asarray(poly, dtype=float).reshape(-1, 2)
            if len(pts) < 3:
                continue
            x = pts[:, 0]; y = pts[:, 1]
            # shoelace area
            area_poly = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
            total_area += area_poly
            # perimeter
            d = np.diff(pts, axis=0)
            total_per += float(np.sum(np.sqrt((d**2).sum(axis=1))))
            total_per += float(np.sqrt(((pts[0]-pts[-1])**2).sum()))
        return float(total_area), float(total_per), None

    # RLE
    if isinstance(seg, dict) and "size" in seg and "counts" in seg:
        H, W = int(seg["size"][0]), int(seg["size"][1])
        counts = seg["counts"]
        if isinstance(counts, list):
            # uncompressed RLE -> decode manually
            arr = np.zeros(H * W, dtype=np.uint8)
            idx = 0
            val = 0
            for run in counts:
                run = int(run)
                run = max(0, min(run, arr.size - idx))
                if val == 1:
                    arr[idx:idx+run] = 1
                idx += run
                val ^= 1
            m = arr.reshape((H, W), order="F")
            area_px = float(m.sum())
            # perimeter via simple 4-neighborhood transitions
            # (coarse but consistent)
            per = 0.0
            # horizontal
            per += float(np.sum(m[:, 1:] != m[:, :-1]))
            # vertical
            per += float(np.sum(m[1:, :] != m[:-1, :]))
            return area_px, per, None
        else:
            # compressed RLE (string)
            if HAS_PYCOCO:
                rle = {"size": [H, W], "counts": counts}
                m = mask_utils.decode(rle)
                area_px = float(mask_utils.area(rle))
                # perimeter approx via boundary transitions
                per = 0.0
                per += float(np.sum(m[:, 1:] != m[:, :-1]))
                per += float(np.sum(m[1:, :] != m[:-1, :]))
                return area_px, per, None
            else:
                # fallback: cannot decode; use no-mask fallback
                # if numeric area in annotation exists, caller will use it
                return None, None, None

    # Unknown
    return None, None, None

# ---------- Loader with robust category handling & fallbacks ----------
def load_coco_many(
    folder: Path,
    dataset_label: str,
    crystal_ids, nucleus_ids, defect_ids,
    crystal_names, nucleus_names, defect_names,
    verbose=True, diag_dir: Path = None
):
    jsons = sorted(folder.rglob("*.json"))
    if verbose:
        print(f"Analyzing: {folder}  [{dataset_label}]  JSONs found: {len(jsons)}")

    cryst_id_set   = set(crystal_ids)
    nuc_id_set     = set(nucleus_ids)
    defect_id_set  = set(defect_ids)
    cryst_name_l   = set([s.lower() for s in crystal_names])
    nuc_name_l     = set([s.lower() for s in nucleus_names])
    defect_name_l  = set([s.lower() for s in defect_names])

    # Diagnostics: counts of categories we observed
    cat_count = {}
    saw_compressed_rle = False

    rows = []
    for jp in jsons:
        try:
            data = load_json(jp)
        except Exception:
            continue

        if isinstance(data, dict):
            anns = data.get("annotations", [])
            images = {im["id"]: im for im in data.get("images", [])} if "images" in data else {}
            cat_by_id = {}
            if "categories" in data:
                for c in data["categories"]:
                    cid = c.get("id")
                    cname = c.get("name")
                    if cid is not None:
                        cat_by_id[cid] = cname
        elif isinstance(data, list):
            anns = data
            images = {}
            cat_by_id = {}
        else:
            anns = []
            images = {}
            cat_by_id = {}

        # Collect nuclei for this file (image-level)
        nuclei_pts_by_image = {}
        defects_by_image = {}

        # First pass: nuclei & defect masks
        for ann in anns:
            cid = ann.get("category_id")
            cname = ann.get("category") or (cat_by_id.get(cid) if cid in cat_by_id else ann.get("category"))

            # Update category histogram
            key = (cid, str(cname).lower() if cname else None)
            cat_count[key] = cat_count.get(key, 0) + 1

            # image size (if any)
            H = W = None
            im_id = ann.get("image_id")
            if im_id in images:
                im = images[im_id]
                H = im.get("height"); W = im.get("width")

            seg = ann.get("segmentation", None)
            if isinstance(seg, dict) and "size" in seg and isinstance(seg.get("counts"), str):
                saw_compressed_rle = True

            # nucleus?
            is_nucleus = False
            if cid in nuc_id_set:
                is_nucleus = True
            if cname and name_matches(cname, nucleus_names):
                is_nucleus = True
            if (not is_nucleus) and ann.get("keypoints"):
                is_nucleus = True

            if is_nucleus:
                # try keypoints; else use bbox center; else NaN
                k = ann.get("keypoints")
                if k and len(k) >= 2:
                    nx, ny = float(k[0]), float(k[1])
                else:
                    bbox = ann.get("bbox")
                    if bbox and len(bbox) >= 4:
                        nx, ny = centroid_from_bbox(bbox)
                    else:
                        nx = ny = np.nan
                nuclei_pts_by_image.setdefault(im_id, []).append((nx, ny))
                continue

            # defects?
            is_defect = False
            if cid in defect_id_set:
                is_defect = True
            if cname and name_matches(cname, defect_names):
                is_defect = True

            if is_defect:
                # we won’t build a pixel union if we can't decode; we’ll flag presence via fraction later if possible
                # For simplicity, we’ll just mark that a defect exists in this image; fraction needs overlap mask
                # If area-only JSONs, we can’t get overlap, so set fraction to 0 (conservative).
                defects_by_image[im_id] = True

        # Second pass: crystals
        for ann in anns:
            cid = ann.get("category_id")
            cname = ann.get("category") or (cat_by_id.get(cid) if cid in cat_by_id else ann.get("category"))
            seg = ann.get("segmentation", None)
            im_id = ann.get("image_id")

            # is crystal?
            is_crystal = False
            if cid in cryst_id_set:
                is_crystal = True
            if cname and name_matches(cname, crystal_names):
                is_crystal = True
            # fallback: if it has segmentation or area field, consider it a crystal unless it's already used as nucleus/defect by ID/name
            # (this opens the gate for your files that lack COCO categories)
            if not is_crystal:
                if seg is not None or ("area" in ann and ann["area"] is not None):
                    if not ( (cid in nuc_id_set) or (cname and name_matches(cname, nucleus_names)) or
                             (cid in defect_id_set) or (cname and name_matches(cname, defect_names)) ):
                        is_crystal = True
            if not is_crystal:
                continue

            # area/perimeter from seg if possible
            area_px = None
            per_px  = None
            used_mask = False
            if seg is not None:
                a, p, mask = area_perimeter_from_segmentation(seg)
                if (a is None or p is None) and isinstance(seg, dict) and "counts" in seg and isinstance(seg["counts"], str) and HAS_PYCOCO:
                    # try again via pycocotools (compressed RLE)
                    rle = {"size": seg["size"], "counts": seg["counts"]}
                    a = float(mask_utils.area(rle))
                    m = mask_utils.decode(rle)
                    p = 0.0
                    p += float(np.sum(m[:,1:] != m[:,:-1]))
                    p += float(np.sum(m[1:,:] != m[:-1,:]))
                area_px = a if a is not None else None
                per_px  = p if p is not None else None
                used_mask = (area_px is not None and per_px is not None)

            # fallback to numeric 'area' (COCO field) if present
            if area_px is None:
                if "area" in ann and ann["area"] is not None:
                    area_px = float(ann["area"])
                    # perimeter approx from equivalent circle
                    per_px = eq_circle_perimeter_from_area(area_px)
                else:
                    # nothing to do
                    continue

            # centroid fallback (bbox center), used only for nearest nucleus distance in this version
            cx, cy = (np.nan, np.nan)
            bbox = ann.get("bbox")
            if bbox and len(bbox) >= 4:
                cx, cy = centroid_from_bbox(bbox)

            # nearest nucleus distance (if any nucleus exists in that image)
            nx = ny = nd = np.nan
            if im_id in nuclei_pts_by_image and len(nuclei_pts_by_image[im_id]) > 0 and np.isfinite(cx) and np.isfinite(cy):
                pts = np.array(nuclei_pts_by_image[im_id], float)
                d2 = (pts[:,0]-cx)**2 + (pts[:,1]-cy)**2
                j = int(np.argmin(d2))
                nx, ny = float(pts[j,0]), float(pts[j,1])
                nd     = float(np.sqrt(float(np.min(d2))))

            # simple defect fraction: we don’t have pixel overlap without masks, set 0 if unknown
            defect_frac = 0.0
            # (If you later provide pixel masks for defects, we can compute overlap fraction)

            circ = circularity(area_px, per_px)

            rows.append({
                "dataset": dataset_label,
                "file": str(jp),
                "image_id": im_id,
                "area_px": float(area_px),
                "perimeter_px": float(per_px),
                "circularity": float(circ),
                "defect_frac": float(defect_frac),
                "centroid_x": float(cx), "centroid_y": float(cy),
                "nuc_x": float(nx), "nuc_y": float(ny), "nuc_dist_px": float(nd),
                "used_mask": bool(used_mask)
            })

    df = pd.DataFrame(rows)
    if diag_dir is not None:
        safe_mkdir(diag_dir)
        # category inventory
        # (we built inside the loop; rebuild quickly from df 'file' not available -> skip; we already printed runtime summary below)
        pass

    if df.empty:
        print(f"[SUMMARY {dataset_label}] crystals=0, nuclei=?, defects=?, rows_kept=0")
        if saw_compressed_rle and not HAS_PYCOCO:
            print("[HINT] Detected compressed RLE but pycocotools is not installed. "
                  "Install it to decode masks: pip install pycocotools")
        return df

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["area_px"])
    crystals = (df["area_px"] > 0).sum()
    nuclei = np.isfinite(df["nuc_x"]).sum()
    print(f"[SUMMARY {dataset_label}] crystals={crystals}, nuclei={nuclei}, rows_kept={len(df)}")
    if saw_compressed_rle and not HAS_PYCOCO:
        print("[NOTE] Some annotations used compressed RLE and were handled by numeric area fallback. "
              "Install pycocotools for exact masks/perimeters.")

    return df

# ---------- Nucleation & growth ----------
def rank_to_t0_ms(areas):
    a = np.asarray(areas, float)
    if a.size == 0:
        return np.array([], float)
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    denom = max(1, len(a) - 1)
    q = ranks / float(denom)
    return 60.0 * q

def fit_logN(t):
    t = np.asarray(t, float)
    t = t[(t > 0) & (t <= 60) & np.isfinite(t)]
    if len(t) < 5:
        return None
    l = np.log(t); mu=float(l.mean()); sigma=float(l.std(ddof=1))
    grid = np.arange(0, 61, 1, dtype=float)
    pdf = np.zeros_like(grid); g = grid > 0
    pdf[g] = stats.lognorm.pdf(grid[g], s=sigma, scale=np.exp(mu))
    ll = np.sum(stats.lognorm.logpdf(t, s=sigma, scale=np.exp(mu)))
    k = 2; n = len(t); aic = 2*k - 2*ll; bic = k*np.log(n) - 2*ll
    return {"model":"lognormal","grid_t":grid,"pdf":pdf,"mu":mu,"sigma":sigma,"AIC":aic,"BIC":bic}

def fit_gamma(t):
    t = np.asarray(t, float)
    t = t[(t > 0) & (t <= 60) & np.isfinite(t)]
    if len(t) < 5:
        return None
    k, loc, theta = stats.gamma.fit(t, floc=0)
    grid = np.arange(0, 61, 1, dtype=float)
    pdf = stats.gamma.pdf(grid, a=k, loc=0, scale=theta)
    ll  = np.sum(stats.gamma.logpdf(t, a=k, loc=0, scale=theta))
    p=2; n=len(t); aic=2*p-2*ll; bic=p*np.log(n)-2*ll
    return {"model":"gamma","grid_t":grid,"pdf":pdf,"k":k,"theta":theta,"AIC":aic,"BIC":bic}

def pick_I_model(t0_ms):
    g = fit_gamma(t0_ms); l = fit_logN(t0_ms)
    if g is None and l is None: return None
    if g is None: return l
    if l is None: return g
    return g if g["BIC"] < l["BIC"] else l

def calibrate_v0_alpha_beta(df, use_penalties=True):
    A = np.asarray(df["area_px"], float)
    R = np.sqrt(A / np.pi)
    C = np.clip(np.asarray(df["circularity"], float), 0, 1)
    phi = np.clip(np.asarray(df["defect_frac"], float), 0, 1)
    t0 = np.asarray(df["t0_ms"], float)
    dT = np.clip(600.0 - t0, 1.0, None)

    if not use_penalties:
        v0 = np.median(R / dT)
        return v0, 0.0, 0.0

    alphas = [0.0, 0.5, 1.0, 1.5]
    betas  = [0.0, 0.5, 1.0, 1.5]
    best = None
    for a in alphas:
        for b in betas:
            f = np.exp(-a*(1.0 - C)) * np.exp(-b*phi)
            v0s = (R / dT) / (f + 1e-9)
            v0  = np.median(v0s[np.isfinite(v0s) & (v0s > 0)])
            Rp  = v0 * f * dT
            err = np.nanmean((Rp - R)**2)
            if (best is None) or (err < best[0]):
                best = (err, v0, a, b)
    _, v0, a_opt, b_opt = best
    return v0, a_opt, b_opt

def build_X_pred(df, v0, alpha, beta):
    A = np.asarray(df["area_px"], float)
    R = np.sqrt(A/np.pi)
    C = np.clip(np.asarray(df["circularity"], float), 0, 1)
    phi = np.clip(np.asarray(df["defect_frac"], float), 0, 1)
    t0 = np.asarray(df["t0_ms"], float)

    f = np.exp(-alpha*(1.0 - C)) * np.exp(-beta*phi)
    t = np.arange(0, 601, 1, dtype=float)
    A_sum = np.zeros_like(t)
    for Ri, t0i, fi in zip(R, t0, f):
        ri = np.maximum(0.0, t - t0i) * (v0 * fi)
        Ai = np.pi * (ri**2)
        A_sum += Ai
    denom = max(A.sum(), 1.0)
    X = np.clip(A_sum / denom, 0, 1)
    return t, X

def fit_avrami_K(t_ms, X_pred, n_fixed):
    def loss(K):
        return np.nanmean((1.0 - np.exp(-K * (t_ms**n_fixed)) - X_pred)**2)
    res = optimize.minimize(lambda z: loss(z[0]),
                            x0=np.array([1e-3]),
                            bounds=[(1e-9, 1e3)])
    return float(res.x[0])

def bootstrap_dn_dt(t0_ms, n_boot=0):
    if n_boot <= 0: return None
    rng = np.random.default_rng(42)
    t0 = np.asarray(t0_ms, float)
    t0 = t0[(t0 > 0) & (t0 <= 60) & np.isfinite(t0)]
    if len(t0) < 10: return None
    curves = []
    for _ in range(n_boot):
        tb = rng.choice(t0, size=len(t0), replace=True)
        r  = pick_I_model(tb)
        if r is not None:
            curves.append(r["pdf"])
    if not curves:
        return None
    C = np.vstack(curves)
    lo = np.percentile(C, 2.5, axis=0)
    hi = np.percentile(C, 97.5, axis=0)
    grid = np.arange(0, 61, 1, dtype=float)
    return {"grid": grid, "lo": lo, "hi": hi}

# ---------- Plots ----------
def plot_dn_dt(resA, labelA, resB, labelB, outpng, bandA=None, bandB=None):
    plt.figure()
    if bandA:
        plt.fill_between(bandA["grid"], bandA["lo"], bandA["hi"], alpha=0.2, label=f"{labelA} 95% CI")
    if bandB:
        plt.fill_between(bandB["grid"], bandB["lo"], bandB["hi"], alpha=0.2, label=f"{labelB} 95% CI")
    if resA is not None:
        plt.plot(resA["grid_t"], resA["pdf"], label=f"{labelA} dn/dt ({resA['model']})")
    if resB is not None:
        plt.plot(resB["grid_t"], resB["pdf"], label=f"{labelB} dn/dt ({resB['model']})")
    plt.xlabel("t (ms)"); plt.ylabel("dn/dt (a.u.)")
    plt.title("Nucleation rate density (0–60 ms)")
    plt.legend(); plt.tight_layout(); plt.savefig(outpng, dpi=200); plt.close()

def plot_X(t, XA, XB, n, K_A, K_B, labelA, labelB, outpng):
    plt.figure()
    plt.plot(t, XA, label=f"X_pred {labelA}")
    plt.plot(t, 1.0 - np.exp(-K_A * (t**n)), "--", label=f"Avrami {labelA} (n={n:.2f}, K={K_A:.3g})")
    plt.plot(t, XB, label=f"X_pred {labelB}")
    plt.plot(t, 1.0 - np.exp(-K_B * (t**n)), "--", label=f"Avrami {labelB} (n={n:.2f}, K={K_B:.3g})")
    plt.xlabel("t (ms)"); plt.ylabel("X(t)"); plt.ylim(0, 1.05)
    plt.title("X(t) and Avrami overlays")
    plt.legend(); plt.tight_layout(); plt.savefig(outpng, dpi=200); plt.close()

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Compare FAPI vs FAPI-TEMPO dn/dt and X(t) from COCO-like JSONs.")
    ap.add_argument("--folders", nargs=2, required=True)
    ap.add_argument("--labels",  nargs=2, required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n_avrami", type=float, default=2.5)
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--calibrate_penalties", action="store_true")

    # Category config (IDs and/or names)
    ap.add_argument("--crystal_ids", nargs="*", type=int, default=[1])
    ap.add_argument("--nucleus_ids", nargs="*", type=int, default=[2])
    ap.add_argument("--defect_ids",  nargs="*", type=int, default=[3])
    ap.add_argument("--crystal_names", nargs="*", default=["crystal","mask","grain","spherulite"])
    ap.add_argument("--nucleus_names", nargs="*", default=["nucleus","seed","nuclei","nucleation","point"])
    ap.add_argument("--defect_names",  nargs="*", default=["defect","void","crack","hole","defective"])

    args = ap.parse_args()

    folderA, folderB = map(lambda p: Path(p).resolve(), args.folders)
    labelA, labelB = args.labels

    outdir = Path(args.out).resolve() if args.out else (folderA.parent / "comparative_outputs_FAPI_vs_TEMPO_v4")
    safe_mkdir(outdir)

    # Load both datasets with fallbacks
    dfA = load_coco_many(folderA, labelA,
                         args.crystal_ids, args.nucleus_ids, args.defect_ids,
                         args.crystal_names, args.nucleus_names, args.defect_names,
                         verbose=True, diag_dir=outdir)
    dfB = load_coco_many(folderB, labelB,
                         args.crystal_ids, args.nucleus_ids, args.defect_ids,
                         args.crystal_names, args.nucleus_names, args.defect_names,
                         verbose=True, diag_dir=outdir)

    if dfA.empty or dfB.empty:
        print("One or both datasets are empty. Aborting comparison after logging.")
        # Save tiny diagnostics
        if not dfA.empty:
            dfA.head(50).to_csv(outdir / f"diagnostic_first_rows_{labelA}.csv", index=False)
        if not dfB.empty:
            dfB.head(50).to_csv(outdir / f"diagnostic_first_rows_{labelB}.csv", index=False)
        return

    # Nucleation pseudo-times by rank (0..60 ms)
    dfA["t0_ms"] = rank_to_t0_ms(dfA["area_px"])
    dfB["t0_ms"] = rank_to_t0_ms(dfB["area_px"])

    # Fit dn/dt and bootstrap bands
    resA  = pick_I_model(dfA["t0_ms"])
    resB  = pick_I_model(dfB["t0_ms"])
    bandA = bootstrap_dn_dt(dfA["t0_ms"], n_boot=args.bootstrap)
    bandB = bootstrap_dn_dt(dfB["t0_ms"], n_boot=args.bootstrap)

    # Save dn/dt CSV/plot
    grid = np.arange(0, 61, 1)
    pdfA = resA["pdf"] if resA is not None else np.zeros_like(grid)
    pdfB = resB["pdf"] if resB is not None else np.zeros_like(grid)
    pd.DataFrame({"t_ms": grid, f"dn_dt_{labelA}": pdfA, f"dn_dt_{labelB}": pdfB}).to_csv(outdir/"dn_dt_compare.csv", index=False)
    plot_dn_dt(resA, labelA, resB, labelB, str(outdir/"dn_dt_compare.png"), bandA, bandB)

    # Growth calibration & X(t)
    v0A, aA, bA = calibrate_v0_alpha_beta(dfA, use_penalties=args.calibrate_penalties)
    v0B, aB, bB = calibrate_v0_alpha_beta(dfB, use_penalties=args.calibrate_penalties)
    t, XA = build_X_pred(dfA, v0A, aA, bA)
    _, XB = build_X_pred(dfB, v0B, aB, bB)
    K_A = fit_avrami_K(t, XA, args.n_avrami)
    K_B = fit_avrami_K(t, XB, args.n_avrami)

    # Save X overlays
    pd.DataFrame({
        "t_ms": t,
        f"X_pred_{labelA}": XA,
        f"X_pred_{labelB}": XB,
        f"X_Avrami_{labelA}": 1.0 - np.exp(-K_A * (t**args.n_avrami)),
        f"X_Avrami_{labelB}": 1.0 - np.exp(-K_B * (t**args.n_avrami)),
    }).to_csv(outdir/"X_overlays.csv", index=False)
    plot_X(t, XA, XB, args.n_avrami, K_A, K_B, labelA, labelB, str(outdir/"X_overlays.png"))

    # Parameter summary
    def pack(label, res, v0, aa, bb, K):
        d = {"dataset": label, "v0_px_per_ms": v0, "alpha": aa, "beta": bb, "K": K}
        if res is None:
            d.update({"I_model":"NA"})
        else:
            d.update({"I_model": res["model"], "AIC": res["AIC"], "BIC": res["BIC"]})
            if res["model"] == "gamma":
                d.update({"k": res["k"], "theta": res["theta"]})
            else:
                d.update({"mu": res["mu"], "sigma": res["sigma"]})
        return d

    params = pd.DataFrame([pack(labelA, resA, v0A, aA, bA, K_A),
                           pack(labelB, resB, v0B, aB, bB, K_B)])
    params.to_csv(outdir/"fit_parameters_summary.csv", index=False)

    # Export objects
    keep = ["dataset","file","image_id","area_px","perimeter_px","circularity",
            "defect_frac","centroid_x","centroid_y","nuc_x","nuc_y","nuc_dist_px","t0_ms","used_mask"]
    dfA[keep].to_csv(outdir/f"objects_{labelA}.csv", index=False)
    dfB[keep].to_csv(outdir/f"objects_{labelB}.csv", index=False)

    print("=== Done ===")
    print(f"Output: {outdir}")

if __name__ == "__main__":
    main()
