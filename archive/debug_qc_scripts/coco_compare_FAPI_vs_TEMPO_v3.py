import os, json, math, argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize
from skimage.draw import polygon2mask
from skimage.measure import find_contours

# ---------- Helpers ----------
def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def as_iter(x):
    return x if isinstance(x, (list, tuple)) else [x]

def infer_hw_from_poly(seg):
    xs, ys = [], []
    for poly in as_iter(seg):
        arr = np.asarray(poly, dtype=float).reshape(-1, 2)
        xs.append(arr[:,0]); ys.append(arr[:,1])
    x = np.concatenate(xs) if xs else np.array([0.0])
    y = np.concatenate(ys) if ys else np.array([0.0])
    W = int(np.clip(np.ceil(x.max())+4, 8, 16384))
    H = int(np.clip(np.ceil(y.max())+4, 8, 16384))
    return H, W

def mask_from_annotation(ann, H=None, W=None):
    seg = ann.get("segmentation", None)
    if seg is None:
        return None, H, W
    # Uncompressed RLE
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
                run = max(0, min(run, arr.size-idx))
                if val == 1:
                    arr[idx:idx+run] = 1
                idx += run
                val ^= 1
            m = arr.reshape((H, W), order="F")
            return m, H, W
        else:
            # Compressed RLE (string) requires pycocotools -> skip gracefully
            return None, H, W
    # Polygons
    if isinstance(seg, list):
        if H is None or W is None:
            H, W = infer_hw_from_poly(seg)
        m = np.zeros((H, W), dtype=np.uint8)
        for poly in seg:
            pts = np.asarray(poly, dtype=float).reshape(-1, 2)
            m |= polygon2mask((H, W), np.fliplr(pts)).astype(np.uint8)
        return m, H, W
    return None, H, W

def area_perimeter(m):
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

def centroid(m):
    if m is None or m.sum()==0:
        return (np.nan, np.nan)
    ys, xs = np.nonzero(m)
    return float(xs.mean()), float(ys.mean())

def nearest_point_to_mask(m, x, y):
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return np.nan, np.nan, np.inf
    d2 = (xs-x)**2 + (ys-y)**2
    j = int(np.argmin(d2))
    return float(xs[j]), float(ys[j]), float(np.sqrt(d2[j]))

def circularity(area, per):
    if area<=0 or per<=0: return 0.0
    return float(4*math.pi*area/(per**2))

def name_matches(n, targets_lower):
    if not n: return False
    nl = str(n).strip().lower()
    return any(t in nl for t in targets_lower)

# ---------- Loader with configurable categories ----------
def load_coco_many(
    folder: Path,
    dataset_label: str,
    crystal_ids, nucleus_ids, defect_ids,
    crystal_names, nucleus_names, defect_names,
    verbose=True
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

        nuclei_pts = {}
        defects_union = {}
        size_cache = {}

        # First pass: collect nuclei and defects per image
        for ann in anns:
            cid = ann.get("category_id")
            cname = ann.get("category") or (cat_by_id.get(cid) if cid in cat_by_id else ann.get("category"))

            # image H,W
            H=W=None
            if "image_id" in ann and ann["image_id"] in images:
                im = images[ann["image_id"]]
                H = im.get("height"); W = im.get("width")
            seg = ann.get("segmentation")
            if isinstance(seg, dict) and "size" in seg:
                H = H or int(seg["size"][0]); W = W or int(seg["size"][1])
            elif isinstance(seg, list) and (H is None or W is None):
                H,W = infer_hw_from_poly(seg)
            if "image_id" in ann and (H and W):
                size_cache[ann["image_id"]] = (int(H), int(W))

            # nucleus?
            is_nucleus = False
            if cid in nuc_id_set: is_nucleus=True
            if cname and name_matches(cname, nucleus_names): is_nucleus=True
            if (not is_nucleus) and ann.get("keypoints"): is_nucleus=True

            # defect?
            is_defect = False
            if cid in defect_id_set: is_defect=True
            if cname and name_matches(cname, defect_names): is_defect=True

            if is_nucleus:
                k = ann.get("keypoints")
                if k and len(k)>=2:
                    nx,ny=float(k[0]),float(k[1])
                else:
                    # centroid fallback if provided as tiny mask
                    try:
                        m,HH,WW = mask_from_annotation(ann,H,W)
                    except RuntimeError:
                        m=None; HH=W=None
                    if m is not None:
                        nx,ny = centroid(m)
                    else:
                        nx=ny=np.nan
                nuclei_pts.setdefault(ann.get("image_id"),[]).append((nx,ny))
            elif is_defect:
                try:
                    m,HH,WW = mask_from_annotation(ann,H,W)
                except RuntimeError:
                    m=None; HH=W=None
                if m is not None:
                    mu = defects_union.get(ann.get("image_id"))
                    if mu is None:
                        mu = np.zeros_like(m, dtype=np.uint8)
                    mu |= m.astype(np.uint8)
                    defects_union[ann.get("image_id")] = mu

        # Second pass: crystals
        for ann in anns:
            cid = ann.get("category_id")
            cname = ann.get("category") or (cat_by_id.get(cid) if cid in cat_by_id else ann.get("category"))
            is_crystal = False
            if cid in cryst_id_set: is_crystal=True
            if cname and name_matches(cname, cryst_name_l): is_crystal=True
            # fallback: any annotation with a segmentation might be a crystal
            if not is_crystal and ann.get("segmentation") is not None:
                is_crystal=True

            if not is_crystal:
                continue

            image_id = ann.get("image_id")
            H,W = size_cache.get(image_id,(None,None))
            try:
                m,HH,WW = mask_from_annotation(ann,H,W)
            except RuntimeError:
                continue
            if m is None:
                continue

            A,P = area_perimeter(m)
            if A<=0:
                continue
            C = circularity(A,P)
            cx,cy = centroid(m)

            # nearest nucleus to centroid
            nx=ny=np.nan; nd=np.nan
            if image_id in nuclei_pts and len(nuclei_pts[image_id])>0:
                pts = np.array(nuclei_pts[image_id],float)
                d2 = (pts[:,0]-cx)**2 + (pts[:,1]-cy)**2
                j = int(np.argmin(d2))
                nx,ny = float(pts[j,0]), float(pts[j,1])
                _,_,nd = nearest_point_to_mask(m, nx, ny)

            # defect overlap fraction
            phi=0.0
            mu = defects_union.get(image_id)
            if mu is not None:
                inter = (mu & m).sum()
                phi = float(inter)/float(A+1e-6)

            rows.append({
                "dataset": dataset_label,
                "file": str(jp),
                "image_id": image_id,
                "area_px": float(A),
                "perimeter_px": float(P),
                "circularity": float(C),
                "centroid_x": float(cx), "centroid_y": float(cy),
                "nuc_x": float(nx), "nuc_y": float(ny), "nuc_dist_px": float(nd),
                "defect_frac": float(phi)
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"[SUMMARY {dataset_label}] crystals=0, nuclei=?, defects=?, rows_kept=0")
        return df

    df = df.replace([np.inf,-np.inf], np.nan).dropna(subset=["area_px"])
    crystals = (df["area_px"]>0).sum()
    nuclei   = np.isfinite(df["nuc_x"]).sum()
    defects  = (df["defect_frac"]>0).sum()
    print(f"[SUMMARY {dataset_label}] crystals={crystals}, nuclei={nuclei}, defects={defects}, rows_kept={len(df)}")
    return df

# ---------- Nucleation & growth ----------
def rank_to_t0_ms(areas):
    a = np.asarray(areas,float)
    order = np.argsort(a)
    ranks = np.empty_like(order,dtype=float)
    ranks[order] = np.arange(len(a),dtype=float)
    denom = max(1, len(a)-1)
    q = ranks/float(denom)
    return 60.0*q

def fit_logN(t):
    t = np.asarray(t,float)
    t = t[(t>0)&(t<=60)&np.isfinite(t)]
    if len(t)<5: return None
    l = np.log(t); mu=float(l.mean()); sigma=float(l.std(ddof=1))
    grid = np.arange(0,61,1,dtype=float)
    pdf = np.zeros_like(grid); g = grid>0
    pdf[g] = stats.lognorm.pdf(grid[g], s=sigma, scale=np.exp(mu))
    ll = np.sum(stats.lognorm.logpdf(t, s=sigma, scale=np.exp(mu)))
    k=2; n=len(t); aic=2*k-2*ll; bic=k*np.log(n)-2*ll
    return {"model":"lognormal","grid_t":grid,"pdf":pdf,"mu":mu,"sigma":sigma,"AIC":aic,"BIC":bic}

def fit_gamma(t):
    t = np.asarray(t,float)
    t = t[(t>0)&(t<=60)&np.isfinite(t)]
    if len(t)<5: return None
    k,loc,theta = stats.gamma.fit(t, floc=0)
    grid = np.arange(0,61,1,dtype=float)
    pdf = stats.gamma.pdf(grid, a=k, loc=0, scale=theta)
    ll  = np.sum(stats.gamma.logpdf(t, a=k, loc=0, scale=theta))
    p=2; n=len(t); aic=2*p-2*ll; bic=p*np.log(n)-2*ll
    return {"model":"gamma","grid_t":grid,"pdf":pdf,"k":k,"theta":theta,"AIC":aic,"BIC":bic}

def pick_I_model(t0_ms):
    g = fit_gamma(t0_ms); l = fit_logN(t0_ms)
    if g is None and l is None: return None
    if g is None: return l
    if l is None: return g
    return g if g["BIC"]<l["BIC"] else l

def calibrate_v0_alpha_beta(df, use_penalties=True):
    A = np.asarray(df["area_px"], float)
    R = np.sqrt(A/np.pi)
    C = np.clip(np.asarray(df["circularity"], float),0,1)
    phi=np.clip(np.asarray(df["defect_frac"], float),0,1)
    t0 = np.asarray(df["t0_ms"], float)
    dT = np.clip(600.0 - t0, 1.0, None)

    if not use_penalties:
        v0 = np.median(R/dT)
        return v0, 0.0, 0.0

    alphas=[0.0,0.5,1.0,1.5]; betas=[0.0,0.5,1.0,1.5]
    best=None
    for a in alphas:
        for b in betas:
            f = np.exp(-a*(1.0-C))*np.exp(-b*phi)
            v0s = (R/dT)/(f+1e-9)
            v0  = np.median(v0s[np.isfinite(v0s)&(v0s>0)])
            Rp  = v0*f*dT
            err = np.nanmean((Rp-R)**2)
            if (best is None) or (err<best[0]):
                best=(err,v0,a,b)
    _,v0,a_opt,b_opt = best
    return v0,a_opt,b_opt

def build_X_pred(df, v0, alpha, beta):
    A = np.asarray(df["area_px"], float)
    R = np.sqrt(A/np.pi)
    C = np.clip(np.asarray(df["circularity"], float),0,1)
    phi=np.clip(np.asarray(df["defect_frac"], float),0,1)
    t0 = np.asarray(df["t0_ms"], float)

    f = np.exp(-alpha*(1.0-C))*np.exp(-beta*phi)
    t = np.arange(0,601,1,dtype=float)
    A_sum = np.zeros_like(t)
    for Ri,t0i,fi in zip(R,t0,f):
        ri = np.maximum(0.0, t-t0i)*(v0*fi)
        Ai = np.pi*(ri**2)
        A_sum += Ai
    denom = max(A.sum(), 1.0)
    X = np.clip(A_sum/denom, 0,1)
    return t,X

def fit_avrami_K(t_ms,X_pred,n_fixed):
    def loss(K):
        return np.nanmean((1.0-np.exp(-K*(t_ms**n_fixed))-X_pred)**2)
    res = optimize.minimize(lambda z: loss(z[0]), x0=np.array([1e-3]), bounds=[(1e-9,1e3)])
    return float(res.x[0])

def bootstrap_dn_dt(t0_ms, n_boot=0):
    if n_boot<=0: return None
    rng = np.random.default_rng(42)
    t0 = np.asarray(t0_ms,float)
    t0 = t0[(t0>0)&(t0<=60)&np.isfinite(t0)]
    if len(t0)<10: return None
    curves=[]
    for _ in range(n_boot):
        tb = rng.choice(t0, size=len(t0), replace=True)
        r  = pick_I_model(tb)
        if r is not None:
            curves.append(r["pdf"])
    if not curves: return None
    C = np.vstack(curves)
    lo = np.percentile(C,2.5,axis=0); hi = np.percentile(C,97.5,axis=0)
    grid = np.arange(0,61,1,dtype=float)
    return {"grid":grid,"lo":lo,"hi":hi}

# ---------- Plots ----------
def plot_dn_dt(resA,labelA,resB,labelB,outpng,bandA=None,bandB=None):
    plt.figure()
    if bandA:
        plt.fill_between(bandA["grid"],bandA["lo"],bandA["hi"],alpha=0.2,label=f"{labelA} 95% CI")
    if bandB:
        plt.fill_between(bandB["grid"],bandB["lo"],bandB["hi"],alpha=0.2,label=f"{labelB} 95% CI")
    if resA is not None:
        plt.plot(resA["grid_t"],resA["pdf"],label=f"{labelA} dn/dt ({resA['model']})")
    if resB is not None:
        plt.plot(resB["grid_t"],resB["pdf"],label=f"{labelB} dn/dt ({resB['model']})")
    plt.xlabel("t (ms)"); plt.ylabel("dn/dt (a.u.)"); plt.title("Nucleation rate density (0–60 ms)")
    plt.legend(); plt.tight_layout(); plt.savefig(outpng,dpi=200); plt.close()

def plot_X(t,XA,XB,n,K_A,K_B,labelA,labelB,outpng):
    plt.figure()
    plt.plot(t,XA,label=f"X_pred {labelA}")
    plt.plot(t,1.0-np.exp(-K_A*(t**n)),"--",label=f"Avrami {labelA} (n={n:.2f}, K={K_A:.3g})")
    plt.plot(t,XB,label=f"X_pred {labelB}")
    plt.plot(t,1.0-np.exp(-K_B*(t**n)),"--",label=f"Avrami {labelB} (n={n:.2f}, K={K_B:.3g})")
    plt.xlabel("t (ms)"); plt.ylabel("X(t)"); plt.ylim(0,1.05); plt.title("X(t) and Avrami overlays")
    plt.legend(); plt.tight_layout(); plt.savefig(outpng,dpi=200); plt.close()

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Compare FAPI vs FAPI-TEMPO dn/dt and X(t) from COCO JSONs.")
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
    ap.add_argument("--nucleus_names", nargs="*", default=["nucleus","seed","nuclei","nucleation"])
    ap.add_argument("--defect_names",  nargs="*", default=["defect","void","crack","hole"])

    args = ap.parse_args()
    folderA, folderB = map(lambda p: Path(p).resolve(), args.folders)
    labelA, labelB = args.labels

    if args.out:
        outdir = Path(args.out).resolve()
    else:
        outdir = folderA.parent / "comparative_outputs_FAPI_vs_TEMPO"
    safe_mkdir(outdir)

    dfA = load_coco_many(folderA, labelA,
                         args.crystal_ids, args.nucleus_ids, args.defect_ids,
                         args.crystal_names, args.nucleus_names, args.defect_names)
    dfB = load_coco_many(folderB, labelB,
                         args.crystal_ids, args.nucleus_ids, args.defect_ids,
                         args.crystal_names, args.nucleus_names, args.defect_names)

    if dfA.empty or dfB.empty:
        print("One or both datasets are empty. Aborting comparison after logging.")
        return

    # nucleation times by rank (0..60 ms)
    dfA["t0_ms"] = rank_to_t0_ms(dfA["area_px"])
    dfB["t0_ms"] = rank_to_t0_ms(dfB["area_px"])

    # dn/dt fit + bootstrap
    resA = pick_I_model(dfA["t0_ms"])
    resB = pick_I_model(dfB["t0_ms"])
    bandA = bootstrap_dn_dt(dfA["t0_ms"], n_boot=args.bootstrap)
    bandB = bootstrap_dn_dt(dfB["t0_ms"], n_boot=args.bootstrap)

    # save dn/dt csv + plot
    gridA = resA["grid_t"] if resA is not None else np.arange(0,61,1)
    gridB = resB["grid_t"] if resB is not None else gridA
    pdfA  = resA["pdf"]     if resA is not None else np.zeros_like(gridA)
    pdfB  = resB["pdf"]     if resB is not None else np.zeros_like(gridB)
    pd.DataFrame({
        "t_ms": gridA,
        f"dn_dt_{labelA}": pdfA,
        f"dn_dt_{labelB}": pdfB
    }).to_csv(outdir/"dn_dt_compare.csv", index=False)
    plot_dn_dt(resA,labelA,resB,labelB,str(outdir/"dn_dt_compare.png"), bandA, bandB)

    # growth calibration & X(t) with circularity/defect penalties if requested
    v0A, aA, bA = calibrate_v0_alpha_beta(dfA, use_penalties=args.calibrate_penalties)
    v0B, aB, bB = calibrate_v0_alpha_beta(dfB, use_penalties=args.calibrate_penalties)

    t, XA = build_X_pred(dfA, v0A, aA, bA)
    _, XB = build_X_pred(dfB, v0B, aB, bB)
    K_A = fit_avrami_K(t, XA, args.n_avrami)
    K_B = fit_avrami_K(t, XB, args.n_avrami)

    pd.DataFrame({
        "t_ms": t,
        f"X_pred_{labelA}": XA,
        f"X_pred_{labelB}": XB,
        f"X_Avrami_{labelA}": 1.0-np.exp(-K_A*(t**args.n_avrami)),
        f"X_Avrami_{labelB}": 1.0-np.exp(-K_B*(t**args.n_avrami)),
    }).to_csv(outdir/"X_overlays.csv", index=False)
    plot_X(t, XA, XB, args.n_avrami, K_A, K_B, labelA, labelB, str(outdir/"X_overlays.png"))

    # parameters summary
    def pack_row(label,res,v0,aa,bb,K):
        d={"dataset":label,"v0_px_per_ms":v0,"alpha":aa,"beta":bb,"K":K}
        if res is None:
            d.update({"I_model":"NA"})
        else:
            d.update({"I_model":res["model"],"AIC":res["AIC"],"BIC":res["BIC"]})
            if res["model"]=="gamma":
                d.update({"k":res["k"],"theta":res["theta"]})
            else:
                d.update({"mu":res["mu"],"sigma":res["sigma"]})
        return d

    params_df = pd.DataFrame([
        pack_row(labelA,resA,v0A,aA,bA,K_A),
        pack_row(labelB,resB,v0B,aB,bB,K_B),
    ])
    params_df.to_csv(outdir/"fit_parameters_summary.csv", index=False)

    # objects export
    keep = ["dataset","file","image_id","area_px","perimeter_px","circularity",
            "defect_frac","centroid_x","centroid_y","nuc_x","nuc_y","nuc_dist_px","t0_ms"]
    dfA[keep].to_csv(outdir/f"objects_{labelA}.csv", index=False)
    dfB[keep].to_csv(outdir/f"objects_{labelB}.csv", index=False)

    print("=== Done ===")
    print(f"Output: {outdir}")

if __name__ == "__main__":
    main()
