import argparse, json, math, os, glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats, optimize
import matplotlib.pyplot as plt

# ---------- Geometry helpers ----------
def polygon_area_perimeter(seg):
    # seg is a list of flat [x1,y1,x2,y2,...]
    xs = seg[0::2]; ys = seg[1::2]
    n  = len(xs)
    if n < 3: return 0.0, 0.0
    # area (shoelace)
    area = 0.0
    per  = 0.0
    for i in range(n):
        j = (i+1)%n
        area += xs[i]*ys[j] - xs[j]*ys[i]
        dx = xs[j]-xs[i]; dy = ys[j]-ys[i]
        per += (dx*dx + dy*dy)**0.5
    return abs(area)/2.0, per

def rle_counts_to_area(rle_counts, size):
    # simple COCO-style uncompressed counts (list of run lengths)
    # counts alternate white/black; area = sum of black runs
    # size = [height, width]
    if not isinstance(rle_counts, (list, tuple)):
        return None
    arr = list(rle_counts)
    if len(arr)==0: return 0.0
    # If starts with white, black runs are arr[1], arr[3],...
    # If starts with black (rare in COCO), adjust. Typical: start with white.
    area = 0
    for i, run in enumerate(arr):
        if i % 2 == 1:  # black
            area += int(run)
    return float(area)

def safe_float(x):
    try: return float(x)
    except: return None

# ---------- Flexible JSON reader ----------
def read_json_flexible(p: Path):
    txt = p.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return {}, []
    try:
        data = json.loads(txt)
    except Exception:
        # try JSONL
        anns = []
        cats = []
        with p.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line=line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict) and ("annotations" in obj or "images" in obj):
                    cats.extend(obj.get("categories", []))
                    anns.extend(obj.get("annotations", []))
                elif isinstance(obj, dict):
                    anns.append(obj)
                elif isinstance(obj, list):
                    for el in obj:
                        if isinstance(el, dict):
                            anns.append(el)
        return {"categories": cats}, anns

    if isinstance(data, dict):
        return data, data.get("annotations", [])
    elif isinstance(data, list):
        return {}, data
    else:
        return {}, []

def normalize_cat_id(cid):
    # accept string/int
    if isinstance(cid, str):
        try: return int(cid)
        except: return cid  # leave as is
    return cid

def make_cat_maps(cats):
    id2name = {}
    name2id = {}
    for c in cats or []:
        cid = normalize_cat_id(c.get("id"))
        nm  = c.get("name")
        if cid is not None and nm:
            id2name[cid] = nm
            name2id[nm]  = cid
    return id2name, name2id

# ---------- One-file to tidy ----------
def load_one_coco(json_path,
                  crystal_ids, nucleus_ids, defect_ids,
                  crystal_names=None, nucleus_names=None, defect_names=None):
    data, anns = read_json_flexible(json_path)
    cats = data.get("categories", [])
    id2name, name2id = make_cat_maps(cats)

    # Allow matching by names (optional)
    def matches_category(a, wanted_ids, wanted_names):
        cid = normalize_cat_id(a.get("category_id"))
        nm  = a.get("category_name") or a.get("name")
        # map cat_name from official table if available
        if cid in id2name and nm is None:
            nm = id2name[cid]
        in_id = (cid in wanted_ids) if wanted_ids else False
        in_nm = (str(nm).lower() in {s.lower() for s in (wanted_names or [])}) if nm else False
        return in_id or in_nm

    rows = []
    nuc_pts = []  # (image_id, x, y)

    for a in anns:
        if not isinstance(a, dict): continue
        # image id fallback 0 if absent
        iid = a.get("image_id", 0)
        cid = a.get("category_id")
        nm  = a.get("category_name") or a.get("name")

        # unify “type”
        is_crystal = matches_category(a, crystal_ids, crystal_names)
        is_defect  = matches_category(a, defect_ids, defect_names)
        is_nuc     = matches_category(a, nucleus_ids, nucleus_names)

        # extract area/perimeter if present
        area  = safe_float(a.get("area"))
        perim = safe_float(a.get("perimeter"))

        # derive from segmentation if missing
        if area is None or perim is None:
            seg = a.get("segmentation")
            if isinstance(seg, dict) and "counts" in seg and "size" in seg:
                # uncompressed counts = list -> we can get area
                area_from_rle = rle_counts_to_area(seg["counts"], seg["size"])
                if area is None: area = area_from_rle
                # perimeter from RLE requires contour extraction; skip if not available
            elif isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], list):
                # multiple polygons
                a_sum = 0.0; p_sum = 0.0
                for poly in seg:
                    a_i, p_i = polygon_area_perimeter(poly)
                    a_sum += a_i; p_sum += p_i
                if area is None:  area  = a_sum
                if perim is None: perim = p_sum
            elif isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], (int,float)):
                # single polygon flat list
                a_i, p_i = polygon_area_perimeter(seg)
                if area is None:  area  = a_i
                if perim is None: perim = p_i

        if is_crystal:
            rows.append(dict(
                dataset=Path(json_path).stem,
                image_id=iid,
                object_id=a.get("id"),
                area_px=area if area is not None else np.nan,
                perimeter_px=perim if perim is not None else np.nan,
                is_crystal=True,
                is_defect=False
            ))
        elif is_defect:
            rows.append(dict(
                dataset=Path(json_path).stem,
                image_id=iid,
                object_id=a.get("id"),
                area_px=area if area is not None else np.nan,
                perimeter_px=perim if perim is not None else np.nan,
                is_crystal=False,
                is_defect=True
            ))
        elif is_nuc:
            # attempt to get a point
            x = y = None
            if "keypoints" in a and isinstance(a["keypoints"], (list,tuple)) and len(a["keypoints"])>=2:
                x,y = a["keypoints"][:2]
            elif "point" in a and isinstance(a["point"], (list,tuple)) and len(a["point"])>=2:
                x,y = a["point"][:2]
            elif "bbox" in a and isinstance(a["bbox"], (list,tuple)) and len(a["bbox"])>=4:
                bx,by,bw,bh = a["bbox"]
                x = bx + bw/2.0; y = by + bh/2.0
            if x is not None and y is not None:
                nuc_pts.append((iid,float(x),float(y)))

    if not rows:
        return pd.DataFrame(columns=["dataset","image_id","object_id","area_px","perimeter_px",
                                     "circularity","defect_area_px","defect_frac","nucleus_x","nucleus_y"])

    df = pd.DataFrame(rows)
    crystals = df[df["is_crystal"]].copy()
    if crystals.empty:
        # (No crystals matched names/ids)
        return pd.DataFrame(columns=["dataset","image_id","object_id","area_px","perimeter_px",
                                     "circularity","defect_area_px","defect_frac","nucleus_x","nucleus_y"])

    # circularity (fallback to 1 if perimeter missing)
    A = crystals["area_px"].astype(float).to_numpy()
    P = crystals["perimeter_px"].astype(float).to_numpy()
    circ = np.ones_like(A)
    valid = (A>0) & (P>0)
    circ[valid] = (4.0*np.pi*A[valid])/(P[valid]**2)
    crystals["circularity"] = np.clip(circ, 0, 1)

    crystals["defect_area_px"] = 0.0
    crystals["defect_frac"]    = 0.0

    # attach nuclei (mean per image as fallback)
    crystals["nucleus_x"] = np.nan; crystals["nucleus_y"] = np.nan
    if nuc_pts:
        nu = pd.DataFrame(nuc_pts, columns=["image_id","x","y"])
        for iid, idxs in crystals.groupby("image_id").groups.items():
            pts = nu[nu["image_id"]==iid]
            if len(pts):
                cx, cy = pts["x"].mean(), pts["y"].mean()
                crystals.loc[idxs, "nucleus_x"] = cx
                crystals.loc[idxs, "nucleus_y"] = cy

    keep = crystals[["dataset","image_id","object_id","area_px","perimeter_px",
                     "circularity","defect_area_px","defect_frac","nucleus_x","nucleus_y"]].copy()
    return keep.replace([np.inf,-np.inf], np.nan)

def load_coco_many(folder, crystal_ids, nucleus_ids, defect_ids, crystal_names, nucleus_names, defect_names):
    frames = []
    for jp in sorted(Path(folder).glob("*.json")):
        df = load_one_coco(jp, crystal_ids, nucleus_ids, defect_ids, crystal_names, nucleus_names, defect_names)
        if len(df): frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["dataset","image_id","object_id","area_px","perimeter_px",
                                     "circularity","defect_area_px","defect_frac","nucleus_x","nucleus_y"])
    df = pd.concat(frames, ignore_index=True)
    # clean
    df = df.dropna(subset=["area_px"])
    df["circularity"] = df["circularity"].fillna(1.0).clip(0,1)
    df["defect_frac"] = df["defect_frac"].fillna(0.0).clip(0,1)
    return df

# ---------- Kinetics (same as before) ----------
def rank_to_t0_ms(area_px):
    area = np.asarray(area_px, float)
    order = np.argsort(area)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(area))
    q = ranks/(max(len(area)-1,1))
    return 60.0*q

def t0_from_backcalc(df, v0, alpha, beta):
    A   = np.asarray(df["area_px"], float)
    R   = np.sqrt(np.maximum(A,0.0)/np.pi)
    C   = np.clip(np.asarray(df["circularity"], float), 0, 1)
    phi = np.clip(np.asarray(df["defect_frac"], float), 0, 1)
    f   = np.exp(-alpha*(1.0 - C)) * np.exp(-beta*phi)
    dt  = R/np.maximum(v0*np.maximum(f,1e-9), 1e-9)
    return np.clip(600.0 - dt, 0.0, 60.0)

def calibrate_v0_alpha_beta(df, use_penalties=True):
    A   = np.asarray(df["area_px"], float)
    R   = np.sqrt(np.maximum(A,0.0)/np.pi)
    t0  = rank_to_t0_ms(df["area_px"])
    dt  = np.maximum(600.0 - t0, 1.0)
    C   = np.clip(np.asarray(df["circularity"], float), 0, 1)
    phi = np.clip(np.asarray(df["defect_frac"], float), 0, 1)

    if not use_penalties:
        v0 = np.median(R/dt)
        return float(v0), 0.0, 0.0

    def resid(params):
        v0, a, b = params
        f   = np.exp(-a*(1.0 - C)) * np.exp(-b*phi)
        pred= v0*np.maximum(f,1e-9)*dt
        return R - pred

    x0 = np.array([np.median(R/dt), 0.5, 0.5])
    bounds = ([1e-6, 0.0, 0.0], [np.inf, 10.0, 10.0])
    try:
        res = optimize.least_squares(resid, x0=x0, bounds=bounds, max_nfev=2000)
        v0, a, b = res.x
    except Exception:
        v0, a, b = x0
    return float(v0), float(a), float(b)

def fit_lognormal(t0_ms):
    t = np.asarray(t0_ms, float)
    t = t[(t>0) & (t<=60.0)]
    if len(t) < 3: return None
    sigma, loc, scale = stats.lognorm.fit(t, floc=0)
    mu = math.log(scale)
    ll = np.sum(stats.lognorm.logpdf(t, s=sigma, loc=0, scale=scale))
    k  = 2
    aic= 2*k - 2*ll
    bic= k*math.log(len(t)) - 2*ll
    return dict(model="lognormal", mu=mu, sigma=sigma, aic=aic, bic=bic, n=len(t))

def fit_gamma(t0_ms):
    t = np.asarray(t0_ms, float)
    t = t[(t>0) & (t<=60.0)]
    if len(t) < 3: return None
    k, loc, theta = stats.gamma.fit(t, floc=0)
    ll = np.sum(stats.gamma.logpdf(t, a=k, loc=0, scale=theta))
    npar=2
    aic= 2*npar - 2*ll
    bic= npar*math.log(len(t)) - 2*ll
    return dict(model="gamma", k=k, theta=theta, aic=aic, bic=bic, n=len(t))

def best_dn_dt_fit(t0_ms):
    g = fit_gamma(t0_ms)
    l = fit_lognormal(t0_ms)
    c = [m for m in (g,l) if m is not None]
    if not c: return None
    c.sort(key=lambda d: (d["bic"], d["aic"]))
    return c[0]

def sample_dn_dt_curve(fit, t_grid_ms):
    if fit is None: return np.zeros_like(t_grid_ms)
    if fit["model"] == "gamma":
        return stats.gamma.pdf(t_grid_ms, a=fit["k"], loc=0, scale=fit["theta"])
    else:
        sigma = fit["sigma"]; scale = math.exp(fit["mu"])
        return stats.lognorm.pdf(t_grid_ms, s=sigma, loc=0, scale=scale)

def build_Xpred(df, v0, alpha, beta, t_grid_ms, A_eff=None):
    A   = np.asarray(df["area_px"], float)
    R   = np.sqrt(np.maximum(A,0.0)/np.pi)
    C   = np.clip(np.asarray(df["circularity"], float), 0, 1)
    phi = np.clip(np.asarray(df["defect_frac"], float), 0, 1)
    t0  = np.asarray(df["t0_ms"], float)
    f   = np.exp(-alpha*(1.0 - C)) * np.exp(-beta*phi)
    X = np.zeros_like(t_grid_ms, dtype=float)
    for i in range(len(df)):
        dt = np.maximum(t_grid_ms - t0[i], 0.0)
        r  = np.minimum(v0*np.maximum(f[i],1e-9)*dt, R[i])
        Ai = np.pi*(r**2)
        X += Ai
    if A_eff is None or A_eff <= 0:
        A_eff = max(np.sum(A), 1.0)
    return X/float(A_eff)

def fit_avrami_to(X_pred, t_grid_ms, n_fixed):
    t = t_grid_ms/1000.0
    def model(t, K): return 1.0 - np.exp(-(K*(t**n_fixed)))
    def resid(K):    return X_pred - model(t, K[0])
    K0 = np.array([1e-3])
    bounds = ([1e-12],[1e+6])
    try:
        res = optimize.least_squares(resid, K0, bounds=bounds, max_nfev=2000)
        K = float(res.x[0])
    except Exception:
        K = float(K0[0])
    return K

# ---------- Plots ----------
def plot_dn_dt_two(t_ms, yA, yB, labelA, labelB, out_png):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    plt.plot(t_ms, yA, label=f"{labelA} dn/dt")
    plt.plot(t_ms, yB, label=f"{labelB} dn/dt")
    plt.xlabel("time (ms)"); plt.ylabel("dn/dt (a.u.)")
    plt.title("Nucleation rate density")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=200); plt.close()

def plot_X_two(t_ms, XA, XB, labelA, labelB, KA, KB, n, out_png):
    import matplotlib.pyplot as plt
    t = t_ms/1000.0
    XA_fit = 1.0 - np.exp(-(KA*(t**n)))
    XB_fit = 1.0 - np.exp(-(KB*(t**n)))
    plt.figure(figsize=(6,4))
    plt.plot(t_ms, XA, label=f"{labelA} X_pred")
    plt.plot(t_ms, XB, label=f"{labelB} X_pred")
    plt.plot(t_ms, XA_fit, "--", label=f"{labelA} Avrami (K={KA:.3g}, n={n})")
    plt.plot(t_ms, XB_fit, "--", label=f"{labelB} Avrami (K={KB:.3g}, n={n})")
    plt.xlabel("time (ms)"); plt.ylabel("X(t) (a.u.)")
    plt.title("Bulk transformed fraction (proxy)")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=200); plt.close()

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folders", nargs=2, required=True)
    ap.add_argument("--labels",  nargs=2, default=["FAPI","FAPI-TEMPO"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--n_avrami", type=float, default=2.5)
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--calibrate_penalties", action="store_true")
    ap.add_argument("--t0_mode", choices=["rank","backcalc"], default="backcalc")
    ap.add_argument("--crystal_ids", nargs="*", type=int, default=[1])
    ap.add_argument("--nucleus_ids", nargs="*", type=int, default=[2])
    ap.add_argument("--defect_ids",  nargs="*", type=int, default=[3])
    ap.add_argument("--crystal_names", nargs="*", default=["crystal","Crystal mask"])
    ap.add_argument("--nucleus_names", nargs="*", default=["nucleus","Nucleus point"])
    ap.add_argument("--defect_names",  nargs="*", default=["defect","Defect"])
    args = ap.parse_args()

    folderA, folderB = args.folders
    labelA,  labelB  = args.labels

    outdir = args.out or str(Path(folderA).parent / "comparative_outputs_backcalc_v2")
    Path(outdir).mkdir(parents=True, exist_ok=True)

    print(f"Analyzing: {folderA}  [{labelA}]")
    dfA = load_coco_many(folderA,
                         set(args.crystal_ids), set(args.nucleus_ids), set(args.defect_ids),
                         args.crystal_names, args.nucleus_names, args.defect_names)
    print(f"[SUMMARY {labelA}] rows={len(dfA)}")

    print(f"Analyzing: {folderB}  [{labelB}]")
    dfB = load_coco_many(folderB,
                         set(args.crystal_ids), set(args.nucleus_ids), set(args.defect_ids),
                         args.crystal_names, args.nucleus_names, args.defect_names)
    print(f"[SUMMARY {labelB}] rows={len(dfB)}")

    if len(dfA)==0 or len(dfB)==0:
        print("One or both datasets are empty after loading/cleaning. Aborting.")
        return

    v0A, aA, bA = calibrate_v0_alpha_beta(dfA, use_penalties=args.calibrate_penalties)
    v0B, aB, bB = calibrate_v0_alpha_beta(dfB, use_penalties=args.calibrate_penalties)

    if args.t0_mode == "rank":
        dfA["t0_ms"] = rank_to_t0_ms(dfA["area_px"])
        dfB["t0_ms"] = rank_to_t0_ms(dfB["area_px"])
    else:
        dfA["t0_ms"] = t0_from_backcalc(dfA, v0A, aA, bA)
        dfB["t0_ms"] = t0_from_backcalc(dfB, v0B, aB, bB)

    # dn/dt fits
    fitA = best_dn_dt_fit(dfA["t0_ms"])
    fitB = best_dn_dt_fit(dfB["t0_ms"])
    t_dn = np.arange(0.0, 60.0+1e-9, 1.0)
    dnA = sample_dn_dt_curve(fitA, t_dn)
    dnB = sample_dn_dt_curve(fitB, t_dn)

    # X_pred, Avrami
    t_ms = np.arange(0.0, 600.0+1e-9, 1.0)
    XA = build_Xpred(dfA, v0A, aA, bA, t_ms, A_eff=None)
    XB = build_Xpred(dfB, v0B, aB, bB, t_ms, A_eff=None)
    KA = fit_avrami_to(XA, t_ms, args.n_avrami)
    KB = fit_avrami_to(XB, t_ms, args.n_avrami)

    # Exports
    pd.DataFrame({"t_ms":t_dn, f"dn_dt_{labelA}":dnA}).to_csv(Path(outdir)/f"dn_dt_{labelA}.csv", index=False)
    pd.DataFrame({"t_ms":t_dn, f"dn_dt_{labelB}":dnB}).to_csv(Path(outdir)/f"dn_dt_{labelB}.csv", index=False)
    pd.DataFrame({"t_ms":t_ms, f"X_pred_{labelA}":XA, f"X_pred_{labelB}":XB}).to_csv(Path(outdir)/"X_pred_both.csv", index=False)

    params_rows = []
    def add_fit(label, fit, v0, a, b, K):
        row = dict(label=label, v0=v0, alpha=a, beta=b, K=K)
        if fit is not None and fit["model"]=="gamma":
            row.update(model="gamma", k=fit["k"], theta=fit["theta"], AIC=fit["aic"], BIC=fit["bic"])
        elif fit is not None and fit["model"]=="lognormal":
            row.update(model="lognormal", mu=fit["mu"], sigma=fit["sigma"], AIC=fit["aic"], BIC=fit["bic"])
        else:
            row.update(model="NA")
        params_rows.append(row)
    add_fit(labelA, fitA, v0A, aA, bA, KA)
    add_fit(labelB, fitB, v0B, aB, bB, KB)
    pd.DataFrame(params_rows).to_csv(Path(outdir)/"model_params.csv", index=False)

    # plots
    plot_dn_dt_two(t_dn, dnA, dnB, labelA, labelB, str(Path(outdir)/"dn_dt_compare.png"))
    plot_X_two(t_ms, XA, XB, labelA, labelB, KA, KB, args.n_avrami, str(Path(outdir)/"X_pred_Avrami_compare.png"))

    keep_cols = ["dataset","image_id","object_id","area_px","perimeter_px","circularity","defect_frac","t0_ms"]
    dfA[keep_cols].to_csv(Path(outdir)/f"objects_{labelA}.csv", index=False)
    dfB[keep_cols].to_csv(Path(outdir)/f"objects_{labelB}.csv", index=False)

    print("=== Done ===")
    print(f"{labelA}: v0={v0A:.4g}, alpha={aA:.3g}, beta={bA:.3g}")
    print(f"{labelB}: v0={v0B:.4g}, alpha={aB:.3g}, beta={bB:.3g}")
    if fitA: print(f"{labelA} dn/dt best: {fitA}")
    if fitB: print(f"{labelB} dn/dt best: {fitB}")
    print(f"Avrami n={args.n_avrami}: K_{labelA}={KA:.4g}, K_{labelB}={KB:.4g}")
    print(f"Outputs in: {outdir}")

if __name__ == "__main__":
    main()
