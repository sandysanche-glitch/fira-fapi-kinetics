import argparse, json, math, os, glob, io
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats, optimize
import matplotlib.pyplot as plt

CRYSTAL_DEFAULT_IDS = {1}
NUCLEUS_DEFAULT_IDS = {2}
DEFECT_DEFAULT_IDS  = {3}

# -------------- Robust JSON reader --------------
def read_json_flexible(p: str):
    """
    Returns a tuple (images, annotations):
      - If file is a valid COCO dict: (data['images'], data['annotations'])
      - If file is a list of annotations: ([], list)
      - If file is JSON-Lines: parses each line; if COCO dict lines, merges;
        if annotation lines, returns combined list.
    """
    path = Path(p)
    txt  = path.read_text(encoding="utf-8").strip()
    if not txt:
        return [], []

    # Try normal JSON first
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        # Try JSON-Lines
        images, anns = [], []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict) and ("annotations" in obj or "images" in obj):
                    images.extend(obj.get("images", []))
                    anns.extend(obj.get("annotations", []))
                elif isinstance(obj, dict):
                    # assume it's a single annotation-like record
                    anns.append(obj)
                elif isinstance(obj, list):
                    for el in obj:
                        if isinstance(el, dict):
                            anns.append(el)
        return images, anns

    # If we parsed JSON successfully:
    if isinstance(data, dict):
        return data.get("images", []) or [], data.get("annotations", []) or []
    elif isinstance(data, list):
        # It's a flat list of annotations
        return [], data
    else:
        return [], []

# -------------- Helpers --------------
def is_number(x):
    try:
        float(x); return True
    except Exception:
        return False

def safe_get(d, keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d

# -------------- Single-file loader --------------
def load_one_coco(json_path,
                  crystal_ids,
                  nucleus_ids,
                  defect_ids,
                  crystal_names=None,
                  nucleus_names=None,
                  defect_names=None):
    """
    Build tidy DataFrame with per-object snapshot:
      dataset, image_id, object_id, area_px, perimeter_px,
      circularity, defect_area_px, defect_frac, nucleus_x, nucleus_y
    Compatible with COCO dicts OR plain lists of annotations OR JSON-Lines.
    """
    images, anns = read_json_flexible(json_path)
    dataset_label = Path(json_path).stem

    # image sizes (optional)
    img_size = {}
    for im in images:
        iid = im.get("id")
        W   = safe_get(im, ["width"], None)
        H   = safe_get(im, ["height"], None)
        img_size[iid] = (W, H)

    rows = []
    nuclei_points = []

    def norm_num(val):
        if is_number(val): return float(val)
        return None

    # Accept per-annotation dicts (possibly not strictly COCO)
    for a in anns:
        if not isinstance(a, dict): 
            continue

        cid = a.get("category_id")
        # category name fallback (optional)
        cname = a.get("category_name", a.get("name", None))
        if cid is None and cname is not None:
            # If your files rely on names, you could map them here
            pass

        iid = a.get("image_id", 0)  # default 0 if missing
        oid = a.get("id", None)

        area = a.get("area", None)
        if area is None:
            area = safe_get(a, ["attributes", "area"], None)
        area = norm_num(area)

        perim = a.get("perimeter", None)
        if perim is None:
            perim = safe_get(a, ["attributes", "perimeter"], None)
        perim = norm_num(perim)

        # Decide type
        if cid in crystal_ids:
            rows.append(dict(
                dataset=dataset_label,
                image_id=iid,
                object_id=oid,
                area_px=area if area is not None else np.nan,
                perimeter_px=perim if perim is not None else np.nan,
                is_crystal=True,
                is_defect=False
            ))
        elif cid in defect_ids:
            rows.append(dict(
                dataset=dataset_label,
                image_id=iid,
                object_id=oid,
                area_px=area if area is not None else np.nan,
                perimeter_px=perim if perim is not None else np.nan,
                is_crystal=False,
                is_defect=True
            ))
        elif cid in nucleus_ids:
            # Extract a point (x,y) from common fields
            x = y = None
            kp = a.get("keypoints")
            if isinstance(kp, (list, tuple)) and len(kp) >= 2:
                x, y = kp[0], kp[1]
            if (x is None or y is None) and "point" in a:
                pt = a.get("point")
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    x, y = pt[0], pt[1]
            if (x is None or y is None) and "bbox" in a:
                bx, by, bw, bh = a["bbox"]
                x = bx + bw/2.0
                y = by + bh/2.0
            if x is not None and y is not None:
                nuclei_points.append((dataset_label, iid, float(x), float(y)))

    if not rows:
        return pd.DataFrame(columns=[
            "dataset","image_id","object_id","area_px","perimeter_px",
            "circularity","defect_area_px","defect_frac","nucleus_x","nucleus_y"
        ])

    df = pd.DataFrame(rows)
    crystals = df[df["is_crystal"]].copy()
    defects  = df[df["is_defect"]].copy()

    # Circularity from area and perimeter
    circ = np.full(len(crystals), np.nan, dtype=float)
    if "perimeter_px" in crystals and "area_px" in crystals:
        A = crystals["area_px"].to_numpy(float)
        P = crystals["perimeter_px"].to_numpy(float)
        valid = (A > 0) & (P > 0)
        circ[valid] = (4.0*np.pi*A[valid])/(P[valid]**2)
    crystals["circularity"] = circ

    # If per-object defect area is carried in attributes, you can add it similarly.
    crystals["defect_area_px"] = 0.0
    crystals["defect_frac"]    = 0.0

    # Nucleus points: map per image (fallback = mean nucleus per image)
    crystals["nucleus_x"] = np.nan
    crystals["nucleus_y"] = np.nan
    if nuclei_points:
        nu = pd.DataFrame(nuclei_points, columns=["dataset","image_id","x","y"])
        for iid, grp in crystals.groupby("image_id"):
            pts = nu[nu["image_id"] == iid]
            if len(pts) == 0:
                continue
            cx, cy = pts["x"].mean(), pts["y"].mean()
            crystals.loc[grp.index, "nucleus_x"] = cx
            crystals.loc[grp.index, "nucleus_y"] = cy

    keep = crystals[[
        "dataset","image_id","object_id","area_px","perimeter_px",
        "circularity","defect_area_px","defect_frac","nucleus_x","nucleus_y"
    ]].copy()
    return keep

def load_coco_many(folder, crystal_ids, nucleus_ids, defect_ids,
                   crystal_names, nucleus_names, defect_names):
    jsons = sorted(glob.glob(str(Path(folder) / "*.json")))
    frames = []
    for jp in jsons:
        df = load_one_coco(jp, crystal_ids, nucleus_ids, defect_ids,
                           crystal_names, nucleus_names, defect_names)
        if len(df):
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=[
            "dataset","image_id","object_id","area_px","perimeter_px",
            "circularity","defect_area_px","defect_frac","nucleus_x","nucleus_y"
        ])
    df = pd.concat(frames, ignore_index=True)
    df = df.replace([np.inf,-np.inf], np.nan)
    df = df.dropna(subset=["area_px"])
    df["circularity"] = np.clip(df["circularity"].fillna(1.0), 0.0, 1.0)
    df["defect_frac"] = np.clip(df["defect_frac"].fillna(0.0), 0.0, 1.0)
    return df

# -------------- Kinetics & fitting --------------
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
    dt  = R/np.maximum(v0*f, 1e-9)
    t0  = 600.0 - dt
    return np.clip(t0, 0.0, 60.0)

def t0_from_nucleus_ray(df, v0, alpha, beta):
    if "ray_edge_px" not in df.columns or df["ray_edge_px"].isna().all():
        return t0_from_backcalc(df, v0, alpha, beta)
    ray = np.asarray(df["ray_edge_px"], float)
    C   = np.clip(np.asarray(df["circularity"], float), 0, 1)
    phi = np.clip(np.asarray(df["defect_frac"], float), 0, 1)
    f   = np.exp(-alpha*(1.0 - C)) * np.exp(-beta*phi)
    dt  = ray/np.maximum(v0*f, 1e-9)
    t0  = 600.0 - dt
    return np.clip(t0, 0.0, 60.0)

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
    candidates = [m for m in (g,l) if m is not None]
    if not candidates: return None
    candidates.sort(key=lambda d: (d["bic"], d["aic"]))
    return candidates[0]

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

# -------------- Plots --------------
def plot_dn_dt_two(t_ms, yA, yB, labelA, labelB, out_png):
    plt.figure(figsize=(6,4))
    plt.plot(t_ms, yA, label=f"{labelA} dn/dt")
    plt.plot(t_ms, yB, label=f"{labelB} dn/dt")
    plt.xlabel("time (ms)"); plt.ylabel("dn/dt (a.u.)")
    plt.title("Nucleation rate density")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=200); plt.close()

def plot_X_two(t_ms, XA, XB, labelA, labelB, KA, KB, n, out_png):
    t = t_ms/1000.0
    XA_fit = 1.0 - np.exp(-(KA*(t**n)))
    XB_fit = 1.0 - np.exp(-(KB*(t**n)))
    plt.figure(figsize=(6,4))
    plt.plot(t_ms, XA, label=f"{labelA} X_pred")
    plt.plot(t_ms, XB, label=f"{labelB} X_pred")
    plt.plot(t_ms, XA_fit, "--", label=f"{labelA} Avrami fit (K={KA:.3g}, n={n})")
    plt.plot(t_ms, XB_fit, "--", label=f"{labelB} Avrami fit (K={KB:.3g}, n={n})")
    plt.xlabel("time (ms)"); plt.ylabel("X(t) (a.u.)")
    plt.title("Bulk transformed fraction (proxy)")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=200); plt.close()

# -------------- Main --------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folders", nargs=2, required=True, help="Two folders with COCO JSONs")
    ap.add_argument("--labels",  nargs=2, default=["FAPI","FAPI-TEMPO"])
    ap.add_argument("--out", default=None, help="Output folder; default next to folders")
    ap.add_argument("--n_avrami", type=float, default=2.5)
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--calibrate_penalties", action="store_true")
    ap.add_argument("--t0_mode", choices=["rank","backcalc","nucleus_ray"], default="backcalc")
    ap.add_argument("--crystal_ids", nargs="*", type=int, default=list(CRYSTAL_DEFAULT_IDS))
    ap.add_argument("--nucleus_ids", nargs="*", type=int, default=list(NUCLEUS_DEFAULT_IDS))
    ap.add_argument("--defect_ids",  nargs="*", type=int, default=list(DEFECT_DEFAULT_IDS))
    ap.add_argument("--crystal_names", nargs="*", default=None)
    ap.add_argument("--nucleus_names", nargs="*", default=None)
    ap.add_argument("--defect_names",  nargs="*", default=None)
    args = ap.parse_args()

    folderA, folderB = args.folders
    labelA,  labelB  = args.labels

    outdir = args.out
    if outdir is None:
        outdir = str(Path(folderA).parent / "comparative_outputs_backcalc")
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
    elif args.t0_mode == "backcalc":
        dfA["t0_ms"] = t0_from_backcalc(dfA, v0A, aA, bA)
        dfB["t0_ms"] = t0_from_backcalc(dfB, v0B, aB, bB)
    else:
        dfA["t0_ms"] = t0_from_nucleus_ray(dfA, v0A, aA, bA)
        dfB["t0_ms"] = t0_from_nucleus_ray(dfB, v0B, aB, bB)

    fitA = best_dn_dt_fit(dfA["t0_ms"])
    fitB = best_dn_dt_fit(dfB["t0_ms"])
    t_dn = np.arange(0.0, 60.0+1e-9, 1.0)
    dnA = sample_dn_dt_curve(fitA, t_dn)
    dnB = sample_dn_dt_curve(fitB, t_dn)

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

    plot_dn_dt_two(t_dn, dnA, dnB, labelA, labelB, str(Path(outdir)/"dn_dt_compare.png"))
    plot_X_two(t_ms, XA, XB, labelA, labelB, KA, KB, args.n_avrami, str(Path(outdir)/"X_pred_Avrami_compare.png"))

    keep_cols = ["dataset","image_id","object_id","area_px","circularity","defect_frac","t0_ms"]
    (dfA[keep_cols]).to_csv(Path(outdir)/f"objects_{labelA}.csv", index=False)
    (dfB[keep_cols]).to_csv(Path(outdir)/f"objects_{labelB}.csv", index=False)

    print("=== Done ===")
    print(f"v0, alpha, beta: {labelA} = {v0A:.4g}, {aA:.3g}, {bA:.3g} | {labelB} = {v0B:.4g}, {aB:.3g}, {bB:.3g}")
    if fitA: print(f"{labelA} dn/dt best: {fitA}")
    if fitB: print(f"{labelB} dn/dt best: {fitB}")
    print(f"Avrami n={args.n_avrami}: K_{labelA}={KA:.4g}, K_{labelB}={KB:.4g}")
    print(f"Outputs in: {outdir}")

if __name__ == "__main__":
    main()
