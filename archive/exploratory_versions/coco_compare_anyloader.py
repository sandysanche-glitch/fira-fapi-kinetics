import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import optimize, stats
import matplotlib.pyplot as plt

# ----------------- Robust readers -----------------
def read_flexible(p: Path):
    txt = p.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return {}
    try:
        return json.loads(txt)
    except Exception:
        # JSONL fallback
        out = []
        with p.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line=line.strip()
                if not line: 
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
        return {"__jsonl__": out}

def to_float(x, default=np.nan):
    try:
        # strings like "123", "123.4" -> float; arrays/lists -> fail to default
        if isinstance(x, (list, tuple, dict)):
            return default
        return float(x)
    except Exception:
        return default

def polygon_area_perimeter(seg):
    xs = seg[0::2]; ys = seg[1::2]
    n  = len(xs)
    if n < 3: 
        return np.nan, np.nan
    area = 0.0; per = 0.0
    for i in range(n):
        j = (i+1)%n
        area += xs[i]*ys[j] - xs[j]*ys[i]
        dx = xs[j]-xs[i]; dy = ys[j]-ys[i]
        per += (dx*dx + dy*dy)**0.5
    return abs(area)/2.0, per

def rle_counts_to_area(counts):
    # We only handle uncompressed counts (list of runs). If it's a string (compressed),
    # we cannot decode without pycocotools; return NaN to skip.
    if isinstance(counts, (list,tuple)):
        area = 0
        for i, run in enumerate(counts):
            # standard convention alternates background/foreground; if different, this is still a rough proxy
            try:
                r = int(run)
            except Exception:
                return np.nan
            if i % 2 == 1:
                area += r
        return float(area)
    return np.nan

def derive_area_perim_from_ann(a):
    """
    Try in order:
      1) use numeric 'area'/'perimeter' if present
      2) polygons -> area, perimeter
      3) uncompressed RLE counts -> area (perimeter NaN)
      4) bbox -> area= w*h, perimeter ~ 2*(w+h)
    Return (area_float, perim_float); any missing as NaN.
    """
    area  = to_float(a.get("area"), np.nan)
    perim = to_float(a.get("perimeter"), np.nan)

    seg = a.get("segmentation")
    if (not np.isfinite(area)) or (not np.isfinite(perim)):
        # polygons?
        if isinstance(seg, list) and seg:
            if isinstance(seg[0], list):  # multiple polys
                A_sum = 0.0; P_sum = 0.0
                ok = False
                for poly in seg:
                    if isinstance(poly, list) and len(poly) >= 6:
                        a_i, p_i = polygon_area_perimeter(poly)
                        if np.isfinite(a_i): A_sum += a_i; ok=True
                        if np.isfinite(p_i): P_sum += p_i
                if not np.isfinite(area)  and ok: area  = A_sum
                if not np.isfinite(perim) and ok: perim = P_sum if P_sum>0 else np.nan
            elif isinstance(seg[0], (int,float)) and len(seg) >= 6:
                a_i, p_i = polygon_area_perimeter(seg)
                if not np.isfinite(area):  area  = a_i
                if not np.isfinite(perim): perim = p_i

        # RLE dict (uncompressed counts only)
        if (not np.isfinite(area)) and isinstance(seg, dict) and "counts" in seg:
            area_from_rle = rle_counts_to_area(seg["counts"])
            if np.isfinite(area_from_rle):
                area = area_from_rle
                # perimeter remains NaN (unknown from counts-only without decode)

    # bbox fallback if still NaN
    if (not np.isfinite(area)) or (not np.isfinite(perim)):
        bbox = a.get("bbox")
        if isinstance(bbox, (list,tuple)) and len(bbox) >= 4:
            _, _, bw, bh = bbox[0], bbox[1], bbox[2], bbox[3]
            bwf = to_float(bw, np.nan); bhf = to_float(bh, np.nan)
            if np.isfinite(bwf) and np.isfinite(bhf):
                if not np.isfinite(area):  area  = bwf*bhf
                if not np.isfinite(perim): perim = 2.0*(bwf+bhf)

    # final numeric cast (ensure floats or NaN)
    area  = to_float(area,  np.nan)
    perim = to_float(perim, np.nan)
    return area, perim

def extract_point(a):
    # try standard spots for a nucleus point
    if "keypoints" in a and isinstance(a["keypoints"], (list,tuple)) and len(a["keypoints"])>=2:
        return to_float(a["keypoints"][0], np.nan), to_float(a["keypoints"][1], np.nan)
    if "point" in a and isinstance(a["point"], (list,tuple)) and len(a["point"])>=2:
        return to_float(a["point"][0], np.nan), to_float(a["point"][1], np.nan)
    if "bbox" in a and isinstance(a["bbox"], (list,tuple)) and len(a["bbox"])>=4:
        bx,by,bw,bh = a["bbox"]
        bx,by,bw,bh = to_float(bx,np.nan), to_float(by,np.nan), to_float(bw,np.nan), to_float(bh,np.nan)
        if np.isfinite(bx) and np.isfinite(by) and np.isfinite(bw) and np.isfinite(bh):
            return bx + bw/2.0, by + bh/2.0
    return (np.nan, np.nan)

def load_folder_any(folder: str):
    rows = []
    nuc_by_image = {}
    defects_by_image = {}
    n_json = 0
    skipped_non_numeric = 0

    for jp in sorted(Path(folder).glob("*.json")):
        n_json += 1
        data = read_flexible(jp)
        if isinstance(data, dict) and "annotations" in data:
            anns = data["annotations"]
        elif isinstance(data, dict) and "__jsonl__" in data:
            anns = []
            for d in data["__jsonl__"]:
                if isinstance(d, dict) and "annotations" in d:
                    anns += d["annotations"]
                elif isinstance(d, dict):
                    anns.append(d)
        elif isinstance(data, list):
            anns = [x for x in data if isinstance(x, dict)]
        else:
            anns = []

        for a in anns:
            if not isinstance(a, dict): 
                continue
            iid = a.get("image_id", 0)
            nm  = str(a.get("category_name") or a.get("name") or "").lower()
            cid = a.get("category_id")

            # Heuristics (relaxed): infer roles
            is_crystal = ("crystal" in nm) or (cid==1) or ("mask" in nm and "nucleus" not in nm and "defect" not in nm)
            is_defect  = ("defect" in nm)  or (cid==3)
            is_nucleus = ("nucleus" in nm) or (cid==2)

            # Try to derive area/perimeter
            area, perim = derive_area_perim_from_ann(a)

            # If no explicit role matched, fallback by area magnitude
            if not (is_crystal or is_defect or is_nucleus):
                if np.isfinite(area):
                    is_crystal = (area >= 50.0)
                    is_nucleus = not is_crystal

            # Now route
            if is_crystal:
                if not np.isfinite(area):
                    skipped_non_numeric += 1
                    continue
                rows.append(dict(
                    dataset=Path(folder).name,
                    jsonfile=jp.name,
                    image_id=iid,
                    object_id=a.get("id"),
                    area_px=area,
                    perimeter_px=perim if np.isfinite(perim) else np.nan
                ))
            elif is_defect:
                if np.isfinite(area):
                    defects_by_image.setdefault(iid, 0.0)
                    defects_by_image[iid] += area
            elif is_nucleus:
                x,y = extract_point(a)
                if np.isfinite(x) and np.isfinite(y):
                    nuc_by_image.setdefault(iid, []).append((x,y))

    if not rows:
        return pd.DataFrame(), n_json

    df = pd.DataFrame(rows)

    # circularity
    A = df["area_px"].astype(float).to_numpy()
    P = df["perimeter_px"].astype(float).to_numpy()
    circ = np.ones_like(A)
    ok = (A>0) & (P>0) & np.isfinite(P)
    circ[ok] = (4.0*np.pi*A[ok])/(P[ok]**2)
    df["circularity"] = np.clip(circ, 0, 1)

    # defect fraction per image (coarse heuristic)
    df["defect_frac"] = 0.0
    if defects_by_image:
        area_by_image = df.groupby("image_id")["area_px"].sum()
        for iid, tot_def in defects_by_image.items():
            tot_area = float(area_by_image.get(iid, 0.0))
            if tot_area > 0:
                phi = np.clip(tot_def/tot_area, 0.0, 1.0)
                df.loc[df["image_id"]==iid, "defect_frac"] = phi

    # nucleus coords: assign mean per image (fallback)
    df["nucleus_x"] = np.nan; df["nucleus_y"] = np.nan
    for iid, pts in nuc_by_image.items():
        if len(pts):
            xs, ys = zip(*pts)
            df.loc[df["image_id"]==iid, "nucleus_x"] = float(np.mean(xs))
            df.loc[df["image_id"]==iid, "nucleus_y"] = float(np.mean(ys))

    # final cleaning: require numeric area
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["area_px"])
    print(f"[loader] {folder}: kept rows={len(df)}  skipped_non_numeric={skipped_non_numeric}")
    return df, n_json

# ----------------- Kinetics (same as before) -----------------
def rank_to_t0_ms(area_px):
    area = np.asarray(area_px, float)
    order = np.argsort(area)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(area))
    q = ranks/(max(len(area)-1,1))
    return 60.0*q

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

    x0 = np.array([np.median(R/dt), 0.2, 0.2])
    bounds = ([1e-8, 0.0, 0.0], [1e3, 10.0, 10.0])
    try:
        res = optimize.least_squares(resid, x0=x0, bounds=bounds, max_nfev=2000)
        v0, a, b = res.x
    except Exception:
        v0, a, b = x0
    return float(v0), float(a), float(b)

def t0_from_backcalc(df, v0, alpha, beta):
    A   = np.asarray(df["area_px"], float)
    R   = np.sqrt(np.maximum(A,0.0)/np.pi)
    C   = np.clip(np.asarray(df["circularity"], float), 0, 1)
    phi = np.clip(np.asarray(df["defect_frac"], float), 0, 1)
    f   = np.exp(-alpha*(1.0 - C)) * np.exp(-beta*phi)
    dt  = R/np.maximum(v0*np.maximum(f,1e-9), 1e-9)
    return np.clip(600.0 - dt, 0.0, 60.0)

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

def best_dn_dt_fit(t0_ms):
    cands = [m for m in (fit_gamma(t0_ms), fit_lognormal(t0_ms)) if m is not None]
    if not cands: return None
    cands.sort(key=lambda d: (d["bic"], d["aic"]))
    return cands[0]

def sample_dn_dt_curve(fit, t_grid_ms):
    if fit is None:
        return np.zeros_like(t_grid_ms)
    if fit["model"]=="gamma":
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
    if not A_eff or A_eff <= 0:
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

# ----------------- Plot helpers -----------------
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
    plt.plot(t_ms, XA_fit, "--", label=f"{labelA} Avrami (K={KA:.3g}, n={n})")
    plt.plot(t_ms, XB_fit, "--", label=f"{labelB} Avrami (K={KB:.3g}, n={n})")
    plt.xlabel("time (ms)"); plt.ylabel("X(t) (a.u.)")
    plt.title("Bulk transformed fraction (proxy)")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=200); plt.close()

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folders", nargs=2, required=True)
    ap.add_argument("--labels",  nargs=2, default=["FAPI","FAPI-TEMPO"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--n_avrami", type=float, default=2.5)
    ap.add_argument("--t0_mode", choices=["rank","backcalc"], default="backcalc")
    ap.add_argument("--calibrate_penalties", action="store_true")
    args = ap.parse_args()

    folderA, folderB = args.folders
    labelA,  labelB  = args.labels
    outdir = args.out or str(Path(folderA).parent / "comparative_outputs_any")
    Path(outdir).mkdir(parents=True, exist_ok=True)

    print(f"Analyzing: {folderA}  [{labelA}]")
    dfA, nA = load_folder_any(folderA)
    print(f"[SUMMARY {labelA}] jsons={nA} rows={len(dfA)}")

    print(f"Analyzing: {folderB}  [{labelB}]")
    dfB, nB = load_folder_any(folderB)
    print(f"[SUMMARY {labelB}] jsons={nB} rows={len(dfB)}")

    if len(dfA)==0 or len(dfB)==0:
        print("One or both datasets empty after tolerant loading. Aborting.")
        return

    # Calibrate penalties & base speeds (dataset-specific)
    v0A, aA, bA = calibrate_v0_alpha_beta(dfA, use_penalties=args.calibrate_penalties)
    v0B, aB, bB = calibrate_v0_alpha_beta(dfB, use_penalties=args.calibrate_penalties)

    # t0 from chosen mode
    if args.t0_mode == "rank":
        dfA["t0_ms"] = rank_to_t0_ms(dfA["area_px"])
        dfB["t0_ms"] = rank_to_t0_ms(dfB["area_px"])
    else:
        dfA["t0_ms"] = t0_from_backcalc(dfA, v0A, aA, bA)
        dfB["t0_ms"] = t0_from_backcalc(dfB, v0B, aB, bB)

    # dn/dt and Avrami
    t_dn = np.arange(0.0, 60.0+1, 1.0)
    fitA = best_dn_dt_fit(dfA["t0_ms"])
    fitB = best_dn_dt_fit(dfB["t0_ms"])
    dnA = sample_dn_dt_curve(fitA, t_dn)
    dnB = sample_dn_dt_curve(fitB, t_dn)

    t_ms = np.arange(0.0, 600.0+1, 1.0)
    XA = build_Xpred(dfA, v0A, aA, bA, t_ms, A_eff=None)
    XB = build_Xpred(dfB, v0B, aB, bB, t_ms, A_eff=None)
    KA = fit_avrami_to(XA, t_ms, args.n_avrami)
    KB = fit_avrami_to(XB, t_ms, args.n_avrami)

    # Exports
    Path(outdir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"t_ms":t_dn, f"dn_dt_{labelA}":dnA}).to_csv(Path(outdir)/f"dn_dt_{labelA}.csv", index=False)
    pd.DataFrame({"t_ms":t_dn, f"dn_dt_{labelB}":dnB}).to_csv(Path(outdir)/f"dn_dt_{labelB}.csv", index=False)
    pd.DataFrame({"t_ms":t_ms, f"X_pred_{labelA}":XA, f"X_pred_{labelB}":XB}).to_csv(Path(outdir)/"X_pred_both.csv", index=False)

    params = []
    def add_params(label, fit, v0, a, b, K):
        row = dict(label=label, v0=v0, alpha=a, beta=b, K=K)
        if fit is not None and fit["model"]=="gamma":
            row.update(model="gamma", k=fit["k"], theta=fit["theta"], AIC=fit["aic"], BIC=fit["bic"])
        elif fit is not None and fit["model"]=="lognormal":
            row.update(model="lognormal", mu=fit["mu"], sigma=fit["sigma"], AIC=fit["aic"], BIC=fit["bic"])
        else:
            row.update(model="NA")
        params.append(row)
    add_params(labelA, fitA, v0A, aA, bA, KA)
    add_params(labelB, fitB, v0B, aB, bB, KB)
    pd.DataFrame(params).to_csv(Path(outdir)/"model_params.csv", index=False)

    # Plots
    plot_dn_dt_two(t_dn, dnA, dnB, labelA, labelB, str(Path(outdir)/"dn_dt_compare.png"))
    plot_X_two(t_ms, XA, XB, labelA, labelB, KA, KB, args.n_avrami, str(Path(outdir)/"X_pred_Avrami_compare.png"))

    # object tables
    keep = ["dataset","jsonfile","image_id","object_id","area_px","perimeter_px","circularity","defect_frac","nucleus_x","nucleus_y","t0_ms"]
    dfA[keep].to_csv(Path(outdir)/f"objects_{labelA}.csv", index=False)
    dfB[keep].to_csv(Path(outdir)/f"objects_{labelB}.csv", index=False)

    print("=== Done ===")
    print(f"{labelA}: v0={v0A:.4g}, alpha={aA:.3g}, beta={bA:.3g}")
    print(f"{labelB}: v0={v0B:.4g}, alpha={aB:.3g}, beta={bB:.3g}")
    if fitA: print(f"{labelA} dn/dt best: {fitA}")
    if fitB: print(f"{labelB} dn/dt best: {fitB}")
    print(f"Avrami n={args.n_avrami}: K_{labelA}={KA:.4g}, K_{labelB}={KB:.4g}")
    print(f"Outputs in: {outdir}")

if __name__=="__main__":
    main()
