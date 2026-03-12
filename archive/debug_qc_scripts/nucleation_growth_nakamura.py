from pathlib import Path
import sys, json, argparse
import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt

R_gas = 8.314462618
EV_TO_J_PER_MOL = 96485.33212

CFG = {
    "nucleation_window_ms": 60.0,
    "growth_window_ms": 540.0,
    "grid_ms": 1.0,
    "avrami_n": {"FAPI": 2.5, "FAPI-TEMPO": 2.5},
    "arrhenius": {
        "FAPI": {"Ea_eV": 0.45, "k0": 1.0},
        "FAPI-TEMPO": {"Ea_eV": 0.45, "k0": 1.0}
    },
    "bootstrap": 500,
    "min_events": 20
}

def load_metrics(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p, sep=r"\t", engine="python")
    df.rename(columns={c: c.strip() for c in df.columns}, inplace=True)
    for col in ("Dataset","Time","Temperature"):
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    df["Dataset"] = df["Dataset"].astype(str).str.strip()
    df = df[~df["Dataset"].str.lower().isin({"um","μm","units","unit"})]
    df["t_ms"] = df["Time"].astype(float)*1000.0
    df["T_K"]  = df["Temperature"].astype(float)+273.15
    cand_area = [c for c in df.columns if c.lower() in {"area","area_px","area_um2","area (px)","area (µm²)"}]
    df["_AREA_COL_"] = cand_area[0] if cand_area else None
    cand_id = [c for c in df.columns if c.lower() in {"id","objectid","label","grain_id"}]
    df["_ID_COL_"] = cand_id[0] if cand_id else None
    return df

def infer_nucleation_times(dsub: pd.DataFrame, nuc_ms: float) -> np.ndarray:
    d = dsub[(dsub["t_ms"]>=0)&(dsub["t_ms"]<=nuc_ms)].copy()
    if d.empty: return np.array([])
    if d["_AREA_COL_"].notna().all() and d["_ID_COL_"].notna().all():
        area_col = d["_AREA_COL_"].iloc[0]; id_col = d["_ID_COL_"].iloc[0]
        if area_col in d.columns and id_col in d.columns:
            nuc=[]; 
            for oid,g in d.groupby(id_col):
                gg=g.sort_values("t_ms"); 
                a=gg[area_col].astype(float).to_numpy()
                if np.nanmax(a)>0:
                    thr=0.05*np.nanmax(a)
                    idx=np.where(a>thr)[0]
                    if idx.size: nuc.append(gg["t_ms"].iloc[idx[0]])
            return np.array(nuc,float)
    cand_area=[c for c in d.columns if c.lower().startswith("area")]
    if cand_area:
        A=d.groupby("t_ms")[cand_area[0]].sum().sort_index()
        t=A.index.to_numpy()
        if t.size>=3:
            dA=np.gradient(A.to_numpy(),t)
            mask=dA>np.percentile(dA[dA>0],25) if np.any(dA>0) else dA>0
            t_sig=t[mask]
            if t_sig.size:
                cutoff=np.min(t)+0.10*(np.max(t)-np.min(t))
                return t_sig[t_sig<=cutoff]
    return d["t_ms"].dropna().unique()

def fit_lognormal(times):
    x=np.asarray(times,float); x=x[x>0]; 
    if x.size<3: return None
    shape,loc,scale=stats.lognorm.fit(x,floc=0)
    ll=np.sum(stats.lognorm.logpdf(x,shape,loc=loc,scale=scale))
    k=2; aic=2*k-2*ll; bic=k*np.log(x.size)-2*ll
    return {"model":"lognormal","shape":shape,"loc":loc,"scale":scale,"aic":aic,"bic":bic}

def fit_gamma(times):
    x=np.asarray(times,float); x=x[x>0]
    if x.size<3: return None
    a,loc,scale=stats.gamma.fit(x,floc=0)
    ll=np.sum(stats.gamma.logpdf(x,a,loc=loc,scale=scale))
    k=2; aic=2*k-2*ll; bic=k*np.log(x.size)-2*ll
    return {"model":"gamma","a":a,"loc":loc,"scale":scale,"aic":aic,"bic":bic}

def best_model_fit(times):
    m1=fit_lognormal(times); m2=fit_gamma(times)
    if m1 is None: return m2
    if m2 is None: return m1
    return m1 if (m1["bic"]<m2["bic"] or (abs(m1["bic"]-m2["bic"])<1e-9 and m1["aic"]<=m2["aic"])) else m2

def pdf_eval(fit,t):
    t=np.asarray(t,float)
    if fit["model"]=="lognormal":
        return stats.lognorm.pdf(t,fit["shape"],loc=fit["loc"],scale=fit["scale"])
    return stats.gamma.pdf(t,fit["a"],loc=fit["loc"],scale=fit["scale"])

def arrhenius_k(TK,k0,Ea_eV):
    Ea=Ea_eV*EV_TO_J_PER_MOL
    return k0*np.exp(-Ea/(R_gas*TK))

def theta_effective_time(t_ms,TK,k0,Ea_eV):
    t_s=np.asarray(t_ms,float)/1000.0
    TK=np.asarray(TK,float)
    kT=arrhenius_k(TK,k0,Ea_eV)
    t_s=t_s-t_s.min()
    return integrate.cumulative_trapezoid(kT,t_s,initial=0.0)

def jmak_nakamura(theta,n):
    theta=np.asarray(theta,float)
    return 1.0-np.exp(-(theta**n))

def bootstrap_bands(times,fit,grid_ms,B=500,conf=0.95,rng=None):
    rng=np.random.default_rng(rng)
    tgrid=np.arange(0,max(1.0,np.max(times))+grid_ms,grid_ms)
    N=len(times); samps=[]
    for _ in range(B):
        if fit["model"]=="lognormal":
            sim=stats.lognorm.rvs(fit["shape"],loc=fit["loc"],scale=fit["scale"],size=N,random_state=rng)
            f=stats.lognorm.pdf(tgrid,fit["shape"],loc=fit["loc"],scale=fit["scale"])
        else:
            sim=stats.gamma.rvs(fit["a"],loc=fit["loc"],scale=fit["scale"],size=N,random_state=rng)
            f=stats.gamma.pdf(tgrid,fit["a"],loc=fit["loc"],scale=fit["scale"])
        samps.append(f)
    arr=np.vstack(samps); lo=np.quantile(arr,(1-conf)/2.0,axis=0); hi=np.quantile(arr,1-(1-conf)/2.0,axis=0)
    return pd.DataFrame({"t_ms":tgrid,"dn_dt_lo":lo,"dn_dt_hi":hi})

def pick_engine():
    try:
        import xlsxwriter  # noqa
        return "xlsxwriter"
    except Exception:
        return "openpyxl"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--metrics",default="Sts_metrics.txt")
    ap.add_argument("--cfg",default=None)
    ap.add_argument("--outdir",default=".")
    args=ap.parse_args()

    cfg=CFG.copy()
    if args.cfg and Path(args.cfg).exists():
        with open(args.cfg,"r") as fh: user=json.load(fh)
        for k,v in user.items():
            if isinstance(v,dict) and k in cfg: cfg[k].update(v)
            else: cfg[k]=v

    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    df=load_metrics(Path(args.metrics))
    datasets=sorted(df["Dataset"].dropna().unique())
    print("Datasets:",datasets)

    results={}
    for ds in datasets:
        d=df[df["Dataset"]==ds].copy()
        thermal=d[["t_ms","T_K"]].drop_duplicates().sort_values("t_ms")
        t_all=thermal["t_ms"].to_numpy(); T_all=thermal["T_K"].to_numpy()

        t_nuc=infer_nucleation_times(d,cfg["nucleation_window_ms"])
        t_nuc=t_nuc[np.isfinite(t_nuc)]; t_nuc=np.unique(np.round(t_nuc,3))

        fit=None
        if t_nuc.size>=max(3,cfg["min_events"]):
            fit=best_model_fit(t_nuc)

        if fit is not None:
            tgrid=np.arange(0.0,cfg["nucleation_window_ms"]+cfg["grid_ms"],cfg["grid_ms"])
            pdf=pdf_eval(fit,tgrid)
            dn_dt=pd.DataFrame({"t_ms":tgrid,"dn_dt":pdf})
            bands=bootstrap_bands(t_nuc,fit,cfg["grid_ms"],B=cfg["bootstrap"])
        else:
            dn_dt=pd.DataFrame({"t_ms":[], "dn_dt":[]})
            bands=pd.DataFrame({"t_ms":[], "dn_dt_lo":[], "dn_dt_hi":[]})

        pars=cfg["arrhenius"].get(ds,cfg["arrhenius"]["FAPI"])
        n_av=cfg["avrami_n"].get(ds,cfg["avrami_n"]["FAPI"])
        theta=theta_effective_time(t_all,T_all,k0=pars["k0"],Ea_eV=pars["Ea_eV"])
        X=jmak_nakamura(theta,n_av)
        Tmed=np.median(T_all); k_iso=arrhenius_k(Tmed,pars["k0"],pars["Ea_eV"])
        theta_iso=(t_all-t_all.min())/1000.0*k_iso
        X_iso=jmak_nakamura(theta_iso,n_av)

        results[ds]={
            "fit":fit,
            "dn_dt":dn_dt,
            "bands":bands,
            "thermal":pd.DataFrame({"t_ms":t_all,"T_K":T_all}),
            "theta":pd.DataFrame({"t_ms":t_all,"theta_eff_s":theta}),
            "X_t":pd.DataFrame({"t_ms":t_all,"X_nakamura":X,"X_iso_ref":X_iso}),
            "t_nuc":pd.DataFrame({"t_ms":t_nuc})
        }

        dn_dt.to_csv(outdir/f"{ds.lower().replace('-','_')}_dn_dt_fits_1ms.csv",index=False)
        bands.to_csv(outdir/f"{ds.lower().replace('-','_')}_dn_dt_bootstrap_bands_1ms.csv",index=False)
        results[ds]["X_t"].to_csv(outdir/f"{ds.lower().replace('-','_')}_X_t_overlay.csv",index=False)

        if not dn_dt.empty:
            plt.figure(); plt.plot(dn_dt["t_ms"],dn_dt["dn_dt"],label=f"{fit['model']} fit")
            if not bands.empty:
                plt.fill_between(bands["t_ms"],bands["dn_dt_lo"],bands["dn_dt_hi"],alpha=0.3,label="95% band")
            plt.xlabel("Time (ms)"); plt.ylabel("dn/dt (a.u.)"); plt.title(f"{ds}: nucleation rate"); plt.legend(); plt.tight_layout()
            plt.savefig(outdir/f"dn_dt_{ds.replace('-','_')}.png",dpi=200); plt.close()

        plt.figure(); 
        plt.plot(results[ds]["X_t"]["t_ms"],results[ds]["X_t"]["X_nakamura"],label="Nakamura")
        plt.plot(results[ds]["X_t"]["t_ms"],results[ds]["X_t"]["X_iso_ref"],"--",label=f"Isothermal @ {Tmed:.1f} K")
        plt.xlabel("Time (ms)"); plt.ylabel("X(t)"); plt.title(f"{ds}: JMAK overlay (n={n_av})"); plt.legend(); plt.tight_layout()
        plt.savefig(outdir/f"X_t_{ds.replace('-','_')}.png",dpi=200); plt.close()

    engine=("xlsxwriter" if "xlsxwriter" in sys.modules else "openpyxl")
    xlsx=outdir/"nucleation_analysis_workbook.xlsx"
    with pd.ExcelWriter(xlsx,engine=engine) as w:
        wrote=False
        rows=[]
        for ds,pack in results.items():
            if not pack["dn_dt"].empty:
                pack["dn_dt"].to_excel(w,sheet_name=f"{ds}_dn_dt",index=False); wrote=True
                pack["bands"].to_excel(w,sheet_name=f"{ds}_dn_dt_bands",index=False); wrote=True
            pack["X_t"].to_excel(w,sheet_name=f"{ds}_X_t",index=False); wrote=True
            pack["thermal"].to_excel(w,sheet_name=f"{ds}_thermal",index=False); wrote=True
            pack["theta"].to_excel(w,sheet_name=f"{ds}_theta",index=False); wrote=True
            pack["t_nuc"].to_excel(w,sheet_name=f"{ds}_t_nuc",index=False); wrote=True
            if pack["fit"] is not None:
                f=pack["fit"]
                if f["model"]=="lognormal":
                    rows.append({"Dataset":ds,"Model":"lognormal","shape/a":f["shape"],"scale":f["scale"],"AIC":f["aic"],"BIC":f["bic"]})
                else:
                    rows.append({"Dataset":ds,"Model":"gamma","shape/a":f["a"],"scale":f["scale"],"AIC":f["aic"],"BIC":f["bic"]})
        if rows:
            pd.DataFrame(rows).to_excel(w,sheet_name="model_selection",index=False); wrote=True
        if not wrote:
            pd.DataFrame({"info":["no results available"]}).to_excel(w,sheet_name="empty",index=False)

    print("Done.")
    for ds in results:
        tag=ds.lower().replace("-","_")
        print(f" - {tag}_dn_dt_fits_1ms.csv")
        print(f" - {tag}_dn_dt_bootstrap_bands_1ms.csv")
        print(f" - {tag}_X_t_overlay.csv")
        print(f" - dn_dt_{ds.replace('-','_')}.png")
        print(f" - X_t_{ds.replace('-','_')}.png")
    print(" - nucleation_analysis_workbook.xlsx")

if __name__=="__main__":
    main()
