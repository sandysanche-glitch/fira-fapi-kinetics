import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_dndt_csv(p, time_col=None, rate_col=None):
    df = pd.read_csv(p)
    if time_col is None:
        for c in ["t_ms","time_ms","ms","t"]:
            if c in df.columns:
                time_col = c; break
    if rate_col is None:
        # accept generic or dataset-specific (e.g., dn_dt_FAPI)
        for c in ["dn_dt","dn_dt_total","dn_dt_field","rate","density"]:
            if c in df.columns:
                rate_col = c; break
        if rate_col is None:
            # any column starting with dn_dt_
            cand = [c for c in df.columns if c.startswith("dn_dt")]
            if cand:
                rate_col = cand[0]
    if time_col is None or rate_col is None:
        raise ValueError(f"{p} missing time/rate columns. Found: {list(df.columns)}")
    t_ms = df[time_col].astype(float).to_numpy()
    dn_dt = df[rate_col].astype(float).to_numpy()
    return t_ms, dn_dt, time_col, rate_col

def load_objects_csv(p):
    df = pd.read_csv(p)
    need = ["area_px","t0_ms"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{p} missing columns: {missing}")
    return df

def compute_I_SI(dn_dt_ms, area_eff_px, px_per_um):
    um_per_px = 1.0 / px_per_um
    m_per_px  = um_per_px * 1e-6
    area_eff_m2 = area_eff_px * (m_per_px**2)
    # count/ms -> count/s (×1000), then / m²
    return (dn_dt_ms * 1000.0) / max(area_eff_m2, 1e-30), area_eff_m2

def compute_G_SI_per_object(objects_df, px_per_um):
    um_per_px = 1.0 / px_per_um
    m_per_px  = um_per_px * 1e-6

    A_px = np.asarray(objects_df["area_px"], float)
    t0_ms = np.asarray(objects_df["t0_ms"], float)
    # final radius in pixels
    R_px = np.sqrt(np.maximum(A_px, 0.0) / np.pi)
    # available growth time in seconds
    dt_s = np.maximum((600.0 - t0_ms)/1000.0, 1e-9)
    # m/s
    G_mps = (R_px * m_per_px) / dt_s

    out = objects_df.copy()
    out["R_px"] = R_px
    out["dt_s"] = dt_s
    out["G_m_per_s"] = G_mps
    return out

def summarize_I(t_ms, I_SI):
    # restrict to 0-60 ms for nucleation window summaries
    mask = (t_ms >= 0) & (t_ms <= 60)
    t_s = t_ms[mask] / 1000.0
    Iw = I_SI[mask]
    # integral via trapezoid, count/m^2
    N_per_m2 = np.trapz(Iw, t_s)
    I_peak = np.max(Iw) if Iw.size else np.nan
    I_mean = np.trapz(Iw, t_s) / (t_s[-1]-t_s[0]) if Iw.size > 1 else np.nan
    return dict(I_peak=I_peak, I_mean_0_60ms=I_mean, N_per_m2=N_per_m2)

def summarize_G(objects_with_G):
    g = np.asarray(objects_with_G["G_m_per_s"], float)
    g = g[np.isfinite(g) & (g>=0)]
    if g.size == 0:
        return dict(G_median=np.nan, G_IQR=np.nan, G_mean=np.nan, G_std=np.nan, n=0)
    q25, q50, q75 = np.percentile(g, [25,50,75])
    return dict(G_median=q50, G_IQR=q75-q25, G_mean=float(np.mean(g)), G_std=float(np.std(g)), n=len(g))

def main():
    ap = argparse.ArgumentParser(description="Compute SI nucleation I(t) and growth G from exported CSVs.")
    ap.add_argument("--dir", required=True, help="Folder with dn_dt_<label>.csv and objects_<label>.csv")
    ap.add_argument("--label", required=True, help="Dataset label, e.g., FAPI or FAPI-TEMPO")
    ap.add_argument("--px_per_um", type=float, default=2.20014)
    ap.add_argument("--area_eff_px", type=float, required=True, help="Effective area in pixels")
    ap.add_argument("--outdir", default=None, help="Output folder (default: --dir)")
    ap.add_argument("--time_col", default=None)
    ap.add_argument("--rate_col", default=None)
    args = ap.parse_args()

    base = Path(args.dir)
    out  = Path(args.outdir) if args.outdir else base
    out.mkdir(parents=True, exist_ok=True)

    dn_csv = base / f"dn_dt_{args.label}.csv"
    ob_csv = base / f"objects_{args.label}.csv"

    # --- load
    t_ms, dn_dt, used_t, used_r = load_dndt_csv(dn_csv, args.time_col, args.rate_col)
    obj = load_objects_csv(ob_csv)

    # --- I(t) in SI
    I_SI, Aeff_m2 = compute_I_SI(dn_dt, args.area_eff_px, args.px_per_um)
    I_sum = summarize_I(t_ms, I_SI)

    # --- G per object in SI
    objG = compute_G_SI_per_object(obj, args.px_per_um)
    G_sum = summarize_G(objG)

    # --- save CSVs
    siI = pd.DataFrame({"t_ms": t_ms, "I_count_per_s_per_m2": I_SI})
    siI.to_csv(out / f"I_SI_{args.label}.csv", index=False)
    objG.to_csv(out / f"objects_with_G_{args.label}.csv", index=False)

    # --- quick plots
    plt.figure(figsize=(7,4))
    plt.plot(t_ms, I_SI)
    plt.xlabel("Time (ms)"); plt.ylabel("I (count s$^{-1}$ m$^{-2}$)")
    plt.title(f"I(t) in SI — {args.label}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / f"I_SI_{args.label}.png", dpi=300)

    plt.figure(figsize=(7,4))
    vals = objG["G_m_per_s"].values
    vals = vals[np.isfinite(vals) & (vals>=0)]
    if vals.size > 0:
        plt.hist(vals, bins=50)
    plt.xlabel("G (m s$^{-1}$)"); plt.ylabel("count")
    plt.title(f"Growth speed distribution — {args.label}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / f"G_hist_{args.label}.png", dpi=300)

    # --- print summary
    print(f"\n=== {args.label} ===")
    print(f"Pixels/µm: {args.px_per_um:.5f}  |  A_eff: {args.area_eff_px:.0f} px  ({Aeff_m2:.3e} m²)")
    print(f"I(t): columns used: time='{used_t}', rate='{used_r}'")
    for k,v in I_sum.items():
        print(f"{k}: {v:.3e}")
    for k,v in G_sum.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
