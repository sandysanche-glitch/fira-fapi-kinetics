# decouple_nucleation_growth.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------- user settings --------
STS_PATH = Path("Sts_metrics.txt")  # change if needed
TOTAL_MS = 600.0                    # assumed total solidification window
BIN_MS   = 10.0                     # histogram/bin width for time
# --------------------------------

def load_sts_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\t", engine="python")
    cols = list(df.columns)

    def pick(patterns):
        out = {}
        for key, pats in patterns.items():
            found = None
            for c in cols:
                c_norm = c.replace(" ", "").lower()
                for p in pats:
                    if p in c_norm:
                        found = c
                        break
                if found: break
            out[key] = found
        return out

    # Left block (FAPI-ish)
    left_map = pick({
        "file":["file_name"],
        "dataset":["dataset"],
        "area":["fapi_area_"],
        "perim":["fapi_perimeter"],
        "cd":["cd"],
        "entropy":["fapi_entropy"],
    })
    # Right block (FAPI-TEMPO-ish)
    right_map = pick({
        "file":["file_name.1","filename.1"],
        "dataset":["dataset.1"],
        "area":["fapitempo_area_","fapi tempo_area_"],
        "perim":["fapitempo_perimeter","fapi tempo_perimeter"],
        "cd":["cd.1"],
        "entropy":["fapitempo_entropy","fapi tempo_entropy"],
    })

    parts = []
    if left_map["dataset"] and left_map["area"]:
        ldf = pd.DataFrame({
            "SampleID": df[left_map["file"]] if left_map["file"] else None,
            "Dataset": df[left_map["dataset"]],
            "Area_um2": pd.to_numeric(df[left_map["area"]], errors="coerce"),
            "Perimeter_um": pd.to_numeric(df[left_map["perim"]], errors="coerce") if left_map["perim"] else np.nan,
            "CD": pd.to_numeric(df[left_map["cd"]], errors="coerce") if left_map["cd"] else np.nan,
            "Entropy": pd.to_numeric(df[left_map["entropy"]], errors="coerce") if left_map["entropy"] else np.nan,
        })
        parts.append(ldf)
    if right_map["dataset"] and right_map["area"]:
        rdf = pd.DataFrame({
            "SampleID": df[right_map["file"]] if right_map["file"] else None,
            "Dataset": df[right_map["dataset"]],
            "Area_um2": pd.to_numeric(df[right_map["area"]], errors="coerce"),
            "Perimeter_um": pd.to_numeric(df[right_map["perim"]], errors="coerce") if right_map["perim"] else np.nan,
            "CD": pd.to_numeric(df[right_map["cd"]], errors="coerce") if right_map["cd"] else np.nan,
            "Entropy": pd.to_numeric(df[right_map["entropy"]], errors="coerce") if right_map["entropy"] else np.nan,
        })
        parts.append(rdf)

    tidy = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["SampleID","Dataset","Area_um2","Perimeter_um","CD","Entropy"])
    tidy = tidy.dropna(subset=["Dataset","Area_um2"])
    tidy["Dataset"] = tidy["Dataset"].astype(str).str.strip().replace({
        "FAPI TEMPO":"FAPI-TEMPO",
        "FAPI_TEMPO":"FAPI-TEMPO",
        "FAPI TEMPO_":"FAPI-TEMPO"
    })
    return tidy

def map_to_time(df: pd.DataFrame, total_ms: float) -> pd.DataFrame:
    w = df.copy()
    # Largest areas interpreted as earliest pseudo-time
    w["area_rank_desc"] = w["Area_um2"].rank(method="average", ascending=False, pct=True)
    w["t_ms"] = (1.0 - w["area_rank_desc"]) * total_ms
    w["radius_um"] = np.sqrt(w["Area_um2"] / np.pi)
    return w

def estimate_dn_dt(t_ms: np.ndarray, bin_ms: float, total_ms: float):
    if t_ms.size == 0:
        return np.array([]), np.array([])
    edges = np.arange(0, max(total_ms, float(np.nanmax(t_ms))) + bin_ms, bin_ms)
    hist, edges = np.histogram(t_ms, bins=edges)
    centers = 0.5*(edges[:-1] + edges[1:])
    dn_dt = hist / bin_ms  # counts per ms
    # light smoothing
    if dn_dt.size >= 3:
        dn_dt = pd.Series(dn_dt).rolling(3, center=True, min_periods=1).mean().to_numpy()
    return centers, dn_dt

def estimate_growth_rate(t_ms: np.ndarray, r_um: np.ndarray, bin_ms: float, total_ms: float):
    if t_ms.size == 0:
        return np.array([]), np.array([]), np.array([])
    edges = np.arange(0, max(total_ms, float(np.nanmax(t_ms))) + bin_ms, bin_ms)
    idx = np.digitize(t_ms, edges) - 1
    med_r, centers = [], []
    for b in range(len(edges)-1):
        sel = r_um[idx==b]
        if sel.size:
            med_r.append(np.median(sel))
            centers.append(0.5*(edges[b]+edges[b+1]))
    med_r = np.asarray(med_r, float)
    centers = np.asarray(centers, float)
    if centers.size >= 2:
        drdt = np.gradient(med_r, centers)  # um per ms
        if drdt.size >= 3:
            drdt = pd.Series(drdt).rolling(3, center=True, min_periods=1).mean().to_numpy()
    else:
        drdt = np.zeros_like(med_r)
    return centers, med_r, drdt

def main():
    tidy = load_sts_metrics(STS_PATH)

    fapi = tidy[ tidy["Dataset"].str.contains(r"\bFAPI\b", case=False, regex=True) &
                 ~tidy["Dataset"].str.contains("TEMPO", case=False) ].copy()
    ftempo = tidy[ tidy["Dataset"].str.contains("FAPI-TEMPO", case=False) ].copy()

    fapi_t   = map_to_time(fapi, TOTAL_MS)
    ftempo_t = map_to_time(ftempo, TOTAL_MS)

    # Nucleation (dn/dt)
    c_f,  dn_f  = estimate_dn_dt(fapi_t["t_ms"].to_numpy(),   BIN_MS, TOTAL_MS)
    c_t,  dn_t  = estimate_dn_dt(ftempo_t["t_ms"].to_numpy(), BIN_MS, TOTAL_MS)

    # Growth (median radius & derivative)
    cg_f, mr_f, dr_f = estimate_growth_rate(fapi_t["t_ms"].to_numpy(),   fapi_t["radius_um"].to_numpy(),   BIN_MS, TOTAL_MS)
    cg_t, mr_t, dr_t = estimate_growth_rate(ftempo_t["t_ms"].to_numpy(), ftempo_t["radius_um"].to_numpy(), BIN_MS, TOTAL_MS)

    # Plots
    plt.figure(figsize=(7,4.5))
    if c_f.size: plt.plot(c_f, dn_f, label="FAPI")
    if c_t.size: plt.plot(c_t, dn_t, label="FAPI-TEMPO")
    plt.xlabel("Pseudo-time (ms)"); plt.ylabel("Estimated dn/dt (counts/ms)")
    plt.title("Nucleation rate proxy (decoupled)")
    plt.legend(); plt.tight_layout()
    plt.savefig("decoupled_dn_dt.png", dpi=160); plt.close()

    plt.figure(figsize=(7,4.5))
    if cg_f.size: plt.plot(cg_f, dr_f, label="FAPI")
    if cg_t.size: plt.plot(cg_t, dr_t, label="FAPI-TEMPO")
    plt.xlabel("Pseudo-time (ms)"); plt.ylabel("d⟨radius⟩/dt (μm/ms)")
    plt.title("Growth rate proxy (decoupled)")
    plt.legend(); plt.tight_layout()
    plt.savefig("decoupled_growth_rate.png", dpi=160); plt.close()

    # CSV exports
    dn_dt_df = pd.concat([
        pd.DataFrame({"t_ms": c_f, "dn_dt_counts_per_ms": dn_f, "Dataset":"FAPI"}),
        pd.DataFrame({"t_ms": c_t, "dn_dt_counts_per_ms": dn_t, "Dataset":"FAPI-TEMPO"})
    ], ignore_index=True)
    gr_df = pd.concat([
        pd.DataFrame({"t_ms": cg_f, "median_radius_um": mr_f, "drdt_um_per_ms": dr_f, "Dataset":"FAPI"}),
        pd.DataFrame({"t_ms": cg_t, "median_radius_um": mr_t, "drdt_um_per_ms": dr_t, "Dataset":"FAPI-TEMPO"})
    ], ignore_index=True)

    dn_dt_df.to_csv("decoupled_dn_dt_proxy.csv", index=False)
    gr_df.to_csv("decoupled_growth_proxy.csv", index=False)

    print("Saved:")
    print(" - decoupled_dn_dt.png")
    print(" - decoupled_growth_rate.png")
    print(" - decoupled_dn_dt_proxy.csv")
    print(" - decoupled_growth_proxy.csv")

if __name__ == "__main__":
    main()
