import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------- SET YOUR OUTPUT FOLDER HERE ----------
OUT_DIR = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\sam_cuda_vith_clean\FAPI\stable_nucleation_compare_win60_v3"

fapi_events  = os.path.join(OUT_DIR, "nucleation_events_filtered_FAPI.csv")
tempo_events = os.path.join(OUT_DIR, "nucleation_events_filtered_FAPI_TEMPO.csv")
out_png      = os.path.join(OUT_DIR, "overlay_dn_dt_Rnuc_gate.png")
out_csv_fapi = os.path.join(OUT_DIR, "dn_dt_filtered_FAPI_RnucGate.csv")
out_csv_tmp  = os.path.join(OUT_DIR, "dn_dt_filtered_FAPI_TEMPO_RnucGate.csv")

BIN_MS = 20
R_NUC_MAX = 60  # try 40, 50, 60

def edges_from_both(t1, t2, bin_ms):
    t1 = np.asarray(t1, dtype=float)
    t2 = np.asarray(t2, dtype=float)
    all_t = np.concatenate([t1, t2]) if (t1.size + t2.size) else np.array([0.0])
    tmax = bin_ms * np.ceil(np.max(all_t) / bin_ms)
    edges = np.arange(0.0, tmax + bin_ms, bin_ms, dtype=float)
    if edges.size < 2:
        edges = np.array([0.0, bin_ms], dtype=float)
    return edges

def dn_dt(times, edges, bin_ms):
    times = np.asarray(times, dtype=float)
    counts, _ = np.histogram(times, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rate = counts / (bin_ms / 1000.0)  # 1/s
    cumN = np.cumsum(counts)
    return centers, rate, cumN, counts

# ---- read inputs ----
if not os.path.exists(fapi_events):
    raise FileNotFoundError(f"Missing: {fapi_events}")
if not os.path.exists(tempo_events):
    raise FileNotFoundError(f"Missing: {tempo_events}")

fapi  = pd.read_csv(fapi_events)
tempo = pd.read_csv(tempo_events)

# ---- gate: reject "already large at birth" ----
fapi_g  = fapi [fapi ["R_nuc_px"] <= R_NUC_MAX].copy()
tempo_g = tempo[tempo["R_nuc_px"] <= R_NUC_MAX].copy()

edges = edges_from_both(fapi_g["nuc_time_ms"].values, tempo_g["nuc_time_ms"].values, BIN_MS)

cF, rF, cumF, nF = dn_dt(fapi_g["nuc_time_ms"].values, edges, BIN_MS)
cT, rT, cumT, nT = dn_dt(tempo_g["nuc_time_ms"].values, edges, BIN_MS)

# ---- write gated curves ----
pd.DataFrame({
    "bin_center_ms": cF,
    "n_nucleated": nF,
    "dn_dt_per_s": rF,
    "cum_n": cumF
}).to_csv(out_csv_fapi, index=False)

pd.DataFrame({
    "bin_center_ms": cT,
    "n_nucleated": nT,
    "dn_dt_per_s": rT,
    "cum_n": cumT
}).to_csv(out_csv_tmp, index=False)

# ---- plot ----
plt.figure()
plt.plot(cF, rF, marker="o", label=f"FAPI (n={len(fapi_g)})")
plt.plot(cT, rT, marker="o", label=f"FAPI-TEMPO (n={len(tempo_g)})")
plt.xlabel("time (ms)")
plt.ylabel("dn/dt (1/s)")
plt.title(f"Stable nucleation rate with R_nuc <= {R_NUC_MAX} px (bin={BIN_MS} ms)")
plt.legend()
plt.tight_layout()
plt.savefig(out_png, dpi=200)
plt.close()

print("[OK] Read:")
print("  ", fapi_events)
print("  ", tempo_events)
print("[OK] Wrote:")
print("  ", out_png)
print("  ", out_csv_fapi)
print("  ", out_csv_tmp)
print(f"[OK] Kept events: FAPI {len(fapi_g)}/{len(fapi)} | TEMPO {len(tempo_g)}/{len(tempo)}")
