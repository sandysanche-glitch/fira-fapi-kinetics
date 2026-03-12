import os
import pandas as pd
import matplotlib.pyplot as plt

FAPI_KIN  = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\sam_cuda_vith_clean\FAPI\kinetics"
TEMPO_KIN = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\out\FAPI_TEMPO\idmap_kinetics_win60"

OUT = os.path.join(os.path.dirname(FAPI_KIN), "compare_FAPI_vs_TEMPO_win60")
os.makedirs(OUT, exist_ok=True)

TAU_MAX = 5000

def read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    return pd.read_csv(path)

# --- Load ---
dn_fapi = read_csv(os.path.join(FAPI_KIN,  "dn_dt_nucleation.csv"))
dn_tmp  = read_csv(os.path.join(TEMPO_KIN,"dn_dt_nucleation.csv"))

ac_fapi = read_csv(os.path.join(FAPI_KIN,  "active_tracks.csv"))
ac_tmp  = read_csv(os.path.join(TEMPO_KIN,"active_tracks.csv"))

gr_fapi = read_csv(os.path.join(FAPI_KIN,  "growth_rate_vs_time.csv"))
gr_tmp  = read_csv(os.path.join(TEMPO_KIN,"growth_rate_vs_time.csv"))

tau_fapi_path = os.path.join(FAPI_KIN, "tau_fits_filtered_tau5000.csv")
tau_fapi = read_csv(tau_fapi_path) if os.path.exists(tau_fapi_path) else read_csv(os.path.join(FAPI_KIN,"tau_fits.csv"))
tau_tmp  = read_csv(os.path.join(TEMPO_KIN,"tau_fits.csv"))

tau_fapi = tau_fapi[tau_fapi["tau_ms"] <= TAU_MAX].copy()
tau_tmp  = tau_tmp[tau_tmp["tau_ms"] <= TAU_MAX].copy()

# ---------- Plot 1: nucleation rate ----------
plt.figure()
plt.plot(dn_fapi["bin_center_ms"], dn_fapi["dn_dt_per_s"], marker="o", label="FAPI")
plt.plot(dn_tmp ["bin_center_ms"], dn_tmp ["dn_dt_per_s"], marker="o", label="FAPI-TEMPO")
plt.xlabel("time (ms)")
plt.ylabel("dn/dt (1/s)")
plt.title("Nucleation rate (bin=20 ms)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "overlay_nucleation_rate.png"), dpi=200)
plt.close()

# ---------- Plot 1b: cumulative nucleation ----------
plt.figure()
plt.plot(dn_fapi["bin_center_ms"], dn_fapi["cum_n"], marker="o", label="FAPI")
plt.plot(dn_tmp ["bin_center_ms"], dn_tmp ["cum_n"], marker="o", label="FAPI-TEMPO")
plt.xlabel("time (ms)")
plt.ylabel("cumulative nucleated grains")
plt.title("Cumulative nucleation N(t)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "overlay_cum_nucleation.png"), dpi=200)
plt.close()

# ---------- Plot 2: active grains ----------
# try common column names; fall back to 2nd column
def pick_active(df):
    for c in ["active_tracks","active","n_active","active_grains"]:
        if c in df.columns:
            return c
    return df.columns[1]

ac_y_f = pick_active(ac_fapi)
ac_y_t = pick_active(ac_tmp)

plt.figure()
plt.plot(ac_fapi[ac_fapi.columns[0]], ac_fapi[ac_y_f], marker="o", label="FAPI")
plt.plot(ac_tmp [ac_tmp.columns[0]],  ac_tmp [ac_y_t], marker="o", label="FAPI-TEMPO")
plt.xlabel(ac_fapi.columns[0])
plt.ylabel("active grains")
plt.title("Active grains vs time")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "overlay_active_grains.png"), dpi=200)
plt.close()

# ---------- Plot 3: median growth rate vs time ----------
def pick_growth(df):
    for c in ["median_um_per_s","median_growth_um_per_s","growth_um_per_s","median"]:
        if c in df.columns:
            return c
    return df.columns[1]

gr_y_f = pick_growth(gr_fapi)
gr_y_t = pick_growth(gr_tmp)

plt.figure()
plt.plot(gr_fapi[gr_fapi.columns[0]], gr_fapi[gr_y_f], marker="o", label="FAPI")
plt.plot(gr_tmp [gr_tmp.columns[0]],  gr_tmp [gr_y_t], marker="o", label="FAPI-TEMPO")
plt.xlabel(gr_fapi.columns[0])
plt.ylabel("growth rate (um/s)")
plt.title("Median growth rate vs time")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "overlay_growth_rate.png"), dpi=200)
plt.close()

# ---------- Plot 4: tau histogram ----------
plt.figure()
plt.hist(tau_fapi["tau_ms"], bins=30, alpha=0.6, label="FAPI")
plt.hist(tau_tmp ["tau_ms"], bins=30, alpha=0.6, label="FAPI-TEMPO")
plt.xlabel("tau (ms)")
plt.ylabel("count")
plt.title(f"Tau distribution (tau<={TAU_MAX} ms)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "overlay_tau_hist.png"), dpi=200)
plt.close()

print("Wrote overlays to:", OUT)
for fn in [
    "overlay_nucleation_rate.png",
    "overlay_cum_nucleation.png",
    "overlay_active_grains.png",
    "overlay_growth_rate.png",
    "overlay_tau_hist.png",
]:
    print(" -", os.path.join(OUT, fn))
