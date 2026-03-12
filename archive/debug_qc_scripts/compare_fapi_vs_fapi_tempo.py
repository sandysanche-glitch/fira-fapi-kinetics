# compare_fapi_vs_fapi_tempo.py
# ------------------------------------------------------------
# Comparative plotting: FAPI vs FAPI-TEMPO
#
# This script supports two input styles:
#
# MODE A (matched kinetics exports; preferred for your current workflow)
#   Each dataset folder contains:
#     - kinetics_tau0p3_Nt.csv
#     - kinetics_tau0p3_rate.csv
#   Optional:
#     - growth_per_frame_tau0p3.csv
#
# MODE B (legacy kinetics folders; auto-detected fallback)
#   Each dataset folder may contain files like:
#     - dn_dt_nucleation.csv
#     - active_tracks.csv
#     - growth_rate_vs_time.csv
#     - tau_fits.csv (or tau_fits_filtered_tau5000.csv)
#
# Outputs (PNG + CSV summary) are written to OUT_DIR.
#
# Notes:
# - Use "FAPI" and "FAPI-TEMPO" labels (as requested).
# - Figure titles include a provenance banner to avoid mixing processing levels.
# ------------------------------------------------------------

import os
import math
from typing import Optional, Tuple, Dict, List

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# EDIT THESE PATHS
# ============================================================

# --- FAPI-TEMPO ---
# Option 1 (artifact-filtered kinetics exports from your v15 workflow)
BASE_FAPI_TEMPO = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\stable_v15_out_overlap03_poly_pad400"

# --- FAPI ---
# Option 1 (matched kinetics built from retrack folder)
# Example (recommended):
# BASE_FAPI = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\out\FAPI\retrack_cuda_vith\matched_kinetics_for_compare"
#
# If not built yet, you can point to retrack folder directly and the script will use MODE B if files exist.
BASE_FAPI = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\out\FAPI\retrack_cuda_vith\matched_kinetics_for_compare_symmetric_bbox"

# Output folder
OUT_DIR = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\out\stable_nucleation_compare_final\compare_FAPI_vs_FAPI_TEMPO"

# Provenance shown in figure titles / summary
FAPI_PROVENANCE = "retrack/stable-gated"
FAPI_TEMPO_PROVENANCE = "artifact-filtered bbox τ=0.3"   # change if using stable-gated TEMPO instead

# Optional tau histogram cap (MODE B only)
TAU_MAX_MS = 5000

# ============================================================
# Helpers
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def exists(path: str) -> bool:
    return os.path.exists(path)

def read_csv(path: str, label: str) -> pd.DataFrame:
    if not exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")
    return pd.read_csv(path)

def first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if exists(p):
            return p
    return None

def numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def mode_a_paths(base_dir: str) -> Dict[str, str]:
    """Matched kinetics files."""
    return {
        "Nt": first_existing([
            os.path.join(base_dir, "kinetics_tau0p3_Nt.csv"),
            os.path.join(base_dir, "kinetics_tau0p2_Nt.csv"),
            os.path.join(base_dir, "kinetics_tau0p4_Nt.csv"),
        ]) or "",
        "rate": first_existing([
            os.path.join(base_dir, "kinetics_tau0p3_rate.csv"),
            os.path.join(base_dir, "kinetics_tau0p2_rate.csv"),
            os.path.join(base_dir, "kinetics_tau0p4_rate.csv"),
        ]) or "",
        "growth_pf": first_existing([
            os.path.join(base_dir, "growth_per_frame_tau0p3.csv"),
            os.path.join(base_dir, "growth_per_frame_tau0p2.csv"),
            os.path.join(base_dir, "growth_per_frame_tau0p4.csv"),
        ]) or "",
    }

def mode_b_paths(base_dir: str) -> Dict[str, str]:
    """Legacy kinetics-style files."""
    return {
        "dn_dt_nucleation": first_existing([os.path.join(base_dir, "dn_dt_nucleation.csv")]) or "",
        "active_tracks": first_existing([os.path.join(base_dir, "active_tracks.csv")]) or "",
        "growth_rate_vs_time": first_existing([os.path.join(base_dir, "growth_rate_vs_time.csv")]) or "",
        "tau_fits": first_existing([
            os.path.join(base_dir, f"tau_fits_filtered_tau{int(TAU_MAX_MS)}.csv"),
            os.path.join(base_dir, "tau_fits.csv"),
        ]) or "",
    }

def detect_mode(base_dir: str) -> str:
    a = mode_a_paths(base_dir)
    if a["Nt"] and a["rate"]:
        return "A"
    b = mode_b_paths(base_dir)
    # minimal MODE B requirement: dn_dt_nucleation
    if b["dn_dt_nucleation"]:
        return "B"
    raise FileNotFoundError(
        f"Could not detect supported input mode in folder:\n  {base_dir}\n"
        f"Expected either matched kinetics files (kinetics_tau0p3_*.csv) "
        f"or legacy files (dn_dt_nucleation.csv, etc.)."
    )

# ============================================================
# Standardization to common internal schema
# ============================================================

def standardize_rate_mode_a(df: pd.DataFrame) -> pd.DataFrame:
    # expected: t_center_ms, t_center_s, dNdt_per_s
    out = df.copy()
    if "t_center_s" not in out.columns and "t_center_ms" in out.columns:
        out["t_center_s"] = pd.to_numeric(out["t_center_ms"], errors="coerce") / 1000.0
    out = numeric(out, ["t_center_ms", "t_center_s", "dNdt_per_s"])
    req = ["t_center_s", "dNdt_per_s"]
    missing = [c for c in req if c not in out.columns]
    if missing:
        raise ValueError(f"MODE A rate file missing columns {missing}. Found: {list(out.columns)}")
    if "t_center_ms" not in out.columns:
        out["t_center_ms"] = out["t_center_s"] * 1000.0
    return out.dropna(subset=["t_center_s", "dNdt_per_s"]).sort_values("t_center_s").reset_index(drop=True)

def standardize_Nt_mode_a(df: pd.DataFrame) -> pd.DataFrame:
    # expected: t_ms, t_s, N
    out = df.copy()
    if "t_s" not in out.columns and "t_ms" in out.columns:
        out["t_s"] = pd.to_numeric(out["t_ms"], errors="coerce") / 1000.0
    out = numeric(out, ["t_ms", "t_s", "N"])
    req = ["t_s", "N"]
    missing = [c for c in req if c not in out.columns]
    if missing:
        raise ValueError(f"MODE A Nt file missing columns {missing}. Found: {list(out.columns)}")
    if "t_ms" not in out.columns:
        out["t_ms"] = out["t_s"] * 1000.0
    out = out.dropna(subset=["t_s", "N"]).sort_values("t_s").reset_index(drop=True)
    out["N"] = out["N"].astype(int)
    return out

def standardize_rate_mode_b(df: pd.DataFrame) -> pd.DataFrame:
    # expected typical: bin_center_ms, dn_dt_per_s, cum_n
    out = df.copy()
    x_col = pick_col(out, ["bin_center_ms", "t_center_ms", "t_ms"])
    y_col = pick_col(out, ["dn_dt_per_s", "dNdt_per_s", "rate_per_s"])
    if x_col is None or y_col is None:
        raise ValueError(f"MODE B dn_dt file missing rate/time columns. Found: {list(out.columns)}")
    out = out.rename(columns={x_col: "t_center_ms", y_col: "dNdt_per_s"})
    out = numeric(out, ["t_center_ms", "dNdt_per_s"])
    out["t_center_s"] = out["t_center_ms"] / 1000.0
    return out.dropna(subset=["t_center_ms", "dNdt_per_s"]).sort_values("t_center_ms").reset_index(drop=True)

def standardize_Nt_mode_b(df: pd.DataFrame) -> pd.DataFrame:
    # derive N(t) from dn_dt_nucleation.csv if cum_n exists
    out = df.copy()
    x_col = pick_col(out, ["bin_center_ms", "t_center_ms", "t_ms"])
    n_col = pick_col(out, ["cum_n", "cumulative_nucleations", "N"])
    if x_col is None or n_col is None:
        raise ValueError(f"MODE B dn_dt file missing cumulative column. Found: {list(out.columns)}")
    out = out.rename(columns={x_col: "t_ms", n_col: "N"})
    out = numeric(out, ["t_ms", "N"])
    out["t_s"] = out["t_ms"] / 1000.0
    out = out.dropna(subset=["t_ms", "N"]).sort_values("t_ms").reset_index(drop=True)
    out["N"] = out["N"].astype(int)
    return out[["t_ms", "t_s", "N"]]

def standardize_active_mode_b(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    x_col = pick_col(out, ["time_ms", "bin_center_ms", "t_ms", out.columns[0]])
    y_col = pick_col(out, ["active_tracks", "active", "n_active", "active_grains"])
    if y_col is None:
        # fallback: second column
        y_col = out.columns[1] if len(out.columns) > 1 else None
    if x_col is None or y_col is None:
        return pd.DataFrame(columns=["t_ms", "t_s", "active_tracks"])
    out = out.rename(columns={x_col: "t_ms", y_col: "active_tracks"})
    out = numeric(out, ["t_ms", "active_tracks"])
    out["t_s"] = out["t_ms"] / 1000.0
    return out.dropna(subset=["t_ms", "active_tracks"]).sort_values("t_ms").reset_index(drop=True)

def standardize_growth_rate_mode_b(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    x_col = pick_col(out, ["time_ms", "bin_center_ms", "t_ms", out.columns[0]])
    y_col = pick_col(out, ["median_um_per_s", "median_growth_um_per_s", "growth_um_per_s", "median"])
    if y_col is None:
        y_col = out.columns[1] if len(out.columns) > 1 else None
    if x_col is None or y_col is None:
        return pd.DataFrame(columns=["t_ms", "t_s", "growth_rate_um_per_s"])
    out = out.rename(columns={x_col: "t_ms", y_col: "growth_rate_um_per_s"})
    out = numeric(out, ["t_ms", "growth_rate_um_per_s"])
    out["t_s"] = out["t_ms"] / 1000.0
    return out.dropna(subset=["t_ms", "growth_rate_um_per_s"]).sort_values("t_ms").reset_index(drop=True)

def standardize_tau_mode_b(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    tau_col = pick_col(out, ["tau_ms", "tau", "tau_fit_ms"])
    if tau_col is None:
        return pd.DataFrame(columns=["tau_ms"])
    out = out.rename(columns={tau_col: "tau_ms"})
    out = numeric(out, ["tau_ms"])
    out = out.dropna(subset=["tau_ms"]).copy()
    out = out[out["tau_ms"] <= TAU_MAX_MS]
    return out.reset_index(drop=True)

def load_growth_per_frame_mode_a(path: str) -> pd.DataFrame:
    """Optional: aggregate per-frame growth to median equivalent radius if available."""
    g = pd.read_csv(path)
    if "t_s" not in g.columns and "t_ms" in g.columns:
        g["t_s"] = pd.to_numeric(g["t_ms"], errors="coerce") / 1000.0
    if "t_ms" not in g.columns and "t_s" in g.columns:
        g["t_ms"] = pd.to_numeric(g["t_s"], errors="coerce") * 1000.0
    # derive R_px if only area exists
    if "R_px" not in g.columns and "area_px" in g.columns:
        area = pd.to_numeric(g["area_px"], errors="coerce")
        g["R_px"] = (area / math.pi) ** 0.5
    g = numeric(g, ["t_ms", "t_s", "R_px"])
    # We plot median R_px vs time (not dR/dt) since growth_per_frame often lacks explicit rates
    if "t_ms" in g.columns and "R_px" in g.columns:
        agg = (
            g.dropna(subset=["t_ms", "R_px"])
             .groupby("t_ms", as_index=False)["R_px"]
             .median()
             .sort_values("t_ms")
             .reset_index(drop=True)
        )
        agg["t_s"] = agg["t_ms"] / 1000.0
        agg = agg.rename(columns={"R_px": "median_R_px"})
        return agg
    return pd.DataFrame(columns=["t_ms", "t_s", "median_R_px"])

# ============================================================
# Dataset loader
# ============================================================

def load_dataset(base_dir: str, label: str) -> Dict[str, pd.DataFrame]:
    mode = detect_mode(base_dir)
    data: Dict[str, pd.DataFrame] = {"mode": pd.DataFrame([{"mode": mode}])}  # simple carrier

    if mode == "A":
        p = mode_a_paths(base_dir)
        rate_df = read_csv(p["rate"], f"{label} MODE A rate")
        nt_df = read_csv(p["Nt"], f"{label} MODE A Nt")
        data["rate"] = standardize_rate_mode_a(rate_df)
        data["Nt"] = standardize_Nt_mode_a(nt_df)

        # Optional growth per-frame -> median_R_px(t)
        if p["growth_pf"]:
            try:
                data["growth_proxy"] = load_growth_per_frame_mode_a(p["growth_pf"])
            except Exception as e:
                print(f"[WARN] {label}: could not load growth_per_frame file ({e})")
                data["growth_proxy"] = pd.DataFrame(columns=["t_ms", "t_s", "median_R_px"])
        else:
            data["growth_proxy"] = pd.DataFrame(columns=["t_ms", "t_s", "median_R_px"])

        # Active tracks / tau may not exist in MODE A outputs
        data["active"] = pd.DataFrame(columns=["t_ms", "t_s", "active_tracks"])
        data["tau"] = pd.DataFrame(columns=["tau_ms"])

        # metadata
        data["meta"] = pd.DataFrame([{
            "label": label,
            "mode": "A",
            "base_dir": base_dir,
            "rate_file": p["rate"],
            "Nt_file": p["Nt"],
            "growth_file": p["growth_pf"] or "",
        }])

    elif mode == "B":
        p = mode_b_paths(base_dir)
        dn = read_csv(p["dn_dt_nucleation"], f"{label} dn_dt_nucleation")
        data["rate"] = standardize_rate_mode_b(dn)
        data["Nt"] = standardize_Nt_mode_b(dn)

        if p["active_tracks"]:
            data["active"] = standardize_active_mode_b(read_csv(p["active_tracks"], f"{label} active_tracks"))
        else:
            data["active"] = pd.DataFrame(columns=["t_ms", "t_s", "active_tracks"])

        if p["growth_rate_vs_time"]:
            data["growth_rate"] = standardize_growth_rate_mode_b(read_csv(p["growth_rate_vs_time"], f"{label} growth_rate_vs_time"))
        else:
            data["growth_rate"] = pd.DataFrame(columns=["t_ms", "t_s", "growth_rate_um_per_s"])

        if p["tau_fits"]:
            data["tau"] = standardize_tau_mode_b(read_csv(p["tau_fits"], f"{label} tau_fits"))
        else:
            data["tau"] = pd.DataFrame(columns=["tau_ms"])

        data["growth_proxy"] = pd.DataFrame(columns=["t_ms", "t_s", "median_R_px"])

        data["meta"] = pd.DataFrame([{
            "label": label,
            "mode": "B",
            "base_dir": base_dir,
            "dn_dt_file": p["dn_dt_nucleation"],
            "active_file": p["active_tracks"] or "",
            "growth_rate_file": p["growth_rate_vs_time"] or "",
            "tau_file": p["tau_fits"] or "",
        }])

    else:
        raise RuntimeError("Unsupported mode")

    return data

# ============================================================
# Metrics
# ============================================================

def compute_nt_metrics(nt_df: pd.DataFrame) -> Dict[str, float]:
    if nt_df.empty:
        return {
            "N_final": math.nan,
            "onset_s": math.nan,
            "t10_s": math.nan,
            "t50_s": math.nan,
            "t90_s": math.nan,
            "t_last_s": math.nan,
        }

    nt = nt_df.sort_values("t_s").reset_index(drop=True)
    N_final = int(nt["N"].iloc[-1])
    onset_s = float(nt["t_s"].iloc[0]) if len(nt) else math.nan
    t_last_s = float(nt["t_s"].iloc[-1]) if len(nt) else math.nan

    def first_time_for_fraction(frac: float) -> float:
        if N_final <= 0:
            return math.nan
        target = math.ceil(frac * N_final)
        row = nt.loc[nt["N"] >= target]
        return float(row["t_s"].iloc[0]) if not row.empty else math.nan

    return {
        "N_final": N_final,
        "onset_s": onset_s,
        "t10_s": first_time_for_fraction(0.10),
        "t50_s": first_time_for_fraction(0.50),
        "t90_s": first_time_for_fraction(0.90),
        "t_last_s": t_last_s,
    }

def compute_rate_metrics(rate_df: pd.DataFrame) -> Dict[str, float]:
    if rate_df.empty:
        return {"peak_dNdt_per_s": math.nan, "t_peak_s": math.nan}
    r = rate_df.sort_values("t_center_s").reset_index(drop=True)
    i = r["dNdt_per_s"].idxmax()
    peak = r.loc[i]
    return {
        "peak_dNdt_per_s": float(peak["dNdt_per_s"]),
        "t_peak_s": float(peak["t_center_s"]),
    }

def compute_active_peak_metrics(active_df: pd.DataFrame) -> Dict[str, float]:
    if active_df.empty:
        return {"peak_active_tracks": math.nan, "t_peak_active_s": math.nan}
    a = active_df.sort_values("t_s").reset_index(drop=True)
    i = a["active_tracks"].idxmax()
    row = a.loc[i]
    return {
        "peak_active_tracks": float(row["active_tracks"]),
        "t_peak_active_s": float(row["t_s"]),
    }

def compute_growth_metrics(data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    # Prefer explicit growth rate (MODE B), else proxy median_R_px(t) (MODE A)
    if "growth_rate" in data and not data["growth_rate"].empty:
        g = data["growth_rate"].sort_values("t_s").reset_index(drop=True)
        i = g["growth_rate_um_per_s"].idxmax()
        row = g.loc[i]
        return {
            "growth_metric_name": "peak_growth_rate_um_per_s",
            "growth_metric_value": float(row["growth_rate_um_per_s"]),
            "t_growth_metric_s": float(row["t_s"]),
        }
    if "growth_proxy" in data and not data["growth_proxy"].empty:
        gp = data["growth_proxy"].sort_values("t_s").reset_index(drop=True)
        i = gp["median_R_px"].idxmax()
        row = gp.loc[i]
        return {
            "growth_metric_name": "max_median_R_px",
            "growth_metric_value": float(row["median_R_px"]),
            "t_growth_metric_s": float(row["t_s"]),
        }
    return {
        "growth_metric_name": "",
        "growth_metric_value": math.nan,
        "t_growth_metric_s": math.nan,
    }

# ============================================================
# Plotting
# ============================================================

def provenance_banner() -> str:
    return f"FAPI: {FAPI_PROVENANCE} | FAPI-TEMPO: {FAPI_TEMPO_PROVENANCE}"

def plot_rate(fapi: Dict[str, pd.DataFrame], tempo: Dict[str, pd.DataFrame], out_dir: str) -> None:
    plt.figure(figsize=(6.6, 4.4))
    plt.plot(fapi["rate"]["t_center_ms"], fapi["rate"]["dNdt_per_s"], marker="o", label=f"FAPI (n={int(fapi['Nt']['N'].max()) if not fapi['Nt'].empty else 'NA'})")
    plt.plot(tempo["rate"]["t_center_ms"], tempo["rate"]["dNdt_per_s"], marker="o", label=f"FAPI-TEMPO (n={int(tempo['Nt']['N'].max()) if not tempo['Nt'].empty else 'NA'})")
    plt.xlabel("time (ms)")
    plt.ylabel("dn/dt (1/s)")
    plt.title("Nucleation rate dN/dt\n" + provenance_banner())
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compare_dn_dt_FAPI_vs_FAPI-TEMPO.png"), dpi=300)
    plt.close()

def plot_Nt(fapi: Dict[str, pd.DataFrame], tempo: Dict[str, pd.DataFrame], out_dir: str) -> None:
    plt.figure(figsize=(6.6, 4.4))
    plt.step(fapi["Nt"]["t_ms"], fapi["Nt"]["N"], where="post", label=f"FAPI (n={int(fapi['Nt']['N'].max()) if not fapi['Nt'].empty else 'NA'})")
    plt.step(tempo["Nt"]["t_ms"], tempo["Nt"]["N"], where="post", label=f"FAPI-TEMPO (n={int(tempo['Nt']['N'].max()) if not tempo['Nt'].empty else 'NA'})")
    plt.xlabel("time (ms)")
    plt.ylabel("cumulative stable nuclei N(t)")
    plt.title("Cumulative nucleation N(t)\n" + provenance_banner())
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compare_Nt_FAPI_vs_FAPI-TEMPO.png"), dpi=300)
    plt.close()

def plot_active_if_available(fapi: Dict[str, pd.DataFrame], tempo: Dict[str, pd.DataFrame], out_dir: str) -> bool:
    if fapi["active"].empty and tempo["active"].empty:
        return False
    plt.figure(figsize=(6.6, 4.4))
    if not fapi["active"].empty:
        plt.plot(fapi["active"]["t_ms"], fapi["active"]["active_tracks"], marker="o", label="FAPI")
    if not tempo["active"].empty:
        plt.plot(tempo["active"]["t_ms"], tempo["active"]["active_tracks"], marker="o", label="FAPI-TEMPO")
    plt.xlabel("time (ms)")
    plt.ylabel("active tracks")
    plt.title("Active tracks vs time\n" + provenance_banner())
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compare_active_tracks_FAPI_vs_FAPI-TEMPO.png"), dpi=300)
    plt.close()
    return True

def plot_growth_if_available(fapi: Dict[str, pd.DataFrame], tempo: Dict[str, pd.DataFrame], out_dir: str) -> bool:
    # Case 1: explicit growth_rate in both or either (MODE B)
    has_explicit = ("growth_rate" in fapi and not fapi["growth_rate"].empty) or ("growth_rate" in tempo and not tempo["growth_rate"].empty)
    if has_explicit:
        plt.figure(figsize=(6.6, 4.4))
        if "growth_rate" in fapi and not fapi["growth_rate"].empty:
            plt.plot(fapi["growth_rate"]["t_ms"], fapi["growth_rate"]["growth_rate_um_per_s"], marker="o", label="FAPI")
        if "growth_rate" in tempo and not tempo["growth_rate"].empty:
            plt.plot(tempo["growth_rate"]["t_ms"], tempo["growth_rate"]["growth_rate_um_per_s"], marker="o", label="FAPI-TEMPO")
        plt.xlabel("time (ms)")
        plt.ylabel("growth rate (um/s)")
        plt.title("Growth rate vs time\n" + provenance_banner())
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "compare_growth_rate_FAPI_vs_FAPI-TEMPO.png"), dpi=300)
        plt.close()
        return True

    # Case 2: proxy from growth_per_frame_tau0p3.csv (MODE A)
    has_proxy = ("growth_proxy" in fapi and not fapi["growth_proxy"].empty) or ("growth_proxy" in tempo and not tempo["growth_proxy"].empty)
    if has_proxy:
        plt.figure(figsize=(6.6, 4.4))
        if "growth_proxy" in fapi and not fapi["growth_proxy"].empty:
            plt.plot(fapi["growth_proxy"]["t_ms"], fapi["growth_proxy"]["median_R_px"], marker="o", label="FAPI")
        if "growth_proxy" in tempo and not tempo["growth_proxy"].empty:
            plt.plot(tempo["growth_proxy"]["t_ms"], tempo["growth_proxy"]["median_R_px"], marker="o", label="FAPI-TEMPO")
        plt.xlabel("time (ms)")
        plt.ylabel("median equivalent radius (px)")
        plt.title("Growth proxy (median R) vs time\n" + provenance_banner())
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "compare_growth_proxy_medianR_FAPI_vs_FAPI-TEMPO.png"), dpi=300)
        plt.close()
        return True

    return False

def plot_tau_hist_if_available(fapi: Dict[str, pd.DataFrame], tempo: Dict[str, pd.DataFrame], out_dir: str) -> bool:
    if fapi["tau"].empty and tempo["tau"].empty:
        return False
    plt.figure(figsize=(6.6, 4.4))
    if not fapi["tau"].empty:
        plt.hist(fapi["tau"]["tau_ms"], bins=30, alpha=0.6, label="FAPI")
    if not tempo["tau"].empty:
        plt.hist(tempo["tau"]["tau_ms"], bins=30, alpha=0.6, label="FAPI-TEMPO")
    plt.xlabel("tau (ms)")
    plt.ylabel("count")
    plt.title(f"Tau distribution (tau <= {TAU_MAX_MS} ms)\n" + provenance_banner())
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compare_tau_hist_FAPI_vs_FAPI-TEMPO.png"), dpi=300)
    plt.close()
    return True

# ============================================================
# Main
# ============================================================

def main():
    ensure_dir(OUT_DIR)

    print("[INFO] Loading datasets...")
    fapi = load_dataset(BASE_FAPI, "FAPI")
    tempo = load_dataset(BASE_FAPI_TEMPO, "FAPI-TEMPO")

    fapi_mode = fapi["meta"]["mode"].iloc[0]
    tempo_mode = tempo["meta"]["mode"].iloc[0]

    print(f"[INFO] FAPI mode       = {fapi_mode} ({BASE_FAPI})")
    print(f"[INFO] FAPI-TEMPO mode = {tempo_mode} ({BASE_FAPI_TEMPO})")

    # Core plots (always)
    plot_rate(fapi, tempo, OUT_DIR)
    plot_Nt(fapi, tempo, OUT_DIR)

    # Optional plots
    made_active = plot_active_if_available(fapi, tempo, OUT_DIR)
    made_growth = plot_growth_if_available(fapi, tempo, OUT_DIR)
    made_tau = plot_tau_hist_if_available(fapi, tempo, OUT_DIR)

    # Metrics summary
    rows = []
    for label, ds in [("FAPI", fapi), ("FAPI-TEMPO", tempo)]:
        row = {
            "dataset": label,
            "mode": ds["meta"]["mode"].iloc[0],
            "base_dir": ds["meta"]["base_dir"].iloc[0],
            "provenance_label": FAPI_PROVENANCE if label == "FAPI" else FAPI_TEMPO_PROVENANCE,
        }
        row.update(compute_nt_metrics(ds["Nt"]))
        row.update(compute_rate_metrics(ds["rate"]))
        row.update(compute_active_peak_metrics(ds["active"]))
        row.update(compute_growth_metrics(ds))
        row["n_tau_fits"] = int(len(ds["tau"])) if "tau" in ds else 0
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(OUT_DIR, "compare_summary_metrics_FAPI_vs_FAPI-TEMPO.csv")
    summary.to_csv(summary_path, index=False)

    # Export aligned curves for easy post-analysis
    fapi["rate"].assign(dataset="FAPI").to_csv(os.path.join(OUT_DIR, "rate_curve_FAPI.csv"), index=False)
    tempo["rate"].assign(dataset="FAPI-TEMPO").to_csv(os.path.join(OUT_DIR, "rate_curve_FAPI-TEMPO.csv"), index=False)
    fapi["Nt"].assign(dataset="FAPI").to_csv(os.path.join(OUT_DIR, "Nt_curve_FAPI.csv"), index=False)
    tempo["Nt"].assign(dataset="FAPI-TEMPO").to_csv(os.path.join(OUT_DIR, "Nt_curve_FAPI-TEMPO.csv"), index=False)

    # Save provenance + file map
    manifest_rows = []
    for ds in [fapi, tempo]:
        manifest_rows.extend(ds["meta"].to_dict(orient="records"))
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(OUT_DIR, "compare_input_manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    print("[OK] Wrote outputs to:")
    print(" ", OUT_DIR)
    print("[OK] Main figures:")
    print("  - compare_dn_dt_FAPI_vs_FAPI-TEMPO.png")
    print("  - compare_Nt_FAPI_vs_FAPI-TEMPO.png")
    if made_active:
        print("  - compare_active_tracks_FAPI_vs_FAPI-TEMPO.png")
    if made_growth:
        print("  - compare_growth_rate_FAPI_vs_FAPI-TEMPO.png  (or growth proxy figure)")
    if made_tau:
        print("  - compare_tau_hist_FAPI_vs_FAPI-TEMPO.png")
    print("[OK] Tables:")
    print("  - compare_summary_metrics_FAPI_vs_FAPI-TEMPO.csv")
    print("  - compare_input_manifest.csv")
    print("  - rate_curve_FAPI.csv / rate_curve_FAPI-TEMPO.csv")
    print("  - Nt_curve_FAPI.csv / Nt_curve_FAPI-TEMPO.csv")

if __name__ == "__main__":
    main()