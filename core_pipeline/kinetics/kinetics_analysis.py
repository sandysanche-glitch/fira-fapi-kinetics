import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics")
IN_FILES = {
    "FAPI": ROOT / "sam" / "FAPI" / "frame_kinetics.csv",
    "FAPI_TEMPO": ROOT / "sam" / "FAPI_TEMPO" / "frame_kinetics.csv",
}
OUT_DIR = ROOT / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Ensure numeric
    df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce")
    df["area_px"] = pd.to_numeric(df["area_px"], errors="coerce")
    df = df.dropna(subset=["time_ms", "area_px"])
    # Sort
    df = df.sort_values(["time_ms", "frame_id"]).reset_index(drop=True)
    return df


def compute_nucleation(df: pd.DataFrame, time_bin_ms: float = 5.0) -> pd.DataFrame:
    """
    n(t): cumulative nuclei count
    dn/dt: binned nucleation rate
    """
    nuc = df[df["class"] == "nucleus"].copy()
    if nuc.empty:
        return pd.DataFrame(columns=["time_ms", "n", "dn_dt_per_ms", "dn_dt_per_s"])

    # Each frame has per-frame ids; approximate nucleus birth by first appearance of a nucleus object.
    # Since IDs reset each frame, we instead use "count of nuclei per frame" and define births by increases.
    # Robust alternative: births = nuclei count in frame (if nuclei persist poorly); for kinetics, this works well.
    frame_n = nuc.groupby("time_ms").size().reset_index(name="nuclei_in_frame")
    frame_n = frame_n.sort_values("time_ms").reset_index(drop=True)

    # Cumulative nucleation count by counting new nuclei occurrences over time:
    # We take births as positive increments of nuclei_in_frame relative to the running maximum.
    running_max = frame_n["nuclei_in_frame"].cummax()
    n_cum = running_max
    frame_n["n"] = n_cum

    # Bin in time for dn/dt
    t0 = frame_n["time_ms"].min()
    t1 = frame_n["time_ms"].max()
    bins = np.arange(t0, t1 + time_bin_ms, time_bin_ms)
    frame_n["bin"] = pd.cut(frame_n["time_ms"], bins=bins, include_lowest=True)

    b = frame_n.groupby("bin", observed=True).agg(
        time_ms=("time_ms", "mean"),
        n=("n", "max"),
    ).reset_index(drop=True)

    # dn/dt on bins
    b["dn"] = b["n"].diff().fillna(b["n"])
    b["dt_ms"] = b["time_ms"].diff().fillna(time_bin_ms)
    b["dn_dt_per_ms"] = b["dn"] / b["dt_ms"]
    b["dn_dt_per_s"] = b["dn_dt_per_ms"] * 1000.0
    return b[["time_ms", "n", "dn_dt_per_ms", "dn_dt_per_s"]]


def compute_growth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Growth proxy from total cell area:
      A_total(t) = sum cell areas at time t
      R_eff(t) = sqrt(A_total/pi)
      v_eff(t) = dR_eff/dt
    This is robust even without persistent tracking IDs.
    """
    cells = df[df["class"] == "cell"].copy()
    if cells.empty:
        return pd.DataFrame(columns=["time_ms", "A_total", "R_eff", "v_eff_per_ms", "v_eff_per_s"])

    g = cells.groupby("time_ms").agg(A_total=("area_px", "sum")).reset_index()
    g = g.sort_values("time_ms").reset_index(drop=True)

    g["R_eff"] = np.sqrt(g["A_total"] / np.pi)

    # finite difference (central where possible)
    t = g["time_ms"].values
    R = g["R_eff"].values
    v = np.zeros_like(R, dtype=float)

    if len(R) >= 2:
        # forward/backward at ends, central inside
        v[0] = (R[1] - R[0]) / (t[1] - t[0])
        v[-1] = (R[-1] - R[-2]) / (t[-1] - t[-2])
        for i in range(1, len(R) - 1):
            v[i] = (R[i + 1] - R[i - 1]) / (t[i + 1] - t[i - 1])

    g["v_eff_per_ms"] = v
    g["v_eff_per_s"] = g["v_eff_per_ms"] * 1000.0
    return g[["time_ms", "A_total", "R_eff", "v_eff_per_ms", "v_eff_per_s"]]


def plot_series(label: str, nuc_df: pd.DataFrame, growth_df: pd.DataFrame):
    # n(t)
    if not nuc_df.empty:
        plt.figure()
        plt.plot(nuc_df["time_ms"], nuc_df["n"])
        plt.xlabel("Time (ms)")
        plt.ylabel("Cumulative nuclei n(t)")
        plt.title(f"{label}: n(t)")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{label}_n_t.png", dpi=200)
        plt.close()

        plt.figure()
        plt.plot(nuc_df["time_ms"], nuc_df["dn_dt_per_s"])
        plt.xlabel("Time (ms)")
        plt.ylabel("dn/dt (1/s)")
        plt.title(f"{label}: dn/dt (binned)")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{label}_dn_dt.png", dpi=200)
        plt.close()

    # growth
    if not growth_df.empty:
        plt.figure()
        plt.plot(growth_df["time_ms"], growth_df["R_eff"])
        plt.xlabel("Time (ms)")
        plt.ylabel("R_eff (px)")
        plt.title(f"{label}: effective radius R(t)")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{label}_R_t.png", dpi=200)
        plt.close()

        plt.figure()
        plt.plot(growth_df["time_ms"], growth_df["v_eff_per_s"])
        plt.xlabel("Time (ms)")
        plt.ylabel("v_eff (px/s)")
        plt.title(f"{label}: effective growth rate dR/dt")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{label}_v_t.png", dpi=200)
        plt.close()


all_summary = []

for label, path in IN_FILES.items():
    df = load_and_clean(path)

    nuc = compute_nucleation(df, time_bin_ms=5.0)
    growth = compute_growth(df)

    nuc.to_csv(OUT_DIR / f"{label}_nucleation.csv", index=False)
    growth.to_csv(OUT_DIR / f"{label}_growth.csv", index=False)

    plot_series(label, nuc, growth)

    # quick summary numbers
    summary = {
        "dataset": label,
        "t_start_ms": float(df["time_ms"].min()),
        "t_end_ms": float(df["time_ms"].max()),
        "max_n": float(nuc["n"].max()) if not nuc.empty else np.nan,
        "max_dn_dt_per_s": float(nuc["dn_dt_per_s"].max()) if not nuc.empty else np.nan,
        "max_R_eff_px": float(growth["R_eff"].max()) if not growth.empty else np.nan,
        "max_v_eff_px_per_s": float(growth["v_eff_per_s"].max()) if not growth.empty else np.nan,
    }
    all_summary.append(summary)

summary_df = pd.DataFrame(all_summary)
summary_df.to_csv(OUT_DIR / "summary_kinetics.csv", index=False)

print("Done.")
print(f"Outputs in: {OUT_DIR}")
