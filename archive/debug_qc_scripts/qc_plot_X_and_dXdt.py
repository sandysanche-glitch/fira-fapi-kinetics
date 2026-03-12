# qc_plot_X_and_dXdt.py
# Plot X(t)=sum_bbox_area_frac_kept and its derivative dX/dt from counts_shifted.csv
# Robust to duplicated / unsorted / irregular time samples.
#
# Usage (Windows cmd / miniconda):
#   python qc_plot_X_and_dXdt.py
#   python qc_plot_X_and_dXdt.py --csv sam\segfreeze_v1_fapi_vs_tempo\counts_shifted.csv
#   python qc_plot_X_and_dXdt.py --dataset FAPI --outdir qc_plots
#
# Outputs (PNG):
#   qc_X_vs_t_<DATASET>.png
#   qc_dXdt_vs_t_<DATASET>.png
#   qc_n_vs_t_<DATASET>.png
#   qc_dndt_vs_t_<DATASET>.png

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def centered_rolling_mean(y: np.ndarray, win: int) -> np.ndarray:
    """Centered rolling mean with edge handling."""
    if win <= 1:
        return y.astype(float)
    s = pd.Series(y.astype(float))
    return s.rolling(win, center=True, min_periods=max(2, win // 3)).mean().to_numpy()


def safe_gradient(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute dy/dt safely:
      - requires strictly increasing t
      - uses np.gradient on cleaned arrays
    """
    if len(y) < 2:
        return np.full_like(y, np.nan, dtype=float)
    return np.gradient(y.astype(float), t.astype(float))


def load_and_clean(csv_path: str, dataset: str | None, tcol: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Optional dataset filter
    if dataset is not None:
        if "dataset" not in df.columns:
            raise ValueError("CSV has no 'dataset' column, but --dataset was provided.")
        df = df[df["dataset"].astype(str) == str(dataset)].copy()
        if df.empty:
            raise ValueError(f"No rows found for dataset={dataset} in {csv_path}")

    # Ensure numeric time
    if tcol not in df.columns:
        raise ValueError(f"Time column '{tcol}' not found. Columns: {df.columns.tolist()}")
    df[tcol] = pd.to_numeric(df[tcol], errors="coerce")

    # Drop rows with NaN time
    df = df.dropna(subset=[tcol]).copy()

    # Sort by time
    df = df.sort_values(tcol).reset_index(drop=True)

    # Collapse duplicate times (THIS fixes your divide-by-zero / inf gradient warnings)
    # We take mean of numeric columns at the same time stamp.
    # Keep dataset string if present.
    group_cols = [tcol]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Make sure tcol is included as numeric (it is)
    agg = {c: "mean" for c in numeric_cols}
    df_g = df.groupby(group_cols, as_index=False).agg(agg)

    # If dataset column exists and you want to keep it:
    if "dataset" in df.columns:
        # fill with a constant if filtered, otherwise keep as 'mixed'
        if dataset is not None:
            df_g["dataset"] = dataset
        else:
            df_g["dataset"] = "mixed"

    # If frame exists, it got averaged; that’s ok for plotting vs time.
    return df_g


def plot_series(t, y, title, xlabel, ylabel, outpath):
    plt.figure()
    plt.plot(t, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=r"sam\segfreeze_v1_fapi_vs_tempo\counts_shifted.csv",
                    help="Path to counts_shifted.csv")
    ap.add_argument("--dataset", default=None, help="Dataset name, e.g. FAPI or FAPI_TEMPO (optional)")
    ap.add_argument("--tcol", default="t_shifted_ms", help="Time column to use (default: t_shifted_ms)")
    ap.add_argument("--xcol", default="sum_bbox_area_frac_kept", help="X column (default: sum_bbox_area_frac_kept)")
    ap.add_argument("--ncol", default="n_kept", help="Count column (default: n_kept)")
    ap.add_argument("--smooth_win", type=int, default=7, help="Smoothing window for derivatives (default: 7)")
    ap.add_argument("--outdir", default=".", help="Output directory for plots (default: current dir)")
    args = ap.parse_args()

    csv_path = args.csv
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    df = load_and_clean(csv_path, args.dataset, args.tcol)

    # Basic checks
    for col in [args.xcol, args.ncol]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing. Columns: {df.columns.tolist()}")

    t = df[args.tcol].to_numpy(dtype=float)

    # Ensure strictly increasing t (after grouping it should be, but double-check)
    dt = np.diff(t)
    if np.any(dt <= 0):
        # Last-resort fix: drop non-increasing steps
        keep = np.ones(len(t), dtype=bool)
        for i in range(1, len(t)):
            if t[i] <= t[i-1]:
                keep[i] = False
        df = df[keep].reset_index(drop=True)
        t = df[args.tcol].to_numpy(dtype=float)

    ds_label = args.dataset if args.dataset is not None else str(df.get("dataset", ["mixed"])[0])

    print(f"Loaded: {os.path.abspath(csv_path)}")
    print("Rows after clean/group:", len(df))
    print("Columns:", df.columns.tolist())
    print("Dataset label:", ds_label)
    print("Time range:", float(t.min()), "to", float(t.max()), "ms")
    if len(t) >= 2:
        print("dt min/median/max:", float(np.min(np.diff(t))), float(np.median(np.diff(t))), float(np.max(np.diff(t))))

    # ---- X(t) ----
    X = df[args.xcol].to_numpy(dtype=float)
    out_X = os.path.join(outdir, f"qc_X_vs_t_{ds_label}.png")
    plot_series(
        t, X,
        title=f"{ds_label}: X vs shifted time",
        xlabel=f"{args.tcol} (ms)",
        ylabel=args.xcol,
        outpath=out_X,
    )
    print("Saved:", out_X)

    # ---- dX/dt ----
    X_s = centered_rolling_mean(X, args.smooth_win)
    dXdt = safe_gradient(X_s, t)
    out_dX = os.path.join(outdir, f"qc_dXdt_vs_t_{ds_label}.png")
    plot_series(
        t, dXdt,
        title=f"{ds_label}: dX/dt (smoothed win={args.smooth_win})",
        xlabel=f"{args.tcol} (ms)",
        ylabel=f"d({args.xcol})/dt",
        outpath=out_dX,
    )
    print("Saved:", out_dX)

    # ---- n(t) ----
    n = df[args.ncol].to_numpy(dtype=float)
    out_n = os.path.join(outdir, f"qc_n_vs_t_{ds_label}.png")
    plot_series(
        t, n,
        title=f"{ds_label}: n_kept vs shifted time",
        xlabel=f"{args.tcol} (ms)",
        ylabel=args.ncol,
        outpath=out_n,
    )
    print("Saved:", out_n)

    # ---- dn/dt ----
    n_s = centered_rolling_mean(n, args.smooth_win)
    dndt = safe_gradient(n_s, t)
    out_dn = os.path.join(outdir, f"qc_dndt_vs_t_{ds_label}.png")
    plot_series(
        t, dndt,
        title=f"{ds_label}: dn/dt (smoothed win={args.smooth_win})",
        xlabel=f"{args.tcol} (ms)",
        ylabel=f"d({args.ncol})/dt",
        outpath=out_dn,
    )
    print("Saved:", out_dn)


if __name__ == "__main__":
    main()