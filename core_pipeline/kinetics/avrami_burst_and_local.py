# avrami_burst_and_local.py
# ------------------------------------------------------------
# Burst-window Avrami fit for FAPI_TEMPO + local (instantaneous) Avrami exponent n(t)
# - Reads counts_shifted.csv (from your segfreeze checkpoint)
# - Builds X from sum_bbox_area_frac_kept normalized by p99 (robust)
# - Enforces monotonic X via running maximum
# - TEMPO: detects burst onset (dn/dt peak OR d2X/dt2 peak) and fits a tight window
# - FAPI (and optionally TEMPO): computes local n(t) = d ln[-ln(1-X)] / d ln t using sliding regression
#
# Outputs:
#   avrami_burst_results.csv
#   local_avrami_n.csv
#   figures:
#     qc_X_mono_<dataset>.png
#     tempo_burst_fit.png
#     tempo_burst_residuals.png
#     local_n_<dataset>.png
#
# Usage:
#   python avrami_burst_and_local.py --csv "sam/segfreeze_v1_fapi_vs_tempo/counts_shifted.csv"
#   python avrami_burst_and_local.py --csv ... --tempo_method d2Xdt2 --tempo_halfwin 10
#   python avrami_burst_and_local.py --csv ... --local_halfwin 7
# ------------------------------------------------------------

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def safe_mkdir(p):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def robust_p99(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return np.quantile(x, 0.99)


def enforce_monotonic_running_max(x):
    x = np.asarray(x, float)
    out = np.copy(x)
    # propagate NaNs safely (cummax ignores NaN poorly), so fill NaN with -inf then restore
    nanmask = ~np.isfinite(out)
    tmp = np.copy(out)
    tmp[nanmask] = -np.inf
    tmp = np.maximum.accumulate(tmp)
    tmp[nanmask] = np.nan
    return tmp


def gradient_safe(y, t):
    """First derivative dy/dt using np.gradient, guarding against repeated t."""
    y = np.asarray(y, float)
    t = np.asarray(t, float)

    # If any repeated t, add a tiny epsilon to make strictly increasing locally
    t2 = t.copy()
    for i in range(1, len(t2)):
        if not np.isfinite(t2[i]) or not np.isfinite(t2[i - 1]):
            continue
        if t2[i] <= t2[i - 1]:
            t2[i] = t2[i - 1] + 1e-9

    return np.gradient(y, t2)


def avrami_transform(X):
    """Return y = ln[-ln(1-X)] with safe clipping."""
    X = np.asarray(X, float)
    eps = 1e-9
    Xc = np.clip(X, eps, 1 - eps)
    return np.log(-np.log(1 - Xc))


def linear_fit(x, y):
    """y = a + b x; returns b, a, r2, yhat"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        return np.nan, np.nan, np.nan, None

    A = np.vstack([x, np.ones_like(x)]).T
    b, a = np.linalg.lstsq(A, y, rcond=None)[0]  # y = b*x + a
    yhat = b * x + a

    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return b, a, r2, (x, y, yhat)


def select_X_range(df, Xcol="X_mono", xmin=0.05, xmax=0.8):
    X = df[Xcol].to_numpy(float)
    return (np.isfinite(X)) & (X >= xmin) & (X <= xmax)


def detect_burst_onset(df, method="dndt", smooth_win=7):
    """
    method:
      - 'dndt'   : peak of smoothed dn/dt (n_kept)
      - 'd2Xdt2' : peak of smoothed d2X/dt2 (X_mono)
    returns index (row position in df, not frame id).
    """
    t = df["t_pos_ms"].to_numpy(float)

    if method == "dndt":
        n = df["n_kept"].to_numpy(float)
        d1 = gradient_safe(n, t)
        d1s = pd.Series(d1).rolling(smooth_win, center=True, min_periods=1).mean().to_numpy()
        # onset is where derivative is maximal
        idx = int(np.nanargmax(d1s))
        return idx

    if method == "d2Xdt2":
        X = df["X_mono"].to_numpy(float)
        d1 = gradient_safe(X, t)
        d2 = gradient_safe(d1, t)
        d2s = pd.Series(d2).rolling(smooth_win, center=True, min_periods=1).mean().to_numpy()
        idx = int(np.nanargmax(d2s))
        return idx

    raise ValueError(f"Unknown method: {method}")


def burst_window_indices(df, onset_idx, halfwin=10, Xmin=0.05, Xmax=0.8):
    """
    Start from an onset-centered index window, then filter to X-range and finite.
    Ensures positive time and valid X.
    """
    i0 = max(0, onset_idx - halfwin)
    i1 = min(len(df), onset_idx + halfwin + 1)
    sub = df.iloc[i0:i1].copy()

    m = select_X_range(sub, "X_mono", Xmin, Xmax)
    # also require finite transforms and positive time
    t = sub["t_pos_ms"].to_numpy(float)
    m &= np.isfinite(t) & (t > 0)

    return sub.loc[m].copy(), (i0, i1)


def local_avrami_n(df, halfwin=7, Xmin=0.05, Xmax=0.8):
    """
    Local n(t) via sliding linear regression of:
      y = ln[-ln(1-X)]
      x = ln(t)
    on a window of size (2*halfwin+1)
    returns df with columns: ln_t, avrami_y, n_local, r2_local
    """
    out = df.copy()

    t = out["t_pos_ms"].to_numpy(float)
    X = out["X_mono"].to_numpy(float)

    good = np.isfinite(t) & (t > 0) & np.isfinite(X) & (X >= Xmin) & (X <= Xmax)
    ln_t = np.full_like(t, np.nan, dtype=float)
    yA = np.full_like(t, np.nan, dtype=float)
    ln_t[good] = np.log(t[good])
    yA[good] = avrami_transform(X[good])

    n_local = np.full_like(t, np.nan, dtype=float)
    r2_local = np.full_like(t, np.nan, dtype=float)

    for i in range(len(out)):
        j0 = max(0, i - halfwin)
        j1 = min(len(out), i + halfwin + 1)

        xw = ln_t[j0:j1]
        yw = yA[j0:j1]
        m = np.isfinite(xw) & np.isfinite(yw)
        if np.sum(m) < 5:
            continue

        slope, intercept, r2, _ = linear_fit(xw[m], yw[m])
        n_local[i] = slope
        r2_local[i] = r2

    out["ln_t"] = ln_t
    out["avrami_y"] = yA
    out["n_local"] = n_local
    out["r2_local"] = r2_local
    return out


def plot_X_mono(df, dataset, outdir):
    plt.figure()
    plt.plot(df["t_shifted_ms"], df["X_raw"], label="X raw")
    plt.plot(df["t_shifted_ms"], df["X_mono"], label="X mono (running max)")
    plt.xlabel("t_shifted_ms")
    plt.ylabel("X (normalized)")
    plt.title(f"{dataset}: X(t) raw vs monotonic")
    plt.legend()
    fn = os.path.join(outdir, f"qc_X_mono_{dataset}.png")
    plt.savefig(fn, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to counts_shifted.csv")
    ap.add_argument("--outdir", default="avrami_out", help="Output folder")
    ap.add_argument("--Xmin", type=float, default=0.05, help="Lower X bound for Avrami transforms")
    ap.add_argument("--Xmax", type=float, default=0.8, help="Upper X bound for Avrami transforms")
    ap.add_argument("--tempo_method", choices=["dndt", "d2Xdt2"], default="dndt",
                    help="Burst onset detector for TEMPO")
    ap.add_argument("--tempo_halfwin", type=int, default=10, help="Half-window (frames) around onset for TEMPO fit")
    ap.add_argument("--smooth_win", type=int, default=7, help="Smoothing window for onset detector")
    ap.add_argument("--local_halfwin", type=int, default=7, help="Half-window for local n(t) regression")
    args = ap.parse_args()

    safe_mkdir(args.outdir)

    df = pd.read_csv(args.csv)
    need_cols = ["dataset", "t_shifted_ms", "n_kept", "sum_bbox_area_frac_kept"]
    for c in need_cols:
        if c not in df.columns:
            raise SystemExit(f"Missing column '{c}'. Columns are: {df.columns.tolist()}")

    results_rows = []
    local_rows = []

    for dataset in sorted(df["dataset"].unique()):
        d = df[df["dataset"] == dataset].copy()

        # sort by shifted time
        d = d.sort_values("t_shifted_ms").reset_index(drop=True)

        # Positive time axis for Avrami (exclude negative induction time)
        d["t_pos_ms"] = d["t_shifted_ms"].astype(float)
        # We'll keep the original for plotting, but for Avrami we require t_pos_ms > 0

        # Robust normalization -> X_raw
        p99 = robust_p99(d["sum_bbox_area_frac_kept"].to_numpy(float))
        if not np.isfinite(p99) or p99 <= 0:
            raise SystemExit(f"{dataset}: invalid p99 normalization = {p99}")

        d["X_raw"] = d["sum_bbox_area_frac_kept"].astype(float) / float(p99)
        d["X_raw"] = d["X_raw"].clip(lower=0.0, upper=1.5)  # allow slight overshoot pre-mono
        d["X_mono"] = enforce_monotonic_running_max(d["X_raw"].to_numpy(float))
        d["X_mono"] = np.clip(d["X_mono"], 0.0, 0.999999999)  # avoid 1 exactly

        plot_X_mono(d, dataset, args.outdir)

        # ---------- Local n(t) for ALL datasets (useful for FAPI especially) ----------
        dloc = local_avrami_n(d, halfwin=args.local_halfwin, Xmin=args.Xmin, Xmax=args.Xmax)
        # store
        tmp = dloc[["dataset", "frame", "t_shifted_ms", "t_pos_ms", "X_raw", "X_mono", "n_local", "r2_local"]].copy()
        local_rows.append(tmp)

        # plot local n(t)
        plt.figure()
        plt.plot(dloc["t_shifted_ms"], dloc["n_local"])
        plt.xlabel("t_shifted_ms (ms)")
        plt.ylabel("local n(t)")
        plt.title(f"{dataset}: local Avrami exponent n(t)  (halfwin={args.local_halfwin})")
        fn = os.path.join(args.outdir, f"local_n_{dataset}.png")
        plt.savefig(fn, dpi=200, bbox_inches="tight")
        plt.close()

        # ---------- Burst Avrami fit only for TEMPO ----------
        if dataset.upper() == "FAPI_TEMPO":
            # Restrict for onset detection to t>0 (avoid negative induction)
            dp = d[d["t_pos_ms"] > 0].copy().reset_index(drop=True)

            onset_idx = detect_burst_onset(dp, method=args.tempo_method, smooth_win=args.smooth_win)
            sub, (i0, i1) = burst_window_indices(dp, onset_idx, halfwin=args.tempo_halfwin,
                                                Xmin=args.Xmin, Xmax=args.Xmax)

            # Avrami regression in transformed space
            t = sub["t_pos_ms"].to_numpy(float)
            X = sub["X_mono"].to_numpy(float)

            ln_t = np.log(t)
            yA = avrami_transform(X)

            slope, intercept, r2, pack = linear_fit(ln_t, yA)
            n_fit = slope
            k_fit = np.exp(intercept)  # because y = ln(k) + n ln(t)

            results_rows.append({
                "dataset": dataset,
                "tempo_method": args.tempo_method,
                "tempo_halfwin": args.tempo_halfwin,
                "Xmin": args.Xmin,
                "Xmax": args.Xmax,
                "p99_norm": p99,
                "onset_idx_in_pos": int(onset_idx),
                "onset_t_ms": float(dp.loc[onset_idx, "t_pos_ms"]),
                "fit_i0": int(i0),
                "fit_i1": int(i1),
                "fit_n_points": int(len(sub)),
                "n_avrami": float(n_fit),
                "k_avrami": float(k_fit),
                "r2": float(r2)
            })

            # Plot Avrami fit
            if pack is not None:
                xfit, yfit, yhat = pack
                plt.figure()
                plt.scatter(ln_t, yA, s=18)
                # line across fit domain
                xs = np.linspace(np.nanmin(xfit), np.nanmax(xfit), 100)
                plt.plot(xs, n_fit * xs + intercept)
                plt.xlabel("ln(t)")
                plt.ylabel("ln[-ln(1-X)]")
                plt.title(f"{dataset}: burst Avrami fit  n={n_fit:.3f}, k={k_fit:.3e}, R²={r2:.3f}")
                fn = os.path.join(args.outdir, "tempo_burst_fit.png")
                plt.savefig(fn, dpi=200, bbox_inches="tight")
                plt.close()

                # residuals
                plt.figure()
                resid = yfit - yhat
                plt.scatter(xfit, resid, s=18)
                plt.axhline(0)
                plt.xlabel("ln(t)")
                plt.ylabel("residual")
                plt.title(f"{dataset}: burst Avrami residuals")
                fn = os.path.join(args.outdir, "tempo_burst_residuals.png")
                plt.savefig(fn, dpi=200, bbox_inches="tight")
                plt.close()

            # Plot X(t) and highlight selected window
            plt.figure()
            plt.plot(dp["t_pos_ms"], dp["X_mono"], label="X_mono")
            if len(sub) > 0:
                plt.plot(sub["t_pos_ms"], sub["X_mono"], linewidth=3, label="fit window")
            plt.xlabel("t_pos_ms (ms)")
            plt.ylabel("X")
            plt.title(f"{dataset}: X(t) with burst-fit window (method={args.tempo_method})")
            plt.legend()
            fn = os.path.join(args.outdir, "tempo_burst_window_on_X.png")
            plt.savefig(fn, dpi=200, bbox_inches="tight")
            plt.close()

    # Write outputs
    if results_rows:
        pd.DataFrame(results_rows).to_csv(os.path.join(args.outdir, "avrami_burst_results.csv"), index=False)
        print("Saved:", os.path.join(args.outdir, "avrami_burst_results.csv"))
    else:
        print("No burst results written (did not find dataset == FAPI_TEMPO).")

    if local_rows:
        pd.concat(local_rows, ignore_index=True).to_csv(os.path.join(args.outdir, "local_avrami_n.csv"), index=False)
        print("Saved:", os.path.join(args.outdir, "local_avrami_n.csv"))

    print("Done. Figures are in:", args.outdir)


if __name__ == "__main__":
    main()