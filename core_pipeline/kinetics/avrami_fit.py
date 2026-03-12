# avrami_fit.py
# Fit Avrami: ln[-ln(1-X)] = ln(k) + n ln(t)
# Uses counts_shifted.csv produced by your segfreeze pipeline.

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_CSV = r"sam\segfreeze_v1_fapi_vs_tempo\counts_shifted.csv"

def rolling_smooth(y, win):
    """Centered rolling median (robust). If win<=1, returns y."""
    if win is None or win <= 1:
        return y.copy()
    s = pd.Series(y)
    return s.rolling(window=win, center=True, min_periods=max(2, win // 2)).median().to_numpy()

def linear_fit(x, y):
    """Return slope, intercept, yhat, residuals, R^2."""
    # np.polyfit returns [slope, intercept]
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    resid = y - yhat
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y - np.mean(y))**2) if len(y) > 1 else np.nan
    r2 = 1 - ss_res / ss_tot if ss_tot not in (0, np.nan) else np.nan
    return slope, intercept, yhat, resid, r2

def avrami_transform(t, X):
    """Compute ln(t) and ln[-ln(1-X)], masking invalid points."""
    # Valid domain: t>0, 0<X<1
    mask = (t > 0) & (X > 0) & (X < 1)
    t2 = t[mask]
    X2 = X[mask]
    lx = np.log(t2)
    ly = np.log(-np.log(1 - X2))
    return mask, lx, ly

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV, help="Path to counts_shifted.csv")
    ap.add_argument("--xcol", default="sum_bbox_area_frac_kept", help="Column used as X proxy")
    ap.add_argument("--tcol", default="t_shifted_ms", help="Time column (shifted)")
    ap.add_argument("--xmin", type=float, default=0.05, help="Lower X bound for fit window")
    ap.add_argument("--xmax", type=float, default=0.80, help="Upper X bound for fit window")
    ap.add_argument("--smooth_win", type=int, default=7, help="Rolling median window for X (odd recommended)")
    ap.add_argument("--outdir", default=".", help="Output directory for plots/results")
    args = ap.parse_args()

    csv_path = args.csv
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}\nTip: pass --csv with full path")

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(csv_path)
    needed = {"dataset", args.tcol, args.xcol}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}\nColumns present: {df.columns.tolist()}")

    results = []

    for dataset, g in df.groupby("dataset"):
        g = g.sort_values(args.tcol).copy()

        # 1) Exclude pre-induction times
        g = g[g[args.tcol] >= 0].copy()
        if len(g) < 10:
            print(f"[WARN] {dataset}: too few points after t>=0 filter ({len(g)}). Skipping.")
            continue

        t = g[args.tcol].to_numpy(dtype=float)

        # 2) Build X(t) from chosen proxy, normalize per dataset
        raw = g[args.xcol].to_numpy(dtype=float)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        # Smooth proxy first (reduces frame-to-frame segmentation jitter)
        raw_s = rolling_smooth(raw, args.smooth_win)

        # Normalize by dataset max (robust: use 99th percentile to avoid a single outlier frame)
        denom = np.percentile(raw_s, 99)
        if denom <= 0:
            print(f"[WARN] {dataset}: {args.xcol} is ~0. Skipping.")
            continue

        X = raw_s / denom

        # Clip into [0, 0.999999] to keep logs stable
        X = np.clip(X, 0.0, 0.999999)

        # 3) Fit window selection: X in [xmin, xmax]
        win_mask = (X >= args.xmin) & (X <= args.xmax) & (t > 0)

        # If window is fragmented due to noise, keep the largest contiguous block
        idx = np.where(win_mask)[0]
        if len(idx) < 10:
            print(f"[WARN] {dataset}: not enough points in X-window [{args.xmin},{args.xmax}] (n={len(idx)}).")
            continue

        # Find largest contiguous run in idx
        runs = []
        start = idx[0]
        prev = idx[0]
        for i in idx[1:]:
            if i == prev + 1:
                prev = i
            else:
                runs.append((start, prev))
                start = i
                prev = i
        runs.append((start, prev))
        run_lengths = [(b - a + 1) for a, b in runs]
        best_run = runs[int(np.argmax(run_lengths))]
        a, b = best_run
        sel = np.zeros_like(win_mask, dtype=bool)
        sel[a:b+1] = True

        t_fit = t[sel]
        X_fit = X[sel]

        # 4) Avrami transform
        _, lx, ly = avrami_transform(t_fit, X_fit)
        if len(lx) < 8:
            print(f"[WARN] {dataset}: too few valid Avrami points after transform ({len(lx)}).")
            continue

        # 5) Linear regression
        n, ln_k, yhat, resid, r2 = linear_fit(lx, ly)
        k = float(np.exp(ln_k))

        # Save results
        results.append({
            "dataset": dataset,
            "n": float(n),
            "ln_k": float(ln_k),
            "k": float(k),
            "R2": float(r2),
            "t_fit_min_ms": float(np.min(t_fit)),
            "t_fit_max_ms": float(np.max(t_fit)),
            "X_fit_min": float(np.min(X_fit)),
            "X_fit_max": float(np.max(X_fit)),
            "n_points": int(len(lx)),
            "denom_used": float(denom),
            "xmin": args.xmin,
            "xmax": args.xmax,
            "smooth_win": args.smooth_win,
        })

        # ---- Plots ----
        # (A) Avrami plot
        plt.figure()
        plt.plot(lx, ly, marker="o", linestyle="None", markersize=3)
        xx = np.linspace(np.min(lx), np.max(lx), 200)
        plt.plot(xx, n*xx + ln_k)
        plt.title(f"{dataset}: Avrami fit  (n={n:.3f}, k={k:.3e}, R²={r2:.3f})")
        plt.xlabel("ln(t)")
        plt.ylabel("ln[-ln(1-X)]")
        out1 = os.path.join(args.outdir, f"avrami_{dataset}.png")
        plt.tight_layout()
        plt.savefig(out1, dpi=150)
        plt.close()

        # (B) Residuals vs ln(t)
        plt.figure()
        plt.plot(lx, resid, marker="o", linestyle="None", markersize=3)
        plt.axhline(0.0)
        plt.title(f"{dataset}: residuals (Avrami)")
        plt.xlabel("ln(t)")
        plt.ylabel("residual")
        out2 = os.path.join(args.outdir, f"avrami_residuals_{dataset}.png")
        plt.tight_layout()
        plt.savefig(out2, dpi=150)
        plt.close()

        # (C) Diagnostic: X(t) and fit window
        plt.figure()
        plt.plot(t, X, linewidth=1)
        plt.plot(t_fit, X_fit, linewidth=2)
        plt.title(f"{dataset}: X(t) (norm by p99={denom:.4g})  | fit window highlighted")
        plt.xlabel("t_shifted_ms")
        plt.ylabel("X (normalized)")
        out3 = os.path.join(args.outdir, f"avrami_Xwindow_{dataset}.png")
        plt.tight_layout()
        plt.savefig(out3, dpi=150)
        plt.close()

        print(f"[OK] {dataset}: n={n:.3f}, k={k:.3e}, R²={r2:.3f} | fit t=[{np.min(t_fit):.1f},{np.max(t_fit):.1f}] ms")

    if not results:
        print("[ERROR] No fits produced. Check xmin/xmax, time shift, and columns.")
        return

    res_df = pd.DataFrame(results).sort_values("dataset")
    out_csv = os.path.join(args.outdir, "avrami_results.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    main()