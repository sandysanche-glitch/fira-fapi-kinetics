#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


def robust_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def smooth(y: np.ndarray, sigma_pts: float = 2.0) -> np.ndarray:
    y = np.asarray(y, float)
    radius = max(1, int(round(4 * sigma_pts)))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-(x**2) / (2 * sigma_pts**2))
    k /= k.sum()
    return np.convolve(y, k, mode="same")


def parse_file_name(name: str, sample: str) -> tuple[str, str]:
    s = str(name)
    nums = re.findall(r"(\d+)", s)
    mg = nums[0] if len(nums) >= 1 else s
    roi = nums[1] if len(nums) >= 2 else "0"
    return f"{sample}_MG_{mg}", f"{sample}_ROI_{mg}_{roi}"


def prep(path: Path, sample: str) -> pd.DataFrame:
    raw = pd.read_csv(path)

    out = pd.DataFrame(index=raw.index)
    out["sample"] = sample
    out["file_name"] = raw["file_name"].astype(str)

    parsed = out["file_name"].apply(lambda x: parse_file_name(x, sample))
    out["micrograph_id"] = [a for a, b in parsed]
    out["roi_id"] = [b for a, b in parsed]

    colmap = {
        "area_(µm²)": "area_um2",
        "t0_ms": "t0_ms",
        "R_um_final": "R_um_final",
        "v_eff_um_per_ms": "v_eff_um_per_ms",
    }
    for src, dst in colmap.items():
        out[dst] = robust_num(raw[src]) if src in raw.columns else np.nan

    out["t0_ms"] = out["t0_ms"].clip(lower=0)
    out["count_weight"] = 1.0
    out["area_weight"] = out["area_um2"]
    out["R2_weight"] = out["R_um_final"] ** 2
    return out


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)

    ok = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    values = values[ok]
    weights = weights[ok]

    if len(values) == 0 or weights.sum() <= 0:
        return np.nan

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cum_w = np.cumsum(weights)
    target = q * weights.sum()
    idx = np.searchsorted(cum_w, target, side="left")
    idx = min(idx, len(values) - 1)
    return float(values[idx])


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    values = values[ok]
    weights = weights[ok]
    if len(values) == 0 or weights.sum() <= 0:
        return np.nan
    return float(np.sum(values * weights) / np.sum(weights))


def weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    mu = weighted_mean(values, weights)
    if not np.isfinite(mu):
        return np.nan
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    values = values[ok]
    weights = weights[ok]
    if len(values) == 0 or weights.sum() <= 0:
        return np.nan
    var = np.sum(weights * (values - mu) ** 2) / np.sum(weights)
    return float(np.sqrt(var))


def weighted_wasserstein_1d(
    x1: np.ndarray,
    w1: np.ndarray,
    x2: np.ndarray,
    w2: np.ndarray,
) -> float:
    x1 = np.asarray(x1, float)
    w1 = np.asarray(w1, float)
    x2 = np.asarray(x2, float)
    w2 = np.asarray(w2, float)

    ok1 = np.isfinite(x1) & np.isfinite(w1) & (w1 >= 0)
    ok2 = np.isfinite(x2) & np.isfinite(w2) & (w2 >= 0)
    x1, w1 = x1[ok1], w1[ok1]
    x2, w2 = x2[ok2], w2[ok2]

    if len(x1) == 0 or len(x2) == 0 or w1.sum() <= 0 or w2.sum() <= 0:
        return np.nan

    w1 = w1 / w1.sum()
    w2 = w2 / w2.sum()

    xs = np.sort(np.unique(np.concatenate([x1, x2])))
    if len(xs) < 2:
        return 0.0

    def cdf_at(grid: np.ndarray, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        order = np.argsort(x)
        x = x[order]
        w = w[order]
        cw = np.cumsum(w)
        idx = np.searchsorted(x, grid, side="right") - 1
        out = np.zeros_like(grid, dtype=float)
        good = idx >= 0
        out[good] = cw[np.clip(idx[good], 0, len(cw) - 1)]
        return out

    f1 = cdf_at(xs, x1, w1)
    f2 = cdf_at(xs, x2, w2)
    dx = np.diff(xs)
    avg_diff = np.abs(f1[:-1] - f2[:-1])
    return float(np.sum(avg_diff * dx))


def normalize_density(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    t = np.asarray(t, float)
    y = np.clip(y, 0, None)
    area = np.trapz(y, t)
    if area <= 0 or not np.isfinite(area):
        return np.full_like(y, np.nan)
    return y / area


def jensen_shannon_distance_from_curves(y1: np.ndarray, y2: np.ndarray, t: np.ndarray) -> float:
    p = normalize_density(y1, t)
    q = normalize_density(y2, t)
    ok = np.isfinite(p) & np.isfinite(q)
    p = p[ok]
    q = q[ok]
    if len(p) == 0:
        return np.nan
    eps = 1e-12
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    js = 0.5 * (kl_pm + kl_qm)
    return float(np.sqrt(js))


def reconstruct(gs: pd.DataFrame, weight_col: str, n_grid: int = 500) -> pd.DataFrame:
    g = gs[["t0_ms", weight_col]].dropna().sort_values("t0_ms")
    g = g[g[weight_col] >= 0]

    if len(g) < 2:
        raise ValueError(f"Not enough data for reconstruction using {weight_col}")

    t = g["t0_ms"].to_numpy(float)
    w = g[weight_col].to_numpy(float)

    t_grid = np.linspace(t.min(), t.max(), n_grid)
    n = np.searchsorted(t, t_grid, side="right").astype(float)

    cws = np.cumsum(w)
    idx = np.searchsorted(t, t_grid, side="right") - 1
    x_num = np.where(idx >= 0, cws[np.clip(idx, 0, len(cws) - 1)], 0.0)
    X = x_num / w.sum() if w.sum() > 0 else np.full_like(t_grid, np.nan)

    dn_dt = smooth(np.gradient(n, t_grid), 2.0)
    dX_dt = smooth(np.gradient(X, t_grid), 2.0)

    return pd.DataFrame(
        {
            "time_ms": t_grid,
            "n": n,
            "dn_dt": dn_dt,
            "X": X,
            "dX_dt": dX_dt,
        }
    )


def inverse_time_at_fraction(t: np.ndarray, X: np.ndarray, frac: float) -> float:
    t = np.asarray(t, float)
    X = np.asarray(X, float)
    ok = np.isfinite(t) & np.isfinite(X)
    t = t[ok]
    X = X[ok]
    if len(t) < 2:
        return np.nan
    Xm = np.maximum.accumulate(np.clip(X, 0, 1))
    ux, idx = np.unique(Xm, return_index=True)
    ut = t[idx]
    if ux[0] > 0:
        ux = np.r_[0.0, ux]
        ut = np.r_[t[0], ut]
    if ux[-1] < 1:
        ux = np.r_[ux, 1.0]
        ut = np.r_[ut, t[-1]]
    return float(np.interp(frac, ux, ut))


def halfmax_width(t: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if len(t) < 3 or not np.isfinite(y).any():
        return np.nan, np.nan, np.nan

    peak_idx = int(np.nanargmax(y))
    peak_val = y[peak_idx]
    if not np.isfinite(peak_val) or peak_val <= 0:
        return np.nan, np.nan, np.nan

    half = 0.5 * peak_val
    above = np.where(y >= half)[0]
    if len(above) < 2:
        return np.nan, np.nan, np.nan

    left_idx = above[0]
    right_idx = above[-1]
    left_w = t[peak_idx] - t[left_idx]
    right_w = t[right_idx] - t[peak_idx]
    fwhm = t[right_idx] - t[left_idx]
    return float(left_w), float(right_w), float(fwhm)


def curve_metrics(curve: pd.DataFrame) -> dict[str, float]:
    t = curve["time_ms"].to_numpy(float)
    X = curve["X"].to_numpy(float)
    R = curve["dX_dt"].to_numpy(float)

    t10 = inverse_time_at_fraction(t, X, 0.10)
    t50 = inverse_time_at_fraction(t, X, 0.50)
    t90 = inverse_time_at_fraction(t, X, 0.90)

    peak_idx = int(np.nanargmax(R))
    peak_time = float(t[peak_idx])
    peak_height = float(R[peak_idx])

    left_w, right_w, fwhm = halfmax_width(t, R)
    asym_ratio = right_w / left_w if np.isfinite(left_w) and left_w > 0 else np.nan

    return {
        "t10_ms": t10,
        "t50_ms": t50,
        "t90_ms": t90,
        "rise_10_90_ms": t90 - t10 if np.isfinite(t10) and np.isfinite(t90) else np.nan,
        "early_mid_width_ms": t50 - t10 if np.isfinite(t10) and np.isfinite(t50) else np.nan,
        "late_tail_width_ms": t90 - t50 if np.isfinite(t50) and np.isfinite(t90) else np.nan,
        "peak_time_dXdt_ms": peak_time,
        "peak_height_dXdt": peak_height,
        "left_halfwidth_ms": left_w,
        "right_halfwidth_ms": right_w,
        "FWHM_dXdt_ms": fwhm,
        "peak_asymmetry_right_over_left": asym_ratio,
    }


def weighted_t0_metrics(gs: pd.DataFrame, weight_col: str) -> dict[str, float]:
    g = gs[["t0_ms", weight_col]].dropna()
    t = g["t0_ms"].to_numpy(float)
    w = g[weight_col].to_numpy(float)

    q25 = weighted_quantile(t, w, 0.25)
    q50 = weighted_quantile(t, w, 0.50)
    q75 = weighted_quantile(t, w, 0.75)

    return {
        "weighted_t0_mean_ms": weighted_mean(t, w),
        "weighted_t0_median_ms": q50,
        "weighted_t0_std_ms": weighted_std(t, w),
        "weighted_t0_IQR_ms": q75 - q25 if np.isfinite(q25) and np.isfinite(q75) else np.nan,
    }


def area_between_curves(t: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> float:
    return float(np.trapz(np.abs(y1 - y2), t))


def build_curves_and_metrics(
    fapi_path: Path,
    tempo_path: Path,
    n_grid: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fapi = prep(fapi_path, "FAPI")
    tempo = prep(tempo_path, "FAPI-TEMPO")
    df = pd.concat([fapi, tempo], ignore_index=True)

    curves_rows = []
    sample_metric_rows = []

    weightings = [
        ("count_weighted", "count_weight"),
        ("area_weighted", "area_weight"),
        ("R2_weighted", "R2_weight"),
    ]

    for sample in ["FAPI", "FAPI-TEMPO"]:
        gs = df[df["sample"] == sample].copy()
        for weighting_name, weight_col in weightings:
            curve = reconstruct(gs, weight_col, n_grid=n_grid)
            curve["sample"] = sample
            curve["weighting"] = weighting_name
            curves_rows.append(curve)

            m = curve_metrics(curve)
            m.update(weighted_t0_metrics(gs, weight_col))
            m["sample"] = sample
            m["weighting"] = weighting_name
            sample_metric_rows.append(m)

    curves = pd.concat(curves_rows, ignore_index=True)
    per_sample_metrics = pd.DataFrame(sample_metric_rows)

    pair_rows = []
    for weighting_name, weight_col in weightings:
        cf = curves[(curves["sample"] == "FAPI") & (curves["weighting"] == weighting_name)].sort_values("time_ms")
        ct = curves[(curves["sample"] == "FAPI-TEMPO") & (curves["weighting"] == weighting_name)].sort_values("time_ms")

        t = cf["time_ms"].to_numpy(float)
        x1 = cf["X"].to_numpy(float)
        x2 = ct["X"].to_numpy(float)
        r1 = cf["dX_dt"].to_numpy(float)
        r2 = ct["dX_dt"].to_numpy(float)

        gf = df[df["sample"] == "FAPI"][["t0_ms", weight_col]].dropna()
        gt = df[df["sample"] == "FAPI-TEMPO"][["t0_ms", weight_col]].dropna()

        pair_rows.append(
            {
                "weighting": weighting_name,
                "AUC_abs_diff_X": area_between_curves(t, x1, x2),
                "AUC_abs_diff_dXdt": area_between_curves(t, r1, r2),
                "JS_distance_dXdt": jensen_shannon_distance_from_curves(r1, r2, t),
                "Wasserstein_t0_ms": weighted_wasserstein_1d(
                    gf["t0_ms"].to_numpy(float),
                    gf[weight_col].to_numpy(float),
                    gt["t0_ms"].to_numpy(float),
                    gt[weight_col].to_numpy(float),
                ),
            }
        )

    pair_metrics = pd.DataFrame(pair_rows)
    return curves, per_sample_metrics, pair_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fapi", type=Path, required=True, help="Path to morpho_kinetics_from_cm_full_FAPI_with_veff.csv")
    parser.add_argument("--tempo", type=Path, required=True, help="Path to morpho_kinetics_from_cm_full_FAPITEMPO_with_veff.csv")
    parser.add_argument("--outdir", type=Path, default=Path("bridge_metrics_out"))
    parser.add_argument("--n_grid", type=int, default=500)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    curves, per_sample_metrics, pair_metrics = build_curves_and_metrics(
        fapi_path=args.fapi,
        tempo_path=args.tempo,
        n_grid=args.n_grid,
    )

    curves.to_csv(args.outdir / "reconstructed_curves_with_veff.csv", index=False)
    per_sample_metrics.to_csv(args.outdir / "per_sample_curve_metrics.csv", index=False)
    pair_metrics.to_csv(args.outdir / "pairwise_curve_comparison_metrics.csv", index=False)

    # Main-text compact table: area-weighted focus
    main_cols = [
        "sample",
        "t50_ms",
        "rise_10_90_ms",
        "peak_time_dXdt_ms",
        "peak_height_dXdt",
        "FWHM_dXdt_ms",
    ]
    main_table = per_sample_metrics[per_sample_metrics["weighting"] == "area_weighted"][main_cols].copy()
    main_table.to_csv(args.outdir / "main_text_curve_metrics_area_weighted.csv", index=False)

    print(f"Wrote outputs to: {args.outdir.resolve()}")
    print("\nArea-weighted main metrics:")
    print(main_table.to_string(index=False))
    print("\nPairwise comparison metrics:")
    print(pair_metrics.to_string(index=False))


if __name__ == "__main__":
    main()