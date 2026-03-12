import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from matplotlib.collections import PolyCollection

# ---- PATHS ----
INP = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\radial_entropy_results.csv"
OUT_SUMMARY = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\paired_entropy_stats.csv"
OUT_DELTA_CSV = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\paired_entropy_deltas.csv"
OUT_PNG = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\paired_delta_entropy.png"

# ---- GROUPS ----
G_FAPI = "FAPI"
G_TEMPO = "FAPI-TEMPO"

# ---- ENTROPY SETTINGS (match what you used) ----
BINS = 256
MASK_EPS = 1e-12  # norm01 pixels > eps are "mask pixels"

# ---- PLOT STYLE ----
VIOLIN_ALPHA = 0.20
POINT_ALPHA = 0.70

def shannon_entropy_bits(values: np.ndarray, bins: int, vmin: float, vmax: float) -> float:
    if values.size == 0:
        return np.nan
    hist, _ = np.histogram(values, bins=bins, range=(vmin, vmax))
    p = hist.astype(np.float64)
    p = p[p > 0]
    if p.size == 0:
        return np.nan
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())

def load_norm01_png(path_png: str) -> np.ndarray:
    im = Image.open(path_png)
    im = im.convert("L")  # robust: use grayscale
    arr = np.asarray(im).astype(np.float32)
    if arr.max() > 1.0:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0)

def compute_full_entropy_raw_reconstructed_mask(row, base_dir) -> float:
    rel = str(row["heatmap_norm01_path"])
    png_path = rel if os.path.isabs(rel) else os.path.join(base_dir, rel)
    if not os.path.exists(png_path):
        return np.nan

    norm = load_norm01_png(png_path)
    mask = norm > MASK_EPS

    mn0 = float(row["heatmap_min_before"])
    mx0 = float(row["heatmap_max_before"])
    raw = norm * (mx0 - mn0) + mn0

    vals = raw[mask].ravel()
    return shannon_entropy_bits(vals, bins=BINS, vmin=mn0, vmax=mx0)

def cohen_dz(d: np.ndarray) -> float:
    d = d[np.isfinite(d)]
    if d.size < 2:
        return np.nan
    sd = np.std(d, ddof=1)
    if sd == 0:
        return np.nan
    return float(np.mean(d) / sd)

def rank_biserial_from_diffs(d: np.ndarray) -> float:
    """Rank-biserial correlation for paired diffs using Wilcoxon ranks."""
    d = d[np.isfinite(d)]
    d = d[d != 0]
    n = d.size
    if n == 0:
        return np.nan

    absd = np.abs(d)
    # average ranks for ties
    order = absd.argsort()
    ranks = np.empty(n, dtype=float)
    sorted_abs = absd[order]

    i = 0
    r = 1
    while i < n:
        j = i
        while j < n and sorted_abs[j] == sorted_abs[i]:
            j += 1
        avg_rank = (r + (r + (j - i) - 1)) / 2.0
        ranks[order[i:j]] = avg_rank
        r += (j - i)
        i = j

    Wpos = ranks[d > 0].sum()
    Wneg = ranks[d < 0].sum()
    denom = n * (n + 1) / 2.0
    return float((Wpos - Wneg) / denom)

def set_violin_alpha(ax, alpha: float):
    for coll in ax.collections:
        if isinstance(coll, PolyCollection):
            coll.set_alpha(alpha)

def main():
    # SciPy tests (with a clear error if missing)
    try:
        from scipy.stats import ttest_rel, wilcoxon
    except Exception as e:
        raise RuntimeError(
            "This script needs SciPy for paired t-test/Wilcoxon.\n"
            "Install: conda install scipy  (or pip install scipy)\n"
            f"Original error: {e}"
        )

    df = pd.read_csv(INP)
    base_dir = os.path.dirname(INP)

    # Required columns
    need = {
        "group", "pair_id",
        "entropy_inner_norm01", "entropy_middle_norm01", "entropy_outer_norm01",
        "heatmap_norm01_path", "heatmap_min_before", "heatmap_max_before"
    }
    missing = sorted(list(need - set(df.columns)))
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}\nFound: {list(df.columns)}")

    # Compute full entropy (raw reconstructed, mask pixels)
    df["full_entropy_raw_maskpix"] = df.apply(
        lambda r: compute_full_entropy_raw_reconstructed_mask(r, base_dir),
        axis=1
    )

    # Keep only the two groups & pairs that have both
    df = df[df["group"].isin([G_FAPI, G_TEMPO])].copy()

    # Build paired (wide) table per metric
    metrics = {
        "Full entropy (raw_reconstructed, mask)": "full_entropy_raw_maskpix",
        "Inner entropy (norm01)": "entropy_inner_norm01",
        "Middle entropy (norm01)": "entropy_middle_norm01",
        "Outer entropy (norm01)": "entropy_outer_norm01",
    }

    deltas_long = []
    stats_rows = []

    for label, col in metrics.items():
        wide = df.pivot_table(index="pair_id", columns="group", values=col, aggfunc="mean")
        wide = wide.dropna(subset=[G_FAPI, G_TEMPO])  # require both sides
        d = (wide[G_TEMPO] - wide[G_FAPI]).to_numpy()

        # paired tests
        t_res = ttest_rel(wide[G_TEMPO], wide[G_FAPI], nan_policy="omit")
        try:
            w_res = wilcoxon(d[d != 0])  # wilcoxon ignores zeros poorly, so we drop them
            w_stat = float(w_res.statistic)
            w_p = float(w_res.pvalue)
        except Exception:
            w_stat, w_p = np.nan, np.nan

        dz = cohen_dz(d)
        rrb = rank_biserial_from_diffs(d)

        stats_rows.append({
            "metric": label,
            "n_pairs": int(np.isfinite(d).sum()),
            "delta_mean(TEMPO-FAPI)": float(np.nanmean(d)),
            "delta_median(TEMPO-FAPI)": float(np.nanmedian(d)),
            "t_stat": float(t_res.statistic),
            "t_p": float(t_res.pvalue),
            "wilcoxon_W": w_stat,
            "wilcoxon_p": w_p,
            "cohen_dz": dz,
            "rank_biserial_r": rrb,
        })

        # long for plotting
        tmp = pd.DataFrame({
            "pair_id": wide.index.values,
            "metric": label,
            "delta": d
        })
        deltas_long.append(tmp)

    stats_df = pd.DataFrame(stats_rows)
    deltas_df = pd.concat(deltas_long, ignore_index=True)

    stats_df.to_csv(OUT_SUMMARY, index=False)
    deltas_df.to_csv(OUT_DELTA_CSV, index=False)

    print("Saved stats:", OUT_SUMMARY)
    print("Saved deltas:", OUT_DELTA_CSV)

    # ---- Plot Δ per pair ----
    sns.set_style("whitegrid")
    order = list(metrics.keys())

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
    fig.suptitle("Paired Δ entropy per pair (TEMPO − FAPI)", y=1.03)

    for ax, m in zip(axes, order):
        sub = deltas_df[deltas_df["metric"] == m].copy()

        # violin + points (single category), with transparency
        sns.violinplot(data=sub, x="metric", y="delta", inner=None, cut=0, linewidth=1, ax=ax)
        set_violin_alpha(ax, VIOLIN_ALPHA)

        sns.stripplot(data=sub, x="metric", y="delta", jitter=0.28, size=3, alpha=POINT_ALPHA, ax=ax)

        # zero reference line
        ax.axhline(0, linewidth=1.2, linestyle="--", color="k", alpha=0.6)

        # BLACK mean bar
        mu = float(sub["delta"].mean())
        ax.hlines(mu, -0.22, 0.22, linewidth=3.0, color="k", zorder=7)

        ax.set_xlabel("")
        ax.set_title(m.replace(" entropy", ""))
        ax.set_xticklabels([""])  # cleaner
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("Δ Entropy (bits or norm01 units)")
    for ax in axes[1:]:
        ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=250, bbox_inches="tight")
    print("Saved plot:", OUT_PNG)

if __name__ == "__main__":
    main()
