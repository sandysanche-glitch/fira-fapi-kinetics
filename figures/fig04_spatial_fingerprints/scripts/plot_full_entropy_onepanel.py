import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from matplotlib.collections import PolyCollection

# ---- INPUT / OUTPUT ----
INP = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\radial_entropy_results.csv"
OUT_PNG = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\full_entropy_onepanel.png"
OUT_CSV = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\radial_entropy_with_full_entropy.csv"

# ---- PLOT STYLE ----
GROUP_ORDER = ["FAPI", "FAPI-TEMPO"]
PALETTE = {"FAPI": "#1f77b4", "FAPI-TEMPO": "#ff7f0e"}   # keep previous colors
VIOLIN_ALPHA = 0.25
POINT_ALPHA = 0.70

# ---- ENTROPY SETTINGS ----
BINS = 256
MASK_EPS = 1e-12

# Choose what "full area" means:
USE_MASK_ONLY = True   # True = full mask pixels (recommended); False = whole image incl. background

# Choose entropy definition for comparison to your old “non-normalized” entropy:
# - "raw_reconstructed" uses min/max_before to reconstruct raw values from norm01 PNG.
# - "norm01" uses the PNG values directly in 0..1 range.
ENTROPY_MODE = "raw_reconstructed"   # or "norm01"

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
    if im.mode in ("RGB", "RGBA"):
        im = im.convert("L")
    else:
        im = im.convert("L")
    arr = np.asarray(im).astype(np.float32)
    if arr.max() > 1.0:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0)

def set_violin_alpha(ax, alpha: float):
    for coll in ax.collections:
        if isinstance(coll, PolyCollection):
            coll.set_alpha(alpha)

def main():
    df = pd.read_csv(INP)
    base_dir = os.path.dirname(INP)

    required = {"group", "heatmap_norm01_path"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}\nFound: {list(df.columns)}")

    if ENTROPY_MODE == "raw_reconstructed":
        needed_raw = {"heatmap_min_before", "heatmap_max_before"}
        miss_raw = sorted(list(needed_raw - set(df.columns)))
        if miss_raw:
            raise ValueError(
                f"ENTROPY_MODE='raw_reconstructed' requires columns: {miss_raw}\n"
                f"Found: {list(df.columns)}"
            )

    full_entropy = []
    n_ok, n_missing = 0, 0

    for _, row in df.iterrows():
        rel = str(row["heatmap_norm01_path"])
        png_path = rel if os.path.isabs(rel) else os.path.join(base_dir, rel)

        if not os.path.exists(png_path):
            full_entropy.append(np.nan)
            n_missing += 1
            continue

        norm = load_norm01_png(png_path)

        if USE_MASK_ONLY:
            vals_norm = norm[norm > MASK_EPS].ravel()
        else:
            vals_norm = norm.ravel()

        if ENTROPY_MODE == "norm01":
            e = shannon_entropy_bits(vals_norm, bins=BINS, vmin=0.0, vmax=1.0)

        elif ENTROPY_MODE == "raw_reconstructed":
            mn0 = float(row["heatmap_min_before"])
            mx0 = float(row["heatmap_max_before"])
            raw = norm * (mx0 - mn0) + mn0
            vals_raw = raw[norm > MASK_EPS].ravel() if USE_MASK_ONLY else raw.ravel()
            e = shannon_entropy_bits(vals_raw, bins=BINS, vmin=mn0, vmax=mx0)

        else:
            raise ValueError("ENTROPY_MODE must be 'raw_reconstructed' or 'norm01'.")

        full_entropy.append(e)
        n_ok += 1

    colname = f"full_entropy_{ENTROPY_MODE}_{'mask' if USE_MASK_ONLY else 'allpix'}"
    df[colname] = full_entropy
    df.to_csv(OUT_CSV, index=False)

    print("Loaded:", INP)
    print("Rows:", len(df), "OK PNGs:", n_ok, "Missing PNGs:", n_missing)
    print("Saved CSV:", OUT_CSV)

    # ---- Plot: ONE PANEL ----
    sub = df[df["group"].isin(GROUP_ORDER)].copy()

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))

    title_area = "mask pixels" if USE_MASK_ONLY else "all pixels"
    ax.set_title(f"Full Shannon entropy ({ENTROPY_MODE}, {title_area})")

    sns.violinplot(
        data=sub, x="group", y=colname,
        order=GROUP_ORDER,
        hue="group", hue_order=GROUP_ORDER,
        palette=PALETTE,
        inner=None, cut=0, linewidth=1,
        legend=False, ax=ax
    )
    set_violin_alpha(ax, VIOLIN_ALPHA)

    sns.stripplot(
        data=sub, x="group", y=colname,
        order=GROUP_ORDER,
        hue="group", hue_order=GROUP_ORDER,
        palette=PALETTE,
        jitter=0.28, size=3, alpha=POINT_ALPHA,
        dodge=False, legend=False, ax=ax
    )

    # BLACK mean / min / max bars
    stats = sub.groupby("group")[colname].agg(["mean", "min", "max"]).reindex(GROUP_ORDER)
    for i, g in enumerate(GROUP_ORDER):
        mu = float(stats.loc[g, "mean"])
        mn = float(stats.loc[g, "min"])
        mx = float(stats.loc[g, "max"])
        ax.hlines(mu, i - 0.22, i + 0.22, linewidth=3.0, color="k", zorder=7)  # mean
        ax.hlines(mn, i - 0.18, i + 0.18, linewidth=1.3, color="k", zorder=7)  # min
        ax.hlines(mx, i - 0.18, i + 0.18, linewidth=1.3, color="k", zorder=7)  # max

    ax.set_xlabel("")
    ax.set_ylabel("Entropy (bits)")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=250, bbox_inches="tight")
    print("Saved plot:", OUT_PNG)

if __name__ == "__main__":
    main()
