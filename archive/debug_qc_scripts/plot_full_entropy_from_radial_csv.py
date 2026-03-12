import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import PolyCollection

INP = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\radial_entropy_results.csv"
OUT = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\annular_entropy_violin.png"

REGIONS = ["inner", "middle", "outer"]
GROUP_ORDER = ["FAPI", "FAPI-TEMPO"]
PALETTE = {"FAPI": "#1f77b4", "FAPI-TEMPO": "#ff7f0e"}

VIOLIN_ALPHA = 0.25
POINT_ALPHA  = 0.70

df = pd.read_csv(INP)

expected_cols = {"group", "entropy_inner_norm01", "entropy_middle_norm01", "entropy_outer_norm01"}
missing = sorted(list(expected_cols - set(df.columns)))
if missing:
    raise ValueError(
        "Wrong file loaded (missing expected entropy columns).\n"
        f"Loaded: {INP}\nMissing: {missing}\nFound columns: {list(df.columns)}"
    )

value_cols = [f"entropy_{r}_norm01" for r in REGIONS]
dfl = df.melt(
    id_vars=[c for c in ["group", "pair_id", "sample_id"] if c in df.columns],
    value_vars=value_cols,
    var_name="region",
    value_name="entropy"
)
dfl["region"] = (
    dfl["region"].str.replace("entropy_", "", regex=False)
                 .str.replace("_norm01", "", regex=False)
)

sns.set_style("whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
fig.suptitle("Annular entropy (0..1 normalized) — violin + points (mean/min/max)", y=1.02)

for ax, region in zip(axes, REGIONS):
    sub = dfl[dfl["region"] == region].copy()

    # Violin
    sns.violinplot(
        data=sub, x="group", y="entropy",
        order=GROUP_ORDER,
        hue="group", hue_order=GROUP_ORDER,
        palette=PALETTE, inner=None, cut=0, linewidth=1,
        legend=False, ax=ax
    )

    # Make violin fills transparent
    for coll in ax.collections:
        if isinstance(coll, PolyCollection):
            coll.set_alpha(VIOLIN_ALPHA)

    # Points
    sns.stripplot(
        data=sub, x="group", y="entropy",
        order=GROUP_ORDER,
        hue="group", hue_order=GROUP_ORDER,
        palette=PALETTE, jitter=0.28, size=3, alpha=POINT_ALPHA,
        dodge=False, legend=False, ax=ax
    )

    # BLACK mean / min / max bars
    stats = sub.groupby("group")["entropy"].agg(["mean", "min", "max"]).reindex(GROUP_ORDER)
    for i, g in enumerate(GROUP_ORDER):
        mu = float(stats.loc[g, "mean"])
        mn = float(stats.loc[g, "min"])
        mx = float(stats.loc[g, "max"])

        ax.hlines(mu, i - 0.22, i + 0.22, linewidth=3.0, color="k", zorder=7)  # mean
        ax.hlines(mn, i - 0.18, i + 0.18, linewidth=1.3, color="k", zorder=7)  # min
        ax.hlines(mx, i - 0.18, i + 0.18, linewidth=1.3, color="k", zorder=7)  # max

    ax.set_title(region.capitalize())
    ax.set_xlabel("")
    ax.set_ylim(0, 1.02)

axes[0].set_ylabel("Entropy (normalized 0..1)")
for ax in axes[1:]:
    ax.set_ylabel("")

plt.tight_layout()
plt.savefig(OUT, dpi=250, bbox_inches="tight")
print("Saved:", OUT)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from matplotlib.collections import PolyCollection

# ---- USER PATHS ----
INP = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\radial_entropy_results.csv"
OUT_PNG = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\full_entropy_comparison.png"
OUT_CSV = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\radial_entropy_results_with_full_entropy.csv"

# ---- STYLE ----
GROUP_ORDER = ["FAPI", "FAPI-TEMPO"]
PALETTE = {"FAPI": "#1f77b4", "FAPI-TEMPO": "#ff7f0e"}  # same as your previous plots
VIOLIN_ALPHA = 0.25
POINT_ALPHA = 0.70

# ---- ENTROPY SETTINGS ----
BINS_NORM = 256          # bins for 0..1 values
BINS_RAW = 256           # bins for reconstructed raw
MASK_EPS = 1e-12         # pixels > MASK_EPS are considered "mask"

def shannon_entropy_bits(values: np.ndarray, bins: int, vmin=None, vmax=None) -> float:
    """Shannon entropy in bits for a 1D array of values (histogram-based)."""
    if values.size == 0:
        return np.nan
    if vmin is None:
        vmin = float(np.nanmin(values))
    if vmax is None:
        vmax = float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.nan

    hist, _ = np.histogram(values, bins=bins, range=(vmin, vmax))
    p = hist.astype(np.float64)
    p = p[p > 0]
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())

def load_norm01_heatmap_as_float01(path_png: str) -> np.ndarray:
    """Loads PNG and returns a float array in [0,1]. Assumes grayscale or uses luminance if RGB."""
    im = Image.open(path_png)

    # If RGB/RGBA, convert to grayscale (luminance). This is safest if your PNG is already grayscale.
    if im.mode in ("RGB", "RGBA"):
        im = im.convert("L")
    else:
        im = im.convert("L")

    arr = np.asarray(im).astype(np.float32)

    # Typical PNGs are 0..255
    if arr.max() > 1.0:
        arr = arr / 255.0

    arr = np.clip(arr, 0.0, 1.0)
    return arr

def add_transparency_to_violins(ax, alpha: float):
    for coll in ax.collections:
        if isinstance(coll, PolyCollection):
            coll.set_alpha(alpha)

def violin_points_blackbars(ax, sub, ycol: str, title: str):
    sns.violinplot(
        data=sub, x="group", y=ycol,
        order=GROUP_ORDER,
        hue="group", hue_order=GROUP_ORDER,
        palette=PALETTE,
        inner=None, cut=0, linewidth=1,
        legend=False, ax=ax
    )
    add_transparency_to_violins(ax, VIOLIN_ALPHA)

    sns.stripplot(
        data=sub, x="group", y=ycol,
        order=GROUP_ORDER,
        hue="group", hue_order=GROUP_ORDER,
        palette=PALETTE,
        jitter=0.28, size=3, alpha=POINT_ALPHA,
        dodge=False, legend=False, ax=ax
    )

    stats = sub.groupby("group")[ycol].agg(["mean", "min", "max"]).reindex(GROUP_ORDER)
    for i, g in enumerate(GROUP_ORDER):
        mu = float(stats.loc[g, "mean"])
        mn = float(stats.loc[g, "min"])
        mx = float(stats.loc[g, "max"])
        ax.hlines(mu, i - 0.22, i + 0.22, linewidth=3.0, color="k", zorder=7)  # mean
        ax.hlines(mn, i - 0.18, i + 0.18, linewidth=1.3, color="k", zorder=7)  # min
        ax.hlines(mx, i - 0.18, i + 0.18, linewidth=1.3, color="k", zorder=7)  # max

    ax.set_title(title)
    ax.set_xlabel("")
    ax.grid(True, alpha=0.2)

def main():
    df = pd.read_csv(INP)

    # Hard checks so you don't accidentally read another CSV
    required = {"group", "heatmap_norm01_path", "heatmap_min_before", "heatmap_max_before"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}\nFound: {list(df.columns)}")

    base_dir = os.path.dirname(INP)

    full_norm_all = []
    full_norm_mask = []
    full_raw_all = []
    full_raw_mask = []

    n_ok = 0
    n_missing_png = 0

    for _, row in df.iterrows():
        rel = str(row["heatmap_norm01_path"])
        png_path = rel if os.path.isabs(rel) else os.path.join(base_dir, rel)

        if not os.path.exists(png_path):
            n_missing_png += 1
            full_norm_all.append(np.nan)
            full_norm_mask.append(np.nan)
            full_raw_all.append(np.nan)
            full_raw_mask.append(np.nan)
            continue

        norm = load_norm01_heatmap_as_float01(png_path)
        vals_all = norm.ravel()
        vals_mask = norm[norm > MASK_EPS].ravel()

        # normalized entropy (0..1)
        e_norm_all = shannon_entropy_bits(vals_all, bins=BINS_NORM, vmin=0.0, vmax=1.0)
        e_norm_mask = shannon_entropy_bits(vals_mask, bins=BINS_NORM, vmin=0.0, vmax=1.0)

        # reconstruct raw using stored min/max (per-image)
        mn0 = float(row["heatmap_min_before"])
        mx0 = float(row["heatmap_max_before"])
        raw = norm * (mx0 - mn0) + mn0

        raw_all = raw.ravel()
        raw_mask = raw[norm > MASK_EPS].ravel()

        e_raw_all = shannon_entropy_bits(raw_all, bins=BINS_RAW, vmin=mn0, vmax=mx0)
        e_raw_mask = shannon_entropy_bits(raw_mask, bins=BINS_RAW, vmin=mn0, vmax=mx0)

        full_norm_all.append(e_norm_all)
        full_norm_mask.append(e_norm_mask)
        full_raw_all.append(e_raw_all)
        full_raw_mask.append(e_raw_mask)
        n_ok += 1

    df["full_entropy_norm_allpix"] = full_norm_all
    df["full_entropy_norm_maskpix"] = full_norm_mask
    df["full_entropy_raw_allpix"] = full_raw_all
    df["full_entropy_raw_maskpix"] = full_raw_mask

    df.to_csv(OUT_CSV, index=False)
    print("Loaded:", INP)
    print("Rows:", len(df), "OK PNGs:", n_ok, "Missing PNGs:", n_missing_png)
    print("Saved computed table:", OUT_CSV)

    # Plot: compare mask-only (most comparable to “entropy over mask”)
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    fig.suptitle("Full Shannon entropy from heatmaps (mask-only pixels)", y=1.03)

    sub = df[df["group"].isin(GROUP_ORDER)].copy()

    violin_points_blackbars(
        axes[0], sub, "full_entropy_raw_maskpix",
        "Full entropy (raw reconstructed) — mask pixels"
    )
    axes[0].set_ylabel("Entropy (bits)")

    violin_points_blackbars(
        axes[1], sub, "full_entropy_norm_maskpix",
        "Full entropy (0..1 from PNG) — mask pixels"
    )
    axes[1].set_ylabel("Entropy (bits)")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=250, bbox_inches="tight")
    print("Saved plot:", OUT_PNG)

if __name__ == "__main__":
    main()
