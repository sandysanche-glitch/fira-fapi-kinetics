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
