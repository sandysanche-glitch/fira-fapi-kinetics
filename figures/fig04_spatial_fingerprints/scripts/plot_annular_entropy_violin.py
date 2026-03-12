import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# READ ONLY THIS FILE (absolute path)
INP = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\radial_entropy_results.csv"
OUT = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\annular_entropy_violin.png"

REGIONS = ["inner", "middle", "outer"]
GROUP_ORDER = ["FAPI", "FAPI-TEMPO"]

# ---- Load (explicit) ----
df = pd.read_csv(INP)

# ---- Hard sanity checks to avoid “wrong CSV” mistakes ----
expected_cols = {
    "group",
    "entropy_inner_norm01", "entropy_middle_norm01", "entropy_outer_norm01"
}
missing = sorted(list(expected_cols - set(df.columns)))
if missing:
    raise ValueError(
        "This does NOT look like the radial entropy CSV.\n"
        f"Loaded: {INP}\n"
        f"Missing required columns: {missing}\n"
        f"Columns found: {list(df.columns)}"
    )

# Show what was loaded (so you can confirm immediately)
print("Loaded:", INP)
print("Shape:", df.shape)
print("Groups:", df["group"].unique()[:10])
print(df[["group", "sample_id", "pair_id"]].head())

# ---- Long format ----
value_cols = [f"entropy_{r}_norm01" for r in REGIONS]
dfl = df.melt(
    id_vars=[c for c in ["group", "pair_id", "sample_id"] if c in df.columns],
    value_vars=value_cols,
    var_name="region",
    value_name="entropy"
)
dfl["region"] = (
    dfl["region"]
    .str.replace("entropy_", "", regex=False)
    .str.replace("_norm01", "", regex=False)
)

# ---- Plot ----
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
fig.suptitle("Annular entropy (0..1 normalized) — violin + points (mean/median/max)", y=1.02)

for ax, region in zip(axes, REGIONS):
    sub = dfl[dfl["region"] == region].copy()

    sns.violinplot(
        data=sub, x="group", y="entropy",
        order=GROUP_ORDER, inner=None, cut=0,
        linewidth=1, ax=ax
    )

    sns.stripplot(
        data=sub, x="group", y="entropy",
        order=GROUP_ORDER, jitter=0.28,
        size=3, alpha=0.55, ax=ax
    )

    stats = sub.groupby("group")["entropy"].agg(["mean", "median", "max"]).reindex(GROUP_ORDER)
    for i, g in enumerate(GROUP_ORDER):
        mu = float(stats.loc[g, "mean"])
        med = float(stats.loc[g, "median"])
        mx = float(stats.loc[g, "max"])

        ax.hlines(mu,  i - 0.22, i + 0.22, linewidth=3)    # mean (thick)
        ax.hlines(med, i - 0.20, i + 0.20, linewidth=2)    # median (mid)
        ax.hlines(mx,  i - 0.18, i + 0.18, linewidth=1.2)  # max (thin)

    ax.set_title(region.capitalize())
    ax.set_xlabel("")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.2)

axes[0].set_ylabel("Entropy (normalized 0..1)")
for ax in axes[1:]:
    ax.set_ylabel("")

plt.tight_layout()
plt.savefig(OUT, dpi=250, bbox_inches="tight")
print("Saved:", OUT)
