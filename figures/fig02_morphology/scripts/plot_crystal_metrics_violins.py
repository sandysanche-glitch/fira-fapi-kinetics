#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# seaborn is optional but strongly recommended for the "entropy-like" look
try:
    import seaborn as sns
except ImportError as e:
    raise SystemExit(
        "This script needs seaborn. Install it with:\n"
        "  pip install seaborn\n"
        "or\n"
        "  conda install seaborn"
    )

def infer_dataset_from_filename(name: str) -> str:
    """
    Infer dataset label from file_name. Expected patterns:
      - FAPI_TEMPO_...
      - FAPI-TEMPO_...
      - FAPI_TEMPO ...
      - FAPI_...
    """
    s = str(name).strip()
    up = s.upper()
    if "TEMPO" in up:
        return "FAPI-TEMPO"
    # fallback: assume FAPI
    return "FAPI"


def load_metrics(in_dir: Path, pattern: str) -> pd.DataFrame:
    csvs = sorted(in_dir.glob(pattern))
    if not csvs:
        raise FileNotFoundError(f"No CSVs matching '{pattern}' found in: {in_dir}")

    dfs = []
    for p in csvs:
        df = pd.read_csv(p)
        df["__source_csv__"] = p.name
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)

    if "file_name" not in out.columns:
        raise ValueError(f"Expected a 'file_name' column. Found: {out.columns.tolist()}")

    out["dataset"] = out["file_name"].apply(infer_dataset_from_filename)

    # compute defect area ratio (robustly)
    area_col = "area_(µm²)" if "area_(µm²)" in out.columns else None
    def_area_col = "defects_area_(µm²)" if "defects_area_(µm²)" in out.columns else None

    if area_col and def_area_col:
        area = pd.to_numeric(out[area_col], errors="coerce")
        darea = pd.to_numeric(out[def_area_col], errors="coerce")
        out["defect_area_ratio"] = darea / area.replace(0, np.nan)
    else:
        out["defect_area_ratio"] = np.nan

    return out


def pretty_ylabel(metric: str) -> str:
    if metric == "area_(µm²)":
        return "Area (µm²)"
    if metric == "perimeter_(µm)":
        return "Perimeter (µm)"
    if metric == "circularity_distortion":
        return "Circularity distortion"
    if metric == "defect_area_ratio":
        return "Defect area ratio (A_defects / A_grain)"
    return metric


def plot_violin_like_entropy(df: pd.DataFrame, metric: str, out_path: Path, title: str):
    # Clean + enforce order
    d = df[["dataset", metric]].copy()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna(subset=[metric])

    order = ["FAPI", "FAPI-TEMPO"]
    d["dataset"] = pd.Categorical(d["dataset"], categories=order, ordered=True)
    d = d.dropna(subset=["dataset"])

    if d.empty:
        raise ValueError(f"No valid data to plot for metric '{metric}' after cleaning.")

    sns.set_theme(style="whitegrid", context="talk")

    fig, ax = plt.subplots(figsize=(12, 7))

    # Violin (no inner stats)
    sns.violinplot(
        data=d,
        x="dataset",
        y=metric,
        order=order,
        inner=None,
        cut=0,
        linewidth=1.5,
        ax=ax
    )

    # Jittered points
    sns.stripplot(
        data=d,
        x="dataset",
        y=metric,
        order=order,
        jitter=0.25,
        size=5,
        alpha=0.65,
        ax=ax
    )

    # Overlay boxplot to mimic thick median line + whiskers
    sns.boxplot(
        data=d,
        x="dataset",
        y=metric,
        order=order,
        width=0.25,
        showfliers=False,
        boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 2},
        whiskerprops={"color": "black", "linewidth": 2},
        capprops={"color": "black", "linewidth": 2},
        medianprops={"color": "black", "linewidth": 6},
        ax=ax
    )

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(pretty_ylabel(metric))

    # light y-grid like your example
    ax.grid(True, axis="y", alpha=0.35)
    ax.grid(False, axis="x")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path.with_suffix(".png"), dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_dir",
        required=True,
        help=r'Folder containing crystal_metrics*.csv (e.g. "D:\SWITCHdrive\Institution\Sts_grain morphology_ML")'
    )
    ap.add_argument(
        "--pattern",
        default="crystal_metrics*.csv",
        help="Glob pattern for CSVs inside --in_dir (default: crystal_metrics*.csv)"
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help=r'Output folder for plots (e.g. "D:\...\plots")'
    )
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    df = load_metrics(in_dir, args.pattern)

    # Metrics requested
    metrics = [
        "area_(µm²)",
        "perimeter_(µm)",
        "circularity_distortion",
        "defect_area_ratio",
    ]

    # Check presence; allow missing but warn
    missing = [m for m in metrics if m not in df.columns]
    if missing:
        print("[WARN] Missing columns:", missing)
        print("[INFO] Available columns:", df.columns.tolist())

    # Plot each metric separately
    for m in metrics:
        if m not in df.columns:
            continue
        title = f"{pretty_ylabel(m)} (crystal_metrics, per-instance)"
        out_path = out_dir / f"violin_{re.sub(r'[^A-Za-z0-9_]+','_',m)}"
        plot_violin_like_entropy(df, m, out_path, title)
        print("[OK] Saved:", out_path.with_suffix(".png"))

    # quick summary counts
    print("\n[INFO] Dataset counts:")
    print(df["dataset"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
