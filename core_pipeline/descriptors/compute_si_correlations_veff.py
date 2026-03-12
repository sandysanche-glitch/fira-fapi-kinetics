#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def robust_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def add_defect_fraction(
    df: pd.DataFrame,
    defect_area_col: str,
    grain_area_col: str,
) -> pd.DataFrame:
    df = df.copy()
    if defect_area_col in df.columns and grain_area_col in df.columns:
        num = robust_num(df[defect_area_col])
        den = robust_num(df[grain_area_col])
        with np.errstate(divide="ignore", invalid="ignore"):
            df["defect_fraction"] = np.where(den > 0, num / den, np.nan)
    return df


def p_to_str(p: float) -> str:
    if not np.isfinite(p):
        return "nan"
    if p < 1e-300:
        return "<1e-300"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def corr_row(
    df: pd.DataFrame,
    sample_name: str,
    descriptor_col: str,
    descriptor_label: str,
    veff_col: str,
) -> dict:
    x = robust_num(df[descriptor_col])
    y = robust_num(df[veff_col])

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m].to_numpy(float)
    y = y[m].to_numpy(float)

    n = len(x)

    if n < 3:
        return {
            "sample": sample_name,
            "descriptor": descriptor_label,
            "descriptor_col": descriptor_col,
            "n": n,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "spearman_p_str": "nan",
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "pearson_p_str": "nan",
        }

    sp = spearmanr(x, y, nan_policy="omit")
    pr = pearsonr(x, y)

    return {
        "sample": sample_name,
        "descriptor": descriptor_label,
        "descriptor_col": descriptor_col,
        "n": n,
        "spearman_rho": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
        "spearman_p_str": p_to_str(float(sp.pvalue)),
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "pearson_p_str": p_to_str(float(pr.pvalue)),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Compute SI Spearman/Pearson correlations between morphology descriptors and v_eff."
    )
    ap.add_argument("--fapi", required=True, help="Path to FAPI with_veff CSV")
    ap.add_argument("--tempo", required=True, help="Path to FAPI-TEMPO with_veff CSV")
    ap.add_argument("--out", required=True, help="Output SI table CSV path")
    ap.add_argument("--veff-col", default="v_eff_um_per_ms", help="Effective growth-rate column")
    ap.add_argument("--circ-col", default="circularity_distortion", help="Circularity distortion column")
    ap.add_argument("--entropy-col", default="entropy_hm_(bits)", help="Entropy column")
    ap.add_argument("--defect-area-col", default="defects_area_(µm²)", help="Defect area column")
    ap.add_argument("--grain-area-col", default="area_(µm²)", help="Grain area column")
    args = ap.parse_args()

    fapi = pd.read_csv(args.fapi)
    tempo = pd.read_csv(args.tempo)

    fapi = add_defect_fraction(fapi, args.defect_area_col, args.grain_area_col)
    tempo = add_defect_fraction(tempo, args.defect_area_col, args.grain_area_col)

    # check required v_eff column
    for name, df in [("FAPI", fapi), ("FAPI-TEMPO", tempo)]:
        if args.veff_col not in df.columns:
            raise KeyError(f"Missing '{args.veff_col}' in {name} table")

    descriptors = [
        (args.circ_col, "Circularity distortion"),
        ("defect_fraction", "Defect fraction"),
        (args.entropy_col, "Entropy"),
    ]

    rows = []
    for col, label in descriptors:
        if col in fapi.columns:
            rows.append(corr_row(fapi, "FAPI", col, label, args.veff_col))
        else:
            rows.append(
                {
                    "sample": "FAPI",
                    "descriptor": label,
                    "descriptor_col": col,
                    "n": 0,
                    "spearman_rho": np.nan,
                    "spearman_p": np.nan,
                    "spearman_p_str": "missing",
                    "pearson_r": np.nan,
                    "pearson_p": np.nan,
                    "pearson_p_str": "missing",
                }
            )

        if col in tempo.columns:
            rows.append(corr_row(tempo, "FAPI-TEMPO", col, label, args.veff_col))
        else:
            rows.append(
                {
                    "sample": "FAPI-TEMPO",
                    "descriptor": label,
                    "descriptor_col": col,
                    "n": 0,
                    "spearman_rho": np.nan,
                    "spearman_p": np.nan,
                    "spearman_p_str": "missing",
                    "pearson_r": np.nan,
                    "pearson_p": np.nan,
                    "pearson_p_str": "missing",
                }
            )

    out_df = pd.DataFrame(rows)

    # nice SI ordering
    descriptor_order = {
        "Circularity distortion": 0,
        "Defect fraction": 1,
        "Entropy": 2,
    }
    sample_order = {
        "FAPI": 0,
        "FAPI-TEMPO": 1,
    }

    out_df["_dord"] = out_df["descriptor"].map(descriptor_order)
    out_df["_sord"] = out_df["sample"].map(sample_order)
    out_df = out_df.sort_values(["_dord", "_sord"]).drop(columns=["_dord", "_sord"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"[OK] Saved SI correlation table: {out_path.resolve()}")
    print("\nSI correlation table:")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()