#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_dn_dt_small.py

Re-plot nucleation density rate dn/dt for FAPI and FAPI–TEMPO
on a canvas that is ~20% smaller in both width and height.

Reads:
  D:\SWITCHdrive\Institution\Sts_grain morphology_ML\combined_out_penalties\combined_dn_dt.csv

Writes:
  D:\SWITCHdrive\Institution\Sts_grain morphology_ML\combined_out_penalties\combined_dn_dt_small.png
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    base_dir = Path(
        r"D:\SWITCHdrive\Institution\Sts_grain morphology_ML\combined_out_penalties"
    )
    csv_path = base_dir / "combined_dn_dt.csv"
    out_path = base_dir / "combined_dn_dt_small.png"

    df = pd.read_csv(csv_path)

    # --- detect columns ---
    cols_lower = {c.lower(): c for c in df.columns}

    # time column: prefer 't_ms', then 't', else first column
    if "t_ms" in cols_lower:
        t_col = cols_lower["t_ms"]
    elif "t" in cols_lower:
        t_col = cols_lower["t"]
    else:
        t_col = df.columns[0]

    # FAPI dn/dt column: name containing both 'fapi' and 'dn'
    fapi_candidates = [
        c for c in df.columns
        if ("fapi" in c.lower()) and ("dn" in c.lower() or "rate" in c.lower())
    ]
    if not fapi_candidates:
        raise RuntimeError("Could not find FAPI dn/dt column in combined_dn_dt.csv")
    fapi_col = fapi_candidates[0]

    # FAPI–TEMPO dn/dt column: contains 'tempo' and dn/rate
    tempo_candidates = [
        c for c in df.columns
        if ("tempo" in c.lower()) and ("dn" in c.lower() or "rate" in c.lower())
    ]
    if not tempo_candidates:
        raise RuntimeError("Could not find FAPI–TEMPO dn/dt column in combined_dn_dt.csv")
    tempo_col = tempo_candidates[0]

    t_ms = df[t_col].values
    dn_fapi = df[fapi_col].values
    dn_tempo = df[tempo_col].values

    # --- plot with smaller canvas (~20% smaller than ~7x4.5) ---
    fig, ax = plt.subplots(figsize=(5.0, 4.0))

    ax.plot(t_ms, dn_fapi, label="FAPI", lw=2)
    ax.plot(t_ms, dn_tempo, label="FAPI-TEMPO", lw=2)

    ax.set_xlabel("t (ms)")
    ax.set_ylabel("dn/dt  [events / (ms·mm²)]")
    ax.set_title("Nucleation density rate dn/dt — both datasets")

    ax.legend()
    ax.grid(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"[OK] saved resized dn/dt plot: {out_path}")


if __name__ == "__main__":
    main()
