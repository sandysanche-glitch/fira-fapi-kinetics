import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Helpers

def load_dndt(dndt_csv):
    """
    Flexible loader for dn/dt CSV.
    Accepts time under: t_ms, time_ms, ms, t
    Accepts rate under: dn_dt, dn_dt_total, dn_dt_field, rate, density
    """
    d = pd.read_csv(dndt_csv)
    t_candidates  = ["t_ms", "time_ms", "ms", "t"]
    r_candidates  = ["dn_dt", "dn_dt_total", "dn_dt_field", "rate", "density"]
    tcol = next((c for c in t_candidates if c in d.columns), None)
    rcol = next((c for c in r_candidates if c in d.columns), None)
    if tcol is None or rcol is None:
        raise ValueError(
            f"{dndt_csv} needs a time col {t_candidates} and a rate col {r_candidates}. "
            f"Found: {list(d.columns)}"
        )
    t_ms  = d[tcol].astype(float).to_numpy()
    dn_dt = d[rcol].astype(float).to_numpy()
    return t_ms, dn_dt, d

def load_x_overlay(x_csv, labelA, labelB):
    """
    Flexible loader for X overlay CSV.
    Tries common patterns:
      time columns: t_ms, time_ms, ms, t
      dataset columns: X_FAPI, X_FAPI_TEMPO, XA, XB, X_A, X_B
    """
    d = pd.read_csv(x_csv)
    t_candidates = ["t_ms", "time_ms", "ms", "t"]
    tcol = next((c for c in t_candidates if c in d.columns), None)
    if tcol is None:
        raise ValueError(f"{x_csv} needs a time column {t_candidates}. Found: {list(d.columns)}")

    # try exact labels first
    x_cols_pref = [f"X_{labelA}", f"X_{labelB}"]
    # fallbacks
    fallbacks = [
        ("XA", "XB"), ("X_A", "X_B"),
        ("X_FAPI", "X_FAPI_TEMPO"),  # common pair
    ]

    xAcol, xBcol = None, None
    if all(c in d.columns for c in x_cols_pref):
        xAcol, xBcol = x_cols_pref
    else:
        for a,b in fallbacks:
            if a in d.columns and b in d.columns:
                xAcol, xBcol = a, b
                break

    # last resort: pick two X* columns deterministically
    if xAcol is None or xBcol is None:
        X_like = [c for c in d.columns if c.lower().startswith("x")]
        if len(X_like) >= 2:
            X_like = sorted(X_like)[:2]
            xAcol, xBcol = X_like[0], X_like[1]

    if xAcol is None or xBcol is None:
        raise ValueError(
            f"Could not find two X(·) columns for overlay in {x_csv}. "
            f"Have columns: {list(d.columns)}"
        )

    t_ms = d[tcol].astype(float).to_numpy()
    XA   = d[xAcol].astype(float).to_numpy()
    XB   = d[xBcol].astype(float).to_numpy()
    return t_ms, XA, XB, d, (xAcol, xBcol, tcol)

# ---------- Main

def main():
    ap = argparse.ArgumentParser(description="Save PNG plots for dn/dt and X(t) overlay (FAPI vs FAPI-TEMPO).")
    ap.add_argument("--dir", required=True, help="Folder containing exported CSVs (objects_*.csv, dn_dt_*.csv, X_pred_both.csv).")
    ap.add_argument("--labels", nargs=2, default=["FAPI","FAPI-TEMPO"], help="Dataset labels, e.g., FAPI FAPI-TEMPO")
    ap.add_argument("--px_per_um", type=float, default=2.20014, help="Pixels per micrometer.")
    ap.add_argument("--area_eff_px", type=float, default=2_500_000, help="Effective area (pixels) for SI conversion of rates.")
    ap.add_argument("--x_overlay_csv", default=None, help="Optional explicit path to X overlay CSV.")
    ap.add_argument("--outdir", default=None, help="Output directory for PNGs. Default: <dir>")

    args = ap.parse_args()
    base = Path(args.dir)
    outdir = Path(args.outdir) if args.outdir else base
    outdir.mkdir(parents=True, exist_ok=True)

    labelA, labelB = args.labels
    # --- Files
    dnA_csv = base / f"dn_dt_{labelA}.csv"
    dnB_csv = base / f"dn_dt_{labelB}.csv"
    x_csv   = Path(args.x_overlay_csv) if args.x_overlay_csv else (base / "X_pred_both.csv")

    # --- Load dn/dt
    tA_ms, dnA_dt, dA = load_dndt(dnA_csv)
    tB_ms, dnB_dt, dB = load_dndt(dnB_csv)

    # SI conversion for dn/dt:
    # dn/dt [count/ms] -> [count/s] multiply by 1000
    # If you want areal rate [count/(s·m^2)], divide by area_eff (px) and by pixel area (m^2/px).
    # Here we just plot raw [count/ms] and also a SI version [count/s/m^2] for completeness:
    px_per_um = args.px_per_um
    um_per_px = 1.0 / px_per_um
    m_per_px  = um_per_px * 1e-6
    px_area_m2 = (m_per_px ** 2)
    area_eff_m2 = args.area_eff_px * px_area_m2

    dnA_dt_si = (dnA_dt * 1000.0) / max(area_eff_m2, 1e-30)
    dnB_dt_si = (dnB_dt * 1000.0) / max(area_eff_m2, 1e-30)

    # --- Plot dn/dt (raw units: count/ms)
    fig1, ax1 = plt.subplots(figsize=(7,4))
    ax1.plot(tA_ms, dnA_dt, label=f"{labelA} (count/ms)")
    ax1.plot(tB_ms, dnB_dt, label=f"{labelB} (count/ms)")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Nucleation rate (count / ms)")
    ax1.set_title("dn/dt comparison (raw)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(outdir / "dn_dt_compare_raw.png", dpi=300, bbox_inches="tight")

    # --- Plot dn/dt (SI: count/s/m^2)
    fig2, ax2 = plt.subplots(figsize=(7,4))
    ax2.plot(tA_ms, dnA_dt_si, label=f"{labelA} (count/s/m²)")
    ax2.plot(tB_ms, dnB_dt_si, label=f"{labelB} (count/s/m²)")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Nucleation rate (count / s / m²)")
    ax2.set_title("dn/dt comparison (SI)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(outdir / "dn_dt_compare_SI.png", dpi=300, bbox_inches="tight")

    # --- X overlay
    t_ms, XA, XB, dX, cols = load_x_overlay(x_csv, labelA, labelB)

    # Raw X (dimensionless fraction)
    fig3, ax3 = plt.subplots(figsize=(7,4))
    ax3.plot(t_ms, XA, label=labelA)
    ax3.plot(t_ms, XB, label=labelB)
    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel("Transformed fraction X(t)")
    ax3.set_title("X(t) comparison")
    ax3.set_ylim(0, 1.05)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(outdir / "X_overlay_compare.png", dpi=300, bbox_inches="tight")

    print(f"PNG files written to: {outdir}")

if __name__ == "__main__":
    main()
