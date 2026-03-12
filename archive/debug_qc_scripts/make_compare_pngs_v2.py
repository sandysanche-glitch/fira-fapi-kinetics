import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Helpers

def pick_col(df, candidates, allow_prefix=None):
    """
    Pick the first existing column by explicit candidates or (optionally) a prefix rule.
    Returns (name or None).
    """
    for c in candidates:
        if c in df.columns:
            return c
    if allow_prefix:
        for c in df.columns:
            if c.startswith(allow_prefix):
                return c
    return None

def load_dndt(dndt_csv, time_col=None, rate_col=None):
    """
    Flexible loader for dn/dt CSV.
    - If time_col / rate_col provided, use them.
    - Otherwise:
        * time: one of [t_ms, time_ms, ms, t]
        * rate: one of [dn_dt, dn_dt_total, dn_dt_field, rate, density] OR any column starting with 'dn_dt_'
    """
    d = pd.read_csv(dndt_csv)

    if time_col is None:
        t_candidates = ["t_ms", "time_ms", "ms", "t"]
        time_col = pick_col(d, t_candidates)
    if rate_col is None:
        r_candidates = ["dn_dt", "dn_dt_total", "dn_dt_field", "rate", "density"]
        # also accept dataset-specific names like dn_dt_FAPI, dn_dt_FAPI-TEMPO, etc.
        rate_col = pick_col(d, r_candidates, allow_prefix="dn_dt_")

    if time_col is None or rate_col is None:
        raise ValueError(
            f"{dndt_csv} needs a time column (e.g. t_ms) and a rate column "
            f"(e.g. dn_dt or dn_dt_<label>). Found: {list(d.columns)}"
        )

    t_ms  = d[time_col].astype(float).to_numpy()
    dn_dt = d[rate_col].astype(float).to_numpy()
    return t_ms, dn_dt, d, (time_col, rate_col)

def load_x_overlay(x_csv, labelA, labelB, time_col=None, XA_col=None, XB_col=None):
    """
    Flexible loader for X overlay CSV.
    - If specific columns are provided, use them.
    - Otherwise tries:
        time: one of [t_ms, time_ms, ms, t]
        X cols:
          * X_{labelA}, X_{labelB}
          * fallbacks: (XA, XB), (X_A, X_B), (X_FAPI, X_FAPI_TEMPO)
          * last resort: pick two columns starting with 'X'
    """
    d = pd.read_csv(x_csv)

    if time_col is None:
        t_candidates = ["t_ms", "time_ms", "ms", "t"]
        time_col = pick_col(d, t_candidates)

    if XA_col is None or XB_col is None:
        prefA, prefB = f"X_{labelA}", f"X_{labelB}"
        if prefA in d.columns and prefB in d.columns:
            XA_col, XB_col = prefA, prefB
        else:
            for a,b in [("XA","XB"), ("X_A","X_B"), ("X_FAPI","X_FAPI_TEMPO")]:
                if a in d.columns and b in d.columns:
                    XA_col, XB_col = a, b
                    break
            if XA_col is None or XB_col is None:
                # last resort: first two X* columns
                X_like = [c for c in d.columns if c.lower().startswith("x")]
                if len(X_like) >= 2:
                    X_like = sorted(X_like)[:2]
                    XA_col, XB_col = X_like[0], X_like[1]

    if time_col is None or XA_col is None or XB_col is None:
        raise ValueError(
            f"Could not find time/X columns in {x_csv}. "
            f"Have: {list(d.columns)}"
        )

    t_ms = d[time_col].astype(float).to_numpy()
    XA   = d[XA_col].astype(float).to_numpy()
    XB   = d[XB_col].astype(float).to_numpy()
    return t_ms, XA, XB, d, (time_col, XA_col, XB_col)

# ---------- Main

def main():
    ap = argparse.ArgumentParser(description="Save PNG plots for dn/dt and X(t) overlay (FAPI vs FAPI-TEMPO).")
    ap.add_argument("--dir", required=True, help="Folder with exported CSVs (dn_dt_*.csv, X_pred_both.csv).")
    ap.add_argument("--labels", nargs=2, default=["FAPI","FAPI-TEMPO"], help="Dataset labels, e.g., FAPI FAPI-TEMPO")
    ap.add_argument("--px_per_um", type=float, default=2.20014, help="Pixels per micrometer.")
    ap.add_argument("--area_eff_px", type=float, default=2_500_000, help="Effective area (pixels) for SI conversion of rates.")
    ap.add_argument("--x_overlay_csv", default=None, help="Optional explicit path to X overlay CSV.")
    ap.add_argument("--outdir", default=None, help="Output directory for PNGs. Default: <dir>")

    # Optional explicit column names (if auto-detect fails)
    ap.add_argument("--time_col_A", default=None, help="Time column name in dn_dt_<labelA>.csv (e.g., t_ms)")
    ap.add_argument("--rate_col_A", default=None, help="Rate column name in dn_dt_<labelA>.csv (e.g., dn_dt_FAPI)")
    ap.add_argument("--time_col_B", default=None, help="Time column name in dn_dt_<labelB>.csv")
    ap.add_argument("--rate_col_B", default=None, help="Rate column name in dn_dt_<labelB>.csv")

    ap.add_argument("--x_time_col", default=None, help="Time column in X overlay CSV")
    ap.add_argument("--xA_col", default=None, help="X column for labelA in X overlay CSV")
    ap.add_argument("--xB_col", default=None, help="X column for labelB in X overlay CSV")

    args = ap.parse_args()
    base = Path(args.dir)
    outdir = Path(args.outdir) if args.outdir else base
    outdir.mkdir(parents=True, exist_ok=True)

    labelA, labelB = args.labels

    # --- Files
    dnA_csv = base / f"dn_dt_{labelA}.csv"
    dnB_csv = base / f"dn_dt_{labelB}.csv"
    x_csv   = Path(args.x_overlay_csv) if args.x_overlay_csv else (base / "X_pred_both.csv")

    # --- Load dn/dt (with flexible column discovery or explicit overrides)
    tA_ms, dnA_dt, dA, usedA = load_dndt(dnA_csv, time_col=args.time_col_A, rate_col=args.rate_col_A)
    tB_ms, dnB_dt, dB, usedB = load_dndt(dnB_csv, time_col=args.time_col_B, rate_col=args.rate_col_B)

    print(f"[{labelA}] Using dn/dt columns: time='{usedA[0]}', rate='{usedA[1]}'")
    print(f"[{labelB}] Using dn/dt columns: time='{usedB[0]}', rate='{usedB[1]}'")

    # SI conversion for dn/dt:
    # dn/dt [count/ms] -> [count/s] multiply by 1000
    # [count/s/m^2] divide by effective area (m^2)
    px_per_um = args.px_per_um
    um_per_px = 1.0 / px_per_um
    m_per_px  = um_per_px * 1e-6
    px_area_m2 = (m_per_px ** 2)
    area_eff_m2 = args.area_eff_px * px_area_m2 if args.area_eff_px is not None else np.nan

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
    t_ms, XA, XB, dX, cols = load_x_overlay(
        x_csv,
        labelA, labelB,
        time_col=args.x_time_col,
        XA_col=args.xA_col,
        XB_col=args.xB_col
    )
    print(f"[X overlay] Using columns: time='{cols[0]}', {labelA}='{cols[1]}', {labelB}='{cols[2]}'")

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
