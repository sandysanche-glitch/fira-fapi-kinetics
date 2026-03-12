import argparse, math
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv",  required=True, help="objects_FAPI.csv or objects_FAPI-TEMPO.csv")
    ap.add_argument("--out_csv", required=False, help="output CSV (default: *_aug.csv next to input)")
    # If your column names differ, you can override here:
    ap.add_argument("--area_col",        default="area_px",      help="area column (pixels^2)")
    ap.add_argument("--circ_col",        default="circularity",  help="circularity C=4πA/P^2 in (0,1]")
    ap.add_argument("--centroid_x_col",  default="centroid_x",   help="crystal centroid x (px)")
    ap.add_argument("--centroid_y_col",  default="centroid_y",   help="crystal centroid y (px)")
    ap.add_argument("--nuc_x_col",       default="nuc_x",        help="nucleus x (px)")
    ap.add_argument("--nuc_y_col",       default="nuc_y",        help="nucleus y (px)")
    args = ap.parse_args()

    src = Path(args.in_csv)
    out = Path(args.out_csv) if args.out_csv else src.with_name(src.stem + "_aug.csv")

    df = pd.read_csv(src)

    # --- Area ---
    if args.area_col not in df.columns:
        raise ValueError(f"Missing area column '{args.area_col}' in {src}")

    A = pd.to_numeric(df[args.area_col], errors="coerce").astype(float)

    # --- Perimeter: prefer exact via circularity if available ---
    perim = None
    if args.circ_col in df.columns:
        C = pd.to_numeric(df[args.circ_col], errors="coerce").astype(float)
        # P = sqrt(4 π A / C)  (guard C)
        C_clamped = np.clip(C, 1e-6, 1.0)
        perim = np.sqrt(4.0 * math.pi * np.maximum(A, 0.0) / C_clamped)
    else:
        # fallback: circular perimeter with same area (approximate)
        perim = 2.0 * np.sqrt(math.pi * np.maximum(A, 0.0))

    df["perim_px"] = perim

    # --- ray_px from nucleus -> centroid (if coords exist) ---
    rx = ry = None
    for col in (args.centroid_x_col, args.centroid_y_col, args.nuc_x_col, args.nuc_y_col):
        if col not in df.columns:
            print(f"[WARN] Missing column '{col}'. ray_px will be NaN unless all four exist.")
    if all(c in df.columns for c in (args.centroid_x_col, args.centroid_y_col, args.nuc_x_col, args.nuc_y_col)):
        cx = pd.to_numeric(df[args.centroid_x_col], errors="coerce").astype(float)
        cy = pd.to_numeric(df[args.centroid_y_col], errors="coerce").astype(float)
        nx = pd.to_numeric(df[args.nuc_x_col], errors="coerce").astype(float)
        ny = pd.to_numeric(df[args.nuc_y_col], errors="coerce").astype(float)
        df["ray_px"] = np.sqrt((cx - nx)**2 + (cy - ny)**2)
    else:
        df["ray_px"] = np.nan

    # Basic sanity clamps
    df.loc[~np.isfinite(df["perim_px"]), "perim_px"] = np.nan
    df.loc[~np.isfinite(df["ray_px"]),   "ray_px"]   = np.nan

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[OK] Wrote augmented file: {out}")

if __name__ == "__main__":
    main()
