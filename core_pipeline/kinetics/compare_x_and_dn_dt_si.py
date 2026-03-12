import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Folder where per-dataset CSVs were exported")
    ap.add_argument("--labelA", default="FAPI")
    ap.add_argument("--labelB", default="FAPI-TEMPO")
    ap.add_argument("--out", required=True, help="Output folder for combined plots")
    args = ap.parse_args()

    indir = Path(args.dir)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    # Load X overlays
    XA = pd.read_csv(indir / f"{args.labelA}_X_pred_Avrami.csv")
    XB = pd.read_csv(indir / f"{args.labelB}_X_pred_Avrami.csv")

    # Load dn/dt
    dA = pd.read_csv(indir / f"{args.labelA}_nucleation_dn_dt.csv")
    dB = pd.read_csv(indir / f"{args.labelB}_nucleation_dn_dt.csv")

    import matplotlib.pyplot as plt

    # X overlay — milliseconds
    plt.figure()
    plt.plot(XA["t_ms"], XA["X_pred"], label=f"{args.labelA} X_pred")
    plt.plot(XA["t_ms"], XA["X_Avrami"], linestyle="--", label=f"{args.labelA} Avrami")
    plt.plot(XB["t_ms"], XB["X_pred"], label=f"{args.labelB} X_pred")
    plt.plot(XB["t_ms"], XB["X_Avrami"], linestyle="--", label=f"{args.labelB} Avrami")
    plt.xlabel("t (ms)"); plt.ylabel("X(t) (fraction)")
    plt.title("X(t) vs Avrami — both datasets (ms)")
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "X_overlay_both_ms.png", dpi=200)
    plt.close()

    # X overlay — seconds
    # If t_s missing (older files), compute quickly
    for X in (XA, XB):
        if "t_s" not in X.columns:
            X["t_s"] = X["t_ms"] / 1000.0

    plt.figure()
    plt.plot(XA["t_s"], XA["X_pred"], label=f"{args.labelA} X_pred")
    plt.plot(XA["t_s"], XA["X_Avrami"], linestyle="--", label=f"{args.labelA} Avrami")
    plt.plot(XB["t_s"], XB["X_pred"], label=f"{args.labelB} X_pred")
    plt.plot(XB["t_s"], XB["X_Avrami"], linestyle="--", label=f"{args.labelB} Avrami")
    plt.xlabel("t (s)"); plt.ylabel("X(t) (fraction)")
    plt.title("X(t) vs Avrami — both datasets (seconds)")
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "X_overlay_both_s.png", dpi=200)
    plt.close()

    # dn/dt (per ms·mm²), milliseconds
    plt.figure()
    plt.plot(dA["t_ms"], dA["dn_dt_per_ms_mm2"], label=args.labelA)
    plt.plot(dB["t_ms"], dB["dn_dt_per_ms_mm2"], label=args.labelB)
    plt.xlabel("t (ms)"); plt.ylabel("dn/dt  [events / (ms·mm²)]")
    plt.title("Nucleation density rate dn/dt — both datasets")
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "dn_dt_both_ms.png", dpi=200)
    plt.close()

    print(f"[OK] Wrote combined plots to: {outdir}")

if __name__ == "__main__":
    main()
