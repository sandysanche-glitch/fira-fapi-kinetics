import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ---------- Pixel→metric calibration ----------
PX_PER_UM = 2.20014
S_UM_PER_PX = 1.0 / PX_PER_UM           # µm/px
S_M_PER_PX  = S_UM_PER_PX * 1e-6        # m/px
S_M2_PER_PX2 = S_M_PER_PX**2            # m²/px²


def ensure_morphology_cols(df: pd.DataFrame, eps: float = 1e-9) -> pd.DataFrame:
    """Ensure 'circ' and 'defect_frac' exist. Compute if missing using area/perimeter/defect_area."""
    out = df.copy()
    if "circ" not in out.columns:
        if not {"area_px", "perim_px"}.issubset(out.columns):
            raise ValueError("Need columns 'area_px' and 'perim_px' to compute circularity 'circ'.")
        P2 = np.clip(np.asarray(out["perim_px"], float)**2, eps, np.inf)
        out["circ"] = (4.0*np.pi*np.asarray(out["area_px"], float) / P2).clip(0.0, 1.0)
    if "defect_frac" not in out.columns:
        if "defect_area_px" in out.columns:
            A = np.clip(np.asarray(out["area_px"], float), eps, np.inf)
            out["defect_frac"] = (np.asarray(out["defect_area_px"], float) / A).clip(0.0, 1.0)
        else:
            out["defect_frac"] = 0.0
    return out


def growth_speed_mps_from_ray(df: pd.DataFrame,
                              v0_px_per_ms: float | None = None,
                              alpha: float = 0.0,
                              beta: float = 0.0,
                              use_penalties: bool = False) -> np.ndarray:
    """
    Compute radial growth speed v in m/s.
    If use_penalties=False or v0 is None: v = ray/dt (data-derived) -> convert to m/s.
    If use_penalties=True and v0 provided: v = v0 * exp(-alpha*(1-C)) * exp(-beta*phi) -> convert to m/s.
    Requires columns: ray_px, t0_ms, plus circ/defect_frac (computed if absent).
    """
    df = ensure_morphology_cols(df)
    C   = np.asarray(df["circ"], float)
    phi = np.asarray(df["defect_frac"], float)
    ray = np.asarray(df["ray_px"], float)
    dt  = np.clip(600.0 - np.asarray(df["t0_ms"], float), 1e-9, np.inf)  # ms

    if not use_penalties or v0_px_per_ms is None:
        v_px_per_ms = ray / dt
        v_m_per_s   = v_px_per_ms * S_M_PER_PX * 1e3
    else:
        f = np.exp(-alpha*(1.0 - C)) * np.exp(-beta*phi)
        v_m_per_s = (v0_px_per_ms * f) * S_M_PER_PX * 1e3
    return v_m_per_s


def nucleation_I_SI(dn_dt_ms: np.ndarray, A_eff_px: float) -> np.ndarray:
    """
    Convert dn/dt (nuclei per ms) to nucleation rate density I(t) in m^-2 s^-1,
    using effective observed area A_eff in pixels.
    """
    A_m2 = A_eff_px * S_M2_PER_PX2
    if A_m2 <= 0:
        raise ValueError("A_eff_px must be > 0 to compute I(t) in SI units.")
    dn_dt_s = np.asarray(dn_dt_ms, float) * 1e3  # nuclei/s
    I = dn_dt_s / A_m2
    return I


def main():
    ap = argparse.ArgumentParser(
        description="Convert growth speeds and nucleation rates to SI units (m/s and m^-2 s^-1)."
    )
    ap.add_argument("--objects_csv", required=True,
                    help="Per-object table CSV with at least columns: ray_px, t0_ms, area_px, perim_px. Optional: defect_area_px, circ, defect_frac.")
    ap.add_argument("--dndt_csv", required=True,
                    help="Time series CSV with columns: t_ms, dn_dt (nuclei per ms).")
    ap.add_argument("--label", default="FAPI", help="Label used in output filenames.")
    ap.add_argument("--outdir", required=True, help="Output folder.")
    ap.add_argument("--area_eff_px", type=float, default=None,
                    help="Effective observed area in pixels. If omitted, uses sum(area_px) as a proxy.")
    ap.add_argument("--v0_px_per_ms", type=float, default=None,
                    help="Optional base growth speed v0 in px/ms if you want to apply penalties mode.")
    ap.add_argument("--alpha", type=float, default=0.0, help="Penalty weight for (1-circ).")
    ap.add_argument("--beta",  type=float, default=0.0, help="Penalty weight for defect_frac.")
    ap.add_argument("--use_penalties", action="store_true",
                    help="If set with v0_px_per_ms, compute v=v0*exp(-alpha*(1-C))*exp(-beta*phi); else v=ray/dt.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    obj = pd.read_csv(args.objects_csv)
    it  = pd.read_csv(args.dndt_csv)

    # Basic checks/cleanup
    needed = {"ray_px", "t0_ms", "area_px", "perim_px"}
    missing = needed - set(obj.columns)
    if missing:
        raise ValueError(f"Missing required columns in objects CSV: {sorted(missing)}")

    if not {"t_ms", "dn_dt"}.issubset(it.columns):
        raise ValueError("dndt_csv must contain columns: 't_ms' and 'dn_dt'")

    obj = obj.replace([np.inf, -np.inf], np.nan).dropna(subset=["ray_px", "t0_ms", "area_px", "perim_px"])

    # Effective area choice
    if args.area_eff_px is None:
        # Proxy: sum of areas (you can replace with union area if available)
        A_eff_px = float(np.nansum(obj["area_px"].to_numpy()))
        area_note = "A_eff from sum(area_px)"
    else:
        A_eff_px = float(args.area_eff_px)
        area_note = "A_eff provided by user"

    # Growth speeds (m/s)
    v_mps = growth_speed_mps_from_ray(
        obj,
        v0_px_per_ms=args.v0_px_per_ms,
        alpha=args.alpha,
        beta=args.beta,
        use_penalties=args.use_penalties
    )
    obj_SI = obj.copy()
    obj_SI["v_m_per_s"] = v_mps
    obj_SI["ray_um"] = np.asarray(obj_SI["ray_px"], float) * S_UM_PER_PX
    obj_SI["ray_m"]  = np.asarray(obj_SI["ray_px"], float) * S_M_PER_PX

    # Nucleation I(t) (m^-2 s^-1)
    I_SI = nucleation_I_SI(it["dn_dt"].to_numpy(), A_eff_px)
    it_SI = it.copy()
    it_SI["I_m^-2_s^-1"] = I_SI

    # Save outputs
    obj_out = outdir / f"{args.label}_per_object_SI.csv"
    it_out  = outdir / f"{args.label}_nucleation_I_SI.csv"
    obj_SI.to_csv(obj_out, index=False)
    it_SI.to_csv(it_out, index=False)

    # Summary
    with open(outdir / f"{args.label}_SI_summary.txt", "w") as f:
        f.write(f"Label: {args.label}\n")
        f.write(f"Scale: {S_UM_PER_PX:.8f} µm/px; {S_M_PER_PX:.8e} m/px; {S_M2_PER_PX2:.3e} m²/px²\n")
        f.write(f"A_eff_px = {A_eff_px:.0f} px  ({area_note})\n")
        f.write(f"A_eff_m2 ≈ {A_eff_px*S_M2_PER_PX2:.3e} m²\n")
        f.write(f"v (median) ≈ {np.median(v_mps):.3e} m/s\n")
        f.write(f"v (mean)   ≈ {np.mean(v_mps):.3e} m/s\n")
        f.write(f"v (P90)    ≈ {np.percentile(v_mps,90):.3e} m/s\n")
        if args.use_penalties and args.v0_px_per_ms is not None:
            f.write(f"Penalties used with v0={args.v0_px_per_ms} px/ms, alpha={args.alpha}, beta={args.beta}\n")
        else:
            f.write("Penalties not used; v=ray/dt derived from data.\n")
        f.write("Outputs:\n")
        f.write(f" - {obj_out.name}\n")
        f.write(f" - {it_out.name}\n")

    print("=== Done ===")
    print(f"Per-object SI metrics: {obj_out}")
    print(f"Nucleation I(t) SI:   {it_out}")
    print(f"Summary:              {outdir / (args.label + '_SI_summary.txt')}")
    print(f"Area note: {area_note}")


if __name__ == "__main__":
    main()
