#!/usr/bin/env python
import pandas as pd
import numpy as np

# -------------------------------------------------------------------
# USER INPUT: paths to your crystal-metrics CSV files
# -------------------------------------------------------------------
FAPI_CSV = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\crystal_metrics_fapi.csv"
FAPITEMPO_CSV = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\crystal_metrics_fapi_tempo.csv"

# -------------------------------------------------------------------
# USER INPUT: column names in those CSVs
#   Adjust these if your column names differ!
# -------------------------------------------------------------------
# area and defect areas (for defect fraction phi)
AREA_COL = "area_(µm²)"            # total grain area
DEFECT_AREA_COL = "defects_area_(µm²)"  # overlapping defect area

# entropy columns for ΔH = H0 - Hσ
ENTROPY_RAW_COL = "entropy(bits)"       # raw entropy
ENTROPY_HM_COL = "entropy_hm_(bits)"    # blurred / heat-map entropy

# OPTIONAL columns (only used if present)
POLAR_ANISO_COL = "A_polar"            # shape anisotropy (if you wrote it)
TEXTURE_ANISO_COL = "A_tex"            # texture anisotropy (if present)
VEFF_COL = "v_eff"                     # effective growth rate (if present)


# -------------------------------------------------------------------
def load_metrics(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"\n[INFO] Loaded {path}")
    print("[INFO] Columns:", list(df.columns))
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add defect_fraction and Delta_H columns if possible."""
    # defect fraction phi = A_def / A_grain
    if "defect_fraction" not in df.columns:
        if DEFECT_AREA_COL in df.columns and AREA_COL in df.columns:
            df["defect_fraction"] = df[DEFECT_AREA_COL] / df[AREA_COL]
            print("[INFO] Created 'defect_fraction' = "
                  f"{DEFECT_AREA_COL} / {AREA_COL}")
        else:
            print("[WARN] Could not create 'defect_fraction' "
                  f"(need '{DEFECT_AREA_COL}' and '{AREA_COL}')")

    # multiscale entropy contrast ΔH = H0 - Hσ
    if "Delta_H" not in df.columns:
        if ENTROPY_RAW_COL in df.columns and ENTROPY_HM_COL in df.columns:
            df["Delta_H"] = df[ENTROPY_RAW_COL] - df[ENTROPY_HM_COL]
            print("[INFO] Created 'Delta_H' = "
                  f"{ENTROPY_RAW_COL} - {ENTROPY_HM_COL}")
        else:
            print("[WARN] Could not create 'Delta_H' "
                  f"(need '{ENTROPY_RAW_COL}' and '{ENTROPY_HM_COL}')")

    return df


def mean_std(df: pd.DataFrame, col: str):
    vals = df[col].dropna().values
    if vals.size == 0:
        return np.nan, np.nan
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1))  # sample std
    return mean, std


def report_descriptor(df_fapi: pd.DataFrame,
                      df_tempo: pd.DataFrame,
                      col: str,
                      label_latex: str,
                      description: str = ""):
    if col not in df_fapi.columns or col not in df_tempo.columns:
        print(f"\n[SKIP] Column '{col}' not present in both datasets.")
        return

    m_f, s_f = mean_std(df_fapi, col)
    m_t, s_t = mean_std(df_tempo, col)

    print("\n" + "-" * 70)
    print(f"Descriptor {label_latex} (from column '{col}')")
    if description:
        print("  " + description)
    print(f"  FAPI       = {m_f:.4g} ± {s_f:.4g}")
    print(f"  FAPI–TEMPO = {m_t:.4g} ± {s_t:.4g}")

    # LaTeX snippet to paste into the table
    print("  LaTeX values field:")
    print(f"    ${m_f:.3g}\\!\\pm\\!{s_f:.3g}$ / "
          f"${m_t:.3g}\\!\\pm\\!{s_t:.3g}$ \\\\")


def main():
    # load
    df_fapi = load_metrics(FAPI_CSV)
    df_tempo = load_metrics(FAPITEMPO_CSV)

    # derive phi and ΔH
    df_fapi = add_derived_columns(df_fapi)
    df_tempo = add_derived_columns(df_tempo)

    # --- now report the descriptors for the table -------------------
    # A_polar (shape anisotropy) – only if you have already stored it
    report_descriptor(
        df_fapi, df_tempo,
        POLAR_ANISO_COL,
        r"A_{\mathrm{polar}}",
        "shape anisotropy from nucleation-centre polar profiles"
    )

    # A_tex (texture anisotropy)
    report_descriptor(
        df_fapi, df_tempo,
        TEXTURE_ANISO_COL,
        r"A_{\mathrm{tex}}",
        "texture anisotropy from gradient-orientation histograms"
    )

    # ΔH (multiscale entropy contrast)
    report_descriptor(
        df_fapi, df_tempo,
        "Delta_H",
        r"\Delta H",
        "multiscale entropy contrast = H_0 - H_\\sigma"
    )

    # defect fraction φ
    report_descriptor(
        df_fapi, df_tempo,
        "defect_fraction",
        r"\phi",
        "defect area fraction = A_def / A_grain"
    )

    # kinetic heterogeneity: CV(v_eff) = σ(v_eff)/⟨v_eff⟩
    if VEFF_COL in df_fapi.columns and VEFF_COL in df_tempo.columns:
        for name, df in [("FAPI", df_fapi), ("FAPI–TEMPO", df_tempo)]:
            vals = df[VEFF_COL].dropna().values
            if vals.size == 0:
                print(f"[WARN] No v_eff values for {name}")
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1))
            cv = std / mean if mean != 0 else np.nan
            print(f"\nKinetic heterogeneity for {name} "
                  f"(from '{VEFF_COL}'):")
            print(f"  mean v_eff = {mean:.4g}, std = {std:.4g}, "
                  f"CV = {cv:.4g}")

        # If you want a LaTeX row for CV(v_eff), just read off the CVs
        print("\n[NOTE] Use the two CV(v_eff) values above in the "
              "kinetic-heterogeneity row of your table.")
    else:
        print("\n[WARN] v_eff column not available; cannot compute "
              "kinetic heterogeneity from CSV.")


if __name__ == "__main__":
    main()
