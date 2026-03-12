#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
harmonize_video_pair_inputs.py

Checkpoint-tree harmonizer for the video-pair:
  - FAPI
  - FAPI_TEMPO

Reads canonical_inputs/* and produces harmonized_tables/*:
  - active_table_harmonized.csv (from frame_kinetics.csv; same "active" criterion)
  - growth_table_harmonized.csv (standardized µm/s + time)
  - tau_table_harmonized.csv (if available; carries QC metadata)
  - qc_summary.csv (per dataset)

Also writes tau_unavailable.flag under canonical_inputs/<dataset>/ if tau is missing.

Assumes folder tree:
Kinetics/harmonization_checkpoints/v001_video_pair_fapi_vs_fapi_tempo/
  canonical_inputs/
    FAPI/
    FAPI_TEMPO/
  harmonized_tables/
    FAPI/
    FAPI_TEMPO/
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# User-controlled constants
# ----------------------------
CHECKPOINT_VERSION = "v001_video_pair_fapi_vs_fapi_tempo"
UM_PER_PX = 0.065  # µm/px for both datasets (as agreed)

DATASETS = ["FAPI", "FAPI_TEMPO"]


# ----------------------------
# Logging helpers
# ----------------------------
def info(msg: str) -> None:
    print(f"[INFO] {msg}")

def warn(msg: str) -> None:
    print(f"[WARN] {msg}")

def ok(msg: str) -> None:
    print(f"[OK] {msg}")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def find_checkpoint_dir() -> Path:
    """
    Find .../harmonization_checkpoints/v001_video_pair_fapi_vs_fapi_tempo
    from cwd or script path.
    """
    candidates = []
    try:
        candidates.append(Path(__file__).resolve())
    except Exception:
        pass
    candidates.append(Path.cwd().resolve())

    for start in candidates:
        for p in [start] + list(start.parents):
            if p.name == CHECKPOINT_VERSION and p.parent.name == "harmonization_checkpoints":
                return p
            nested = p / "harmonization_checkpoints" / CHECKPOINT_VERSION
            if nested.exists():
                return nested.resolve()

    raise FileNotFoundError(
        f"Could not locate checkpoint directory '{CHECKPOINT_VERSION}'. "
        "Run from Kinetics root or from checkpoint/scripts."
    )


# ----------------------------
# Column utilities
# ----------------------------
def pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find any of columns {candidates}. Found: {list(df.columns)}")

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def standardize_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee:
      - time_ms (float)
      - t_s (float)
    Accepts possible time columns: time_ms, t_ms, t, t_s
    """
    out = df.copy()

    if "time_ms" not in out.columns:
        if "t_ms" in out.columns:
            out["time_ms"] = to_num(out["t_ms"])
        elif "t" in out.columns:
            out["time_ms"] = to_num(out["t"])
        elif "t_s" in out.columns:
            out["time_ms"] = to_num(out["t_s"]) * 1000.0

    if "t_s" not in out.columns:
        if "time_ms" in out.columns:
            out["t_s"] = to_num(out["time_ms"]) / 1000.0

    return out


# ----------------------------
# ACTIVE: frame_kinetics -> active_table_harmonized.csv
# ----------------------------
def compute_active_from_frame_kinetics(frame_df: pd.DataFrame, dataset_name: str) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Active criterion (symmetric, dataset-agnostic):
      For each frame:
        - Identify background as the single object with the largest area_px in that frame.
        - Exclude it.
        - Active grains = count of remaining objects.
      Additionally produce:
        - n_total_objects (all objects in frame)
        - n_background_excluded (always 1 if frame has >=1 object)
        - sum_area_px / sum_area_um2 for non-background objects
        - mean_R_px / mean_R_um for non-background objects
    Required columns: frame_id (or frame_i), time_ms (or t_ms), area_px, R_px, annotation_id optional.
    """
    df = frame_df.copy()

    # Basic sanity: must have frame identifier + area + radius
    frame_col = "frame_i" if "frame_i" in df.columns else ("frame_id" if "frame_id" in df.columns else None)
    if frame_col is None:
        raise KeyError(f"{dataset_name} frame_kinetics: missing frame_i/frame_id. cols={list(df.columns)}")
    if "area_px" not in df.columns and "area" not in df.columns:
        raise KeyError(f"{dataset_name} frame_kinetics: missing area_px. cols={list(df.columns)}")

    if "area_px" not in df.columns:
        df["area_px"] = to_num(df["area"])
    else:
        df["area_px"] = to_num(df["area_px"])

    # Radius: prefer R_px; otherwise compute from area
    if "R_px" not in df.columns:
        df["R_px"] = np.sqrt(df["area_px"] / math.pi)
    else:
        df["R_px"] = to_num(df["R_px"])

    # time columns
    df = standardize_time_columns(df)
    if "time_ms" not in df.columns:
        raise KeyError(f"{dataset_name} frame_kinetics: missing time_ms/t_ms/t_s. cols={list(df.columns)}")

    # frame index normalization
    # If we have "frame_id" like frame_00012_t24.00ms, we keep it; also make numeric frame_i if possible.
    if frame_col == "frame_id" and "frame_i" not in df.columns:
        # Best-effort extract integer from frame_id
        extracted = df["frame_id"].astype(str).str.extract(r"frame_(\d+)", expand=False)
        df["frame_i"] = pd.to_numeric(extracted, errors="coerce")
    elif frame_col == "frame_i":
        df["frame_i"] = to_num(df["frame_i"])

    # Drop rows without core fields
    df = df.dropna(subset=["time_ms", "area_px", "R_px"]).copy()

    # Group by frame and compute background exclusion by max area
    rows = []
    for frame_key, g in df.groupby(frame_col, sort=True):
        g = g.copy()
        g = g.dropna(subset=["area_px"])
        n_total = int(len(g))
        if n_total == 0:
            continue

        # background = max area object
        bg_idx = g["area_px"].idxmax()
        g_nonbg = g.drop(index=bg_idx)

        n_bg_excl = 1
        n_active = int(len(g_nonbg))

        sum_area_px = float(g_nonbg["area_px"].sum()) if n_active > 0 else 0.0
        sum_area_um2 = sum_area_px * (UM_PER_PX ** 2)

        mean_R_px = float(g_nonbg["R_px"].mean()) if n_active > 0 else float("nan")
        mean_R_um = mean_R_px * UM_PER_PX if np.isfinite(mean_R_px) else float("nan")

        # time for this frame: use median of time_ms in the group (all should match)
        t_ms = float(np.nanmedian(g["time_ms"].to_numpy(dtype=float)))
        t_s = t_ms / 1000.0

        # numeric frame_i
        if "frame_i" in g.columns and pd.notna(g["frame_i"]).any():
            fi = float(np.nanmedian(pd.to_numeric(g["frame_i"], errors="coerce")))
        else:
            fi = float("nan")

        rows.append({
            "frame_i": int(fi) if np.isfinite(fi) else "",
            "frame_key": str(frame_key),
            "time_ms": t_ms,
            "t_s": t_s,

            # counts
            "n_total_objects": n_total,
            "n_background_excluded": n_bg_excl,
            "n_active_grains": n_active,

            # aliases for compare script robustness
            "n_active": n_active,
            "active": n_active,

            # size summaries (non-background)
            "sum_area_px": sum_area_px,
            "sum_area_um2": sum_area_um2,
            "mean_R_px": mean_R_px,
            "mean_R_um": mean_R_um,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"{dataset_name}: frame_kinetics produced empty active table.")

    out = out.sort_values("time_ms").reset_index(drop=True)

    qc = {
        "active_rows": int(len(out)),
        "active_time_ms_min": float(out["time_ms"].min()),
        "active_time_ms_max": float(out["time_ms"].max()),
        "active_n_active_mean": float(pd.to_numeric(out["n_active_grains"], errors="coerce").mean()),
        "active_n_active_max": int(pd.to_numeric(out["n_active_grains"], errors="coerce").max()),
    }
    return out, qc


# ----------------------------
# GROWTH: growth_rate_vs_time -> growth_table_harmonized.csv
# ----------------------------
def standardize_growth_table(growth_df: pd.DataFrame, dataset_name: str) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Normalize growth table to:
      - time_ms
      - t_s
      - median_growth_um_per_s  (best effort from available columns)
    Accepts:
      time_ms/t_ms/t_s
      growth columns like:
        median_um_per_s_clipped, median_um_per_s_raw, median_um_per_s,
        median_px_per_s, median_growth_px_per_s, growth_px_per_s,
        median_growth_um_per_s, growth_um_per_s, etc.
    """
    g = growth_df.copy()
    g = standardize_time_columns(g)

    if "time_ms" not in g.columns:
        raise KeyError(f"{dataset_name} growth_rate_vs_time: missing time column. cols={list(g.columns)}")

    # pick best growth column
    candidates = [
        # already in µm/s (preferred)
        "median_growth_um_per_s",
        "median_um_per_s_clipped",
        "median_um_per_s_raw",
        "median_um_per_s",
        "growth_um_per_s",
        "um_per_s",
        # px/s (convert)
        "median_growth_px_per_s",
        "median_px_per_s",
        "growth_px_per_s",
        "px_per_s",
        # generic
        "median",
        "growth_rate",
        "rate",
    ]
    growth_col = pick_first_existing(g, candidates)

    vals = to_num(g[growth_col])

    # convert px/s -> um/s when needed
    if growth_col.endswith("_px_per_s") or growth_col in {"median", "growth_rate", "rate", "px_per_s", "median_px_per_s", "median_growth_px_per_s", "growth_px_per_s"}:
        # if values look already small/um scale, still treat as px/s only if explicit; otherwise keep as-is
        # We convert only if column indicates px.
        if "px" in growth_col:
            vals_um_s = vals * UM_PER_PX
        else:
            vals_um_s = vals  # unknown; leave as-is
    else:
        vals_um_s = vals  # assumed already µm/s

    out = pd.DataFrame({
        "time_ms": to_num(g["time_ms"]),
        "t_s": to_num(g["time_ms"]) / 1000.0,
        "median_growth_um_per_s": vals_um_s,
        "growth_source_col": growth_col,
    }).dropna(subset=["time_ms"]).sort_values("time_ms").reset_index(drop=True)

    qc = {
        "growth_rows": int(len(out)),
        "growth_time_ms_min": float(out["time_ms"].min()) if len(out) else float("nan"),
        "growth_time_ms_max": float(out["time_ms"].max()) if len(out) else float("nan"),
        "growth_col_used": growth_col,
        "growth_median_um_s_median": float(pd.to_numeric(out["median_growth_um_per_s"], errors="coerce").median()) if len(out) else float("nan"),
    }
    return out, qc


# ----------------------------
# TAU: tau_fits -> tau_table_harmonized.csv
# ----------------------------
def standardize_tau_table(tau_df: pd.DataFrame, dataset_name: str) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Standardize tau table to contain:
      track_id, tau_ms, tau_s,
      fit_status, tau_qc_pass, tau_qc_failure_reasons,
      and QC metadata passthrough:
        fit_r2, fit_nrmse_range, fit_window_over_tau, tau_source_status

    Accepts tau_ms (preferred) or tau (assumed ms) or tau_s (convert).
    """
    t = tau_df.copy()

    if "track_id" not in t.columns:
        raise KeyError(f"{dataset_name} tau table missing track_id. cols={list(t.columns)}")

    t["track_id"] = pd.to_numeric(t["track_id"], errors="coerce")
    t = t.dropna(subset=["track_id"]).copy()
    t["track_id"] = t["track_id"].astype(int)

    # tau candidates
    if "tau_ms" in t.columns:
        t["tau_ms"] = to_num(t["tau_ms"])
    elif "tau" in t.columns:
        t["tau_ms"] = to_num(t["tau"])
    elif "tau_s" in t.columns:
        t["tau_ms"] = to_num(t["tau_s"]) * 1000.0
    else:
        raise KeyError(f"{dataset_name} tau table missing tau_ms/tau/tau_s. cols={list(t.columns)}")

    t["tau_s"] = t["tau_ms"] / 1000.0

    # carry QC fields if present; otherwise create empty columns
    for c in ["fit_status", "tau_qc_pass", "tau_qc_failure_reasons", "fit_r2", "fit_nrmse_range", "fit_window_over_tau", "tau_source_status"]:
        if c not in t.columns:
            t[c] = "" if c in {"fit_status", "tau_qc_failure_reasons", "tau_source_status"} else np.nan

    # normalize tau_qc_pass to bool-ish
    if "tau_qc_pass" in t.columns:
        # handle strings/ints
        t["tau_qc_pass"] = t["tau_qc_pass"].apply(lambda x: str(x).strip().lower() in {"true", "1", "yes", "y"})

    out_cols = [
        "track_id",
        "tau_ms", "tau_s",
        "fit_status",
        "tau_qc_pass",
        "tau_qc_failure_reasons",
        "fit_r2", "fit_nrmse_range", "fit_window_over_tau",
        "tau_source_status",
    ]

    # keep extra columns too (helpful), but ensure required set present
    base = t[out_cols].copy()

    qc = {
        "tau_rows": int(len(base)),
        "tau_ms_median": float(pd.to_numeric(base["tau_ms"], errors="coerce").median()) if len(base) else float("nan"),
        "tau_qc_pass_count": int(base["tau_qc_pass"].sum()) if len(base) else 0,
        "tau_fit_status_ok_count": int((base["fit_status"].astype(str) == "ok").sum()) if len(base) else 0,
    }
    return base.sort_values("track_id").reset_index(drop=True), qc


# ----------------------------
# QC Summary writer
# ----------------------------
def write_qc_summary(out_path: Path, dataset_name: str, qc_items: Dict[str, object], notes: List[str]) -> None:
    rows = [{"dataset": dataset_name, "key": k, "value": v} for k, v in qc_items.items()]
    if notes:
        for i, n in enumerate(notes, start=1):
            rows.append({"dataset": dataset_name, "key": f"note_{i:02d}", "value": n})
    pd.DataFrame(rows).to_csv(out_path, index=False)


# ----------------------------
# Main
# ----------------------------
def main() -> int:
    info("Harmonizing video-pair inputs (active/growth/tau) ...")
    checkpoint_dir = find_checkpoint_dir()
    info(f"CHECKPOINT_DIR  : {checkpoint_dir}")
    info(f"UM_PER_PX       : {UM_PER_PX:.6f}")

    canonical_root = checkpoint_dir / "canonical_inputs"
    harmonized_root = checkpoint_dir / "harmonized_tables"
    ensure_dir(harmonized_root)

    for ds in DATASETS:
        info(f"--- {ds} ---")
        ds_can = canonical_root / ds
        ds_out = harmonized_root / ds
        ensure_dir(ds_out)

        qc: Dict[str, object] = {
            "um_per_px": UM_PER_PX,
        }
        notes: List[str] = []

        # -------- ACTIVE --------
        fk_path = ds_can / "frame_kinetics.csv"
        if fk_path.exists():
            try:
                fk = pd.read_csv(fk_path)
                active_table, qc_active = compute_active_from_frame_kinetics(fk, ds)
                active_out = ds_out / "active_table_harmonized.csv"
                active_table.to_csv(active_out, index=False)
                qc.update(qc_active)
                ok(f"{ds}: active_table_harmonized.csv ({len(active_table)} rows)")
            except Exception as e:
                notes.append(f"active_failed:{e}")
                warn(f"{ds}: active harmonization failed: {e}")
        else:
            notes.append("active_missing_frame_kinetics")
            warn(f"{ds}: Missing frame_kinetics.csv in canonical_inputs. Active harmonization not possible.")

        # -------- GROWTH --------
        gr_path = ds_can / "growth_rate_vs_time.csv"
        if gr_path.exists():
            try:
                gr = pd.read_csv(gr_path)
                growth_table, qc_growth = standardize_growth_table(gr, ds)
                growth_out = ds_out / "growth_table_harmonized.csv"
                growth_table.to_csv(growth_out, index=False)
                qc.update(qc_growth)
                ok(f"{ds}: growth_table_harmonized.csv ({len(growth_table)} rows)")
            except Exception as e:
                notes.append(f"growth_failed:{e}")
                warn(f"{ds}: growth standardization failed: {e}")
        else:
            notes.append("growth_missing_growth_rate_vs_time")
            warn(f"{ds}: Missing growth_rate_vs_time.csv in canonical_inputs. Growth harmonization skipped.")

        # -------- TAU --------
        tau_path = ds_can / "tau_fits.csv"
        tau_flag = ds_can / "tau_unavailable.flag"

        if tau_path.exists():
            try:
                tau_df = pd.read_csv(tau_path)
                tau_table, qc_tau = standardize_tau_table(tau_df, ds)
                tau_out = ds_out / "tau_table_harmonized.csv"
                tau_table.to_csv(tau_out, index=False)
                qc.update(qc_tau)
                ok(f"{ds}: tau_table_harmonized.csv ({len(tau_table)} rows)")

                # If tau exists, remove unavailable flag if present
                if tau_flag.exists():
                    try:
                        tau_flag.unlink()
                        ok(f"{ds}: removed {tau_flag.name}")
                    except Exception as e:
                        warn(f"{ds}: could not remove tau_unavailable.flag: {e}")

            except Exception as e:
                notes.append(f"tau_failed:{e}")
                warn(f"{ds}: tau standardization failed: {e}")
                # Ensure flag exists
                try:
                    tau_flag.write_text("tau unavailable or invalid; see qc_summary\n", encoding="utf-8")
                except Exception:
                    pass
        else:
            notes.append("tau_missing_tau_fits")
            warn(f"{ds}: Missing tau_fits.csv in canonical_inputs. Tau harmonization unavailable.")
            # Ensure flag exists
            try:
                tau_flag.write_text("tau unavailable; provide tau_fits.csv to enable tau panel\n", encoding="utf-8")
            except Exception:
                pass

        # -------- QC summary --------
        qc_out = ds_out / "qc_summary.csv"
        write_qc_summary(qc_out, ds, qc, notes)
        ok(f"{ds}: qc_summary.csv written")

    ok("Harmonization tables completed.")
    info("Output folder:")
    print(f"  {harmonized_root}")
    info("Next: run compare script on harmonized_tables/ for a fully symmetric publication-robust comparison.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())