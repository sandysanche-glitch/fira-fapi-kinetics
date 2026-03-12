import os
import re
import math
import argparse
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def ensure_exists(path: str, label: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")


def parse_frame_from_frame_id(series: pd.Series) -> pd.Series:
    # expects strings like frame_00030_t60.00ms
    out = series.astype(str).str.extract(r"frame_(\d+)")[0]
    return pd.to_numeric(out, errors="coerce")


def bbox_iou_one_to_many(x, y, w, h, cand_df: pd.DataFrame) -> np.ndarray:
    if cand_df.empty:
        return np.array([], dtype=float)

    x1 = float(x)
    y1 = float(y)
    x2 = x1 + float(w)
    y2 = y1 + float(h)

    cx1 = cand_df["bbox_x"].to_numpy(dtype=float)
    cy1 = cand_df["bbox_y"].to_numpy(dtype=float)
    cx2 = cx1 + cand_df["bbox_w"].to_numpy(dtype=float)
    cy2 = cy1 + cand_df["bbox_h"].to_numpy(dtype=float)

    ix1 = np.maximum(x1, cx1)
    iy1 = np.maximum(y1, cy1)
    ix2 = np.minimum(x2, cx2)
    iy2 = np.minimum(y2, cy2)

    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih

    a = max(0.0, float(w)) * max(0.0, float(h))
    ca = np.maximum(0.0, cand_df["bbox_w"].to_numpy(dtype=float)) * np.maximum(0.0, cand_df["bbox_h"].to_numpy(dtype=float))
    union = a + ca - inter

    out = np.zeros_like(inter, dtype=float)
    m = union > 0
    out[m] = inter[m] / union[m]
    return out


def first_existing(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            return c
    return None


def build_rate_from_events(events_df: pd.DataFrame, bin_ms: float = 20.0) -> pd.DataFrame:
    """
    Build kinetics_tau0p3_rate.csv-like output:
    columns: t_center_ms, t_center_s, dN, dt_ms, dt_s, dNdt_per_s
    """
    if events_df.empty:
        return pd.DataFrame(columns=["t_center_ms", "t_center_s", "dN", "dt_ms", "dt_s", "dNdt_per_s"])

    t_ms = pd.to_numeric(events_df["t_ms"], errors="coerce").dropna().to_numpy(dtype=float)
    t_ms = np.sort(t_ms)

    t_min = float(np.min(t_ms))
    t_max = float(np.max(t_ms))

    # align bins to 20 ms grid
    start = math.floor(t_min / bin_ms) * bin_ms
    end = math.ceil(t_max / bin_ms) * bin_ms + bin_ms  # include last event

    edges = np.arange(start, end + 1e-9, bin_ms, dtype=float)
    counts, edges = np.histogram(t_ms, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    out = pd.DataFrame({
        "t_center_ms": centers,
        "t_center_s": centers / 1000.0,
        "dN": counts.astype(int),
        "dt_ms": float(bin_ms),
        "dt_s": float(bin_ms) / 1000.0,
    })
    out["dNdt_per_s"] = out["dN"] / out["dt_s"]
    return out


def build_Nt_from_events(events_df: pd.DataFrame, dt_ms: float = 2.0) -> pd.DataFrame:
    """
    Build kinetics_tau0p3_Nt.csv-like output on a regular 2 ms grid from onset to last event.
    columns: t_ms, t_s, N
    """
    if events_df.empty:
        return pd.DataFrame(columns=["t_ms", "t_s", "N"])

    ev = events_df.sort_values("t_ms").copy()
    t = pd.to_numeric(ev["t_ms"], errors="coerce").dropna().to_numpy(dtype=float)
    t = np.sort(t)

    t0 = float(np.min(t))
    t1 = float(np.max(t))

    grid = np.arange(t0, t1 + 1e-9, dt_ms, dtype=float)

    # cumulative count at each grid point
    counts = np.searchsorted(t, grid, side="right").astype(int)

    out = pd.DataFrame({
        "t_ms": grid,
        "t_s": grid / 1000.0,
        "N": counts
    })
    return out


# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Symmetric bbox artifact filtering for FAPI stable nucleation events")
    ap.add_argument("--events_csv", required=True, help="FAPI stable nucleation events (e.g., nucleation_events_filtered_FAPI.csv)")
    ap.add_argument("--tracks_csv", required=True, help="FAPI tracks.csv")
    ap.add_argument("--frame_kinetics_csv", required=True, help="FAPI frame_kinetics.csv")
    ap.add_argument("--out_dir", required=True, help="Output folder")
    ap.add_argument("--tau", type=float, default=0.3, help="bbox IoU threshold (default=0.3)")
    ap.add_argument("--bin_ms", type=float, default=20.0, help="rate bin width in ms (default=20)")
    ap.add_argument("--nt_dt_ms", type=float, default=2.0, help="N(t) grid spacing in ms (default=2)")
    args = ap.parse_args()

    ensure_exists(args.events_csv, "events_csv")
    ensure_exists(args.tracks_csv, "tracks_csv")
    ensure_exists(args.frame_kinetics_csv, "frame_kinetics_csv")
    os.makedirs(args.out_dir, exist_ok=True)

    print("[INFO] Loading CSVs...")
    ev = pd.read_csv(args.events_csv)
    tr = pd.read_csv(args.tracks_csv)
    fk = pd.read_csv(args.frame_kinetics_csv)

    # Standardize event columns
    if "nuc_frame_i" not in ev.columns and "nuc_frame_idx" in ev.columns:
        ev = ev.rename(columns={"nuc_frame_idx": "nuc_frame_i"})
    if "nuc_time_ms" not in ev.columns and "t_ms" in ev.columns:
        ev = ev.rename(columns={"t_ms": "nuc_time_ms"})

    required_ev = {"track_id", "nuc_frame_i"}
    miss_ev = [c for c in required_ev if c not in ev.columns]
    if miss_ev:
        raise ValueError(f"events_csv missing columns: {miss_ev}. Found: {list(ev.columns)}")

    # Standardize tracks columns
    if "frame" not in tr.columns:
        if "frame_id" in tr.columns:
            tr["frame"] = parse_frame_from_frame_id(tr["frame_id"])
        elif "frame_idx" in tr.columns:
            tr["frame"] = pd.to_numeric(tr["frame_idx"], errors="coerce")
    tr["frame"] = pd.to_numeric(tr["frame"], errors="coerce")
    if "time_ms" not in tr.columns and "t_ms" in tr.columns:
        tr["time_ms"] = pd.to_numeric(tr["t_ms"], errors="coerce")

    for c in ["track_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]:
        if c not in tr.columns:
            raise ValueError(f"tracks_csv missing required column: {c}")
        tr[c] = pd.to_numeric(tr[c], errors="coerce")

    # Standardize frame_kinetics columns
    if "frame" not in fk.columns:
        if "frame_id" in fk.columns:
            fk["frame"] = parse_frame_from_frame_id(fk["frame_id"])
        elif "frame_idx" in fk.columns:
            fk["frame"] = pd.to_numeric(fk["frame_idx"], errors="coerce")
    fk["frame"] = pd.to_numeric(fk["frame"], errors="coerce")

    for c in ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]:
        if c not in fk.columns:
            raise ValueError(f"frame_kinetics_csv missing required column: {c}")
        fk[c] = pd.to_numeric(fk[c], errors="coerce")

    # Optional background filtering
    if "class" in fk.columns:
        cls = fk["class"].astype(str).str.lower()
        fk = fk[~cls.isin(["background", "bg"])].copy()

    # Drop invalid boxes
    tr = tr.dropna(subset=["track_id", "frame", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]).copy()
    fk = fk.dropna(subset=["frame", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]).copy()

    # Build per-frame candidate lookup for previous frame detections
    fk_by_frame = {int(k): g.copy() for k, g in fk.groupby(fk["frame"].astype(int))}
    tr_by_tf = {}
    for _, r in tr.iterrows():
        key = (int(r["track_id"]), int(r["frame"]))
        # keep first row if duplicates happen
        if key not in tr_by_tf:
            tr_by_tf[key] = r

    print(f"[INFO] Loaded events={len(ev)}, tracks_rows={len(tr)}, frame_kinetics_rows={len(fk)}")

    # Compute symmetric bbox artifact score for each FAPI stable event
    rows = []
    for i, r in ev.iterrows():
        track_id = int(pd.to_numeric(r["track_id"], errors="coerce"))
        nuc_frame = int(pd.to_numeric(r["nuc_frame_i"], errors="coerce"))

        rec = dict(r)
        rec["track_id"] = track_id
        rec["nuc_frame_i"] = nuc_frame

        key = (track_id, nuc_frame)
        if key not in tr_by_tf:
            rec.update({
                "bbox_iou_prev_max": np.nan,
                "n_prev_cands": 0,
                "status": "no_track_row_at_nuc_frame",
            })
            rows.append(rec)
            continue

        rr = tr_by_tf[key]
        x = float(rr["bbox_x"])
        y = float(rr["bbox_y"])
        w = float(rr["bbox_w"])
        h = float(rr["bbox_h"])

        prev_frame = nuc_frame - 1
        cands = fk_by_frame.get(prev_frame, None)
        if cands is None or cands.empty:
            rec.update({
                "bbox_iou_prev_max": np.nan,
                "n_prev_cands": 0,
                "status": "empty_prev_frame",
            })
            rows.append(rec)
            continue

        ious = bbox_iou_one_to_many(x, y, w, h, cands)
        rec.update({
            "bbox_iou_prev_max": float(np.max(ious)) if len(ious) else np.nan,
            "n_prev_cands": int(len(ious)),
            "status": "ok",
        })
        rows.append(rec)

    out_all = pd.DataFrame(rows)

    # Ensure time column for downstream kinetics
    if "nuc_time_ms" in out_all.columns:
        out_all["t_ms"] = pd.to_numeric(out_all["nuc_time_ms"], errors="coerce")
    elif "time_ms" in out_all.columns:
        out_all["t_ms"] = pd.to_numeric(out_all["time_ms"], errors="coerce")
    else:
        # fallback from frame if 2 ms spacing
        out_all["t_ms"] = pd.to_numeric(out_all["nuc_frame_i"], errors="coerce") * 2.0

    # Save raw per-event scored table
    scored_path = os.path.join(args.out_dir, "nucleation_events_filtered_FAPI_with_bboxiou_symmetric.csv")
    out_all.to_csv(scored_path, index=False)

    # Tau sweep (same style as TEMPO)
    tau_values = [0.2, 0.3, 0.4]
    sweep_rows = []
    for tau in tau_values:
        kept = out_all[(out_all["status"] == "ok") & (pd.to_numeric(out_all["bbox_iou_prev_max"], errors="coerce") < tau)].copy()
        kept = kept.sort_values("t_ms")
        tag = str(tau).replace(".", "p")
        out_tau = os.path.join(args.out_dir, f"clean_bbox_tau{tag}_FAPI.csv")
        kept.to_csv(out_tau, index=False)
        sweep_rows.append({
            "tau": tau,
            "kept_events": int(len(kept)),
            "eligible_ok": int((out_all["status"] == "ok").sum())
        })

        if abs(tau - args.tau) < 1e-12:
            # Build matched kinetics files for selected tau
            events_out = kept[["track_id", "nuc_frame_i", "t_ms"]].copy()
            events_out["t_s"] = events_out["t_ms"] / 1000.0
            events_out = events_out.sort_values("t_ms")
            events_path = os.path.join(args.out_dir, "kinetics_tau0p3_events_FAPI.csv")
            events_out.to_csv(events_path, index=False)

            Nt = build_Nt_from_events(events_out, dt_ms=args.nt_dt_ms)
            Nt_path = os.path.join(args.out_dir, "kinetics_tau0p3_Nt.csv")
            Nt.to_csv(Nt_path, index=False)

            rate = build_rate_from_events(events_out, bin_ms=args.bin_ms)
            rate_path = os.path.join(args.out_dir, "kinetics_tau0p3_rate.csv")
            rate.to_csv(rate_path, index=False)

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_path = os.path.join(args.out_dir, "bbox_tau_sweep_summary_FAPI_symmetric.csv")
    sweep_df.to_csv(sweep_path, index=False)

    # Summary print
    ok_mask = out_all["status"] == "ok"
    ok_n = int(ok_mask.sum())
    print("[OK] Wrote:")
    print(f"  {scored_path}")
    print(f"  {sweep_path}")
    print(f"[SUMMARY] status counts:\n{out_all['status'].value_counts().to_string()}")
    if ok_n > 0:
        vals = pd.to_numeric(out_all.loc[ok_mask, "bbox_iou_prev_max"], errors="coerce")
        print(f"[SUMMARY] bbox_iou_prev_max on eligible events: min={vals.min():.6f}, median={vals.median():.6f}, mean={vals.mean():.6f}, max={vals.max():.6f}")
    print("[OK] If tau=0.3 was selected, also wrote:")
    print(f"  {os.path.join(args.out_dir, 'kinetics_tau0p3_events_FAPI.csv')}")
    print(f"  {os.path.join(args.out_dir, 'kinetics_tau0p3_Nt.csv')}")
    print(f"  {os.path.join(args.out_dir, 'kinetics_tau0p3_rate.csv')}")


if __name__ == "__main__":
    main()