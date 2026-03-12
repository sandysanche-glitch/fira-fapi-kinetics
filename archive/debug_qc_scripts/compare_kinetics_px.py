import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_frame_times(path: Path) -> pd.DataFrame:
    ft = pd.read_csv(path)
    if not {"frame", "t_s"}.issubset(ft.columns):
        raise ValueError(f"{path} must contain columns: frame,t_s")
    ft = ft.sort_values("frame").drop_duplicates("frame")
    ft["frame"] = ft["frame"].astype(int)
    ft["t_s"] = ft["t_s"].astype(float)
    ft["t_rel_s"] = ft["t_s"] - float(ft["t_s"].min())
    return ft


def make_lookup(ft, use_rel=True):
    key = "t_rel_s" if use_rel else "t_s"
    return dict(zip(ft["frame"].astype(int).values, ft[key].astype(float).values))


def median_dt(ft: pd.DataFrame) -> float:
    t = ft["t_rel_s"].to_numpy(float)
    if len(t) < 3:
        return np.nan
    return float(np.median(np.diff(t)))


def central_diff(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.full_like(y, np.nan, dtype=float)
    if len(y) < 3:
        return out
    for i in range(1, len(y) - 1):
        dt = t[i + 1] - t[i - 1]
        if dt > 0:
            out[i] = (y[i + 1] - y[i - 1]) / dt
    return out


def try_video_shape(video_path: str):
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        w = int(cap.get(3))
        h = int(cap.get(4))
        cap.release()
        if w > 0 and h > 0:
            return (w, h)
        return None
    except Exception:
        return None


def guess_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


# -----------------------------
# Nucleation: support 3 formats
# -----------------------------
def compute_dn_dt_from_histogram_csv(hist_csv: Path, roi_area_px2: float) -> pd.DataFrame:
    # t_center_ms, n_nucleated, j_rate_per_ms
    df = pd.read_csv(hist_csv)
    needed = {"t_center_ms", "n_nucleated", "j_rate_per_ms"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{hist_csv} missing {needed}. Found: {df.columns.tolist()}")

    out = pd.DataFrame()
    out["t_s"] = df["t_center_ms"].astype(float) / 1000.0
    j_per_s_total = df["j_rate_per_ms"].astype(float) * 1000.0
    out["dn_dt_per_px2_per_s"] = j_per_s_total / roi_area_px2
    out["n_new"] = df["n_nucleated"].astype(float)
    out["n_cum"] = out["n_new"].cumsum()
    out["n_density_per_px2"] = out["n_cum"] / roi_area_px2
    return out


def compute_dn_dt_from_frame_counts(frame_counts_csv: Path, ft_lookup: dict, roi_area_px2: float, dt_default: float) -> pd.DataFrame:
    """
    Format:
      frame, new_tracks
    """
    df = pd.read_csv(frame_counts_csv)
    if not {"frame", "new_tracks"}.issubset(df.columns):
        raise ValueError(f"{frame_counts_csv} must have columns frame,new_tracks. Found: {df.columns.tolist()}")

    df["frame"] = df["frame"].astype(int)
    df["new_tracks"] = df["new_tracks"].astype(float)

    df["t_s"] = df["frame"].map(ft_lookup)
    df = df.dropna(subset=["t_s"]).sort_values("t_s").copy()
    if df.empty:
        return pd.DataFrame(columns=["t_s","dn_dt_per_px2_per_s","n_new","n_cum","n_density_per_px2"])

    # use dt from mapping if possible, else dt_default
    # (here times are uniform anyway, but we compute a local dt for robustness)
    t = df["t_s"].to_numpy(float)
    dt = np.diff(t)
    dt_med = float(np.median(dt)) if len(dt) else float(dt_default)
    if dt_med <= 0:
        dt_med = float(dt_default)

    out = pd.DataFrame()
    out["t_s"] = t
    out["n_new"] = df["new_tracks"].to_numpy(float)
    out["n_cum"] = np.cumsum(out["n_new"].to_numpy(float))
    out["n_density_per_px2"] = out["n_cum"] / roi_area_px2
    out["dn_dt_per_px2_per_s"] = (out["n_new"] / dt_med) / roi_area_px2
    return out


def compute_dn_dt_from_track_summary(nuc_csv: Path, ft_lookup: dict, roi_area_px2: float,
                                    dt_bin_s: float, min_detections: int, min_final_area_px: float) -> pd.DataFrame:
    df = pd.read_csv(nuc_csv)

    c_nf = guess_col(df, ["nuc_frame", "nuc_fram", "nucFrame"])
    if c_nf is None:
        raise ValueError(f"{nuc_csv} is not track-summary format (no nuc_frame). Columns: {df.columns.tolist()}")

    df = df.rename(columns={c_nf: "nuc_frame"})
    df["nuc_frame"] = df["nuc_frame"].astype(int)

    if "n_detections" in df.columns:
        df = df[df["n_detections"].fillna(0).astype(int) >= int(min_detections)]

    for ac in ["final_area_px", "final_area", "final_area_px2"]:
        if ac in df.columns:
            df = df[df[ac].fillna(0).astype(float) >= float(min_final_area_px)]
            break

    df["t_nuc_s"] = df["nuc_frame"].map(ft_lookup)
    df = df.dropna(subset=["t_nuc_s"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["t_s","dn_dt_per_px2_per_s","n_new","n_cum","n_density_per_px2"])

    t0 = float(df["t_nuc_s"].min())
    t1 = float(df["t_nuc_s"].max())
    edges = np.arange(t0, t1 + dt_bin_s, dt_bin_s)
    counts, edges = np.histogram(df["t_nuc_s"].to_numpy(float), bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    dn_dt = (counts / dt_bin_s) / roi_area_px2
    n_cum = np.cumsum(counts)
    n_density = n_cum / roi_area_px2

    return pd.DataFrame({
        "t_s": centers,
        "dn_dt_per_px2_per_s": dn_dt,
        "n_new": counts,
        "n_cum": n_cum,
        "n_density_per_px2": n_density,
    })


def compute_dn_dt_auto(nuc_csv: Path, ft_lookup: dict | None, roi_area_px2: float,
                       dt_bin_s: float, dt_default: float,
                       min_detections: int, min_final_area_px: float) -> pd.DataFrame:
    peek = pd.read_csv(nuc_csv, nrows=5)
    cols = set(peek.columns)

    if {"t_center_ms", "n_nucleated", "j_rate_per_ms"}.issubset(cols):
        return compute_dn_dt_from_histogram_csv(nuc_csv, roi_area_px2)

    if {"frame", "new_tracks"}.issubset(cols):
        if ft_lookup is None:
            raise ValueError("frame/new_tracks nucleation needs frame->time mapping.")
        return compute_dn_dt_from_frame_counts(nuc_csv, ft_lookup, roi_area_px2, dt_default)

    if any(c in cols for c in ["nuc_frame", "nuc_fram", "nucFrame"]):
        if ft_lookup is None:
            raise ValueError("Track-summary nucleation needs frame->time mapping.")
        return compute_dn_dt_from_track_summary(nuc_csv, ft_lookup, roi_area_px2, dt_bin_s, min_detections, min_final_area_px)

    raise ValueError(f"Unknown nucleation CSV format: {nuc_csv}\nColumns: {sorted(list(cols))}")


# -----------------------------
# Growth from per-frame tracks
# -----------------------------
def compute_growth_from_per_frame_tracks(per_frame_csv: Path, ft_lookup: dict) -> pd.DataFrame:
    df = pd.read_csv(per_frame_csv)

    c_frame = guess_col(df, ["frame", "Frame"])
    c_tid   = guess_col(df, ["track_id", "trackId", "id"])
    c_area  = guess_col(df, ["area", "area_px", "area_px2", "Area"])
    if not (c_frame and c_tid and c_area):
        return pd.DataFrame(columns=["track_id","frame","t_s","area_px2","R_px","v_px_per_s"])

    df = df.rename(columns={c_frame: "frame", c_tid: "track_id", c_area: "area_px2"})
    df["frame"] = df["frame"].astype(int)
    df["track_id"] = df["track_id"].astype(int)
    df["area_px2"] = df["area_px2"].astype(float)

    df["t_s"] = df["frame"].map(ft_lookup)
    df = df.dropna(subset=["t_s"]).sort_values(["track_id", "t_s"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["track_id","frame","t_s","area_px2","R_px","v_px_per_s"])

    rows = []
    for tid, g in df.groupby("track_id"):
        t = g["t_s"].to_numpy(float)
        A = g["area_px2"].to_numpy(float)
        R = np.sqrt(np.maximum(A, 0) / np.pi)
        v = central_diff(R, t)

        out = g[["track_id", "frame", "t_s"]].copy()
        out["area_px2"] = A
        out["R_px"] = R
        out["v_px_per_s"] = v
        rows.append(out)

    return pd.concat(rows, ignore_index=True)


def summarize_growth_vs_time(growth_df: pd.DataFrame) -> pd.DataFrame:
    g = growth_df.dropna(subset=["v_px_per_s"]).copy()
    if g.empty:
        return pd.DataFrame(columns=["t_s","v_median_px_per_s","v_mean_px_per_s"])
    return g.groupby("t_s")["v_px_per_s"].agg(
        v_median_px_per_s="median",
        v_mean_px_per_s="mean"
    ).reset_index()


# -----------------------------
# Plotting
# -----------------------------
def plot_compare_dn_dt(df_a, label_a, df_b, label_b, out_png: Path):
    plt.figure(figsize=(7.5, 4.5))
    if not df_a.empty:
        plt.plot(df_a["t_s"], df_a["dn_dt_per_px2_per_s"], label=label_a)
    if not df_b.empty:
        plt.plot(df_b["t_s"], df_b["dn_dt_per_px2_per_s"], label=label_b)
    plt.xlabel("time (s, relative)")
    plt.ylabel("dn/dt (nuclei / (px²·s))")
    plt.title("Nucleation rate density dn/dt (pixel units)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_compare_growth(gsum_a, label_a, gsum_b, label_b, out_png: Path):
    plt.figure(figsize=(7.5, 4.5))
    if not gsum_a.empty:
        plt.plot(gsum_a["t_s"], gsum_a["v_median_px_per_s"], label=f"{label_a} (median)")
    if not gsum_b.empty:
        plt.plot(gsum_b["t_s"], gsum_b["v_median_px_per_s"], label=f"{label_b} (median)")
    plt.xlabel("time (s, relative)")
    plt.ylabel("v_eff (px/s)")
    plt.title("Median effective growth rate (pixel units)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Compute dn/dt (3 CSV formats) and growth (if per-frame tracks exist) in pixel units.")
    ap.add_argument("--out_dir", default="kinetics_outputs")

    ap.add_argument("--fapi_frame_times", default="FAPI_frame_times.csv")
    ap.add_argument("--tempo_frame_times", default="FAPI-TEMPO_frame_times.csv")

    ap.add_argument("--fapi_nuc", default=r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\comparative datasets\FAPI\FAPI_nucleation_hist.csv")
    ap.add_argument("--fapi_growth_tracks", default=None)

    ap.add_argument("--tempo_nuc", default=r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\comparative datasets\FAPI_TEMPO\nucleation_histogram.csv")
    ap.add_argument("--tempo_growth_tracks", default=r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\comparative datasets\FAPI_TEMPO\per_instance_tracks.csv")

    ap.add_argument("--fapi_video", default=r"F:\Sandy_data\Sandy\12.11.2025\sequences\v5\FAPI.avi")
    ap.add_argument("--tempo_video", default=r"F:\Sandy_data\Sandy\12.11.2025\sequences\v4\FAPI-TEMPO.avi")

    ap.add_argument("--min_detections", type=int, default=3)
    ap.add_argument("--min_final_area_px", type=float, default=50.0)
    ap.add_argument("--dt_bin_s", type=float, default=None)

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ROI areas
    sf = try_video_shape(args.fapi_video)
    st = try_video_shape(args.tempo_video)
    if not sf or not st:
        raise RuntimeError("Could not read video sizes. Provide readable AVI paths.")

    roi_fapi = float(sf[0]) * float(sf[1])
    roi_tempo = float(st[0]) * float(st[1])
    print(f"[INFO] FAPI: ROI from video {sf[0]}x{sf[1]} => {roi_fapi:.3e} px^2")
    print(f"[INFO] FAPI-TEMPO: ROI from video {st[0]}x{st[1]} => {roi_tempo:.3e} px^2")

    # time maps
    ft_fapi = load_frame_times(Path(args.fapi_frame_times))
    lookup_fapi = make_lookup(ft_fapi, use_rel=True)
    dt_fapi = median_dt(ft_fapi)

    ft_tempo = load_frame_times(Path(args.tempo_frame_times))
    lookup_tempo = make_lookup(ft_tempo, use_rel=True)
    dt_tempo = median_dt(ft_tempo)

    # binning for track-summary format only (ignored for histogram formats)
    bin_fapi = args.dt_bin_s if (args.dt_bin_s and args.dt_bin_s > 0) else dt_fapi
    bin_tempo = args.dt_bin_s if (args.dt_bin_s and args.dt_bin_s > 0) else dt_tempo

    # nucleation (auto)
    dn_fapi = compute_dn_dt_auto(Path(args.fapi_nuc), lookup_fapi, roi_fapi, bin_fapi, dt_fapi,
                                 args.min_detections, args.min_final_area_px)
    dn_tempo = compute_dn_dt_auto(Path(args.tempo_nuc), lookup_tempo, roi_tempo, bin_tempo, dt_tempo,
                                  args.min_detections, args.min_final_area_px)

    dn_fapi.to_csv(out_dir / "FAPI_dn_dt.csv", index=False)
    dn_tempo.to_csv(out_dir / "FAPI-TEMPO_dn_dt.csv", index=False)

    # growth
    if args.fapi_growth_tracks:
        growth_fapi = compute_growth_from_per_frame_tracks(Path(args.fapi_growth_tracks), lookup_fapi)
        gsum_fapi = summarize_growth_vs_time(growth_fapi)
        growth_fapi.to_csv(out_dir / "FAPI_growth_rates.csv", index=False)
        gsum_fapi.to_csv(out_dir / "FAPI_growth_summary.csv", index=False)
    else:
        gsum_fapi = pd.DataFrame()
        print("[WARN] FAPI: no per-frame growth tracks provided; will not plot FAPI growth curve.")

    growth_tempo = compute_growth_from_per_frame_tracks(Path(args.tempo_growth_tracks), lookup_tempo)
    gsum_tempo = summarize_growth_vs_time(growth_tempo)
    growth_tempo.to_csv(out_dir / "FAPI-TEMPO_growth_rates.csv", index=False)
    gsum_tempo.to_csv(out_dir / "FAPI-TEMPO_growth_summary.csv", index=False)

    # plots
    plot_compare_dn_dt(dn_fapi, "FAPI", dn_tempo, "FAPI-TEMPO", out_dir / "dn_dt_compare.png")

    if not gsum_fapi.empty:
        plot_compare_growth(gsum_fapi, "FAPI", gsum_tempo, "FAPI-TEMPO", out_dir / "growth_rate_compare.png")
    else:
        plot_compare_growth(pd.DataFrame(), "FAPI", gsum_tempo, "FAPI-TEMPO", out_dir / "growth_rate_TEMPO_only.png")

    print(f"[OK] Outputs saved in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
