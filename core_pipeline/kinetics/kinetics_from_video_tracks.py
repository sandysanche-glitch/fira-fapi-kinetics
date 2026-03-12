import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PX_PER_UM = 2.20014
UM_PER_PX = 1.0 / PX_PER_UM


def load_frame_times(path: Path) -> pd.DataFrame:
    ft = pd.read_csv(path)
    if not {"frame", "t_s"}.issubset(ft.columns):
        raise ValueError(f"{path} must contain columns: frame, t_s")
    ft = ft.sort_values("frame").drop_duplicates("frame")
    return ft


def make_lookup(ft: pd.DataFrame) -> dict:
    return dict(zip(ft["frame"].astype(int).values, ft["t_s"].astype(float).values))


def median_dt(ft: pd.DataFrame) -> float:
    t = ft["t_s"].to_numpy(float)
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


def guess_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def compute_growth(grain_tracks_csv: Path, ft_lookup: dict) -> pd.DataFrame:
    df = pd.read_csv(grain_tracks_csv)

    # flexible naming
    c_frame = guess_col(df, ["frame", "Frame"])
    c_tid   = guess_col(df, ["track_id", "trackId", "id"])
    c_area  = guess_col(df, ["area", "area_px", "area_px2", "Area"])

    if not (c_frame and c_tid and c_area):
        raise ValueError(f"{grain_tracks_csv} must include frame/track_id/area columns. Found: {df.columns.tolist()}")

    df = df.rename(columns={c_frame: "frame", c_tid: "track_id", c_area: "area_px2"})
    df["frame"] = df["frame"].astype(int)
    df["track_id"] = df["track_id"].astype(int)
    df["area_px2"] = df["area_px2"].astype(float)

    df["t_s"] = df["frame"].map(ft_lookup)
    df = df.dropna(subset=["t_s"]).sort_values(["track_id", "t_s"]).copy()

    rows = []
    for tid, g in df.groupby("track_id"):
        t = g["t_s"].to_numpy(float)
        A_px2 = g["area_px2"].to_numpy(float)
        R_px = np.sqrt(np.maximum(A_px2, 0) / np.pi)
        v_px_s = central_diff(R_px, t)

        out = g[["track_id", "frame", "t_s"]].copy()
        out["area_px2"] = A_px2
        out["R_px"] = R_px
        out["v_px_per_s"] = v_px_s

        # convert
        out["R_um"] = out["R_px"] * UM_PER_PX
        out["v_um_per_s"] = out["v_px_per_s"] * UM_PER_PX
        out["area_um2"] = out["area_px2"] * (UM_PER_PX ** 2)

        rows.append(out)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def compute_dn_dt(nuc_hist_csv: Path, ft_lookup: dict, roi_area_um2: float, dt_bin_s: float,
                  min_detections: int = 3, min_final_area_px: float = 50.0) -> pd.DataFrame:
    """
    Uses nucleation histogram / track summary style CSV.
    Needs at least: track_id, nuc_frame.
    If present, uses n_detections and final_area_px to filter fragments.
    """
    df = pd.read_csv(nuc_hist_csv)

    c_tid = guess_col(df, ["track_id", "TrackID", "id"])
    c_nf  = guess_col(df, ["nuc_frame", "nuc_fram", "nucFrame"])

    if not (c_tid and c_nf):
        raise ValueError(f"{nuc_hist_csv} must include track_id and nuc_frame. Found: {df.columns.tolist()}")

    df = df.rename(columns={c_tid: "track_id", c_nf: "nuc_frame"})
    df["track_id"] = df["track_id"].astype(int)
    df["nuc_frame"] = df["nuc_frame"].astype(int)

    # filters (if columns exist)
    if "n_detections" in df.columns:
        df = df[df["n_detections"].fillna(0).astype(int) >= min_detections]
    if "final_area_px" in df.columns:
        df = df[df["final_area_px"].fillna(0).astype(float) >= min_final_area_px]
    if "final_area_px2" in df.columns:
        df = df[df["final_area_px2"].fillna(0).astype(float) >= min_final_area_px]
    if "final_area" in df.columns:
        df = df[df["final_area"].fillna(0).astype(float) >= min_final_area_px]

    df["t_nuc_s"] = df["nuc_frame"].map(ft_lookup)
    df = df.dropna(subset=["t_nuc_s"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["t_s", "dn_dt_per_um2_per_s", "n_new", "n_cum"])

    t0 = float(df["t_nuc_s"].min())
    t1 = float(df["t_nuc_s"].max())
    edges = np.arange(t0, t1 + dt_bin_s, dt_bin_s)
    counts, edges = np.histogram(df["t_nuc_s"].to_numpy(float), bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    dn_dt = (counts / dt_bin_s) / roi_area_um2  # nuclei / (um^2*s)
    n_cum = np.cumsum(counts)

    return pd.DataFrame({"t_s": centers, "dn_dt_per_um2_per_s": dn_dt, "n_new": counts, "n_cum": n_cum})


def plot_dn_dt(df, out_png: Path, title: str):
    if df.empty:
        return
    plt.figure(figsize=(7,4))
    plt.plot(df["t_s"], df["dn_dt_per_um2_per_s"])
    plt.xlabel("time (s)")
    plt.ylabel("dn/dt (nuclei / (µm²·s))")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_growth(growth_df, out_png: Path, title: str):
    g = growth_df.dropna(subset=["v_um_per_s"]).copy()
    if g.empty:
        return
    # median across tracks at each time
    summary = g.groupby("t_s")["v_um_per_s"].median().reset_index()
    plt.figure(figsize=(7,4))
    plt.plot(summary["t_s"], summary["v_um_per_s"])
    plt.xlabel("time (s)")
    plt.ylabel("median v_eff (µm/s)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)

    ap.add_argument("--frame_times", required=True)      # produced by extract_video_times.py
    ap.add_argument("--grain_tracks", required=True)     # per-frame grain areas
    ap.add_argument("--nuc_hist", required=True)         # nuc_frame per track (+ filters)

    ap.add_argument("--roi_width_px", type=float, required=True)
    ap.add_argument("--roi_height_px", type=float, required=True)

    ap.add_argument("--dt_bin_s", type=float, default=None)
    ap.add_argument("--min_detections", type=int, default=3)
    ap.add_argument("--min_final_area_px", type=float, default=50.0)

    ap.add_argument("--out_dir", default="kinetics_outputs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ft = load_frame_times(Path(args.frame_times))
    lookup = make_lookup(ft)
    dt = median_dt(ft)
    print(f"[INFO] {args.label}: dt ~ {dt:.6f} s  (~{1/dt:.2f} fps)")

    # ROI area in µm²
    roi_area_px2 = float(args.roi_width_px) * float(args.roi_height_px)
    roi_area_um2 = roi_area_px2 * (UM_PER_PX ** 2)
    print(f"[INFO] ROI area: {roi_area_px2:.3e} px²  = {roi_area_um2:.3e} µm²")

    dt_bin_s = args.dt_bin_s if args.dt_bin_s is not None else dt

    dn_dt = compute_dn_dt(
        Path(args.nuc_hist), lookup, roi_area_um2, dt_bin_s,
        min_detections=args.min_detections,
        min_final_area_px=args.min_final_area_px,
    )
    growth = compute_growth(Path(args.grain_tracks), lookup)

    dn_csv = out_dir / f"{args.label}_dn_dt.csv"
    gr_csv = out_dir / f"{args.label}_growth_rates.csv"
    dn_dt.to_csv(dn_csv, index=False)
    growth.to_csv(gr_csv, index=False)
    print(f"[OK] wrote {dn_csv}")
    print(f"[OK] wrote {gr_csv}")

    plot_dn_dt(dn_dt, out_dir / f"{args.label}_dn_dt.png", f"{args.label}: dn/dt")
    plot_growth(growth, out_dir / f"{args.label}_growth_rate.png", f"{args.label}: growth rate")
    print(f"[OK] plots in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
