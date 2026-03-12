# stable_nucleation_curves_v3_fixed.py
# Robust "stable nucleation" curves for FAPI + FAPI-TEMPO with FIXED bin edges for bootstrap.
#
# Usage (Windows):
# python stable_nucleation_curves_v3_fixed.py ^
#   --fapi_tracks  "...\FAPI\tracks.csv" ^
#   --tempo_tracks "...\FAPI_TEMPO\tracks.csv" ^
#   --out_dir      "...\stable_nucleation_compare_sharedbins" ^
#   --L 5 --amin_px 800 --rmin_px 3 --bin_ms 20 --bootstrap_B 1000 --seed 0 ^
#   --use_rmono_gate --rmono_min 0.6
#
# Outputs:
#  nucleation_events_filtered_FAPI.csv
#  nucleation_events_filtered_FAPI_TEMPO.csv
#  nucleation_events_rejected_*.csv
#  N_t_filtered_*.csv
#  dn_dt_filtered_*.csv
#  bootstrap_CI_FAPI_and_TEMPO.csv

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Standardize track CSV schemas
# -----------------------------
def standardize_tracks(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Convert different tracks.csv schemas into a common format:
      track_id (int)
      frame_idx (int)
      time_ms (float)
      area_px (float)
      R_px (float)
      cx, cy (float, optional)
      R_mono (float, optional)
    """
    d = df.copy()

    # track_id
    if "track_id" not in d.columns:
        raise KeyError(f"[{dataset_name}] missing 'track_id' column. Columns={list(d.columns)}")

    # frame index
    if "frame_idx" in d.columns:
        d["frame_idx"] = d["frame_idx"].astype(int)
    elif "frame" in d.columns:
        # often numeric frame number
        d["frame_idx"] = pd.to_numeric(d["frame"], errors="coerce").astype("Int64")
    elif "frame_id" in d.columns:
        # sometimes string like "frame_00065_t130.00ms"
        # extract 00065
        s = d["frame_id"].astype(str)
        d["frame_idx"] = s.str.extract(r"frame_(\d+)", expand=False).astype("Int64")
    else:
        raise KeyError(f"[{dataset_name}] tracks.csv must contain 'frame_idx' or 'frame' or 'frame_id'.")

    # time (ms)
    if "time_ms" in d.columns:
        d["time_ms"] = pd.to_numeric(d["time_ms"], errors="coerce")
    elif "t_ms" in d.columns:
        d["time_ms"] = pd.to_numeric(d["t_ms"], errors="coerce")
    else:
        # If time is missing, approximate by frame_idx * 2 ms? (NOT recommended)
        raise KeyError(f"[{dataset_name}] missing 'time_ms' or 't_ms' column.")

    # area & radius
    for col in ["area_px", "R_px"]:
        if col not in d.columns:
            raise KeyError(f"[{dataset_name}] missing '{col}' column. Columns={list(d.columns)}")
        d[col] = pd.to_numeric(d[col], errors="coerce")

    # optional fields
    for col in ["cx", "cy", "R_mono"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    # drop impossible rows
    d = d.dropna(subset=["track_id", "frame_idx", "time_ms", "area_px", "R_px"]).copy()
    d["track_id"] = pd.to_numeric(d["track_id"], errors="coerce").astype(int)
    d["frame_idx"] = d["frame_idx"].astype(int)

    # enforce sort order
    d = d.sort_values(["track_id", "frame_idx"]).reset_index(drop=True)
    return d


# -----------------------------
# Stable nucleation event logic
# -----------------------------
def find_stable_nucleation_events(
    tracks: pd.DataFrame,
    L: int,
    amin_px: float,
    rmin_px: float,
    use_rmono_gate: bool = False,
    rmono_min: float = 0.6,
):
    """
    For each track, define nucleation time as the first observation that is followed by
    >= L consecutive frames satisfying size criteria (A>=amin AND R>=rmin).
    Optional: require R_mono >= rmono_min (if available) to reject strong non-monotonic growth.

    Returns:
      accepted_events_df, rejected_events_df
    """
    acc = []
    rej = []

    need_rmono = use_rmono_gate

    for tid, g in tracks.groupby("track_id", sort=False):
        g = g.sort_values("frame_idx").reset_index(drop=True)

        frames = g["frame_idx"].to_numpy()
        times = g["time_ms"].to_numpy(dtype=float)
        area = g["area_px"].to_numpy(dtype=float)
        rad  = g["R_px"].to_numpy(dtype=float)

        # size pass per row
        good = (area >= amin_px) & (rad >= rmin_px)

        # find first i where next L frames are consecutive and all good
        nuc_i = None
        for i in range(len(g)):
            j = i + L - 1
            if j >= len(g):
                break
            # consecutive frame requirement
            if not np.all(frames[i:j+1] == np.arange(frames[i], frames[i] + L)):
                continue
            # size requirement over the window
            if not np.all(good[i:j+1]):
                continue
            nuc_i = i
            break

        if nuc_i is None:
            rej.append({
                "track_id": int(tid),
                "reason": f"no_window_L{L}_meets_size_and_consecutive",
                "n_points": int(len(g)),
                "first_time_ms": float(times[0]) if len(times) else np.nan,
                "last_time_ms": float(times[-1]) if len(times) else np.nan,
            })
            continue

        # optional monotonicity proxy gate (TEMPO has R_mono already)
        if need_rmono:
            if "R_mono" not in g.columns:
                # if user requests R_mono gate but it doesn't exist, do a simple proxy:
                # fraction of non-decreasing steps after nuc_i
                r = rad[nuc_i:]
                if len(r) >= 2:
                    frac_nondec = np.mean(np.diff(r) >= -0.5)  # allow tiny jitter
                else:
                    frac_nondec = 1.0
                rmono_val = float(frac_nondec)
            else:
                rmono_val = float(g.loc[nuc_i, "R_mono"]) if pd.notna(g.loc[nuc_i, "R_mono"]) else np.nan

            if not (np.isfinite(rmono_val) and rmono_val >= rmono_min):
                rej.append({
                    "track_id": int(tid),
                    "reason": f"rmono<{rmono_min}",
                    "n_points": int(len(g)),
                    "nuc_time_ms": float(times[nuc_i]),
                    "R_nuc_px": float(rad[nuc_i]),
                    "A_nuc_px": float(area[nuc_i]),
                    "R_mono": rmono_val,
                })
                continue

        acc.append({
            "track_id": int(tid),
            "nuc_frame_idx": int(frames[nuc_i]),
            "nuc_time_ms": float(times[nuc_i]),
            "R_nuc_px": float(rad[nuc_i]),
            "A_nuc_px": float(area[nuc_i]),
            "n_points": int(len(g)),
            "first_time_ms": float(times[0]),
            "last_time_ms": float(times[-1]),
            "R_mono_at_nuc": float(g.loc[nuc_i, "R_mono"]) if ("R_mono" in g.columns and pd.notna(g.loc[nuc_i, "R_mono"])) else np.nan,
        })

    acc_df = pd.DataFrame(acc).sort_values("nuc_time_ms", kind="mergesort").reset_index(drop=True)
    rej_df = pd.DataFrame(rej).reset_index(drop=True)
    return acc_df, rej_df


# -----------------------------
# Fixed-bin curves + bootstrap CI
# -----------------------------
def make_edges(times_ms, bin_ms, t_min=0.0):
    t = np.asarray(times_ms, dtype=float)
    tmax = float(np.nanmax(t)) if len(t) else 0.0
    tmax = bin_ms * np.ceil(tmax / bin_ms)
    edges = np.arange(float(t_min), float(tmax) + float(bin_ms), float(bin_ms), dtype=float)
    if len(edges) < 2:
        edges = np.array([0.0, float(bin_ms)], dtype=float)
    return edges


def curves_from_edges(times_ms, edges, bin_ms):
    counts, _ = np.histogram(np.asarray(times_ms, dtype=float), bins=edges)
    cumN = np.cumsum(counts)
    dn_dt = counts / (bin_ms / 1000.0)  # 1/s
    centers = 0.5 * (edges[:-1] + edges[1:])
    return pd.DataFrame({
        "bin_center_ms": centers,
        "n_nucleated": counts,
        "dn_dt_per_s": dn_dt,
        "cum_n": cumN,
    })


def bootstrap_CI_fixed(times_ms, edges, bin_ms, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.asarray(times_ms, dtype=float)
    n = len(t)
    centers = 0.5 * (edges[:-1] + edges[1:])
    m = len(centers)

    if n == 0:
        z = np.zeros(m, dtype=float)
        return pd.DataFrame({
            "bin_center_ms": centers,
            "N_lo": z, "N_med": z, "N_hi": z,
            "dn_dt_lo": z, "dn_dt_med": z, "dn_dt_hi": z,
        })

    N_boot = np.empty((B, m), dtype=float)
    dn_boot = np.empty((B, m), dtype=float)

    for b in range(B):
        samp = rng.choice(t, size=n, replace=True)
        counts, _ = np.histogram(samp, bins=edges)
        N_boot[b, :] = np.cumsum(counts)
        dn_boot[b, :] = counts / (bin_ms / 1000.0)

    def q(a, p): return np.quantile(a, p, axis=0)

    return pd.DataFrame({
        "bin_center_ms": centers,
        "N_lo": q(N_boot, 0.025),
        "N_med": q(N_boot, 0.50),
        "N_hi": q(N_boot, 0.975),
        "dn_dt_lo": q(dn_boot, 0.025),
        "dn_dt_med": q(dn_boot, 0.50),
        "dn_dt_hi": q(dn_boot, 0.975),
    })


def plot_overlay_ci(ci_f, ci_t, out_png_dn, out_png_N, title_suffix):
    # dn/dt
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(ci_f["bin_center_ms"], ci_f["dn_dt_med"], marker="o", label="FAPI")
    ax.fill_between(ci_f["bin_center_ms"], ci_f["dn_dt_lo"], ci_f["dn_dt_hi"], alpha=0.25)
    ax.plot(ci_t["bin_center_ms"], ci_t["dn_dt_med"], marker="o", label="FAPI-TEMPO")
    ax.fill_between(ci_t["bin_center_ms"], ci_t["dn_dt_lo"], ci_t["dn_dt_hi"], alpha=0.25)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("dn/dt (1/s)")
    ax.set_title("Stable nucleation rate (fixed bins)\n" + title_suffix)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png_dn, dpi=200)
    plt.close(fig)

    # N(t)
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(ci_f["bin_center_ms"], ci_f["N_med"], marker="o", label="FAPI")
    ax.fill_between(ci_f["bin_center_ms"], ci_f["N_lo"], ci_f["N_hi"], alpha=0.25)
    ax.plot(ci_t["bin_center_ms"], ci_t["N_med"], marker="o", label="FAPI-TEMPO")
    ax.fill_between(ci_t["bin_center_ms"], ci_t["N_lo"], ci_t["N_hi"], alpha=0.25)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("Cumulative stable nuclei N(t)")
    ax.set_title("Cumulative nucleation (fixed bins)\n" + title_suffix)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png_N, dpi=200)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fapi_tracks", required=True)
    ap.add_argument("--tempo_tracks", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--L", type=int, default=5)
    ap.add_argument("--amin_px", type=float, default=800)
    ap.add_argument("--rmin_px", type=float, default=3)
    ap.add_argument("--bin_ms", type=float, default=20)
    ap.add_argument("--bootstrap_B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--use_rmono_gate", action="store_true")
    ap.add_argument("--rmono_min", type=float, default=0.6)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("[OK] Reading:")
    print("  FAPI :", args.fapi_tracks)
    print("  TEMPO:", args.tempo_tracks)

    fapi_raw = pd.read_csv(args.fapi_tracks)
    tempo_raw = pd.read_csv(args.tempo_tracks)

    fapi = standardize_tracks(fapi_raw, "FAPI")
    tempo = standardize_tracks(tempo_raw, "FAPI_TEMPO")

    # stable nucleation events
    fapi_acc, fapi_rej = find_stable_nucleation_events(
        fapi, args.L, args.amin_px, args.rmin_px,
        use_rmono_gate=False, rmono_min=args.rmono_min
    )
    tempo_acc, tempo_rej = find_stable_nucleation_events(
        tempo, args.L, args.amin_px, args.rmin_px,
        use_rmono_gate=args.use_rmono_gate, rmono_min=args.rmono_min
    )

    # write events
    p_fapi_acc = os.path.join(args.out_dir, "nucleation_events_filtered_FAPI.csv")
    p_tempo_acc = os.path.join(args.out_dir, "nucleation_events_filtered_FAPI_TEMPO.csv")
    p_fapi_rej = os.path.join(args.out_dir, "nucleation_events_rejected_FAPI.csv")
    p_tempo_rej = os.path.join(args.out_dir, "nucleation_events_rejected_FAPI_TEMPO.csv")

    fapi_acc.to_csv(p_fapi_acc, index=False)
    tempo_acc.to_csv(p_tempo_acc, index=False)
    fapi_rej.to_csv(p_fapi_rej, index=False)
    tempo_rej.to_csv(p_tempo_rej, index=False)

    # fixed bin edges PER DATASET (this file’s outputs are per-dataset)
    edges_f = make_edges(fapi_acc["nuc_time_ms"].to_numpy(float), args.bin_ms)
    edges_t = make_edges(tempo_acc["nuc_time_ms"].to_numpy(float), args.bin_ms)

    # curves
    fapi_curve = curves_from_edges(fapi_acc["nuc_time_ms"].to_numpy(float), edges_f, args.bin_ms)
    tempo_curve = curves_from_edges(tempo_acc["nuc_time_ms"].to_numpy(float), edges_t, args.bin_ms)

    fapi_curve.to_csv(os.path.join(args.out_dir, "dn_dt_filtered_FAPI.csv"), index=False)
    tempo_curve.to_csv(os.path.join(args.out_dir, "dn_dt_filtered_FAPI_TEMPO.csv"), index=False)

    # N(t) tables (same info, separate files to match your old convention)
    fapi_curve[["bin_center_ms", "cum_n"]].to_csv(os.path.join(args.out_dir, "N_t_filtered_FAPI.csv"), index=False)
    tempo_curve[["bin_center_ms", "cum_n"]].to_csv(os.path.join(args.out_dir, "N_t_filtered_FAPI_TEMPO.csv"), index=False)

    # bootstrap CI with FIXED edges (per dataset)
    ci_f = bootstrap_CI_fixed(
        fapi_acc["nuc_time_ms"].to_numpy(float), edges_f, args.bin_ms, B=args.bootstrap_B, seed=args.seed
    )
    ci_t = bootstrap_CI_fixed(
        tempo_acc["nuc_time_ms"].to_numpy(float), edges_t, args.bin_ms, B=args.bootstrap_B, seed=args.seed + 1
    )

    # write combined CI file (long-form)
    ci_f2 = ci_f.copy()
    ci_f2["dataset"] = "FAPI"
    ci_t2 = ci_t.copy()
    ci_t2["dataset"] = "FAPI_TEMPO"
    ci_all = pd.concat([ci_f2, ci_t2], ignore_index=True)
    ci_all.to_csv(os.path.join(args.out_dir, "bootstrap_CI_FAPI_and_TEMPO.csv"), index=False)

    # quick overlay plots for sanity (note: bins differ between datasets here; your PAIR wrapper does shared bins)
    suffix = f"L={args.L}, Amin={args.amin_px:g}px, Rmin={args.rmin_px:g}px, bin={args.bin_ms:g}ms"
    if args.use_rmono_gate:
        suffix += f", Rmono≥{args.rmono_min:g}"
    out_dn = os.path.join(args.out_dir, "overlay_dn_dt_fixedbins_perdataset.png")
    out_N  = os.path.join(args.out_dir, "overlay_N_t_fixedbins_perdataset.png")
    plot_overlay_ci(ci_f, ci_t, out_dn, out_N, suffix)

    # methods note
    methods_txt = os.path.join(args.out_dir, "stable_nucleation_methods.txt")
    with open(methods_txt, "w", encoding="utf-8") as f:
        f.write("Stable nucleation definition\n")
        f.write("---------------------------\n")
        f.write(
            f"Per tracked grain, nucleation time was defined as the first observation that is followed by "
            f"≥ L={args.L} consecutive frames in which the object is continuously detected (no frame gaps) and "
            f"meets the minimum size thresholds: area ≥ {args.amin_px:g} px and equivalent radius ≥ {args.rmin_px:g} px.\n"
        )
        if args.use_rmono_gate:
            f.write(
                f"For FAPI-TEMPO, an additional monotonicity proxy gate was applied at nucleation: "
                f"R_mono ≥ {args.rmono_min:g} (dataset-provided monotonicity score), to reduce split/merge and ID-switch artifacts.\n"
            )
        f.write(
            f"N(t) and dn/dt were computed in {args.bin_ms:g} ms bins. Bootstrap confidence intervals "
            f"(2.5–97.5%) were obtained by resampling nucleation times with replacement (B={args.bootstrap_B}) "
            f"using fixed bin edges to ensure equal-length vectors across resamples.\n"
        )

    print("\n[OK] Wrote:")
    for p in [p_fapi_acc, p_tempo_acc, p_fapi_rej, p_tempo_rej,
              os.path.join(args.out_dir, "dn_dt_filtered_FAPI.csv"),
              os.path.join(args.out_dir, "dn_dt_filtered_FAPI_TEMPO.csv"),
              os.path.join(args.out_dir, "N_t_filtered_FAPI.csv"),
              os.path.join(args.out_dir, "N_t_filtered_FAPI_TEMPO.csv"),
              os.path.join(args.out_dir, "bootstrap_CI_FAPI_and_TEMPO.csv"),
              out_dn, out_N, methods_txt]:
        print(" ", p)


if __name__ == "__main__":
    main()
