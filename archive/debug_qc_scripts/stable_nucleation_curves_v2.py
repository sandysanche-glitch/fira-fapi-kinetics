import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Helpers: normalize input schema
# -----------------------------
def normalize_tracks(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Normalize tracks.csv from either pipeline to a common schema:
      - frame_idx (int)
      - time_ms (float)
      - frame_label (str)
      - track_id (int)
      - area_px (float)
      - R_px (float)
      - R_mono (float or NaN)
    """
    d = df.copy()

    # time column
    if "time_ms" in d.columns:
        d["time_ms"] = pd.to_numeric(d["time_ms"], errors="coerce")
    elif "t_ms" in d.columns:
        d["time_ms"] = pd.to_numeric(d["t_ms"], errors="coerce")
    else:
        raise KeyError(f"[{name}] tracks.csv missing time column (expected time_ms or t_ms).")

    # frame index column
    if "frame_idx" in d.columns:
        d["frame_idx"] = pd.to_numeric(d["frame_idx"], errors="coerce")
    elif "frame" in d.columns:
        d["frame_idx"] = pd.to_numeric(d["frame"], errors="coerce")
    else:
        # Some files only have frame_id like "frame_00065_t130.00ms"
        if "frame_id" in d.columns:
            s = d["frame_id"].astype(str)
            # extract the 5-digit index after frame_
            d["frame_idx"] = pd.to_numeric(s.str.extract(r"frame_(\d+)")[0], errors="coerce")
        else:
            raise KeyError(f"[{name}] tracks.csv missing frame info (expected frame_idx or frame or frame_id).")

    # frame label (for nicer output)
    if "frame_id" in d.columns:
        d["frame_label"] = d["frame_id"].astype(str)
    elif "frame" in d.columns:
        d["frame_label"] = d["frame"].astype(str)
    elif "frame_idx" in d.columns:
        d["frame_label"] = d["frame_idx"].astype(int).map(lambda x: f"frame_{x:05d}")
    else:
        d["frame_label"] = "frame_unknown"

    # required columns
    if "track_id" not in d.columns:
        raise KeyError(f"[{name}] tracks.csv missing track_id.")
    d["track_id"] = pd.to_numeric(d["track_id"], errors="coerce").astype("Int64")

    for col in ["area_px", "R_px"]:
        if col not in d.columns:
            raise KeyError(f"[{name}] tracks.csv missing {col}.")
        d[col] = pd.to_numeric(d[col], errors="coerce")

    # optional monotonic metric
    if "R_mono" in d.columns:
        d["R_mono"] = pd.to_numeric(d["R_mono"], errors="coerce")
    else:
        d["R_mono"] = np.nan

    # drop garbage rows
    d = d.dropna(subset=["frame_idx", "time_ms", "track_id", "area_px", "R_px"]).copy()
    d["frame_idx"] = d["frame_idx"].astype(int)
    d["track_id"] = d["track_id"].astype(int)

    # sort
    d = d.sort_values(["track_id", "frame_idx"]).reset_index(drop=True)
    return d


# -----------------------------
# Stable nucleation detection
# -----------------------------
def find_stable_nucleation_events(
    df: pd.DataFrame,
    L: int,
    amin_px: float,
    rmin_px: float,
    rmono_min: float,
    use_rmono_gate: bool,
    growth_W: int,
    growth_min_ratio: float,
    use_growth_gate: bool,
    dataset_name: str,
):
    """
    For each track:
      - Find first index i such that frames i..i+L-1 exist consecutively (no gaps)
      - And each satisfies area>=amin and R>=rmin
      - Optional gates:
          * R_mono >= rmono_min (if available)
          * R(t+i+W) / R(t+i) >= growth_min_ratio
    Returns (accepted_events_df, rejected_events_df)
    """
    accepted = []
    rejected = []

    for tid, g in df.groupby("track_id", sort=False):
        g = g.sort_values("frame_idx").reset_index(drop=True)
        frames = g["frame_idx"].to_numpy()
        area = g["area_px"].to_numpy()
        R = g["R_px"].to_numpy()
        t = g["time_ms"].to_numpy()
        flabel = g["frame_label"].astype(str).to_numpy()

        # quick size mask
        ok_size = (area >= amin_px) & (R >= rmin_px)

        found = False
        reason = "no_stable_window"

        # slide over possible nucleation points
        for i in range(0, len(g) - L + 1):
            # L consecutive frames?
            window_frames = frames[i : i + L]
            if not np.all(np.diff(window_frames) == 1):
                continue

            # size satisfied across window?
            if not np.all(ok_size[i : i + L]):
                continue

            # Optional monotonic gate
            if use_rmono_gate and ("R_mono" in g.columns) and np.isfinite(g.loc[i, "R_mono"]):
                if float(g.loc[i, "R_mono"]) < rmono_min:
                    reason = f"R_mono<{rmono_min}"
                    continue

            # Optional growth gate
            if use_growth_gate:
                j = i + growth_W
                if j < len(g):
                    ratio = (R[j] / R[i]) if R[i] > 0 else np.nan
                    if not np.isfinite(ratio) or ratio < growth_min_ratio:
                        reason = f"growth_ratio<{growth_min_ratio}_in_{growth_W}f"
                        continue
                else:
                    reason = f"no_growth_window_{growth_W}f"
                    continue

            # accept nucleation at i
            accepted.append(
                {
                    "dataset": dataset_name,
                    "track_id": tid,
                    "nuc_frame_idx": int(frames[i]),
                    "nuc_frame_label": flabel[i],
                    "nuc_time_ms": float(t[i]),
                    "R_nuc_px": float(R[i]),
                    "area_nuc_px": float(area[i]),
                    "R_mono": float(g.loc[i, "R_mono"]) if "R_mono" in g.columns else np.nan,
                    "n_points_track": int(len(g)),
                }
            )
            found = True
            break

        if not found:
            # pick first seen as debug info
            rejected.append(
                {
                    "dataset": dataset_name,
                    "track_id": tid,
                    "first_seen_frame_idx": int(frames[0]),
                    "first_seen_time_ms": float(t[0]),
                    "R_first_px": float(R[0]),
                    "area_first_px": float(area[0]),
                    "reason": reason,
                    "n_points_track": int(len(g)),
                }
            )

    return pd.DataFrame(accepted), pd.DataFrame(rejected)


# -----------------------------
# Curves + bootstrap CI
# -----------------------------
def make_curves(events: pd.DataFrame, bin_ms: float):
    if len(events) == 0:
        return (
            pd.DataFrame(columns=["bin_center_ms", "n_nucleated", "dn_dt_per_s", "cum_n"]),
            pd.DataFrame(columns=["time_ms", "cum_n"]),
        )

    times = np.sort(events["nuc_time_ms"].to_numpy())
    tmin = 0.0
    tmax = float(np.ceil(times.max() / bin_ms) * bin_ms)

    edges = np.arange(tmin, tmax + bin_ms, bin_ms)
    counts, _ = np.histogram(times, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    cum = np.cumsum(counts)
    dn_dt = counts / (bin_ms / 1000.0)  # per second

    dn_df = pd.DataFrame(
        {
            "bin_center_ms": centers,
            "n_nucleated": counts,
            "dn_dt_per_s": dn_dt,
            "cum_n": cum,
        }
    )
    Nt_df = pd.DataFrame({"time_ms": centers, "cum_n": cum})
    return dn_df, Nt_df


def bootstrap_CI(events: pd.DataFrame, bin_ms: float, B: int, seed: int):
    rng = np.random.default_rng(seed)
    if len(events) == 0:
        return pd.DataFrame(columns=["bin_center_ms", "N_lo", "N_hi", "dn_dt_lo", "dn_dt_hi"])

    base_dn, base_Nt = make_curves(events, bin_ms)
    centers = base_dn["bin_center_ms"].to_numpy()

    N_boot = []
    dn_boot = []

    times = events["nuc_time_ms"].to_numpy()

    for _ in range(B):
        sample = rng.choice(times, size=len(times), replace=True)
        tmp = pd.DataFrame({"nuc_time_ms": sample})
        dn_df, _ = make_curves(tmp, bin_ms)
        # align bins
        dn_boot.append(dn_df["dn_dt_per_s"].to_numpy())
        N_boot.append(dn_df["cum_n"].to_numpy())

    N_boot = np.vstack(N_boot)
    dn_boot = np.vstack(dn_boot)

    ci = pd.DataFrame(
        {
            "bin_center_ms": centers,
            "N_lo": np.percentile(N_boot, 2.5, axis=0),
            "N_hi": np.percentile(N_boot, 97.5, axis=0),
            "dn_dt_lo": np.percentile(dn_boot, 2.5, axis=0),
            "dn_dt_hi": np.percentile(dn_boot, 97.5, axis=0),
        }
    )
    return ci


def plot_overlay_with_CI(out_png, fapi_dn, tempo_dn, fapi_ci, tempo_ci, title):
    plt.figure()
    plt.plot(fapi_dn["bin_center_ms"], fapi_dn["dn_dt_per_s"], marker="o", label="FAPI")
    plt.plot(tempo_dn["bin_center_ms"], tempo_dn["dn_dt_per_s"], marker="o", label="FAPI-TEMPO")

    if len(fapi_ci):
        plt.fill_between(fapi_ci["bin_center_ms"], fapi_ci["dn_dt_lo"], fapi_ci["dn_dt_hi"], alpha=0.2)
    if len(tempo_ci):
        plt.fill_between(tempo_ci["bin_center_ms"], tempo_ci["dn_dt_lo"], tempo_ci["dn_dt_hi"], alpha=0.2)

    plt.xlabel("time (ms)")
    plt.ylabel("dn/dt (1/s)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_overlay_N_with_CI(out_png, fapi_Nt, tempo_Nt, fapi_ci, tempo_ci, title):
    plt.figure()
    plt.plot(fapi_Nt["time_ms"], fapi_Nt["cum_n"], marker="o", label="FAPI")
    plt.plot(tempo_Nt["time_ms"], tempo_Nt["cum_n"], marker="o", label="FAPI-TEMPO")

    if len(fapi_ci):
        plt.fill_between(fapi_ci["bin_center_ms"], fapi_ci["N_lo"], fapi_ci["N_hi"], alpha=0.2)
    if len(tempo_ci):
        plt.fill_between(tempo_ci["bin_center_ms"], tempo_ci["N_lo"], tempo_ci["N_hi"], alpha=0.2)

    plt.xlabel("time (ms)")
    plt.ylabel("cumulative nucleated grains")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


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
    ap.add_argument("--rmin_px", type=float, default=3.0)
    ap.add_argument("--bin_ms", type=float, default=20.0)

    ap.add_argument("--bootstrap_B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)

    # Optional physics gates
    ap.add_argument("--use_rmono_gate", action="store_true")
    ap.add_argument("--rmono_min", type=float, default=0.6)

    ap.add_argument("--use_growth_gate", action="store_true")
    ap.add_argument("--growth_W", type=int, default=10)          # frames after nuc
    ap.add_argument("--growth_min_ratio", type=float, default=1.2)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("[OK] Reading:")
    print("  FAPI :", args.fapi_tracks)
    print("  TEMPO:", args.tempo_tracks)

    fapi_raw = pd.read_csv(args.fapi_tracks)
    tempo_raw = pd.read_csv(args.tempo_tracks)

    fapi = normalize_tracks(fapi_raw, "FAPI")
    tempo = normalize_tracks(tempo_raw, "FAPI_TEMPO")

    # detect events
    fapi_acc, fapi_rej = find_stable_nucleation_events(
        fapi, args.L, args.amin_px, args.rmin_px,
        args.rmono_min, args.use_rmono_gate,
        args.growth_W, args.growth_min_ratio, args.use_growth_gate,
        "FAPI"
    )
    tempo_acc, tempo_rej = find_stable_nucleation_events(
        tempo, args.L, args.amin_px, args.rmin_px,
        args.rmono_min, args.use_rmono_gate,
        args.growth_W, args.growth_min_ratio, args.use_growth_gate,
        "FAPI_TEMPO"
    )

    # write events
    fapi_acc.to_csv(os.path.join(args.out_dir, "nucleation_events_filtered_FAPI.csv"), index=False)
    tempo_acc.to_csv(os.path.join(args.out_dir, "nucleation_events_filtered_FAPI_TEMPO.csv"), index=False)
    fapi_rej.to_csv(os.path.join(args.out_dir, "nucleation_events_rejected_FAPI.csv"), index=False)
    tempo_rej.to_csv(os.path.join(args.out_dir, "nucleation_events_rejected_FAPI_TEMPO.csv"), index=False)

    # curves
    fapi_dn, fapi_Nt = make_curves(fapi_acc, args.bin_ms)
    tempo_dn, tempo_Nt = make_curves(tempo_acc, args.bin_ms)

    fapi_dn.to_csv(os.path.join(args.out_dir, "dn_dt_filtered_FAPI.csv"), index=False)
    tempo_dn.to_csv(os.path.join(args.out_dir, "dn_dt_filtered_FAPI_TEMPO.csv"), index=False)
    fapi_Nt.to_csv(os.path.join(args.out_dir, "N_t_filtered_FAPI.csv"), index=False)
    tempo_Nt.to_csv(os.path.join(args.out_dir, "N_t_filtered_FAPI_TEMPO.csv"), index=False)

    # bootstrap CI
    fapi_ci = bootstrap_CI(fapi_acc, args.bin_ms, args.bootstrap_B, args.seed)
    tempo_ci = bootstrap_CI(tempo_acc, args.bin_ms, args.bootstrap_B, args.seed + 1)
    ci_out = pd.merge(
        fapi_ci.add_prefix("FAPI_"),
        tempo_ci.add_prefix("TEMPO_"),
        left_on="FAPI_bin_center_ms",
        right_on="TEMPO_bin_center_ms",
        how="outer"
    )
    ci_out.to_csv(os.path.join(args.out_dir, "bootstrap_CI_FAPI_and_TEMPO.csv"), index=False)

    # plots
    plot_overlay_with_CI(
        os.path.join(args.out_dir, "overlay_dn_dt_filtered_with_CI.png"),
        fapi_dn, tempo_dn,
        fapi_ci.rename(columns={"bin_center_ms":"bin_center_ms"}),
        tempo_ci.rename(columns={"bin_center_ms":"bin_center_ms"}),
        f"Stable nucleation rate (bin={args.bin_ms:g} ms, L={args.L}, Amin={args.amin_px:g}px, Rmin={args.rmin_px:g}px)"
    )
    plot_overlay_N_with_CI(
        os.path.join(args.out_dir, "overlay_N_t_filtered_with_CI.png"),
        fapi_Nt, tempo_Nt,
        fapi_ci.rename(columns={"bin_center_ms":"bin_center_ms"}),
        tempo_ci.rename(columns={"bin_center_ms":"bin_center_ms"}),
        f"Stable cumulative nucleation N(t) (bin={args.bin_ms:g} ms, L={args.L})"
    )

    # methods text
    methods = []
    methods.append("Stable nucleation definition (per track):")
    methods.append(f"- Nucleation time is defined as the first detection followed by ≥L consecutive frames with no frame gaps (L={args.L}).")
    methods.append(f"- Size gate applied over the L-frame window: area ≥ {args.amin_px:g} px and radius ≥ {args.rmin_px:g} px.")
    if args.use_rmono_gate:
        methods.append(f"- Additional physics gate: monotonicity metric R_mono ≥ {args.rmono_min:g} (when available).")
    if args.use_growth_gate:
        methods.append(f"- Additional physics gate: net growth within W={args.growth_W} frames: R(t+W)/R(t) ≥ {args.growth_min_ratio:g}.")
    methods.append(f"- N(t) and dn/dt computed using {args.bin_ms:g} ms bins.")
    methods.append(f"- Uncertainty estimated via bootstrap resampling of nucleation events (B={args.bootstrap_B}).")

    with open(os.path.join(args.out_dir, "stable_nucleation_methods.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(methods) + "\n")

    print("\n=== Summary ===")
    def summary(name, ev):
        if len(ev)==0:
            print(f"{name}: n=0")
            return
        print(f"{name}: n={len(ev)} | nuc_time_ms median={np.median(ev.nuc_time_ms):.2f} | R_nuc_px median={np.median(ev.R_nuc_px):.2f}")
    summary("FAPI", fapi_acc)
    summary("FAPI-TEMPO", tempo_acc)

    print("\n[OK] Wrote outputs to:")
    print(" ", args.out_dir)


if __name__ == "__main__":
    main()
