import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Utilities: fixed shared bins (prevents bootstrap length mismatch)
# ----------------------------
def make_shared_bin_edges(all_event_times_ms, bin_ms, t_min=0.0):
    t = np.asarray(all_event_times_ms, dtype=float)
    if len(t) == 0 or np.all(np.isnan(t)):
        t_max = 0.0
    else:
        t_max = float(np.nanmax(t))
    # snap up to next bin edge so last bin is consistent
    t_max = bin_ms * np.ceil(t_max / bin_ms) if t_max > 0 else 0.0
    edges = np.arange(t_min, t_max + bin_ms, bin_ms, dtype=float)
    # ensure at least one bin
    if len(edges) < 2:
        edges = np.array([0.0, float(bin_ms)], dtype=float)
    return edges


def compute_N_and_dn_dt(event_times_ms, edges, bin_ms):
    counts, _ = np.histogram(event_times_ms, bins=edges)
    cumN = np.cumsum(counts)
    dn_dt = counts / (bin_ms / 1000.0)  # 1/s
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, cumN, dn_dt


def bootstrap_CI_fixed(event_times_ms, edges, bin_ms, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.asarray(event_times_ms, dtype=float)
    n = len(t)

    centers = 0.5 * (edges[:-1] + edges[1:])
    if n == 0:
        z = np.zeros_like(centers)
        return pd.DataFrame({
            "bin_center_ms": centers,
            "N_lo": z, "N_med": z, "N_hi": z,
            "dn_dt_lo": z, "dn_dt_med": z, "dn_dt_hi": z,
        })

    N_boot = []
    dn_boot = []
    for _ in range(B):
        samp = rng.choice(t, size=n, replace=True)
        _, cumN, dn_dt = compute_N_and_dn_dt(samp, edges, bin_ms)
        N_boot.append(cumN)
        dn_boot.append(dn_dt)

    N_boot = np.vstack(N_boot)
    dn_boot = np.vstack(dn_boot)

    def q(a, p):
        return np.quantile(a, p, axis=0)

    return pd.DataFrame({
        "bin_center_ms": centers,
        "N_lo": q(N_boot, 0.025),
        "N_med": q(N_boot, 0.50),
        "N_hi": q(N_boot, 0.975),
        "dn_dt_lo": q(dn_boot, 0.025),
        "dn_dt_med": q(dn_boot, 0.50),
        "dn_dt_hi": q(dn_boot, 0.975),
    })


# ----------------------------
# Load + normalize tracks formats (FAPI vs TEMPO)
# ----------------------------
def normalize_tracks(df, dataset_name):
    """
    Returns a dataframe with at least:
    track_id, time_ms, frame_idx, area_px, R_px, (optional) R_mono
    """
    out = df.copy()

    # time column harmonization
    if "time_ms" in out.columns:
        out["time_ms"] = out["time_ms"].astype(float)
    elif "t_ms" in out.columns:
        out["time_ms"] = out["t_ms"].astype(float)
    else:
        raise KeyError(f"[{dataset_name}] tracks.csv missing time column (time_ms or t_ms).")

    # frame index harmonization
    if "frame_idx" in out.columns:
        out["frame_idx"] = out["frame_idx"].astype(int)
    elif "frame" in out.columns:
        out["frame_idx"] = out["frame"].astype(int)
    elif "frame_id" in out.columns:
        # e.g. 'frame_00065_t130.00ms' -> 65
        out["frame_idx"] = out["frame_id"].astype(str).str.extract(r"frame_(\d+)").astype(int)
    else:
        raise KeyError(f"[{dataset_name}] tracks.csv missing frame identifier (frame_idx/frame/frame_id).")

    # radius + area
    if "R_px" not in out.columns:
        raise KeyError(f"[{dataset_name}] tracks.csv missing R_px.")
    out["R_px"] = out["R_px"].astype(float)

    if "area_px" not in out.columns:
        raise KeyError(f"[{dataset_name}] tracks.csv missing area_px.")
    out["area_px"] = out["area_px"].astype(float)

    # track id
    if "track_id" not in out.columns:
        raise KeyError(f"[{dataset_name}] tracks.csv missing track_id.")
    out["track_id"] = out["track_id"].astype(int)

    # optional monotonic score
    if "R_mono" in out.columns:
        out["R_mono"] = out["R_mono"].astype(float)

    return out


# ----------------------------
# Stable nucleation detection + gates
# ----------------------------
def find_stable_nucleation_events(tracks, L=5, amin_px=800, rmin_px=3,
                                 min_total_after=0,
                                 use_rmono_gate=False, rmono_min=0.0):
    """
    Defines nucleation time per track as the first frame where:
      - R_px >= rmin_px AND area_px >= amin_px
      - and the condition holds for >= L consecutive frames
    Additional gates:
      - min_total_after: require >= this many frames remaining after nucleation (persistence gate)
      - use_rmono_gate: if R_mono exists, require R_mono >= rmono_min
    Outputs:
      events_df with columns:
        track_id, nuc_frame_idx, nuc_time_ms, A_nuc_px, R_nuc_px, n_points_track, (optional) R_mono
      rejected_df with reason strings
    """
    events = []
    rejected = []

    gby = tracks.sort_values(["track_id", "frame_idx"]).groupby("track_id", sort=True)
    for tid, g in gby:
        g = g.reset_index(drop=True)

        # optional R_mono gate
        if use_rmono_gate:
            if "R_mono" not in g.columns:
                rejected.append({"track_id": tid, "reason": "no_R_mono_column"})
                continue
            if float(g["R_mono"].iloc[0]) < rmono_min:
                rejected.append({"track_id": tid, "reason": f"R_mono<{rmono_min}"})
                continue

        ok = (g["area_px"] >= amin_px) & (g["R_px"] >= rmin_px)

        # find first index i such that ok[i:i+L] all True
        found = False
        for i in range(0, len(g) - L + 1):
            if bool(ok.iloc[i:i+L].all()):
                # persistence after nucleation
                if min_total_after > 0 and (len(g) - i) < min_total_after:
                    rejected.append({"track_id": tid, "reason": f"too_short_after_nuc(<{min_total_after})"})
                    found = True  # found a candidate but rejected
                    break

                events.append({
                    "track_id": int(tid),
                    "nuc_frame_idx": int(g.loc[i, "frame_idx"]),
                    "nuc_time_ms": float(g.loc[i, "time_ms"]),
                    "A_nuc_px": float(g.loc[i, "area_px"]),
                    "R_nuc_px": float(g.loc[i, "R_px"]),
                    "n_points_track": int(len(g)),
                    "R_mono": float(g["R_mono"].iloc[0]) if "R_mono" in g.columns else np.nan,
                })
                found = True
                break

        if not found:
            rejected.append({"track_id": tid, "reason": f"no_{L}_consecutive_frames_meeting_size"})

    return pd.DataFrame(events), pd.DataFrame(rejected)


def apply_Rnuc_gate(events_df, R_nuc_max):
    if len(events_df) == 0:
        return events_df.copy()
    return events_df[events_df["R_nuc_px"] <= float(R_nuc_max)].copy()


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fapi_tracks", required=True)
    ap.add_argument("--tempo_tracks", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--L", type=int, default=5)
    ap.add_argument("--amin_px", type=float, default=800)
    ap.add_argument("--rmin_px", type=float, default=3)
    ap.add_argument("--bin_ms", type=float, default=20)

    ap.add_argument("--R_nuc_max", type=float, default=60.0, help="Reject events with R_nuc_px > this.")
    ap.add_argument("--min_total_after", type=int, default=0,
                    help="Require at least this many frames remaining after nucleation (persistence gate).")

    ap.add_argument("--use_rmono_gate", action="store_true")
    ap.add_argument("--rmono_min", type=float, default=0.6)

    ap.add_argument("--bootstrap_B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("[OK] Reading:")
    print("  FAPI :", args.fapi_tracks)
    print("  TEMPO:", args.tempo_tracks)

    fapi_raw = pd.read_csv(args.fapi_tracks)
    tempo_raw = pd.read_csv(args.tempo_tracks)

    fapi = normalize_tracks(fapi_raw, "FAPI")
    tempo = normalize_tracks(tempo_raw, "FAPI_TEMPO")

    # stable nucleation events
    fapi_events, fapi_rej = find_stable_nucleation_events(
        fapi, L=args.L, amin_px=args.amin_px, rmin_px=args.rmin_px,
        min_total_after=args.min_total_after,
        use_rmono_gate=False, rmono_min=args.rmono_min
    )
    tempo_events, tempo_rej = find_stable_nucleation_events(
        tempo, L=args.L, amin_px=args.amin_px, rmin_px=args.rmin_px,
        min_total_after=args.min_total_after,
        use_rmono_gate=args.use_rmono_gate, rmono_min=args.rmono_min
    )

    # R_nuc gate
    fapi_events_g = apply_Rnuc_gate(fapi_events, args.R_nuc_max)
    tempo_events_g = apply_Rnuc_gate(tempo_events, args.R_nuc_max)

    # shared bins across BOTH conditions (for fair overlay + fixed bootstrap)
    edges = make_shared_bin_edges(
        np.concatenate([fapi_events_g["nuc_time_ms"].values, tempo_events_g["nuc_time_ms"].values]),
        args.bin_ms
    )

    # point estimates
    cF, NF, rF = compute_N_and_dn_dt(fapi_events_g["nuc_time_ms"].values, edges, args.bin_ms)
    cT, NT, rT = compute_N_and_dn_dt(tempo_events_g["nuc_time_ms"].values, edges, args.bin_ms)

    # bootstrap CI
    ciF = bootstrap_CI_fixed(fapi_events_g["nuc_time_ms"].values, edges, args.bin_ms, B=args.bootstrap_B, seed=args.seed)
    ciT = bootstrap_CI_fixed(tempo_events_g["nuc_time_ms"].values, edges, args.bin_ms, B=args.bootstrap_B, seed=args.seed + 1)

    # write outputs
    f_fapi_events = os.path.join(args.out_dir, "nucleation_events_filtered_FAPI.csv")
    f_tempo_events = os.path.join(args.out_dir, "nucleation_events_filtered_FAPI_TEMPO.csv")
    f_fapi_rej = os.path.join(args.out_dir, "nucleation_events_rejected_FAPI.csv")
    f_tempo_rej = os.path.join(args.out_dir, "nucleation_events_rejected_FAPI_TEMPO.csv")

    fapi_events_g.to_csv(f_fapi_events, index=False)
    tempo_events_g.to_csv(f_tempo_events, index=False)
    fapi_rej.to_csv(f_fapi_rej, index=False)
    tempo_rej.to_csv(f_tempo_rej, index=False)

    pd.DataFrame({"bin_center_ms": cF, "cum_N": NF}).to_csv(os.path.join(args.out_dir, "N_t_filtered_FAPI.csv"), index=False)
    pd.DataFrame({"bin_center_ms": cT, "cum_N": NT}).to_csv(os.path.join(args.out_dir, "N_t_filtered_FAPI_TEMPO.csv"), index=False)
    pd.DataFrame({"bin_center_ms": cF, "dn_dt_per_s": rF}).to_csv(os.path.join(args.out_dir, "dn_dt_filtered_FAPI.csv"), index=False)
    pd.DataFrame({"bin_center_ms": cT, "dn_dt_per_s": rT}).to_csv(os.path.join(args.out_dir, "dn_dt_filtered_FAPI_TEMPO.csv"), index=False)

    ci_out = ciF.merge(
        ciT, on="bin_center_ms", suffixes=("_FAPI", "_TEMPO")
    )
    ci_out.to_csv(os.path.join(args.out_dir, "bootstrap_CI_FAPI_and_TEMPO.csv"), index=False)

    # plots: dn/dt with CI
    plt.figure()
    plt.plot(cF, rF, marker="o", label=f"FAPI (n={len(fapi_events_g)})")
    plt.plot(cT, rT, marker="o", label=f"FAPI-TEMPO (n={len(tempo_events_g)})")
    plt.fill_between(ciF["bin_center_ms"], ciF["dn_dt_lo"], ciF["dn_dt_hi"], alpha=0.2)
    plt.fill_between(ciT["bin_center_ms"], ciT["dn_dt_lo"], ciT["dn_dt_hi"], alpha=0.2)
    plt.xlabel("time (ms)")
    plt.ylabel("dn/dt (1/s)")
    ttl = f"Stable nucleation rate (bin={args.bin_ms:g} ms), gates: L>={args.L}, Amin>={args.amin_px:g}px, Rmin>={args.rmin_px:g}px, R_nuc<={args.R_nuc_max:g}px"
    if args.min_total_after > 0:
        ttl += f", persist>={args.min_total_after} frames"
    if args.use_rmono_gate:
        ttl += f", R_mono>={args.rmono_min:g}"
    plt.title(ttl)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "overlay_dn_dt_filtered.png"), dpi=200)

    # plots: N(t) with CI
    plt.figure()
    plt.plot(cF, NF, marker="o", label=f"FAPI (n={len(fapi_events_g)})")
    plt.plot(cT, NT, marker="o", label=f"FAPI-TEMPO (n={len(tempo_events_g)})")
    plt.fill_between(ciF["bin_center_ms"], ciF["N_lo"], ciF["N_hi"], alpha=0.2)
    plt.fill_between(ciT["bin_center_ms"], ciT["N_lo"], ciT["N_hi"], alpha=0.2)
    plt.xlabel("time (ms)")
    plt.ylabel("cumulative stable nuclei N(t)")
    plt.title("Cumulative stable nucleation N(t) (same gates)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "overlay_N_t_filtered.png"), dpi=200)

    # methods text
    methods = []
    methods.append("Stable nucleation definition and gating")
    methods.append(f"- Nucleation time per track: first frame where the grain satisfies area_px >= {args.amin_px:g} and R_px >= {args.rmin_px:g} for >= {args.L} consecutive frames.")
    if args.min_total_after > 0:
        methods.append(f"- Persistence gate: track must have >= {args.min_total_after} frames remaining after nucleation (reduces transient detections).")
    methods.append(f"- Birth-size gate: events with R_nuc_px > {args.R_nuc_max:g} were rejected (prevents late re-entries/splits being counted as new nuclei).")
    if args.use_rmono_gate:
        methods.append(f"- Monotonicity gate (TEMPO only if available): R_mono >= {args.rmono_min:g}.")
    methods.append(f"- N(t) and dn/dt computed in fixed {args.bin_ms:g} ms bins shared across both conditions.")
    methods.append(f"- Confidence bands: bootstrap resampling of nucleation event times (B={args.bootstrap_B}), reporting 2.5–97.5% quantiles.")

    with open(os.path.join(args.out_dir, "stable_nucleation_methods.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(methods) + "\n")

    print("\n[OK] Wrote outputs to:", args.out_dir)
    print("FAPI retained:", len(fapi_events_g), " / raw stable:", len(fapi_events))
    print("TEMPO retained:", len(tempo_events_g), " / raw stable:", len(tempo_events))


if __name__ == "__main__":
    main()
