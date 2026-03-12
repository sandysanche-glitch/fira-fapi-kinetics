import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Helpers: column normalization
# -----------------------------
_FRAME_ID_RE = re.compile(r"frame_(\d+)_t")

def _infer_frame_col(df: pd.DataFrame) -> str:
    if "frame_idx" in df.columns:
        return "frame_idx"
    if "frame" in df.columns:
        return "frame"
    if "frame_id" in df.columns:
        return "frame_id"
    raise KeyError("tracks.csv must contain one of: frame_idx, frame, frame_id")

def _infer_time_col(df: pd.DataFrame) -> str:
    if "t_ms" in df.columns:
        return "t_ms"
    if "time_ms" in df.columns:
        return "time_ms"
    raise KeyError("tracks.csv must contain one of: t_ms, time_ms")

def _frame_to_int(x):
    # already numeric?
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float) and np.isfinite(x):
        return int(x)

    s = str(x)
    m = _FRAME_ID_RE.search(s)
    if m:
        return int(m.group(1))
    # fallback: try plain int
    return int(s)

def normalize_tracks(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    df = df.copy()

    frame_col = _infer_frame_col(df)
    time_col = _infer_time_col(df)

    # normalize to standard column names
    df["_frame"] = df[frame_col].map(_frame_to_int)
    df["_t_ms"] = pd.to_numeric(df[time_col], errors="coerce")

    # required columns for gates
    for col in ["track_id ցույց", "track_id"]:
        pass

    if "track_id" not in df.columns:
        raise KeyError(f"{dataset_name}: tracks.csv must contain track_id")

    if "R_px" not in df.columns:
        raise KeyError(f"{dataset_name}: tracks.csv must contain R_px")
    if "area_px" not in df.columns:
        raise KeyError(f"{dataset_name}: tracks.csv must contain area_px")

    # tidy types
    df["track_id"] = pd.to_numeric(df["track_id"], errors="coerce").astype("Int64")
    df["R_px"] = pd.to_numeric(df["R_px"], errors="coerce")
    df["area_px"] = pd.to_numeric(df["area_px"], errors="coerce")

    df = df.dropna(subset=["track_id", "_frame", "_t_ms", "R_px", "area_px"]).copy()
    df["track_id"] = df["track_id"].astype(int)
    df["_frame"] = df["_frame"].astype(int)

    # sort
    df = df.sort_values(["track_id", "_frame"]).reset_index(drop=True)
    return df

# ---------------------------------
# Stable nucleation event extraction
# ---------------------------------
def compute_rmono(R: np.ndarray) -> float:
    """Monotonicity score in [0..1]: fraction of non-negative steps."""
    if len(R) < 2:
        return np.nan
    d = np.diff(R)
    return float(np.mean(d >= 0))

def find_stable_nucleation_events(
    df: pd.DataFrame,
    L: int,
    amin_px: float,
    rmin_px: float,
    use_rmono_gate: bool = False,
    rmono_min: float = 0.0,
    rnuc_max: float | None = None,
    growth_K: int | None = None,
    growth_dR_min: float = 0.0,
):
    """
    Nucleation time for a track = earliest index i such that:
      - frames i..i+L-1 are consecutive (no gaps)
      - each detection satisfies area >= amin_px AND R >= rmin_px
    Optional:
      - rnuc_max: require R_nuc <= rnuc_max
      - rmono gate: require monotonicity >= rmono_min (computed over full track after nuc)
      - growth gate: require R(i+growth_K) - R(i) >= growth_dR_min (if available)
    """
    accepted = []
    rejected = []

    for tid, g in df.groupby("track_id", sort=False):
        frames = g["_frame"].to_numpy()
        t_ms = g["_t_ms"].to_numpy()
        R = g["R_px"].to_numpy()
        A = g["area_px"].to_numpy()

        n = len(g)
        if n < L:
            rejected.append((tid, np.nan, "too_short"))
            continue

        idx_nuc = None
        for i in range(0, n - L + 1):
            # consecutive frames?
            if not np.all(np.diff(frames[i:i+L]) == 1):
                continue
            # size gate on the L-window
            if np.any(A[i:i+L] < amin_px) or np.any(R[i:i+L] < rmin_px):
                continue
            idx_nuc = i
            break

        if idx_nuc is None:
            rejected.append((tid, np.nan, "no_stable_window"))
            continue

        nuc_t = float(t_ms[idx_nuc])
        nuc_frame = int(frames[idx_nuc])
        R_nuc = float(R[idx_nuc])
        A_nuc = float(A[idx_nuc])

        # rnuc gate
        if rnuc_max is not None and R_nuc > rnuc_max:
            rejected.append((tid, nuc_t, f"rnuc_gt_{rnuc_max}"))
            continue

        # growth-after-birth gate
        if growth_K is not None and growth_K > 0:
            j = idx_nuc + growth_K
            if j < n:
                dR = float(R[j] - R[idx_nuc])
                if dR < growth_dR_min:
                    rejected.append((tid, nuc_t, f"growth_dR<{growth_dR_min}_over_{growth_K}"))
                    continue
            # if j out of range, we don't reject; track may end early

        # rmono gate
        rmono = compute_rmono(R[idx_nuc:])  # after nucleation
        if use_rmono_gate and np.isfinite(rmono) and rmono < rmono_min:
            rejected.append((tid, nuc_t, f"rmono_lt_{rmono_min}"))
            continue

        accepted.append({
            "track_id": tid,
            "nuc_time_ms": nuc_t,
            "nuc_frame": nuc_frame,
            "R_nuc_px": R_nuc,
            "A_nuc_px": A_nuc,
            "n_points": int(n),
            "R_max_px": float(np.max(R)),
            "A_max_px": float(np.max(A)),
            "R_mono": rmono,
        })

    acc = pd.DataFrame(accepted).sort_values("nuc_time_ms").reset_index(drop=True)
    rej = pd.DataFrame(rejected, columns=["track_id", "nuc_time_ms", "reject_reason"])
    return acc, rej

# -----------------------------
# Shared-bin N(t), dn/dt + CI
# -----------------------------
def make_shared_bin_edges(all_event_times_ms, bin_ms, t_min=0.0):
    t = np.asarray(all_event_times_ms, dtype=float)
    if len(t) == 0:
        t_max = 0.0
    else:
        t_max = float(np.nanmax(t))
    t_max = bin_ms * np.ceil(t_max / bin_ms)  # snap up
    edges = np.arange(t_min, t_max + bin_ms, bin_ms, dtype=float)
    return edges

def compute_N_and_dn_dt(event_times_ms, edges, bin_ms):
    counts, _ = np.histogram(event_times_ms, bins=edges)
    cumN = np.cumsum(counts)
    dn_dt = counts / (bin_ms / 1000.0)  # 1/s
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, cumN, dn_dt

def bootstrap_CI_fixed(times_ms, edges, bin_ms, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.asarray(times_ms, dtype=float)
    n = len(t)
    centers = 0.5 * (edges[:-1] + edges[1:])

    if n == 0:
        z = np.zeros_like(centers, dtype=float)
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

# -----------------------------
# Plotting
# -----------------------------
def plot_overlay_dn_dt(ci_fapi, ci_tempo, out_png, title):
    plt.figure()
    plt.plot(ci_fapi["bin_center_ms"], ci_fapi["dn_dt_med"], marker="o", label="FAPI")
    plt.fill_between(ci_fapi["bin_center_ms"], ci_fapi["dn_dt_lo"], ci_fapi["dn_dt_hi"], alpha=0.2)

    plt.plot(ci_tempo["bin_center_ms"], ci_tempo["dn_dt_med"], marker="o", label="FAPI-TEMPO")
    plt.fill_between(ci_tempo["bin_center_ms"], ci_tempo["dn_dt_lo"], ci_tempo["dn_dt_hi"], alpha=0.2)

    plt.xlabel("time (ms)")
    plt.ylabel("dn/dt (1/s)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_overlay_N(ci_fapi, ci_tempo, out_png, title):
    plt.figure()
    plt.plot(ci_fapi["bin_center_ms"], ci_fapi["N_med"], marker="o", label="FAPI")
    plt.fill_between(ci_fapi["bin_center_ms"], ci_fapi["N_lo"], ci_fapi["N_hi"], alpha=0.2)

    plt.plot(ci_tempo["bin_center_ms"], ci_tempo["N_med"], marker="o", label="FAPI-TEMPO")
    plt.fill_between(ci_tempo["bin_center_ms"], ci_tempo["N_lo"], ci_tempo["N_hi"], alpha=0.2)

    plt.xlabel("time (ms)")
    plt.ylabel("cumulative stable nuclei N(t)")
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
    ap.add_argument("--amin_px", type=float, default=800.0)
    ap.add_argument("--rmin_px", type=float, default=3.0)
    ap.add_argument("--bin_ms", type=float, default=20.0)
    ap.add_argument("--bootstrap_B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--use_rmono_gate", action="store_true")
    ap.add_argument("--rmono_min", type=float, default=0.6)

    ap.add_argument("--rnuc_max", type=float, default=None)

    ap.add_argument("--growth_K", type=int, default=None,
                    help="Optional: frames after nucleation to check growth (e.g. 5)")
    ap.add_argument("--growth_dR_min", type=float, default=0.0,
                    help="Optional: require R(t+K)-R(t) >= this")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("[OK] Reading:")
    print("  FAPI :", args.fapi_tracks)
    print("  TEMPO:", args.tempo_tracks)

    fapi_raw = pd.read_csv(args.fapi_tracks)
    tempo_raw = pd.read_csv(args.tempo_tracks)

    fapi = normalize_tracks(fapi_raw, "FAPI")
    tempo = normalize_tracks(tempo_raw, "FAPI_TEMPO")

    fapi_acc, fapi_rej = find_stable_nucleation_events(
        fapi, args.L, args.amin_px, args.rmin_px,
        use_rmono_gate=False,  # FAPI has no R_mono column; we compute it anyway but don't require it by default
        rmono_min=args.rmono_min,
        rnuc_max=args.rnuc_max,
        growth_K=args.growth_K,
        growth_dR_min=args.growth_dR_min,
    )

    tempo_acc, tempo_rej = find_stable_nucleation_events(
        tempo, args.L, args.amin_px, args.rmin_px,
        use_rmono_gate=args.use_rmono_gate,
        rmono_min=args.rmono_min,
        rnuc_max=args.rnuc_max,
        growth_K=args.growth_K,
        growth_dR_min=args.growth_dR_min,
    )

    # Write events
    f_fapi = os.path.join(args.out_dir, "nucleation_events_filtered_FAPI.csv")
    f_tempo = os.path.join(args.out_dir, "nucleation_events_filtered_FAPI_TEMPO.csv")
    r_fapi = os.path.join(args.out_dir, "nucleation_events_rejected_FAPI.csv")
    r_tempo = os.path.join(args.out_dir, "nucleation_events_rejected_FAPI_TEMPO.csv")
    fapi_acc.to_csv(f_fapi, index=False)
    tempo_acc.to_csv(f_tempo, index=False)
    fapi_rej.to_csv(r_fapi, index=False)
    tempo_rej.to_csv(r_tempo, index=False)

    # Shared bin edges from BOTH datasets
    all_times = np.concatenate([
        fapi_acc["nuc_time_ms"].to_numpy(dtype=float) if len(fapi_acc) else np.array([], dtype=float),
        tempo_acc["nuc_time_ms"].to_numpy(dtype=float) if len(tempo_acc) else np.array([], dtype=float),
    ])
    edges = make_shared_bin_edges(all_times, args.bin_ms, t_min=0.0)

    # Deterministic curves
    cF, NF, dF = compute_N_and_dn_dt(fapi_acc["nuc_time_ms"].to_numpy(dtype=float), edges, args.bin_ms)
    cT, NT, dT = compute_N_and_dn_dt(tempo_acc["nuc_time_ms"].to_numpy(dtype=float), edges, args.bin_ms)

    nd_fapi = pd.DataFrame({"bin_center_ms": cF, "cum_N": NF, "dn_dt_per_s": dF})
    nd_tempo = pd.DataFrame({"bin_center_ms": cT, "cum_N": NT, "dn_dt_per_s": dT})

    nd_fapi.to_csv(os.path.join(args.out_dir, "N_and_dn_dt_FAPI_sharedbins.csv"), index=False)
    nd_tempo.to_csv(os.path.join(args.out_dir, "N_and_dn_dt_FAPI_TEMPO_sharedbins.csv"), index=False)

    # Bootstrap CIs (fixed edges!)
    ci_fapi = bootstrap_CI_fixed(fapi_acc["nuc_time_ms"].to_numpy(dtype=float), edges, args.bin_ms, args.bootstrap_B, args.seed)
    ci_tempo = bootstrap_CI_fixed(tempo_acc["nuc_time_ms"].to_numpy(dtype=float), edges, args.bin_ms, args.bootstrap_B, args.seed + 1)

    ci_fapi.to_csv(os.path.join(args.out_dir, "bootstrap_CI_FAPI_sharedbins.csv"), index=False)
    ci_tempo.to_csv(os.path.join(args.out_dir, "bootstrap_CI_FAPI_TEMPO_sharedbins.csv"), index=False)

    # Plots
    gate_txt = f"L={args.L}, Amin={args.amin_px}px, Rmin={args.rmin_px}px"
    if args.rnuc_max is not None:
        gate_txt += f", R_nuc<={args.rnuc_max}px"
    if args.use_rmono_gate:
        gate_txt += f", R_mono>={args.rmono_min}"
    if args.growth_K is not None:
        gate_txt += f", dR(K={args.growth_K})>={args.growth_dR_min}px"

    out_dn = os.path.join(args.out_dir, "overlay_dn_dt_sharedbins.png")
    out_N = os.path.join(args.out_dir, "overlay_Nt_sharedbins.png")
    plot_overlay_dn_dt(ci_fapi, ci_tempo, out_dn, f"Stable nucleation rate (bin={args.bin_ms} ms) | {gate_txt}")
    plot_overlay_N(ci_fapi, ci_tempo, out_N, f"Cumulative stable nuclei N(t) (bin={args.bin_ms} ms) | {gate_txt}")

    # Methods text
    methods = os.path.join(args.out_dir, "stable_nucleation_methods.txt")
    with open(methods, "w", encoding="utf-8") as f:
        f.write(
            "Stable nucleation definition (per track):\n"
            f"- Nucleation time is the earliest frame i such that detections exist for >= L={args.L} consecutive frames (no gaps),\n"
            f"  and each detection in that L-window satisfies area >= Amin={args.amin_px} px and radius >= Rmin={args.rmin_px} px.\n"
            "- This defines a 'detectable stable nucleus' and suppresses transient segmentation noise.\n\n"
            "Additional physics/quality gates:\n"
        )
        if args.rnuc_max is not None:
            f.write(f"- Birth-size gate: require R_nuc <= {args.rnuc_max} px (reject already-large-at-birth detections).\n")
        if args.use_rmono_gate:
            f.write(f"- Monotonicity gate: require R_mono >= {args.rmono_min}, where R_mono is the fraction of non-decreasing R steps after nucleation.\n")
        if args.growth_K is not None:
            f.write(f"- Growth-after-birth gate: require R(t+K)-R(t) >= {args.growth_dR_min} px with K={args.growth_K} frames when available.\n")
        f.write(
            "\nBinning + uncertainty:\n"
            f"- Event times from both datasets are pooled to define shared bin edges with width {args.bin_ms} ms.\n"
            "- dn/dt is computed as counts per bin divided by bin duration (s).\n"
            f"- Bootstrap CIs: B={args.bootstrap_B} resamples of nucleation times with replacement; all bootstrap curves are evaluated on the same shared bin edges.\n"
        )

    print("\n[OK] Wrote:")
    print("  ", f_fapi)
    print("  ", f_tempo)
    print("  ", r_fapi)
    print("  ", r_tempo)
    print("  ", out_dn)
    print("  ", out_N)
    print("  ", methods)

    def _summ(ev, name):
        if len(ev) == 0:
            print(f"{name}: n=0")
            return
        late = (ev["nuc_time_ms"] >= 200).sum()
        print(f"{name}: n={len(ev)} | nuc_time max={ev['nuc_time_ms'].max():.1f} ms | late>=200ms: {late}")

    print("\n=== Summary ===")
    _summ(fapi_acc, "FAPI")
    _summ(tempo_acc, "FAPI-TEMPO")

if __name__ == "__main__":
    main()
