# run_stable_nucleation_pair_sharedbins.py
# Run the SAME rebuild+filter (stable nucleation + overlap gate) for FAPI and FAPI-TEMPO,
# then compute shared-bin N(t) and dn/dt with bootstrap CI and save overlay plots.

import os
import argparse
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_shared_bin_edges(all_event_times_ms, bin_ms, t_min=0.0):
    t = np.asarray(all_event_times_ms, dtype=float)
    tmax = float(np.nanmax(t)) if len(t) else 0.0
    tmax = bin_ms * np.ceil(tmax / bin_ms)
    edges = np.arange(t_min, tmax + bin_ms, bin_ms, dtype=float)
    if len(edges) < 2:
        edges = np.array([0.0, bin_ms], dtype=float)
    return edges


def compute_N_and_dn_dt(event_times_ms, edges, bin_ms):
    counts, _ = np.histogram(event_times_ms, bins=edges)
    cumN = np.cumsum(counts)
    dn_dt = counts / (bin_ms / 1000.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, cumN, dn_dt


def bootstrap_CI_fixed(events_df, time_col, edges, bin_ms, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    t = events_df[time_col].to_numpy(dtype=float)
    n = len(t)

    centers = 0.5 * (edges[:-1] + edges[1:])
    if n == 0:
        z = np.zeros_like(centers)
        return pd.DataFrame(
            {"bin_center_ms": centers,
             "N_lo": z, "N_med": z, "N_hi": z,
             "dn_dt_lo": z, "dn_dt_med": z, "dn_dt_hi": z}
        )

    N_boot = np.zeros((B, len(centers)), dtype=float)
    dn_boot = np.zeros((B, len(centers)), dtype=float)

    for b in range(B):
        samp = rng.choice(t, size=n, replace=True)
        _, cumN, dn_dt = compute_N_and_dn_dt(samp, edges, bin_ms)
        N_boot[b, :] = cumN
        dn_boot[b, :] = dn_dt

    def q(a, p): return np.quantile(a, p, axis=0)

    return pd.DataFrame(
        {"bin_center_ms": centers,
         "N_lo": q(N_boot, 0.025), "N_med": q(N_boot, 0.50), "N_hi": q(N_boot, 0.975),
         "dn_dt_lo": q(dn_boot, 0.025), "dn_dt_med": q(dn_boot, 0.50), "dn_dt_hi": q(dn_boot, 0.975)}
    )


def run_one(python_exe, rebuild_script, tracks_csv, json_dir, out_dir, args):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        python_exe, rebuild_script,
        "--tracks_csv", tracks_csv,
        "--json_dir", json_dir,
        "--out_dir", out_dir,
        "--L", str(args.L),
        "--amin_px", str(args.amin_px),
        "--rmin_px", str(args.rmin_px),
        "--overlap_prev_max", str(args.overlap_prev_max),
    ]
    if args.use_rmono_gate:
        cmd += ["--use_rmono_gate", "--rmono_min", str(args.rmono_min)]
    if args.rnuc_max is not None:
        cmd += ["--rnuc_max", str(args.rnuc_max)]

    print("[RUN]", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
        raise RuntimeError(f"Rebuild failed (exit {r.returncode}).")

    return os.path.join(out_dir, "nucleation_events_filtered.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild_script", required=True, help="stable_nucleation_rebuild_from_json.py")
    ap.add_argument("--python_exe", default=None, help="path to python, default uses current interpreter")

    ap.add_argument("--fapi_tracks", required=True)
    ap.add_argument("--fapi_json_dir", required=True)

    ap.add_argument("--tempo_tracks", required=True)
    ap.add_argument("--tempo_json_dir", required=True)

    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--L", type=int, default=5)
    ap.add_argument("--amin_px", type=float, default=800.0)
    ap.add_argument("--rmin_px", type=float, default=3.0)

    ap.add_argument("--use_rmono_gate", action="store_true")
    ap.add_argument("--rmono_min", type=float, default=0.6)

    ap.add_argument("--rnuc_max", type=float, default=None)
    ap.add_argument("--overlap_prev_max", type=float, default=0.5)

    ap.add_argument("--bin_ms", type=float, default=20.0)
    ap.add_argument("--bootstrap_B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    py = args.python_exe or os.environ.get("PYTHON", None) or os.sys.executable
    os.makedirs(args.out_dir, exist_ok=True)

    fapi_out = os.path.join(args.out_dir, "FAPI_rebuild")
    tempo_out = os.path.join(args.out_dir, "FAPI_TEMPO_rebuild")

    fapi_events_csv = run_one(py, args.rebuild_script, args.fapi_tracks, args.fapi_json_dir, fapi_out, args)
    tempo_events_csv = run_one(py, args.rebuild_script, args.tempo_tracks, args.tempo_json_dir, tempo_out, args)

    fapi_events = pd.read_csv(fapi_events_csv)
    tempo_events = pd.read_csv(tempo_events_csv)

    # Shared bins
    edges = make_shared_bin_edges(
        np.concatenate([fapi_events["nuc_time_ms"].values, tempo_events["nuc_time_ms"].values])
        if (len(fapi_events) + len(tempo_events)) else np.array([0.0]),
        args.bin_ms
    )

    # Save N(t) and dn/dt
    cF, NF, rF = compute_N_and_dn_dt(fapi_events["nuc_time_ms"].values, edges, args.bin_ms)
    cT, NT, rT = compute_N_and_dn_dt(tempo_events["nuc_time_ms"].values, edges, args.bin_ms)

    out_F = pd.DataFrame({"bin_center_ms": cF, "N": NF, "dn_dt_per_s": rF})
    out_T = pd.DataFrame({"bin_center_ms": cT, "N": NT, "dn_dt_per_s": rT})

    out_F_csv = os.path.join(args.out_dir, "N_and_dn_dt_FAPI_sharedbins.csv")
    out_T_csv = os.path.join(args.out_dir, "N_and_dn_dt_FAPI_TEMPO_sharedbins.csv")
    out_F.to_csv(out_F_csv, index=False)
    out_T.to_csv(out_T_csv, index=False)

    # Bootstrap CI
    ci_F = bootstrap_CI_fixed(fapi_events, "nuc_time_ms", edges, args.bin_ms, B=args.bootstrap_B, seed=args.seed)
    ci_T = bootstrap_CI_fixed(tempo_events, "nuc_time_ms", edges, args.bin_ms, B=args.bootstrap_B, seed=args.seed + 1)

    ciF_csv = os.path.join(args.out_dir, "bootstrap_CI_FAPI_sharedbins.csv")
    ciT_csv = os.path.join(args.out_dir, "bootstrap_CI_FAPI_TEMPO_sharedbins.csv")
    ci_F.to_csv(ciF_csv, index=False)
    ci_T.to_csv(ciT_csv, index=False)

    # Overlay dn/dt with CI shading
    png_dn = os.path.join(args.out_dir, "overlay_dn_dt_sharedbins.png")
    plt.figure()
    plt.plot(out_F["bin_center_ms"], out_F["dn_dt_per_s"], marker="o", label="FAPI")
    plt.plot(out_T["bin_center_ms"], out_T["dn_dt_per_s"], marker="o", label="FAPI-TEMPO")
    plt.fill_between(ci_F["bin_center_ms"], ci_F["dn_dt_lo"], ci_F["dn_dt_hi"], alpha=0.2)
    plt.fill_between(ci_T["bin_center_ms"], ci_T["dn_dt_lo"], ci_T["dn_dt_hi"], alpha=0.2)
    plt.xlabel("time (ms)")
    plt.ylabel("dn/dt (1/s)")
    plt.title(
        f"Stable nucleation rate (bin={args.bin_ms} ms) | "
        f"L={args.L}, Amin={args.amin_px}px, Rmin={args.rmin_px}px, "
        f"R_nuc<={args.rnuc_max if args.rnuc_max is not None else 'inf'}px, "
        f"overlap_prev<={args.overlap_prev_max}, "
        f"R_mono>={args.rmono_min if args.use_rmono_gate else 'off'}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_dn, dpi=200)

    # Overlay N(t)
    png_N = os.path.join(args.out_dir, "overlay_N_t_sharedbins.png")
    plt.figure()
    plt.plot(out_F["bin_center_ms"], out_F["N"], marker="o", label="FAPI")
    plt.plot(out_T["bin_center_ms"], out_T["N"], marker="o", label="FAPI-TEMPO")
    plt.fill_between(ci_F["bin_center_ms"], ci_F["N_lo"], ci_F["N_hi"], alpha=0.2)
    plt.fill_between(ci_T["bin_center_ms"], ci_T["N_lo"], ci_T["N_hi"], alpha=0.2)
    plt.xlabel("time (ms)")
    plt.ylabel("Cumulative N(t)")
    plt.title("Stable nucleation N(t) (shared bins, bootstrap CI)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_N, dpi=200)

    print("\n[OK] Wrote:")
    print(" ", out_F_csv)
    print(" ", out_T_csv)
    print(" ", ciF_csv)
    print(" ", ciT_csv)
    print(" ", png_dn)
    print(" ", png_N)

    print("\n=== Counts ===")
    print(f"FAPI accepted: {len(fapi_events)}")
    print(f"TEMPO accepted: {len(tempo_events)}")


if __name__ == "__main__":
    main()
