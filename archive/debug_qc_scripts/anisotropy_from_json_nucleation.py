import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from pycocotools import mask as mask_utils
except Exception as e:
    mask_utils = None
    _PYCOCO_ERR = e
else:
    _PYCOCO_ERR = None


def decode_rle_mask(segmentation):
    """
    Decode a COCO-style RLE segmentation into a boolean mask.

    segmentation: dict with keys 'size' = [H, W], 'counts' = RLE string or bytes.
    Returns: np.ndarray[H,W] of bool.
    """
    if mask_utils is None:
        raise RuntimeError(
            "pycocotools is required to decode RLE masks but could not be imported: "
            f"{_PYCOCO_ERR}"
        )

    if not isinstance(segmentation, dict):
        raise ValueError(f"Unsupported segmentation format: {type(segmentation)}")

    if "size" not in segmentation or "counts" not in segmentation:
        raise ValueError("Segmentation dict must contain 'size' and 'counts' keys.")

    rle = {"size": segmentation["size"], "counts": segmentation["counts"]}
    m = mask_utils.decode(rle)
    # decode can return HxW or HxWx1
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(bool)


def compute_area(mask):
    return float(np.count_nonzero(mask))


def compute_centroid(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return math.nan, math.nan
    cx = xs.mean()
    cy = ys.mean()
    return float(cx), float(cy)


def compute_perimeter(mask):
    """
    Crude perimeter estimate as number of boundary pixels (8-connected).
    """
    m = mask.astype(bool)
    if m.size == 0:
        return 0.0
    m_pad = np.pad(m, 1, mode="constant", constant_values=False)
    center = m_pad[1:-1, 1:-1]

    neighbors = [
        m_pad[:-2, 1:-1],   # up
        m_pad[2:, 1:-1],    # down
        m_pad[1:-1, :-2],   # left
        m_pad[1:-1, 2:],    # right
        m_pad[:-2, :-2],    # up-left
        m_pad[:-2, 2:],     # up-right
        m_pad[2:, :-2],     # down-left
        m_pad[2:, 2:],      # down-right
    ]

    interior = center.copy()
    for n in neighbors:
        interior &= n

    boundary = center & (~interior)
    return float(boundary.sum())


def compute_circularity(area_px, perimeter_px):
    if perimeter_px <= 0 or area_px <= 0:
        return math.nan
    C = 4.0 * math.pi * area_px / (perimeter_px ** 2)
    # numerical clipping
    return float(min(max(C, 0.0), 1.0))


def polar_profile_from_mask(mask, cx, cy, n_theta=72):
    """
    Compute per-angle boundary radius for a grain mask, given a center.

    mask: 2D boolean array
    cx, cy: center coordinates (float, in pixel coordinates x=column, y=row)
    Returns:
        theta_centers: 1D array of angle centers in radians (0..2π)
        r_bin: 1D array of max radius in each angular bin
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        theta_centers = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
        return theta_centers, np.zeros_like(theta_centers)

    dx = xs.astype(float) - cx
    dy = ys.astype(float) - cy
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)  # -pi..pi
    theta = np.mod(theta, 2.0 * math.pi)  # 0..2π

    edges = np.linspace(0.0, 2.0 * math.pi, n_theta + 1)
    theta_centers = 0.5 * (edges[:-1] + edges[1:])
    r_bin = np.zeros(n_theta, dtype=float)

    for i in range(n_theta):
        mask_bin = (theta >= edges[i]) & (theta < edges[i + 1])
        if np.any(mask_bin):
            # use max radius in the bin as boundary estimate
            r_bin[i] = float(r[mask_bin].max())
        else:
            r_bin[i] = 0.0

    return theta_centers, r_bin


def anisotropy_from_profile(radii):
    """
    Compute simple anisotropy metrics from 1D radius profile.

    radii: 1D array of non-negative radii.

    Returns:
        r_mean, A_polar, A_std
    """
    radii = np.asarray(radii, dtype=float)
    if len(radii) == 0 or np.all(radii == 0):
        return 0.0, 0.0, 0.0
    # ignore zeros when computing mean (if some angle bins are empty)
    valid = radii > 0
    if not np.any(valid):
        r_mean = float(radii.mean())
    else:
        r_mean = float(radii[valid].mean())
    if r_mean <= 0:
        return r_mean, 0.0, 0.0

    A_polar = (float(radii.max()) - float(radii.min())) / r_mean
    A_std = float(np.std(radii[valid]) / r_mean) if np.any(valid) else 0.0
    return r_mean, A_polar, A_std


def process_dataset(
    folder,
    label,
    nuc_x_key=None,
    nuc_y_key=None,
    grain_cat_id=2,
    px_per_um=2.20014,
    n_theta=72,
):
    """
    Process all JSON files in 'folder' and compute per-grain anisotropy metrics.

    Returns:
        per_grain_df, theta_centers, r_norm_mean, r_norm_std
    """
    folder = Path(folder)
    json_files = sorted(folder.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {folder}")

    rows = []
    all_profiles = []

    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)

        if isinstance(data, dict) and "annotations" in data:
            objs = data["annotations"]
        elif isinstance(data, list):
            objs = data
        else:
            raise ValueError(f"Unsupported JSON structure in {jf}")

        for idx, obj in enumerate(objs):
            seg = obj.get("segmentation", None)
            if seg is None:
                continue

            cat = obj.get("category_id", grain_cat_id)
            # keep only grains
            if grain_cat_id is not None and cat != grain_cat_id:
                continue

            try:
                mask = decode_rle_mask(seg)
            except Exception as e:
                print(f"[WARN] Skipping object {idx} in {jf.name}: {e}")
                continue

            area_px = compute_area(mask)
            if area_px <= 0:
                continue

            # nucleation center: explicit keys if present, else centroid
            if nuc_x_key and nuc_y_key and nuc_x_key in obj and nuc_y_key in obj:
                nuc_x = float(obj[nuc_x_key])
                nuc_y = float(obj[nuc_y_key])
            else:
                nuc_x, nuc_y = compute_centroid(mask)

            cx, cy = compute_centroid(mask)  # geometric centroid (for reference)

            perimeter_px = compute_perimeter(mask)
            circularity = compute_circularity(area_px, perimeter_px)

            theta_centers, r_bins = polar_profile_from_mask(
                mask, nuc_x, nuc_y, n_theta=n_theta
            )
            r_mean, A_polar, A_std = anisotropy_from_profile(r_bins)

            # normalized radial profile
            if r_mean > 0:
                r_norm = r_bins / r_mean
            else:
                r_norm = np.zeros_like(r_bins)

            all_profiles.append(r_norm)

            rows.append(
                {
                    "label": label,
                    "json_file": jf.name,
                    "grain_index": idx,
                    "area_px": area_px,
                    "perimeter_px": perimeter_px,
                    "circularity": circularity,
                    "nuc_x": nuc_x,
                    "nuc_y": nuc_y,
                    "centroid_x": cx,
                    "centroid_y": cy,
                    "R_mean_px": r_mean,
                    "R_mean_um": r_mean / px_per_um if px_per_um > 0 else math.nan,
                    "A_polar": A_polar,
                    "A_std": A_std,
                }
            )

    if not rows:
        raise RuntimeError(
            f"No grain objects found in {folder} with category_id={grain_cat_id}"
        )

    per_grain_df = pd.DataFrame(rows)

    # Dataset-averaged normalized profile
    profiles = np.vstack(all_profiles)  # [N_grains, n_theta]
    r_norm_mean = np.nanmean(profiles, axis=0)
    r_norm_std = np.nanstd(profiles, axis=0)
    theta_centers = np.linspace(
        0.0, 2.0 * math.pi, profiles.shape[1], endpoint=False
    )

    return per_grain_df, theta_centers, r_norm_mean, r_norm_std


def main():
    ap = argparse.ArgumentParser(
        description="Anisotropy metrics from JSON grain masks (FAPI vs FAPI-TEMPO)."
    )
    ap.add_argument("--fapi-dir", required=True, help="Folder with FAPI JSON files")
    ap.add_argument(
        "--tempo-dir", required=True, help="Folder with FAPI-TEMPO JSON files"
    )
    ap.add_argument(
        "--out-prefix", required=True, help="Output prefix (folder + base name)"
    )
    ap.add_argument(
        "--grain-cat-id", type=int, default=2, help="category_id for grains (default 2)"
    )
    ap.add_argument(
        "--px-per-um", type=float, default=2.20014,
        help="Pixels per micron (default 2.20014)",
    )
    ap.add_argument(
        "--n-theta", type=int, default=72,
        help="Number of angular bins for polar profiles",
    )
    ap.add_argument(
        "--nuc-x-key", type=str, default=None,
        help="Optional JSON key for nucleation x-coordinate",
    )
    ap.add_argument(
        "--nuc-y-key", type=str, default=None,
        help="Optional JSON key for nucleation y-coordinate",
    )
    args = ap.parse_args()

    out_prefix = Path(args.out_prefix)
    out_dir = out_prefix.parent
    if str(out_dir) != "":
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Processing FAPI dataset in {args.fapi_dir} ...")
    fapi_df, fapi_theta, fapi_r_mean, fapi_r_std = process_dataset(
        args.fapi_dir,
        label="FAPI",
        nuc_x_key=args.nuc_x_key,
        nuc_y_key=args.nuc_y_key,
        grain_cat_id=args.grain_cat_id,
        px_per_um=args.px_per_um,
        n_theta=args.n_theta,
    )

    print(f"[INFO] Processing FAPI-TEMPO dataset in {args.tempo_dir} ...")
    tempo_df, tempo_theta, tempo_r_mean, tempo_r_std = process_dataset(
        args.tempo_dir,
        label="FAPI-TEMPO",
        nuc_x_key=args.nuc_x_key,
        nuc_y_key=args.nuc_y_key,
        grain_cat_id=args.grain_cat_id,
        px_per_um=args.px_per_um,
        n_theta=args.n_theta,
    )

    # Save per-grain metrics
    fapi_csv = out_prefix.with_name(out_prefix.name + "_FAPI_anisotropy_per_grain.csv")
    tempo_csv = out_prefix.with_name(
        out_prefix.name + "_FAPITEMPO_anisotropy_per_grain.csv"
    )
    fapi_df.to_csv(fapi_csv, index=False)
    tempo_df.to_csv(tempo_csv, index=False)
    print(
        f"[OK] Saved per-grain anisotropy CSVs:\n  {fapi_csv}\n  {tempo_csv}"
    )

    # Save averaged polar profiles
    theta_deg_fapi = np.degrees(fapi_theta)
    theta_deg_tempo = np.degrees(tempo_theta)

    fapi_prof_csv = out_prefix.with_name(
        out_prefix.name + "_FAPI_polar_profile.csv"
    )
    tempo_prof_csv = out_prefix.with_name(
        out_prefix.name + "_FAPITEMPO_polar_profile.csv"
    )

    pd.DataFrame(
        {
            "theta_deg": theta_deg_fapi,
            "r_norm_mean": fapi_r_mean,
            "r_norm_std": fapi_r_std,
        }
    ).to_csv(fapi_prof_csv, index=False)

    pd.DataFrame(
        {
            "theta_deg": theta_deg_tempo,
            "r_norm_mean": tempo_r_mean,
            "r_norm_std": tempo_r_std,
        }
    ).to_csv(tempo_prof_csv, index=False)

    print(f"[OK] Saved polar profiles:\n  {fapi_prof_csv}\n  {tempo_prof_csv}")

    # Plots
    try:
        import matplotlib.pyplot as plt

        # 1) Histogram of A_polar for both datasets
        plt.figure()
        bins = np.linspace(
            0.0,
            max(
                fapi_df["A_polar"].max(),
                tempo_df["A_polar"].max(),
                1e-3,
            ),
            40,
        )
        plt.hist(
            fapi_df["A_polar"].dropna(),
            bins=bins,
            histtype="step",
            label="FAPI",
        )
        plt.hist(
            tempo_df["A_polar"].dropna(),
            bins=bins,
            histtype="step",
            label="FAPI-TEMPO",
        )
        plt.xlabel("Anisotropy A_polar")
        plt.ylabel("Count")
        plt.title("Grain shape anisotropy (polar, per grain)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            out_prefix.with_name(out_prefix.name + "_A_polar_hist.png"),
            dpi=200,
        )
        plt.close()

        # 2) Dataset-averaged normalized polar profiles (Cartesian)
        plt.figure()
        plt.plot(theta_deg_fapi, fapi_r_mean, label="FAPI")
        plt.plot(theta_deg_tempo, tempo_r_mean, label="FAPI-TEMPO")
        plt.xlabel("Angle (deg)")
        plt.ylabel("Normalized radius r(θ)/⟨r⟩")
        plt.title("Average grain shape in polar coordinates")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            out_prefix.with_name(
                out_prefix.name + "_polar_avg_cartesian.png"
            ),
            dpi=200,
        )
        plt.close()

        # 3) Same as polar plot (rose plot) – dataset-averaged
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="polar")
        # close the loop for nicer polar plot
        th_fapi = np.append(fapi_theta, fapi_theta[0])
        r_fapi = np.append(fapi_r_mean, fapi_r_mean[0])
        th_tempo = np.append(tempo_theta, tempo_theta[0])
        r_tempo = np.append(tempo_r_mean, tempo_r_mean[0])
        ax.plot(th_fapi, r_fapi, label="FAPI")
        ax.plot(th_tempo, r_tempo, label="FAPI-TEMPO")
        ax.set_title("Average normalized grain shape (polar)")
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
        plt.tight_layout()
        plt.savefig(
            out_prefix.with_name(out_prefix.name + "_polar_avg_rose.png"),
            dpi=200,
        )
        plt.close()

        print("[OK] Saved anisotropy plots.")
    except Exception as e:
        print(f"[WARN] Plotting skipped: {e}")

    print("[DONE] Anisotropy analysis finished.")


if __name__ == "__main__":
    main()
