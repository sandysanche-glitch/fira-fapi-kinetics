#!/usr/bin/env python
"""
intragrain_radial_defect_from_listjson.py

Compute average radial defect profiles for FAPI and FAPI–TEMPO from
list-JSON segmentation outputs and corresponding heat-map images.

Defects are defined by thresholding the per-pixel heat-map intensity
inside each grain. For each grain, we:
  - decode the RLE 'segmentation' into a binary mask
  - compute distances from the grain centroid
  - normalize radius r by the grain's maximum radius R -> r/R in [0, 1]
  - bin pixels by r/R and compute:
        defect_fraction(bin) = (# defect pixels in bin) /
                               (# grain pixels in bin)
Then we average these radial defect profiles over all grains in a dataset.

Outputs:
  - PNG plot of FAPI vs FAPI–TEMPO mean radial defect fraction
  - optional .npz with raw per-grain profiles (for further analysis)
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import imageio.v2 as imageio

try:
    from pycocotools import mask as maskUtils
except ImportError as e:
    raise ImportError(
        "pycocotools is required to decode RLE segmentations.\n"
        "Install it e.g. with:\n"
        "  pip install pycocotools\n"
        "or (on Windows / conda):\n"
        "  conda install -c conda-forge pycocotools"
    ) from e

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute average radial defect profiles for FAPI and FAPI–TEMPO "
            "from list-JSON segmentations and heat-map images."
        )
    )
    parser.add_argument(
        "--fapi-dir",
        required=True,
        help="Directory containing FAPI_*.json and FAPI_*_heatmap.png",
    )
    parser.add_argument(
        "--tempo-dir",
        required=True,
        help="Directory containing FAPI_TEMPO_*.json and FAPI_TEMPO_*_heatmap.png",
    )
    parser.add_argument(
        "--out-prefix",
        required=True,
        help="Output prefix for plots and data (e.g. path/to/intragrain_defects)",
    )
    parser.add_argument(
        "--intensity-suffix",
        default="_heatmap.png",
        help="Suffix for heat-map images (default: '_heatmap.png')",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=25,
        help="Number of radial bins between r/R=0 and r/R=1 (default: 25)",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=200,
        help="Minimum number of pixels in a grain mask to be considered (default: 200)",
    )
    parser.add_argument(
        "--defect-threshold",
        type=float,
        default=0.6,
        help=(
            "Threshold on normalized heat-map intensity (0–1) to flag defects "
            "(default: 0.6). Intensities are normalized per image to max=1."
        ),
    )
    parser.add_argument(
        "--save-npz",
        action="store_true",
        help="If set, save per-grain profiles to a .npz next to the PNG.",
    )
    return parser.parse_args()


def decode_rle_to_mask(segm: dict) -> np.ndarray:
    """Decode a COCO-style RLE segmentation to a 2D boolean mask."""
    m = maskUtils.decode(segm)
    # pycocotools can return (H, W, 1); squeeze if needed
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(bool)


def compute_radial_defect_profile(
    mask: np.ndarray,
    intensity: np.ndarray,
    defect_threshold: float,
    n_bins: int,
    min_pixels: int,
) -> np.ndarray | None:
    """
    For a single grain mask + intensity image, compute radial defect fraction
    as a function of normalized radius r/R.

    Returns:
        profile: shape (n_bins,) with defect fraction per bin,
                 or None if the grain is too small or degenerate.
    """
    # basic sanity
    if mask.shape != intensity.shape:
        raise ValueError(
            f"Mask and intensity shape mismatch: {mask.shape} vs {intensity.shape}"
        )

    ys, xs = np.nonzero(mask)
    n_pix = xs.size
    if n_pix < min_pixels:
        return None

    # centroid of the grain (in pixel coords)
    cx = xs.mean()
    cy = ys.mean()

    # distances from centroid
    dx = xs - cx
    dy = ys - cy
    r = np.sqrt(dx * dx + dy * dy)
    r_max = r.max()
    if r_max <= 0:
        return None

    r_norm = r / r_max  # r/R in [0, 1]

    # grab intensity at grain pixels and normalize per-image to [0,1]
    vals = intensity[ys, xs].astype(np.float32)
    vmax = vals.max()
    if vmax > 0:
        vals_norm = vals / vmax
    else:
        # completely flat image; no meaningful defects
        return None

    defect = vals_norm >= defect_threshold

    # bin r_norm into n_bins
    bin_idx = np.floor(r_norm * n_bins).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    profile = np.full(n_bins, np.nan, dtype=np.float32)
    for b in range(n_bins):
        in_bin = bin_idx == b
        n_in_bin = in_bin.sum()
        if n_in_bin == 0:
            continue
        # defect fraction in this annulus
        n_def = np.logical_and(in_bin, defect).sum()
        profile[b] = n_def / n_in_bin

    return profile


def process_dataset(
    dir_path: str | Path,
    intensity_suffix: str,
    n_bins: int,
    min_pixels: int,
    defect_threshold: float,
    label: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process all JSON files in a directory and compute the mean + std radial
    defect profile over grains.

    Returns:
        r_centers: (n_bins,) radial coordinate centers in r/R
        mean_profile: (n_bins,) mean defect fraction
        std_profile:  (n_bins,) std over grains
    """
    dir_path = Path(dir_path)
    json_paths = sorted(dir_path.glob("*.json"))

    logging.info(f"[{label}] Found {len(json_paths)} JSON files in {dir_path}")
    all_profiles: list[np.ndarray] = []

    for jf in json_paths:
        base = jf.stem  # e.g. 'FAPI_0'
        img_path = jf.with_name(base + intensity_suffix)

        if not img_path.is_file():
            logging.warning(f"[{label}] No heat-map image for {base}; skipping file.")
            continue

        try:
            img = imageio.imread(img_path)
        except Exception as e:
            logging.warning(f"[{label}] Failed to read {img_path}: {e}")
            continue

        # convert to grayscale if RGB
        if img.ndim == 3:
            img = img.mean(axis=2)

        img = img.astype(np.float32)

        # load list-JSON
        try:
            with open(jf, "r") as f:
                data = json.load(f)
        except Exception as e:
            logging.warning(f"[{label}] Failed to load JSON {jf}: {e}")
            continue

        if not isinstance(data, list):
            logging.warning(
                f"[{label}] JSON {jf} is not a list; skipping (expected list of annotations)."
            )
            continue

        for ann in data:
            segm = ann.get("segmentation", None)
            if segm is None:
                continue

            try:
                mask = decode_rle_to_mask(segm)
            except Exception as e:
                logging.warning(
                    f"[{label}] Failed to decode RLE in {jf} (mask_name={ann.get('mask_name')}): {e}"
                )
                continue

            # ensure same shape as intensity
            if mask.shape != img.shape:
                logging.warning(
                    f"[{label}] Shape mismatch for {jf} mask vs image: "
                    f"{mask.shape} vs {img.shape}; skipping this grain."
                )
                continue

            prof = compute_radial_defect_profile(
                mask=mask,
                intensity=img,
                defect_threshold=defect_threshold,
                n_bins=n_bins,
                min_pixels=min_pixels,
            )
            if prof is not None:
                all_profiles.append(prof)

    if not all_profiles:
        raise RuntimeError(f"[{label}] No valid grains found in {dir_path}")

    profiles = np.stack(all_profiles, axis=0)  # (n_grains, n_bins)
    mean_profile = np.nanmean(profiles, axis=0)
    std_profile = np.nanstd(profiles, axis=0)

    r_edges = np.linspace(0.0, 1.0, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    logging.info(
        f"[{label}] used {profiles.shape[0]} grains; mean defect fraction over all bins "
        f"= {np.nanmean(mean_profile):.4f}"
    )

    return r_centers, mean_profile, std_profile, profiles


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # FAPI
    r_fapi, m_fapi, s_fapi, profs_fapi = process_dataset(
        dir_path=args.fapi_dir,
        intensity_suffix=args.intensity_suffix,
        n_bins=args.n_bins,
        min_pixels=args.min_pixels,
        defect_threshold=args.defect_threshold,
        label="FAPI",
    )

    # FAPI–TEMPO
    r_tempo, m_tempo, s_tempo, profs_tempo = process_dataset(
        dir_path=args.tempo_dir,
        intensity_suffix=args.intensity_suffix,
        n_bins=args.n_bins,
        min_pixels=args.min_pixels,
        defect_threshold=args.defect_threshold,
        label="FAPI–TEMPO",
    )

    # --- plotting ---
    fig, ax = plt.subplots(figsize=(5.0, 4.0))

    ax.plot(r_fapi, m_fapi, label="FAPI", lw=2)
    ax.fill_between(
        r_fapi, m_fapi - s_fapi, m_fapi + s_fapi, alpha=0.3
    )

    ax.plot(r_tempo, m_tempo, label="FAPI–TEMPO", lw=2)
    ax.fill_between(
        r_tempo, m_tempo - s_tempo, m_tempo + s_tempo, alpha=0.3
    )

    ax.set_xlabel("Normalized radius $r/R$")
    ax.set_ylabel("Radial defect fraction")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    out_png = out_prefix.with_name(out_prefix.name + "_radial_defect_profiles.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    logging.info(f"Saved radial defect profile plot to: {out_png}")

    # optional: save raw profiles
    if args.save_npz:
        out_npz = out_prefix.with_name(out_prefix.name + "_radial_defect_profiles.npz")
        np.savez(
            out_npz,
            r=r_fapi,  # same for both, since it's just (0.5 + bin)/n_bins
            mean_fapi=m_fapi,
            std_fapi=s_fapi,
            mean_tempo=m_tempo,
            std_tempo=s_tempo,
            profiles_fapi=profs_fapi,
            profiles_tempo=profs_tempo,
        )
        logging.info(f"Saved raw radial defect profiles to: {out_npz}")


if __name__ == "__main__":
    main()
