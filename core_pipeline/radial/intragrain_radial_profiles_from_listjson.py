import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools import mask as maskUtils


# -----------------------------
# Argument parsing
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Average intragrain radial heat-map profiles from list-style JSON + heatmaps."
    )
    parser.add_argument(
        "--fapi-dir",
        type=str,
        required=True,
        help="Folder with FAPI_*.json and FAPI_*_heatmap.png",
    )
    parser.add_argument(
        "--tempo-dir",
        type=str,
        required=True,
        help="Folder with FAPI_TEMPO_*.json and FAPI_TEMPO_*_heatmap.png",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        required=True,
        help="Prefix for output PNG (no extension).",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=25,
        help="Number of radial bins between 0 and 1.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=200,
        help="Minimum grain area (pixels) to include.",
    )
    return parser.parse_args()


# -----------------------------
# Utilities
# -----------------------------
def load_list_json(json_path: Path):
    """Load your list-style JSON (one list of dicts with bbox/score/segmentation/mask_name)."""
    with open(json_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{json_path} is not a list-style JSON as expected.")
    return data


def decode_mask_from_seg(segmentation: dict) -> np.ndarray:
    """
    Decode COCO-compressed RLE (size + counts) to a boolean mask.
    segmentation: {'size': [H, W], 'counts': <string>}
    """
    size = segmentation["size"]  # [H, W]
    counts = segmentation["counts"]

    # pycocotools expects counts as bytes for compressed RLE
    if isinstance(counts, str):
        rle = {"size": size, "counts": counts.encode("utf-8")}
    else:
        rle = {"size": size, "counts": counts}

    mask = maskUtils.decode(rle)  # H x W, uint8
    mask = mask.astype(bool)
    return mask


def radial_profile_from_mask(mask: np.ndarray, heatmap: np.ndarray, n_bins: int):
    """
    Compute radial profile of heatmap intensity inside a single grain mask.
    Radius is measured from the mask centroid and normalised by the max radius.
    """
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None

    yc = ys.mean()
    xc = xs.mean()

    rs = np.sqrt((ys - yc) ** 2 + (xs - xc) ** 2)
    r_max = rs.max()
    if r_max <= 0:
        return None

    r_norm = rs / r_max
    vals = heatmap[ys, xs]

    # Bin 0..1 into n_bins
    bin_idx = np.floor(r_norm * n_bins).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    sums = np.bincount(bin_idx, weights=vals, minlength=n_bins)
    counts = np.bincount(bin_idx, minlength=n_bins)

    prof = np.full(n_bins, np.nan, dtype=float)
    nonzero = counts > 0
    prof[nonzero] = sums[nonzero] / counts[nonzero]

    return prof


def load_heatmap(path: Path) -> np.ndarray:
    """Load heatmap as grayscale and normalise to [0,1]."""
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32)
    mn = arr.min()
    mx = arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr[:] = 0.0
    return arr


def process_dataset(
    dir_path: Path,
    pattern: str,
    n_bins: int,
    min_pixels: int,
    label: str,
):
    """
    Process all <pattern>.json in dir_path, with matching <stem>_heatmap.png.
    Returns:
        r_centers, mean_profile, std_profile, n_grains
    """
    json_files = sorted(dir_path.glob(pattern))
    all_profiles = []

    for jf in json_files:
        stem = jf.stem  # e.g. FAPI_0
        heatmap_path = dir_path / f"{stem}_heatmap.png"

        if not heatmap_path.exists():
            print(f"[WARN] no heatmap found for {stem}; skipping")
            continue

        # Load heatmap once per file
        heatmap = load_heatmap(heatmap_path)

        # Load list-style predictions
        try:
            preds = load_list_json(jf)
        except Exception as e:
            print(f"[WARN] skipping {jf.name}: cannot parse JSON ({e})")
            continue

        for ann in preds:
            seg = ann.get("segmentation", None)
            if seg is None:
                continue

            try:
                mask = decode_mask_from_seg(seg)
            except Exception as e:
                print(f"[WARN] segmentation decode failed in {jf.name}: {e}")
                continue

            area = mask.sum()
            if area < min_pixels:
                continue

            prof = radial_profile_from_mask(mask, heatmap, n_bins)
            if prof is None:
                continue

            all_profiles.append(prof)

    all_profiles = np.array(all_profiles, dtype=float)
    if all_profiles.size == 0:
        raise RuntimeError(f"No valid grains found in {dir_path}")

    r_centers = (np.arange(n_bins) + 0.5) / n_bins
    mean_profile = np.nanmean(all_profiles, axis=0)
    std_profile = np.nanstd(all_profiles, axis=0)

    print(f"[INFO] {label}: used {all_profiles.shape[0]} grains")

    return r_centers, mean_profile, std_profile, all_profiles.shape[0]


def plot_comparison(
    r,
    m_fapi,
    s_fapi,
    m_tempo,
    s_tempo,
    out_png: str,
):
    plt.figure(figsize=(5, 4))
    ax = plt.gca()

    ax.plot(r, m_fapi, label="FAPI", linewidth=2)
    ax.fill_between(
        r, m_fapi - s_fapi, m_fapi + s_fapi, alpha=0.2
    )

    ax.plot(r, m_tempo, label="FAPI–TEMPO", linewidth=2)
    ax.fill_between(
        r, m_tempo - s_tempo, m_tempo + s_tempo, alpha=0.2
    )

    ax.set_xlabel("Normalised radius $r/R$")
    ax.set_ylabel("Normalised heat-map intensity")
    ax.set_xlim(0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[OK] Saved comparison plot to {out_png}")


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    fapi_dir = Path(args.fapi_dir)
    tempo_dir = Path(args.tempo_dir)
    out_prefix = Path(args.out_prefix)

    # FAPI
    print(f"[INFO] Processing FAPI in {fapi_dir} ...")
    r_fapi, m_fapi, s_fapi, n_fapi = process_dataset(
        fapi_dir,
        pattern="FAPI_*.json",
        n_bins=args.n_bins,
        min_pixels=args.min_pixels,
        label="FAPI",
    )

    # FAPI–TEMPO
    print(f"[INFO] Processing FAPI–TEMPO in {tempo_dir} ...")
    r_tempo, m_tempo, s_tempo, n_tempo = process_dataset(
        tempo_dir,
        pattern="FAPI_TEMPO_*.json",
        n_bins=args.n_bins,
        min_pixels=args.min_pixels,
        label="FAPI–TEMPO",
    )

    # Output path (no Path.with_suffix trick to avoid previous error)
    out_png = str(out_prefix) + "_radial_heatmap_profiles.png"
    plot_comparison(r_fapi, m_fapi, s_fapi, m_tempo, s_tempo, out_png)


if __name__ == "__main__":
    main()
