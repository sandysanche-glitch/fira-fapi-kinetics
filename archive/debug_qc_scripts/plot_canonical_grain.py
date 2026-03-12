import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

# 1) Decode COCO-style RLE to a boolean mask
try:
    from pycocotools import mask as mask_utils

    def decode_rle(segmentation):
        """
        segmentation: dict with keys 'size' [H, W] and 'counts' (compressed RLE string)
        returns: boolean mask (H, W)
        """
        rle = segmentation
        counts = rle["counts"]
        if isinstance(counts, str):
            rle = {"size": rle["size"], "counts": counts.encode("ascii")}
        m = mask_utils.decode(rle)  # uint8 {0,1}, shape (H, W)
        return m.astype(bool)

except ImportError:
    def decode_rle(segmentation):
        raise ImportError(
            "pycocotools is required to decode the RLE. "
            "Install with `pip install pycocotools`."
        )


def load_heatmap_image(image_path):
    """Load intensity/heat-map image as float32 array in [0,1]."""
    img = np.array(Image.open(image_path)).astype(np.float32)
    if img.ndim == 3:
        # convert RGB → grayscale by simple average
        img = img.mean(axis=2)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    return img


# ----------------------------------------------------------------------
# Main plotting function
# ----------------------------------------------------------------------

def plot_canonical_grain(
    json_path,
    image_path,
    grain_index=0,
    n_annuli=6,
    defect_threshold=0.6,
    rng_seed=0,
):
    """
    json_path: path to FAPI_0.json (list of dicts with 'segmentation', 'bbox', etc.)
    image_path: path to corresponding heat-map / CL image (e.g. FAPI_0_heatmap.png)
    grain_index: which grain in the JSON list to show
    n_annuli: number of concentric rings to draw
    defect_threshold: threshold on normalized intensity to define "defect" pixels
    """

    json_path = Path(json_path)
    image_path = Path(image_path)

    # --- Load data ---
    with open(json_path, "r") as f:
        annos = json.load(f)

    if grain_index < 0 or grain_index >= len(annos):
        raise IndexError(f"grain_index {grain_index} out of range (0..{len(annos)-1})")

    anno = annos[grain_index]
    seg = anno["segmentation"]
    grain_mask = decode_rle(seg)  # (H, W) bool

    img = load_heatmap_image(image_path)

    if img.shape != grain_mask.shape:
        raise ValueError(
            f"Image shape {img.shape} and mask shape {grain_mask.shape} do not match."
        )

    # --- Compute grain geometry ---
    ys, xs = np.nonzero(grain_mask)
    if ys.size == 0:
        raise RuntimeError("Selected grain has empty mask.")

    # Center of mass in pixel coordinates (x, y)
    cx = xs.mean()
    cy = ys.mean()

    # Max radius (in pixels) from center to any grain pixel
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    R = r.max()

    # --- Defect pixels: intensity above threshold within the grain ---
    # (normalized already in load_heatmap_image)
    norm_intensity = img
    defect_mask = (norm_intensity > defect_threshold) & grain_mask

    # --- Simple orientation from image gradients ---
    gy, gx = np.gradient(norm_intensity)
    # orientation angle of gradients
    theta = np.arctan2(gy, gx)

    # Randomly sample some pixels in the grain to show arrows
    rng = np.random.default_rng(rng_seed)
    n_arrows = min(150, xs.size)  # cap to avoid clutter
    sample_idx = rng.choice(xs.size, size=n_arrows, replace=False)
    xs_s = xs[sample_idx]
    ys_s = ys[sample_idx]
    theta_s = theta[ys_s, xs_s]

    # Directions for quiver (note image coordinates: y points down)
    u = np.cos(theta_s)
    v = -np.sin(theta_s)  # minus sign because y-axis is inverted in imshow

    # --- Prepare figure ---
    fig, ax = plt.subplots(figsize=(4, 4), dpi=200)

    # Show underlying intensity restricted to the grain
    base = np.zeros_like(img)
    base[grain_mask] = img[grain_mask]

    im = ax.imshow(base, cmap="magma", origin="upper")
    ax.set_axis_off()

    # Draw grain boundary as contour
    ax.contour(
        grain_mask.astype(float),
        levels=[0.5],
        colors="white",
        linewidths=1.0,
    )

    # --- Draw concentric annuli ---
    radii = np.linspace(0.0, 1.0, n_annuli + 1)[1:]  # skip 0
    for rr in radii:
        circle = plt.Circle(
            (cx, cy),
            rr * R,
            edgecolor="white",
            linestyle="--",
            linewidth=0.7,
            fill=False,
            alpha=0.6,
        )
        ax.add_patch(circle)

    # Label center and rim
    ax.text(
        cx,
        cy,
        "center\nr/R=0",
        color="white",
        fontsize=6,
        ha="center",
        va="center",
    )
    ax.text(
        cx + R * 0.85,
        cy,
        "r/R=1",
        color="white",
        fontsize=6,
        ha="center",
        va="center",
    )

    # --- Overlay defects as red points ---
    dy, dx = np.nonzero(defect_mask)
    ax.scatter(
        dx,
        dy,
        s=4,
        c="red",
        alpha=0.8,
        linewidths=0,
        label="Defect pixels",
    )

    # --- Draw orientation arrows (local texture) ---
    ax.quiver(
        xs_s,
        ys_s,
        u,
        v,
        angles="xy",
        scale_units="xy",
        scale=40.0,
        width=0.002,
        color="cyan",
        alpha=0.6,
        label="Local orientation",
    )

    # Optional legend (small)
    ax.legend(
        loc="lower right",
        fontsize=6,
        frameon=True,
        framealpha=0.6,
    )

    ax.set_title("Canonical grain with radial annuli", fontsize=9)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    base_dir = Path(r"D:\SWITCHdrive\Institution\Sts_grain morphology_ML\comparative datasets\FAPI")
    json_path = base_dir / "FAPI_0.json"
    image_path = base_dir / "FAPI_0_heatmap.png"  # adjust if your suffix differs

    plot_canonical_grain(
        json_path=json_path,
        image_path=image_path,
        grain_index=0,      # try different indices to find a nice, big grain
        n_annuli=6,
        defect_threshold=0.6,
    )
