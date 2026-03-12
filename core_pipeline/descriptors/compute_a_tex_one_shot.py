#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compute_A_tex_one_shot.py

One-shot extractor for texture anisotropy A_tex from list-style JSONs
(top-level list of annotations) + PNG images in the SAME folder.

Outputs:
  - A_tex_FAPI.csv
  - A_tex_FAPI_TEMPO.csv

Also prints mean ± std + LaTeX-ready values line for your Table.

Assumptions that match your dataset:
  - JSON files are list-style: FAPI_*.json and FAPI_TEMPO_*.json
  - Each annotation contains:
      ann["segmentation"] = {"size": [H, W], "counts": <COCO RLE>}
    where "counts" is the standard compressed COCO RLE string.
  - The intensity image for each JSON is:
      <stem> + intensity_suffix  (default: "_heatmap.png")
    e.g., FAPI_0_heatmap.png
    If missing, it falls back to <stem>.png

Dependencies:
  - numpy
  - pandas
  - Pillow
  - pycocotools  (for robust COCO RLE decoding)

Install if needed:
  pip install numpy pandas pillow pycocotools
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

try:
    from pycocotools import mask as mask_utils
except ImportError as e:
    raise ImportError(
        "pycocotools is required to decode COCO RLE.\n"
        "Install with: pip install pycocotools"
    ) from e


# --------- Defaults matching your current storage ----------------------------
DEFAULT_FAPI_DIR = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\comparative datasets\FAPI"
DEFAULT_TEMPO_DIR = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\comparative datasets\FAPI-TEMPO"
DEFAULT_OUT_DIR = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\comparative datasets"


# --------- Core math --------------------------------------------------------
def compute_a_tex_from_mask_and_image(mask: np.ndarray, img: np.ndarray, grad_percentile: float = 70.0):
    """
    Compute scalar texture anisotropy A_tex for one grain.

    A_tex = |Σ w exp(i 2θ)| / Σ w   with axial orientation and weights w=|∇I|.

    Parameters
    ----------
    mask : bool array, shape (H, W)
    img  : float array, shape (H, W)
    grad_percentile : float
        Keep only pixels with gradient magnitude >= this percentile
        within the grain to suppress noise.

    Returns
    -------
    float or np.nan
    """
    if mask.sum() == 0:
        return np.nan

    # mild safety: ensure float
    I = img.astype(np.float32)

    # gradients
    gy, gx = np.gradient(I)  # note order: y then x
    mag = np.hypot(gx, gy)

    # select inside mask
    mag_m = mag[mask]
    gx_m = gx[mask]
    gy_m = gy[mask]

    # suppress weak gradients
    if mag_m.size == 0:
        return np.nan
    thr = np.percentile(mag_m, grad_percentile)
    keep = mag_m >= thr

    mag_k = mag_m[keep]
    gx_k = gx_m[keep]
    gy_k = gy_m[keep]

    if mag_k.size == 0 or np.nansum(mag_k) <= 0:
        return np.nan

    # axial orientation: angle of gradient, doubled in order parameter
    theta = np.arctan2(gy_k, gx_k)

    # weights
    w = mag_k

    num = np.abs(np.nansum(w * np.exp(1j * 2.0 * theta)))
    den = np.nansum(w)

    if den <= 0:
        return np.nan

    return float(num / den)


def decode_rle_to_mask(seg):
    """
    Decode COCO RLE dict {"size":[H,W], "counts":...} to boolean mask.
    """
    if not isinstance(seg, dict) or "size" not in seg or "counts" not in seg:
        return None

    h, w = seg["size"]
    rle = {"size": [h, w], "counts": seg["counts"]}

    m = mask_utils.decode(rle)
    # pycocotools may return HxWx1
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(bool)


def load_intensity_image(folder: Path, stem: str, intensity_suffix: str):
    """
    Try <stem><intensity_suffix>, else <stem>.png
    Returns float32 grayscale image or None.
    """
    cand1 = folder / f"{stem}{intensity_suffix}"
    cand2 = folder / f"{stem}.png"

    img_path = None
    if cand1.exists():
        img_path = cand1
    elif cand2.exists():
        img_path = cand2

    if img_path is None:
        return None

    img = Image.open(img_path).convert("L")
    return np.array(img, dtype=np.float32)


# --------- Dataset processing ----------------------------------------------
def process_dataset(dir_path: Path, label: str, out_csv: Path, intensity_suffix: str,
                    min_pixels: int = 200, grad_percentile: float = 70.0):
    """
    Loop over list-JSONs in folder, compute A_tex per annotation.
    """
    json_files = sorted(dir_path.glob(f"{label.replace('–','_').replace('-','_')}_*.json"))
    # The above pattern might be too strict for your naming.
    # So we also fall back to the simpler known stems:
    if label == "FAPI":
        json_files = sorted(dir_path.glob("FAPI_*.json"))
    elif label == "FAPI–TEMPO":
        json_files = sorted(dir_path.glob("FAPI_TEMPO_*.json"))

    print(f"[INFO] Processing {label} in {dir_path} ...")
    print(f"[INFO] Found {len(json_files)} JSON files.")

    rows = []
    total_anns = 0
    used = 0

    for jf in json_files:
        stem = jf.stem  # e.g., FAPI_0 or FAPI_TEMPO_0
        img = load_intensity_image(dir_path, stem, intensity_suffix)
        if img is None:
            # skip quietly; you already have many missing intensity images
            continue

        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue

        if not isinstance(data, list):
            continue

        for ai, ann in enumerate(data):
            total_anns += 1
            seg = ann.get("segmentation", None)
            mask = decode_rle_to_mask(seg)
            if mask is None:
                continue

            area = int(mask.sum())
            if area < min_pixels:
                continue

            # protect against size mismatch
            if mask.shape != img.shape:
                # try a safe skip rather than risky resize
                continue

            atex = compute_a_tex_from_mask_and_image(mask, img, grad_percentile=grad_percentile)
            if not np.isfinite(atex):
                continue

            used += 1
            rows.append({
                "file_name": stem,
                "ann_index": ai,
                "area_px": area,
                "A_tex": atex,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No usable grains found for {label}. Check RLE + image suffix.")

    df.to_csv(out_csv, index=False)

    mean = float(df["A_tex"].mean())
    std = float(df["A_tex"].std(ddof=1))

    print(f"[INFO] {label}: total anns scanned = {total_anns}")
    print(f"[INFO] {label}: grains used       = {used}")
    print(f"[RESULT] {label}: A_tex = {mean:.3f} ± {std:.3f}")
    print(f"[INFO] Saved: {out_csv}")

    return mean, std, len(df)


# --------- CLI / one-shot runner -------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Compute A_tex from list-JSON + PNG images.")
    p.add_argument("--fapi-dir", default=DEFAULT_FAPI_DIR, help="Folder with FAPI_*.json and images.")
    p.add_argument("--tempo-dir", default=DEFAULT_TEMPO_DIR, help="Folder with FAPI_TEMPO_*.json and images.")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output folder for CSVs.")
    p.add_argument("--intensity-suffix", default="_heatmap.png",
                   help="Preferred image suffix per JSON stem (fallback is .png).")
    p.add_argument("--min-pixels", type=int, default=200, help="Min grain area (px) to keep.")
    p.add_argument("--grad-percentile", type=float, default=70.0,
                   help="Gradient-magnitude percentile threshold within each grain.")
    return p.parse_args()


def main():
    args = parse_args()

    fapi_dir = Path(args.fapi_dir)
    tempo_dir = Path(args.tempo_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_fapi = out_dir / "A_tex_FAPI.csv"
    out_tempo = out_dir / "A_tex_FAPI_TEMPO.csv"

    m_f, s_f, n_f = process_dataset(
        fapi_dir, "FAPI", out_fapi,
        intensity_suffix=args.intensity_suffix,
        min_pixels=args.min_pixels,
        grad_percentile=args.grad_percentile,
    )

    m_t, s_t, n_t = process_dataset(
        tempo_dir, "FAPI–TEMPO", out_tempo,
        intensity_suffix=args.intensity_suffix,
        min_pixels=args.min_pixels,
        grad_percentile=args.grad_percentile,
    )

    # LaTeX-ready line for your table
    print("\n----------------------------------------------------------------------")
    print("Descriptor A_tex (texture anisotropy)")
    print(f"  FAPI       = {m_f:.4f} ± {s_f:.4f}   (N = {n_f})")
    print(f"  FAPI–TEMPO = {m_t:.4f} ± {s_t:.4f}   (N = {n_t})")

    # compact rounding for table
    # (match your style with \!\pm\!)
    def fmt(mean, std):
        # keep 3 decimals unless very small
        return f"{mean:.3f}", f"{std:.3f}"

    mf_s, sf_s = fmt(m_f, s_f)
    mt_s, st_s = fmt(m_t, s_t)

    print("  LaTeX values field:")
    print(f"    ${mf_s}\\!\\pm\\!{sf_s}$ / ${mt_s}\\!\\pm\\!{st_s}$ \\\\")


if __name__ == "__main__":
    main()
