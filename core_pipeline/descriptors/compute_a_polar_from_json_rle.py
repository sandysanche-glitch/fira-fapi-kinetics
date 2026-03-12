#!/usr/bin/env python
"""
compute_A_polar_from_json_rle.py

Compute shape anisotropy A_polar for FAPI and FAPI–TEMPO from
list-style JSON files containing COCO RLE 'segmentation' (and optionally 'category_id').

Data layout (your machine):

  JSONs + PNG images:
    F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\comparative datasets\FAPI
    F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\comparative datasets\FAPI-TEMPO

For THESE JSONs, there is no 'category_id', so we treat ALL annotations
in each file as grains. If 'category_id' exists, only those equal to GRAIN_CAT_ID are used.
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pycocotools import mask as maskUtils   # pip install pycocotools


# ------------------ paths & constants ------------------------------------

BASE = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\comparative datasets"
FAPI_DIR = Path(BASE) / "FAPI"
TEMPO_DIR = Path(BASE) / "FAPI-TEMPO"

GRAIN_CAT_ID = 1   # grains (if category_id is present)
N_THETA = 360      # angular resolution for r(theta)


# ------------------ core functions ---------------------------------------

def compute_A_polar_from_mask(mask: np.ndarray, n_theta: int = 360) -> float:
    """
    Compute A_polar from a binary grain mask (True/1 inside grain).

    A_polar = < [ r(theta)/<r> - 1 ]^2 >_theta
    """
    mask = mask.astype(bool)
    if mask.sum() < 10:
        return np.nan

    ys, xs = np.nonzero(mask)
    cy = ys.mean()
    cx = xs.mean()

    dy = ys - cy
    dx = xs - cx
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    theta = np.mod(theta, 2 * np.pi)

    edges = np.linspace(0.0, 2 * np.pi, n_theta + 1)
    idx = np.digitize(theta, edges) - 1
    idx = np.clip(idx, 0, n_theta - 1)

    r_theta = np.full(n_theta, np.nan)
    for k in range(n_theta):
        r_k = r[idx == k]
        if r_k.size > 0:
            r_theta[k] = np.max(r_k)

    valid = np.isfinite(r_theta)
    if valid.sum() < 10:
        return np.nan

    r_theta = r_theta[valid]
    r_mean = np.mean(r_theta)
    A_polar = np.mean((r_theta / r_mean - 1.0) ** 2)
    return float(A_polar)


def decode_segmentation_rle(seg):
    """
    Decode COCO-style RLE segmentation (with 'size' and 'counts' string)
    into a H x W boolean array using pycocotools.
    """
    rle = seg  # already {'size':[H,W], 'counts': '...'}
    m = maskUtils.decode(rle)   # returns H x W uint8 or HxWx1
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(bool)


def process_dataset(json_dir: Path, label: str, out_csv: Path,
                    grain_cat_id: int = 1,
                    n_theta: int = 360):
    print(f"\n[INFO] === Processing {label} in {json_dir} ===")
    json_files = sorted(json_dir.glob("*.json"))
    print(f"[INFO] Found {len(json_files)} JSON files.")

    if not json_files:
        print(f"[ERROR] No JSON files found in {json_dir}")
        return

    # quick debug introspection of the first JSON
    first = json_files[0]
    print(f"[DEBUG] Inspecting first JSON: {first}")
    with open(first, "r") as f:
        data0 = json.load(f)
    print(f"[DEBUG] Top-level type: {type(data0)}")
    if isinstance(data0, list) and data0:
        print(f"[DEBUG] First element keys: {list(data0[0].keys())}")
    elif isinstance(data0, dict):
        print(f"[DEBUG] Dict keys: {list(data0.keys())}")

    total_anns = 0
    grain_anns = 0
    decoded_masks = 0
    used_grains = 0

    rows = []

    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)

        # These FAPI JSONs are list-style: [ {bbox, score, segmentation, mask_name}, ... ]
        if not isinstance(data, list):
            # skip COCO-dict style; this script is for list-JSONs
            continue

        for i, ann in enumerate(data):
            total_anns += 1

            # NEW LOGIC:
            # if 'category_id' is present, use it to filter.
            # If it's missing (your case), treat this annotation as a grain.
            cat_id = ann.get("category_id", grain_cat_id)
            if cat_id != grain_cat_id:
                continue

            grain_anns += 1

            seg = ann.get("segmentation", None)
            if not isinstance(seg, dict) or "counts" not in seg:
                continue

            try:
                mask_bin = decode_segmentation_rle(seg)
            except Exception as e:
                print(f"[WARN] Failed to decode RLE in {jf.name} idx {i}: {e}")
                continue

            decoded_masks += 1

            if mask_bin.sum() < 10:
                continue

            A_p = compute_A_polar_from_mask(mask_bin, n_theta=n_theta)
            used_grains += 1

            rows.append(
                {
                    "json_file": jf.name,
                    "grain_index": i,
                    "A_polar": A_p,
                }
            )

    print("\n[INFO] Summary for", label)
    print(f"  total annotations (all cats):        {total_anns}")
    print(f"  grain annotations (after filter):    {grain_anns}")
    print(f"  decoded RLE masks:                   {decoded_masks}")
    print(f"  grains with usable masks:            {used_grains}")

    if not rows:
        print(f"[WARN] No usable grain masks found for {label}.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    mean_A = df["A_polar"].mean()
    std_A  = df["A_polar"].std(ddof=1)
    print(f"[RESULT] {label}: A_polar = {mean_A:.3f} ± {std_A:.3f} (N = {len(df)})")
    print(f"[INFO] Saved detailed values to {out_csv}")


def main():
    out_fapi  = Path(BASE) / "A_polar_FAPI.csv"
    out_tempo = Path(BASE) / "A_polar_FAPI_TEMPO.csv"

    process_dataset(FAPI_DIR, "FAPI", out_fapi,
                    grain_cat_id=GRAIN_CAT_ID, n_theta=N_THETA)
    process_dataset(TEMPO_DIR, "FAPI–TEMPO", out_tempo,
                    grain_cat_id=GRAIN_CAT_ID, n_theta=N_THETA)


if __name__ == "__main__":
    main()
