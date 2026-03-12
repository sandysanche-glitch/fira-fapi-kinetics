#!/usr/bin/env python
"""
compute_A_polar_from_json_debug.py

Debug version to compute shape anisotropy A_polar for FAPI and FAPI–TEMPO
from list-style JSON files with `mask_name`.

Data layout (as on your machine):

  FAPI JSON + masks:
    F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\comparative datasets\FAPI

  FAPI–TEMPO JSON + masks:
    F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\comparative datasets\FAPI-TEMPO

This script will:
  - print how many JSONs it finds,
  - inspect the first JSON,
  - count how many annotations have mask_name,
  - count how many masks actually exist and are non-empty,
  - compute A_polar where possible,
  - write results to:
      A_polar_FAPI.csv
      A_polar_FAPI_TEMPO.csv
in the "comparative datasets" folder.
"""

import os
import json
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
import pandas as pd


# --- paths ---------------------------------------------------------------

BASE = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\comparative datasets"
FAPI_DIR = Path(BASE) / "FAPI"
TEMPO_DIR = Path(BASE) / "FAPI-TEMPO"


# --- core computation ----------------------------------------------------

def compute_A_polar_from_mask(mask: np.ndarray, n_theta: int = 360) -> float:
    """Compute A_polar from a binary mask (True/1 inside grain)."""
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


def process_dataset(json_dir: Path, label: str, out_csv: Path, n_theta: int = 360):
    print(f"\n[INFO] === Processing {label} in {json_dir} ===")
    json_files = sorted(json_dir.glob("*.json"))
    print(f"[INFO] Found {len(json_files)} JSON files.")

    if not json_files:
        print(f"[ERROR] No JSON files found in {json_dir}")
        return

    # --- inspect first JSON for debugging ---
    first = json_files[0]
    print(f"[DEBUG] Inspecting first JSON: {first}")
    with open(first, "r") as f:
        data0 = json.load(f)
    print(f"[DEBUG] Top-level type: {type(data0)}")

    if isinstance(data0, list):
        print(f"[DEBUG] List length: {len(data0)}")
        if data0:
            print(f"[DEBUG] First element keys: {list(data0[0].keys())}")
    elif isinstance(data0, dict):
        print(f"[DEBUG] Dict keys: {list(data0.keys())}")
    else:
        print("[DEBUG] Unexpected JSON structure.")

    total_anns = 0
    with_mask_name = 0
    masks_found = 0
    masks_big_enough = 0

    rows = []

    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            # This script is for list-style JSON. Skip COCO dicts.
            continue

        for i, ann in enumerate(data):
            total_anns += 1
            mask_name = ann.get("mask_name", None)
            if mask_name is None:
                continue
            with_mask_name += 1

            mask_path = json_dir / mask_name
            if not mask_path.is_file():
                # mask PNG missing
                continue
            masks_found += 1

            mask_img = imageio.imread(mask_path)
            if mask_img.ndim == 3:
                mask_img = mask_img[..., 0]
            mask_bin = mask_img > 0
            if mask_bin.sum() < 10:
                # too small / empty
                continue
            masks_big_enough += 1

            A_polar = compute_A_polar_from_mask(mask_bin, n_theta=n_theta)

            rows.append(
                {
                    "json_file": jf.name,
                    "grain_index": i,
                    "mask_name": mask_name,
                    "A_polar": A_polar,
                }
            )

    print("\n[INFO] Summary of what I saw:")
    print(f"  total annotations (all JSONs, list-style only): {total_anns}")
    print(f"  with mask_name:                               {with_mask_name}")
    print(f"  mask PNG exists on disk:                      {masks_found}")
    print(f"  masks with >= 10 pixels:                      {masks_big_enough}")
    print(f"  rows accumulated (grains used):               {len(rows)}")

    if not rows:
        print(f"[WARN] No usable grains found for {label}.")
        print("[WARN] Please check that the JSONs are list-style and that "
              "mask_name PNGs are present in the same folder.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    mean_A = df["A_polar"].mean()
    std_A = df["A_polar"].std(ddof=1)
    print(f"[INFO] {label}: {len(df)} grains, A_polar = {mean_A:.3f} ± {std_A:.3f}")
    print(f"[INFO] Saved to {out_csv}")


def main():
    out_fapi = Path(BASE) / "A_polar_FAPI.csv"
    out_tempo = Path(BASE) / "A_polar_FAPI_TEMPO.csv"

    process_dataset(FAPI_DIR, "FAPI", out_fapi, n_theta=360)
    process_dataset(TEMPO_DIR, "FAPI–TEMPO", out_tempo, n_theta=360)


if __name__ == "__main__":
    main()
