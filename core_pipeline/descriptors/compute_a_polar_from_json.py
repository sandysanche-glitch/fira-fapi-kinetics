#!/usr/bin/env python
"""
compute_A_polar_from_json.py

Compute shape anisotropy A_polar for FAPI and FAPI–TEMPO using
list-style JSON files and their mask_name PNGs.

Configured for:
  FAPI JSON + masks:
    F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\comparative datasets\FAPI
  FAPI–TEMPO JSON + masks:
    F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\comparative datasets\FAPI-TEMPO

Outputs:
  A_polar_FAPI.csv
  A_polar_FAPI_TEMPO.csv
in the same "comparative datasets" folder.
"""

import os
import json
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
import pandas as pd


# --- paths (adapt here if needed) -----------------------------------------

BASE = r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\comparative datasets"
FAPI_DIR = Path(BASE) / "FAPI"
TEMPO_DIR = Path(BASE) / "FAPI-TEMPO"


# --- core computation -----------------------------------------------------

def compute_A_polar_from_mask(mask, n_theta=360):
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
    rows = []
    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"No JSON files found in {json_dir}")

    print(f"[INFO] Processing {label} in {json_dir} ...")
    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            # skip COCO-style – this script is for list-JSON
            continue

        for i, ann in enumerate(data):
            mask_name = ann.get("mask_name", None)
            if mask_name is None:
                continue

            mask_path = json_dir / mask_name
            if not mask_path.is_file():
                continue

            mask_img = imageio.imread(mask_path)
            if mask_img.ndim == 3:
                mask_img = mask_img[..., 0]
            mask_bin = mask_img > 0

            A_polar = compute_A_polar_from_mask(mask_bin, n_theta=n_theta)
            rows.append(
                {
                    "json_file": jf.name,
                    "grain_index": i,
                    "mask_name": mask_name,
                    "A_polar": A_polar,
                }
            )

    if not rows:
        raise RuntimeError(f"No grains processed for {label} – check masks/JSON.")

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
