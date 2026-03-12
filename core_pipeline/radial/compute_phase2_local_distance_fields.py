#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import distance_transform_edt, center_of_mass

try:
    from pycocotools import mask as maskUtils
except ImportError as e:
    raise ImportError(
        "pycocotools is required.\n"
        "conda install -c conda-forge pycocotools\n"
        "or pip install pycocotools / pycocotools-windows"
    ) from e


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def normalize_key(x: str) -> str:
    x = str(x).strip().replace("\\", "/").split("/")[-1]
    xl = x.lower()
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".json"]:
        if xl.endswith(ext):
            x = x[: -len(ext)]
            break
    return x


def iter_annotations(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict) and isinstance(obj.get("annotations", None), list):
        return [x for x in obj["annotations"] if isinstance(x, dict)]
    return []


def decode_mask(seg: Dict[str, Any]) -> np.ndarray:
    if not isinstance(seg, dict) or "size" not in seg or "counts" not in seg:
        raise ValueError("Segmentation must be COCO RLE with size/counts.")
    rle = {"size": seg["size"], "counts": seg["counts"]}
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("utf-8")
    m = maskUtils.decode(rle)
    if m.ndim == 3:
        m = m[..., 0]
    return m.astype(bool)


def is_category(ann: Dict[str, Any], target: int, missing_means_true: bool = False) -> bool:
    if "category_id" not in ann:
        return missing_means_true
    try:
        return int(ann["category_id"]) == int(target)
    except Exception:
        return False


def get_mask_name(stem: str, ann: Dict[str, Any], idx: int) -> str:
    if "mask_name" in ann and ann["mask_name"] is not None:
        return normalize_key(ann["mask_name"])
    return normalize_key(f"{stem}_{idx}")


def mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return 0, 0, 0, 0
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1, y1


def crop_with_margin(arr: np.ndarray, x0: int, y0: int, x1: int, y1: int, margin: int = 8):
    h, w = arr.shape[:2]
    xx0 = max(0, x0 - margin)
    yy0 = max(0, y0 - margin)
    xx1 = min(w - 1, x1 + margin)
    yy1 = min(h - 1, y1 + margin)
    return arr[yy0:yy1 + 1, xx0:xx1 + 1], xx0, yy0


# ------------------------------------------------------------
# local fields
# ------------------------------------------------------------
def compute_boundary_distance_field(crystal_mask: np.ndarray) -> np.ndarray:
    """
    Distance from each crystal pixel to the nearest crystal boundary pixel.
    Defined only inside crystal mask, NaN outside.
    """
    # distance to background, evaluated on crystal pixels
    d = distance_transform_edt(crystal_mask)
    out = np.full(crystal_mask.shape, np.nan, dtype=np.float32)
    out[crystal_mask] = d[crystal_mask].astype(np.float32)
    return out


def compute_defect_distance_field(crystal_mask: np.ndarray, defect_mask: np.ndarray | None) -> np.ndarray:
    """
    Distance from each crystal pixel to the nearest defect pixel.
    NaN outside crystal.
    If no defect exists, returns NaN inside crystal.
    """
    out = np.full(crystal_mask.shape, np.nan, dtype=np.float32)
    if defect_mask is None or not np.any(defect_mask):
        return out

    # distance to nearest True pixel in defect mask:
    # edt on inverse defect mask
    d = distance_transform_edt(~defect_mask)
    out[crystal_mask] = d[crystal_mask].astype(np.float32)
    return out


def compute_nucleus_distance_field(
    crystal_mask: np.ndarray,
    nucleus_mask: np.ndarray | None,
    nucleus_mode: str = "mask",
) -> np.ndarray:
    """
    Distance from each crystal pixel to nucleus.
    nucleus_mode:
      - 'mask': distance to nearest nucleus pixel
      - 'centroid': distance to nucleus centroid
    NaN outside crystal.
    If no nucleus is available, returns NaN inside crystal.
    """
    out = np.full(crystal_mask.shape, np.nan, dtype=np.float32)
    if nucleus_mask is None or not np.any(nucleus_mask):
        return out

    ys, xs = np.nonzero(crystal_mask)
    if xs.size == 0:
        return out

    if nucleus_mode == "mask":
        d = distance_transform_edt(~nucleus_mask)
        out[crystal_mask] = d[crystal_mask].astype(np.float32)
        return out

    elif nucleus_mode == "centroid":
        cy, cx = center_of_mass(nucleus_mask.astype(np.uint8))
        if not np.isfinite(cx) or not np.isfinite(cy):
            return out
        yy, xx = np.indices(crystal_mask.shape)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        out[crystal_mask] = d[crystal_mask].astype(np.float32)
        return out

    else:
        raise ValueError("nucleus_mode must be 'mask' or 'centroid'")


# ------------------------------------------------------------
# JSON parsing per image
# ------------------------------------------------------------
def parse_image_annotations(
    jf: Path,
    crystal_category_id: int = 1,
    nucleus_category_id: int = 2,
    defect_category_id: int = 3,
    assume_all_are_crystals_if_missing_category: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    data = json.loads(jf.read_text(encoding="utf-8"))
    anns = iter_annotations(data)

    crystals = []
    nuclei = []
    defects = []

    for i, ann in enumerate(anns):
        seg = ann.get("segmentation", None)
        if seg is None:
            continue

        cat_present = "category_id" in ann
        if not cat_present and assume_all_are_crystals_if_missing_category:
            crystals.append((i, ann))
            continue

        if is_category(ann, crystal_category_id, missing_means_true=False):
            crystals.append((i, ann))
        elif is_category(ann, nucleus_category_id, missing_means_true=False):
            nuclei.append((i, ann))
        elif is_category(ann, defect_category_id, missing_means_true=False):
            defects.append((i, ann))

    return crystals, nuclei, defects


def union_masks(masks: List[np.ndarray], shape: Tuple[int, int]) -> np.ndarray | None:
    if not masks:
        return None
    u = np.zeros(shape, dtype=bool)
    for m in masks:
        u |= m
    return u


# ------------------------------------------------------------
# main per-dataset processing
# ------------------------------------------------------------
def summarize_field(field: np.ndarray, valid_mask: np.ndarray) -> Dict[str, float]:
    vals = field[valid_mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "p10": np.nan,
            "p90": np.nan,
            "max": np.nan,
        }
    p10, p90 = np.percentile(vals, [10, 90])
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p10": float(p10),
        "p90": float(p90),
        "max": float(np.max(vals)),
    }


def save_qc_overlay(
    crystal_mask: np.ndarray,
    field: np.ndarray,
    out_png: Path,
    title: str,
):
    x0, y0, x1, y1 = mask_bbox(crystal_mask)
    field_crop, xx0, yy0 = crop_with_margin(field, x0, y0, x1, y1, margin=8)
    mask_crop, _, _ = crop_with_margin(crystal_mask.astype(float), x0, y0, x1, y1, margin=8)

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.imshow(mask_crop, cmap="gray", alpha=0.25, interpolation="nearest")
    im = ax.imshow(field_crop, cmap="viridis", interpolation="nearest")
    ax.set_title(title)
    ax.set_axis_off()
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def process_dataset(
    json_dir: Path,
    out_dir: Path,
    dataset_label: str,
    crystal_category_id: int,
    nucleus_category_id: int,
    defect_category_id: int,
    nucleus_mode: str,
    min_crystal_pixels: int,
    save_npz: bool,
    save_qc_pngs: bool,
    max_qc_per_dataset: int,
) -> pd.DataFrame:
    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"No JSON files found in {json_dir}")

    rows = []
    qc_count = 0

    fields_dir = out_dir / f"{dataset_label}_field_npz"
    qc_dir = out_dir / f"{dataset_label}_qc_png"
    if save_npz:
        fields_dir.mkdir(parents=True, exist_ok=True)
    if save_qc_pngs:
        qc_dir.mkdir(parents=True, exist_ok=True)

    for jf in json_files:
        stem = jf.stem

        crystals, nuclei, defects = parse_image_annotations(
            jf,
            crystal_category_id=crystal_category_id,
            nucleus_category_id=nucleus_category_id,
            defect_category_id=defect_category_id,
            assume_all_are_crystals_if_missing_category=True,
        )

        if not crystals:
            continue

        # decode optional shared masks first
        nuclei_masks = []
        defect_masks = []

        for _, ann in nuclei:
            try:
                nuclei_masks.append(decode_mask(ann["segmentation"]))
            except Exception:
                pass

        for _, ann in defects:
            try:
                defect_masks.append(decode_mask(ann["segmentation"]))
            except Exception:
                pass

        shared_shape = None
        if nuclei_masks:
            shared_shape = nuclei_masks[0].shape
        elif defect_masks:
            shared_shape = defect_masks[0].shape
        else:
            # will infer per crystal
            pass

        nucleus_union = union_masks(nuclei_masks, shared_shape) if shared_shape is not None else None
        defect_union = union_masks(defect_masks, shared_shape) if shared_shape is not None else None

        for ann_idx, ann in crystals:
            try:
                crystal_mask = decode_mask(ann["segmentation"])
            except Exception:
                continue

            if int(crystal_mask.sum()) < min_crystal_pixels:
                continue

            # if shared masks absent, create None; if present but shape mismatch, skip mismatch
            nuc_mask = nucleus_union
            def_mask = defect_union

            if nuc_mask is not None and nuc_mask.shape != crystal_mask.shape:
                nuc_mask = None
            if def_mask is not None and def_mask.shape != crystal_mask.shape:
                def_mask = None

            boundary_field = compute_boundary_distance_field(crystal_mask)
            defect_field = compute_defect_distance_field(crystal_mask, def_mask)
            nucleus_field = compute_nucleus_distance_field(crystal_mask, nuc_mask, nucleus_mode=nucleus_mode)

            grain_key = normalize_key(f"{stem}_{ann_idx}")

            boundary_stats = summarize_field(boundary_field, crystal_mask)
            defect_stats = summarize_field(defect_field, crystal_mask)
            nucleus_stats = summarize_field(nucleus_field, crystal_mask)

            rows.append(
                {
                    "dataset": dataset_label,
                    "json_file": jf.name,
                    "json_stem": stem,
                    "ann_index": int(ann_idx),
                    "file_name": grain_key,
                    "mask_name": normalize_key(get_mask_name(stem, ann, ann_idx)),
                    "crystal_area_px": int(crystal_mask.sum()),
                    "has_nucleus": int(nuc_mask is not None and np.any(nuc_mask)),
                    "has_defect": int(def_mask is not None and np.any(def_mask)),
                    "boundary_mean_px": boundary_stats["mean"],
                    "boundary_median_px": boundary_stats["median"],
                    "boundary_p10_px": boundary_stats["p10"],
                    "boundary_p90_px": boundary_stats["p90"],
                    "boundary_max_px": boundary_stats["max"],
                    "defectdist_mean_px": defect_stats["mean"],
                    "defectdist_median_px": defect_stats["median"],
                    "defectdist_p10_px": defect_stats["p10"],
                    "defectdist_p90_px": defect_stats["p90"],
                    "defectdist_max_px": defect_stats["max"],
                    "nucdist_mean_px": nucleus_stats["mean"],
                    "nucdist_median_px": nucleus_stats["median"],
                    "nucdist_p10_px": nucleus_stats["p10"],
                    "nucdist_p90_px": nucleus_stats["p90"],
                    "nucdist_max_px": nucleus_stats["max"],
                }
            )

            if save_npz:
                np.savez_compressed(
                    fields_dir / f"{grain_key}_local_fields.npz",
                    crystal_mask=crystal_mask.astype(np.uint8),
                    boundary_distance_px=boundary_field.astype(np.float32),
                    defect_distance_px=defect_field.astype(np.float32),
                    nucleus_distance_px=nucleus_field.astype(np.float32),
                )

            if save_qc_pngs and qc_count < max_qc_per_dataset:
                save_qc_overlay(
                    crystal_mask,
                    boundary_field,
                    qc_dir / f"{grain_key}_boundary.png",
                    f"{grain_key} | distance-to-boundary",
                )
                save_qc_overlay(
                    crystal_mask,
                    defect_field,
                    qc_dir / f"{grain_key}_defectdist.png",
                    f"{grain_key} | distance-to-defect",
                )
                save_qc_overlay(
                    crystal_mask,
                    nucleus_field,
                    qc_dir / f"{grain_key}_nucdist.png",
                    f"{grain_key} | distance-to-nucleus",
                )
                qc_count += 1

    if not rows:
        raise RuntimeError(f"No usable crystal grains found in {json_dir}")

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Phase 2 true local pixelwise fields: distance-to-boundary, distance-to-defect, distance-to-nucleus."
    )
    ap.add_argument("--fapi-json-dir", required=True)
    ap.add_argument("--tempo-json-dir", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--crystal-category-id", type=int, default=1)
    ap.add_argument("--nucleus-category-id", type=int, default=2)
    ap.add_argument("--defect-category-id", type=int, default=3)

    ap.add_argument("--nucleus-mode", choices=["mask", "centroid"], default="mask")
    ap.add_argument("--min-crystal-pixels", type=int, default=200)

    ap.add_argument("--save-npz", action="store_true")
    ap.add_argument("--save-qc-pngs", action="store_true")
    ap.add_argument("--max-qc-per-dataset", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fapi_df = process_dataset(
        json_dir=Path(args.fapi_json_dir),
        out_dir=out_dir,
        dataset_label="FAPI",
        crystal_category_id=args.crystal_category_id,
        nucleus_category_id=args.nucleus_category_id,
        defect_category_id=args.defect_category_id,
        nucleus_mode=args.nucleus_mode,
        min_crystal_pixels=args.min_crystal_pixels,
        save_npz=args.save_npz,
        save_qc_pngs=args.save_qc_pngs,
        max_qc_per_dataset=args.max_qc_per_dataset,
    )

    tempo_df = process_dataset(
        json_dir=Path(args.tempo_json_dir),
        out_dir=out_dir,
        dataset_label="FAPI-TEMPO",
        crystal_category_id=args.crystal_category_id,
        nucleus_category_id=args.nucleus_category_id,
        defect_category_id=args.defect_category_id,
        nucleus_mode=args.nucleus_mode,
        min_crystal_pixels=args.min_crystal_pixels,
        save_npz=args.save_npz,
        save_qc_pngs=args.save_qc_pngs,
        max_qc_per_dataset=args.max_qc_per_dataset,
    )

    fapi_csv = out_dir / "phase2_local_fields_FAPI.csv"
    tempo_csv = out_dir / "phase2_local_fields_FAPITEMPO.csv"
    summary_csv = out_dir / "phase2_local_fields_summary.csv"

    fapi_df.to_csv(fapi_csv, index=False)
    tempo_df.to_csv(tempo_csv, index=False)

    summary = pd.DataFrame(
        [
            {
                "dataset": "FAPI",
                "n_grains": len(fapi_df),
                "n_with_nucleus": int(fapi_df["has_nucleus"].sum()),
                "n_with_defect": int(fapi_df["has_defect"].sum()),
                "boundary_median_px_median": float(np.nanmedian(fapi_df["boundary_median_px"])),
                "defectdist_median_px_median": float(np.nanmedian(fapi_df["defectdist_median_px"])),
                "nucdist_median_px_median": float(np.nanmedian(fapi_df["nucdist_median_px"])),
            },
            {
                "dataset": "FAPI-TEMPO",
                "n_grains": len(tempo_df),
                "n_with_nucleus": int(tempo_df["has_nucleus"].sum()),
                "n_with_defect": int(tempo_df["has_defect"].sum()),
                "boundary_median_px_median": float(np.nanmedian(tempo_df["boundary_median_px"])),
                "defectdist_median_px_median": float(np.nanmedian(tempo_df["defectdist_median_px"])),
                "nucdist_median_px_median": float(np.nanmedian(tempo_df["nucdist_median_px"])),
            },
        ]
    )
    summary.to_csv(summary_csv, index=False)

    print("[OK] Wrote:")
    print(" ", fapi_csv)
    print(" ", tempo_csv)
    print(" ", summary_csv)
    if args.save_npz:
        print(" ", out_dir / "FAPI_field_npz")
        print(" ", out_dir / "FAPI-TEMPO_field_npz")
    if args.save_qc_pngs:
        print(" ", out_dir / "FAPI_qc_png")
        print(" ", out_dir / "FAPI-TEMPO_qc_png")


if __name__ == "__main__":
    main()