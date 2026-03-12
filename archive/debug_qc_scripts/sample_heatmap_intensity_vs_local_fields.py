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
from PIL import Image
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


def is_category(ann: Dict[str, Any], target: int, missing_means_true: bool = False) -> bool:
    if "category_id" not in ann:
        return missing_means_true
    try:
        return int(ann["category_id"]) == int(target)
    except Exception:
        return False


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


def load_heatmap(path: Path) -> np.ndarray:
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 3:
        # if RGB, convert to grayscale intensity
        arr = arr[..., :3].mean(axis=2)
    return arr.astype(np.float32)


# ------------------------------------------------------------
# local fields
# ------------------------------------------------------------
def compute_boundary_distance_field(crystal_mask: np.ndarray) -> np.ndarray:
    d = distance_transform_edt(crystal_mask)
    out = np.full(crystal_mask.shape, np.nan, dtype=np.float32)
    out[crystal_mask] = d[crystal_mask].astype(np.float32)
    return out


def compute_defect_distance_field(crystal_mask: np.ndarray, defect_mask: np.ndarray | None) -> np.ndarray:
    out = np.full(crystal_mask.shape, np.nan, dtype=np.float32)
    if defect_mask is None or not np.any(defect_mask):
        return out
    d = distance_transform_edt(~defect_mask)
    out[crystal_mask] = d[crystal_mask].astype(np.float32)
    return out


def compute_nucleus_distance_field(
    crystal_mask: np.ndarray,
    nucleus_mask: np.ndarray | None,
    nucleus_mode: str = "mask",
) -> np.ndarray:
    out = np.full(crystal_mask.shape, np.nan, dtype=np.float32)
    if nucleus_mask is None or not np.any(nucleus_mask):
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

    raise ValueError("nucleus_mode must be 'mask' or 'centroid'")


# ------------------------------------------------------------
# heatmap pairing
# ------------------------------------------------------------
def find_heatmap_for_json(jf: Path, search_dir: Path | None = None) -> Path | None:
    stem = jf.stem
    candidates = [
        jf.with_name(f"{stem}_heatmap.png"),
        jf.with_name(f"{stem}.png"),
        jf.with_name(f"{stem}_heatmap.jpg"),
        jf.with_name(f"{stem}.jpg"),
    ]
    if search_dir is not None:
        candidates.extend(
            [
                search_dir / f"{stem}_heatmap.png",
                search_dir / f"{stem}.png",
                search_dir / f"{stem}_heatmap.jpg",
                search_dir / f"{stem}.jpg",
            ]
        )
    for c in candidates:
        if c.exists():
            return c
    return None


# ------------------------------------------------------------
# sampling
# ------------------------------------------------------------
def parse_image_annotations(
    jf: Path,
    crystal_category_id: int = 1,
    nucleus_category_id: int = 2,
    defect_category_id: int = 3,
    assume_all_are_crystals_if_missing_category: bool = True,
) -> Tuple[List[Tuple[int, Dict[str, Any]]], List[Tuple[int, Dict[str, Any]]], List[Tuple[int, Dict[str, Any]]]]:
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


def sample_dataset(
    json_dir: Path,
    dataset_label: str,
    heatmap_dir: Path | None,
    nucleus_mode: str,
    crystal_category_id: int,
    nucleus_category_id: int,
    defect_category_id: int,
    min_crystal_pixels: int,
    max_pixels_per_grain: int,
) -> pd.DataFrame:
    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"No JSON files found in {json_dir}")

    rows = []

    for jf in json_files:
        heatmap_path = find_heatmap_for_json(jf, heatmap_dir)
        if heatmap_path is None:
            continue

        heatmap = load_heatmap(heatmap_path)
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

        nucleus_union = union_masks(nuclei_masks, shared_shape) if shared_shape is not None else None
        defect_union = union_masks(defect_masks, shared_shape) if shared_shape is not None else None

        for ann_idx, ann in crystals:
            try:
                crystal_mask = decode_mask(ann["segmentation"])
            except Exception:
                continue

            if crystal_mask.shape != heatmap.shape:
                continue

            if int(crystal_mask.sum()) < min_crystal_pixels:
                continue

            nuc_mask = nucleus_union if nucleus_union is not None and nucleus_union.shape == crystal_mask.shape else None
            def_mask = defect_union if defect_union is not None and defect_union.shape == crystal_mask.shape else None

            boundary_field = compute_boundary_distance_field(crystal_mask)
            defect_field = compute_defect_distance_field(crystal_mask, def_mask)
            nucleus_field = compute_nucleus_distance_field(crystal_mask, nuc_mask, nucleus_mode=nucleus_mode)

            ys, xs = np.nonzero(crystal_mask)
            if len(xs) == 0:
                continue

            # optional downsampling for file size control
            if len(xs) > max_pixels_per_grain:
                idx = np.random.choice(len(xs), size=max_pixels_per_grain, replace=False)
                ys = ys[idx]
                xs = xs[idx]

            grain_key = normalize_key(f"{stem}_{ann_idx}")

            for y, x in zip(ys, xs):
                rows.append(
                    {
                        "dataset": dataset_label,
                        "json_stem": stem,
                        "file_name": grain_key,
                        "heatmap_path": heatmap_path.name,
                        "pixel_x": int(x),
                        "pixel_y": int(y),
                        "intensity": float(heatmap[y, x]),
                        "dist_boundary_px": float(boundary_field[y, x]) if np.isfinite(boundary_field[y, x]) else np.nan,
                        "dist_defect_px": float(defect_field[y, x]) if np.isfinite(defect_field[y, x]) else np.nan,
                        "dist_nucleus_px": float(nucleus_field[y, x]) if np.isfinite(nucleus_field[y, x]) else np.nan,
                    }
                )

    if not rows:
        raise RuntimeError(f"No sampled pixels produced for {dataset_label}")
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# binning and plotting
# ------------------------------------------------------------
def binned_stats(df: pd.DataFrame, xcol: str, ycol: str, nbins: int) -> pd.DataFrame:
    x = pd.to_numeric(df[xcol], errors="coerce")
    y = pd.to_numeric(df[ycol], errors="coerce")
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m].to_numpy(float)
    y = y[m].to_numpy(float)

    if x.size == 0:
        return pd.DataFrame(columns=["bin_center", "n", "mean", "median", "q25", "q75"])

    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmin == xmax:
        xmin -= 1e-6
        xmax += 1e-6

    edges = np.linspace(xmin, xmax, nbins + 1)
    out = []

    for i in range(nbins):
        if i < nbins - 1:
            sel = (x >= edges[i]) & (x < edges[i + 1])
        else:
            sel = (x >= edges[i]) & (x <= edges[i + 1])

        yy = y[sel]
        if yy.size == 0:
            out.append(
                {"bin_center": 0.5 * (edges[i] + edges[i + 1]), "n": 0, "mean": np.nan, "median": np.nan, "q25": np.nan, "q75": np.nan}
            )
            continue

        q25, q75 = np.percentile(yy, [25, 75])
        out.append(
            {
                "bin_center": 0.5 * (edges[i] + edges[i + 1]),
                "n": int(yy.size),
                "mean": float(np.mean(yy)),
                "median": float(np.median(yy)),
                "q25": float(q25),
                "q75": float(q75),
            }
        )

    return pd.DataFrame(out)


def plot_two_datasets(
    A: pd.DataFrame,
    B: pd.DataFrame,
    xlabel: str,
    title: str,
    out_png: Path,
    labels=("FAPI", "FAPI-TEMPO"),
):
    fig, ax = plt.subplots(figsize=(5.8, 4.4))

    ax.plot(A["bin_center"], A["median"], lw=2.2, label=labels[0])
    ax.fill_between(A["bin_center"], A["q25"], A["q75"], alpha=0.18)

    ax.plot(B["bin_center"], B["median"], lw=2.2, label=labels[1])
    ax.fill_between(B["bin_center"], B["q25"], B["q75"], alpha=0.18)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Heatmap intensity")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Sample heatmap intensity against distance-to-boundary / defect / nucleus fields."
    )
    ap.add_argument("--fapi-json-dir", required=True)
    ap.add_argument("--tempo-json-dir", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--fapi-heatmap-dir", default=None, help="Optional separate heatmap directory")
    ap.add_argument("--tempo-heatmap-dir", default=None, help="Optional separate heatmap directory")

    ap.add_argument("--nucleus-mode", choices=["mask", "centroid"], default="mask")
    ap.add_argument("--crystal-category-id", type=int, default=1)
    ap.add_argument("--nucleus-category-id", type=int, default=2)
    ap.add_argument("--defect-category-id", type=int, default=3)

    ap.add_argument("--min-crystal-pixels", type=int, default=200)
    ap.add_argument("--max-pixels-per-grain", type=int, default=5000)
    ap.add_argument("--nbins", type=int, default=30)
    ap.add_argument("--save-raw", action="store_true", help="Save raw sampled per-pixel tables")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fapi_raw = sample_dataset(
        json_dir=Path(args.fapi_json_dir),
        dataset_label="FAPI",
        heatmap_dir=Path(args.fapi_heatmap_dir) if args.fapi_heatmap_dir else None,
        nucleus_mode=args.nucleus_mode,
        crystal_category_id=args.crystal_category_id,
        nucleus_category_id=args.nucleus_category_id,
        defect_category_id=args.defect_category_id,
        min_crystal_pixels=args.min_crystal_pixels,
        max_pixels_per_grain=args.max_pixels_per_grain,
    )

    tempo_raw = sample_dataset(
        json_dir=Path(args.tempo_json_dir),
        dataset_label="FAPI-TEMPO",
        heatmap_dir=Path(args.tempo_heatmap_dir) if args.tempo_heatmap_dir else None,
        nucleus_mode=args.nucleus_mode,
        crystal_category_id=args.crystal_category_id,
        nucleus_category_id=args.nucleus_category_id,
        defect_category_id=args.defect_category_id,
        min_crystal_pixels=args.min_crystal_pixels,
        max_pixels_per_grain=args.max_pixels_per_grain,
    )

    if args.save_raw:
        fapi_raw.to_csv(out_dir / "intensity_vs_local_fields_raw_FAPI.csv", index=False)
        tempo_raw.to_csv(out_dir / "intensity_vs_local_fields_raw_FAPITEMPO.csv", index=False)

    # Binned curves
    descriptors = [
        ("dist_boundary_px", "Distance to boundary (px)", "Heatmap intensity vs distance-to-boundary", "boundary"),
        ("dist_defect_px", "Distance to defect (px)", "Heatmap intensity vs distance-to-defect", "defect"),
        ("dist_nucleus_px", "Distance to nucleus (px)", "Heatmap intensity vs distance-to-nucleus", "nucleus"),
    ]

    summary_rows = []

    for xcol, xlabel, title, tag in descriptors:
        fapi_b = binned_stats(fapi_raw, xcol, "intensity", args.nbins)
        tempo_b = binned_stats(tempo_raw, xcol, "intensity", args.nbins)

        fapi_b.insert(0, "dataset", "FAPI")
        tempo_b.insert(0, "dataset", "FAPI-TEMPO")

        out_csv = out_dir / f"binned_intensity_vs_{tag}.csv"
        pd.concat([fapi_b, tempo_b], ignore_index=True).to_csv(out_csv, index=False)

        plot_two_datasets(
            A=fapi_b,
            B=tempo_b,
            xlabel=xlabel,
            title=title,
            out_png=out_dir / f"binned_intensity_vs_{tag}.png",
        )

        summary_rows.append(
            {
                "descriptor": tag,
                "FAPI_n_valid_pixels": int(np.isfinite(pd.to_numeric(fapi_raw[xcol], errors="coerce")).sum()),
                "FAPITEMPO_n_valid_pixels": int(np.isfinite(pd.to_numeric(tempo_raw[xcol], errors="coerce")).sum()),
                "csv": out_csv.name,
                "png": f"binned_intensity_vs_{tag}.png",
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "binned_intensity_vs_local_fields_summary.csv", index=False)

    print("[OK] Wrote:")
    if args.save_raw:
        print(" ", out_dir / "intensity_vs_local_fields_raw_FAPI.csv")
        print(" ", out_dir / "intensity_vs_local_fields_raw_FAPITEMPO.csv")
    for _, _, _, tag in descriptors:
        print(" ", out_dir / f"binned_intensity_vs_{tag}.csv")
        print(" ", out_dir / f"binned_intensity_vs_{tag}.png")
    print(" ", out_dir / "binned_intensity_vs_local_fields_summary.csv")


if __name__ == "__main__":
    main()