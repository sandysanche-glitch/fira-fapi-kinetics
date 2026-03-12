#!/usr/bin/env python3
"""
project_idmap_to_frame_jsons.py

Create per-frame COCO-style JSON mask files whose instance IDs match a FINAL-frame
spherulite ID map (seeded watershed result).

Typical use:
1) You already have per-frame segmentation JSONs (e.g., from SAM) for the SAME image size.
2) You have a final-frame ID map (uint16) where each nucleus/spherulite has a unique label.

This script:
- Builds a per-frame *foreground* mask by OR-ing all masks in the input frame JSON.
- Projects that foreground onto the final ID map: idmask = foreground * idmap_final
- Exports a new per-frame JSON where each item corresponds to one spherulite ID present in that frame.

Why this is useful:
- IDs are temporally consistent without manual segmentation of every frame.
- Fragmentation is reduced because each pixel is assigned by a global partition (the final map).

Notes:
- If your input frame JSONs contain COCO RLE with "counts" as a string (compressed), this script decodes them.
- Output uses compressed RLE strings (COCO-style) to match typical SAM exports.
- If sizes do not match, either regenerate the ID map at the correct size, or enable --resize_idmap.
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Any, Tuple

import numpy as np

try:
    import cv2  # optional (only used for resizing and simple morphology)
except Exception:
    cv2 = None

try:
    import tifffile  # optional (for reading .tif ID maps)
except Exception:
    tifffile = None


# -------------------------
# COCO RLE helpers (compressed)
# -------------------------

def rle_string_to_counts(s: str) -> List[int]:
    cnts: List[int] = []
    p = 0
    m = len(s)
    while p < m:
        x = 0
        k = 0
        more = 1
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1f) << (5 * k)
            more = c & 0x20
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        cnts.append(int(x))
    # undo delta encoding
    for i in range(2, len(cnts)):
        cnts[i] += cnts[i - 2]
    return cnts


def rle_counts_to_string(counts: List[int]) -> str:
    # delta encode
    cnts = list(map(int, counts))
    for i in range(len(cnts) - 1, 1, -1):
        cnts[i] -= cnts[i - 2]

    out_chars: List[str] = []
    for x in cnts:
        x = int(x)
        while True:
            c = x & 0x1f
            x >>= 5
            # termination condition (per COCO API)
            if (x == 0 and (c & 0x10) == 0) or (x == -1 and (c & 0x10) != 0):
                out_chars.append(chr(c + 48))
                break
            else:
                out_chars.append(chr((c | 0x20) + 48))
    return "".join(out_chars)


def rle_decode(rle: Dict[str, Any]) -> np.ndarray:
    """Decode COCO RLE (counts list or compressed string) into (H,W) uint8 mask."""
    h, w = rle["size"]
    counts = rle["counts"]
    if isinstance(counts, list):
        cnts = counts
    elif isinstance(counts, str):
        cnts = rle_string_to_counts(counts)
    else:
        raise TypeError(f"Unsupported RLE counts type: {type(counts)}")

    n = h * w
    flat = np.zeros(n, dtype=np.uint8)
    idx = 0
    val = 0
    for run in cnts:
        run = int(run)
        if run < 0:
            raise ValueError("Negative run length encountered in RLE.")
        end = idx + run
        if end > n:
            end = n
        if val == 1:
            flat[idx:end] = 1
        idx = end
        val ^= 1
        if idx >= n:
            break

    # COCO uses Fortran order (column-major)
    return flat.reshape((w, h), order="F").T


def rle_encode_compressed(mask: np.ndarray) -> Dict[str, Any]:
    """Encode (H,W) binary mask to COCO compressed RLE dict."""
    mask = (mask > 0).astype(np.uint8)
    h, w = mask.shape
    # flatten in Fortran order
    pixels = mask.T.flatten(order="F")

    counts: List[int] = []
    prev = 0
    run = 0
    for p in pixels:
        p = int(p)
        if p == prev:
            run += 1
        else:
            counts.append(run)
            run = 1
            prev = p
    counts.append(run)

    return {"size": [int(h), int(w)], "counts": rle_counts_to_string(counts)}


def mask_to_bbox(mask: np.ndarray) -> List[int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return [0, 0, 0, 0]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, int(x1 - x0 + 1), int(y1 - y0 + 1)]


# -------------------------
# IO helpers
# -------------------------

def load_idmap(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext in [".tif", ".tiff"]:
        if tifffile is None:
            raise RuntimeError("tifffile not installed. Either pip install tifffile, or provide a .npy ID map.")
        arr = tifffile.imread(path)
    else:
        raise ValueError("Unsupported idmap extension. Use .npy or .tif/.tiff")
    if arr.ndim != 2:
        raise ValueError("ID map must be 2D (H,W).")
    return arr.astype(np.uint16)


def resize_nearest(arr: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    th, tw = target_hw
    if arr.shape == (th, tw):
        return arr
    if cv2 is None:
        # fallback nearest-neighbor without cv2
        y_idx = (np.linspace(0, arr.shape[0] - 1, th)).round().astype(int)
        x_idx = (np.linspace(0, arr.shape[1] - 1, tw)).round().astype(int)
        return arr[y_idx][:, x_idx]
    return cv2.resize(arr, (tw, th), interpolation=cv2.INTER_NEAREST)


def union_foreground_from_frame_json(frame_json_path: str, min_area_px: int = 0) -> np.ndarray:
    """OR all instance masks in a per-frame JSON to get a foreground mask."""
    anns = json.load(open(frame_json_path, "r"))
    if not anns:
        return None
    h, w = anns[0]["segmentation"]["size"]
    fg = np.zeros((h, w), dtype=np.uint8)
    for a in anns:
        # fast filter on provided area if present
        area = a.get("area", None)
        if area is not None and min_area_px and int(area) < int(min_area_px):
            continue
        m = rle_decode(a["segmentation"])
        if min_area_px and int(m.sum()) < int(min_area_px):
            continue
        fg |= m
    return fg


def simple_morphology(mask: np.ndarray, close_px: int = 0, open_px: int = 0) -> np.ndarray:
    """Optional cleanup to reduce holes / speckle in the foreground."""
    if (close_px <= 0 and open_px <= 0) or cv2 is None:
        return mask
    out = mask.astype(np.uint8) * 255
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    if open_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    return (out > 0).astype(np.uint8)


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_json_glob", required=True, help='Glob for per-frame JSONs, e.g. "frames/frame_*.json"')
    ap.add_argument("--idmap_path", required=True, help="Final-frame spherulite ID map (.npy or .tif)")
    ap.add_argument("--out_dir", required=True, help="Output directory for idmapped per-frame JSONs")

    ap.add_argument("--min_area_px", type=int, default=0, help="Ignore tiny masks when building foreground union")
    ap.add_argument("--close_px", type=int, default=0, help="Optional MORPH_CLOSE kernel size (px) on foreground")
    ap.add_argument("--open_px", type=int, default=0, help="Optional MORPH_OPEN kernel size (px) on foreground")

    ap.add_argument("--resize_idmap", action="store_true",
                    help="If sizes differ, resize the ID map (nearest-neighbor) to match each frame JSON size.")
    ap.add_argument("--name_suffix", default="_idmapped", help="Suffix added to output json filenames")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    idmap = load_idmap(args.idmap_path)

    frame_paths = sorted(glob.glob(args.frame_json_glob))
    if not frame_paths:
        raise SystemExit(f"No frame JSONs found for glob: {args.frame_json_glob}")

    for fp in frame_paths:
        fg = union_foreground_from_frame_json(fp, min_area_px=args.min_area_px)
        if fg is None:
            continue

        # size matching
        if fg.shape != idmap.shape:
            if not args.resize_idmap:
                raise SystemExit(
                    f"Size mismatch for {os.path.basename(fp)}: frame {fg.shape} vs idmap {idmap.shape}. "
                    f"Regenerate the ID map at the correct size, or enable --resize_idmap."
                )
            idmap_use = resize_nearest(idmap, fg.shape)
        else:
            idmap_use = idmap

        fg = simple_morphology(fg, close_px=args.close_px, open_px=args.open_px)

        labeled = (fg.astype(np.uint16) * idmap_use.astype(np.uint16))
        labs = np.unique(labeled)
        labs = labs[labs != 0]

        # Build per-frame annotations: 1 entry per spherulite label present in this frame.
        anns_out: List[Dict[str, Any]] = []

        # Try to keep image_id consistent with your existing naming convention if possible
        # (If you don't care, this can be 1 always.)
        image_id = 1
        bn = os.path.basename(fp)
        # Example filename: frame_00460_t920.00ms.json  -> image_id=461
        try:
            if bn.startswith("frame_"):
                image_id = int(bn.split("_")[1]) + 1
        except Exception:
            image_id = 1

        for lab in labs:
            mask = (labeled == lab).astype(np.uint8)
            area = int(mask.sum())
            if area <= 0:
                continue
            anns_out.append({
                "id": int(lab),                 # spherulite ID (stable across frames)
                "image_id": int(image_id),
                "category_id": 1,
                "segmentation": rle_encode_compressed(mask),
                "area": area,
                "bbox": mask_to_bbox(mask),
                "iscrowd": 0
            })

        out_name = os.path.splitext(bn)[0] + args.name_suffix + ".json"
        out_path = os.path.join(args.out_dir, out_name)
        with open(out_path, "w") as f:
            json.dump(anns_out, f)

        print(f"Wrote {out_path}  (labels={len(anns_out)})")


if __name__ == "__main__":
    main()
