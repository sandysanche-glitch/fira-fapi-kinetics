#!/usr/bin/env python3
import argparse, glob, json, os
import numpy as np

import tifffile

try:
    from pycocotools import mask as mask_utils
except Exception as e:
    raise RuntimeError(
        "pycocotools is required. Install inside samcuda env:\n"
        "  pip install pycocotools\n"
    ) from e


def load_anns(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "annotations" in data:
        return data["annotations"]
    if isinstance(data, list):
        return data
    return []


def decode_rle(seg):
    return mask_utils.decode(seg).astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idmap_tif", required=True)
    ap.add_argument("--in_glob", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_area_px", type=float, default=800.0)
    ap.add_argument("--min_overlap_frac", type=float, default=0.30,
                    help="Mask pixels that fall on nonzero idmap labels / mask pixels")
    ap.add_argument("--min_purity", type=float, default=0.0,
                    help="Optional: overlap with chosen label / mask pixels (boundary-crossing filter). Try 0.6-0.8.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    idmap = tifffile.imread(args.idmap_tif)
    if idmap.ndim != 2:
        raise ValueError("ID map must be 2D.")
    H, W = idmap.shape

    files = sorted(glob.glob(args.in_glob))
    if not files:
        raise FileNotFoundError(f"No files match: {args.in_glob}")

    for fp in files:
        anns = load_anns(fp)
        out = []

        for a in anns:
            seg = a.get("segmentation", None)
            if not isinstance(seg, dict) or "counts" not in seg:
                continue

            area = float(a.get("area", 0.0))
            if area < args.min_area_px:
                continue

            m = decode_rle(seg)
            if m.shape != (H, W):
                # size mismatch means wrong idmap size or wrong json size
                continue

            ids = idmap[m]
            mask_px = int(m.sum())
            if mask_px <= 0:
                continue

            ids_nz = ids[ids > 0]
            if ids_nz.size == 0:
                continue

            overlap_frac = float(ids_nz.size) / float(mask_px)
            if overlap_frac < args.min_overlap_frac:
                continue

            # choose most frequent label within the mask
            binc = np.bincount(ids_nz.astype(np.int64))
            gid = int(binc.argmax())
            purity = float(binc[gid]) / float(mask_px)

            if purity < args.min_purity:
                continue

            a2 = dict(a)
            a2["id"] = gid
            a2["idmap_id"] = gid
            a2["overlap_frac"] = overlap_frac
            a2["purity"] = purity
            out.append(a2)

        base = os.path.basename(fp).replace(".json", "_idmapped.json")
        out_path = os.path.join(args.out_dir, base)
        if (not args.overwrite) and os.path.exists(out_path):
            continue
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f)

    print("[OK] Wrote idmapped JSONs to:", args.out_dir)


if __name__ == "__main__":
    main()
