import argparse
import numpy as np
from PIL import Image
from skimage.morphology import binary_dilation, disk
from skimage.measure import label, regionprops
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt
import tifffile as tiff

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--handseg_img", required=True, help="RGB image with black boundaries + red nuclei")
    ap.add_argument("--out_tif", required=True, help="Output uint16 label map tif")
    ap.add_argument("--target_w", type=int, required=True)
    ap.add_argument("--target_h", type=int, required=True)
    ap.add_argument("--boundary_black_thr", type=int, default=60, help="RGB threshold for black lines")
    ap.add_argument("--red_r_min", type=int, default=150)
    ap.add_argument("--red_g_max", type=int, default=110)
    ap.add_argument("--red_b_max", type=int, default=110)
    ap.add_argument("--boundary_dilate_px", type=int, default=2)
    args = ap.parse_args()

    img = np.array(Image.open(args.handseg_img).convert("RGB"))
    h, w = img.shape[:2]

    # --- detect black boundaries ---
    boundary = (img[:,:,0] < args.boundary_black_thr) & (img[:,:,1] < args.boundary_black_thr) & (img[:,:,2] < args.boundary_black_thr)
    if args.boundary_dilate_px > 0:
        boundary = binary_dilation(boundary, disk(args.boundary_dilate_px))

    # --- detect red nuclei (connected components -> centroids) ---
    red = (img[:,:,0] >= args.red_r_min) & (img[:,:,1] <= args.red_g_max) & (img[:,:,2] <= args.red_b_max)
    red_lbl = label(red)
    props = [p for p in regionprops(red_lbl) if p.area >= 8]
    if len(props) < 5:
        raise RuntimeError(f"Too few red components detected ({len(props)}). Adjust thresholds.")

    seeds = np.zeros((h, w), dtype=np.int32)
    for i, p in enumerate(props, start=1):
        cy, cx = map(int, np.round(p.centroid))
        cy = np.clip(cy, 0, h-1)
        cx = np.clip(cx, 0, w-1)
        seeds[cy, cx] = i

    # --- watershed inside non-boundary region ---
    allowed = ~boundary
    dist = distance_transform_edt(allowed)
    labels = watershed(-dist, markers=seeds, mask=allowed).astype(np.uint16)

    # --- resize to target (NEAREST to preserve labels) ---
    lab_img = Image.fromarray(labels)
    lab_resized = lab_img.resize((args.target_w, args.target_h), resample=Image.NEAREST)
    labels_out = np.array(lab_resized, dtype=np.uint16)

    tiff.imwrite(args.out_tif, labels_out)
    print(f"[OK] Wrote ID map: {args.out_tif}  (shape={labels_out.shape}, max_id={labels_out.max()})")

if __name__ == "__main__":
    main()
