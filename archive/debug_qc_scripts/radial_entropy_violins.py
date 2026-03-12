# radial_entropy_violins.py
#
# Computes annular (inner/middle/outer) Shannon entropy of a HEATMAP (normalized 0..1)
# inside each grain mask, then aggregates per sample and plots paired violins for
# two datasets (FAPI-TEMPO vs FAPI).
#
# Input expected per sample:
#   - a JSON file that is EITHER:
#       (A) COCO dict with "annotations": [...]
#       (B) a top-level LIST of annotation dicts  (your case)
#     Each annotation must contain:
#       - "segmentation": {"size":[H,W], "counts": ...}  # COCO RLE (compressed string or list)
#     Optional keys:
#       - "score", "bbox", "mask_name", etc.
#
#   - a heatmap image file matching the JSON stem (best-effort):
#       e.g. FAPI_0.json  ↔  FAPI_0_heatmap.png  (or any image whose stem contains "FAPI_0")
#
# Outputs:
#   - radial_entropy_results.csv (sample-level)
#   - heatmaps_norm01/<GROUP>/<SAMPLE>_heatmap_norm01.png (per-image normalized heatmaps)
#   - radial_entropy_violins.png (inner/middle/outer; entropy normalized 0..1)
#
# Dependencies:
#   pip install numpy pandas pillow matplotlib pycocotools
#   (Windows alt): pip install pycocotools-windows

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# --- REQUIRED for COCO RLE decoding ---
try:
    from pycocotools import mask as mask_utils
except Exception as e:
    raise SystemExit(
        "pycocotools is required to decode COCO RLE masks.\n"
        "Try:\n"
        "  pip install pycocotools\n"
        "or on Windows:\n"
        "  pip install pycocotools-windows\n"
        f"Import error: {e}"
    )


# -----------------------------
# Core math helpers
# -----------------------------
def norm01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def shannon_entropy_from_vals(vals01: np.ndarray, nbins: int) -> Tuple[float, float]:
    """
    Histogram-based Shannon entropy over [0,1].
    Returns (entropy_raw, entropy_norm01) where entropy_norm01 = entropy_raw / log(nbins).
    """
    if vals01.size == 0:
        return (np.nan, np.nan)

    vals01 = np.clip(vals01.astype(np.float32), 0, 1)
    hist, _ = np.histogram(vals01, bins=nbins, range=(0.0, 1.0))
    s = int(hist.sum())
    if s <= 0:
        return (np.nan, np.nan)

    p = hist.astype(np.float64) / float(s)
    p = p[p > 0]
    ent = float(-(p * np.log(p)).sum())

    ent_max = float(np.log(nbins)) if nbins > 1 else np.nan
    ent_norm = float(ent / ent_max) if np.isfinite(ent_max) and ent_max > 0 else np.nan
    return ent, ent_norm


# -----------------------------
# IO helpers
# -----------------------------
def load_annotations_any(json_path: Path) -> List[dict]:
    """
    Supports:
      - dict with key "annotations"
      - top-level list of annotations
    """
    with json_path.open("r", encoding="utf-8") as f:
        d = json.load(f)

    if isinstance(d, dict):
        anns = d.get("annotations", [])
        return anns if isinstance(anns, list) else []

    if isinstance(d, list):
        return [x for x in d if isinstance(x, dict)]

    return []


def load_heatmap_gray01(img_path: Path) -> Tuple[np.ndarray, float, float]:
    """
    Load image -> grayscale float32 -> min-max normalize to 0..1.
    Returns (heat01, min_before, max_before).
    """
    im = Image.open(img_path)
    # Many of your heatmaps are RGB; convert to L
    if im.mode not in ("L", "I;16", "I"):
        im = im.convert("L")
    arr = np.array(im).astype(np.float32)
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    return norm01(arr), mn, mx


def save_heatmap_gray01_png(heat01: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray((np.clip(heat01, 0, 1) * 255).astype(np.uint8), mode="L")
    im.save(out_path)


def decode_segmentation_to_mask(seg: Any) -> Optional[np.ndarray]:
    """
    Supports COCO RLE dict:
        {"size":[H,W], "counts": "...."} or counts as list.
    Returns mask (H,W) uint8, or None if unsupported.
    """
    if not isinstance(seg, dict):
        return None
    if "size" not in seg or "counts" not in seg:
        return None

    try:
        rle = {"size": seg["size"], "counts": seg["counts"]}
        m = mask_utils.decode(rle)  # (H,W,1) or (H,W)
        if m.ndim == 3:
            m = m[:, :, 0]
        return (m > 0).astype(np.uint8)
    except Exception:
        return None


def extract_pair_id(sample_id: str) -> str:
    """
    Pairing heuristic: last integer in sample_id.
    Example: FAPI_12 -> "12"
    """
    m = re.findall(r"(\d+)", sample_id)
    return m[-1] if m else ""


# -----------------------------
# Pair JSON ↔ heatmap image
# -----------------------------
def build_image_index(root: Path) -> Dict[str, Path]:
    imgs = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        imgs.extend(root.rglob(ext))

    idx: Dict[str, Path] = {}
    for p in imgs:
        stem = p.stem.lower()
        # include common normalized keys
        idx[stem] = p
        idx[re.sub(r"(_|-)?heatmap$", "", stem)] = p
    return idx


def find_heatmap_for_sample(sample_id: str, img_index: Dict[str, Path]) -> Optional[Path]:
    sid = (sample_id or "").strip().lower()
    if not sid:
        return None

    # 1) direct
    if sid in img_index:
        return img_index[sid]

    # 2) common pattern: <sid>_heatmap
    k = f"{sid}_heatmap"
    if k in img_index:
        return img_index[k]

    # 3) contains
    for stem, p in img_index.items():
        if sid in stem:
            return p

    return None


# -----------------------------
# Entropy per grain annuli
# -----------------------------
def annular_entropies(mask: np.ndarray, heat01: np.ndarray, annuli: Dict[str, Tuple[float, float]], nbins: int) -> Dict[str, Tuple[float, float]]:
    """
    Compute entropy in inner/middle/outer annuli for one grain.
    Annuli are based on radius normalized to rmax within the mask from centroid.
    """
    ys, xs = np.nonzero(mask)
    if ys.size < 10:
        return {k: (np.nan, np.nan) for k in annuli.keys()}

    cy = float(ys.mean())
    cx = float(xs.mean())
    dy = ys.astype(np.float32) - cy
    dx = xs.astype(np.float32) - cx
    r = np.sqrt(dx * dx + dy * dy)
    rmax = float(r.max())
    if rmax <= 1e-6:
        return {k: (np.nan, np.nan) for k in annuli.keys()}

    rn = r / rmax
    vals = heat01[ys, xs]

    out: Dict[str, Tuple[float, float]] = {}
    for name, (a, b) in annuli.items():
        sel = (rn >= a) & (rn < b)
        out[name] = shannon_entropy_from_vals(vals[sel], nbins=nbins)
    return out


# -----------------------------
# Plotting
# -----------------------------
def plot_three_paired_violins(df: pd.DataFrame, out_png: Path, group_col="group", pair_col="pair_id") -> None:
    """
    3 panels (inner/middle/outer) using entropy_*_norm01 columns.
    """
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    needed = ["entropy_inner_norm01", "entropy_middle_norm01", "entropy_outer_norm01", group_col]
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}'. Available: {df.columns.tolist()}")

    df = df.dropna(subset=[group_col]).copy()
    if df.empty:
        print("[WARN] No data to plot.")
        return

    groups = sorted(df[group_col].astype(str).unique().tolist())
    if len(groups) < 2:
        print(f"[WARN] Need at least 2 groups for paired violins; found {groups}.")
        return

    g0, g1 = groups[0], groups[1]
    annuli = [("Inner", "entropy_inner_norm01"),
              ("Middle", "entropy_middle_norm01"),
              ("Outer", "entropy_outer_norm01")]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    rng = np.random.default_rng(0)
    for ax, (title, col) in zip(axes, annuli):
        d0 = df[df[group_col].astype(str) == g0][col].astype(float).dropna().values
        d1 = df[df[group_col].astype(str) == g1][col].astype(float).dropna().values

        pos0, pos1 = 0.85, 1.15
        ax.violinplot([d0, d1], positions=[pos0, pos1], widths=0.25,
                      showmeans=False, showmedians=True, showextrema=False)

        ax.scatter(pos0 + rng.normal(0, 0.03, size=len(d0)), d0, s=12, alpha=0.7)
        ax.scatter(pos1 + rng.normal(0, 0.03, size=len(d1)), d1, s=12, alpha=0.7)

        # paired lines if pair ids overlap
        if pair_col in df.columns and df[pair_col].astype(str).str.len().gt(0).any():
            piv = df.pivot_table(index=pair_col, columns=group_col, values=col, aggfunc="mean")
            if g0 in piv.columns and g1 in piv.columns:
                for _, row in piv.iterrows():
                    y0 = row.get(g0, np.nan)
                    y1 = row.get(g1, np.nan)
                    if np.isfinite(y0) and np.isfinite(y1):
                        ax.plot([pos0, pos1], [y0, y1], alpha=0.25, linewidth=1)

        ax.set_title(title)
        ax.set_xticks([pos0, pos1])
        ax.set_xticklabels([g0, g1])
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("Entropy (normalized 0..1)")
    fig.suptitle("Annular entropy (inner / middle / outer) from heatmap (0..1 normalized)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved plot: {out_png}")


# -----------------------------
# Dataset processing
# -----------------------------
def process_dataset(
    dataset_root: Path,
    group_name: str,
    out_heatmap_root: Path,
    nbins: int,
    annuli: Dict[str, Tuple[float, float]],
    agg: str,
) -> pd.DataFrame:
    jsons = sorted(dataset_root.rglob("*.json"))
    img_index = build_image_index(dataset_root)

    print(f"[INFO] {group_name}: found {len(jsons)} JSON files under {dataset_root}")

    rows: List[Dict[str, Any]] = []
    for i, jp in enumerate(jsons, start=1):
        sample_id = jp.stem
        hp = find_heatmap_for_sample(sample_id, img_index)
        if hp is None:
            continue

        # heatmap -> gray 0..1 (per image)
        heat01, hmn, hmx = load_heatmap_gray01(hp)
        heat_out = out_heatmap_root / group_name / f"{sample_id}_heatmap_norm01.png"
        save_heatmap_gray01_png(heat01, heat_out)

        anns = load_annotations_any(jp)
        if not anns:
            continue

        # treat ALL annotations with valid RLE segmentation as grains
        per_grain_raw = {k: [] for k in annuli.keys()}
        per_grain_n01 = {k: [] for k in annuli.keys()}
        n_used = 0

        for a in anns:
            seg = a.get("segmentation", None)
            m = decode_segmentation_to_mask(seg)
            if m is None:
                continue
            if m.shape != heat01.shape:
                # skip mismatched masks
                continue

            ents = annular_entropies(m, heat01, annuli=annuli, nbins=nbins)
            n_used += 1
            for name in annuli.keys():
                eraw, en01 = ents[name]
                per_grain_raw[name].append(eraw)
                per_grain_n01[name].append(en01)

        if n_used == 0:
            continue

        def aggregate(vals: List[float]) -> float:
            x = np.asarray(vals, dtype=float)
            x = x[np.isfinite(x)]
            if x.size == 0:
                return np.nan
            if agg == "mean":
                return float(np.nanmean(x))
            return float(np.nanmedian(x))

        row = {
            "group": group_name,
            "sample_id": sample_id,
            "pair_id": extract_pair_id(sample_id),
            "json_path": str(jp),
            "heatmap_path": str(hp),
            "heatmap_norm01_path": str(heat_out),
            "heatmap_min_before": hmn,
            "heatmap_max_before": hmx,
            "n_grains_used": int(n_used),
        }

        for name in annuli.keys():
            row[f"entropy_{name}_raw"] = aggregate(per_grain_raw[name])
            row[f"entropy_{name}_norm01"] = aggregate(per_grain_n01[name])

        rows.append(row)

        if i % 200 == 0:
            print(f"[INFO] {group_name}: processed {i}/{len(jsons)} JSONs...")

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fapi_tempo", type=str, default=r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\comparative datasets\FAPI-TEMPO")
    ap.add_argument("--fapi", type=str, default=r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\comparative datasets\FAPI")
    ap.add_argument("--out_csv", type=str, default="radial_entropy_results.csv")
    ap.add_argument("--out_png", type=str, default="radial_entropy_violins.png")
    ap.add_argument("--out_heatmaps", type=str, default="heatmaps_norm01")
    ap.add_argument("--nbins", type=int, default=32)
    ap.add_argument("--agg", type=str, default="median", choices=["median", "mean"])
    args = ap.parse_args()

    # three annuli (inner/middle/outer)
    annuli = {
        "inner":  (0.0, 1.0 / 3.0),
        "middle": (1.0 / 3.0, 2.0 / 3.0),
        "outer":  (2.0 / 3.0, 1.0000001),
    }

    out_heat_root = Path(args.out_heatmaps)
    out_heat_root.mkdir(parents=True, exist_ok=True)

    df_a = process_dataset(Path(args.fapi_tempo), "FAPI-TEMPO", out_heat_root, args.nbins, annuli, args.agg)
    df_b = process_dataset(Path(args.fapi),       "FAPI",       out_heat_root, args.nbins, annuli, args.agg)

    df = pd.concat([df_a, df_b], ignore_index=True)

    # ensure headers even if empty
    expected_cols = [
        "group","sample_id","pair_id","json_path","heatmap_path","heatmap_norm01_path",
        "heatmap_min_before","heatmap_max_before","n_grains_used",
        "entropy_inner_raw","entropy_middle_raw","entropy_outer_raw",
        "entropy_inner_norm01","entropy_middle_norm01","entropy_outer_norm01",
    ]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = np.nan

    out_csv = Path(args.out_csv)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Saved results to: {out_csv.resolve()}")

    if df.empty:
        print("[WARN] No rows produced. Likely heatmaps weren't matched to JSON stems. Check naming.")
        return

    # summaries
    print("\n[Summary: entropy_*_norm01]")
    for a in ["inner", "middle", "outer"]:
        col = f"entropy_{a}_norm01"
        s = df.groupby("group")[col].agg(["count", "mean", "median", "std", "min", "max"])
        print(f"\n--- {col} ---")
        print(s.to_string())

    # plot
    out_png = Path(args.out_png)
    plot_three_paired_violins(df, out_png=out_png)


if __name__ == "__main__":
    main()
