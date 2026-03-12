import os, sys, json, time, math, argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from shapely.geometry import Polygon
    HAS_SHAPELY = True
except Exception:
    HAS_SHAPELY = False


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def find_frame_json(json_dir: str, frame_idx: int) -> Optional[str]:
    # expects filenames like frame_00050_..._idmapped.json
    prefix = f"frame_{frame_idx:05d}_"
    hits = [f for f in os.listdir(json_dir) if f.startswith(prefix) and f.endswith("_idmapped.json")]
    if not hits:
        return None
    # deterministic pick
    hits.sort()
    return os.path.join(json_dir, hits[0])

def bbox_center(b: List[float]) -> Tuple[float, float]:
    x, y, w, h = b
    return (x + 0.5 * w, y + 0.5 * h)

def bbox_iou(b1: List[float], b2: List[float]) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa1, ya1, xa2, ya2 = x1, y1, x1 + w1, y1 + h1
    xb1, yb1, xb2, yb2 = x2, y2, x2 + w2, y2 + h2
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a1 = w1 * h1
    a2 = w2 * h2
    union = a1 + a2 - inter
    return float(inter / union) if union > 0 else 0.0

def rle_to_polygon(seg: Any) -> Optional[Polygon]:
    # Your JSON has COCO RLE in seg['counts']. Converting RLE to polygon robustly
    # requires pycocotools. If you don't have it, polygon IoU cannot be computed.
    return None

def poly_iou_from_rle(segA: Any, segB: Any) -> Optional[float]:
    # If you have pycocotools installed, do exact mask IoU.
    # Otherwise, skip poly IoU (fallback to bbox IoU).
    try:
        from pycocotools import mask as maskUtils
    except Exception:
        return None

    try:
        rleA = segA
        rleB = segB
        # Ensure proper dict format
        if isinstance(rleA, dict) and "counts" in rleA and "size" in rleA:
            pass
        else:
            return None
        if isinstance(rleB, dict) and "counts" in rleB and "size" in rleB:
            pass
        else:
            return None

        # maskUtils.iou expects list of RLEs
        i = maskUtils.iou([rleA], [rleB], [0])
        return float(i[0][0])
    except Exception:
        return None


# -----------------------------
# Offset estimation (robust)
# -----------------------------
def estimate_offsets(
    tracks: pd.DataFrame,
    json_dir: str,
    out_csv: str,
    min_tracks: int,
    smooth_window: int,
    use_prev_if_empty: bool,
    match_radius_px: float = 250.0,
    max_pairs_per_frame: int = 300,
) -> pd.DataFrame:
    """
    Robustly estimate mapping between track (cx,cy) and JSON bbox centers.

    Convention used here:
      track_xy ≈ json_xy + offset
    so:
      json_xy = track_xy - offset

    We estimate offset per frame using nearest-neighbor pairing (after coarse median).
    """
    frames = sorted(tracks["frame_idx"].unique().tolist())
    rows = []

    prev_dx, prev_dy = 0.0, 0.0

    for f in frames:
        sub = tracks[tracks["frame_idx"] == f]
        ntr = len(sub)
        jf = find_frame_json(json_dir, int(f))
        if jf is None:
            if use_prev_if_empty:
                rows.append((f, prev_dx, prev_dy, ntr, 0, "missing_json_reuse_prev"))
            else:
                rows.append((f, 0.0, 0.0, ntr, 0, "missing_json_zero"))
            continue

        try:
            obj = json.load(open(jf, "r"))
        except Exception:
            if use_prev_if_empty:
                rows.append((f, prev_dx, prev_dy, ntr, 0, "bad_json_reuse_prev"))
            else:
                rows.append((f, 0.0, 0.0, ntr, 0, "bad_json_zero"))
            continue

        if not isinstance(obj, list) or len(obj) == 0 or ntr < min_tracks:
            if use_prev_if_empty:
                rows.append((f, prev_dx, prev_dy, ntr, len(obj) if isinstance(obj, list) else 0, "reuse_prev_low_tracks_or_empty"))
            else:
                rows.append((f, 0.0, 0.0, ntr, len(obj) if isinstance(obj, list) else 0, "zero_low_tracks_or_empty"))
            continue

        # Build arrays
        tx = sub["cx"].to_numpy(dtype=float)
        ty = sub["cy"].to_numpy(dtype=float)

        bx = np.array([bbox_center(a["bbox"])[0] for a in obj], dtype=float)
        by = np.array([bbox_center(a["bbox"])[1] for a in obj], dtype=float)

        # coarse offset via medians
        coarse_dx = float(np.median(tx) - np.median(bx))
        coarse_dy = float(np.median(ty) - np.median(by))

        # refine: pair track points to nearest json centers after applying coarse
        # predicted json = track - coarse_offset
        jx_pred = tx - coarse_dx
        jy_pred = ty - coarse_dy

        # brute-force nearest (robust, OK for your sizes); cap pairs
        pairs = []
        for i in range(min(len(jx_pred), max_pairs_per_frame)):
            dxs = bx - jx_pred[i]
            dys = by - jy_pred[i]
            d2 = dxs * dxs + dys * dys
            j = int(np.argmin(d2))
            d = float(math.sqrt(d2[j]))
            if d <= match_radius_px:
                # offset = track - json
                pairs.append((tx[i] - bx[j], ty[i] - by[j]))

        if len(pairs) < max(5, min_tracks // 5):
            # fallback: use coarse
            dx_f, dy_f = coarse_dx, coarse_dy
            note = f"coarse_only_pairs={len(pairs)}"
        else:
            dx_f = float(np.median([p[0] for p in pairs]))
            dy_f = float(np.median([p[1] for p in pairs]))
            note = f"nn_pairs={len(pairs)}"

        prev_dx, prev_dy = dx_f, dy_f
        rows.append((f, dx_f, dy_f, ntr, len(obj), note))

    df = pd.DataFrame(rows, columns=["frame_idx", "dx", "dy", "n_tracks", "n_json", "note"])

    # smooth
    if smooth_window and smooth_window > 1:
        df["dx_smooth"] = df["dx"].rolling(smooth_window, center=True, min_periods=1).median()
        df["dy_smooth"] = df["dy"].rolling(smooth_window, center=True, min_periods=1).median()
    else:
        df["dx_smooth"] = df["dx"]
        df["dy_smooth"] = df["dy"]

    df.to_csv(out_csv, index=False)
    return df


# -----------------------------
# Main nucleation logic
# -----------------------------
@dataclass
class RunConfig:
    L: int
    amin_px: float
    rnuc_max: float
    strict_overlap: bool
    overlap_prev_max: float
    lookahead_k: int
    prev_search_pad_px: float
    prev_max_candidates: int
    k_closest: int
    overlap_unknown_policy: str  # keep / reject
    overlap_per_call_timeout_s: float
    progress_every: int
    checkpoint_every: int
    accept_empty_prev: bool
    accept_no_candidate_prev: bool
    auto_offset: bool
    offset_min_tracks: int
    offset_smooth_window: int
    offset_use_prev_if_empty: bool
    global_fallback_k: int
    bbox_then_poly_topN: int

def select_prev_candidates(
    prev_objs: List[Dict[str, Any]],
    cxj: float,
    cyj: float,
    pad: float,
    prev_max_candidates: int,
    global_fallback_k: int,
) -> Tuple[List[Dict[str, Any]], str]:
    if not prev_objs:
        return [], "prev_empty"

    # ROI filter
    roi = []
    x0, x1 = cxj - pad, cxj + pad
    y0, y1 = cyj - pad, cyj + pad
    for a in prev_objs:
        bx, by = bbox_center(a["bbox"])
        if (x0 <= bx <= x1) and (y0 <= by <= y1):
            roi.append(a)

    if roi:
        # if too many, pick closest
        if len(roi) > prev_max_candidates:
            roi.sort(key=lambda a: (bbox_center(a["bbox"])[0] - cxj) ** 2 + (bbox_center(a["bbox"])[1] - cyj) ** 2)
            roi = roi[:prev_max_candidates]
        return roi, f"roi(n={len(roi)})"

    # fallback: global K-nearest
    all_list = list(prev_objs)
    all_list.sort(key=lambda a: (bbox_center(a["bbox"])[0] - cxj) ** 2 + (bbox_center(a["bbox"])[1] - cyj) ** 2)
    cand = all_list[: max(1, global_fallback_k)]
    return cand, f"global_knn(n={len(cand)})"


def compute_overlap_prev(
    curr_obj: Dict[str, Any],
    prev_cands: List[Dict[str, Any]],
    cfg: RunConfig,
) -> Tuple[Optional[float], str]:
    if not prev_cands:
        return None, "no_candidates"

    t0 = time.time()

    # 1) bbox max IoU
    b_curr = curr_obj["bbox"]
    ious = []
    for a in prev_cands:
        ious.append(bbox_iou(b_curr, a["bbox"]))
        if (time.time() - t0) > cfg.overlap_per_call_timeout_s:
            # partial result still valid for bbox stage
            break
    bbox_max = float(max(ious)) if ious else 0.0
    note = f"bbox_max(n={len(prev_cands)})(max={bbox_max:.3f})"

    # If we cannot do polygon, stop here
    if cfg.bbox_then_poly_topN <= 0:
        return bbox_max, note

    # 2) optional poly refinement near threshold (needs pycocotools)
    # Only worth it if bbox suggests potential overlap.
    if bbox_max <= 0.01:
        return bbox_max, note

    # refine top-N by bbox IoU
    idx = np.argsort(-np.array(ious + [0.0] * (len(prev_cands) - len(ious))))[: min(cfg.bbox_then_poly_topN, len(prev_cands))]
    poly_max = None
    poly_count = 0

    seg_curr = curr_obj.get("segmentation", None)
    if seg_curr is None:
        return bbox_max, note + "|no_curr_seg"

    for j in idx:
        a = prev_cands[int(j)]
        seg_prev = a.get("segmentation", None)
        if seg_prev is None:
            continue
        v = poly_iou_from_rle(seg_curr, seg_prev)
        if v is None:
            # cannot compute poly IoU (no pycocotools) → keep bbox
            return bbox_max, note + "|poly_unavailable"
        poly_count += 1
        poly_max = v if poly_max is None else max(poly_max, v)
        if (time.time() - t0) > cfg.overlap_per_call_timeout_s:
            break

    if poly_max is None:
        return bbox_max, note + "|poly_none"
    return float(poly_max), note + f"|poly_max(n={poly_count})(max={poly_max:.3f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks_csv", required=True)
    ap.add_argument("--json_dir", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--L", type=int, default=5)
    ap.add_argument("--amin_px", type=float, default=800)
    ap.add_argument("--rnuc_max", type=float, default=60)

    ap.add_argument("--strict_overlap", action="store_true")
    ap.add_argument("--overlap_prev_max", type=float, default=0.3)

    ap.add_argument("--lookahead_k", type=int, default=150)

    ap.add_argument("--prev_search_pad_px", type=float, default=250)
    ap.add_argument("--prev_max_candidates", type=int, default=50)
    ap.add_argument("--k_closest", type=int, default=15)

    ap.add_argument("--overlap_unknown_policy", choices=["keep", "reject"], default="keep")
    ap.add_argument("--overlap_per_call_timeout_s", type=float, default=0.6)

    ap.add_argument("--progress_every", type=int, default=25)
    ap.add_argument("--checkpoint_every", type=int, default=100)

    ap.add_argument("--accept_empty_prev", action="store_true")
    ap.add_argument("--accept_no_candidate_prev", action="store_true")

    ap.add_argument("--auto_offset", action="store_true")
    ap.add_argument("--offset_min_tracks", type=int, default=20)
    ap.add_argument("--offset_smooth_window", type=int, default=21)
    ap.add_argument("--offset_use_prev_if_empty", action="store_true")

    ap.add_argument("--global_fallback_k", type=int, default=30)
    ap.add_argument("--bbox_then_poly_topN", type=int, default=10)

    args = ap.parse_args()

    cfg = RunConfig(
        L=args.L,
        amin_px=args.amin_px,
        rnuc_max=args.rnuc_max,
        strict_overlap=args.strict_overlap,
        overlap_prev_max=args.overlap_prev_max,
        lookahead_k=args.lookahead_k,
        prev_search_pad_px=args.prev_search_pad_px,
        prev_max_candidates=args.prev_max_candidates,
        k_closest=args.k_closest,
        overlap_unknown_policy=args.overlap_unknown_policy,
        overlap_per_call_timeout_s=args.overlap_per_call_timeout_s,
        progress_every=args.progress_every,
        checkpoint_every=args.checkpoint_every,
        accept_empty_prev=args.accept_empty_prev,
        accept_no_candidate_prev=args.accept_no_candidate_prev,
        auto_offset=args.auto_offset,
        offset_min_tracks=args.offset_min_tracks,
        offset_smooth_window=args.offset_smooth_window,
        offset_use_prev_if_empty=args.offset_use_prev_if_empty,
        global_fallback_k=args.global_fallback_k,
        bbox_then_poly_topN=args.bbox_then_poly_topN,
    )

    out_dir = args.out_dir
    ensure_dir(out_dir)

    tracks = pd.read_csv(args.tracks_csv)
    if "frame_idx" not in tracks.columns:
        raise RuntimeError("tracks.csv must have frame_idx column")

    offsets_df = None
    if cfg.auto_offset:
        print("[OK] Estimating per-frame offsets (tracks vs JSON bbox centers)...", flush=True)
        offsets_path = os.path.join(out_dir, "offsets_estimated.csv")
        offsets_df = estimate_offsets(
            tracks=tracks,
            json_dir=args.json_dir,
            out_csv=offsets_path,
            min_tracks=cfg.offset_min_tracks,
            smooth_window=cfg.offset_smooth_window,
            use_prev_if_empty=cfg.offset_use_prev_if_empty,
        )
        print(f"[OK] Offset estimation done. median dx={float(offsets_df['dx_smooth'].median()):.2f}, dy={float(offsets_df['dy_smooth'].median()):.2f}. Saved: {offsets_path}", flush=True)

    # Group by track_id
    print(f"[OK] Tracks: {tracks['track_id'].nunique()} unique track_id", flush=True)
    g = tracks.groupby("track_id", sort=True)

    kept_rows = []
    rej_rows = []

    processed = 0

    for track_id, tdf in g:
        processed += 1

        tdf = tdf.sort_values("frame_idx")
        # find first frame where area passes gate
        gate = tdf[(tdf["area_px"] >= cfg.amin_px) & (tdf["R_px"] <= cfg.rnuc_max)]
        if len(gate) == 0:
            rej_rows.append({
                "track_id": int(track_id),
                "reason": "trackgate_no_frame_passes_area_rnuc",
            })
            continue

        nuc_frame = int(gate.iloc[0]["frame_idx"])
        cx_t = float(gate.iloc[0]["cx"])
        cy_t = float(gate.iloc[0]["cy"])

        # offset: track_xy ≈ json_xy + offset  => json = track - offset
        if offsets_df is not None:
            orow = offsets_df[offsets_df["frame_idx"] == nuc_frame]
            if len(orow) == 0:
                dx = float(offsets_df["dx_smooth"].iloc[-1])
                dy = float(offsets_df["dy_smooth"].iloc[-1])
            else:
                dx = float(orow["dx_smooth"].iloc[0])
                dy = float(orow["dy_smooth"].iloc[0])
        else:
            dx, dy = 0.0, 0.0

        cx_j = cx_t - dx
        cy_j = cy_t - dy

        # load current frame JSON to get the "current object"
        curr_json_path = find_frame_json(args.json_dir, nuc_frame)
        if curr_json_path is None:
            rej_rows.append({
                "track_id": int(track_id),
                "nuc_frame_i": nuc_frame,
                "reason": "missing_curr_json",
            })
            continue
        curr_objs = json.load(open(curr_json_path, "r"))
        if not curr_objs:
            rej_rows.append({
                "track_id": int(track_id),
                "nuc_frame_i": nuc_frame,
                "reason": "empty_curr_json",
            })
            continue

        # choose current obj as closest bbox center to cx_j,cy_j
        curr_objs.sort(key=lambda a: (bbox_center(a["bbox"])[0]-cx_j)**2 + (bbox_center(a["bbox"])[1]-cy_j)**2)
        curr_obj = curr_objs[0]

        # prev frame candidates
        prev_frame = nuc_frame - 1
        prev_json_path = find_frame_json(args.json_dir, prev_frame)
        prev_objs = []
        if prev_json_path is not None:
            prev_objs = json.load(open(prev_json_path, "r"))

        if not prev_objs:
            if cfg.accept_empty_prev:
                kept_rows.append({
                    "track_id": int(track_id),
                    "nuc_frame_i": nuc_frame,
                    "cx": cx_t, "cy": cy_t,
                    "dx": dx, "dy": dy,
                    "overlap_prev": 0.0,
                    "overlap_note": "prev_empty_assumed_0",
                    "curr_json_file": os.path.basename(curr_json_path),
                    "prev_json_file": os.path.basename(prev_json_path) if prev_json_path else None,
                })
            else:
                rej_rows.append({
                    "track_id": int(track_id),
                    "nuc_frame_i": nuc_frame,
                    "reason": "prev_empty_reject",
                })
            continue

        prev_cands, cand_note = select_prev_candidates(
            prev_objs, cx_j, cy_j,
            pad=cfg.prev_search_pad_px,
            prev_max_candidates=cfg.prev_max_candidates,
            global_fallback_k=cfg.global_fallback_k,
        )

        if not prev_cands:
            if cfg.accept_no_candidate_prev:
                kept_rows.append({
                    "track_id": int(track_id),
                    "nuc_frame_i": nuc_frame,
                    "cx": cx_t, "cy": cy_t,
                    "dx": dx, "dy": dy,
                    "overlap_prev": 0.0,
                    "overlap_note": "prev_nocand_assumed_0",
                    "cand_note": cand_note,
                })
            else:
                rej_rows.append({
                    "track_id": int(track_id),
                    "nuc_frame_i": nuc_frame,
                    "reason": "prev_nocand_reject",
                    "cand_note": cand_note,
                })
            continue

        ov, ov_note = compute_overlap_prev(curr_obj, prev_cands, cfg)
        if ov is None:
            # unknown overlap (timeout / failure)
            if cfg.overlap_unknown_policy == "keep":
                kept_rows.append({
                    "track_id": int(track_id),
                    "nuc_frame_i": nuc_frame,
                    "cx": cx_t, "cy": cy_t,
                    "dx": dx, "dy": dy,
                    "overlap_prev": np.nan,
                    "overlap_note": "overlap_unknown_keep|" + cand_note,
                })
            else:
                rej_rows.append({
                    "track_id": int(track_id),
                    "nuc_frame_i": nuc_frame,
                    "reason": "overlap_unknown_reject",
                    "overlap_note": cand_note,
                })
            continue

        # decision
        if cfg.strict_overlap and (ov > cfg.overlap_prev_max):
            rej_rows.append({
                "track_id": int(track_id),
                "nuc_frame_i": nuc_frame,
                "cx": cx_t, "cy": cy_t,
                "dx": dx, "dy": dy,
                "overlap_prev": ov,
                "overlap_note": cand_note + "|" + ov_note,
                "reason": f"overlap_prev>{cfg.overlap_prev_max}",
            })
        else:
            kept_rows.append({
                "track_id": int(track_id),
                "nuc_frame_i": nuc_frame,
                "cx": cx_t, "cy": cy_t,
                "dx": dx, "dy": dy,
                "overlap_prev": ov,
                "overlap_note": cand_note + "|" + ov_note,
            })

        if cfg.progress_every and (processed % cfg.progress_every == 0):
            print(f"[PROGRESS] {processed}/{tracks['track_id'].nunique()} | kept={len(kept_rows)} | rej={len(rej_rows)}", flush=True)

        if cfg.checkpoint_every and (processed % cfg.checkpoint_every == 0):
            pd.DataFrame(kept_rows).to_csv(os.path.join(out_dir, f"nucleation_events_filtered_ckpt_{processed:05d}.csv"), index=False)
            pd.DataFrame(rej_rows).to_csv(os.path.join(out_dir, f"nucleation_events_rejected_ckpt_{processed:05d}.csv"), index=False)

    # write final outputs
    kept_df = pd.DataFrame(kept_rows)
    rej_df = pd.DataFrame(rej_rows)

    kept_df.to_csv(os.path.join(out_dir, "nucleation_events_filtered.csv"), index=False)
    rej_df.to_csv(os.path.join(out_dir, "nucleation_events_rejected.csv"), index=False)

    state = {
        "kept": int(len(kept_df)),
        "rejected": int(len(rej_df)),
        "has_shapely": bool(HAS_SHAPELY),
        "has_pycocotools": _has_pycocotools(),
        "config": vars(args),
    }
    with open(os.path.join(out_dir, "state.json"), "w") as f:
        json.dump(state, f, indent=2)

    print(f"[OK] Wrote outputs to: {out_dir}", flush=True)
    print(f"[OK] kept={len(kept_df)} | rejected={len(rej_df)} | has_shapely={HAS_SHAPELY} | has_pycocotools={state['has_pycocotools']}", flush=True)
    print(f"[OK] State file: {os.path.join(out_dir, 'state.json')}", flush=True)

def _has_pycocotools() -> bool:
    try:
        import pycocotools  # noqa
        return True
    except Exception:
        return False

if __name__ == "__main__":
    main()