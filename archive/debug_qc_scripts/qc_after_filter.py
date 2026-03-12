import os, glob, json, re
from statistics import median

# ---- EDIT THESE IF NEEDED ----
W, H = 4704, 3424  # image width/height
RUN_ROOT = r"sam\segfreeze_v1_fapi_vs_tempo"
DATASETS = ["FAPI", "FAPI_TEMPO"]

# Filter rule: drop "giant" masks by bbox fraction
GIANT_BBOX_FRAC = 0.5

# How many frames to summarize at start/end
HEAD = 25
TAIL = 10


def parse_frame_index_and_time_ms(fname: str):
    # expects frame_00012_t24.00ms.json
    m = re.search(r"frame_(\d+)_t([0-9.]+)ms", os.path.basename(fname))
    if not m:
        return None, None
    return int(m.group(1)), float(m.group(2))


def load_ann_list(path):
    with open(path, "r") as f:
        data = json.load(f)
    # expected: list of dicts, each with "bbox": [x,y,w,h]
    if not isinstance(data, list):
        raise ValueError(f"Unexpected JSON format (not a list): {path}")
    return data


def bbox_frac(ann):
    x, y, w, h = ann["bbox"]
    return (w * h) / (W * H)


def summarize_dataset(ds):
    pattern = os.path.join(RUN_ROOT, ds, "coco_rle", "frame_*_t*ms.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"\n[{ds}] No files found at: {pattern}")
        return

    n_total = len(files)

    n_empty_raw = 0
    n_empty_filtered = 0
    n_giant_frames = 0

    first_nonempty_filtered = None  # (frame_idx, t_ms, n_after)
    last_frame_stats = None         # (frame_idx, t_ms, n_raw, n_after, giants_dropped)

    counts_raw = []
    counts_after = []

    for f in files:
        frame_idx, t_ms = parse_frame_index_and_time_ms(f)
        anns = load_ann_list(f)
        n_raw = len(anns)
        counts_raw.append(n_raw)

        if n_raw == 0:
            n_empty_raw += 1
            n_after = 0
            dropped = 0
        else:
            fracs = [bbox_frac(a) for a in anns]
            dropped = sum(fr > GIANT_BBOX_FRAC for fr in fracs)
            n_after = n_raw - dropped

            if dropped > 0:
                n_giant_frames += 1

            if n_after == 0:
                n_empty_filtered += 1

        counts_after.append(n_after)

        if first_nonempty_filtered is None and n_after > 0:
            first_nonempty_filtered = (frame_idx, t_ms, n_after)

        last_frame_stats = (frame_idx, t_ms, n_raw, n_after, dropped)

    def safe_median(x):
        return median(x) if x else 0

    # ---- Report ----
    print(f"\n=== {ds} ===")
    print(f"JSON files: {n_total}")
    print(f"Empty (raw): {n_empty_raw}  | Empty (after filter): {n_empty_filtered}")
    print(f"Frames containing >=1 giant bbox (>{GIANT_BBOX_FRAC}): {n_giant_frames}  ({n_giant_frames/n_total:.3f})")
    print(f"Median n_masks raw: {safe_median(counts_raw)}  | after filter: {safe_median(counts_after)}")

    if first_nonempty_filtered is None:
        print("First appearance (after filter): NONE (all frames empty)")
    else:
        fi, tm, n = first_nonempty_filtered
        print(f"First appearance (after filter): frame={fi:05d}  t={tm:.2f} ms  n_masks={n}")

    li, ltm, nraw, naft, dropped = last_frame_stats
    print(f"Last frame: frame={li:05d} t={ltm:.2f} ms  raw={nraw}  after={naft}  dropped_giant={dropped}")

    # Quick head/tail preview of counts
    print("\nCounts after filter (head):")
    for i, f in enumerate(files[:HEAD]):
        idx, tm = parse_frame_index_and_time_ms(f)
        print(f"  frame={idx:05d} t={tm:7.2f} ms  n_after={counts_after[i]}")

    print("\nCounts after filter (tail):")
    start = max(0, n_total - TAIL)
    for j in range(start, n_total):
        idx, tm = parse_frame_index_and_time_ms(files[j])
        print(f"  frame={idx:05d} t={tm:7.2f} ms  n_after={counts_after[j]}")


def main():
    print("=== QC after giant-bbox filter ===")
    print(f"RUN_ROOT: {RUN_ROOT}")
    print(f"W,H: {W},{H}")
    print(f"GIANT_BBOX_FRAC: {GIANT_BBOX_FRAC}")
    for ds in DATASETS:
        summarize_dataset(ds)


if __name__ == "__main__":
    main()