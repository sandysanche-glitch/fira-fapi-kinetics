from pathlib import Path

FPS = 500  # VERIFY
ROOT = Path(r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\frames")

DATASETS = {
    "FAPI_TEMPO": ROOT / "FAPI_TEMPO",
    "FAPI": ROOT / "FAPI",
}

for label, folder in DATASETS.items():
    frames = sorted(folder.glob("*.png"))

    if len(frames) == 0:
        raise RuntimeError(f"No PNG files found in {folder}")

    print(f"\nProcessing {label}: {len(frames)} frames")

    # ---- Stage 1: temporary rename
    tmp_names = []
    for i, f in enumerate(frames):
        tmp = folder / f"__tmp_{i:05d}.png"
        f.rename(tmp)
        tmp_names.append(tmp)

    # ---- Stage 2: final rename with timestamps
    for i, tmp in enumerate(tmp_names):
        t_ms = i * 1000 / FPS
        final_name = folder / f"frame_{i:05d}_t{t_ms:.2f}ms.png"
        tmp.rename(final_name)

    print(f"✔ {label}: timestamps added safely")
