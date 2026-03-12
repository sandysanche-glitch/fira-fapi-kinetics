# build_checkpoint_manifest.py
# Robust checkpoint builder for:
# Kinetics/harmonization_checkpoints/v001_video_pair_fapi_vs_fapi_tempo
#
# What it does:
# 1) Verifies / creates checkpoint tree (if missing)
# 2) Copies curated source files into:
#      - raw_snapshot/... (folder-level snapshots, file-wise copy into mapped subfolders)
#      - canonical_inputs/FAPI and canonical_inputs/FAPI_TEMPO
# 3) Writes manifests:
#      - source_manifest_raw.csv
#      - source_manifest_curated.csv
#      - schema_report.csv
#      - hash_manifest.csv
# 4) Writes clear warnings for missing tau/growth sources
#
# Notes:
# - Safe to rerun (files overwritten by default for deterministic refresh)
# - Uses only stdlib + pandas
# - Designed for your current FAPI / FAPI-TEMPO video-pair checkpoint workflow

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# =========================
# USER CONFIG (EDIT HERE)
# =========================
KINETICS_ROOT = Path(r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics")
CHECKPOINT_VERSION = "v001_video_pair_fapi_vs_fapi_tempo"

# Calibration for metadata placeholders / provenance notes
UM_PER_PX = 0.065

# If True, existing copied files in checkpoint will be overwritten on rerun
OVERWRITE_EXISTING = True


# =========================
# Helpers
# =========================
def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_text_if_missing(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def copy_file(src: Path, dst: Path, overwrite: bool = True) -> Tuple[bool, str]:
    """
    Returns (copied_ok, message)
    """
    try:
        ensure_dir(dst.parent)
        if dst.exists():
            if overwrite:
                shutil.copy2(src, dst)
                return True, "overwritten"
            return True, "exists_skipped"
        shutil.copy2(src, dst)
        return True, "copied"
    except Exception as e:
        return False, f"copy_error: {e}"


def safe_read_csv(path: Path) -> Tuple[Optional[pd.DataFrame], str]:
    try:
        df = pd.read_csv(path)
        return df, ""
    except Exception as e:
        return None, str(e)


def infer_schema_type(columns: List[str]) -> str:
    cols = set(c.lower() for c in columns)

    if {"time_ms", "dn_dt"}.issubset(cols) or {"time_ms", "dn_dt_per_s"}.issubset(cols):
        return "rate_curve"
    if {"time_ms", "nt"}.issubset(cols) or {"time_ms", "n"}.issubset(cols):
        return "Nt_curve"
    if "tau_ms" in cols or "tau" in cols:
        return "tau_table"
    if "active_grains" in cols:
        return "active_curve"
    if "frame_id" in cols and "area_px" in cols:
        return "frame_kinetics"
    if "track_id" in cols and ("t_nuc_ms" in cols or "time_ms" in cols):
        return "track_summary_or_events"
    if "time_ms" in cols and ("median_um_per_s_raw" in cols or "median_um_per_s_clipped" in cols or "median_um_per_s" in cols):
        return "growth_curve"
    return "unknown"


def csv_columns_json(df: pd.DataFrame) -> str:
    return json.dumps(list(df.columns), ensure_ascii=False)


def rel_to(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path)


# =========================
# Data classes
# =========================
@dataclass
class SourceRecord:
    dataset: str
    group: str               # raw_snapshot / canonical_inputs
    label: str               # semantic label
    original_path: str
    exists_at_copy_time: bool
    copied_to: str
    copy_status: str
    notes: str


@dataclass
class CuratedRecord:
    dataset: str
    label: str
    canonical_path: str
    source_path: str
    status: str              # copied / missing / placeholder / warning
    notes: str


@dataclass
class SchemaRecord:
    dataset: str
    label: str
    file_path: str
    n_rows: Optional[int]
    n_cols: Optional[int]
    columns_json: str
    schema_type: str
    parse_ok: bool
    warnings: str


@dataclass
class HashRecord:
    file_path: str
    sha256: str
    size_bytes: int
    modified_time: str


# =========================
# Checkpoint tree bootstrap (minimal)
# =========================
def ensure_checkpoint_tree(checkpoint_root: Path) -> None:
    dirs = [
        checkpoint_root,
        checkpoint_root / "manifests",
        checkpoint_root / "raw_snapshot" / "FAPI" / "matched_kinetics_for_compare",
        checkpoint_root / "raw_snapshot" / "FAPI" / "retrack_cuda_vith",
        checkpoint_root / "raw_snapshot" / "FAPI" / "sam_cuda_vith_clean_FAPI_kinetics",
        checkpoint_root / "raw_snapshot" / "FAPI_TEMPO" / "stable_v15_out_overlap03_poly_pad400",
        checkpoint_root / "raw_snapshot" / "FAPI_TEMPO" / "out_FAPI_TEMPO_track_summary",
        checkpoint_root / "raw_snapshot" / "FAPI_TEMPO" / "idmap_kinetics_win60",
        checkpoint_root / "raw_snapshot" / "FAPI_TEMPO" / "sam_cuda_vith_clean_FAPI_TEMPO",
        checkpoint_root / "canonical_inputs" / "FAPI",
        checkpoint_root / "canonical_inputs" / "FAPI_TEMPO",
        checkpoint_root / "harmonized_tables" / "FAPI",
        checkpoint_root / "harmonized_tables" / "FAPI_TEMPO",
        checkpoint_root / "compare_outputs" / "csv",
        checkpoint_root / "compare_outputs" / "png",
        checkpoint_root / "scripts",
    ]
    for d in dirs:
        ensure_dir(d)

    write_text_if_missing(
        checkpoint_root / "README_scope.md",
        f"# Harmonization checkpoint\nCreated/updated: {now_iso()}\n\nVideo-pair FAPI vs FAPI-TEMPO checkpoint.\n"
    )
    write_text_if_missing(
        checkpoint_root / "decisions_log.md",
        f"# Decisions log\nCreated/updated: {now_iso()}\n\n- Calibration target: {UM_PER_PX} um/px for both datasets\n"
    )


# =========================
# Source mapping (curated)
# =========================
def build_source_map(kin: Path) -> Dict[str, Dict]:
    """
    Returns a nested structure for both datasets.
    You can edit this mapping as paths evolve.
    """
    fapi_base_retrack = kin / "out" / "FAPI" / "retrack_cuda_vith"
    fapi_matched = fapi_base_retrack / "matched_kinetics_for_compare"
    fapi_clean_kin = kin / "sam_cuda_vith_clean" / "FAPI" / "kinetics"
    fapi_clean_root = kin / "sam_cuda_vith_clean" / "FAPI"

    tempo_stable = kin / "stable_v15_out_overlap03_poly_pad400"
    tempo_track_summary_dir = kin / "out" / "FAPI_TEMPO"
    tempo_idmap60 = kin / "out" / "FAPI_TEMPO" / "idmap_kinetics_win60"
    tempo_clean_root = kin / "sam_cuda_vith_clean" / "FAPI_TEMPO"

    return {
        "FAPI": {
            "raw_groups": {
                "matched_kinetics_for_compare": {
                    "src_dir": fapi_matched,
                    "files": [
                        "kinetics_tau0p3_events_FAPI.csv",
                        "kinetics_tau0p3_rate.csv",
                        "kinetics_tau0p3_Nt.csv",
                    ],
                },
                "retrack_cuda_vith": {
                    "src_dir": fapi_base_retrack,
                    "files": [
                        "FAPI_active_tracks.csv",
                        "FAPI_tau_fits.csv",
                        "FAPI_track_summary.csv",
                        "FAPI_retracked_tracks.csv",
                    ],
                },
                "sam_cuda_vith_clean_FAPI_kinetics": {
                    "src_dir": fapi_clean_kin,
                    "files": [
                        "growth_rate_vs_time.csv",
                    ],
                },
            },
            "canonical": {
                "events.csv": fapi_matched / "kinetics_tau0p3_events_FAPI.csv",
                "rate_curve.csv": fapi_matched / "kinetics_tau0p3_rate.csv",
                "Nt_curve.csv": fapi_matched / "kinetics_tau0p3_Nt.csv",
                "active_tracks_source.csv": fapi_base_retrack / "FAPI_active_tracks.csv",
                "tracks_source.csv": fapi_base_retrack / "FAPI_retracked_tracks.csv",  # if available
                "tau_fits.csv": fapi_base_retrack / "FAPI_tau_fits.csv",
                "growth_rate_vs_time.csv": fapi_clean_kin / "growth_rate_vs_time.csv",
                "frame_kinetics.csv": fapi_clean_root / "frame_kinetics.csv",          # optional but useful
                "metadata.yaml": None,  # generated
            },
        },
        "FAPI_TEMPO": {
            "raw_groups": {
                "stable_v15_out_overlap03_poly_pad400": {
                    "src_dir": tempo_stable,
                    "files": [
                        "kinetics_tau0p3_events.csv",
                        "kinetics_tau0p3_rate.csv",
                        "kinetics_tau0p3_Nt.csv",
                    ],
                },
                "out_FAPI_TEMPO_track_summary": {
                    "src_dir": tempo_track_summary_dir,
                    "files": [
                        "track_summary.csv",  # NOT tau fits; may be used as provisional track summary
                    ],
                },
                "idmap_kinetics_win60": {
                    "src_dir": tempo_idmap60,
                    "files": [
                        "growth_rate_vs_time.csv",
                    ],
                },
                "sam_cuda_vith_clean_FAPI_TEMPO": {
                    "src_dir": tempo_clean_root,
                    "files": [
                        "frame_kinetics.csv",  # optional; user confirmed location
                    ],
                },
            },
            "canonical": {
                "events.csv": tempo_stable / "kinetics_tau0p3_events.csv",
                "rate_curve.csv": tempo_stable / "kinetics_tau0p3_rate.csv",
                "Nt_curve.csv": tempo_stable / "kinetics_tau0p3_Nt.csv",
                # Harmonization target for active source: use same source TYPE if possible.
                # If no precomputed active table, this may be generated later from tracks/frame_kinetics.
                "active_tracks_source.csv": None,  # not directly available in current known files
                "tracks_source.csv": None,         # not directly available in current known files
                # True tau fits needed for publication-robust tau comparison:
                "tau_fits.csv": None,              # missing currently
                "growth_rate_vs_time.csv": tempo_idmap60 / "growth_rate_vs_time.csv",
                "frame_kinetics.csv": tempo_clean_root / "frame_kinetics.csv",
                "metadata.yaml": None,  # generated
                "tau_unavailable.flag": None,  # generated if tau_fits missing
            },
        },
    }


# =========================
# Metadata generation
# =========================
def write_metadata_yaml(path: Path, dataset_name: str, source_notes: Dict[str, str]) -> None:
    ensure_dir(path.parent)
    lines = [
        f"dataset: {dataset_name}",
        f"display_name: {dataset_name.replace('_', '-')}",
        "video_type: crystallization_sequence",
        f"calibration_um_per_px: {UM_PER_PX}",
        "time_unit: ms",
        f"generated_at: '{now_iso()}'",
        "source_notes:",
    ]
    for k, v in source_notes.items():
        v_esc = str(v).replace('"', "'")
        lines.append(f'  {k}: "{v_esc}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =========================
# Schema + hash collection
# =========================
def append_schema_record(schema_records: List[SchemaRecord], dataset: str, label: str, fpath: Path) -> None:
    if not fpath.exists():
        schema_records.append(
            SchemaRecord(
                dataset=dataset,
                label=label,
                file_path=str(fpath),
                n_rows=None,
                n_cols=None,
                columns_json="[]",
                schema_type="missing",
                parse_ok=False,
                warnings="file_missing",
            )
        )
        return

    if fpath.suffix.lower() != ".csv":
        schema_records.append(
            SchemaRecord(
                dataset=dataset,
                label=label,
                file_path=str(fpath),
                n_rows=None,
                n_cols=None,
                columns_json="[]",
                schema_type="non_csv",
                parse_ok=True,
                warnings="skipped_non_csv",
            )
        )
        return

    df, err = safe_read_csv(fpath)
    if df is None:
        schema_records.append(
            SchemaRecord(
                dataset=dataset,
                label=label,
                file_path=str(fpath),
                n_rows=None,
                n_cols=None,
                columns_json="[]",
                schema_type="csv_parse_failed",
                parse_ok=False,
                warnings=err,
            )
        )
    else:
        schema_records.append(
            SchemaRecord(
                dataset=dataset,
                label=label,
                file_path=str(fpath),
                n_rows=int(df.shape[0]),
                n_cols=int(df.shape[1]),
                columns_json=csv_columns_json(df),
                schema_type=infer_schema_type(list(df.columns)),
                parse_ok=True,
                warnings="",
            )
        )


def append_hash_record(hash_records: List[HashRecord], fpath: Path) -> None:
    if not fpath.exists() or not fpath.is_file():
        return
    st = fpath.stat()
    hash_records.append(
        HashRecord(
            file_path=str(fpath),
            sha256=sha256_file(fpath),
            size_bytes=int(st.st_size),
            modified_time=datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        )
    )


# =========================
# Main builder
# =========================
def main():
    checkpoint_root = KINETICS_ROOT / "harmonization_checkpoints" / CHECKPOINT_VERSION
    ensure_checkpoint_tree(checkpoint_root)

    manifests_dir = checkpoint_root / "manifests"

    source_manifest_records: List[SourceRecord] = []
    curated_manifest_records: List[CuratedRecord] = []
    schema_records: List[SchemaRecord] = []
    hash_records: List[HashRecord] = []

    warnings_global: List[str] = []

    srcmap = build_source_map(KINETICS_ROOT)

    print("[INFO] Building harmonization checkpoint manifests...")
    print(f"[INFO] KINETICS_ROOT   : {KINETICS_ROOT}")
    print(f"[INFO] CHECKPOINT_ROOT : {checkpoint_root}")

    # ----------------------------------------
    # 1) raw_snapshot copy
    # ----------------------------------------
    for dataset, ds_cfg in srcmap.items():
        for raw_group_name, raw_cfg in ds_cfg["raw_groups"].items():
            src_dir = Path(raw_cfg["src_dir"])
            dst_dir = checkpoint_root / "raw_snapshot" / dataset / raw_group_name

            ensure_dir(dst_dir)

            for fname in raw_cfg["files"]:
                src_file = src_dir / fname
                dst_file = dst_dir / fname

                exists = src_file.exists()
                if exists:
                    ok, msg = copy_file(src_file, dst_file, overwrite=OVERWRITE_EXISTING)
                    copy_status = msg if ok else msg
                    notes = ""
                else:
                    copy_status = "missing_source"
                    notes = "Source file not found at expected path."

                source_manifest_records.append(
                    SourceRecord(
                        dataset=dataset,
                        group="raw_snapshot",
                        label=f"{raw_group_name}/{fname}",
                        original_path=str(src_file),
                        exists_at_copy_time=bool(exists),
                        copied_to=str(dst_file),
                        copy_status=copy_status,
                        notes=notes,
                    )
                )

                # schema/hash for copied destination (or record missing destination)
                append_schema_record(schema_records, dataset, f"raw:{raw_group_name}/{fname}", dst_file)
                if dst_file.exists():
                    append_hash_record(hash_records, dst_file)

    # ----------------------------------------
    # 2) canonical_inputs copy + placeholders
    # ----------------------------------------
    for dataset, ds_cfg in srcmap.items():
        canonical_dir = checkpoint_root / "canonical_inputs" / dataset
        ensure_dir(canonical_dir)
        source_notes_for_metadata: Dict[str, str] = {}

        for canon_name, src_path in ds_cfg["canonical"].items():
            dst_path = canonical_dir / canon_name

            # metadata handled later
            if canon_name == "metadata.yaml":
                continue

            # explicitly generated flag if tau missing (tempo)
            if canon_name == "tau_unavailable.flag":
                continue

            if src_path is None:
                # missing optional / required source
                status = "missing"
                notes = "No source path configured / not available yet."

                # Special handling for FAPI_TEMPO tau
                if canon_name == "tau_fits.csv":
                    flag_path = canonical_dir / "tau_unavailable.flag"
                    flag_text = (
                        f"missing_tau_fits: true\n"
                        f"dataset: {dataset}\n"
                        f"generated_at: '{now_iso()}'\n"
                        f"reason: true tau-fit table (schema like FAPI_tau_fits.csv) not yet curated\n"
                    )
                    flag_path.write_text(flag_text, encoding="utf-8")
                    curated_manifest_records.append(
                        CuratedRecord(
                            dataset=dataset,
                            label="tau_unavailable.flag",
                            canonical_path=str(flag_path),
                            source_path="",
                            status="generated_flag",
                            notes="Tau fits unavailable; drop tau plot or generate true tau-fit table.",
                        )
                    )
                    append_schema_record(schema_records, dataset, "canonical:tau_unavailable.flag", flag_path)
                    append_hash_record(hash_records, flag_path)
                    warnings_global.append(
                        "[WARN] FAPI_TEMPO missing true tau_fits.csv (publication-robust tau comparison not yet valid)."
                    )

                # Special handling for active/growth warnings
                if canon_name == "active_tracks_source.csv":
                    warnings_global.append(
                        f"[WARN] {dataset} missing active_tracks_source.csv. Harmonize active-grains source/type before robust comparison."
                    )
                if canon_name == "growth_rate_vs_time.csv":
                    warnings_global.append(
                        f"[WARN] {dataset} missing growth_rate_vs_time.csv."
                    )

                curated_manifest_records.append(
                    CuratedRecord(
                        dataset=dataset,
                        label=canon_name,
                        canonical_path=str(dst_path),
                        source_path="",
                        status=status,
                        notes=notes,
                    )
                )
                append_schema_record(schema_records, dataset, f"canonical:{canon_name}", dst_path)
                continue

            src_file = Path(src_path)
            exists = src_file.exists()
            if exists:
                ok, msg = copy_file(src_file, dst_path, overwrite=OVERWRITE_EXISTING)
                status = "copied" if ok else "copy_error"
                notes = msg
                source_notes_for_metadata[canon_name] = str(src_file)
            else:
                status = "missing"
                notes = "Expected source file not found."
                source_notes_for_metadata[canon_name] = f"MISSING: {src_file}"

                if canon_name == "tau_fits.csv":
                    warnings_global.append(
                        f"[WARN] {dataset} tau_fits.csv source missing: {src_file}"
                    )
                if canon_name == "growth_rate_vs_time.csv":
                    warnings_global.append(
                        f"[WARN] {dataset} growth_rate_vs_time.csv source missing: {src_file}"
                    )
                if canon_name == "active_tracks_source.csv":
                    warnings_global.append(
                        f"[WARN] {dataset} active_tracks_source.csv source missing."
                    )

            curated_manifest_records.append(
                CuratedRecord(
                    dataset=dataset,
                    label=canon_name,
                    canonical_path=str(dst_path),
                    source_path=str(src_file),
                    status=status,
                    notes=notes,
                )
            )

            append_schema_record(schema_records, dataset, f"canonical:{canon_name}", dst_path)
            if dst_path.exists():
                append_hash_record(hash_records, dst_path)

        # metadata.yaml (generated)
        metadata_path = canonical_dir / "metadata.yaml"
        write_metadata_yaml(metadata_path, dataset, source_notes_for_metadata)
        curated_manifest_records.append(
            CuratedRecord(
                dataset=dataset,
                label="metadata.yaml",
                canonical_path=str(metadata_path),
                source_path="generated",
                status="generated",
                notes="Generated provenance metadata for canonical inputs.",
            )
        )
        append_schema_record(schema_records, dataset, "canonical:metadata.yaml", metadata_path)
        append_hash_record(hash_records, metadata_path)

    # ----------------------------------------
    # 3) Write manifest CSVs
    # ----------------------------------------
    src_df = pd.DataFrame([asdict(r) for r in source_manifest_records])
    cur_df = pd.DataFrame([asdict(r) for r in curated_manifest_records])
    sch_df = pd.DataFrame([asdict(r) for r in schema_records])
    hsh_df = pd.DataFrame([asdict(r) for r in hash_records])

    # Sort for deterministic outputs
    if not src_df.empty:
        src_df = src_df.sort_values(["dataset", "group", "label"]).reset_index(drop=True)
    if not cur_df.empty:
        cur_df = cur_df.sort_values(["dataset", "label"]).reset_index(drop=True)
    if not sch_df.empty:
        sch_df = sch_df.sort_values(["dataset", "label", "file_path"]).reset_index(drop=True)
    if not hsh_df.empty:
        hsh_df = hsh_df.sort_values(["file_path"]).reset_index(drop=True)

    src_manifest_csv = manifests_dir / "source_manifest_raw.csv"
    cur_manifest_csv = manifests_dir / "source_manifest_curated.csv"
    schema_csv = manifests_dir / "schema_report.csv"
    hash_csv = manifests_dir / "hash_manifest.csv"

    src_df.to_csv(src_manifest_csv, index=False)
    cur_df.to_csv(cur_manifest_csv, index=False)
    sch_df.to_csv(schema_csv, index=False)
    hsh_df.to_csv(hash_csv, index=False)

    # ----------------------------------------
    # 4) Update README + decisions log (append notes)
    # ----------------------------------------
    readme_path = checkpoint_root / "README_scope.md"
    if readme_path.exists():
        txt = readme_path.read_text(encoding="utf-8")
    else:
        txt = "# Harmonization checkpoint\n"
    summary_block = (
        f"\n## Last checkpoint manifest build\n"
        f"- time: {now_iso()}\n"
        f"- source_manifest_raw rows: {len(src_df)}\n"
        f"- source_manifest_curated rows: {len(cur_df)}\n"
        f"- schema_report rows: {len(sch_df)}\n"
        f"- hash_manifest rows: {len(hsh_df)}\n"
    )
    if "## Last checkpoint manifest build" in txt:
        # leave existing; just append fresh run note
        txt += summary_block
    else:
        txt += summary_block
    readme_path.write_text(txt, encoding="utf-8")

    decisions_log = checkpoint_root / "decisions_log.md"
    if decisions_log.exists():
        dlog = decisions_log.read_text(encoding="utf-8")
    else:
        dlog = "# Decisions log\n"
    dlog += (
        f"\n## Manifest build run ({now_iso()})\n"
        f"- Calibration reference retained: {UM_PER_PX} um/px for both datasets.\n"
        f"- This script snapshots sources and canonical inputs only (no harmonization transform yet).\n"
    )
    if warnings_global:
        dlog += "- Warnings observed:\n"
        for w in warnings_global:
            dlog += f"  - {w.replace('[WARN] ', '')}\n"
    decisions_log.write_text(dlog, encoding="utf-8")

    # ----------------------------------------
    # 5) Console summary
    # ----------------------------------------
    print("[OK] Wrote manifests:")
    print(f"  {src_manifest_csv}")
    print(f"  {cur_manifest_csv}")
    print(f"  {schema_csv}")
    print(f"  {hash_csv}")

    print("[OK] Canonical inputs refreshed under:")
    print(f"  {checkpoint_root / 'canonical_inputs'}")

    if warnings_global:
        print("\n".join(warnings_global))
    else:
        print("[OK] No missing tau/growth warnings detected.")

    # Helpful checks for your next step
    print("\n[NEXT] Recommended harmonization checks:")
    print("  1) Active-grains source harmonization (same source type / same active criterion)")
    print("  2) Tau source harmonization (true tau-fit table for FAPI-TEMPO or drop tau panel)")
    print("  3) Freeze exact compare script version into checkpoint/scripts/")


if __name__ == "__main__":
    main()