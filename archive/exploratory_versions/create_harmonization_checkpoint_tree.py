# create_harmonization_checkpoint_tree.py
from pathlib import Path
from datetime import datetime

def ensure_file(path: Path, content: str = ""):
    """Create file only if it doesn't exist yet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content, encoding="utf-8")

def main():
    # ------------------------------------------------------------------
    # EDIT THIS if needed: point to your Kinetics folder
    # ------------------------------------------------------------------
    kinetics_root = Path(r"F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics")

    checkpoint_root = (
        kinetics_root
        / "harmonization_checkpoints"
        / "v001_video_pair_fapi_vs_fapi_tempo"
    )

    # -------------------------
    # Directories to create
    # -------------------------
    dirs = [
        checkpoint_root,

        checkpoint_root / "manifests",

        checkpoint_root / "raw_snapshot",
        checkpoint_root / "raw_snapshot" / "FAPI",
        checkpoint_root / "raw_snapshot" / "FAPI" / "matched_kinetics_for_compare",
        checkpoint_root / "raw_snapshot" / "FAPI" / "retrack_cuda_vith",
        checkpoint_root / "raw_snapshot" / "FAPI" / "sam_cuda_vith_clean_FAPI_kinetics",

        checkpoint_root / "raw_snapshot" / "FAPI_TEMPO",
        checkpoint_root / "raw_snapshot" / "FAPI_TEMPO" / "stable_v15_out_overlap03_poly_pad400",
        checkpoint_root / "raw_snapshot" / "FAPI_TEMPO" / "out_FAPI_TEMPO_track_summary",
        checkpoint_root / "raw_snapshot" / "FAPI_TEMPO" / "idmap_kinetics_win60",
        checkpoint_root / "raw_snapshot" / "FAPI_TEMPO" / "sam_cuda_vith_clean_FAPI_TEMPO",

        checkpoint_root / "canonical_inputs",
        checkpoint_root / "canonical_inputs" / "FAPI",
        checkpoint_root / "canonical_inputs" / "FAPI_TEMPO",

        checkpoint_root / "harmonized_tables",
        checkpoint_root / "harmonized_tables" / "FAPI",
        checkpoint_root / "harmonized_tables" / "FAPI_TEMPO",

        checkpoint_root / "compare_outputs",
        checkpoint_root / "compare_outputs" / "csv",
        checkpoint_root / "compare_outputs" / "png",

        checkpoint_root / "scripts",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # -------------------------
    # Root docs
    # -------------------------
    readme_content = f"""# Harmonization Checkpoint
Created: {now}

## Scope
Video-sequence pair harmonization checkpoint for:
- FAPI
- FAPI-TEMPO

## Goal
Freeze the most relevant source files and harmonized intermediate tables
to support publication-robust comparison of crystallization kinetics.

## Notes
- Copy-only checkpoint (do not edit original source files).
- Rebuild comparisons from files inside this checkpoint once curated.
"""
    ensure_file(checkpoint_root / "README_scope.md", readme_content)

    decisions_content = f"""# Decisions Log
Created: {now}

## Initial placeholders
- Calibration (um/px): TBD (target currently 0.065 um/px for both)
- Time bin width (ms): TBD (e.g., 20 ms)
- Active-grains source type: TBD (tracks.csv vs active_tracks.csv)
- Active criterion: TBD
- Tau source harmonization: TBD (true tau-fit table required for both)
- Include tau panel in final compare: TBD
"""
    ensure_file(checkpoint_root / "decisions_log.md", decisions_content)

    # -------------------------
    # Manifests (starter CSVs)
    # -------------------------
    ensure_file(
        checkpoint_root / "manifests" / "source_manifest_raw.csv",
        "dataset,group,label,original_path,exists_at_copy_time,notes\n"
    )
    ensure_file(
        checkpoint_root / "manifests" / "source_manifest_curated.csv",
        "dataset,label,canonical_path,source_path,status,notes\n"
    )
    ensure_file(
        checkpoint_root / "manifests" / "schema_report.csv",
        "dataset,label,file_path,n_rows,n_cols,columns_json,schema_type,parse_ok,warnings\n"
    )
    ensure_file(
        checkpoint_root / "manifests" / "hash_manifest.csv",
        "file_path,sha256,size_bytes,modified_time\n"
    )

    # -------------------------
    # Canonical inputs placeholders
    # -------------------------
    # FAPI
    fapi_ci = checkpoint_root / "canonical_inputs" / "FAPI"
    ensure_file(fapi_ci / "events.csv", "time_ms,track_id\n")
    ensure_file(fapi_ci / "rate_curve.csv", "time_ms,dn_dt_per_s\n")
    ensure_file(fapi_ci / "Nt_curve.csv", "time_ms,Nt\n")
    ensure_file(fapi_ci / "active_tracks_source.csv", "time_ms,active_grains\n")
    ensure_file(fapi_ci / "tracks_source.csv", "")  # optional placeholder
    ensure_file(fapi_ci / "tau_fits.csv", "track_id,tau_ms\n")
    ensure_file(fapi_ci / "growth_rate_vs_time.csv", "time_ms,median_um_per_s\n")
    ensure_file(fapi_ci / "frame_kinetics.csv", "")  # optional placeholder
    ensure_file(
        fapi_ci / "metadata.yaml",
        """dataset: FAPI
display_name: FAPI
video_type: crystallization_sequence
calibration_um_per_px: 0.065
time_unit: ms
notes: "Fill in provenance and harmonization decisions"
"""
    )

    # FAPI_TEMPO
    tempo_ci = checkpoint_root / "canonical_inputs" / "FAPI_TEMPO"
    ensure_file(tempo_ci / "events.csv", "time_ms,track_id\n")
    ensure_file(tempo_ci / "rate_curve.csv", "time_ms,dn_dt_per_s\n")
    ensure_file(tempo_ci / "Nt_curve.csv", "time_ms,Nt\n")
    ensure_file(tempo_ci / "active_tracks_source.csv", "time_ms,active_grains\n")
    ensure_file(tempo_ci / "tracks_source.csv", "")  # optional placeholder

    # If true tau fits are not available yet, create a flag placeholder.
    # Later you can replace this with a real tau_fits.csv.
    ensure_file(tempo_ci / "tau_unavailable.flag", "True tau-fit table not curated yet.\n")
    # Uncomment the next line later when you have the real file and remove the .flag:
    # ensure_file(tempo_ci / "tau_fits.csv", "track_id,tau_ms\n")

    ensure_file(tempo_ci / "growth_rate_vs_time.csv", "time_ms,median_um_per_s\n")
    ensure_file(tempo_ci / "frame_kinetics.csv", "")  # optional placeholder
    ensure_file(
        tempo_ci / "metadata.yaml",
        """dataset: FAPI_TEMPO
display_name: FAPI-TEMPO
video_type: crystallization_sequence
calibration_um_per_px: 0.065
time_unit: ms
notes: "Fill in provenance and harmonization decisions"
"""
    )

    # -------------------------
    # Harmonized tables placeholders
    # -------------------------
    # FAPI
    fapi_ht = checkpoint_root / "harmonized_tables" / "FAPI"
    ensure_file(fapi_ht / "active_table_harmonized.csv", "time_ms,active_grains\n")
    ensure_file(fapi_ht / "tau_table_harmonized.csv", "track_id,tau_ms\n")
    ensure_file(fapi_ht / "growth_table_harmonized.csv", "time_ms,median_um_per_s\n")
    ensure_file(fapi_ht / "qc_summary.csv", "check,status,details\n")

    # FAPI_TEMPO
    tempo_ht = checkpoint_root / "harmonized_tables" / "FAPI_TEMPO"
    ensure_file(tempo_ht / "active_table_harmonized.csv", "time_ms,active_grains\n")
    ensure_file(tempo_ht / "tau_table_harmonized.csv", "track_id,tau_ms\n")  # may remain empty until tau fits exist
    ensure_file(tempo_ht / "growth_table_harmonized.csv", "time_ms,median_um_per_s\n")
    ensure_file(tempo_ht / "qc_summary.csv", "check,status,details\n")

    # -------------------------
    # Script stubs
    # -------------------------
    scripts_dir = checkpoint_root / "scripts"

    ensure_file(
        scripts_dir / "build_checkpoint_manifest.py",
        '''"""Build source/curated/schema/hash manifests for the harmonization checkpoint."""
from pathlib import Path

def main():
    print("TODO: implement manifest builder")

if __name__ == "__main__":
    main()
'''
    )

    ensure_file(
        scripts_dir / "harmonize_video_pair_inputs.py",
        '''"""Harmonize active-grains, tau, growth tables for FAPI vs FAPI-TEMPO."""
from pathlib import Path

def main():
    print("TODO: implement harmonization")

if __name__ == "__main__":
    main()
'''
    )

    ensure_file(
        scripts_dir / "compare_video_pair_harmonized.py",
        '''"""Generate final comparison CSV/PNG outputs only from harmonized checkpoint inputs."""
from pathlib import Path

def main():
    print("TODO: implement comparison plotting/export")

if __name__ == "__main__":
    main()
'''
    )

    print("[OK] Created harmonization checkpoint tree:")
    print(f"  {checkpoint_root}")

if __name__ == "__main__":
    main()