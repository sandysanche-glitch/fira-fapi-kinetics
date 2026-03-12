# Data policy

This document summarizes which classes of data are included in the repository and which are excluded or only partially represented.

## Included in the repository

The repository includes the following data products required for manuscript-level traceability and partial reproducibility:

- processed CSV outputs used in descriptor extraction, kinetics analysis, radial fingerprinting, and manuscript tables
- selected exported figure assets and panel-level outputs
- manifests and workflow notes documenting curated and raw-source organization
- selected representative analysis-ready inputs where practical

These files are primarily stored in:
- `data/processed/`
- `data/manifests/`
- `docs/`

## Typically not included in full

The complete project workflow may rely on raw or intermediate inputs that are too large or too numerous for a lightweight GitHub repository. These may include:

- full raw crystallization video files
- full raw bright-field microscopy image libraries
- full-resolution scanning-stage image collections
- large intermediate segmentation exports
- temporary or debugging analysis outputs
- training artefacts or model checkpoints not required for manuscript-level interpretation

These large or secondary items are generally not required to understand the manuscript workflow and are therefore not necessarily included in the public repository.

## Included instead of full raw data

Where full raw datasets are not included, the repository provides:
- processed CSV outputs used directly in figures and tables
- curated manifests describing relevant source files
- a figure-to-script map documenting how the manuscript outputs were generated
- archived exploratory and support files retained for traceability where useful

## Reproducibility scope

This repository is intended to support:

- understanding of the analysis workflow
- inspection of the core scripts used in the study
- access to processed outputs that support manuscript figures and tables
- practical traceability between scripts, processed CSVs, and exported figures

It is **not guaranteed** to provide a single-command full rebuild of the entire project from raw acquisition to final manuscript figures, especially where:
- raw inputs are too large for GitHub,
- figure panels were assembled externally,
- or some legacy intermediate steps were retained only as archived support material.

## External assembly note

Some final manuscript figures were assembled from multiple panel-level plots generated in Python and then combined externally. This is documented in:
- `docs/figure_to_script_map.md`

## Archived support material

Files that were useful during development but are not part of the public-facing workflow have been moved into:
- `archive/`

This includes exploratory scripts, older panel variants, QC summaries, and support manifests.

## Future archival option

For a later archival release, a larger companion dataset could be deposited externally and linked from this repository, for example through:
- Zenodo
- institutional storage
- project-specific data archive

If such an archive is created, this file should be updated with:
- accession / DOI
- contents of the external dataset
- relation between the external dataset and this GitHub repository