# FIRA FAPI crystallization kinetics and spatial fingerprints

This repository contains the analysis code, processed data products, and figure-support files used for the manuscript on crystallization kinetics and spatial fingerprints in flash infrared annealed (FIRA) FAPbI\(_3\) films with and without TEMPO additive.

## Overview

The study combines bright-field optical microscopy, segmentation-informed computer vision, grain-resolved descriptor extraction, and video-anchored kinetics reconstruction to compare:

- pristine FAPI
- FAPI--TEMPO

The workflow links:
- time-resolved crystallization dynamics,
- grain-scale morphology,
- radial microstructural fingerprints,
- and effective growth-rate heterogeneity.

The repository is organized to preserve both:
- a **core reproducible pipeline**, and
- a **figure-centric structure**, since several manuscript figures were assembled from multiple panel-level scripts and then combined externally.

## Repository structure

```text
core_pipeline/
  segmentation/   # mask generation, JSON cleaning, ID mapping
  tracking/       # grain tracking and per-grain growth reconstruction
  kinetics/       # X(t), dX/dt, burst metrics, local Avrami analysis
  descriptors/    # A_polar, A_tex, entropy, crowding, descriptor tables
  radial/         # canonical-grain mapping, radial fingerprints, local fields

figures/
  fig02_morphology/              # morphology figure panel scripts and outputs
  fig03_kinetics/                # kinetics figure panel scripts and outputs
  fig04_spatial_fingerprints/    # spatial-fingerprint panel scripts and outputs
  figSI/                         # supplementary-figure support scripts and outputs

data/
  processed/
    kinetics/
    descriptors/
    radial_profiles/
    manuscript_tables/
  manifests/

docs/
  workflow_overview.md
  figure_to_script_map.md
  data_policy.md
  README_scope.md
  decisions_log.md
  source_manifest_curated.csv
  source_manifest_raw.csv
  kept_manifest_refined.csv
  archive_manifest_refined.csv

archive/
  exploratory_versions/
  debug_qc_scripts/
  manifests_support/
  notes_docs/
  old_panels/