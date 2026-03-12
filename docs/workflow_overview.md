# Workflow overview

This document summarizes the analysis workflow implemented in this repository and explains how the different code, data, and figure folders relate to the manuscript.

The repository is organized around a **segmentation-informed analysis of crystallization kinetics and spatial microstructural fingerprints** in FIRA-processed FAPbI\(_3\) films with and without TEMPO additive.

## Scientific objective

The workflow is designed to connect:

- **time-resolved crystallization dynamics** from in situ bright-field videos,
- **grain-resolved morphology** from large static microscopy datasets,
- and **spatial microstructural fingerprints** extracted from validated grain masks.

The two compositions compared throughout are:

- pristine **FAPI**
- **FAPI--TEMPO**

The central idea is that the final spherulitic microstructure stores a readable signature of the non-equilibrium crystallization pathway, and that this signature can be extracted quantitatively from bright-field microscopy using segmentation-informed computer vision.

---

## Workflow summary

The full workflow proceeds in five main stages:

1. **Segmentation and grain validation**
2. **Tracking and video-based kinetics extraction**
3. **Grain-level descriptor extraction**
4. **Morphology-transported kinetics reconstruction**
5. **Canonical radial fingerprint analysis**

These stages are reflected in the repository layout.

---

## 1. Segmentation and grain validation

### Goal

Convert bright-field microscopy images into validated grain masks suitable for quantitative analysis.

### Inputs

- time-resolved bright-field frames
- static bright-field micrographs
- JSON mask annotations or segmentation exports
- grain, nucleus, and defect mask information where available

### Core logic

Candidate masks are generated and cleaned through a segmentation-informed workflow. The project uses a combination of:
- zero-shot proposal generation,
- mask decoding and cleaning,
- manual or semi-supervised refinement,
- and post-segmentation validity filtering.

The central requirement is that only **physically meaningful spherulitic grains** are retained for downstream analysis.

### Repository location

- `core_pipeline/segmentation/`

### Typical outputs

- cleaned and harmonized grain masks
- ID-mapped mask representations
- mask exports suitable for tracking and descriptor extraction

---

## 2. Tracking and video-based kinetics extraction

### Goal

Use the time-resolved videos to extract the direct experimental crystallization timeline.

### Inputs

- frame-resolved grain masks
- tracked grain identities across frames
- frame timing information

### Core logic

For each tracked grain, the segmented area \(A_i(t)\) is followed over time and converted to an equivalent radius:

\[
R_i(t)=\sqrt{\frac{A_i(t)}{\pi}}.
\]

From this, an effective growth-rate proxy is derived. At the film level, the transformed fraction is computed as:

\[
X(t)=\frac{1}{A_{\mathrm{FOV}}}\sum_i A_i(t),
\]

and the corresponding transformation-rate curve \(dX/dt\) is obtained numerically.

These video-derived observables are used to quantify:
- burst timing,
- burst width,
- synchrony,
- local Avrami-type descriptors,
- and grain-level effective growth-rate distributions.

### Repository location

- `core_pipeline/tracking/`
- `core_pipeline/kinetics/`

### Typical outputs

- per-grain track tables
- \(X(t)\), \(dX/dt\), \(n(t)\), \(dn/dt\)
- burst metrics
- effective Avrami summary parameters

---

## 3. Grain-level descriptor extraction

### Goal

Convert validated grain masks into compact quantitative descriptors of morphology, disorder, texture, and defect loading.

### Inputs

- validated grain masks
- corresponding bright-field images
- defect masks / nucleus masks where available
- per-grain or per-pixel image-derived features

### Core logic

Each grain is characterized by a set of geometric and image-derived descriptors, including:
- area
- perimeter
- equivalent radius
- circularity-related shape measures
- defect fraction
- entropy-based disorder metrics
- anisotropy descriptors such as \(A_{\mathrm{polar}}\) and \(A_{\mathrm{tex}}\)

These descriptors define the grain-scale morphology space used throughout the manuscript.

### Repository location

- `core_pipeline/descriptors/`

### Typical outputs

- per-grain descriptor CSV files
- merged descriptor tables
- manuscript-ready summary tables
- descriptor values used in the radar/fingerprint comparison

---

## 4. Morphology-transported kinetics reconstruction

### Goal

Link the large static grain population to the experimentally observed kinetic timeline, despite having only one crystallization video per composition.

### Inputs

- video-derived kinetic anchor
- static grain-size and descriptor distributions
- merged grain-level morphology–kinetics tables

### Core logic

The morphology-transported kinetics framework maps the larger static grain population onto an effective time axis anchored by the direct video-derived kinetics.

This allows reconstruction of:
- transported \(X(t)\)
- transported \(dX/dt\)
- effective Avrami-style compact kinetics descriptors

Different weighting schemes may be compared, including:
- count weighting
- area weighting
- optional \(R^2\)-style weighting

The purpose is not to replace direct kinetics, but to project the experimentally observed timeline onto the much larger static grain population.

### Repository location

- `core_pipeline/kinetics/`
- supporting processed tables in `data/processed/kinetics/`

### Typical outputs

- transported kinetics curves
- weighting-comparison plots
- transported Avrami fit tables
- burst-comparison summaries

---

## 5. Canonical radial fingerprint analysis

### Goal

Identify where within the final grains the signatures of non-equilibrium growth are stored.

### Inputs

- validated grain masks
- descriptor fields
- defect-associated maps
- crowding metrics
- kinetic-heterogeneity summaries

### Core logic

Each grain is mapped onto a normalized radial coordinate:

\[
r/R_{\mathrm{eq}},
\]

so that grains of different size can be compared on a common basis.

Radial profiles are then computed for observables such as:
- defect fraction
- entropy
- texture anisotropy
- polar anisotropy
- heat-map intensity
- crowding-related measures
- kinetic heterogeneity

This produces the canonical-grain radial fingerprints used in the main text and SI.

A related extension computes local geometric distance fields, such as:
- distance-to-boundary
- distance-to-defect
- distance-to-nucleus

These preserve full intragrain spatial geometry and provide a bridge to possible future pixel-level analyses.

### Repository location

- `core_pipeline/radial/`

### Typical outputs

- radial descriptor profiles
- radial summary tables
- crowding-profile tables
- local distance-field maps

---

## Figure generation logic

The repository preserves a **figure-centric structure** because several manuscript figures were built from multiple panel-level scripts and then assembled externally.

### Figure folders

- `figures/fig02_morphology/`
- `figures/fig03_kinetics/`
- `figures/fig04_spatial_fingerprints/`
- `figures/figSI/`

Each figure folder may contain:
- panel-generation scripts
- relevant processed inputs
- exported outputs
- figure-specific notes

For the detailed mapping between manuscript figures and supporting scripts, see:

- `docs/figure_to_script_map.md`

---

## Processed data products

The repository includes processed CSV outputs organized by analysis role.

### `data/processed/kinetics/`
Contains:
- burst metrics
- transported kinetics tables
- morphology–kinetics merged tables
- Avrami-related outputs

### `data/processed/descriptors/`
Contains:
- anisotropy tables
- entropy summaries
- merged descriptor CSVs
- descriptor-level correlation outputs

### `data/processed/radial_profiles/`
Contains:
- radial entropy profiles
- radial crowding profiles
- radial kinetic heterogeneity profiles
- radial polar / intragrain profile summaries

### `data/processed/manuscript_tables/`
Contains:
- final or near-final CSV tables supporting manuscript and SI values

---

## Supporting documentation

The main documentation files are:

- `README.md`  
  Project-level overview and repository structure

- `docs/figure_to_script_map.md`  
  Mapping between figures, scripts, processed inputs, and exported outputs

- `docs/data_policy.md`  
  What is included in the repository and what is excluded or only partially represented

- `docs/README_scope.md`  
  Scope note preserved from repository construction / harmonization

- `docs/decisions_log.md`  
  Curated decision trail for organization and file-selection logic

---

## Archived material

The repository also retains archived material for transparency and traceability:

- `archive/exploratory_versions/`
- `archive/debug_qc_scripts/`
- `archive/manifests_support/`
- `archive/notes_docs/`
- `archive/old_panels/`

These folders contain:
- exploratory or versioned scripts
- QC-only artefacts
- old panel variants
- support manifests
- non-core notes and legacy material

They are preserved to document workflow evolution, but they are not part of the intended public-facing core pipeline.

---

## Recommended reading order

For a reader new to the repository, the best order is:

1. `README.md`
2. `docs/workflow_overview.md`
3. `docs/figure_to_script_map.md`
4. `docs/data_policy.md`
5. `core_pipeline/`
6. `figures/`
7. `data/processed/`

---

## Final note

This repository is intended to make the manuscript workflow understandable, traceable, and reusable at the level of:
- core analysis logic,
- processed outputs,
- and figure support.

It is not necessarily a single-command rebuild of the entire project from raw acquisition to final manuscript layout, especially where:
- raw inputs are too large for GitHub,
- some figures were assembled externally,
- or legacy support material has been archived rather than integrated into the core pipeline.