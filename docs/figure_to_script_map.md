
---

## `docs/figure_to_script_map.md`

```markdown
# Figure-to-script map

This document maps manuscript figures to their supporting scripts, processed inputs, and exported outputs.

Because several manuscript figures were assembled from multiple panel-level plots, the entries below distinguish:
- **panel-generation scripts**
- **processed CSV inputs**
- **final or near-final exported outputs**
- **external assembly step**, where applicable

---

## Main-text figures

## Fig. 1 — segmentation-informed workflow

**Purpose**  
Overview schematic of the segmentation-informed analysis workflow.

**Likely source**  
- conceptual / manually assembled workflow figure
- supporting logic documented in:
  - `core_pipeline/segmentation/`
  - `docs/workflow_overview.md`

**Notes**  
This figure is primarily schematic rather than directly regenerated from a single analysis script.

---

## Fig. 2 — grain-resolved morphology

**Purpose**  
Comparison of grain-scale morphology distributions between FAPI and FAPI--TEMPO.

**Panel-generation scripts**
- `figures/fig02_morphology/scripts/plot_crystal_metrics_violins.py`

**Relevant processed inputs**
- descriptor and morphology CSV files in:
  - `data/processed/descriptors/`
  - `data/processed/manuscript_tables/`

**Representative outputs**
- morphology-panel PNGs in:
  - `figures/fig02_morphology/outputs/`
  - or final exported figure assets in `results/` if retained

**Notes**
This figure summarizes grain area, perimeter, circularity distortion, and entropy distributions.

---

## Fig. 3 — measured and morphology-transported kinetics

**Purpose**  
Comparison of video-derived and morphology-transported crystallization kinetics for FAPI and FAPI--TEMPO.

**Panel-generation scripts**
- `figures/fig03_kinetics/scripts/plot_main_morphology_transported_kinetics.py`
- `figures/fig03_kinetics/scripts/make_main_video_bridge_panel_final.py`
- `figures/fig03_kinetics/scripts/plot_combined_x_and_growth.py`
- `figures/fig03_kinetics/scripts/plot_dn_dt_small.py`

**Relevant processed inputs**
- `data/processed/kinetics/burst_metrics.csv`
- `data/processed/kinetics/metrics_burst_metrics.csv`
- `data/processed/kinetics/transported_avrami_fit_params.csv`
- `data/processed/kinetics/transported_avrami_continuous_fit_params.csv`
- morphology–kinetics merged CSVs in `data/processed/kinetics/`

**Representative outputs**
- `figures/fig03_kinetics/outputs/main_morphology_transported_kinetics.png`
- `figures/fig03_kinetics/outputs/main_video_bridge_panel_final.png`

**External assembly**
- Final figure may have been assembled from multiple exported panels outside Python.

**Notes**
This figure includes direct kinetics \(X(t)\), \(dX/dt\), and transported-kinetics representations.

---

## Fig. 4 — spatial fingerprints and crowding-linked kinetics

**Purpose**  
Film-level descriptor comparison and canonical-grain radial fingerprint analysis.

**Panel-generation scripts**
- `figures/fig04_spatial_fingerprints/scripts/plot_large_dataset_veff_correlations.py`
- `figures/fig04_spatial_fingerprints/scripts/plot_annular_entropy_violin.py`
- `figures/fig04_spatial_fingerprints/scripts/plot_full_entropy_onepanel.py`

**Related core analysis scripts**
- `core_pipeline/radial/intragrain_radial_master.py`
- `core_pipeline/radial/intragrain_radial_profiles_from_listjson.py`
- `core_pipeline/radial/intragrain_radial_defect_profiles.py`
- `core_pipeline/radial/intragrain_radial_polar_kinetic_from_listjson.py`
- `core_pipeline/radial/radial_crowding_profiles_from_json.py`
- `core_pipeline/radial/radial_kinetic_heterogeneity_from_json_and_veff_v3.py`
- `core_pipeline/radial/update_radial_kinetic_heterogeneity_from_veff_v2.py`
- `core_pipeline/radial/quantify_radial_profiles.py`

**Relevant processed inputs**
- `data/processed/radial_profiles/`
- `data/processed/descriptors/`
- `data/processed/manuscript_tables/`

**Representative outputs**
- descriptor radar and radial-profile outputs in:
  - `figures/fig04_spatial_fingerprints/outputs/`

**External assembly**
- Final multi-panel figure may have been assembled from separate exported panels.

**Notes**
This figure combines descriptor-level comparison, canonical-grain construction, radial defect/entropy/anisotropy trends, and crowding-linked observables.

---

## Supplementary figures

## SI segmentation / classifier / QC figures

**Related workflow**
- `core_pipeline/segmentation/`
- classifier and QC support retained in archive and manifests

**Representative figures**
- segmentation workflow
- classifier examples
- classifier confusion matrix

**Notes**
These are partly derived from the segmentation-validation workflow and partly assembled from exported outputs.

---

## SI kinetics diagnostics

**Relevant scripts**
- `figures/figSI/scripts/plot_si_comparisons.py`
- `figures/figSI/scripts/plot_si_weighting_comparisons.py`
- `core_pipeline/kinetics/kinetics_analysis.py`
- `core_pipeline/kinetics/avrami_burst_and_local.py`
- `core_pipeline/kinetics/qc_extract_burst_metrics.py`

**Relevant processed inputs**
- kinetics and burst-related CSVs in:
  - `data/processed/kinetics/`

**Representative outputs**
- transported Avrami and SI kinetics outputs in:
  - `figures/figSI/outputs/`

---

## SI descriptor and radial-support figures

**Relevant scripts**
- radial-analysis scripts in `core_pipeline/radial/`
- descriptor scripts in `core_pipeline/descriptors/`

**Relevant processed inputs**
- descriptor CSVs in `data/processed/descriptors/`
- radial-profile CSVs in `data/processed/radial_profiles/`
- manuscript SI tables in `data/processed/manuscript_tables/`

**Notes**
These figures document radial support, weighting robustness, descriptor correlations, and local spatial fields.

---

## SI local geometric distance fields

**Relevant scripts**
- `core_pipeline/radial/compute_phase2_local_distance_fields.py`

**Purpose**
Construct representative distance-to-boundary, distance-to-defect, and distance-to-nucleus maps inside segmented grains.

---

## Remaining work before public release

The mapping above captures the current public-facing structure, but can still be improved by adding, for each final figure:
- exact input CSV filenames
- exact exported panel names
- whether final panel assembly occurred in PowerPoint / Illustrator / other external software
- the final manuscript figure number used in the submitted version