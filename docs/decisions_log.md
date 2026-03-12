# Decisions Log
Created: 2026-02-24 15:56:43

## Initial placeholders
- Calibration (um/px): TBD (target currently 0.065 um/px for both)
- Time bin width (ms): TBD (e.g., 20 ms)
- Active-grains source type: TBD (tracks.csv vs active_tracks.csv)
- Active criterion: TBD
- Tau source harmonization: TBD (true tau-fit table required for both)
- Include tau panel in final compare: TBD

## Manifest build run (2026-02-24 16:16:13)
- Calibration reference retained: 0.065 um/px for both datasets.
- This script snapshots sources and canonical inputs only (no harmonization transform yet).
- Warnings observed:
  - FAPI_TEMPO missing active_tracks_source.csv. Harmonize active-grains source/type before robust comparison.
  - FAPI_TEMPO missing true tau_fits.csv (publication-robust tau comparison not yet valid).
