Tau Recovery Debug Summary
=========================

warnings_csv: F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\harmonization_checkpoints\v001_video_pair_fapi_vs_fapi_tempo\compare_outputs\csv\tempo_tau_recovery_warnings.csv
tracks_csv  : F:\Sandy_data\AI segemetation work\Sts_grain morphology_ML\Kinetics\out\FAPI_TEMPO\tracks.csv

warnings_rows_total: 958
warnings_unique_tracks: 958
tracks_rows_total: 29683
tracks_unique_tracks: 977
fit_ok_count: 360
qc_pass_count: 0
qc_fail_count: 958
tau_nonnull_count: 360
r2_nonnull_count: 360

Top failure modes (failed rows only):
  - fit_status:too_few_points: 526
  - unspecified: 360
  - fit_status:insufficient_growth_dynamic: 72

Recommendations:
  [1] QC strictness vs fit formulation: All tracks fail QC despite some fit_status=ok. Inspect representative fits by mode before loosening thresholds. Prioritize diagnosing tau-bound hits and fit-window adequacy.
      evidence: fit_ok=360, qc_pass=0, total=958
  [2] Low R² prevalence: R² appears low overall. Inspect whether growth model matches actual kinetics (single-exponential rise may be too simple), and verify radius extraction smoothness/noise. Consider weighting, smoothing, or monotonic filtering for fit input.
      evidence: median R²=0.879
