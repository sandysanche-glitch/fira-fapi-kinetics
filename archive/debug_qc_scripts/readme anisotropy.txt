cd D:\SWITCHdrive\Institution\Sts_grain morphology_ML

python merge_anisotropy_and_metrics.py ^
  --ani-fapi "anisotropy_out_FAPI_anisotropy_per_grain.csv" ^
  --ani-tempo "anisotropy_out_FAPITEMPO_anisotropy_per_grain.csv" ^
  --metrics-fapi "crystal_metrics.csv" ^
  --metrics-tempo "crystal_metrics 1.csv" ^
  --polar-fapi "anisotropy_out_FAPI_polar_profile.csv" ^
  --polar-tempo "anisotropy_out_FAPITEMPO_polar_profile.csv" ^
  --out-prefix "anisotropy_merged"
