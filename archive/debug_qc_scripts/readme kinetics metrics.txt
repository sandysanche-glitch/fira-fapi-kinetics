cd D:\SWITCHdrive\Institution\Sts_grain morphology_ML

python morpho_kinetics_from_crystal_metrics.py ^
  --cm-fapi "crystal_metrics.csv" ^
  --cm-tempo "crystal_metrics 1.csv" ^
  --out-prefix "morpho_kinetics_from_cm" ^
  --t-max-ms 600 ^
  --t-win-ms 60 ^
  --circ-col circularity_distortion ^
  --entropy-col entropy_hm_(bits) ^
  --defect-area-col "defects_area_(µm²)" ^
  --grain-area-col "area_(µm²)"
