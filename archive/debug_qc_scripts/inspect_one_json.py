import json
import os

# <<< EDIT THIS IF NEEDED >>>
json_path = r"D:\SWITCHdrive\Institution\Sts_grain morphology_ML\comparative datasets\FAPI\FAPI_0.json"

print("Inspecting:", json_path)
with open(json_path, "r") as f:
    data = json.load(f)

print("Top-level type:", type(data))

if isinstance(data, dict):
    print("Top-level keys:", list(data.keys()))
    for k, v in data.items():
        print(f"  key={k!r}, type={type(v)}, len={len(v) if hasattr(v,'__len__') else 'n/a'}")
        break  # just first key for sanity
elif isinstance(data, list):
    print("List length:", len(data))
    if len(data) > 0 and isinstance(data[0], dict):
        print("First element keys:", list(data[0].keys()))
else:
    print("Unexpected JSON structure")
