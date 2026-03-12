#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pathlib import Path

def main():
    json_path = Path(
        r"D:\SWITCHdrive\Institution\Sts_grain morphology_ML\comparative datasets\FAPI\FAPI_0.json"
    )

    print(f"[INFO] Inspecting {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    # If the top-level is a dict, print the keys
    if isinstance(data, dict):
        print("[INFO] Top-level type: dict")
        print("[INFO] Keys:", list(data.keys()))
        if "categories" in data:
            print("\n[INFO] Categories:")
            for c in data["categories"]:
                print(c)
        if "annotations" in data:
            print("\n[INFO] First 3 annotations:")
            for ann in data["annotations"][:3]:
                print(ann)
    # If the top-level is a list, print the first element
    elif isinstance(data, list):
        print("[INFO] Top-level type: list")
        print(f"[INFO] Length: {len(data)}")
        print("[INFO] First element:")
        print(data[0])
    else:
        print("[WARN] Unexpected top-level type:", type(data))

if __name__ == "__main__":
    main()
