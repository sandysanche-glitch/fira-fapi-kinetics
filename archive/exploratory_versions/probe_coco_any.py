import json
from pathlib import Path
import sys

def read_flexible(p: Path):
    txt = p.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return {}
    try:
        return json.loads(txt)
    except Exception:
        # Try JSONL
        out = []
        with p.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line=line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
        return {"__jsonl__": out}

def summarize(path: Path, max_show=5):
    data = read_flexible(path)
    anns = []
    cats = []
    images = []
    if isinstance(data, dict):
        anns = data.get("annotations", [])
        cats = data.get("categories", [])
        images = data.get("images", [])
        if not anns and "__jsonl__" in data:
            for d in data["__jsonl__"]:
                if isinstance(d, dict):
                    anns += d.get("annotations", []) if "annotations" in d else [d]
                    cats += d.get("categories", [])
    elif isinstance(data, list):
        anns = data
    print(f"\n== {path.name} ==")
    print(f"keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
    print(f"images: {len(images)}  categories: {len(cats)}  annotations: {len(anns)}")
    # Categories summary
    id2name = {}
    for c in cats:
        cid = c.get("id")
        nm  = c.get("name")
        id2name[cid] = nm
    if id2name:
        print("categories id->name:", id2name)
    # Count by category_id and by name (if present)
    by_id = {}
    by_name = {}
    for a in anns:
        if not isinstance(a, dict): continue
        cid = a.get("category_id")
        nm  = a.get("category_name") or a.get("name")
        by_id[cid] = by_id.get(cid, 0)+1
        if nm:
            by_name[nm] = by_name.get(nm, 0)+1
    print("by category_id:", by_id)
    print("by category_name/name:", by_name)
    # Show a few example annotations
    print("-- sample annotations --")
    shown = 0
    for a in anns:
        if not isinstance(a, dict): 
            continue
        keys = list(a.keys())
        print({k:a[k] for k in keys[:12]})
        shown += 1
        if shown >= max_show:
            break

if __name__=="__main__":
    if len(sys.argv)<2:
        print("Usage: python probe_coco_any.py <folder_or_json>")
        sys.exit(1)
    p = Path(sys.argv[1])
    if p.is_dir():
        files = sorted(list(p.glob("*.json")))
        if not files:
            print("[WARN] No JSON files in folder.")
        for f in files[:10]:
            summarize(f)
        if len(files)>10:
            print("...(showing first 10)")
    else:
        summarize(p)
