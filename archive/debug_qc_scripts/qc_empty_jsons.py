import os, glob, json, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", required=True)
    ap.add_argument("--pattern", default="frame_*_idmapped.json")
    args = ap.parse_args()

    fs = sorted(glob.glob(os.path.join(args.json_dir, args.pattern)))
    n = len(fs)
    empty = 0
    bad = 0
    nonempty = 0

    for f in fs:
        try:
            with open(f, "r") as fh:
                d = json.load(fh)
            if isinstance(d, list) and len(d) == 0:
                empty += 1
            elif isinstance(d, list):
                nonempty += 1
            else:
                bad += 1
        except Exception:
            bad += 1

    denom = max(n, 1)
    print("json_dir:", args.json_dir)
    print("files:", n)
    print("nonempty:", nonempty)
    print("empty:", empty)
    print("bad:", bad)
    print("empty_or_bad:", empty + bad, "frac:", (empty + bad) / denom)

if __name__ == "__main__":
    main()
