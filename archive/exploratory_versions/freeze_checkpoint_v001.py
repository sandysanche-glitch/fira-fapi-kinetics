#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
freeze_checkpoint_v001.py

Creates a dated, hash-locked frozen snapshot inside:
  harmonization_checkpoints/v001_video_pair_fapi_vs_fapi_tempo/frozen/

Copies:
  - compare_outputs/ (csv + png + manifests + warnings)
  - tau_recovery_debug/ (if exists)
  - compare_outputs/csv/tempo_tau_recovery_*.csv (if exists)
  - canonical_inputs/
  - harmonized_tables/
  - manifests/
  - scripts/ (optionally filtered)
  - README_scope.md, decisions_log.md (if exist)

Writes:
  - freeze_manifest.csv  (what was copied)
  - freeze_hash_manifest.csv (sha256 for every copied file)
  - FREEZE_README.txt (what this is + how to reproduce)

Safe:
  - file/dir existence checks
  - clear warnings
  - no dependence on absolute paths outside checkpoint

Usage:
  python freeze_checkpoint_v001.py
Optional:
  python freeze_checkpoint_v001.py --checkpoint_dir "F:\\...\\v001_video_pair_fapi_vs_fapi_tempo"
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


# -----------------------------
# Config (edit if you want)
# -----------------------------
DEFAULT_CHECKPOINT_DIR = None  # auto-detect relative to this script if None

FREEZE_PARENT_NAME = "frozen"
FREEZE_TAG_PREFIX = "v001_video_pair_freeze"

# What to copy from the checkpoint root (relative paths)
# (Some may not exist; script will warn and skip.)
COPY_DIRS = [
    "compare_outputs",
    "tau_recovery_debug",
    "canonical_inputs",
    "harmonized_tables",
    "manifests",
    "scripts",
]

COPY_FILES = [
    "README_scope.md",
    "decisions_log.md",
    # any other top-level notes you maintain
]

# Optional: exclude big/junk patterns inside scripts snapshot
EXCLUDE_DIR_NAMES = {
    "__pycache__",
    ".ipynb_checkpoints",
}
EXCLUDE_FILE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".tmp",
    ".log",
}


# -----------------------------
# Utilities
# -----------------------------
def now_stamp() -> str:
    # Local time stamp; safe for filenames
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_excluded_dirname(name: str) -> bool:
    return name in EXCLUDE_DIR_NAMES


def is_excluded_filename(name: str) -> bool:
    lower = name.lower()
    return any(lower.endswith(sfx) for sfx in EXCLUDE_FILE_SUFFIXES)


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def relpath_safe(path: str, root: str) -> str:
    try:
        return os.path.relpath(path, root)
    except Exception:
        return path


def copy_file(src: str, dst: str) -> None:
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)


def copy_tree(src_dir: str, dst_dir: str) -> Tuple[int, int]:
    """
    Returns (files_copied, files_skipped).
    Skips excluded dirs and excluded suffixes.
    """
    files_copied = 0
    files_skipped = 0

    for root, dirs, files in os.walk(src_dir):
        # filter dirs in-place
        dirs[:] = [d for d in dirs if not is_excluded_dirname(d)]

        rel = os.path.relpath(root, src_dir)
        out_root = os.path.join(dst_dir, rel) if rel != "." else dst_dir
        ensure_dir(out_root)

        for fn in files:
            if is_excluded_filename(fn):
                files_skipped += 1
                continue
            s = os.path.join(root, fn)
            d = os.path.join(out_root, fn)
            copy_file(s, d)
            files_copied += 1

    return files_copied, files_skipped


@dataclass
class CopiedItem:
    kind: str            # "dir" or "file"
    src_rel: str
    dst_rel: str
    status: str          # "copied" or "missing" or "skipped"
    note: str = ""


def find_checkpoint_dir_from_script() -> str:
    """
    Assumes this script lives in: <checkpoint>/scripts/freeze_checkpoint_v001.py
    """
    here = os.path.abspath(os.path.dirname(__file__))
    # If it's in scripts/, then checkpoint is parent
    parent = os.path.abspath(os.path.join(here, ".."))
    return parent


def write_csv(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def collect_all_files(root_dir: str) -> List[str]:
    out = []
    for r, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if not is_excluded_dirname(d)]
        for fn in files:
            if is_excluded_filename(fn):
                continue
            out.append(os.path.join(r, fn))
    return out


def print_box(title: str) -> None:
    line = "=" * 100
    print(line)
    print(title)
    print(line)


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", default=DEFAULT_CHECKPOINT_DIR, help="Path to v001_video_pair_fapi_vs_fapi_tempo")
    ap.add_argument("--tag", default=None, help="Optional extra tag in folder name (e.g., 'tau_enabled')")
    args = ap.parse_args()

    checkpoint_dir = args.checkpoint_dir
    if not checkpoint_dir:
        checkpoint_dir = find_checkpoint_dir_from_script()

    checkpoint_dir = os.path.abspath(checkpoint_dir)

    print_box("FREEZE CHECKPOINT v001")
    print(f"[INFO] CHECKPOINT_DIR : {checkpoint_dir}")

    if not os.path.isdir(checkpoint_dir):
        print(f"[ERROR] Not a directory: {checkpoint_dir}")
        return 2

    freeze_parent = os.path.join(checkpoint_dir, FREEZE_PARENT_NAME)
    ensure_dir(freeze_parent)

    stamp = now_stamp()
    extra = f"_{args.tag}" if args.tag else ""
    freeze_name = f"{FREEZE_TAG_PREFIX}_{stamp}{extra}"
    freeze_dir = os.path.join(freeze_parent, freeze_name)
    ensure_dir(freeze_dir)

    print(f"[INFO] FREEZE_DIR     : {freeze_dir}")

    copied: List[CopiedItem] = []
    warnings: List[str] = []

    # Copy directories
    for rel in COPY_DIRS:
        src = os.path.join(checkpoint_dir, rel)
        dst = os.path.join(freeze_dir, rel)
        if not os.path.exists(src):
            warnings.append(f"Missing directory: {rel}")
            copied.append(CopiedItem(kind="dir", src_rel=rel, dst_rel=rel, status="missing"))
            continue
        if not os.path.isdir(src):
            warnings.append(f"Expected directory but got file: {rel}")
            copied.append(CopiedItem(kind="dir", src_rel=rel, dst_rel=rel, status="skipped", note="not a directory"))
            continue

        files_copied, files_skipped = copy_tree(src, dst)
        copied.append(CopiedItem(
            kind="dir",
            src_rel=rel,
            dst_rel=rel,
            status="copied",
            note=f"files_copied={files_copied}, files_skipped={files_skipped}",
        ))
        print(f"[OK] Copied dir: {rel}  ({files_copied} files, skipped {files_skipped})")

    # Copy top-level files
    for rel in COPY_FILES:
        src = os.path.join(checkpoint_dir, rel)
        dst = os.path.join(freeze_dir, rel)
        if not os.path.exists(src):
            warnings.append(f"Missing file: {rel}")
            copied.append(CopiedItem(kind="file", src_rel=rel, dst_rel=rel, status="missing"))
            continue
        if not os.path.isfile(src):
            warnings.append(f"Expected file but got dir: {rel}")
            copied.append(CopiedItem(kind="file", src_rel=rel, dst_rel=rel, status="skipped", note="not a file"))
            continue
        copy_file(src, dst)
        copied.append(CopiedItem(kind="file", src_rel=rel, dst_rel=rel, status="copied"))
        print(f"[OK] Copied file: {rel}")

    # Extra: make sure tau recovery summary/warnings are captured even if you didn’t put them under tau_recovery_debug
    # Many runs store these under compare_outputs/csv/
    extra_tau_candidates = [
        os.path.join(checkpoint_dir, "compare_outputs", "csv", "tempo_tau_recovery_summary.csv"),
        os.path.join(checkpoint_dir, "compare_outputs", "csv", "tempo_tau_recovery_warnings.csv"),
    ]
    for src in extra_tau_candidates:
        if os.path.isfile(src):
            rel = relpath_safe(src, checkpoint_dir)
            dst = os.path.join(freeze_dir, rel)
            copy_file(src, dst)
            copied.append(CopiedItem(kind="file", src_rel=rel, dst_rel=rel, status="copied", note="extra_tau_candidate"))
            print(f"[OK] Copied extra tau QC file: {rel}")

    # Write freeze manifest
    manifest_rows = []
    for it in copied:
        manifest_rows.append({
            "kind": it.kind,
            "src_rel": it.src_rel,
            "dst_rel": it.dst_rel,
            "status": it.status,
            "note": it.note,
        })

    manifest_path = os.path.join(freeze_dir, "freeze_manifest.csv")
    write_csv(manifest_path, manifest_rows, ["kind", "src_rel", "dst_rel", "status", "note"])
    print(f"[OK] Wrote: {manifest_path}")

    # Hash-lock everything copied into freeze_dir
    print("[INFO] Hashing frozen files (sha256) ...")
    all_files = collect_all_files(freeze_dir)
    hash_rows = []
    for fp in sorted(all_files):
        # skip the hash manifest itself (it doesn't exist yet) — but after writing it, it would change anyway
        rel = relpath_safe(fp, freeze_dir)
        if rel in {"freeze_hash_manifest.csv"}:
            continue
        try:
            h = sha256_file(fp)
        except Exception as e:
            warnings.append(f"Hash failed for {rel}: {e}")
            continue
        hash_rows.append({
            "rel_path": rel.replace("\\", "/"),
            "sha256": h,
            "bytes": os.path.getsize(fp),
        })

    hash_path = os.path.join(freeze_dir, "freeze_hash_manifest.csv")
    write_csv(hash_path, hash_rows, ["rel_path", "sha256", "bytes"])
    print(f"[OK] Wrote: {hash_path}  (files_hashed={len(hash_rows)})")

    # Write freeze readme
    readme_path = os.path.join(freeze_dir, "FREEZE_README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("FROZEN CHECKPOINT SNAPSHOT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Created: {stamp}\n")
        f.write(f"Source checkpoint: {checkpoint_dir}\n")
        f.write(f"Freeze folder: {freeze_dir}\n\n")

        f.write("Purpose:\n")
        f.write("  Immutable snapshot of the video-pair harmonization+comparison state.\n")
        f.write("  Hash-locks allow verification that plots/tables were generated from exactly these files.\n\n")

        f.write("Key entry points (typical):\n")
        f.write("  - compare_outputs/png/*\n")
        f.write("  - compare_outputs/csv/*\n")
        f.write("  - canonical_inputs/*\n")
        f.write("  - harmonized_tables/*\n")
        f.write("  - scripts/*  (exact scripts used at freeze time)\n\n")

        if warnings:
            f.write("Warnings during freeze:\n")
            for w in warnings:
                f.write(f"  - {w}\n")
        else:
            f.write("Warnings during freeze: none\n")

    print(f"[OK] Wrote: {readme_path}")

    # Write warnings log
    warn_path = os.path.join(freeze_dir, "freeze_warnings.txt")
    with open(warn_path, "w", encoding="utf-8") as f:
        for w in warnings:
            f.write(w + "\n")
    print(f"[OK] Wrote: {warn_path}")

    print_box("FREEZE COMPLETE")
    print(f"[OK] Frozen snapshot: {freeze_dir}")
    if warnings:
        print(f"[WARN] {len(warnings)} warnings (see freeze_warnings.txt)")
    else:
        print("[OK] No warnings.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())