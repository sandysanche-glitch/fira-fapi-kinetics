#!/usr/bin/env python3
"""
make_video_with_scalebar_and_time.py

Create an MP4 from a directory of image frames, adding:
- a scale bar (e.g., 50 µm) using proper Unicode "µ" via PIL
- a time counter (ms / s / frame index)

Example:
python make_video_with_scalebar_and_time.py ^
  --in-dir "F:\Sandy_data\Sandy\12.11.2025\sequences\v5\FAPI-files_shorten" ^
  --out-mp4 "F:\Sandy_data\Sandy\12.11.2025\sequences\v5\FAPI_scalebar_time.mp4" ^
  --fps 20 ^
  --px-per-um 2.20014 ^
  --scale-bar-um 50 ^
  --scale-bar-label ^
  --time-mode ms ^
  --t0-ms 0 ^
  --dt-ms 5 ^
  --resize-width 1200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# -----------------------------
# Utilities
# -----------------------------
def list_frames(in_dir: Path,
                exts: Tuple[str, ...],
                recursive: bool = True) -> List[Path]:
    if not in_dir.exists() or not in_dir.is_dir():
        raise FileNotFoundError(f"--in-dir does not exist or is not a directory: {in_dir}")

    if recursive:
        cand = []
        for ext in exts:
            cand.extend(in_dir.rglob(f"*{ext}"))
    else:
        cand = []
        for ext in exts:
            cand.extend(in_dir.glob(f"*{ext}"))

    # Filter to files and sort naturally by name
    cand = [p for p in cand if p.is_file()]

    # Natural-ish sort: split digits
    def keyfun(p: Path):
        s = p.name
        out = []
        buf = ""
        is_digit = False
        for ch in s:
            d = ch.isdigit()
            if buf and d != is_digit:
                out.append(int(buf) if is_digit else buf.lower())
                buf = ch
                is_digit = d
            else:
                buf += ch
                is_digit = d
        if buf:
            out.append(int(buf) if is_digit else buf.lower())
        return out

    cand.sort(key=keyfun)
    return cand


def read_bgr(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        # fallback
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    return img


def resize_keep_aspect(img_bgr: np.ndarray, target_w: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if w == target_w:
        return img_bgr
    scale = target_w / float(w)
    target_h = int(round(h * scale))
    return cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)


def load_font(font_px: int) -> ImageFont.FreeTypeFont:
    # DejaVuSans usually exists with PIL/Python distributions and supports "µ"
    for name in ["DejaVuSans.ttf", "arial.ttf", "LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(name, font_px)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_text_unicode_bgr(img_bgr: np.ndarray,
                          text: str,
                          xy: Tuple[int, int],
                          font_px: int = 28,
                          color_bgr: Tuple[int, int, int] = (255, 255, 255),
                          stroke_px: int = 2,
                          stroke_color_bgr: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Draw Unicode text (e.g., µ) onto a BGR OpenCV image using PIL.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = load_font(font_px)

    fill_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    stroke_rgb = (stroke_color_bgr[2], stroke_color_bgr[1], stroke_color_bgr[0])

    # PIL stroke supported in modern Pillow; if not, fallback to manual shadow
    try:
        draw.text(xy, text, font=font, fill=fill_rgb,
                  stroke_width=stroke_px, stroke_fill=stroke_rgb)
    except TypeError:
        # crude outline fallback
        x, y = xy
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                draw.text((x + dx, y + dy), text, font=font, fill=stroke_rgb)
        draw.text(xy, text, font=font, fill=fill_rgb)

    out_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return out_bgr


def format_time(i: int, t0_ms: float, dt_ms: float, mode: str) -> str:
    t_ms = t0_ms + i * dt_ms
    if mode == "ms":
        return f"{t_ms:.0f} ms"
    if mode == "s":
        return f"{t_ms/1000.0:.2f} s"
    if mode == "frame":
        return f"frame {i}"
    raise ValueError(f"Unknown --time-mode: {mode}")


def draw_scale_bar(img_bgr: np.ndarray,
                   px_per_um: float,
                   scale_bar_um: float,
                   show_label: bool,
                   bar_thickness_px: int = 12,
                   margin_px: int = 30,
                   font_px: int = 28) -> np.ndarray:
    """
    Draw a scale bar in the bottom-left corner.
    Uses PIL for label so µ renders correctly.
    """
    h, w = img_bgr.shape[:2]
    length_px = int(round(scale_bar_um * px_per_um))
    length_px = max(5, min(length_px, w - 2 * margin_px))

    x0 = margin_px
    y0 = h - margin_px  # baseline
    x1 = x0 + length_px
    y1 = y0

    # Bar rectangle coordinates (filled)
    top = y1 - bar_thickness_px
    left = x0
    right = x1
    bottom = y1

    # Draw filled white rectangle with black outline for contrast
    cv2.rectangle(img_bgr, (left, top), (right, bottom), (255, 255, 255), thickness=-1)
    cv2.rectangle(img_bgr, (left, top), (right, bottom), (0, 0, 0), thickness=2)

    if show_label:
        label = f"{scale_bar_um:g} µm"  # real mu
        # place label above bar, left-aligned
        tx = left
        ty = top - int(font_px * 1.4)
        ty = max(5, ty)
        img_bgr = draw_text_unicode_bgr(
            img_bgr, label, (tx, ty),
            font_px=font_px,
            color_bgr=(255, 255, 255),
            stroke_px=3,
            stroke_color_bgr=(0, 0, 0),
        )

    return img_bgr


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="Directory containing frames (images). Can be nested.")
    ap.add_argument("--out-mp4", required=True, help="Output MP4 path.")
    ap.add_argument("--fps", type=float, default=20.0, help="Video frames per second.")
    ap.add_argument("--px-per-um", type=float, required=True, help="Pixels per micrometer (µm).")
    ap.add_argument("--scale-bar-um", type=float, default=50.0, help="Scale bar length in µm.")
    ap.add_argument("--scale-bar-label", action="store_true", help="Draw the scale bar label (e.g., 50 µm).")

    ap.add_argument("--time-mode", choices=["ms", "s", "frame"], default="ms",
                    help="How to display time overlay.")
    ap.add_argument("--t0-ms", type=float, default=0.0, help="Time at first frame (ms).")
    ap.add_argument("--dt-ms", type=float, default=5.0, help="Time increment per frame (ms).")

    ap.add_argument("--time-prefix", default="", help='Prefix for time text (e.g. "t=").')
    ap.add_argument("--time-pos", choices=["tl", "tr", "bl", "br"], default="tl",
                    help="Time text position: top-left/top-right/bottom-left/bottom-right.")

    ap.add_argument("--resize-width", type=int, default=0,
                    help="If >0, resize frames to this width (keeps aspect ratio).")

    ap.add_argument("--exts", nargs="*", default=[".png", ".jpg", ".jpeg", ".tif", ".tiff"],
                    help="Frame extensions to include.")
    ap.add_argument("--no-recursive", action="store_true",
                    help="Disable recursive frame search; only look in --in-dir itself.")
    ap.add_argument("--font-px", type=int, default=30, help="Font size for overlays (pixels).")
    ap.add_argument("--bar-thickness-px", type=int, default=12, help="Scale bar thickness in pixels.")
    ap.add_argument("--margin-px", type=int, default=30, help="Overlay margin from edges in pixels.")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_mp4 = Path(args.out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    exts = tuple(e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.exts)
    frames = list_frames(in_dir, exts=exts, recursive=(not args.no_recursive))
    if not frames:
        raise FileNotFoundError(f"No frames found in {in_dir} with extensions {exts} "
                                f"(recursive={not args.no_recursive}).")

    print(f"[INFO] Found {len(frames)} frames. First 5:")
    for p in frames[:5]:
        print("  ", p)

    first = read_bgr(frames[0])
    if args.resize_width and args.resize_width > 0:
        first = resize_keep_aspect(first, args.resize_width)

    h, w = first.shape[:2]

    # VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, float(args.fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_mp4}. "
                           f"Try a different output path or codec.")

    # Time text anchor
    def time_xy(img_w: int, img_h: int, text: str) -> Tuple[int, int]:
        # Rough width estimate for placement; we will just pad generously.
        m = args.margin_px
        if args.time_pos == "tl":
            return (m, m)
        if args.time_pos == "tr":
            return (max(m, img_w - m - 400), m)
        if args.time_pos == "bl":
            return (m, max(m, img_h - m - 50))
        if args.time_pos == "br":
            return (max(m, img_w - m - 400), max(m, img_h - m - 50))
        return (m, m)

    try:
        for i, fp in enumerate(frames):
            img = read_bgr(fp)
            if args.resize_width and args.resize_width > 0:
                img = resize_keep_aspect(img, args.resize_width)

            # Defensive: enforce same size
            if img.shape[1] != w or img.shape[0] != h:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

            # Scale bar
            img = draw_scale_bar(
                img,
                px_per_um=args.px_per_um,
                scale_bar_um=args.scale_bar_um,
                show_label=bool(args.scale_bar_label),
                bar_thickness_px=args.bar_thickness_px,
                margin_px=args.margin_px,
                font_px=args.font_px,
            )

            # Time overlay (PIL so unicode safe if you later add symbols)
            ttxt = args.time_prefix + format_time(i, args.t0_ms, args.dt_ms, args.time_mode)
            tx, ty = time_xy(w, h, ttxt)
            img = draw_text_unicode_bgr(
                img, ttxt, (tx, ty),
                font_px=args.font_px,
                color_bgr=(255, 255, 255),
                stroke_px=3,
                stroke_color_bgr=(0, 0, 0),
            )

            writer.write(img)

            if (i + 1) % 200 == 0:
                print(f"[INFO] Wrote {i+1}/{len(frames)} frames...")

    finally:
        writer.release()

    print(f"[OK] Saved video: {out_mp4}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)