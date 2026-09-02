#!/usr/bin/env python3
"""
把 data/customedFace 的圖片拼成「Label | Example(四張)」表格圖。

用法:
  conda run -n SFM python scripts/make_customed_face_table.py
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "data" / "customedFace"
OUT_PATH = ROOT / "plots" / "customedFace_examples_table.png"

LABEL_ORDER = [
    ("masked_face", "Masked face"),
    ("glasses_face", "Glasses face"),
    ("sunglasses_face", "Sunglasses face"),
    ("partial_face", "Partial face"),
    ("sketch_face", "Sketch face"),
]
SUBTYPE_RE = re.compile(
    r"^(masked_face|glasses_face|sunglasses_face|partial_face|sketch_face|normal_face)",
    re.I,
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def collect_by_subtype(image_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(image_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        m = SUBTYPE_RE.match(path.stem)
        if not m:
            continue
        groups[m.group(1).lower()].append(path)
    return groups


def load_square(path: Path, size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    nw, nh = max(size, int(round(w * scale))), max(size, int(round(h * scale)))
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - size) // 2
    top = (nh - size) // 2
    return img.crop((left, top, left + size, top + size))


def get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf" if bold
        else "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def make_table(
    image_dir: Path,
    out_path: Path,
    n_examples: int = 4,
    thumb_size: int = 160,
) -> Path:
    groups = collect_by_subtype(image_dir)
    rows = []
    for key, label in LABEL_ORDER:
        paths = groups.get(key, [])
        if not paths:
            continue
        rows.append((label, paths[:n_examples]))
    if not rows:
        raise RuntimeError(f"{image_dir} 找不到可用圖片")

    # 版面參數（像素）
    border = 2
    pad = 14
    gap = 12
    label_col_w = 220
    header_h = 56
    row_h = pad * 2 + thumb_size
    example_col_w = pad * 2 + n_examples * thumb_size + (n_examples - 1) * gap

    table_w = border + label_col_w + border + example_col_w + border
    table_h = border + header_h + len(rows) * row_h + border

    canvas = Image.new("RGB", (table_w, table_h), "white")
    draw = ImageDraw.Draw(canvas)
    font_header = get_font(28, bold=True)
    font_label = get_font(24, bold=False)

    # 外框
    draw.rectangle([0, 0, table_w - 1, table_h - 1], outline="black", width=border)

    # 垂直分隔：Label | Example（只有這一條）
    x_split = border + label_col_w
    draw.line([(x_split, 0), (x_split, table_h - 1)], fill="black", width=border)

    # Header 底線
    y = border + header_h
    draw.line([(0, y), (table_w - 1, y)], fill="black", width=border)

    # Header 文字
    for text, x0, x1 in (
        ("Label", border, x_split),
        ("Example", x_split + border, table_w - border),
    ):
        tw, th = text_size(draw, text, font_header)
        draw.text(
            (x0 + (x1 - x0 - tw) // 2, border + (header_h - th) // 2),
            text, fill="black", font=font_header,
        )

    # Data rows
    for i, (label, paths) in enumerate(rows):
        y0 = border + header_h + i * row_h
        y1 = y0 + row_h
        if i < len(rows) - 1:
            draw.line([(0, y1), (table_w - 1, y1)], fill="black", width=border)

        # Label
        tw, th = text_size(draw, label, font_label)
        draw.text(
            (border + (label_col_w - tw) // 2, y0 + (row_h - th) // 2),
            label, fill="black", font=font_label,
        )

        # Examples：緊貼 Example 欄左側，不再置中留白
        x = x_split + border + pad
        y_img = y0 + pad
        for path in paths:
            thumb = load_square(path, thumb_size)
            canvas.paste(thumb, (x, y_img))
            x += thumb_size + gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="拼 customedFace 範例表格圖")
    p.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    p.add_argument("--out", type=Path, default=OUT_PATH)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--thumb-size", type=int, default=160)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = args.image_dir if args.image_dir.is_absolute() else ROOT / args.image_dir
    out = args.out if args.out.is_absolute() else ROOT / args.out
    path = make_table(
        image_dir=image_dir,
        out_path=out,
        n_examples=args.n,
        thumb_size=args.thumb_size,
    )
    print(f"✓ 已輸出: {path.resolve()}")


if __name__ == "__main__":
    main()
