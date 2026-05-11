#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

try:
    from PIL import Image
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit(
        "找不到 Pillow。請先安裝：\n"
        "  python3 -m pip install --user Pillow\n"
    ) from e


DEFAULT_BASE_DIR = Path(
    "mergingViT_plot_tool/plots/patch_2x2/kmeans_all_stages_blocks_128_256_512_1024"
)
DEFAULT_OUT_DIR = Path("mergingViT_plot_tool/plots/split_pic")
DEFAULT_STAGES = ["stage3_block0", "stage2_block0", "stage1_block0", "stage0_block0"]


@dataclass(frozen=True)
class Box:
    x0: int
    y0: int
    x1: int
    y1: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _iter_images(stage_dir: Path) -> list[Path]:
    exts = (".png", ".jpg", ".jpeg", ".webp")
    paths = [p for p in stage_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(paths, key=lambda p: p.name)


def _parse_stage_limits(items: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"stage limit must be 'stage=NUM', got: {item}")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise ValueError(f"empty stage name in: {item}")
        try:
            n = int(v)
        except ValueError as e:
            raise ValueError(f"invalid NUM in: {item}") from e
        if n < 0:
            raise ValueError(f"NUM must be >= 0 in: {item}")
        out[k] = n
    return out


def _contiguous_ranges(flags: list[bool]) -> list[tuple[int, int]]:
    """
    flags: 1D bool list. Return list of (start, end), end exclusive.
    """
    if not flags:
        return []

    ranges: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, v in enumerate(flags):
        if v and not in_run:
            in_run = True
            start = i
        elif not v and in_run:
            in_run = False
            ranges.append((start, i))
    if in_run:
        ranges.append((start, len(flags)))
    return ranges


def _merge_ranges(ranges: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    if not ranges:
        return []
    merged = [ranges[0]]
    for s, e in ranges[1:]:
        last_s, last_e = merged[-1]
        if s - last_e <= max_gap:
            merged[-1] = (last_s, e)
        else:
            merged.append((s, e))
    return merged


def _moving_average(xs: list[float], window: int) -> list[float]:
    if not xs:
        return []
    if window <= 1:
        return xs[:]

    w = min(window, len(xs))
    out: list[float] = []
    s = sum(xs[:w])
    out.append(s / w)
    for i in range(w, len(xs)):
        s += xs[i] - xs[i - w]
        out.append(s / w)

    # pad to same length
    pad_left = (len(xs) - len(out)) // 2
    pad_right = len(xs) - len(out) - pad_left
    return [out[0]] * pad_left + out + [out[-1]] * pad_right


def _content_bbox_pil(gray: Image.Image, white_thr: int) -> Box:
    """
    Bounding box of all non-white pixels in the whole image.
    """
    w, h = gray.size
    pix = gray.load()

    x0, y0 = w, h
    x1, y1 = -1, -1

    for y in range(h):
        for x in range(w):
            if pix[x, y] < white_thr:
                if x < x0:
                    x0 = x
                if y < y0:
                    y0 = y
                if x > x1:
                    x1 = x
                if y > y1:
                    y1 = y

    if x1 < x0 or y1 < y0:
        return Box(0, 0, w, h)
    return Box(x0, y0, x1 + 1, y1 + 1)


def _tight_crop_pil(img: Image.Image, white_thr: int, pad: int = 0) -> Image.Image:
    gray = img.convert("L")
    w, h = gray.size
    pix = gray.load()

    x0, y0 = w, h
    x1, y1 = -1, -1

    for y in range(h):
        for x in range(w):
            if pix[x, y] < white_thr:
                if x < x0:
                    x0 = x
                if y < y0:
                    y0 = y
                if x > x1:
                    x1 = x
                if y > y1:
                    y1 = y

    if x1 < x0 or y1 < y0:
        return img

    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + 1 + pad)
    y1 = min(h, y1 + 1 + pad)
    return img.crop((x0, y0, x1, y1))


def _trim_vertical_by_projection(
    img: Image.Image,
    white_thr: int,
    frac_thr: float,
    pad: int,
) -> Image.Image:
    """
    Remove thin top/bottom bleed by using row-wise non-white fraction.
    This is useful when a row band accidentally includes a sliver of adjacent rows.
    """
    gray = img.convert("L")
    w, h = gray.size
    if w <= 0 or h <= 0:
        return img

    pix = gray.load()
    row_frac: list[float] = []
    for y in range(h):
        cnt = 0
        for x in range(w):
            if pix[x, y] < white_thr:
                cnt += 1
        row_frac.append(cnt / max(1, w))

    top = 0
    while top < h and row_frac[top] <= frac_thr:
        top += 1
    bottom = h - 1
    while bottom >= 0 and row_frac[bottom] <= frac_thr:
        bottom -= 1

    if bottom <= top:
        return img

    top = max(0, top - pad)
    bottom = min(h - 1, bottom + pad)
    return img.crop((0, top, w, bottom + 1))


def _row_nonwhite_fractions_pil(gray: Image.Image, bbox: Box, white_thr: int) -> list[float]:
    pix = gray.load()
    width = max(1, bbox.x1 - bbox.x0)
    fracs: list[float] = []

    for y in range(bbox.y0, bbox.y1):
        cnt = 0
        for x in range(bbox.x0, bbox.x1):
            if pix[x, y] < white_thr:
                cnt += 1
        fracs.append(cnt / width)

    return fracs


def _split_rows_by_projection_pil(
    gray: Image.Image,
    bbox: Box,
    num_rows: int,
    white_thr: int,
    row_nonwhite_frac_thr=0.02,
    smooth_window=5,
    min_row_height=25,
    merge_gap=8,
) -> list[tuple[int, int]]:
    """
    Use horizontal projection to find cluster rows.
    This handles the top title because the title becomes its own small band,
    and we later keep the largest `num_rows` row bands.
    """
    row_frac = _row_nonwhite_fractions_pil(gray, bbox=bbox, white_thr=white_thr)
    smooth = _moving_average(row_frac, window=smooth_window)

    active = [v > row_nonwhite_frac_thr for v in smooth]
    ranges = _contiguous_ranges(active)
    ranges = _merge_ranges(ranges, max_gap=merge_gap)

    # convert to absolute y and filter tiny bands
    abs_ranges = []
    for s, e in ranges:
        y0 = bbox.y0 + s
        y1 = bbox.y0 + e
        if (y1 - y0) >= min_row_height:
            abs_ranges.append((y0, y1))

    # Usually title is a short band; cluster rows are taller.
    # Keep the tallest `num_rows` bands, then sort by y.
    if len(abs_ranges) >= num_rows:
        abs_ranges = sorted(abs_ranges, key=lambda r: (r[1] - r[0]), reverse=True)[:num_rows]
        abs_ranges = sorted(abs_ranges, key=lambda r: r[0])
        return abs_ranges

    # Fallback: if not enough bands found, use coarse equal split inside bbox
    h = bbox.y1 - bbox.y0
    step = h / num_rows
    fallback = []
    for i in range(num_rows):
        y0 = int(round(bbox.y0 + i * step))
        y1 = int(round(bbox.y0 + (i + 1) * step))
        fallback.append((y0, y1))
    return fallback


def _col_nonwhite_fractions_pil(
    gray: Image.Image, y0: int, y1: int, x0: int, x1: int, white_thr: int
) -> list[float]:
    pix = gray.load()
    height = max(1, y1 - y0)
    fracs: list[float] = []

    for x in range(x0, x1):
        cnt = 0
        for y in range(y0, y1):
            if pix[x, y] < white_thr:
                cnt += 1
        fracs.append(cnt / height)

    return fracs


def _crop_cluster_strip_pil(
    img: Image.Image,
    gray: Image.Image,
    y0: int,
    y1: int,
    x_min: int,
    x_max: int,
    white_thr: int,
    col_nonwhite_frac_thr: float = 0.01,
    min_segment_width: int = 12,
    merge_gap: int = 35,
    tight_pad: int = 2,
    vtrim_frac_thr: float = 0.02,
    vtrim_pad: int = 1,
) -> Image.Image:
    """
    Keep the whole row, including the left label and the 4 thumbnails.

    Strategy:
    - detect all active x-ranges in this row
    - remove tiny noise
    - merge nearby segments
    - take the full envelope from the first meaningful segment
      to the last meaningful segment
    """
    col_frac = _col_nonwhite_fractions_pil(
        gray,
        y0=y0,
        y1=y1,
        x0=x_min,
        x1=x_max,
        white_thr=white_thr,
    )

    smooth = _moving_average(col_frac, window=7)
    active = [v > col_nonwhite_frac_thr for v in smooth]
    ranges = _contiguous_ranges(active)
    ranges = [(s, e) for (s, e) in ranges if (e - s) >= min_segment_width]
    ranges = _merge_ranges(ranges, max_gap=merge_gap)

    if not ranges:
        crop = img.crop((x_min, y0, x_max, y1))
        crop = _trim_vertical_by_projection(
            crop, white_thr=white_thr, frac_thr=vtrim_frac_thr, pad=vtrim_pad
        )
        return _tight_crop_pil(crop, white_thr=white_thr, pad=tight_pad)

    xs = ranges[0][0]
    xe = ranges[-1][1]

    crop = img.crop((x_min + xs, y0, x_min + xe, y1))
    crop = _trim_vertical_by_projection(
        crop, white_thr=white_thr, frac_thr=vtrim_frac_thr, pad=vtrim_pad
    )
    return _tight_crop_pil(crop, white_thr=white_thr, pad=tight_pad)


def split_montage_into_clusters(
    image_path: Path,
    num_clusters: int = 10,
    white_thr: int = 245,
    row_pad_up: int = 2,
    row_pad_down: int = 18,
    tight_pad: int = 2,
    vtrim_frac_thr: float = 0.02,
    vtrim_pad: int = 1,
) -> list[Image.Image]:
    """
    Split a montage image into per-cluster row images.
    Each output image contains one cluster row:
    - left label kept
    - 4 representative thumbnails kept
    - top title ignored during row selection
    """
    img = Image.open(image_path).convert("RGB")
    gray = img.convert("L")

    bbox = _content_bbox_pil(gray, white_thr=white_thr)

    row_ranges = _split_rows_by_projection_pil(
        gray=gray,
        bbox=bbox,
        num_rows=num_clusters,
        white_thr=white_thr,
        row_nonwhite_frac_thr=0.015,
        smooth_window=9,
        min_row_height=25,
        merge_gap=18,
    )

    crops: list[Image.Image] = []
    for (y0, y1) in row_ranges:
        # Expand row band a bit to avoid cutting the bottom edge.
        y0e = max(bbox.y0, y0 - row_pad_up)
        y1e = min(bbox.y1, y1 + row_pad_down)
        crop = _crop_cluster_strip_pil(
            img=img,
            gray=gray,
            y0=y0e,
            y1=y1e,
            x_min=bbox.x0,
            x_max=bbox.x1,
            white_thr=white_thr,
            col_nonwhite_frac_thr=0.01,
            min_segment_width=12,
            merge_gap=35,
            tight_pad=tight_pad,
            vtrim_frac_thr=vtrim_frac_thr,
            vtrim_pad=vtrim_pad,
        )
        crops.append(crop)

    return crops


def process_stage(
    base_dir: Path,
    out_dir: Path,
    stage: str,
    limit: int | None,
    num_clusters: int,
    white_thr: int,
    row_pad_up: int,
    row_pad_down: int,
    tight_pad: int,
    vtrim_frac_thr: float,
    vtrim_pad: int,
) -> tuple[int, int]:
    stage_dir = base_dir / stage
    if not stage_dir.exists():
        raise FileNotFoundError(f"stage directory not found: {stage_dir}")

    images = _iter_images(stage_dir)
    if limit is not None:
        images = images[:limit]

    stage_out = out_dir / stage
    _ensure_dir(stage_out)

    total_in = 0
    total_out = 0

    for img_path in images:
        total_in += 1
        crops = split_montage_into_clusters(
            img_path,
            num_clusters=num_clusters,
            white_thr=white_thr,
            row_pad_up=row_pad_up,
            row_pad_down=row_pad_down,
            tight_pad=tight_pad,
            vtrim_frac_thr=vtrim_frac_thr,
            vtrim_pad=vtrim_pad,
        )
        stem = img_path.stem
        for idx, crop in enumerate(crops):
            out_path = stage_out / f"{stem}_cluster{idx}.png"
            crop.save(out_path)
            total_out += 1

    return total_in, total_out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Split montage images into one image per cluster row."
    )
    p.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Base directory that contains stage folders.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory. Each stage will have its own subfolder here.",
    )
    p.add_argument(
        "--stages",
        nargs="+",
        default=DEFAULT_STAGES,
        help="Stage folders to process (relative to --base-dir).",
    )
    p.add_argument(
        "--limit-per-stage",
        type=int,
        default=None,
        help="Process at most N images per stage (default: all).",
    )
    p.add_argument(
        "--stage-limits",
        nargs="*",
        default=[],
        help="Override per-stage limits, format: stage3_block0=20 stage2_block0=5",
    )
    p.add_argument(
        "--num-clusters",
        type=int,
        default=10,
        help="Number of clusters per montage (default: 10).",
    )
    p.add_argument(
        "--white-thr",
        type=int,
        default=245,
        help="Pixel value threshold considered 'white background' (default: 245).",
    )
    # Backward-compatible knob (applies to both up/down) + the recommended split knobs.
    p.add_argument(
        "--row-pad",
        type=int,
        default=None,
        help="(Deprecated) Expand each detected cluster row band by N pixels on both top/bottom.",
    )
    p.add_argument(
        "--row-pad-up",
        type=int,
        default=2,
        help="Expand each detected cluster row band upwards by N pixels (default: 2).",
    )
    p.add_argument(
        "--row-pad-down",
        type=int,
        default=18,
        help="Expand each detected cluster row band downwards by N pixels (default: 18).",
    )
    p.add_argument(
        "--tight-pad",
        type=int,
        default=2,
        help="Padding (pixels) added after tight-crop (default: 2).",
    )
    p.add_argument(
        "--vtrim-frac-thr",
        type=float,
        default=0.02,
        help="Vertical trim threshold: row non-white fraction <= thr will be trimmed (default: 0.02).",
    )
    p.add_argument(
        "--vtrim-pad",
        type=int,
        default=1,
        help="Padding (pixels) kept after vertical trim (default: 1).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    base_dir: Path = args.base_dir
    out_dir: Path = args.out_dir
    stages: list[str] = args.stages
    limit_per_stage: int | None = args.limit_per_stage
    stage_limits = _parse_stage_limits(args.stage_limits)
    num_clusters: int = args.num_clusters
    white_thr: int = args.white_thr
    # If user provided the old --row-pad, apply it to both sides.
    if args.row_pad is not None:
        row_pad_up = int(args.row_pad)
        row_pad_down = int(args.row_pad)
    else:
        row_pad_up = int(args.row_pad_up)
        row_pad_down = int(args.row_pad_down)
    tight_pad: int = args.tight_pad
    vtrim_frac_thr: float = float(args.vtrim_frac_thr)
    vtrim_pad: int = int(args.vtrim_pad)

    _ensure_dir(out_dir)

    grand_in = 0
    grand_out = 0

    for stage in stages:
        lim = stage_limits.get(stage, limit_per_stage)
        total_in, total_out = process_stage(
            base_dir=base_dir,
            out_dir=out_dir,
            stage=stage,
            limit=lim,
            num_clusters=num_clusters,
            white_thr=white_thr,
            row_pad_up=row_pad_up,
            row_pad_down=row_pad_down,
            tight_pad=tight_pad,
            vtrim_frac_thr=vtrim_frac_thr,
            vtrim_pad=vtrim_pad,
        )
        print(f"[{stage}] processed {total_in} images -> {total_out} cluster crops")
        grand_in += total_in
        grand_out += total_out

    print(f"[ALL] processed {grand_in} images -> {grand_out} cluster crops")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())