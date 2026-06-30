from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class DatasetSummary:
    request_name: str
    dataset_name: str
    class_count: int | str
    train_samples: int | str
    test_samples: int | str
    raw_image_size: str
    source: str
    status: str = "ok"


def candidate_paths(root: Path, relative_path: str) -> Iterable[Path]:
    yield root / relative_path
    yield root / "data" / relative_path


def first_existing_path(root: Path, relative_path: str) -> Path:
    for path in candidate_paths(root, relative_path):
        if path.exists():
            return path
    raise FileNotFoundError(
        f"找不到 {relative_path}；已檢查: "
        + ", ".join(str(path) for path in candidate_paths(root, relative_path))
    )


def format_array_image_size(shape: tuple[int, ...]) -> str:
    if len(shape) == 4:
        return f"{shape[1]}x{shape[2]}x{shape[3]}"
    if len(shape) == 3:
        return f"{shape[1]}x{shape[2]}"
    return str(shape[1:])


def summarize_colored_numpy(
    root: Path,
    request_name: str,
    dataset_name: str,
    folder: str,
) -> DatasetSummary:
    data_dir = first_existing_path(root, folder)
    train_imgs = np.load(data_dir / "Train_imgs.npy", mmap_mode="r")
    test_imgs = np.load(data_dir / "Test_imgs.npy", mmap_mode="r")
    train_labels = np.load(data_dir / "Train_labels.npy", mmap_mode="r")
    test_labels = np.load(data_dir / "Test_labels.npy", mmap_mode="r")

    labels = np.concatenate([np.asarray(train_labels), np.asarray(test_labels)])
    class_count = len(set(labels.tolist()))

    return DatasetSummary(
        request_name=request_name,
        dataset_name=dataset_name,
        class_count=class_count,
        train_samples=int(train_imgs.shape[0]),
        test_samples=int(test_imgs.shape[0]),
        raw_image_size=format_array_image_size(tuple(train_imgs.shape)),
        source=str(data_dir),
    )


def summarize_medmnist(
    root: Path,
    request_name: str,
    dataset_name: str,
    flag: str,
    size: int = 28,
) -> DatasetSummary:
    import medmnist
    from medmnist import INFO

    info = INFO[flag]
    class_count = len(info["label"])
    samples = info.get("n_samples")

    if isinstance(samples, dict) and {"train", "test"} <= samples.keys():
        train_samples = int(samples["train"])
        test_samples = int(samples["test"])
    else:
        DataClass = getattr(medmnist, info["python_class"])
        train_dataset = DataClass(split="train", size=size, download=True)
        test_dataset = DataClass(split="test", size=size, download=True)
        train_samples = len(train_dataset)
        test_samples = len(test_dataset)

    channels = int(info.get("n_channels", 1))
    raw_image_size = f"{size}x{size}x{channels}" if channels > 1 else f"{size}x{size}"

    return DatasetSummary(
        request_name=request_name,
        dataset_name=dataset_name,
        class_count=class_count,
        train_samples=train_samples,
        test_samples=test_samples,
        raw_image_size=raw_image_size,
        source=f"medmnist.INFO['{flag}']",
    )


def image_files_under(folder: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in suffixes)


def summarize_sizes(image_paths: list[Path], max_images: int | None = None) -> str:
    if not image_paths:
        return "unknown"

    selected_paths = image_paths if max_images is None else image_paths[:max_images]
    sizes: list[tuple[int, int]] = []
    modes: set[str] = set()

    for path in selected_paths:
        with Image.open(path) as img:
            width, height = img.size
            sizes.append((height, width))
            modes.add(img.mode)

    unique_sizes = set(sizes)
    mode_text = "/" + ",".join(sorted(modes)) if modes else ""
    if len(unique_sizes) == 1:
        height, width = sizes[0]
        return f"{height}x{width}{mode_text}"

    heights = [height for height, _ in sizes]
    widths = [width for _, width in sizes]
    suffix = "" if max_images is None else f", sampled={len(selected_paths)}"
    return (
        f"varies: H {min(heights)}-{max(heights)}, "
        f"W {min(widths)}-{max(widths)}, unique={len(unique_sizes)}{mode_text}{suffix}"
    )


def summarize_caltech101(
    root: Path,
    request_name: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    scan_all_sizes: bool = True,
) -> DatasetSummary:
    del seed  # Counts only depend on per-class sample totals.

    categories_dir = first_existing_path(root, "caltech101/101_ObjectCategories")
    class_dirs = [
        path
        for path in sorted(categories_dir.iterdir())
        if path.is_dir() and path.name != "BACKGROUND_Google"
    ]
    if not class_dirs:
        raise FileNotFoundError(f"{categories_dir} 底下找不到 Caltech101 類別資料夾")

    train_samples = 0
    test_samples = 0
    all_images: list[Path] = []

    for class_dir in class_dirs:
        images = image_files_under(class_dir)
        if not images:
            continue
        split = max(1, int(len(images) * train_ratio))
        split = min(split, len(images) - 1)
        train_samples += split
        test_samples += len(images) - split
        all_images.extend(images)

    raw_image_size = summarize_sizes(all_images, max_images=None if scan_all_sizes else 200)

    return DatasetSummary(
        request_name=request_name,
        dataset_name="Caltech101",
        class_count=len(class_dirs),
        train_samples=train_samples,
        test_samples=test_samples,
        raw_image_size=raw_image_size,
        source=str(categories_dir),
    )


def summarize_heart_calcification(
    root: Path,
    request_name: str,
    grid_size: int,
    need_resize_height: bool,
    resize_height: int,
    threshold: float,
    contrast_factor: float,
    enhance_method: str,
    use_vessel_mask: bool,
    use_min_count: bool,
    augment_positive: bool,
    augment_multiplier: int,
) -> DatasetSummary:
    with redirect_stdout(StringIO()):
        from dataloader.heart_calcification.heart_calcification_data_processor import (
            HeartCalcificationDataProcessor,
        )

    dataset_dir = first_existing_path(root, "HeartCalcification")

    def count_split(split: str) -> int:
        data_dir = dataset_dir / split
        if not data_dir.exists():
            raise FileNotFoundError(f"找不到 HeartCalcification/{split}: {data_dir}")
        with redirect_stdout(StringIO()):
            processor = HeartCalcificationDataProcessor(
                grid_size=grid_size,
                data_dir=str(data_dir),
                need_resize_height=need_resize_height,
                resize_height=resize_height,
                threshold=threshold,
                contrast_factor=contrast_factor,
                enhance_method=enhance_method,
                use_vessel_mask=use_vessel_mask,
            )
            data = processor.get_model_ready_data(
                use_min_count=use_min_count,
                augment_positive=augment_positive if split == "train" else False,
                augment_multiplier=augment_multiplier,
            )
        return len(data)

    train_samples = count_split("train")
    test_samples = count_split("test")
    raw_images = image_files_under(dataset_dir / "train") + image_files_under(dataset_dir / "test")
    raw_size = summarize_sizes(raw_images, max_images=None)

    return DatasetSummary(
        request_name=request_name,
        dataset_name="HeartCalcification",
        class_count=2,
        train_samples=train_samples,
        test_samples=test_samples,
        raw_image_size=f"patch {grid_size}x{grid_size}; raw {raw_size}",
        source=str(dataset_dir),
    )


def failed_summary(request_name: str, error: Exception) -> DatasetSummary:
    return DatasetSummary(
        request_name=request_name,
        dataset_name=request_name,
        class_count="-",
        train_samples="-",
        test_samples="-",
        raw_image_size="-",
        source=str(error),
        status="error",
    )


def build_registry(args: argparse.Namespace) -> dict[str, Callable[[str], DatasetSummary]]:
    root = args.root.resolve()

    return {
        "colored_mnist": lambda name: summarize_colored_numpy(
            root, name, "Colored_MNIST", "Color_MNIST"
        ),
        "color_mnist": lambda name: summarize_colored_numpy(
            root, name, "Colored_MNIST", "Color_MNIST"
        ),
        "fashion_mnist": lambda name: summarize_colored_numpy(
            root, name, "Colored_FashionMNIST", "Color_FashionMNIST"
        ),
        "colored_fashion_mnist": lambda name: summarize_colored_numpy(
            root, name, "Colored_FashionMNIST", "Color_FashionMNIST"
        ),
        "blood_mnist": lambda name: summarize_medmnist(
            root, name, "BloodMNIST", "bloodmnist", size=28
        ),
        "bloodmnist": lambda name: summarize_medmnist(
            root, name, "BloodMNIST", "bloodmnist", size=28
        ),
        "retina": lambda name: summarize_medmnist(
            root, name, "RetinaMNIST", "retinamnist", size=28
        ),
        "retina_mnist": lambda name: summarize_medmnist(
            root, name, "RetinaMNIST", "retinamnist", size=28
        ),
        "retinamnist": lambda name: summarize_medmnist(
            root, name, "RetinaMNIST", "retinamnist", size=28
        ),
        "retina_224": lambda name: summarize_medmnist(
            root, name, "RetinaMNIST_224", "retinamnist", size=224
        ),
        "retinamnist_224": lambda name: summarize_medmnist(
            root, name, "RetinaMNIST_224", "retinamnist", size=224
        ),
        "caltech101": lambda name: summarize_caltech101(
            root, name, train_ratio=args.caltech_train_ratio, scan_all_sizes=not args.fast
        ),
        "clatech101": lambda name: summarize_caltech101(
            root, name, train_ratio=args.caltech_train_ratio, scan_all_sizes=not args.fast
        ),
        "heart_calcification": lambda name: summarize_heart_calcification(
            root,
            name,
            grid_size=args.heart_grid_size,
            need_resize_height=args.heart_resize_height,
            resize_height=args.heart_resize_height_value,
            threshold=args.heart_threshold,
            contrast_factor=args.heart_contrast_factor,
            enhance_method=args.heart_enhance_method,
            use_vessel_mask=args.heart_use_vessel_mask,
            use_min_count=args.heart_use_min_count,
            augment_positive=args.heart_augment_positive,
            augment_multiplier=args.heart_augment_multiplier,
        ),
        "heart_clacification": lambda name: summarize_heart_calcification(
            root,
            name,
            grid_size=args.heart_grid_size,
            need_resize_height=args.heart_resize_height,
            resize_height=args.heart_resize_height_value,
            threshold=args.heart_threshold,
            contrast_factor=args.heart_contrast_factor,
            enhance_method=args.heart_enhance_method,
            use_vessel_mask=args.heart_use_vessel_mask,
            use_min_count=args.heart_use_min_count,
            augment_positive=args.heart_augment_positive,
            augment_multiplier=args.heart_augment_multiplier,
        ),
    }


def print_markdown(rows: list[DatasetSummary]) -> None:
    headers = [
        "dataset",
        "class_count",
        "train_samples",
        "test_samples",
        "raw_image_size",
        "status",
        "source",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        values = [
            row.dataset_name,
            row.class_count,
            row.train_samples,
            row.test_samples,
            row.raw_image_size,
            row.status,
            row.source,
        ]
        escaped = [str(value).replace("|", "\\|") for value in values]
        print("| " + " | ".join(escaped) + " |")


def print_csv(rows: list[DatasetSummary]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=list(asdict(rows[0]).keys()))
    writer.writeheader()
    writer.writerows(asdict(row) for row in rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="列出專案常用資料集的類別數、train/test 樣本數與原始影像尺寸。"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="資料根目錄；預設為專案根目錄，並會自動嘗試 root/data。",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "colored_mnist",
            "fashion_mnist",
            "blood_mnist",
            "clatech101",
            "heart_clacification",
            "retina",
        ],
        help="要統計的資料集名稱或別名。",
    )
    parser.add_argument("--format", choices=["markdown", "json", "csv"], default="markdown")
    parser.add_argument("--fast", action="store_true", help="大型影像資料集只抽樣估計尺寸範圍。")

    parser.add_argument("--caltech-train-ratio", type=float, default=0.8)

    parser.add_argument("--heart-grid-size", type=int, default=45)
    parser.add_argument("--heart-resize-height", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--heart-resize-height-value", type=int, default=900)
    parser.add_argument("--heart-threshold", type=float, default=0.5)
    parser.add_argument("--heart-contrast-factor", type=float, default=1.5)
    parser.add_argument("--heart-enhance-method", default="none")
    parser.add_argument("--heart-use-vessel-mask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--heart-use-min-count", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--heart-augment-positive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--heart-augment-multiplier", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = build_registry(args)
    rows: list[DatasetSummary] = []

    for dataset in args.datasets:
        key = dataset.lower()
        try:
            if key not in registry:
                raise KeyError(f"未知資料集別名: {dataset}")
            rows.append(registry[key](dataset))
        except Exception as error:  # Keep one bad dataset from hiding the rest.
            rows.append(failed_summary(dataset, error))

    if args.format == "json":
        print(json.dumps([asdict(row) for row in rows], ensure_ascii=False, indent=2))
    elif args.format == "csv":
        print_csv(rows)
    else:
        print_markdown(rows)


if __name__ == "__main__":
    main()
