#!/usr/bin/env python3
"""
對「特定圖片」用指定 expxx checkpoint 做分類推論 / 評估。

用法（專案根目錄）:
  # 資料夾依 ImageFolder 結構放標籤：face/*.png, non_face/*.png
  conda run -n SFM python eval_images.py --exp 330 --image-dir data/my_16_images

  # 扁平資料夾（只推論，無 GT）
  conda run -n SFM python eval_images.py --exp runs/train/exp330 --image-dir data/my_16_images

  # 檔案列表：每行 path，或 path\\tlabel / path label
  conda run -n SFM python eval_images.py --exp 330 --list my_16.txt
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.autonotebook import tqdm

import models
from dataloader.get_dataloader import dataset_classes

ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def load_config_from_dir(exp_dir: Path):
    config_path = exp_dir / "config.py"
    if not config_path.is_file():
        raise FileNotFoundError(f"找不到 config.py: {config_path}")
    spec = importlib.util.spec_from_file_location("exp_config", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.config, mod.arch


def resolve_exp_dir(exp: str) -> Path:
    exp = exp.strip().rstrip("/")
    candidates = []
    p = Path(exp)
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(ROOT / exp)
        if exp.isdigit():
            candidates.append(ROOT / "runs" / "train" / f"exp{exp}")
        elif exp.startswith("exp") and exp[3:].isdigit():
            candidates.append(ROOT / "runs" / "train" / exp)
        elif not exp.startswith("runs/"):
            candidates.append(ROOT / "runs" / "train" / exp)

    for c in candidates:
        if c.is_dir():
            return c.resolve()
    raise FileNotFoundError(
        f"找不到 experiment 目錄: {exp}\n嘗試過: " + ", ".join(str(c) for c in candidates)
    )


def find_checkpoint(exp_dir: Path, arch_name: str) -> Path:
    for name in (f"{arch_name}_best.pth", "best_epoch.pth"):
        path = exp_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"找不到 checkpoint：{exp_dir}")


def build_eval_transform(dataset_name: str, input_size):
    """對齊 get_dataloader 的 test transform。"""
    h, w = input_size[0], input_size[1]
    if dataset_name == "Caltech101":
        return transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize(256),
            transforms.CenterCrop(h),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize([h, w]),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
    ])


def get_class_names(dataset_name: str, num_classes: int | None, data_root: Path) -> list[str]:
    if dataset_name in dataset_classes:
        try:
            ds = dataset_classes[dataset_name](str(data_root), train=False, transform=None)
            names = list(getattr(ds, "classes", []))
            if names:
                return names
        except Exception:
            pass
    if num_classes is not None:
        return [str(i) for i in range(num_classes)]
    return []


def parse_label(token: str, class_names: list[str]) -> int:
    token = token.strip()
    if token.isdigit() or (token.startswith("-") and token[1:].isdigit()):
        return int(token)
    if token in class_names:
        return class_names.index(token)
    # 常見別名
    aliases = {
        "face": "face",
        "non_face": "non_face",
        "nonface": "non_face",
        "non-face": "non_face",
    }
    key = aliases.get(token.lower())
    if key is not None and key in class_names:
        return class_names.index(key)
    raise ValueError(f"無法解析標籤 '{token}'；可用類別名: {class_names}")


def collect_from_list(list_path: Path, class_names: list[str]) -> list[tuple[Path, int | None]]:
    samples = []
    for line_no, raw in enumerate(list_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            path_str, label_str = line.split("\t", 1)
        else:
            parts = line.split()
            if len(parts) == 1:
                path_str, label_str = parts[0], None
            else:
                path_str, label_str = parts[0], parts[-1]

        path = Path(path_str)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"{list_path}:{line_no} 找不到圖片: {path}")

        label = parse_label(label_str, class_names) if label_str is not None else None
        samples.append((path.resolve(), label))
    if not samples:
        raise RuntimeError(f"列表沒有有效圖片: {list_path}")
    return samples


def collect_from_dir(image_dir: Path, class_names: list[str]) -> list[tuple[Path, int | None]]:
    image_dir = image_dir.resolve()
    samples: list[tuple[Path, int | None]] = []

    # ImageFolder：子資料夾名 = 類別名（例如 face/、non_face/）
    if class_names:
        class_dirs = [
            d for d in sorted(image_dir.iterdir())
            if d.is_dir() and d.name in class_names
        ]
        if class_dirs:
            for class_dir in class_dirs:
                label = class_names.index(class_dir.name)
                for path in sorted(class_dir.rglob("*")):
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                        samples.append((path, label))
            if samples:
                return samples

    # 扁平資料夾：無標籤
    for path in sorted(image_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            samples.append((path, None))
    if not samples:
        raise RuntimeError(f"資料夾沒有可用圖片: {image_dir}")
    return samples


class ImagePathDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int | None]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if label is None:
            y = torch.tensor(-1, dtype=torch.long)
        else:
            y = torch.tensor(label, dtype=torch.long)
        return img, y, str(path)


def print_multiclass_metrics(targets, preds):
    targets = np.asarray(targets)
    preds = np.asarray(preds)
    print(f"\nAccuracy      : {accuracy_score(targets, preds):.4f}")
    print(f"Precision     : {precision_score(targets, preds, average='macro', zero_division=0):.4f} (macro)")
    print(f"Recall        : {recall_score(targets, preds, average='macro', zero_division=0):.4f} (macro)")
    print(f"F1-score      : {f1_score(targets, preds, average='macro', zero_division=0):.4f} (macro)")
    print(f"Balanced Acc. : {balanced_accuracy_score(targets, preds):.4f}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="特定圖片 × expxx checkpoint 分類推論")
    p.add_argument("--exp", required=True, help="331 / exp331 / runs/train/exp331")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image-dir", type=Path, help="圖片資料夾（可含子類別資料夾）")
    g.add_argument("--list", type=Path, help="檔案列表：path 或 path\\tlabel")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument(
        "--save-csv",
        type=Path,
        default=None,
        help="可選：把每張圖預測結果存成 CSV",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    exp_dir = resolve_exp_dir(args.exp)
    config, arch = load_config_from_dir(exp_dir)
    device = config.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if isinstance(device, str):
        device = torch.device(device)

    dataset_name = config["dataset"]
    input_size = tuple(config["input_shape"])
    batch_size = args.batch_size or int(config.get("batch_size", 16))
    num_classes = arch.get("args", {}).get("num_classes")
    data_root = ROOT / "data"
    class_names = get_class_names(dataset_name, num_classes, data_root)

    if args.list is not None:
        list_path = args.list if args.list.is_absolute() else ROOT / args.list
        samples = collect_from_list(list_path, class_names)
    else:
        image_dir = args.image_dir if args.image_dir.is_absolute() else ROOT / args.image_dir
        if not image_dir.is_dir():
            raise FileNotFoundError(f"找不到圖片資料夾: {image_dir}")
        samples = collect_from_dir(image_dir, class_names)

    ckpt_path = find_checkpoint(exp_dir, arch["name"])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_weights", ckpt)

    print("=" * 60)
    print("Classification inference on custom images")
    print(f"  exp        : {exp_dir}")
    print(f"  checkpoint : {ckpt_path}")
    print(f"  dataset    : {dataset_name}")
    print(f"  classes    : {class_names}")
    print(f"  n_images   : {len(samples)}")
    print(f"  has_labels : {sum(1 for _, y in samples if y is not None)}/{len(samples)}")
    if "best_epoch" in ckpt:
        print(f"  best_epoch : {ckpt['best_epoch']}")
        print(f"  val_acc    : {ckpt.get('valid_acc', '?')}")
    print("=" * 60)

    model = getattr(getattr(models, arch["name"]), arch["name"])(**dict(arch["args"]))
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    transform = build_eval_transform(dataset_name, input_size)
    loader = DataLoader(
        ImagePathDataset(samples, transform),
        batch_size=batch_size,
        shuffle=False,
    )

    rows = []
    all_targets, all_preds = [], []
    correct, labeled = 0, 0

    with torch.no_grad():
        for X, y, paths in tqdm(loader, total=len(loader)):
            X = X.to(device)
            logits = model(X)
            probs = F.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)

            for i in range(len(paths)):
                pred_i = int(preds[i].item())
                conf_i = float(confs[i].item())
                gt = int(y[i].item())
                gt_name = class_names[gt] if gt >= 0 and gt < len(class_names) else "-"
                pred_name = class_names[pred_i] if pred_i < len(class_names) else str(pred_i)
                ok = (gt >= 0 and pred_i == gt)
                if gt >= 0:
                    labeled += 1
                    all_targets.append(gt)
                    all_preds.append(pred_i)
                    if ok:
                        correct += 1
                mark = "✓" if ok else ("·" if gt < 0 else "✗")
                print(
                    f"{mark} {Path(paths[i]).name:40s}  "
                    f"pred={pred_name:12s} ({conf_i:.3f})  gt={gt_name}"
                )
                rows.append({
                    "path": paths[i],
                    "pred": pred_i,
                    "pred_name": pred_name,
                    "confidence": conf_i,
                    "gt": gt if gt >= 0 else "",
                    "gt_name": gt_name if gt >= 0 else "",
                    "correct": int(ok) if gt >= 0 else "",
                })

    if labeled:
        print(f"\nLabeled Accuracy : {correct / labeled:.4f}  ({correct}/{labeled})")
        print_multiclass_metrics(all_targets, all_preds)
    else:
        print("\n（無 GT 標籤，只輸出預測；可用類別子資料夾或 --list path\\tlabel 提供標籤）")

    if args.save_csv is not None:
        import csv
        out = args.save_csv if args.save_csv.is_absolute() else ROOT / args.save_csv
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"✓ 已存 CSV: {out.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"錯誤: {e}", file=sys.stderr)
        raise
