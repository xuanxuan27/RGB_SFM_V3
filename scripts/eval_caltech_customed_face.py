#!/usr/bin/env python3
"""
載入表 4.8 的 Caltech101 模型（各 exp 的 config + checkpoint），
對 data/customedFace 的圖片做分類，看是否辨識為人臉類別（Faces / Faces_easy）。

用法（專案根目錄）:
  conda run -n SFM python scripts/eval_caltech_customed_face.py
  conda run -n SFM python scripts/eval_caltech_customed_face.py --models DenseNet ViT
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.autonotebook import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import models  # noqa: E402

IMAGE_DIR = ROOT / "data" / "customedFace"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# 表 4.8：label -> exp
TABLE_4_8_MODELS = {
    "MergingViT": "exp219",
    "ResNet": "exp242",
    "DenseNet": "exp300",
    "ViT": "exp312",
    "Swin_tiny": "exp240",
    "PVTv2": "exp280",
}

FACE_CLASS_NAMES = ("Faces", "Faces_easy")
SUBTYPE_RE = re.compile(
    r"^(masked_face|glasses_face|sunglasses_face|partial_face|sketch_face|normal_face)",
    re.I,
)


def load_config_from_dir(exp_dir: Path):
    config_path = exp_dir / "config.py"
    if not config_path.is_file():
        raise FileNotFoundError(f"找不到 config.py: {config_path}")
    spec = importlib.util.spec_from_file_location(f"exp_cfg_{exp_dir.name}", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.config, mod.arch


def resolve_exp_dir(exp: str) -> Path:
    exp = exp.strip().rstrip("/")
    candidates = [Path(exp), ROOT / exp]
    if exp.isdigit():
        candidates.append(ROOT / "runs" / "train" / f"exp{exp}")
    elif exp.startswith("exp") and exp[3:].isdigit():
        candidates.append(ROOT / "runs" / "train" / exp)
    for c in candidates:
        if c.is_dir():
            return c.resolve()
    raise FileNotFoundError(f"找不到 experiment 目錄: {exp}")


def find_checkpoint(exp_dir: Path, arch_name: str) -> Path:
    for name in (f"{arch_name}_best.pth", "best_epoch.pth"):
        path = exp_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"找不到 checkpoint：{exp_dir}")


def get_caltech_class_names() -> list[str]:
    from dataloader.Caltech101 import Caltech101Dataset

    ds = Caltech101Dataset(str(ROOT / "data"), train=False, transform=None)
    return list(ds.classes)


def build_caltech_transform(input_size):
    h = input_size[0]
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(256),
        transforms.CenterCrop(h),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def infer_subtype(path: Path) -> str:
    m = SUBTYPE_RE.match(path.stem)
    return m.group(1).lower() if m else "unknown"


def collect_images(image_dir: Path) -> list[tuple[Path, str]]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"找不到圖片資料夾: {image_dir}")
    paths = sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise RuntimeError(f"資料夾沒有圖片: {image_dir}")
    return [(p.resolve(), infer_subtype(p)) for p in paths]


class PathDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, str]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, subtype = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, subtype, str(path)


def eval_one_model(
    model_label: str,
    exp_name: str,
    samples: list[tuple[Path, str]],
    class_names: list[str],
    face_indices: set[int],
    device: torch.device,
    batch_size: int,
) -> list[dict]:
    exp_dir = resolve_exp_dir(exp_name)
    config, arch = load_config_from_dir(exp_dir)
    ckpt_path = find_checkpoint(exp_dir, arch["name"])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_weights", ckpt)

    print(f"  config     : {exp_dir / 'config.py'}")
    print(f"  arch       : {arch['name']}")
    print(f"  checkpoint : {ckpt_path}")
    if isinstance(ckpt, dict) and "valid_acc" in ckpt:
        print(f"  val_acc    : {ckpt['valid_acc']:.4f}  best_epoch={ckpt.get('best_epoch', '?')}")

    model = getattr(getattr(models, arch["name"]), arch["name"])(**dict(arch["args"]))
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    input_size = tuple(config.get("input_shape", (224, 224)))
    loader = DataLoader(
        PathDataset(samples, build_caltech_transform(input_size)),
        batch_size=batch_size,
        shuffle=False,
    )

    rows = []
    with torch.no_grad():
        for X, subtypes, paths in tqdm(loader, desc=model_label, leave=False):
            X = X.to(device)
            logits = model(X)
            probs = F.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)
            top5_prob, top5_idx = probs.topk(min(5, probs.shape[1]), dim=1)

            for i in range(len(paths)):
                pred_i = int(preds[i].item())
                pred_name = class_names[pred_i] if pred_i < len(class_names) else str(pred_i)
                is_face = pred_i in face_indices
                top5 = [
                    (
                        f"{class_names[int(top5_idx[i, j])]}:{float(top5_prob[i, j]):.3f}"
                        if int(top5_idx[i, j]) < len(class_names)
                        else f"{int(top5_idx[i, j])}:{float(top5_prob[i, j]):.3f}"
                    )
                    for j in range(top5_idx.shape[1])
                ]
                rows.append({
                    "model": model_label,
                    "exp": exp_name,
                    "arch": arch["name"],
                    "subtype": subtypes[i],
                    "image": Path(paths[i]).name,
                    "path": paths[i],
                    "pred_idx": pred_i,
                    "pred_name": pred_name,
                    "confidence": float(confs[i].item()),
                    "is_face": int(is_face),
                    "top5": " | ".join(top5),
                })
                mark = "✓" if is_face else "✗"
                print(
                    f"  {mark} {subtypes[i]:<16} {Path(paths[i]).name:<28} "
                    f"pred={pred_name:<16} conf={float(confs[i]):.3f}"
                )
    return rows


def print_summary(rows: list[dict], model_order: list[str]) -> None:
    subtypes = []
    for r in rows:
        if r["subtype"] not in subtypes:
            subtypes.append(r["subtype"])

    print("\n" + "=" * 72)
    print("推廣性摘要：辨識為人臉（Faces / Faces_easy）比例")
    print("=" * 72)
    header = f"{'Model':<12} {'Exp':<8} {'Overall':>8}" + "".join(
        f" {s.replace('_face', ''):>12}" for s in subtypes
    )
    print(header)
    print("-" * len(header))

    for model in model_order:
        mrows = [r for r in rows if r["model"] == model]
        if not mrows:
            continue
        exp = mrows[0]["exp"]
        overall = sum(r["is_face"] for r in mrows) / len(mrows)
        cells = [f"{exp:<8}{overall:8.1%}"]
        for subtype in subtypes:
            srows = [r for r in mrows if r["subtype"] == subtype]
            rate = sum(r["is_face"] for r in srows) / len(srows) if srows else float("nan")
            cells.append(f"{rate:12.1%}" if srows else f"{'n/a':>12}")
        print(f"{model:<12}" + "".join(cells))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Caltech101 多模型 × customedFace 分類測試")
    p.add_argument(
        "--image-dir",
        type=Path,
        default=IMAGE_DIR,
        help="測試圖片資料夾（預設 data/customedFace）",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"要跑的模型（預設全部）。可選: {', '.join(TABLE_4_8_MODELS)}",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "plots" / "eval_caltech_customed_face.csv",
    )
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=ROOT / "plots" / "eval_caltech_customed_face_summary.csv",
    )
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = args.image_dir if args.image_dir.is_absolute() else ROOT / args.image_dir
    samples = collect_images(image_dir)

    class_names = get_caltech_class_names()
    face_indices = {class_names.index(n) for n in FACE_CLASS_NAMES if n in class_names}
    if not face_indices:
        raise RuntimeError(f"Caltech101 classes 找不到 {FACE_CLASS_NAMES}")

    model_items = TABLE_4_8_MODELS
    if args.models:
        unknown = [m for m in args.models if m not in TABLE_4_8_MODELS]
        if unknown:
            raise ValueError(f"未知模型: {unknown}；可選 {list(TABLE_4_8_MODELS)}")
        model_items = {m: TABLE_4_8_MODELS[m] for m in args.models}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("Caltech101 models × data/customedFace")
    print(f"  image_dir   : {image_dir.resolve()}  ({len(samples)} images)")
    print(f"  face classes: {[class_names[i] for i in sorted(face_indices)]}")
    print(f"  models      : {list(model_items.items())}")
    print(f"  device      : {device}")
    print("=" * 72)

    all_rows: list[dict] = []
    for label, exp_name in model_items.items():
        print(f"\n>>> {label} ({exp_name})")
        rows = eval_one_model(
            model_label=label,
            exp_name=exp_name,
            samples=samples,
            class_names=class_names,
            face_indices=face_indices,
            device=device,
            batch_size=args.batch_size,
        )
        hit = sum(r["is_face"] for r in rows)
        print(f"  face-hit: {hit}/{len(rows)} ({hit / len(rows):.1%})")
        all_rows.extend(rows)

    print_summary(all_rows, list(model_items.keys()))

    out_csv = args.out_csv if args.out_csv.is_absolute() else ROOT / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n✓ 逐張結果: {out_csv.resolve()}")

    subtypes = []
    for r in all_rows:
        if r["subtype"] not in subtypes:
            subtypes.append(r["subtype"])
    summary_rows = []
    for model, exp in model_items.items():
        mrows = [r for r in all_rows if r["model"] == model]
        row = {
            "model": model,
            "exp": exp,
            "n": len(mrows),
            "face_hit": sum(r["is_face"] for r in mrows),
            "face_rate": round(sum(r["is_face"] for r in mrows) / len(mrows), 4) if mrows else 0.0,
        }
        for subtype in subtypes:
            srows = [r for r in mrows if r["subtype"] == subtype]
            row[f"{subtype}_rate"] = (
                round(sum(r["is_face"] for r in srows) / len(srows), 4) if srows else ""
            )
        summary_rows.append(row)

    summary_csv = args.summary_csv if args.summary_csv.is_absolute() else ROOT / args.summary_csv
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"✓ 摘要結果: {summary_csv.resolve()}")


if __name__ == "__main__":
    main()
