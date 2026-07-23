#!/usr/bin/env python3
"""計算專案內常見模型的參數量。"""

from __future__ import annotations

import argparse
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Callable

import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# models 套件導入可能連帶執行 config.py 的 print，先壓掉雜訊
with redirect_stdout(StringIO()):
    from models.DenseNet import DenseNet
    from models.MergingViT import MergingViT
    from models.PVTv2 import PVTv2
    from models.ResNet import ResNet
    from models.Swin_tiny import Swin_tiny
    from models.VIT import VIT


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M ({n:,})"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K ({n:,})"
    return f"{n:,}"


# 預設以 Caltech-101 常見設定 num_classes=101 做公平比較；
# MergingViT 另附專案 config 常見的小尺寸設定。
MODEL_BUILDERS: dict[str, Callable[[], nn.Module]] = {
    "MergingViT (default)": lambda: MergingViT(
        img_size=224,
        patch_size=8,
        in_chans=3,
        num_classes=101,
        num_heads=[2, 4, 8, 16],
        embed_dims=[32, 64, 128, 256],
        depths=[1, 1, 1, 1],
        merge_size=2,
    ),
    "MergingViT (config-like)": lambda: MergingViT(
        img_size=28,
        patch_size=2,
        in_chans=3,
        num_classes=101,
        num_heads=[2, 4, 8, 16],
        embed_dims=[32, 64, 128, 256],
        depths=[1, 1, 1, 1],
        merge_size=[(2, 2), (2, 2), (2, 2)],
    ),
    "Swin_tiny": lambda: Swin_tiny(in_channels=3, out_channels=101),
    "PVTv2 (pvt_v2_b0)": lambda: PVTv2(
        in_channels=3,
        num_classes=101,
        model_name="pvt_v2_b0",
        pretrained=False,
    ),
    "VIT (vit_tiny_patch16_224)": lambda: VIT(
        in_channels=3,
        num_classes=101,
        model_name="vit_tiny_patch16_224",
        pretrained=False,
        img_size=224,
    ),
    "DenseNet (densenet121)": lambda: DenseNet(in_channels=3, out_channels=101),
    "ResNet (resnet18)": lambda: ResNet(layers=18, in_channels=3, out_channels=101),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="計算模型參數量")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="只計算指定模型名稱（部分字串匹配即可）。預設計算全部。",
    )
    return parser


def selected_builders(names: list[str] | None) -> dict[str, Callable[[], nn.Module]]:
    if not names:
        return MODEL_BUILDERS
    selected: dict[str, Callable[[], nn.Module]] = {}
    for key, builder in MODEL_BUILDERS.items():
        if any(name.lower() in key.lower() for name in names):
            selected[key] = builder
    if not selected:
        available = ", ".join(MODEL_BUILDERS)
        raise SystemExit(f"找不到符合的模型。可用：{available}")
    return selected


def main() -> None:
    args = build_parser().parse_args()
    builders = selected_builders(args.models)

    rows: list[dict[str, Any]] = []
    print(f"{'Model':<32} {'Total':>22} {'Trainable':>22}")
    print("-" * 80)

    for name, builder in builders.items():
        try:
            model = builder()
            total, trainable = count_parameters(model)
            rows.append({"name": name, "total": total, "trainable": trainable, "error": None})
            print(f"{name:<32} {format_params(total):>22} {format_params(trainable):>22}")
            del model
        except Exception as exc:  # noqa: BLE001 - 列出各模型失敗原因即可
            rows.append({"name": name, "total": None, "trainable": None, "error": str(exc)})
            print(f"{name:<32} {'ERROR':>22} {str(exc)[:40]}")

    ok_rows = [r for r in rows if r["total"] is not None]
    if ok_rows:
        print("-" * 80)
        print(f"{'合計模型數':<32} {len(ok_rows)}")
        smallest = min(ok_rows, key=lambda r: r["total"])
        largest = max(ok_rows, key=lambda r: r["total"])
        print(f"最小: {smallest['name']} = {format_params(smallest['total'])}")
        print(f"最大: {largest['name']} = {format_params(largest['total'])}")


if __name__ == "__main__":
    main()
