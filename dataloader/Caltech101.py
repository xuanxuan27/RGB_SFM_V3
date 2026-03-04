"""
Custom wrapper for Caltech-101 to fit the project's Dataset interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable, Tuple, Any

import torch
from torch.utils.data import Dataset
from torchvision.datasets import Caltech101


class Caltech101Dataset(Dataset):
    """
    Wraps torchvision.datasets.Caltech101 and exposes a train/test style split.

    The official Caltech-101 dataset only ships a single split, therefore we
    deterministically partition it by slicing the ordered sample list based on
    `train_ratio`. This keeps behaviour reproducible without relying on any
    extra metadata files.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        train_ratio: float = 0.8,
        download: bool = False,
        seed: int = 42,
    ) -> None:
        if not 0.0 < train_ratio < 1.0:
            raise ValueError("train_ratio 必須介於 0 與 1 之間")

        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.base_dataset = Caltech101(
            root=str(self.root),
            download=download,
        )

        total = len(self.base_dataset)
        # split_index = max(1, int(total * train_ratio))
        # split_index = min(split_index, total - 1)

        # if self.train:
        #     self.indices = range(0, split_index)
        # else:
        #     self.indices = range(split_index, total)

        # self.classes = self.base_dataset.categories

        # --- 修正開始 ---
        # 1. 產生所有索引
        all_indices = list(range(total))
        
        # 2. 使用固定種子打亂 (確保 train=True 和 train=False 亂掉的順序是一樣的)
        g = torch.Generator()
        g.manual_seed(seed)
        # 使用 torch.randperm 產生隨機順序
        shuffled_indices = torch.randperm(total, generator=g).tolist()
        
        split_index = max(1, int(total * train_ratio))
        split_index = min(split_index, total - 1)

        if self.train:
            # 取打亂後的前半段
            self.indices = shuffled_indices[:split_index]
        else:
            # 取打亂後的後半段
            self.indices = shuffled_indices[split_index:]
        # --- 修正結束 ---

        self.classes = self.base_dataset.categories

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        real_index = self.indices[index]
        img, target = self.base_dataset[real_index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        target = torch.tensor(target, dtype=torch.long)

        return img, target

