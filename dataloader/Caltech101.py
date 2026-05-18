"""
Custom wrapper for Caltech-101 to fit the project's Dataset interface.
"""

from __future__ import annotations

from collections import defaultdict
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

    Split 策略採用 per-class stratified split：每個類別各自按 train_ratio 切分，
    確保 train/val 的類別分佈一致，避免小類別在 val set 幾乎沒有樣本的問題。
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

        # --- Stratified split：每個類別各自切 train_ratio ---
        # 先把每個類別的 index 分組
        class_to_indices = defaultdict(list)
        for idx in range(len(self.base_dataset)):
            _, label = self.base_dataset[idx]
            class_to_indices[label].append(idx)

        # 固定種子，確保 train=True / train=False 結果一致
        g = torch.Generator()
        g.manual_seed(seed)

        train_indices, val_indices = [], []
        for label, idxs in sorted(class_to_indices.items()):
            # 對這個類別的 indices 做 shuffle
            perm = torch.randperm(len(idxs), generator=g).tolist()
            shuffled = [idxs[i] for i in perm]

            # 至少保留 1 筆在 train，1 筆在 val
            split = max(1, int(len(idxs) * train_ratio))
            split = min(split, len(idxs) - 1)

            train_indices.extend(shuffled[:split])
            val_indices.extend(shuffled[split:])

        self.indices = train_indices if self.train else val_indices
        # --- Stratified split 結束 ---

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