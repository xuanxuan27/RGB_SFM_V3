"""
FaceVsNonFace binary dataset loader for non-classic / varied face images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


class NonclassicFaceDataset(Dataset):
    """
    Binary classification: face (0) vs non_face (1).

    Expected layout under ``{root}/FaceVsNonFace``::

        FaceVsNonFace/
          train/
            face/<subclass>/*.png
            non_face/*.png
          test/
            face/<subclass>/*.png
            non_face/*.png
    """

    CLASS_TO_IDX = {"face": 0, "non_face": 1}
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.classes = ["face", "non_face"]

        split = "train" if train else "test"
        self.data_dir = self.root / "FaceVsNonFace" / split
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"找不到資料集目錄: {self.data_dir}")

        self.samples: List[Tuple[Path, int]] = self._collect_samples()
        if len(self.samples) == 0:
            raise RuntimeError(f"資料集為空: {self.data_dir}")

    def _collect_samples(self) -> List[Tuple[Path, int]]:
        samples: List[Tuple[Path, int]] = []
        for class_name, label in self.CLASS_TO_IDX.items():
            class_dir = self.data_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"找不到類別目錄: {class_dir}")

            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in self.IMAGE_EXTENSIONS:
                    samples.append((path, label))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        target = torch.tensor(target, dtype=torch.long)
        return img, target
