import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, Callable, Optional, Tuple
from PIL import Image

from config import config
from .heart_calcification.heart_calcification_data_processor import HeartCalcificationDataProcessor


class HeartCalcificationDataset(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        augmentation: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        color_mode: str = 'L',  # 新增參數
        grid_size: int = 45,
        need_resize_height: bool = False,
        resize_height : int = 900,
        threshold: float = 1.0,
        contrast_factor : float = 1.0,
        enhance_method: str = 'none',
        use_vessel_mask: bool = True,
        use_min_count:bool = True,
        augment_positive: bool = False,
        augment_multiplier: int = 1
    ) -> None:
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.grid_size = grid_size
        self.color_mode = color_mode

        self.data_dir = f"{self.root}/HeartCalcification/{'train' if self.train else 'test'}"
        self.data_processor = HeartCalcificationDataProcessor(
            grid_size=self.grid_size, data_dir=self.data_dir,
            need_resize_height = need_resize_height, resize_height= resize_height,
            threshold=threshold, contrast_factor = contrast_factor, enhance_method=enhance_method,
            use_vessel_mask=use_vessel_mask)

        need_augmentation =  augment_positive
        if not self.train:
            need_augmentation = False
            # 還原真實情況，不要做資料擴充
            # use_min_count = False


        print(f"is train :{train}")
        print(f"augment_positive :{augment_positive}")
        print(f"need_augmentation :{need_augmentation}")

        self.model_ready_data = self.data_processor.get_model_ready_data(use_min_count, need_augmentation, augment_multiplier)

        print("model data count : ",  len(self.model_ready_data))
        self.data_processor.display_label_counts()

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        key, img, label = self.model_ready_data[idx]

        img = torch.from_numpy(img).float() / 255.0  # 将 np.ndarray 转换为张量

        if self.color_mode == 'RGB':
            if img.ndim == 3 and img.shape[-1] == 3:
                img = img.permute(2, 0, 1)  # (H, W, 3) → (3, H, W)
            elif img.ndim == 2:
                img = img.unsqueeze(0).repeat(3, 1, 1)  # Grayscale to 3-channel
            else:
                raise ValueError(f"Unexpected image shape for RGB mode: {img.shape}")
        else:
            # Assume grayscale: squeeze any channel dimension and make shape (1, H, W)
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img.squeeze(-1)
            img = img.unsqueeze(0)  # (H, W) → (1, H, W)


        y_onehot = np.eye(2)[label]
        y_onehot = torch.from_numpy(y_onehot)
        label = y_onehot

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self) -> int:
        return len(self.model_ready_data)


class HeartCalcificationColor(HeartCalcificationDataset):
    def __init__(self, *args, **kwargs):
        grid_size = config["heart_calcification"]["grid_size"]
        resize_height = config["heart_calcification"]["resize_height"]
        need_resize_height = config["heart_calcification"]["need_resize_height"]
        threshold = config["heart_calcification"]["threshold"]
        contrast_factor = config["heart_calcification"]["contrast_factor"]
        enhance_method = config["heart_calcification"]["enhance_method"]
        use_vessel_mask = config["heart_calcification"]["use_vessel_mask"]
        use_min_count = config["heart_calcification"]["use_min_count"]
        augment_positive = config["heart_calcification"]["augment_positive"]
        augment_multiplier = config["heart_calcification"]["augment_multiplier"]
        super().__init__(*args, **kwargs, color_mode='RGB',
                         grid_size=grid_size, resize_height=resize_height,
                         need_resize_height=need_resize_height, threshold=threshold,
                         contrast_factor=contrast_factor, enhance_method=enhance_method,
                         use_vessel_mask=use_vessel_mask, use_min_count = use_min_count,
                         augment_positive = augment_positive, augment_multiplier=augment_multiplier)

class HeartCalcificationGray(HeartCalcificationDataset):
    def __init__(self, *args, **kwargs):
        grid_size = config["heart_calcification"]["grid_size"]
        resize_height = config["heart_calcification"]["resize_height"]
        need_resize_height = config["heart_calcification"]["need_resize_height"]
        threshold = config["heart_calcification"]["threshold"]
        contrast_factor = config["heart_calcification"]["contrast_factor"]
        enhance_method = config["heart_calcification"]["enhance_method"]
        use_vessel_mask = config["heart_calcification"]["use_vessel_mask"]
        use_min_count = config["heart_calcification"]["use_min_count"]
        augment_positive = config["heart_calcification"]["augment_positive"]
        augment_multiplier = config["heart_calcification"]["augment_multiplier"]
        super().__init__(*args, **kwargs, color_mode='L',
                         grid_size = grid_size, resize_height = resize_height,
                         need_resize_height = need_resize_height, threshold=threshold,
                         contrast_factor = contrast_factor, enhance_method=enhance_method,
                         use_vessel_mask=use_vessel_mask, use_min_count = use_min_count,
                         augment_positive=augment_positive, augment_multiplier=augment_multiplier)
