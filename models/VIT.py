import torch.nn as nn
import torch.nn.functional as F
import timm


class VIT(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=None,
        in_chans=None,
        num_classes=None,
        model_name="vit_tiny_patch16_224",
        pretrained=False,
        img_size=224,
        drop_rate=0.1,
        drop_path_rate=0.1,
        auto_resize=False,
        **kwargs,
    ):
        super().__init__()

        if in_chans is not None and in_chans != in_channels:
            raise ValueError("in_channels 與 in_chans 不一致，請只設定其中一個或使用相同值。")
        if out_channels is not None and num_classes is not None and num_classes != out_channels:
            raise ValueError("out_channels 與 num_classes 不一致，請只設定其中一個或使用相同值。")

        self.in_channels = in_channels if in_chans is None else in_chans
        self.out_channels = num_classes if num_classes is not None else out_channels
        if self.out_channels is None:
            self.out_channels = 101

        self.img_size = img_size
        self.auto_resize = auto_resize

        # num_classes=0 讓 timm 只輸出 ViT feature，再接專案自己的分類 head。
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=self.in_channels,
            num_classes=0,
            img_size=img_size,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            **kwargs,
        )
        self.head = nn.Linear(self.model.num_features, self.out_channels)

    def forward(self, x):
        if self.auto_resize:
            height, width = x.shape[-2:]
            if (height, width) != (self.img_size, self.img_size):
                x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

        x = self.model(x)
        return self.head(x)
