import torch.nn as nn
import torch.nn.functional as F
import timm


class PVTv2(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=None,
        in_chans=None,
        num_classes=None,
        model_name="pvt_v2_b0",
        pretrained=False,
        drop_rate=0.2,
        drop_path_rate=0.2,
        auto_resize=True,
        min_input_size=32,
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
        self.auto_resize = auto_resize
        self.min_input_size = min_input_size

        # pvt_v2_b0 是 timm 內 PVTv2 系列最小版本；num_classes=0 讓 timm 只輸出 backbone feature。
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=self.in_channels,
            num_classes=0,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            **kwargs,
        )
        self.head = nn.Linear(self.model.num_features, self.out_channels)

    def forward(self, x):
        if self.auto_resize:
            height, width = x.shape[-2:]
            target_height = max(height, self.min_input_size)
            target_width = max(width, self.min_input_size)
            if (target_height, target_width) != (height, width):
                x = F.interpolate(x, size=(target_height, target_width), mode="bilinear", align_corners=False)

        x = self.model(x)
        return self.head(x)
