import timm
import torch.nn as nn

class Swin_tiny(nn.Module):
    def __init__(self, in_channels=3, out_channels=101):
        super().__init__()
        # 先用 num_classes=0 載入，不建立 head
        self.model = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=False,
            num_classes=0,  # ← 先不建立分類頭
            drop_rate=0.2,
            drop_path_rate=0.2,
        )
        # 再手動加上新的分類頭
        self.head = nn.Linear(768, out_channels)

    def forward(self, x):
        x = self.model(x)  # 輸出 [B, 768]
        return self.head(x)