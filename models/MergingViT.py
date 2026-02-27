'''
來自 2026 葉容瑄的碩士論文:
在每一層 encoder 後合併 patch
觀察不同解析度下的特徵
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 積木層 ---
class FlexiblePatchMerging(nn.Module):
    def __init__(self, dim, m_h=2, m_w=2):
        super().__init__()
        self.m_h = m_h  # 合併高度
        self.m_w = m_w  # 合併寬度
        # 合併後維度會變成 dim * (m_h * m_w)，我們通常用 Linear 縮減到 2*dim
        self.reduction = nn.Linear(dim * m_h * m_w, 2 * dim)
        self.norm = nn.LayerNorm(dim * m_h * m_w)

    def forward(self, x, H, W):
        """
        x: [B, H*W, C]
        H, W: 當前的解析度 (例如 7, 7)
        """
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        # 【關鍵：自動補齊】
        # 如果 H 或 W 不能被 m_h 或 m_w 整除，自動補 Padding
        pad_input = (H % self.m_h != 0) or (W % self.m_w != 0)
        if pad_input:
            # 計算需要補多少 (右側與下側)
            pad_h = (self.m_h - H % self.m_h) % self.m_h
            pad_w = (self.m_w - W % self.m_w) % self.m_w
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h)) # (C_start, C_end, W_start, W_end, H_start, H_end)
            H, W = H + pad_h, W + pad_w

        # 這裡執行合併邏輯 (把鄰近的 Patch 拼接到 Channel 維度)
        # 假設 m_h=2, m_w=2
        # 將 x 切分並拼接成 [B, H/2, W/2, 4*C]
        x_merged = []
        for i in range(self.m_h):
            for j in range(self.m_w):
                x_merged.append(x[:, i::self.m_h, j::self.m_w, :])
        
        x = torch.cat(x_merged, dim=-1) # 拼接通道
        x = x.view(B, -1, C * self.m_h * self.m_w) # 攤平回 [B, new_N, new_C]
        
        x = self.norm(x)
        x = self.reduction(x)

        return x, H // self.m_h, W // self.m_w


# GEMINI 初版
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding """
    def __init__(self, img_size=28, patch_size=1, in_chans=3, embed_dim=64):
        super().__init__()
        # 支援 int 或 (h, w) tuple
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        
        self.img_size = img_size
        self.patch_h, self.patch_w = patch_size
        
        self.grid_h = img_size // self.patch_h
        self.grid_w = img_size // self.patch_w
        self.num_patches = self.grid_h * self.grid_w

        # kernel 與 stride 都要跟著 patch_size 走
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # 檢查輸入圖片大小是否符合預期
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}x{W}) doesn't match model ({self.img_size}x{self.img_size})."
        
        # 1. 通過卷積投影: [B, C, H, W] -> [B, embed_dim, H/P, W/P]
        x = self.proj(x)
        
        # 2. 攤平空間維度 (Flatten): [B, embed_dim, grid, grid] -> [B, embed_dim, N]
        # 3. 轉置維度 (Transpose) 以符合 Transformer 輸入: [B, N, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        
        x = self.norm(x)
        return x

# GEMINI 初版
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 定義 QKV 矩陣，這裡我們一次算出三個，之後再切分
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        
        # 定義融合用的投影層 (Projection)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        
        # 1. 產生 Q, K, V 並切分成多個 Head
        # qkv 形狀: [B, N, 3, num_heads, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 個別形狀為 [B, num_heads, N, head_dim]

        # 2. 計算 Attention Weights (權重)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 3. 得到每個 Head 的加權輸出 (尚未投影)
        # x_heads 形狀: [B, num_heads, N, head_dim]
        x_heads = (attn @ v)
        
        # 【重要提取點：路徑二】
        # 你之後可以從這裡取出 x_heads，reshape 回 [B, N, num_heads, head_dim] 跑 K-means
        self.individual_heads_output = x_heads.transpose(1, 2) 

        # 4. 拼接 (Concatenate)
        x = x_heads.transpose(1, 2).reshape(B, N, C)
        
        # 【重要提取點：路徑一（投影前）】
        self.pre_projection_features = x 

        # 5. 融合投影 (Linear Projection)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

# GEMINI 初版
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        # 如果沒有指定隱藏層維度，預設為輸入維度的 4 倍
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        
        # 第一層映射：低維 -> 高維
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 激活函數
        self.act = act_layer()
        # 第二層映射：高維 -> 低維
        self.fc2 = nn.Linear(hidden_features, out_features)
        # Dropout
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# --- 單元層 ---
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, drop)
        
        self.norm2 = nn.LayerNorm(dim)
        # mlp_ratio 通常設為 4，這就是隱藏層變胖 4 倍的地方
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        # Pre-norm 結構：先過 Norm 再做 Attention/MLP，最後加回殘差
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MergingViT(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=10, 
                 embed_dims=[64, 128, 256, 512], depths=[1, 1, 1, 1]):
        super().__init__()
        
        # 1. 初始 Patch Embedding (通常第一層切較大，如 4x4)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dims[0])
        h, w = self.patch_embed.grid_h, self.patch_embed.grid_w
        
        # 使用 ModuleList 來裝載多個 Stage
        self.stages = nn.ModuleList()
        self.pos_embeds = nn.ParameterList()
        self.merges = nn.ModuleList()
        
        num_stages = len(depths)
        
        for i in range(num_stages):
            # A. 建立當前 Stage 的位置編碼
            self.pos_embeds.append(nn.Parameter(torch.zeros(1, h * w, embed_dims[i])))
            
            # B. 建立當前 Stage 的 Transformer Blocks
            stage_blocks = nn.ModuleList([
                TransformerBlock(dim=embed_dims[i]) for _ in range(depths[i])
            ])
            self.stages.append(stage_blocks)
            
            # C. 建立 Patch Merging (除了最後一個 Stage 以外都要 Merge)
            if i < num_stages - 1:
                m = FlexiblePatchMerging(dim=embed_dims[i], m_h=2, m_w=2)
                self.merges.append(m)
                # 更新下一個 Stage 的解析度
                h, w = self.get_next_resolution(h, w, 2, 2)
            else:
                self.merges.append(nn.Identity()) # 最後一層不 Merge
        
        # 最後的分類頭
        self.norm_final = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)

    @staticmethod # 設定為靜態方法，不需要實例化也能用
    def get_next_resolution(h, w, m_h, m_w):
        """ 計算經過 FlexiblePatchMerging 之後的解析度 """
        new_h = (h + (m_h - h % m_h) % m_h) // m_h
        new_w = (w + (m_w - w % m_w) % m_w) // m_w
        return new_h, new_w

    def forward(self, x):
        x = self.patch_embed(x)
        H, W = self.patch_embed.grid_h, self.patch_embed.grid_w
        
        for i in range(len(self.stages)):
            # 1. 加上位置編碼
            x = x + self.pos_embeds[i]
            
            # 2. 通過該 Stage 的所有 Blocks
            for blk in self.stages[i]:
                x = blk(x)
            
            # 3. 執行 Merging 並更新解析度 (除了最後一階段)
            if not isinstance(self.merges[i], nn.Identity):
                x, H, W = self.merges[i](x, H, W)
                
        x = x.mean(dim=1)
        return self.head(self.norm_final(x))

# if __name__ == "__main__":
#     # 建立模型實例 (這裡可以自由調整 depths 和 embed_dims)
#     # 假設是 224x224 的大圖，patch_size 為 4
#     model = MergingViT(
#         img_size=224, 
#         patch_size=4, 
#         embed_dims=[64, 128, 256, 512], 
#         depths=[2, 2, 2, 2]
#     )
    
#     dummy_input = torch.randn(1, 3, 224, 224)

#     print(f"--- 開始追蹤解析度變化 (輸入: {dummy_input.shape}) ---")
    
#     # 1. Stage 0: Patch Embedding
#     x = model.patch_embed(dummy_input)
#     H, W = model.patch_embed.grid_h, model.patch_embed.grid_w
#     print(f"Stage 0 (Initial): {H}x{W}, Tokens: {x.shape[1]}, Channels: {x.shape[2]}")

#     # 2. 循環追蹤各個 Stage
#     for i in range(len(model.stages)):
#         # A. 加上位置編碼 (確認長度是否匹配)
#         x = x + model.pos_embeds[i]
        
#         # B. 通過該 Stage 的 Blocks
#         for blk in model.stages[i]:
#             x = blk(x)
#         print(f"Stage {i+1} (Encoder 結束): {H}x{W}, Tokens: {x.shape[1]}")

#         # C. 執行 Merging (除了最後一個 Stage)
#         if i < len(model.stages) - 1:
#             x, H, W = model.merges[i](x, H, W)
#             print(f"Merging {i+1} 之後: {H}x{W}, Tokens: {x.shape[1]}, Channels: {x.shape[2]}")
#         else:
#             print("--- 到達最後一個 Stage，準備進入 Header ---")

#     # 3. 最終輸出
#     x = x.mean(dim=1)
#     output = model.head(model.norm_final(x))
    
#     print(f"最終輸出形狀 (Should be [1, 30]): {output.shape}")