"""
backup file
做完針對餘弦相似度的 kmeans 修改
並未針對 head 作個別分析
Kmeans 的對象是做完 attention，但還沒經過 projection 的向量

K-means 分析工具：
- 對每個位置 (position) 的向量獨立做 K-means
- 以 Colored MNIST 為主
- 取得各群聚中心最近的 k 張圖的該位置 patch 作為向量代表圖
- 可判斷輸入圖在該位置的特徵最接近哪個群聚中心

針對 Head 的分析：
- 切割出每個 head 的向量
- 跟前面 Kmeans 的結果做比較，判斷輸入圖在該 head 的特徵最接近哪個群聚中心
"""
import sys
from pathlib import Path
import json

# 確保從專案根目錄 import（無論用 python Kmeans_analysis.py 或 python mergingViT_plot_tool/Kmeans_analysis.py）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from models.MergingViT import MergingViT


def get_features_before_merge(model, x, stage_idx):
    """
    取得指定 stage 在 transformer block 後、merge 前的 pre-projection 特徵。
    
    Returns:
        feat: [B, N, C] 其中 N = H*W
        H, W: 該 stage 的空間解析度
    """
    x = model.patch_embed(x)
    H, W = model.patch_embed.grid_h, model.patch_embed.grid_w
    
    for i in range(len(model.stages)):
        last_pre_proj = None
        x = x + model.pos_embeds[i]
        for blk in model.stages[i]:
            x = blk(x)
            pre_proj = getattr(blk.attn, "pre_projection_features", None)
            if pre_proj is None:
                raise RuntimeError(
                    "找不到 `pre_projection_features`，請確認 Attention.forward 有儲存投影前特徵。"
                )
            last_pre_proj = pre_proj
        if i == stage_idx:
            return last_pre_proj.clone(), H, W
        if not isinstance(model.merges[i], nn.Identity):
            x, H, W = model.merges[i](x, H, W)
    return None


def get_features_at_block(model, x, stage_idx, block_idx):
    """
    取得指定 stage 的指定 block 後的 pre-projection 特徵。
    
    Args:
        stage_idx: stage 索引 (0~num_stages-1)
        block_idx: 該 stage 內的 block 索引 (0~depths[stage_idx]-1)
    
    Returns:
        feat: [B, N, C] 其中 N = H*W（attention concat 後、W_o 前）
        H, W: 該 stage 的空間解析度
    """
    x = model.patch_embed(x)
    H, W = model.patch_embed.grid_h, model.patch_embed.grid_w
    
    for i in range(len(model.stages)):
        x = x + model.pos_embeds[i]
        for b, blk in enumerate(model.stages[i]):
            x = blk(x)
            pre_proj = getattr(blk.attn, "pre_projection_features", None)
            if pre_proj is None:
                raise RuntimeError(
                    "找不到 `pre_projection_features`，請確認 Attention.forward 有儲存投影前特徵。"
                )
            if i == stage_idx and b == block_idx:
                return pre_proj.clone(), H, W
        if not isinstance(model.merges[i], nn.Identity):
            x, H, W = model.merges[i](x, H, W)
    return None


def get_all_features_at_checkpoints(model, x):
    """
    一次 forward，收集每個 (stage, block) 的 pre-projection 特徵。
    避免對每個 checkpoint 重複跑整個模型。
    
    Returns:
        dict[(stage_idx, block_idx)] -> (feat [B,N,C], H, W)
        其中 feat 為 attention concat 後、W_o 前的向量
    """
    results = {}
    x = model.patch_embed(x)
    H, W = model.patch_embed.grid_h, model.patch_embed.grid_w
    
    for i in range(len(model.stages)):
        x = x + model.pos_embeds[i]
        for b, blk in enumerate(model.stages[i]):
            x = blk(x)
            pre_proj = getattr(blk.attn, "pre_projection_features", None)
            if pre_proj is None:
                raise RuntimeError(
                    "找不到 `pre_projection_features`，請確認 Attention.forward 有儲存投影前特徵。"
                )
            results[(i, b)] = (pre_proj.clone(), H, W)
        if not isinstance(model.merges[i], nn.Identity):
            x, H, W = model.merges[i](x, H, W)
    return results


def cluster_vit_patches(features, n_clusters=8, random_state=42, return_grid_shape=False):
    """
    對 ViT patch token 做「餘弦相似度等價」的 KMeans 分群。

    核心想法：
    1) 先對每個 patch 向量做 L2 normalization，讓每個向量長度 = 1。
    2) 在單位球面上，兩個單位向量 u, v 的平方歐氏距離：
         ||u - v||^2 = 2 - 2 * cos(theta)
       與 cos(theta) 單調對應，因此用 Euclidean KMeans
       在數學上等價於以 cosine similarity 做分群。

    Args:
        features: [Num_Patches, Dim] 的 Tensor/ndarray。
                  你的 MergingViT 無 [CLS]，因此此處只接受空間 patch token。
        n_clusters: 群聚數量。
        random_state: KMeans 隨機種子。
        return_grid_shape: 若為 True，額外回傳可 reshape 的 (H, W) 或 None。

    Returns:
        labels: [Num_Spatial_Patches]，只包含空間 patch 的 cluster label。
        label_grid_shape (optional): (H, W) 或 None。
    """
    if isinstance(features, torch.Tensor):
        feats_np = features.detach().cpu().numpy()
    else:
        feats_np = np.asarray(features)

    if feats_np.ndim != 2:
        raise ValueError(
            f"`features` 需為 2D [Num_Patches, Dim]，目前收到 shape={feats_np.shape}"
        )

    n_tokens = feats_np.shape[0]
    side = int(np.sqrt(n_tokens))
    if side * side == n_tokens:
        H, W = side, side
    else:
        # 若無法從 token 數推回方形網格，仍可分群，但不強制推得 H, W。
        H, W = None, None

    # 重要：L2 normalize 後再做 Euclidean KMeans => cosine-equivalent clustering
    spatial_feats_norm = normalize(feats_np, norm='l2', axis=1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(spatial_feats_norm)

    # 預留給可視化使用（若可推得格狀尺寸）：
    label_grid_shape = (H, W) if (H is not None and W is not None) else None
    # labels_2d = labels.reshape(label_grid_shape)  # 後續可直接視覺化

    if return_grid_shape:
        return labels, label_grid_shape
    return labels


class ViTAnalyzer:
    def __init__(self, model, dataloader, img_size=28, device='cuda'):
        self.model = model.to(device).eval()
        self.dataloader = dataloader
        self.device = device
        self.img_size = img_size

    def extract_features_before_merge(self, stage_idx, max_samples=None):
        """
        收集所有圖片在指定 stage 的「transformer block 後、merge 前」的特徵與原始圖片。
        
        Returns:
            features: [Total_Samples, N, C] 其中 N = H*W
            images: [Total_Samples, 3, H_img, W_img]
            H, W: 該 stage 的空間解析度
        """
        return self._extract_features(
            stage_idx, block_idx=None, max_samples=max_samples
        )

    def extract_features_at_block(self, stage_idx, block_idx, max_samples=None):
        """
        收集所有圖片在指定 stage、指定 block 後的特徵與原始圖片。
        
        Returns:
            features: [Total_Samples, N, C] 其中 N = H*W
            images: [Total_Samples, 3, H_img, W_img]
            H, W: 該 stage 的空間解析度
        """
        return self._extract_features(
            stage_idx, block_idx=block_idx, max_samples=max_samples
        )

    def _extract_features(self, stage_idx, block_idx=None, max_samples=None):
        """內部實作：依 block_idx 決定用 get_features_before_merge 或 get_features_at_block"""
        all_feats = []
        all_imgs = []
        
        with torch.no_grad():
            for imgs, _ in self.dataloader:
                imgs = imgs.to(self.device)
                if block_idx is None:
                    feat, H, W = get_features_before_merge(self.model, imgs, stage_idx)
                else:
                    feat, H, W = get_features_at_block(
                        self.model, imgs, stage_idx, block_idx
                    )
                all_feats.append(feat.cpu())
                all_imgs.append(imgs.cpu())
                
                total = sum(f.shape[0] for f in all_feats)
                if max_samples is not None and total >= max_samples:
                    break
        
        features = torch.cat(all_feats, dim=0)
        if max_samples is not None:
            features = features[:max_samples]
        images = torch.cat(all_imgs, dim=0)
        if max_samples is not None:
            images = images[:max_samples]
        print(f"共有 {total} 張圖片")
        print(f"特徵形狀: {features.shape}")
        return features, images, H, W

    def extract_all_features_single_pass(self, max_samples=None):
        """
        一次 forward 收集所有 (stage, block) 的 pre-projection 特徵與圖片。
        只讀一次資料、只跑一次模型，避免重複計算。
        
        Returns:
            all_features: dict[(stage_idx, block_idx)] -> (features [N,C], H, W)
            images: [Total_Samples, 3, H_img, W_img]
        """
        all_feats_per_checkpoint = {}  # (s,b) -> (list of feats, H, W)
        all_imgs = []
        n_batches = len(self.dataloader)
        
        with torch.no_grad():
            for batch_idx, (imgs, _) in enumerate(self.dataloader):
                imgs = imgs.to(self.device)
                checkpoints = get_all_features_at_checkpoints(self.model, imgs)
                
                for (s, b), (feat, H, W) in checkpoints.items():
                    if (s, b) not in all_feats_per_checkpoint:
                        all_feats_per_checkpoint[(s, b)] = ([], H, W)
                    all_feats_per_checkpoint[(s, b)][0].append(feat.cpu())
                
                all_imgs.append(imgs.cpu())
                total = sum(f.shape[0] for f in all_imgs)
                if max_samples is not None and total >= max_samples:
                    break
                if n_batches > 0 and (batch_idx + 1) % max(1, n_batches // 10) == 0:
                    pct = 100 * (batch_idx + 1) / n_batches
                    print(f"\r  收集特徵: batch {batch_idx+1}/{n_batches} ({pct:.0f}%)", end="", flush=True)
        
        images = torch.cat(all_imgs, dim=0)
        if max_samples is not None:
            images = images[:max_samples]
        
        all_features = {}
        for (s, b), (feat_list, H, W) in all_feats_per_checkpoint.items():
            feats = torch.cat(feat_list, dim=0)
            if max_samples is not None:
                feats = feats[:max_samples]
            all_features[(s, b)] = (feats, H, W)
        
        total = images.shape[0]
        if n_batches > 0:
            print()  # 換行結束 \r 輸出
        print(f"單次 forward 共有 {total} 張圖片")
        print("  特徵空間: attention concat 後、W_o 前 (pre-projection)")
        for (s, b), (feats, H, W) in all_features.items():
            print(f"  stage{s}_block{b}: {feats.shape} (H={H}, W={W})")
        
        return all_features, images

    def get_patch_from_image(self, full_img, patch_idx, H, W, patch_h, patch_w):
        """
        根據 patch_idx 還原出該 patch 在原始圖片中的位置並切圖。
        
        Args:
            full_img: [C, H_img, W_img] 原始圖片
            patch_idx: 0~N-1 的位置索引 (row-major)
            H, W: 該 Stage 的解析度 (如 14x14)
            patch_h, patch_w: 該 Stage 一個 Token 代表原始圖片的尺寸
        """
        row = patch_idx // W
        col = patch_idx % W
        
        y1, y2 = row * patch_h, (row + 1) * patch_h
        x1, x2 = col * patch_w, (col + 1) * patch_w
        
        img_np = full_img.permute(1, 2, 0).numpy()
        patch = img_np[int(y1):int(y2), int(x1):int(x2), :]
        return np.clip(patch, 0, 1)

    def run_kmeans_per_position(self, features, n_clusters=10, random_state=42, positions=None, show_progress=True):
        """
        對每個位置 (position) 的向量獨立做 K-means。
        
        Args:
            features: [N_samples, N, C] 其中 N = H*W
            positions: 要分析的位置索引列表，None 則分析全部 N 個位置
            show_progress: 是否顯示進度百分比
        Returns:
            kmeans_dict: dict[pos] -> KMeans 物件
            cluster_centers: dict[pos] -> [n_clusters, C]
        """
        N = features.shape[1]
        if positions is None:
            positions = list(range(N))
        
        kmeans_dict = {}
        cluster_centers = {}
        n_pos = len(positions)
        
        for i, p in enumerate(positions):
            X = features[:, p, :].numpy()  # [N_samples, C]
            # 先將特徵投影到單位球面，Euclidean KMeans 即可等價近似 cosine 分群
            X_norm = normalize(X, norm='l2', axis=1)
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(X_norm)
            kmeans_dict[p] = kmeans
            cluster_centers[p] = kmeans.cluster_centers_
            if show_progress and n_pos > 0:
                pct = 100 * (i + 1) / n_pos
                step = max(1, n_pos // 20)  # 約每 5% 更新一次
                if (i + 1) % step == 0 or i == n_pos - 1:
                    print(f"\r     K-means 位置進度: {i+1}/{n_pos} ({pct:.0f}%)", end="", flush=True)
        if show_progress and n_pos > 0:
            print(f"\r     K-means 位置進度: {n_pos}/{n_pos} (100%)   ")
        
        return kmeans_dict, cluster_centers

    def get_representative_images_per_position(
        self, features, images, kmeans_dict, stage_res, patch_size_in_origin,
        k_nearest=5, show_progress=True
    ):
        """
        對每個位置的每個群聚中心，取得最近的 k 張圖的該位置 patch 作為向量代表圖。
        
        Args:
            features: [N_samples, N, C]
            images: [N_samples, 3, H_img, W_img]
            kmeans_dict: dict[pos] -> KMeans
            stage_res: (H, W)
            patch_size_in_origin: (ph, pw)
            k_nearest: 每個群聚取幾張代表圖
            show_progress: 是否顯示進度百分比
        
        Returns:
            representatives: dict[(pos, cluster)] -> list of [patch_h, patch_w, 3] arrays
        """
        H, W = stage_res
        ph, pw = patch_size_in_origin
        n_clusters = next(iter(kmeans_dict.values())).n_clusters
        
        representatives = {}  # (pos, cluster) -> list of patches
        positions = list(kmeans_dict.keys())
        n_pos = len(positions)
        
        for i, p in enumerate(positions):
            X = features[:, p, :].numpy()
            X_norm = normalize(X, norm='l2', axis=1)
            kmeans = kmeans_dict[p]
            centers = kmeans.cluster_centers_
            
            for c in range(n_clusters):
                dists = np.linalg.norm(X_norm - centers[c], axis=1)
                nearest_indices = np.argsort(dists)[:k_nearest]
                
                patches = []
                for idx in nearest_indices:
                    patch = self.get_patch_from_image(
                        images[idx], p, H, W, ph, pw
                    )
                    patches.append(patch)
                representatives[(p, c)] = patches
            
            if show_progress and n_pos > 0:
                pct = 100 * (i + 1) / n_pos
                step = max(1, n_pos // 20)
                if (i + 1) % step == 0 or i == n_pos - 1:
                    print(f"\r     代表圖進度: {i+1}/{n_pos} ({pct:.0f}%)", end="", flush=True)
        if show_progress and n_pos > 0:
            print(f"\r     代表圖進度: {n_pos}/{n_pos} (100%)   ")
        
        return representatives

    def assign_input_to_clusters(self, features, kmeans_dict):
        """
        判斷輸入圖在每個位置的特徵最接近哪個群聚中心。
        
        Args:
            features: [N_samples, N, C] 或 [1, N, C] 單張圖
            kmeans_dict: dict[pos] -> KMeans
        Returns:
            labels: dict[pos] -> [N_samples] 每個樣本在該位置的群聚標籤
        """
        if features.dim() == 2:
            features = features.unsqueeze(0)
        labels = {}
        for p in kmeans_dict:
            X = features[:, p, :].numpy()
            X_norm = normalize(X, norm='l2', axis=1)
            labels[p] = kmeans_dict[p].predict(X_norm)
        return labels

    def assign_input_to_clusters_head_split(self, features, cluster_centers, num_heads):
        """
        使用原本 position-KMeans 的中心，切割成 head 子向量後做指派。

        Args:
            features: [N_samples, N, C] 或 [1, N, C]
            cluster_centers: dict[pos] -> [n_clusters, C]
            num_heads: head 數量
        Returns:
            labels_per_head: dict[(pos, head)] -> [N_samples]
        """
        if features.dim() == 2:
            features = features.unsqueeze(0)
        if features.dim() != 3:
            raise ValueError(f"`features` 應為 2D 或 3D，收到 shape={features.shape}")

        _, _, embed_dim = features.shape
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} 無法整除 num_heads={num_heads}，無法切割 head。"
            )
        head_dim = embed_dim // num_heads
        labels_per_head = {}

        for pos, centers in cluster_centers.items():
            X = features[:, pos, :].numpy()  # [N_samples, C]
            X_norm = normalize(X, norm='l2', axis=1)
            C_norm = normalize(np.asarray(centers), norm='l2', axis=1)  # [K, C]

            X_heads = X_norm.reshape(X_norm.shape[0], num_heads, head_dim)
            C_heads = C_norm.reshape(C_norm.shape[0], num_heads, head_dim)

            for h in range(num_heads):
                xh = X_heads[:, h, :]      # [N_samples, head_dim]
                ch = C_heads[:, h, :]      # [K, head_dim]
                dists = np.linalg.norm(xh[:, None, :] - ch[None, :, :], axis=2)  # [N_samples, K]
                labels_per_head[(pos, h)] = np.argmin(dists, axis=1)

        return labels_per_head

    def visualize_and_save_head_split_assignment(
        self, img, labels_single_head, H, W, num_heads, n_clusters, save_prefix, heads_per_fig=8
    ):
        """
        儲存 head-split 的 cluster 指派視覺化。
        每張圖包含原圖 + 多個 head 的 HxW cluster map，可分 part 儲存。
        """
        heads = list(range(num_heads))
        n_parts = (len(heads) + heads_per_fig - 1) // heads_per_fig
        img_np = np.clip(img.permute(1, 2, 0).numpy(), 0, 1)
        cmap = plt.cm.get_cmap('tab10', n_clusters) if n_clusters <= 10 else 'viridis'
        vmax = n_clusters - 1 if n_clusters > 1 else 1

        for part_idx in range(n_parts):
            hs = heads[part_idx * heads_per_fig:(part_idx + 1) * heads_per_fig]
            n_panels = 1 + len(hs)
            n_cols = 3
            n_rows = int(np.ceil(n_panels / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 3.4 * n_rows))
            axes = np.asarray(axes).reshape(-1)

            axes[0].imshow(img_np)
            axes[0].set_title("Original")
            axes[0].axis("off")

            for idx, h in enumerate(hs, start=1):
                grid = np.full((H, W), -1.0, dtype=np.float32)
                for p in range(H * W):
                    if (p, h) in labels_single_head:
                        grid[p // W, p % W] = float(labels_single_head[(p, h)])
                im = axes[idx].imshow(grid, cmap=cmap, vmin=0, vmax=vmax)
                axes[idx].set_title(f"Head {h}")
                axes[idx].axis("off")
                plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.02)

            for idx in range(n_panels, len(axes)):
                axes[idx].axis("off")

            plt.tight_layout()
            out_path = Path(f"{save_prefix}_part{part_idx}.png")
            plt.savefig(out_path, dpi=150)
            plt.close(fig)

    def save_head_split_labels_json(self, labels_single_head, out_path, H, W, num_heads):
        """
        儲存單張圖的 head-split 指派結果。
        """
        payload = {
            "H": int(H),
            "W": int(W),
            "num_heads": int(num_heads),
            "labels": [
                {"pos": int(p), "head": int(h), "cluster": int(c)}
                for (p, h), c in sorted(labels_single_head.items())
            ],
        }
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def sample_random_images(self, m, seed=None):
        """
        從 dataloader 隨機取 m 張圖片。
        
        Returns:
            images: [m, 3, H_img, W_img]
        """
        all_imgs = []
        for imgs, _ in self.dataloader:
            all_imgs.append(imgs.cpu())
            if sum(f.shape[0] for f in all_imgs) >= m:
                break
        images = torch.cat(all_imgs, dim=0)
        rng = np.random.default_rng(seed)
        indices = rng.choice(images.shape[0], min(m, images.shape[0]), replace=False)
        return images[indices]

    def extract_features_for_images(self, images, stage_idx, block_idx):
        """
        對給定的 images 提取指定 (stage, block) 的特徵。
        
        Args:
            images: [B, 3, H, W]
        Returns:
            features: [B, N, C], H, W
        """
        with torch.no_grad():
            imgs = images.to(self.device)
            feat, H, W = get_features_at_block(
                self.model, imgs, stage_idx, block_idx
            )
        return feat.cpu(), H, W

    def visualize_and_save_cluster_assignment(
        self, img, cluster_labels, H, W, n_clusters, save_path
    ):
        """
        將原圖與各位置的 cluster 指派視覺化並存檔。
        
        Args:
            img: [3, H_img, W_img]
            cluster_labels: dict[pos] -> int (單張圖) 或 [N] array
            H, W: stage 解析度
            n_clusters: 群聚數量（用於 colormap）
        """
        if isinstance(cluster_labels, dict):
            label_arr = np.array([cluster_labels[p] for p in sorted(cluster_labels)])
        else:
            label_arr = np.asarray(cluster_labels)
        label_grid = label_arr.reshape(H, W).astype(float)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        img_np = img.permute(1, 2, 0).numpy()
        axes[0].imshow(np.clip(img_np, 0, 1))
        axes[0].set_title('Original')
        axes[0].axis('off')

        cmap = plt.cm.get_cmap('tab10', n_clusters) if n_clusters <= 10 else 'viridis'
        vmax = n_clusters - 1 if n_clusters > 1 else 1
        im = axes[1].imshow(label_grid, cmap=cmap, vmin=0, vmax=vmax)
        axes[1].set_title(f'Cluster assignment (pos 0~{H*W-1})')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], label='Cluster')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def visualize_position_clusters(
        self, representatives, stage_res, n_clusters, k_nearest=5,
        positions_to_show=None, save_path=None, show=True,
        clusters_per_fig=10
    ):
        """
        視覺化指定位置的群聚代表圖。
        若 n_clusters 過大，會拆成每 clusters_per_fig 個 cluster 一張圖。
        
        Args:
            representatives: from get_representative_images_per_position
            stage_res: (H, W)
            positions_to_show: 要顯示的位置列表，None 則顯示中心位置
            show: 是否呼叫 plt.show()，存大量圖時可設 False
            clusters_per_fig: 每張圖顯示幾個 cluster，預設 10
        """
        H, W = stage_res
        if positions_to_show is None:
            center_idx = (H // 2) * W + (W // 2)
            positions_to_show = [center_idx]
        
        total_parts = sum((n_clusters + clusters_per_fig - 1) // clusters_per_fig for _ in positions_to_show)
        done = 0
        for pos in positions_to_show:
            n_parts = (n_clusters + clusters_per_fig - 1) // clusters_per_fig
            for part_idx in range(n_parts):
                c_start = part_idx * clusters_per_fig
                c_end = min(c_start + clusters_per_fig, n_clusters)
                n_rows = c_end - c_start
                
                fig, axes = plt.subplots(n_rows, k_nearest + 1, figsize=(12, n_rows * 1.5))
                if n_rows == 1:
                    axes = axes.reshape(1, -1)
                
                for row, c in enumerate(range(c_start, c_end)):
                    axes[row, 0].text(0.5, 0.5, f'Pos {pos}\nCluster {c}', ha='center', va='center')
                    axes[row, 0].axis('off')
                    for i in range(k_nearest):
                        if (pos, c) in representatives and i < len(representatives[(pos, c)]):
                            axes[row, i + 1].imshow(representatives[(pos, c)][i])
                        axes[row, i + 1].axis('off')
                
                plt.suptitle(f'Position {pos} (row={pos//W}, col={pos%W}) clusters {c_start}~{c_end-1}')
                plt.tight_layout()
                if save_path:
                    if n_parts > 1:
                        out_path = Path(save_path).parent / f"{Path(save_path).stem}_pos{pos}_part{part_idx}.png"
                    else:
                        out_path = Path(save_path).parent / f"{Path(save_path).stem}_pos{pos}.png"
                    plt.savefig(out_path, dpi=150)
                    done += 1
                    if total_parts > 0:
                        pct = 100 * done / total_parts
                        print(f"\r     儲存代表圖: {done}/{total_parts} ({pct:.0f}%)", end="", flush=True)
                if show:
                    plt.show()
                else:
                    plt.close(fig)
        if total_parts > 0 and save_path:
            print(f"\r     儲存代表圖: {total_parts}/{total_parts} (100%)   ")

    def full_analysis_pipeline(
        self, stage_idx=0, max_samples=None, n_clusters=10,
        k_nearest=5, positions_to_show=None, positions=None, save_dir=None
    ):
        """
        完整分析流程：提取特徵 -> 每位置 K-means -> 取得代表圖 -> 視覺化
        
        Args:
            positions: 要分析的位置索引，None 則分析全部 (當 N 很大如 784 時可傳子集加速)
        """
        print(f"Stage {stage_idx}: 提取 merge 前特徵 (max {max_samples if max_samples is not None else '全部'} 張)...")
        features, images, H, W = self.extract_features_before_merge(
            stage_idx, max_samples=max_samples
        )
        N = H * W
        print(f"  特徵形狀: {features.shape} (H={H}, W={W}, N={N})")
        
        patch_h = self.img_size // H
        patch_w = self.img_size // W
        print(f"  每 token 對應原圖: {patch_h}x{patch_w} 像素")
        
        print("  對每個位置做 K-means...")
        kmeans_dict, _ = self.run_kmeans_per_position(
            features, n_clusters=n_clusters, positions=positions
        )
        
        print("  取得各群聚的代表圖...")
        representatives = self.get_representative_images_per_position(
            features, images, kmeans_dict, (H, W), (patch_h, patch_w),
            k_nearest=k_nearest
        )
        
        if positions_to_show is None:
            positions_to_show = [(H // 2) * W + (W // 2)]
        
        if save_dir is None:
            save_dir = Path(__file__).resolve().parent / 'plots' / 'kmeans_analysis'
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print("  視覺化...")
        self.visualize_position_clusters(
            representatives, (H, W), n_clusters, k_nearest,
            positions_to_show=positions_to_show, save_path=save_dir,
            clusters_per_fig=10
        )
        
        return {
            'features': features,
            'images': images,
            'kmeans_dict': kmeans_dict,
            'representatives': representatives,
            'H': H, 'W': W,
        }

    def full_analysis_all_stages_blocks(
        self, max_samples=None, n_clusters=10, k_nearest=5,
        positions=None, save_dir=None,
        m_inference=None, inference_seed=42,
        clusters_per_fig=10, n_clusters_per_stage=None
    ):
        """
        對每個 stage、每個 block、每個 position 做 K-means 分析並存代表圖。
        使用單次 forward 收集所有 checkpoint 特徵，避免重複計算。
        若指定 m_inference，會隨機取 m 張圖推論 cluster 指派並存檔。
        
        儲存結構:
            save_dir/stage{s}_block{b}/repr_pos{pos}.png
            save_dir/inference/img_{i}_s{s}_b{b}.png  (若 m_inference 有值)
        
        Args:
            positions: 要分析的位置，None 則分析全部
            m_inference: 隨機取幾張圖做推論並存檔，None 則不做
            inference_seed: 隨機取圖的 seed
            clusters_per_fig: 每張圖顯示幾個 cluster，n_clusters 大時可拆圖
            n_clusters_per_stage: dict[stage_idx -> n_clusters]，若提供則依 stage 使用不同 cluster 數
        """
        if save_dir is None:
            save_dir = Path(__file__).resolve().parent / 'plots' / 'kmeans_all_stages_blocks'
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n=== 單次 forward 收集所有 (stage, block) 特徵 ===")
        all_features, images = self.extract_all_features_single_pass(
            max_samples=max_samples
        )
        
        checkpoints = sorted(all_features.keys())
        total_cp = len(checkpoints)
        all_results = {}
        for cp_idx, ((stage_idx, block_idx), (features, H, W)) in enumerate(
            [(k, all_features[k]) for k in checkpoints]
        ):
            nc = n_clusters_per_stage.get(stage_idx, n_clusters) if n_clusters_per_stage else n_clusters
            stage_dir = save_dir / f"stage{stage_idx}_block{block_idx}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            
            pct_total = 100 * (cp_idx + 1) / total_cp if total_cp > 0 else 0
            print(f"\n=== Stage {stage_idx} Block {block_idx} [總進度 {pct_total:.0f}% ({cp_idx+1}/{total_cp})] (K-means, n_clusters={nc}) ===")
            patch_h = self.img_size // H
            patch_w = self.img_size // W
            
            print(f"  對每個位置做 K-means...")
            kmeans_dict, cluster_centers = self.run_kmeans_per_position(
                features, n_clusters=nc, positions=positions
            )
            
            print(f"  取得各群聚的代表圖...")
            representatives = self.get_representative_images_per_position(
                features, images, kmeans_dict, (H, W), (patch_h, patch_w),
                k_nearest=k_nearest
            )
            
            positions_to_save = list(kmeans_dict.keys())
            print(f"  儲存 {len(positions_to_save)} 個位置的代表圖...")
            self.visualize_position_clusters(
                representatives, (H, W), nc, k_nearest,
                positions_to_show=positions_to_save,
                save_path=stage_dir / "repr",
                show=False,
                clusters_per_fig=clusters_per_fig
            )
            
            all_results[(stage_idx, block_idx)] = {
                'features': features,
                'images': images,
                'kmeans_dict': kmeans_dict,
                'cluster_centers': cluster_centers,
                'representatives': representatives,
                'H': H, 'W': W,
                'n_clusters': nc,
                'num_heads': int(self.model.stages[stage_idx][block_idx].attn.num_heads),
            }
        
        if m_inference is not None and m_inference > 0:
            self._infer_and_save_m_images(
                all_results, m_inference, n_clusters, k_nearest,
                save_dir, inference_seed, n_clusters_per_stage
            )
        
        return all_results

    def _infer_and_save_m_images(
        self, all_results, m, n_clusters, k_nearest, save_dir, seed=42,
        n_clusters_per_stage=None
    ):
        """
        隨機取 m 張圖，對每個 (stage, block) 推論 cluster 指派並存檔。
        使用單次 forward 取得所有 checkpoint 特徵，避免重複計算。
        儲存:
            inference_position/img{i}_original.png
            inference_position/img{i}_s{s}_b{b}.png
            inference_position/img{i}_s{s}_b{b}_repr_part{p}.png
            inference_head_split/img{i}_s{s}_b{b}_head_assign_part{p}.png
            inference_head_split/img{i}_s{s}_b{b}_head_labels.json
            inference_head_split/img{i}_s{s}_b{b}_head_repr_part{p}.png
        """
        inference_position_dir = save_dir / "inference_position"
        inference_head_split_dir = save_dir / "inference_head_split"
        inference_position_dir.mkdir(parents=True, exist_ok=True)
        inference_head_split_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n=== 隨機取 {m} 張圖推論 cluster 指派 ===")
        sample_imgs = self.sample_random_images(m, seed=seed)
        
        with torch.no_grad():
            imgs = sample_imgs.to(self.device)
            infer_checkpoints = get_all_features_at_checkpoints(self.model, imgs)
        
        n_imgs = sample_imgs.shape[0]
        for img_idx in range(n_imgs):
            pct_img = 100 * (img_idx + 1) / n_imgs if n_imgs > 0 else 0
            print(f"\n  推論進度: 圖 {img_idx+1}/{n_imgs} ({pct_img:.0f}%)")
            img = sample_imgs[img_idx]
            img_np = np.clip(img.permute(1, 2, 0).numpy(), 0, 1)
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(img_np)
            ax.axis('off')
            plt.savefig(inference_position_dir / f"img{img_idx}_original.png", bbox_inches='tight', pad_inches=0, dpi=150)
            plt.close(fig)
            for (stage_idx, block_idx), res in all_results.items():
                kmeans_dict = res['kmeans_dict']
                cluster_centers = res.get('cluster_centers')
                representatives = res['representatives']
                H, W = res['H'], res['W']
                num_heads = int(res.get('num_heads') or self.model.stages[stage_idx][block_idx].attn.num_heads)
                nc = res.get('n_clusters') or (
                    n_clusters_per_stage.get(stage_idx, n_clusters) if n_clusters_per_stage else n_clusters
                )
                
                feat = infer_checkpoints[(stage_idx, block_idx)][0][img_idx:img_idx + 1].cpu()
                labels = self.assign_input_to_clusters(feat, kmeans_dict)
                labels_single = {p: labels[p][0] for p in labels}
                
                save_path = inference_position_dir / f"img{img_idx}_s{stage_idx}_b{block_idx}.png"
                self.visualize_and_save_cluster_assignment(
                    sample_imgs[img_idx], labels_single, H, W, nc,
                    save_path
                )
                self._save_inference_repr(
                    sample_imgs[img_idx], labels_single, representatives,
                    k_nearest, H, W, inference_position_dir, img_idx, stage_idx, block_idx,
                    positions_per_fig=14
                )
                if cluster_centers is not None:
                    labels_head = self.assign_input_to_clusters_head_split(
                        feat, cluster_centers, num_heads
                    )
                    labels_single_head = {(p, h): int(arr[0]) for (p, h), arr in labels_head.items()}
                    head_save_prefix = inference_head_split_dir / f"img{img_idx}_s{stage_idx}_b{block_idx}_head_assign"
                    self.visualize_and_save_head_split_assignment(
                        sample_imgs[img_idx], labels_single_head, H, W, num_heads, nc,
                        save_prefix=head_save_prefix, heads_per_fig=8
                    )
                    self.save_head_split_labels_json(
                        labels_single_head,
                        inference_head_split_dir / f"img{img_idx}_s{stage_idx}_b{block_idx}_head_labels.json",
                        H, W, num_heads
                    )
                    self._save_inference_head_split_repr(
                        sample_imgs[img_idx], labels_single_head, representatives,
                        k_nearest, H, W, inference_head_split_dir, img_idx, stage_idx, block_idx,
                        positions_per_fig=16
                    )
            print(f"    已存 img{img_idx} 的原始圖、position 推論與 head-split 推論")

    def _save_inference_repr(
        self, img, labels_single, representatives, k_nearest,
        H, W, inference_dir, img_idx, stage_idx, block_idx,
        positions_per_fig=14
    ):
        """
        對每個 position 顯示其被分到的 cluster 的代表圖，每張圖最多 positions_per_fig 個 pos。
        格式：每行 = Pos X → Cluster Y | 原圖該 pos 的 patch | k 張 cluster 代表 patch。
        """
        patch_h = self.img_size // H
        patch_w = self.img_size // W
        n_cols = 1 + 1 + k_nearest  # 標籤 + 原圖 patch + 代表 patch
        positions = sorted(labels_single.keys())
        n_parts = (len(positions) + positions_per_fig - 1) // positions_per_fig
        
        for part_idx in range(n_parts):
            start = part_idx * positions_per_fig
            end = min(start + positions_per_fig, len(positions))
            pos_subset = positions[start:end]
            
            n_rows = len(pos_subset)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, n_rows * 1.2))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for row, pos in enumerate(pos_subset):
                cluster = labels_single[pos]
                axes[row, 0].text(
                    0.5, 0.5, f'Pos {pos}\n→ Cluster {cluster}',
                    ha='center', va='center', fontsize=10
                )
                axes[row, 0].axis('off')
                orig_patch = self.get_patch_from_image(img, pos, H, W, patch_h, patch_w)
                axes[row, 1].imshow(orig_patch)
                axes[row, 1].set_title('Input', fontsize=8)
                axes[row, 1].axis('off')
                for i in range(k_nearest):
                    if (pos, cluster) in representatives and i < len(representatives[(pos, cluster)]):
                        axes[row, 2 + i].imshow(representatives[(pos, cluster)][i])
                    axes[row, 2 + i].axis('off')
            
            plt.suptitle(f'img{img_idx} s{stage_idx}_b{block_idx} repr (pos {pos_subset[0]}~{pos_subset[-1]})')
            plt.tight_layout()
            save_path = inference_dir / f"img{img_idx}_s{stage_idx}_b{block_idx}_repr_part{part_idx}.png"
            plt.savefig(save_path, dpi=150)
            plt.close(fig)

    def _save_inference_head_split_repr(
        self, img, labels_single_head, representatives, k_nearest,
        H, W, inference_dir, img_idx, stage_idx, block_idx,
        positions_per_fig=16
    ):
        """
        對每個 (position, head) 顯示其被分到的 cluster 的代表圖。
        格式：每行 = Pos X / Head H → Cluster Y | 原圖該 pos patch | k 張 cluster 代表 patch。

        注意：
            這裡的 representative 來自原本 position-KMeans 的 (pos, cluster)。
            head-split 僅改變指派，不重新計算 head 專屬中心。
        """
        patch_h = self.img_size // H
        patch_w = self.img_size // W
        n_cols = 1 + 1 + k_nearest

        pairs = sorted(labels_single_head.keys(), key=lambda x: (x[0], x[1]))  # (pos, head)
        n_parts = (len(pairs) + positions_per_fig - 1) // positions_per_fig

        for part_idx in range(n_parts):
            start = part_idx * positions_per_fig
            end = min(start + positions_per_fig, len(pairs))
            pair_subset = pairs[start:end]

            n_rows = len(pair_subset)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, n_rows * 1.2))
            if n_rows == 1:
                axes = axes.reshape(1, -1)

            for row, (pos, head) in enumerate(pair_subset):
                cluster = labels_single_head[(pos, head)]
                axes[row, 0].text(
                    0.5, 0.5, f'Pos {pos}\nHead {head}\n→ Cluster {cluster}',
                    ha='center', va='center', fontsize=9
                )
                axes[row, 0].axis('off')

                orig_patch = self.get_patch_from_image(img, pos, H, W, patch_h, patch_w)
                axes[row, 1].imshow(orig_patch)
                axes[row, 1].set_title('Input', fontsize=8)
                axes[row, 1].axis('off')

                for i in range(k_nearest):
                    key = (pos, cluster)
                    if key in representatives and i < len(representatives[key]):
                        axes[row, 2 + i].imshow(representatives[key][i])
                    axes[row, 2 + i].axis('off')

            pos_start, head_start = pair_subset[0]
            pos_end, head_end = pair_subset[-1]
            plt.suptitle(
                f'img{img_idx} s{stage_idx}_b{block_idx} head-split repr '
                f'(({pos_start},h{head_start})~({pos_end},h{head_end}))'
            )
            plt.tight_layout()
            save_path = inference_dir / f"img{img_idx}_s{stage_idx}_b{block_idx}_head_repr_part{part_idx}.png"
            plt.savefig(save_path, dpi=150)
            plt.close(fig)


def _infer_arch_from_checkpoint(state):
    """從 checkpoint state_dict 推斷 patch_size、embed_dims、depths"""
    # patch_embed.proj.weight: [C_out, 3, patch_h, patch_w]
    pw = state.get('patch_embed.proj.weight')
    if pw is not None:
        _patch = int(pw.shape[-1])
    else:
        _patch = 2
    embed_dims = [64, 128, 256, 512]
    depths = [1, 1, 1, 1]
    return _patch, embed_dims, depths


def run_colored_mnist_analysis(
    model_path=None, dataset='Colored_MNIST', img_size=28, patch_size=1,
    max_samples=None, n_clusters=10, k_nearest=5, stage_idx=0, positions=None
):
    """
    以 Colored MNIST 執行的完整範例。
    若提供 model_path，會從 checkpoint 自動推斷 patch_size 以匹配架構。
    """
    import sys
    sys.path.insert(0, '.')
    from dataloader import get_dataloader
    
    train_loader, _ = get_dataloader(
        dataset=dataset, root='./data/', batch_size=32,
        input_size=(img_size, img_size)
    )
    
    _patch = patch_size
    _embed = [64, 128, 256, 512]
    _depths = [1, 1, 1, 1]
    state_to_load = None
    
    if model_path:
        path = Path(model_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent / model_path
        if path.exists():
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            state_to_load = ckpt.get('model_weights', ckpt)
            _patch, _embed, _depths = _infer_arch_from_checkpoint(state_to_load)
    
    model = MergingViT(
        img_size=img_size, patch_size=_patch, in_chans=3,
        num_classes=30, embed_dims=_embed, depths=_depths
    )
    if state_to_load is not None:
        model.load_state_dict(state_to_load, strict=True)
        print(f"已載入模型: {path}")
    elif model_path:
        print(f"警告: 找不到 checkpoint，使用隨機初始化")
    
    analyzer = ViTAnalyzer(model, train_loader, img_size=img_size)
    result = analyzer.full_analysis_pipeline(
        stage_idx=stage_idx, max_samples=max_samples,
        n_clusters=n_clusters, k_nearest=k_nearest, positions=positions
    )
    
    # 示範：對單張圖做 cluster 指派
    sample_feat = result['features'][:1]
    labels = analyzer.assign_input_to_clusters(sample_feat, result['kmeans_dict'])
    pos_keys = sorted(labels.keys())[:10]
    print(f"範例：第一張圖在各位置的群聚標籤 (前 10 個位置): {[(p, labels[p][0]) for p in pos_keys]}")
    
    return analyzer, result


def run_colored_mnist_analysis_all(
    model_path=None, dataset='Colored_MNIST', img_size=28, patch_size=1,
    max_samples=None, n_clusters=10, k_nearest=5, positions=None, save_dir=None,
    m_inference=None, inference_seed=42, clusters_per_fig=10,
    n_clusters_per_stage=None, model_args=None, data_root=None
):
    """
    對每個 stage、每個 block、每個 position 存代表圖。
    若指定 m_inference，隨機取 m 張圖推論 cluster 指派並存檔。
    儲存至 save_dir/stage{s}_block{b}/repr_pos{pos}.png
    推論圖: save_dir/inference/img{i}_s{s}_b{b}.png

    Args:
        model_args: 若提供（如從 config），用於建立模型；否則用 patch_size 等參數
    """
    import sys
    sys.path.insert(0, '.')
    from dataloader import get_dataloader
    
    _root = data_root if data_root is not None else str(Path(__file__).resolve().parent.parent / 'data')
    train_loader, _ = get_dataloader(
        dataset=dataset, root=_root, batch_size=32,
        input_size=(img_size, img_size)
    )
    
    if model_args is not None:
        # 從 config 的 model args 建立模型，確保與訓練時完全一致
        args = dict(model_args)
        # 若 config 未指定，補上必要參數
        args.setdefault('img_size', img_size)
        args.setdefault('in_chans', 3)
        # num_classes 通常會在 config 裡（如 Caltech101=101），若缺少再給預設值
        args.setdefault('num_classes', 30)
    else:
        # 沒有提供 model_args 時，才使用預設架構 + 從 checkpoint 推斷部分參數
        _patch = patch_size
        _embed = [64, 128, 256, 512]
        _depths = [1, 1, 1, 1]
        _merge = 2

    state_to_load = None
    if model_path:
        path = Path(model_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent / model_path
        if path.exists():
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            state_to_load = ckpt.get('model_weights', ckpt)
            # 只有在沒有 model_args 的情況下，才嘗試從 checkpoint 推斷架構
            if state_to_load and model_args is None:
                _patch, _embed, _depths = _infer_arch_from_checkpoint(state_to_load)
                # merge_size 無法從 checkpoint 推斷，保留預設值
    
    # 建立模型：
    # - 若有 model_args（例如從 runs/train/exp202/config.py），完全照 config 建立
    # - 否則使用預設 / 推斷的架構
    if model_args is not None:
        model = MergingViT(**args)
    else:
        model = MergingViT(
            img_size=img_size, patch_size=_patch, in_chans=3,
            num_classes=30, embed_dims=_embed, depths=_depths, merge_size=_merge
        )
    if state_to_load is not None:
        model.load_state_dict(state_to_load, strict=True)
        print(f"已載入模型: {path}")
    
    analyzer = ViTAnalyzer(model, train_loader, img_size=img_size)
    return analyzer.full_analysis_all_stages_blocks(
        max_samples=max_samples,
        n_clusters=n_clusters,
        k_nearest=k_nearest,
        positions=positions,
        save_dir=save_dir,
        m_inference=m_inference,
        inference_seed=inference_seed,
        clusters_per_fig=clusters_per_fig,
        n_clusters_per_stage=n_clusters_per_stage
    )


if __name__ == "__main__":
    import config as cfg

    # 從 config 讀取設定
    model_args = cfg.config["model"]["args"]
    ckpt_dir = cfg.config.get("kmeans_checkpoint_dir") or cfg.config["save_dir"]
    model_path = str(Path(cfg.config["root"]) / ckpt_dir / f"{cfg.config['model']['name']}_best.pth")
    dataset = cfg.config["dataset"]
    input_shape = cfg.config["input_shape"]
    clusters_list = cfg.config.get("kmeans_clusters_per_stage", [30, 60, 120, 240])
    n_clusters_per_stage = {i: v for i, v in enumerate(clusters_list)}

    # 選項 A：單一 stage 分析（只畫中心位置）
    # analyzer, result = run_colored_mnist_analysis(
    #     model_path=model_path,
    #     n_clusters=8, k_nearest=4, stage_idx=0
    # )

    # 選項 B：每個 stage、每個 block、每個 pos 都存代表圖，並隨機取 m 張圖推論
    # cluster 數量依 config 的 kmeans_clusters_per_stage 設定
    run_colored_mnist_analysis_all(
        model_path=model_path,
        dataset=dataset,
        img_size=input_shape[0],
        patch_size=model_args.get("patch_size", 2),
        max_samples=None,  # 可設 500 加速
        n_clusters_per_stage=n_clusters_per_stage,
        k_nearest=4,
        clusters_per_fig=10,  # 每 10 個 cluster 一張圖
        positions=None,  # None=全部位置；可傳 [0,1,105] 等子集加速
        save_dir=None,   # 預設 plots/kmeans_all_stages_blocks/
        m_inference=10,   # 隨機取 m 張圖推論 cluster 指派並存檔
        inference_seed=42,
        model_args=model_args,  # 從 config 傳入，含 merge_size 等
        data_root=str(Path(cfg.config["root"]) / "data"),
    )
