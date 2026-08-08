"""
K-means 分析工具：針對 MergingViT 的 transformer block 後、merge 前的向量
- 以 Colored MNIST 為主
- 對每個位置 (position) 的向量獨立做 K-means
- 取得各群聚中心最近的 k 張圖的該位置 patch 作為向量代表圖
- 可判斷輸入圖在該位置的特徵最接近哪個群聚中心

token 模式：
- 對每個位置 (position) 的向量獨立做 K-means
- 取得各群聚中心最近的 k 張圖的該位置 patch 作為向量代表圖
- 可判斷輸入圖在該位置的特徵最接近哪個群聚中心

head 模式（per-(position, head)）：
- 對每個 (position, head) 的向量獨立做 K-means
- 取得各群聚中心最近的 k 張圖的該 position patch 作為代表圖（距離用該 head 向量）
- 可判斷輸入圖在該 (position, head) 的特徵最接近哪個群聚中心

修改：
將 cluster 代表圖改為 padding 後的圖片
"""
import sys
from pathlib import Path
import json
import csv
import random

# 確保從專案根目錄 import（無論用 python Kmeans_analysis.py 或 python mergingViT_plot_tool/Kmeans_analysis.py）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from models.MergingViT import MergingViT


def get_features_before_merge(model, x, stage_idx):
    """
    取得指定 stage 在 transformer block 後、merge 前的特徵。
    
    Returns:
        feat: [B, N, C] 其中 N = H*W
        H, W: 該 stage 的空間解析度
    """
    x = model.patch_embed(x)
    H, W = model.patch_embed.grid_h, model.patch_embed.grid_w
    
    for i in range(len(model.stages)):
        x = x + model.pos_embeds[i]
        for blk in model.stages[i]:
            x = blk(x)
        if i == stage_idx:
            return x, H, W
        if not isinstance(model.merges[i], nn.Identity):
            x, H, W = model.merges[i](x, H, W)
    return None


def get_features_at_block(model, x, stage_idx, block_idx):
    """
    取得指定 stage 的指定 block 後的特徵。
    
    Args:
        stage_idx: stage 索引 (0~num_stages-1)
        block_idx: 該 stage 內的 block 索引 (0~depths[stage_idx]-1)
    
    Returns:
        feat: [B, N, C] 其中 N = H*W
        H, W: 該 stage 的空間解析度
    """
    x = model.patch_embed(x)
    H, W = model.patch_embed.grid_h, model.patch_embed.grid_w
    
    for i in range(len(model.stages)):
        x = x + model.pos_embeds[i]
        for b, blk in enumerate(model.stages[i]):
            x = blk(x)
            if i == stage_idx and b == block_idx:
                return x, H, W
        if not isinstance(model.merges[i], nn.Identity):
            x, H, W = model.merges[i](x, H, W)
    return None


def get_all_features_at_checkpoints(model, x):
    """
    一次 forward，收集每個 (stage, block) 後的特徵。
    避免對每個 checkpoint 重複跑整個模型。
    
    Returns:
        dict[(stage_idx, block_idx)] -> (feat [B,N,C], H, W)
    """
    results = {}
    x = model.patch_embed(x)
    H, W = model.patch_embed.grid_h, model.patch_embed.grid_w
    
    for i in range(len(model.stages)):
        x = x + model.pos_embeds[i]
        for b, blk in enumerate(model.stages[i]):
            x = blk(x)
            results[(i, b)] = (x.clone(), H, W)
        if not isinstance(model.merges[i], nn.Identity):
            x, H, W = model.merges[i](x, H, W)
    return results


def get_all_features_and_heads_at_checkpoints(model, x):
    """
    一次 forward，收集每個 (stage, block) 後的 token 特徵與 per-head 特徵。

    Returns:
        dict[(stage_idx, block_idx)] -> (
            feat_tokens [B, N, C],
            feat_heads [B, N, num_heads, head_dim],
            H,
            W,
        )
    """
    results = {}
    x = model.patch_embed(x)
    H, W = model.patch_embed.grid_h, model.patch_embed.grid_w

    for i in range(len(model.stages)):
        x = x + model.pos_embeds[i]
        for b, blk in enumerate(model.stages[i]):
            x = blk(x)
            head_feat = getattr(blk.attn, "individual_heads_output", None)
            if head_feat is None:
                raise RuntimeError(
                    "找不到 `individual_heads_output`，請確認 Attention.forward 有儲存每個 head 輸出。"
                )
            results[(i, b)] = (x.clone(), head_feat.clone(), H, W)
        if not isinstance(model.merges[i], nn.Identity):
            x, H, W = model.merges[i](x, H, W)
    return results


def _js_divergence(p, q, eps=1e-12):
    """Jensen-Shannon divergence，輸入需為機率分布。"""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def _cosine_similarity(a, b, eps=1e-12):
    """向量 cosine similarity。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = max(np.linalg.norm(a), eps)
    nb = max(np.linalg.norm(b), eps)
    return float(np.dot(a, b) / (na * nb))


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


_MERGING_VIT_ALLOWED_ARGS = {
    "img_size",
    "patch_size",
    "in_chans",
    "num_classes",
    "embed_dims",
    "num_heads",
    "depths",
    "merge_size",
    "drop_rate",
    "drop_path_rate",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _validate_mergingvit_model_args(model_args=None, model_name=None):
    """避免把其他模型（例如 Swin_tiny）的 config 傳進 MergingViT。"""
    if model_name is not None and model_name != "MergingViT":
        raise ValueError(
            f"KMeans/GradCAM trace 目前只支援 MergingViT，但 config model name 是 {model_name!r}。"
        )
    if model_args is None:
        return

    unexpected = sorted(set(model_args) - _MERGING_VIT_ALLOWED_ARGS)
    if unexpected:
        raise ValueError(
            "model_args 含有 MergingViT 不支援的參數 "
            f"{unexpected}；請確認 config['model']['name'] 是 'MergingViT'，"
            "或手動傳入 MergingViT 的 model_args。"
        )


def _normalize_label_mapping(label_mapping):
    """把不同來源的 label metadata 統一成 dict[int, str]。"""
    if not label_mapping:
        return {}

    if isinstance(label_mapping, dict):
        normalized = {}
        for key, value in label_mapping.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            normalized[idx] = str(value)
        return normalized

    if isinstance(label_mapping, (list, tuple)):
        return {idx: str(value) for idx, value in enumerate(label_mapping)}

    return {}


def _medmnist_label_names(dataset_name):
    medmnist_flags = {
        "PathMNIST": "pathmnist",
        "PathMNIST_224": "pathmnist",
        "DermaMNIST": "dermamnist",
        "RetinaMNIST": "retinamnist",
        "RetinaMNIST_224": "retinamnist",
        "PreprocessedRetinaMNIST224": "retinamnist",
        "BloodMNIST": "bloodmnist",
    }
    flag = medmnist_flags.get(dataset_name)
    if flag is None:
        return {}

    try:
        medmnist = __import__("medmnist", fromlist=["INFO"])
        info = medmnist.INFO
    except ImportError:
        return {}

    return _normalize_label_mapping(info[flag].get("label"))


def _fallback_label_names(dataset_name):
    fallback = {
        "HeartCalcification_Color": {0: "Normal", 1: "Calcification"},
        "HeartCalcification_Gray": {0: "Normal", 1: "Calcification"},
        "CIFAR10": {
            0: "airplane",
            1: "automobile",
            2: "bird",
            3: "cat",
            4: "deer",
            5: "dog",
            6: "frog",
            7: "horse",
            8: "ship",
            9: "truck",
        },
    }
    return dict(fallback.get(dataset_name, {}))


class ViTAnalyzer:
    def __init__(
        self, model, dataloader, img_size=28, device='cuda',
        display_mean=None, display_std=None, dataset_name=None,
        inference_dataloader=None,
    ):
        self.model = model.to(device).eval()
        self.dataloader = dataloader
        self.inference_dataloader = (
            inference_dataloader if inference_dataloader is not None else dataloader
        )
        self.device = device
        self.img_size = img_size
        self.display_mean = display_mean
        self.display_std = display_std
        self.dataset_name = dataset_name
        self.label_names = self._build_label_names()
        self.stage_resolutions = self._build_stage_resolutions()

    def _build_label_names(self):
        dataset = getattr(self.dataloader, "dataset", None)
        seen = set()
        candidates = []

        while dataset is not None and id(dataset) not in seen:
            seen.add(id(dataset))
            candidates.append(dataset)
            dataset = getattr(dataset, "dataset", None)

        for candidate in candidates:
            info = getattr(candidate, "info", None)
            if isinstance(info, dict):
                label_names = _normalize_label_mapping(info.get("label"))
                if label_names:
                    return label_names

            label_to_idx = getattr(candidate, "label_to_idx", None)
            if isinstance(label_to_idx, dict):
                return {int(idx): str(name) for name, idx in label_to_idx.items()}

            classes = getattr(candidate, "classes", None)
            label_names = _normalize_label_mapping(classes)
            if label_names:
                return label_names

        label_names = _medmnist_label_names(self.dataset_name)
        if label_names:
            return label_names

        return _fallback_label_names(self.dataset_name)

    def format_class_label(self, class_idx):
        class_idx = int(class_idx)
        name = self.label_names.get(class_idx)
        if name is None or name == str(class_idx):
            return str(class_idx)
        return f"{class_idx} ({name})"

    def _build_stage_resolutions(self):
        """
        建立每個 stage 的原始解析度與 merge 時的 padded 解析度。
        這只依賴模型架構，可重複用於每張 inference 圖。
        """
        H, W = self.model.patch_embed.grid_h, self.model.patch_embed.grid_w
        resolutions = []

        for stage_idx in range(len(self.model.stages)):
            entry = {
                "stage": int(stage_idx),
                "H": int(H),
                "W": int(W),
                "num_tokens": int(H * W),
                "has_merge": not isinstance(self.model.merges[stage_idx], nn.Identity),
            }

            if entry["has_merge"]:
                merge = self.model.merges[stage_idx]
                m_h, m_w = int(merge.m_h), int(merge.m_w)
                pad_h = (m_h - H % m_h) % m_h
                pad_w = (m_w - W % m_w) % m_w
                padded_H, padded_W = H + pad_h, W + pad_w
                next_H, next_W = padded_H // m_h, padded_W // m_w
                entry.update({
                    "m_h": m_h,
                    "m_w": m_w,
                    "pad_h": int(pad_h),
                    "pad_w": int(pad_w),
                    "padded_H": int(padded_H),
                    "padded_W": int(padded_W),
                    "next_H": int(next_H),
                    "next_W": int(next_W),
                })
                H, W = next_H, next_W
            else:
                entry.update({
                    "m_h": None,
                    "m_w": None,
                    "pad_h": 0,
                    "pad_w": 0,
                    "padded_H": int(H),
                    "padded_W": int(W),
                    "next_H": None,
                    "next_W": None,
                })

            resolutions.append(entry)

        return resolutions

    def forward_for_gradcam(self, img, target_class=None):
        """
        獨立 GradCAM forward：保留最後 stage 最後 block 後、GAP 前的 token 梯度。
        不複用 get_all_features_at_checkpoints，避免 clone/retain_grad 節點混淆。
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        x = img.to(self.device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[0] != 1:
            raise ValueError("forward_for_gradcam 目前一次只處理單張圖。")

        with torch.enable_grad():
            x = self.model.patch_embed(x)
            H, W = self.model.patch_embed.grid_h, self.model.patch_embed.grid_w
            activation = None
            last_stage_idx = len(self.model.stages) - 1
            last_block_idx = len(self.model.stages[last_stage_idx]) - 1

            for stage_idx in range(len(self.model.stages)):
                x = x + self.model.pos_embeds[stage_idx]
                for block_idx, blk in enumerate(self.model.stages[stage_idx]):
                    x = blk(x)
                    if stage_idx == last_stage_idx and block_idx == last_block_idx:
                        activation = x
                        activation.retain_grad()

                if not isinstance(self.model.merges[stage_idx], nn.Identity):
                    x, H, W = self.model.merges[stage_idx](x, H, W)

            if activation is None:
                raise RuntimeError("無法取得最後 stage 的 token activation。")

            pooled = activation.mean(dim=1)
            pooled = self.model.norm_final(pooled)
            pooled = self.model.head_drop(pooled)
            logits = self.model.head(pooled)
            if target_class is None:
                target_class = int(logits.argmax(dim=1).item())
            score = logits[0, int(target_class)]
            score.backward()

        grad = activation.grad
        if grad is None:
            raise RuntimeError("最後 stage activation 沒有 gradient；請確認 retain_grad 在 backward 前呼叫。")

        cam = (grad * activation).sum(dim=-1).clamp(min=0)
        cam = cam[0].detach().cpu()
        cam_min, cam_max = cam.min(), cam.max()
        if float(cam_max - cam_min) > 1e-12:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        self.model.zero_grad(set_to_none=True)
        return logits.detach().cpu(), int(target_class), cam, int(H), int(W)

    def compute_gradcam_topk(self, img, top_k=5, target_class=None):
        """回傳最後 stage CAM map 與 top-k token positions。"""
        logits, target_class, cam, H, W = self.forward_for_gradcam(
            img, target_class=target_class
        )
        n_tokens = int(cam.numel())
        k = min(int(top_k), n_tokens)
        order = torch.argsort(cam, descending=True)[:k].tolist()
        top_positions = []
        for rank, pos in enumerate(order, start=1):
            pos = int(pos)
            top_positions.append({
                "rank": int(rank),
                "pos": pos,
                "row": int(pos // W),
                "col": int(pos % W),
                "score": float(cam[pos].item()),
            })

        return {
            "target_class": int(target_class),
            "pred_class": int(logits.argmax(dim=1).item()),
            "logits": logits[0].tolist(),
            "H": int(H),
            "W": int(W),
            "cam": cam.view(H, W).numpy(),
            "top_positions": top_positions,
        }

    def _children_for_parent(self, stage_from, parent_pos):
        """取得 stage_from+1 的 parent token 在 stage_from 的來源 positions。"""
        info = self.stage_resolutions[stage_from]
        if not info["has_merge"]:
            return []

        parent_pos = int(parent_pos)
        parent_row = parent_pos // info["next_W"]
        parent_col = parent_pos % info["next_W"]
        children = []

        for dh in range(info["m_h"]):
            for dw in range(info["m_w"]):
                row = parent_row * info["m_h"] + dh
                col = parent_col * info["m_w"] + dw
                is_pad = row >= info["H"] or col >= info["W"]
                pos = None if is_pad else int(row * info["W"] + col)
                children.append({
                    "stage": int(stage_from),
                    "pos": pos,
                    "row": int(row),
                    "col": int(col),
                    "is_pad": bool(is_pad),
                })

        return children

    def _parent_for_stage_position(self, stage_idx, pos):
        """回傳指定 stage/position 在下一個 stage 的 parent；最後 stage 沒有 parent。"""
        stage_idx = int(stage_idx)
        pos = int(pos)
        if stage_idx >= len(self.stage_resolutions) - 1:
            return None

        info = self.stage_resolutions[stage_idx]
        if not info["has_merge"]:
            return None

        row = pos // info["W"]
        col = pos % info["W"]
        parent_row = row // info["m_h"]
        parent_col = col // info["m_w"]
        parent_pos = int(parent_row * info["next_W"] + parent_col)
        return int(stage_idx + 1), parent_pos

    def trace_last_stage_positions(self, last_stage_positions):
        """
        將最後 stage 的 positions 依 padded merge mapping 回溯到 stage 2~0。
        padding source 會保留，但不再繼續往更細 stage 展開。
        """
        last_stage_idx = len(self.stage_resolutions) - 1
        traces = {}

        for item in last_stage_positions:
            last_pos = int(item["pos"] if isinstance(item, dict) else item)
            parents = [{
                "stage": int(last_stage_idx),
                "pos": last_pos,
                "row": int(last_pos // self.stage_resolutions[last_stage_idx]["W"]),
                "col": int(last_pos % self.stage_resolutions[last_stage_idx]["W"]),
                "is_pad": False,
            }]
            per_stage = {}

            for stage_from in range(last_stage_idx - 1, -1, -1):
                children = []
                for parent in parents:
                    if parent.get("is_pad") or parent.get("pos") is None:
                        continue
                    children.extend(
                        self._children_for_parent(stage_from, parent["pos"])
                    )
                per_stage[int(stage_from)] = children
                parents = [child for child in children if not child["is_pad"]]

            traces[last_pos] = per_stage

        return traces

    def _resolve_trace_block_indices(self, stage_idx, trace_block="last"):
        if trace_block == "last":
            return [len(self.model.stages[stage_idx]) - 1]
        if trace_block == "all":
            return list(range(len(self.model.stages[stage_idx])))
        if isinstance(trace_block, int):
            block_idx = int(trace_block)
            if block_idx < 0 or block_idx >= len(self.model.stages[stage_idx]):
                raise ValueError(
                    f"stage {stage_idx} 沒有 block {block_idx}，"
                    f"有效範圍是 0~{len(self.model.stages[stage_idx]) - 1}"
                )
            return [block_idx]
        raise ValueError(f"trace_block 必須是 'last'、'all' 或 int，目前收到 {trace_block!r}")

    def _cluster_records_for_position(
        self, labels_by_checkpoint, stage_idx, pos, trace_block="last"
    ):
        records = []
        for block_idx in self._resolve_trace_block_indices(stage_idx, trace_block):
            labels_single = labels_by_checkpoint.get((stage_idx, block_idx), {})
            if pos in labels_single:
                cluster = int(labels_single[pos])
                missing = False
            else:
                cluster = None
                missing = True
            records.append({
                "block": int(block_idx),
                "cluster": cluster,
                "missing_kmeans": bool(missing),
            })
        return records

    def build_gradcam_trace_payload(
        self, img_idx, gradcam_result, traces, labels_by_checkpoint,
        trace_block="last"
    ):
        """組合單張圖的 GradCAM top-k、回溯來源與 cluster labels。"""
        last_stage_idx = len(self.stage_resolutions) - 1
        payload = {
            "img_idx": int(img_idx),
            "target_class": int(gradcam_result["target_class"]),
            "pred_class": int(gradcam_result["pred_class"]),
            "last_stage": int(last_stage_idx),
            "last_stage_H": int(gradcam_result["H"]),
            "last_stage_W": int(gradcam_result["W"]),
            "trace_block": trace_block,
            "cam": gradcam_result["cam"].tolist(),
            "top_positions": [],
        }

        for top_item in gradcam_result["top_positions"]:
            last_pos = int(top_item["pos"])
            item_payload = {
                **top_item,
                "clusters": self._cluster_records_for_position(
                    labels_by_checkpoint, last_stage_idx, last_pos, trace_block
                ),
                "sources_by_stage": {},
            }

            for stage_idx in range(last_stage_idx - 1, -1, -1):
                source_items = []
                for source in traces[last_pos].get(stage_idx, []):
                    pos = source["pos"]
                    source_payload = dict(source)
                    if source["is_pad"] or pos is None:
                        source_payload["clusters"] = []
                    else:
                        source_payload["clusters"] = self._cluster_records_for_position(
                            labels_by_checkpoint, stage_idx, int(pos), trace_block
                        )
                    source_items.append(source_payload)
                item_payload["sources_by_stage"][str(stage_idx)] = source_items

            payload["top_positions"].append(item_payload)

        return payload

    def _save_gradcam_trace_visualization(self, gradcam_result, traces, save_path):
        """儲存 CAM heatmap 與各 stage source mask，方便檢查 mapping。"""
        last_stage_idx = len(self.stage_resolutions) - 1
        n_cols = 1 + last_stage_idx
        fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 3.2))
        if n_cols == 1:
            axes = np.array([axes])

        cam = gradcam_result["cam"]
        axes[0].imshow(cam, cmap="hot")
        axes[0].set_title("Last-stage GradCAM", fontsize=9)
        for top_item in gradcam_result["top_positions"]:
            axes[0].text(
                top_item["col"], top_item["row"], str(top_item["rank"]),
                color="cyan", ha="center", va="center", fontsize=8
            )
        axes[0].axis("off")

        for ax_idx, stage_idx in enumerate(range(last_stage_idx - 1, -1, -1), start=1):
            info = self.stage_resolutions[stage_idx]
            mask = np.zeros((info["padded_H"], info["padded_W"]), dtype=np.float32)
            for top_item in gradcam_result["top_positions"]:
                last_pos = int(top_item["pos"])
                for source in traces[last_pos].get(stage_idx, []):
                    row, col = int(source["row"]), int(source["col"])
                    if source["is_pad"]:
                        mask[row, col] = -1.0
                    else:
                        mask[row, col] = max(mask[row, col], float(top_item["score"]))
            axes[ax_idx].imshow(mask, cmap="viridis")
            axes[ax_idx].set_title(f"Stage {stage_idx} sources", fontsize=9)
            axes[ax_idx].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    def _image_to_numpy(self, img):
        img_np = img.detach().cpu().permute(1, 2, 0).numpy()
        if self.display_mean is not None and self.display_std is not None:
            mean = np.asarray(self.display_mean, dtype=np.float32).reshape(1, 1, 3)
            std = np.asarray(self.display_std, dtype=np.float32).reshape(1, 1, 3)
            img_np = img_np * std + mean
        return np.clip(img_np, 0, 1)

    def _effective_n_clusters(self, requested_n_clusters, n_samples, context="K-means"):
        """KMeans 的 cluster 數不能大於樣本數；不足時自動降到可執行的上限。"""
        requested_n_clusters = int(requested_n_clusters)
        n_samples = int(n_samples)
        if n_samples < 1:
            raise ValueError(f"{context}: 至少需要 1 筆樣本才能做 K-means。")
        if requested_n_clusters < 1:
            raise ValueError(f"{context}: n_clusters 必須 >= 1，目前是 {requested_n_clusters}。")
        if requested_n_clusters > n_samples:
            print(
                f"  警告: {context} 的 n_clusters={requested_n_clusters} "
                f"大於樣本數 {n_samples}，自動改用 n_clusters={n_samples}。"
            )
            return n_samples
        return requested_n_clusters

    # ------------------------------------------------------------------
    # K-means cache 機制
    # ------------------------------------------------------------------

    @staticmethod
    def _heads_cache_tag(heads):
        """將 heads 列表轉成 cache 檔名用的穩定字串。"""
        if heads is None:
            return "all"
        return "-".join(str(int(h)) for h in sorted(heads))

    @staticmethod
    def _normalize_positions_meta(positions):
        """將 positions 正規化成可寫入 cache meta 的穩定值。"""
        if positions is None:
            return "all"
        return [int(p) for p in sorted(positions)]

    @staticmethod
    def _kmeans_cache_path(cache_dir, stage_idx, block_idx, nc, mode="token", heads=None):
        """回傳 kmeans_dict 的 .pkl 路徑與對應的 _meta.json 路徑。"""
        if mode == "token":
            base = Path(cache_dir) / f"stage{stage_idx}_block{block_idx}_nc{nc}"
        else:
            heads_tag = ViTAnalyzer._heads_cache_tag(heads)
            base = (
                Path(cache_dir)
                / f"stage{stage_idx}_block{block_idx}_nc{nc}_mode{mode}_heads{heads_tag}"
            )
        return base.with_suffix(".pkl"), Path(str(base) + "_meta.json")

    @staticmethod
    def _kmeans_cache_meta(
        model_path, save_dir, stage_idx, block_idx, nc, max_samples,
        mode="token", heads=None,
        dataset_name=None, analysis_split=None, positions=None, random_state=42,
    ):
        """建立 cache 有效性比對用的 meta dict。"""
        import os
        model_path = str(model_path) if model_path else ""
        try:
            mtime = float(os.path.getmtime(model_path)) if model_path else 0.0
        except OSError:
            mtime = 0.0
        heads_meta = None if heads is None else [int(h) for h in sorted(heads)]
        return {
            "model_path": model_path,
            "model_mtime": mtime,
            "save_dir": str(save_dir),
            "dataset_name": None if dataset_name is None else str(dataset_name),
            "analysis_split": None if analysis_split is None else str(analysis_split),
            "positions": ViTAnalyzer._normalize_positions_meta(positions),
            "random_state": int(random_state),
            "stage": int(stage_idx),
            "block": int(block_idx),
            "n_clusters": int(nc),
            "max_samples": max_samples,
            "mode": str(mode),
            "heads": heads_meta,
        }

    @staticmethod
    def _load_kmeans_cache(pkl_path, meta_path, expected_meta):
        """
        比對 meta 後載入 kmeans_dict。
        meta 不符合或檔案不存在時回傳 None。
        """
        import joblib
        if not pkl_path.exists() or not meta_path.exists():
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                saved_meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        if saved_meta != expected_meta:
            return None
        try:
            return joblib.load(pkl_path)
        except Exception:
            return None

    @staticmethod
    def _save_kmeans_cache(kmeans_dict, pkl_path, meta_path, meta):
        """將 kmeans_dict 存成 .pkl，並寫入 meta json。"""
        import joblib
        pkl_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(kmeans_dict, pkl_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
            
    #----------------------------------------------------
    def _source_positions_at_stage(self, stage_idx, pos, target_stage=0):
        """將指定 stage 的 token 展開到 target_stage 的真實 source positions。"""
        if stage_idx == target_stage:
            info = self.stage_resolutions[stage_idx]
            pos = int(pos)
            return [{
                "stage": int(stage_idx),
                "pos": pos,
                "row": int(pos // info["W"]),
                "col": int(pos % info["W"]),
                "is_pad": False,
            }]

        sources = []
        for child in self._children_for_parent(stage_idx - 1, pos):
            if child["is_pad"] or child["pos"] is None:
                continue
            sources.extend(
                self._source_positions_at_stage(
                    child["stage"], child["pos"], target_stage=target_stage
                )
            )
        return sources

    def _stage_patch_bounds(self, stage_idx, pos):
        """
        回傳 stage token 在原圖上的 pixel bbox。
        對 coarse stage 不用 img_size/stage_H 均分，而是用 stage0 source patch union。
        """
        stage0_sources = self._source_positions_at_stage(stage_idx, pos, target_stage=0)
        if not stage0_sources:
            return None

        patch_h = int(self.model.patch_embed.patch_h)
        patch_w = int(self.model.patch_embed.patch_w)
        rows = [source["row"] for source in stage0_sources]
        cols = [source["col"] for source in stage0_sources]
        y1 = max(0, min(rows) * patch_h)
        y2 = min(self.img_size, (max(rows) + 1) * patch_h)
        x1 = max(0, min(cols) * patch_w)
        x2 = min(self.img_size, (max(cols) + 1) * patch_w)
        return int(y1), int(y2), int(x1), int(x2)

    def _draw_stage_box(self, ax, stage_idx, pos, label, color="cyan", linewidth=2):
        bounds = self._stage_patch_bounds(stage_idx, pos)
        if bounds is None:
            return
        y1, y2, x1, x2 = bounds
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False,
            edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(rect)
        ax.text(
            x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2, label,
            color=color, fontsize=8, ha="center", va="center",
            bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=1)
        )

    def save_gradcam_topk_overview(self, img, gradcam_result, save_path):
        """在原圖標出 GradCAM top-k patch，並附最後 stage CAM heatmap。"""
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
        img_np = self._image_to_numpy(img)
        axes[0].imshow(img_np)
        axes[0].set_title("Original with top-k patches", fontsize=10)
        axes[0].axis("off")

        H, W = int(gradcam_result["H"]), int(gradcam_result["W"])
        last_stage_idx = len(self.stage_resolutions) - 1
        for top_item in gradcam_result["top_positions"]:
            self._draw_stage_box(
                axes[0], last_stage_idx, top_item["pos"],
                f"#{top_item['rank']}\npos{top_item['pos']}",
            )

        axes[1].imshow(gradcam_result["cam"], cmap="hot")
        axes[1].set_title("Last-stage GradCAM", fontsize=10)
        for top_item in gradcam_result["top_positions"]:
            axes[1].text(
                top_item["col"], top_item["row"], str(top_item["rank"]),
                color="cyan", ha="center", va="center", fontsize=9,
                bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=1)
            )
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=160)
        plt.close(fig)

    def save_gradcam_trace_summary(self, gradcam_result, traces, save_path):
        """正式的 top-k 合併報告圖：last-stage top-k 與各 stage source masks。"""
        self._save_gradcam_trace_visualization(gradcam_result, traces, save_path)

    def _cluster_and_representatives_for_pos(
        self, all_results, labels_by_checkpoint, stage_idx, pos,
        trace_block="last", k_nearest=4
    ):
        block_indices = self._resolve_trace_block_indices(stage_idx, trace_block)
        block_idx = block_indices[0]
        labels_single = labels_by_checkpoint.get((stage_idx, block_idx), {})
        cluster = labels_single.get(int(pos))
        reps = []
        if cluster is not None and (stage_idx, block_idx) in all_results:
            representatives = all_results[(stage_idx, block_idx)]["representatives"]
            reps = representatives.get((int(pos), int(cluster)), [])[:k_nearest]
        return block_idx, cluster, reps

    def _stage_patch(self, img, stage_idx, pos):
        return self._stage_patch_with_padding(img, stage_idx, pos)

    def _stage_patch_with_padding(self, img, stage_idx, pos):
        """
        裁出 token 在原圖的完整感受野，超出原圖邊界的部分補黑（對應 zero padding）。
        以 padded grid 為基準計算 patch 像素大小，使所有 token 的輸出尺寸一致。
        """
        patch_h = int(self.model.patch_embed.patch_h)
        patch_w = int(self.model.patch_embed.patch_w)

        # 計算此 stage 的 token 覆蓋幾個 stage-0 patch（含 padding 擴張）
        span_h, span_w = 1, 1
        for s in range(stage_idx):
            info = self.stage_resolutions[s]
            if info["has_merge"]:
                span_h *= info["m_h"]
                span_w *= info["m_w"]

        # token 在原圖像素空間的完整感受野大小
        ph = patch_h * span_h
        pw = patch_w * span_w

        # 在當前 stage grid 裡的 row/col
        info = self.stage_resolutions[stage_idx]
        W = info["W"]
        row = pos // W
        col = pos % W

        y1 = row * ph
        x1 = col * pw
        y2 = y1 + ph
        x2 = x1 + pw

        img_np = self._image_to_numpy(img)
        H_img, W_img = img_np.shape[:2]

        # 建立黑色畫布，超出邊界的部分保持為 0（呈現 padding）
        canvas = np.zeros((ph, pw, 3), dtype=np.float32)
        src_y1 = max(0, y1)
        src_y2 = min(H_img, y2)
        src_x1 = max(0, x1)
        src_x2 = min(W_img, x2)

        if src_y2 > src_y1 and src_x2 > src_x1:
            dst_y1 = src_y1 - y1
            dst_y2 = dst_y1 + (src_y2 - src_y1)
            dst_x1 = src_x1 - x1
            dst_x2 = dst_x1 + (src_x2 - src_x1)
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = img_np[src_y1:src_y2, src_x1:src_x2]

        return canvas

    def _render_trace_row(self, axes, row_idx, label, input_patch=None, reps=None, child_patches=None):
        axes[row_idx, 0].text(0.5, 0.5, label, ha="center", va="center", fontsize=8)
        axes[row_idx, 0].axis("off")

        n_cols = axes.shape[1]
        for col_idx in range(1, n_cols):
            axes[row_idx, col_idx].axis("off")

        if input_patch is not None:
            axes[row_idx, 1].imshow(input_patch)
            axes[row_idx, 1].axis("off")

        if child_patches is not None:
            for i, (title, patch) in enumerate(child_patches):
                col_idx = 2 + i
                if col_idx >= n_cols:
                    break
                if patch is None:
                    axes[row_idx, col_idx].text(0.5, 0.5, title, ha="center", va="center", fontsize=8)
                else:
                    axes[row_idx, col_idx].imshow(patch)
                    axes[row_idx, col_idx].set_title(title, fontsize=7)
                axes[row_idx, col_idx].axis("off")

        if reps is not None:
            for i, rep in enumerate(reps):
                col_idx = 2 + i
                if col_idx >= n_cols:
                    break
                axes[row_idx, col_idx].imshow(rep)
                axes[row_idx, col_idx].set_title(f"repr{i + 1}", fontsize=7)
                axes[row_idx, col_idx].axis("off")

    def _expansion_rows(
        self, img, all_results, labels_by_checkpoint, parent_stage, parent_pos,
        trace_block="last", k_nearest=4
    ):
        child_stage = parent_stage - 1
        children = self._children_for_parent(child_stage, parent_pos)
        parent_patch = self._stage_patch(img, parent_stage, parent_pos)
        child_patches = []
        for child in children:
            if child["is_pad"] or child["pos"] is None:
                child_patches.append(("PAD", None))
            else:
                child_patches.append((
                    f"s{child_stage} p{child['pos']}",
                    self._stage_patch(img, child_stage, child["pos"])
                ))

        rows = [{
            "label": f"s{parent_stage} p{parent_pos}\n-> s{child_stage}",
            "input_patch": parent_patch,
            "child_patches": child_patches,
        }]

        for child in children:
            if child["is_pad"] or child["pos"] is None:
                rows.append({
                    "label": f"s{child_stage}\nPAD",
                    "input_patch": None,
                    "reps": [],
                })
                continue
            block_idx, cluster, reps = self._cluster_and_representatives_for_pos(
                all_results, labels_by_checkpoint, child_stage, child["pos"],
                trace_block=trace_block, k_nearest=k_nearest
            )
            rows.append({
                "label": f"s{child_stage} b{block_idx}\npos{child['pos']}\ncluster {cluster}",
                "input_patch": self._stage_patch(img, child_stage, child["pos"]),
                "reps": reps,
            })

        valid_children = [
            child for child in children
            if not child["is_pad"] and child["pos"] is not None
        ]
        return rows, valid_children

    def _save_trace_page(self, rows, save_path, title, n_cols):
        fig_h = max(2.5, 1.6 * len(rows))
        fig, axes = plt.subplots(len(rows), n_cols, figsize=(2.0 * n_cols, fig_h))
        if len(rows) == 1:
            axes = axes.reshape(1, -1)

        for row_idx, row in enumerate(rows):
            self._render_trace_row(
                axes, row_idx, row["label"],
                input_patch=row.get("input_patch"),
                reps=row.get("reps"),
                child_patches=row.get("child_patches"),
            )

        plt.suptitle(title, fontsize=11)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    def _save_single_row_image(self, row, save_path, n_repr=4):
        """
        將一個「input patch + repr1~repr{n_repr}」的橫列存成單張 PNG。
        僅處理有 reps 的列（代表圖列），展開列（有 child_patches）不存。

        Args:
            row: dict，需包含 'input_patch'（np array 或 None）與 'reps'（list of np arrays）
            save_path: 輸出路徑（含檔名）
            n_repr: 最多顯示幾張代表圖，預設 4
        """
        input_patch = row.get("input_patch")
        reps = row.get("reps") or []
        n_cols = 1 + n_repr  # 第 0 格 input patch，後面 n_repr 格代表圖

        fig, axes = plt.subplots(1, n_cols, figsize=(2.2 * n_cols, 2.4))
        if n_cols == 1:
            axes = np.array([axes])

        # 第 0 格：input patch
        if input_patch is not None:
            axes[0].imshow(input_patch)
        else:
            axes[0].text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=9)
        axes[0].set_title("input", fontsize=8)
        axes[0].axis("off")

        # 第 1~n_repr 格：代表圖
        for i in range(n_repr):
            ax = axes[1 + i]
            if i < len(reps) and reps[i] is not None:
                ax.imshow(reps[i])
                ax.set_title(f"repr{i + 1}", fontsize=8)
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _make_stage3_summary_row(
        self, img, all_results, labels_by_checkpoint, top_item,
        trace_block="last", k_nearest=4
    ):
        last_stage_idx = len(self.stage_resolutions) - 1
        pos = int(top_item["pos"])
        block_idx, cluster, reps = self._cluster_and_representatives_for_pos(
            all_results, labels_by_checkpoint, last_stage_idx, pos,
            trace_block=trace_block, k_nearest=k_nearest
        )
        return {
            "label": (
                f"top {top_item['rank']}\n"
                f"s{last_stage_idx} b{block_idx}\n"
                f"pos{pos}\ncluster {cluster}"
            ),
            "input_patch": self._stage_patch(img, last_stage_idx, pos),
            "reps": reps,
        }

    def save_topk_trace_representative_pages(
        self, img, all_results, labels_by_checkpoint, gradcam_result,
        trace_dir, trace_block="last", k_nearest=4,
        max_rows_per_fig=12, expansions_per_fig=2,
        save_single_rows=True,
    ):
        """
        每個 top-k patch 產生分頁 trace card。
        一個 expansion 不切頁；預設每頁 2 組 expansion，共 10 列。

        若 save_single_rows=True，BFS 展開時同步把每個「input patch + repr」列
        單獨存成一張 PNG 至 {trace_dir}/single_rows/，供 VLM 逐列分析。

        single_rows 命名規則：
          top-k 本身：  s{S}_b{B}_pos{P}_top{R}.png
          子 patch：    s{S}_b{B}_pos{P}_parent_s{PS}p{PP}.png
        其中 S=stage, B=block, P=pos, R=rank, PS=parent_stage, PP=parent_pos。
        """
        trace_dir = Path(trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        last_stage_idx = len(self.stage_resolutions) - 1
        n_cols = max(2 + k_nearest, 2 + max(
            int(info["m_h"] or 0) * int(info["m_w"] or 0)
            for info in self.stage_resolutions
        ))
        max_expansion_rows = max(
            1 + int(info["m_h"] or 0) * int(info["m_w"] or 0)
            for info in self.stage_resolutions
            if info["has_merge"]
        )
        max_fit = max(1, int(max_rows_per_fig) // max_expansion_rows)
        expansions_per_fig = max(1, min(int(expansions_per_fig), max_fit))

        single_rows_dir = trace_dir / "single_rows"
        if save_single_rows:
            single_rows_dir.mkdir(parents=True, exist_ok=True)

        def _save_sr_if_repr(row, stage, block_idx, pos, *, rank=None, parent_stage=None, parent_pos=None):
            """若 row 是代表圖列（有 reps），存 single row 圖。展開列（child_patches）略過。"""
            if not save_single_rows:
                return
            if "reps" not in row:
                return  # 展開列，不存
            if not row.get("reps"):
                return  # PAD / 缺 cluster 的空列不存，避免餵給 VLM
            if rank is not None:
                fname = f"s{stage}_b{block_idx}_pos{pos}_top{rank:02d}.png"
            else:
                fname = f"s{stage}_b{block_idx}_pos{pos}_parent_s{parent_stage}p{parent_pos}.png"
            self._save_single_row_image(row, single_rows_dir / fname, n_repr=k_nearest)

        for top_item in gradcam_result["top_positions"]:
            rank = int(top_item["rank"])
            top_pos = int(top_item["pos"])
            queue = [(last_stage_idx, top_pos, top_pos)]
            page = 1

            # --- page 1：summary row + 第一個 expansion ---
            summary_row = self._make_stage3_summary_row(
                img, all_results, labels_by_checkpoint, top_item,
                trace_block=trace_block, k_nearest=k_nearest
            )

            _last_block_indices = self._resolve_trace_block_indices(last_stage_idx, trace_block)
            _last_block = _last_block_indices[0]
            _save_sr_if_repr(summary_row, last_stage_idx, _last_block, top_pos, rank=rank)

            first_rows = [summary_row]
            if queue:
                parent_stage, parent_pos, _ = queue.pop(0)
                rows, children = self._expansion_rows(
                    img, all_results, labels_by_checkpoint,
                    parent_stage, parent_pos,
                    trace_block=trace_block, k_nearest=k_nearest
                )
                first_rows.extend(rows)

                # rows[1:] 對應「全部 children（含 PAD）」，不是 valid_children
                child_stage = parent_stage - 1
                _child_block = self._resolve_trace_block_indices(child_stage, trace_block)[0]
                all_children = self._children_for_parent(child_stage, parent_pos)
                for row, child in zip(rows[1:], all_children):
                    if child["is_pad"] or child["pos"] is None:
                        continue
                    _save_sr_if_repr(
                        row, child_stage, _child_block, int(child["pos"]),
                        parent_stage=parent_stage, parent_pos=parent_pos,
                    )

                for child in children:
                    if child["stage"] > 0:
                        queue.append((child["stage"], child["pos"], top_pos))

            self._save_trace_page(
                first_rows,
                trace_dir / f"top{rank:02d}_pos{top_pos}_trace_page{page:02d}.png",
                f"Top {rank} pos{top_pos} trace page {page}",
                n_cols,
            )
            page += 1

            # --- 後續 page：每次展開 expansions_per_fig 組 ---
            while queue:
                pending_expansions = []

                for _ in range(expansions_per_fig):
                    if not queue:
                        break
                    parent_stage, parent_pos, _top_ancestor = queue.pop(0)
                    rows, children = self._expansion_rows(
                        img, all_results, labels_by_checkpoint,
                        parent_stage, parent_pos,
                        trace_block=trace_block, k_nearest=k_nearest
                    )
                    pending_expansions.extend(rows)

                    child_stage = parent_stage - 1
                    _child_block = self._resolve_trace_block_indices(child_stage, trace_block)[0]
                    all_children = self._children_for_parent(child_stage, parent_pos)
                    for row, child in zip(rows[1:], all_children):
                        if child["is_pad"] or child["pos"] is None:
                            continue
                        _save_sr_if_repr(
                            row, child_stage, _child_block, int(child["pos"]),
                            parent_stage=parent_stage, parent_pos=parent_pos,
                        )

                    for child in children:
                        if child["stage"] > 0:
                            queue.append((child["stage"], child["pos"], _top_ancestor))

                if len(pending_expansions) > max_rows_per_fig:
                    print(
                        f"警告: trace page 有 {len(pending_expansions)} 列，"
                        f"超過 max_rows_per_fig={max_rows_per_fig}"
                    )
                self._save_trace_page(
                    pending_expansions,
                    trace_dir / f"top{rank:02d}_pos{top_pos}_trace_page{page:02d}.png",
                    f"Top {rank} pos{top_pos} trace page {page}",
                    n_cols,
                )
                page += 1

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
        一次 forward 收集所有 (stage, block) 的特徵與圖片。
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
        for (s, b), (feats, H, W) in all_features.items():
            print(f"  stage{s}_block{b}: {feats.shape} (H={H}, W={W})")
        
        return all_features, images

    def extract_all_head_features_single_pass(self, max_samples=None):
        """
        一次 forward 收集所有 (stage, block) 的 per-head 特徵與圖片。
        只讀一次資料、只跑一次模型，避免重複計算。

        Returns:
            all_head_features: dict[(stage_idx, block_idx)] -> (
                features_heads [N_samples, N, num_heads, head_dim], H, W
            )
            images: [N_samples, 3, H_img, W_img]
        """
        all_heads_per_checkpoint = {}  # (s,b) -> (list of head feats, H, W)
        all_imgs = []
        n_batches = len(self.dataloader)

        with torch.no_grad():
            for batch_idx, (imgs, _) in enumerate(self.dataloader):
                imgs = imgs.to(self.device)
                checkpoints = get_all_features_and_heads_at_checkpoints(self.model, imgs)

                for (s, b), (_, feat_heads, H, W) in checkpoints.items():
                    if (s, b) not in all_heads_per_checkpoint:
                        all_heads_per_checkpoint[(s, b)] = ([], H, W)
                    all_heads_per_checkpoint[(s, b)][0].append(feat_heads.cpu())

                all_imgs.append(imgs.cpu())
                total = sum(f.shape[0] for f in all_imgs)
                if max_samples is not None and total >= max_samples:
                    break
                if n_batches > 0 and (batch_idx + 1) % max(1, n_batches // 10) == 0:
                    pct = 100 * (batch_idx + 1) / n_batches
                    print(f"\r  收集 head 特徵: batch {batch_idx+1}/{n_batches} ({pct:.0f}%)", end="", flush=True)

        images = torch.cat(all_imgs, dim=0)
        if max_samples is not None:
            images = images[:max_samples]

        all_head_features = {}
        for (s, b), (feat_list, H, W) in all_heads_per_checkpoint.items():
            feats = torch.cat(feat_list, dim=0)
            if max_samples is not None:
                feats = feats[:max_samples]
            all_head_features[(s, b)] = (feats, H, W)

        total = images.shape[0]
        if n_batches > 0:
            print()
        print(f"單次 forward 共有 {total} 張圖片 (head 特徵)")
        for (s, b), (feats, H, W) in all_head_features.items():
            print(f"  stage{s}_block{b}: {feats.shape} (H={H}, W={W})")

        return all_head_features, images

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
        
        img_np = self._image_to_numpy(full_img)
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
        n_clusters = self._effective_n_clusters(
            n_clusters, features.shape[0], context="per-position K-means"
        )
        
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

    def run_kmeans_per_position_per_head(
        self, features_heads, n_clusters=10, random_state=42,
        positions=None, heads=None, show_progress=True
    ):
        """
        對每個 (position, head) 的向量獨立做 K-means（cosine-equivalent）。

        Args:
            features_heads: [N_samples, N, num_heads, head_dim]
            positions: 要分析的位置索引列表，None 則分析全部位置
            heads: 要分析的 head 索引列表，None 則分析全部 head
        Returns:
            kmeans_dict: dict[(pos, head)] -> KMeans 物件
            cluster_centers: dict[(pos, head)] -> [n_clusters, head_dim]
        """
        if features_heads.dim() != 4:
            raise ValueError(
                f"`features_heads` 應為 [N_samples, N, num_heads, head_dim]，目前為 {features_heads.shape}"
            )
        _, N, num_heads, _ = features_heads.shape
        if positions is None:
            positions = list(range(N))
        if heads is None:
            heads = list(range(num_heads))

        kmeans_dict = {}
        cluster_centers = {}
        total = len(positions) * len(heads)
        done = 0
        n_clusters = self._effective_n_clusters(
            n_clusters, features_heads.shape[0], context="per-head K-means"
        )

        for p in positions:
            for h in heads:
                X = features_heads[:, p, h, :].numpy()  # [N_samples, head_dim]
                # 先 normalize，讓 Euclidean KMeans 等價於 cosine similarity 分群
                X_norm = normalize(X, norm='l2', axis=1)
                kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(X_norm)
                kmeans_dict[(p, h)] = kmeans
                cluster_centers[(p, h)] = kmeans.cluster_centers_

                done += 1
                if show_progress and total > 0:
                    step = max(1, total // 20)
                    if done % step == 0 or done == total:
                        pct = 100 * done / total
                        print(f"\r     K-means (pos,head) 進度: {done}/{total} ({pct:.0f}%)", end="", flush=True)
        if show_progress and total > 0:
            print(f"\r     K-means (pos,head) 進度: {total}/{total} (100%)   ")

        return kmeans_dict, cluster_centers

    def get_representative_images_per_position(
            self, features, images, kmeans_dict, stage_idx,
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
        n_clusters = next(iter(kmeans_dict.values())).n_clusters
    
        representatives = {}
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
                    patch = self._stage_patch_with_padding(images[idx], stage_idx, p)
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

    def get_representative_images_per_position_per_head(
        self, features_heads, images, kmeans_dict, stage_idx,
        k_nearest=5, show_progress=True
    ):
        """
        對每個 (position, head) 的每個群聚中心，取得最近的 k 張圖 patch 作為代表圖。
        距離在該 head 的 normalized 特徵空間計算；裁圖仍用該 position 的感受野。

        Args:
            features_heads: [N_samples, N, num_heads, head_dim]
            kmeans_dict: dict[(pos, head)] -> KMeans
            stage_idx: 用於裁出 padding 後的 position patch
            k_nearest: 每個群聚取幾張代表圖

        Returns:
            representatives: dict[(pos, head, cluster)] -> list of patch arrays
        """
        if features_heads.dim() != 4:
            raise ValueError(
                f"`features_heads` 應為 [N_samples, N, num_heads, head_dim]，目前為 {features_heads.shape}"
            )
        if not kmeans_dict:
            return {}

        n_clusters = next(iter(kmeans_dict.values())).n_clusters
        representatives = {}
        keys = list(kmeans_dict.keys())
        n_keys = len(keys)

        for i, (p, h) in enumerate(keys):
            X = features_heads[:, p, h, :].numpy()
            X_norm = normalize(X, norm='l2', axis=1)
            kmeans = kmeans_dict[(p, h)]
            centers = kmeans.cluster_centers_

            for c in range(n_clusters):
                dists = np.linalg.norm(X_norm - centers[c], axis=1)
                nearest_indices = np.argsort(dists)[:k_nearest]

                patches = []
                for idx in nearest_indices:
                    patch = self._stage_patch_with_padding(images[idx], stage_idx, p)
                    patches.append(patch)
                representatives[(p, h, c)] = patches

            if show_progress and n_keys > 0:
                pct = 100 * (i + 1) / n_keys
                step = max(1, n_keys // 20)
                if (i + 1) % step == 0 or i == n_keys - 1:
                    print(f"\r     代表圖 (pos,head) 進度: {i+1}/{n_keys} ({pct:.0f}%)", end="", flush=True)
        if show_progress and n_keys > 0:
            print(f"\r     代表圖 (pos,head) 進度: {n_keys}/{n_keys} (100%)   ")

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

    def assign_input_to_clusters_per_head(self, features_heads, kmeans_dict):
        """
        per-head 版本的 cluster 指派。

        Args:
            features_heads: [N_samples, N, num_heads, head_dim] 或 [N, num_heads, head_dim]
            kmeans_dict: dict[(pos, head)] -> KMeans
        Returns:
            labels: dict[(pos, head)] -> [N_samples]
        """
        if features_heads.dim() == 3:
            features_heads = features_heads.unsqueeze(0)
        if features_heads.dim() != 4:
            raise ValueError(
                f"`features_heads` 應為 4D 或 3D，收到 shape={features_heads.shape}"
            )
        labels = {}
        for (p, h), km in kmeans_dict.items():
            X = features_heads[:, p, h, :].numpy()
            X_norm = normalize(X, norm='l2', axis=1)
            labels[(p, h)] = km.predict(X_norm)
        return labels

    def sample_random_images(self, m, seed=None):
        """
        從 inference dataloader.dataset 直接依 index 均勻隨機取 m 張圖片。
        同一個 seed 會得到同一批 dataset index，不受 DataLoader shuffle 影響。
        
        Returns:
            images: [m, 3, H_img, W_img]
        """
        if m is None or m <= 0:
            return torch.empty((0, 3, self.img_size, self.img_size)), torch.empty((0,), dtype=torch.long)

        rng = np.random.default_rng(seed)
        dataset = getattr(self.inference_dataloader, "dataset", None)
        if dataset is None or len(dataset) == 0:
            return torch.empty((0, 3, self.img_size, self.img_size)), torch.empty((0,), dtype=torch.long)

        n_pick = min(int(m), len(dataset))
        indices = rng.choice(len(dataset), size=n_pick, replace=False).tolist()
        sampled = []
        for order, idx in enumerate(indices):
            if seed is None:
                img, lbl = dataset[int(idx)]
            else:
                item_seed = int(seed) + int(order)
                py_state = random.getstate()
                np_state = np.random.get_state()
                random.seed(item_seed)
                np.random.seed(item_seed % (2**32))
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(item_seed)
                    img, lbl = dataset[int(idx)]
                random.setstate(py_state)
                np.random.set_state(np_state)

            if not torch.is_tensor(img):
                img = torch.as_tensor(np.asarray(img))
            img = img.cpu()

            if torch.is_tensor(lbl):
                lbl_tensor = lbl.detach().cpu()
                if lbl_tensor.dim() > 0 and lbl_tensor.numel() > 1:
                    lbl_tensor = lbl_tensor.argmax()
                else:
                    lbl_tensor = lbl_tensor.reshape(-1)[0]
                lbl_tensor = lbl_tensor.long()
            else:
                lbl_arr = np.asarray(lbl)
                lbl_value = int(lbl_arr.argmax()) if lbl_arr.ndim > 0 and lbl_arr.size > 1 else int(lbl_arr)
                lbl_tensor = torch.tensor(lbl_value, dtype=torch.long)
            sampled.append((img, lbl_tensor))

        images = torch.stack([item[0] for item in sampled], dim=0)
        labels = torch.stack([item[1] for item in sampled], dim=0)
        return images, labels

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
        img_np = self._image_to_numpy(img)
        axes[0].imshow(img_np)
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

    def visualize_heads_for_position_clusters(
        self, representatives, position, heads, n_clusters, k_nearest=5,
        save_path=None, show=False, clusters_per_fig=10
    ):
        """
        固定 position，比較不同 head 的 cluster 代表圖。
        每列代表一個 (head, cluster)。
        """
        total_parts = (n_clusters + clusters_per_fig - 1) // clusters_per_fig
        for part_idx in range(total_parts):
            c_start = part_idx * clusters_per_fig
            c_end = min(c_start + clusters_per_fig, n_clusters)
            n_rows = len(heads) * (c_end - c_start)
            n_cols = 1 + k_nearest

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 1.2 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)

            row = 0
            for h in heads:
                for c in range(c_start, c_end):
                    axes[row, 0].text(0.5, 0.5, f"Pos {position}\nHead {h}\nCluster {c}",
                                      ha='center', va='center', fontsize=9)
                    axes[row, 0].axis('off')
                    for i in range(k_nearest):
                        key = (position, h, c)
                        if key in representatives and i < len(representatives[key]):
                            axes[row, 1 + i].imshow(representatives[key][i])
                        axes[row, 1 + i].axis('off')
                    row += 1

            plt.suptitle(f'Position {position}: head-wise representatives ({c_start}~{c_end-1})')
            plt.tight_layout()
            if save_path:
                out_path = Path(save_path).parent / f"{Path(save_path).stem}_pos{position}_part{part_idx}.png"
                plt.savefig(out_path, dpi=150)
            if show:
                plt.show()
            else:
                plt.close(fig)

    def visualize_positions_for_head_clusters(
        self, representatives, head, positions, n_clusters, k_nearest=5,
        save_path=None, show=False, clusters_per_fig=10
    ):
        """
        固定 head，比較不同 position 的 cluster 代表圖。
        每列代表一個 (position, cluster)。
        """
        total_parts = (n_clusters + clusters_per_fig - 1) // clusters_per_fig
        for part_idx in range(total_parts):
            c_start = part_idx * clusters_per_fig
            c_end = min(c_start + clusters_per_fig, n_clusters)
            n_rows = len(positions) * (c_end - c_start)
            n_cols = 1 + k_nearest

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 1.2 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)

            row = 0
            for p in positions:
                for c in range(c_start, c_end):
                    axes[row, 0].text(0.5, 0.5, f"Head {head}\nPos {p}\nCluster {c}",
                                      ha='center', va='center', fontsize=9)
                    axes[row, 0].axis('off')
                    for i in range(k_nearest):
                        key = (p, head, c)
                        if key in representatives and i < len(representatives[key]):
                            axes[row, 1 + i].imshow(representatives[key][i])
                        axes[row, 1 + i].axis('off')
                    row += 1

            plt.suptitle(f'Head {head}: position-wise representatives ({c_start}~{c_end-1})')
            plt.tight_layout()
            if save_path:
                out_path = Path(save_path).parent / f"{Path(save_path).stem}_head{head}_part{part_idx}.png"
                plt.savefig(out_path, dpi=150)
            if show:
                plt.show()
            else:
                plt.close(fig)

    def summarize_head_cluster_usage(self, labels_per_head, n_clusters):
        """
        計算每個 head 的 cluster 使用分布，以及 head 間 JS/cosine 距離矩陣。

        Args:
            labels_per_head: dict[(pos, head)] -> [N_samples]
            n_clusters: 群聚數
        Returns:
            summary: dict，包含 histograms、js_divergence、cosine_similarity
        """
        head_indices = sorted({h for (_, h) in labels_per_head.keys()})
        head_to_row = {h: i for i, h in enumerate(head_indices)}
        num_heads = len(head_indices)
        usage_counts = np.zeros((num_heads, n_clusters), dtype=np.float64)
        for (pos, h), arr in labels_per_head.items():
            _ = pos  # 僅做聚合統計，保留 key 結構
            binc = np.bincount(np.asarray(arr, dtype=np.int64), minlength=n_clusters)
            usage_counts[head_to_row[h]] += binc

        usage_probs = usage_counts / np.clip(usage_counts.sum(axis=1, keepdims=True), 1e-12, None)

        js_mat = np.zeros((num_heads, num_heads), dtype=np.float64)
        cos_mat = np.zeros((num_heads, num_heads), dtype=np.float64)
        for i in range(num_heads):
            for j in range(num_heads):
                js_mat[i, j] = _js_divergence(usage_probs[i], usage_probs[j])
                cos_mat[i, j] = _cosine_similarity(usage_probs[i], usage_probs[j])

        summary = {
            "head_indices": head_indices,
            "usage_counts": usage_counts.tolist(),
            "usage_probs": usage_probs.tolist(),
            "js_divergence": js_mat.tolist(),
            "cosine_similarity": cos_mat.tolist(),
        }
        return summary

    def save_head_cluster_summary(self, summary, out_dir):
        """
        將 per-head 統計存成 json 與 csv。
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / "head_cluster_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        usage_probs = np.asarray(summary["usage_probs"])
        js = np.asarray(summary["js_divergence"])
        cos = np.asarray(summary["cosine_similarity"])
        head_indices = summary.get("head_indices", list(range(usage_probs.shape[0])))

        usage_csv = out_dir / "head_cluster_usage_probs.csv"
        with open(usage_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = ["head"] + [f"cluster_{i}" for i in range(usage_probs.shape[1])]
            w.writerow(header)
            for h in range(usage_probs.shape[0]):
                w.writerow([head_indices[h]] + usage_probs[h].tolist())

        js_csv = out_dir / "head_js_divergence.csv"
        with open(js_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["head_i/head_j"] + [f"head_{head_indices[j]}" for j in range(js.shape[1])])
            for i in range(js.shape[0]):
                w.writerow([f"head_{head_indices[i]}"] + js[i].tolist())

        cos_csv = out_dir / "head_cosine_similarity.csv"
        with open(cos_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["head_i/head_j"] + [f"head_{head_indices[j]}" for j in range(cos.shape[1])])
            for i in range(cos.shape[0]):
                w.writerow([f"head_{head_indices[i]}"] + cos[i].tolist())

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
        
        effective_n_clusters = self._effective_n_clusters(
            n_clusters, features.shape[0], context=f"stage {stage_idx} K-means"
        )
        print("  對每個位置做 K-means...")
        kmeans_dict, _ = self.run_kmeans_per_position(
            features, n_clusters=effective_n_clusters, positions=positions
        )
        
        print("  取得各群聚的代表圖...")
        representatives = self.get_representative_images_per_position(
            features, images, kmeans_dict, stage_idx,
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
            representatives, (H, W), effective_n_clusters, k_nearest,
            positions_to_show=positions_to_show, save_path=save_dir,
            clusters_per_fig=10
        )
        
        return {
            'features': features,
            'images': images,
            'kmeans_dict': kmeans_dict,
            'representatives': representatives,
            'H': H, 'W': W,
            'n_clusters': effective_n_clusters,
        }

    @staticmethod
    def _filter_last_block_per_stage(checkpoints):
        """保留每個 stage 中 block index 最大的 checkpoint。"""
        last_block_per_stage = {}
        for stage_idx, block_idx in checkpoints:
            if stage_idx not in last_block_per_stage or block_idx > last_block_per_stage[stage_idx]:
                last_block_per_stage[stage_idx] = block_idx
        return [
            (stage_idx, block_idx)
            for stage_idx, block_idx in checkpoints
            if block_idx == last_block_per_stage[stage_idx]
        ]

    def full_analysis_all_stages_blocks(
        self, max_samples=None, n_clusters=10, k_nearest=5,
        positions=None, save_dir=None,
        m_inference=None, inference_seed=42,
        clusters_per_fig=10, n_clusters_per_stage=None,
        save_cluster_representatives=True,
        mode="token", heads=None,
        gradcam_top_k=10, trace_block="last", save_gradcam_trace=True,
        trace_max_rows_per_fig=12, trace_expansions_per_fig=2,
        save_all_inference_repr=False,
        use_kmeans_cache=True, model_path=None,
        dataset_name=None, analysis_split=None, random_state=42,
    ):
        """
        對每個 stage 的最後一個 block 做 K-means 分析並存代表圖。
        使用單次 forward 收集所有 checkpoint 特徵，避免重複計算。
        若指定 m_inference，會隨機取 m 張圖推論 cluster 指派並存檔。

        Args:
            positions: 要分析的位置，None 則分析全部
            m_inference: 隨機取幾張圖做推論並存檔，None 則不做
            inference_seed: 隨機取圖的 seed
            clusters_per_fig: 每張圖顯示幾個 cluster，n_clusters 大時可拆圖
            n_clusters_per_stage: dict[stage_idx -> n_clusters]，若提供則依 stage 使用不同 cluster 數
            save_cluster_representatives: 是否儲存每個 stage/block/position 的 cluster 代表圖
            mode: "token"（舊流程）或 "head"（per-head 分析）
            heads: mode="head" 時可指定要分析的 head 索引列表，None 表示全部 head
            gradcam_top_k: inference 時用 GradCAM 在最後 stage 取前 k 個 token；None 或 <=0 表示不輸出 trace
            trace_block: "last"、"all" 或指定 block index，用於輸出 traced positions 的 cluster
            save_gradcam_trace: 是否輸出 GradCAM top-k 回溯 JSON/視覺化
            trace_max_rows_per_fig: trace representative page 每張最多列數
            trace_expansions_per_fig: trace representative page 每張最多放幾組 parent expansion
            save_all_inference_repr: 是否保留舊行為，存所有非 top-k position 的 repr 圖
            use_kmeans_cache: 是否啟用 K-means 結果 cache，相同 model/dataset/n_clusters 時跳過重算
            model_path: model checkpoint 路徑，用於 cache meta 比對，None 時 cache 永遠失效
            dataset_name: 資料集名稱，寫入 cache meta
            analysis_split: K-means 使用的 split（train/test），寫入 cache meta
            random_state: K-means 隨機種子，寫入 cache meta 並傳入分群
        """
        if save_dir is None:
            save_dir = Path(__file__).resolve().parent / 'plots' / 'kmeans_all_stages_blocks'
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if mode not in {"token", "head"}:
            raise ValueError(f"mode 必須是 'token' 或 'head'，目前收到: {mode}")

        all_results = {}
        if mode == "token":
            print("\n=== 單次 forward 收集所有 (stage, block) 特徵 ===")
            all_features, images = self.extract_all_features_single_pass(
                max_samples=max_samples
            )

            checkpoints = self._filter_last_block_per_stage(sorted(all_features.keys()))
            total_cp = len(checkpoints)
            for cp_idx, ((stage_idx, block_idx), (features, H, W)) in enumerate(
                [(k, all_features[k]) for k in checkpoints]
            ):
                requested_nc = n_clusters_per_stage.get(stage_idx, n_clusters) if n_clusters_per_stage else n_clusters
                nc = self._effective_n_clusters(
                    requested_nc, features.shape[0],
                    context=f"stage {stage_idx} block {block_idx} K-means"
                )
                stage_dir = save_dir / f"stage{stage_idx}_block{block_idx}"
                stage_dir.mkdir(parents=True, exist_ok=True)

                pct_total = 100 * (cp_idx + 1) / total_cp if total_cp > 0 else 0
                print(f"\n=== Stage {stage_idx} Block {block_idx} [總進度 {pct_total:.0f}% ({cp_idx+1}/{total_cp})] (K-means, n_clusters={nc}) ===")

                # K-means cache 機制
                cache_dir = save_dir / "kmeans_cache"
                pkl_path, meta_path = self._kmeans_cache_path(
                    cache_dir, stage_idx, block_idx, nc, mode="token"
                )
                meta = self._kmeans_cache_meta(
                    model_path, str(save_dir), stage_idx, block_idx, nc, max_samples,
                    mode="token",
                    dataset_name=dataset_name,
                    analysis_split=analysis_split,
                    positions=positions,
                    random_state=random_state,
                )
                kmeans_dict = None
                if use_kmeans_cache:
                    kmeans_dict = self._load_kmeans_cache(pkl_path, meta_path, meta)
                    if kmeans_dict is not None:
                        print(f"  ✓ 載入 K-means cache (stage{stage_idx}_block{block_idx}_nc{nc})")
                if kmeans_dict is None:
                    print(f"  對每個位置做 K-means...")
                    kmeans_dict, _ = self.run_kmeans_per_position(
                        features, n_clusters=nc, positions=positions,
                        random_state=random_state,
                    )
                    if use_kmeans_cache:
                        self._save_kmeans_cache(kmeans_dict, pkl_path, meta_path, meta)
                        print(f"  ✓ 已存 K-means cache (stage{stage_idx}_block{block_idx}_nc{nc})")

                print(f"  取得各群聚的代表圖...")
                representatives = self.get_representative_images_per_position(
                    features, images, kmeans_dict, stage_idx,
                    k_nearest=k_nearest
                )

                if save_cluster_representatives:
                    positions_to_save = list(kmeans_dict.keys())
                    print(f"  儲存 {len(positions_to_save)} 個位置的代表圖...")
                    self.visualize_position_clusters(
                        representatives, (H, W), nc, k_nearest,
                        positions_to_show=positions_to_save,
                        save_path=stage_dir / "repr",
                        show=False,
                        clusters_per_fig=clusters_per_fig
                    )
                else:
                    print("  略過 stage/block 的 cluster 代表圖儲存（save_cluster_representatives=False）")

                all_results[(stage_idx, block_idx)] = {
                    'features': features,
                    'images': images,
                    'kmeans_dict': kmeans_dict,
                    'representatives': representatives,
                    'H': H, 'W': W,
                    'n_clusters': nc,
                    'mode': 'token',
                }

            if m_inference is not None and m_inference > 0:
                self._infer_and_save_m_images(
                    all_results, m_inference, n_clusters, k_nearest,
                    save_dir, inference_seed, n_clusters_per_stage,
                    gradcam_top_k=gradcam_top_k,
                    trace_block=trace_block,
                    save_gradcam_trace=save_gradcam_trace,
                    trace_max_rows_per_fig=trace_max_rows_per_fig,
                    trace_expansions_per_fig=trace_expansions_per_fig,
                    save_all_inference_repr=save_all_inference_repr,
                )
            return all_results

        # mode == "head"
        print("\n=== 單次 forward 收集所有 (stage, block) per-head 特徵 ===")
        all_head_features, images = self.extract_all_head_features_single_pass(
            max_samples=max_samples
        )

        checkpoints = self._filter_last_block_per_stage(sorted(all_head_features.keys()))
        total_cp = len(checkpoints)
        for cp_idx, ((stage_idx, block_idx), (features_heads, H, W)) in enumerate(
            [(k, all_head_features[k]) for k in checkpoints]
        ):
            requested_nc = n_clusters_per_stage.get(stage_idx, n_clusters) if n_clusters_per_stage else n_clusters
            nc = self._effective_n_clusters(
                requested_nc, features_heads.shape[0],
                context=f"stage {stage_idx} block {block_idx} per-head K-means"
            )
            stage_dir = save_dir / f"stage{stage_idx}_block{block_idx}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            per_pos_dir = stage_dir / "per_position"
            per_head_dir = stage_dir / "per_head"
            stat_dir = stage_dir / "stats"
            if save_cluster_representatives:
                per_pos_dir.mkdir(parents=True, exist_ok=True)
                per_head_dir.mkdir(parents=True, exist_ok=True)
            stat_dir.mkdir(parents=True, exist_ok=True)

            pct_total = 100 * (cp_idx + 1) / total_cp if total_cp > 0 else 0
            print(f"\n=== Stage {stage_idx} Block {block_idx} [總進度 {pct_total:.0f}% ({cp_idx+1}/{total_cp})] (per-head K-means, n_clusters={nc}) ===")
            num_heads = int(features_heads.shape[2])
            heads_to_use = list(range(num_heads)) if heads is None else list(heads)

            cache_dir = save_dir / "kmeans_cache"
            pkl_path, meta_path = self._kmeans_cache_path(
                cache_dir, stage_idx, block_idx, nc, mode="head", heads=heads_to_use
            )
            meta = self._kmeans_cache_meta(
                model_path, str(save_dir), stage_idx, block_idx, nc, max_samples,
                mode="head", heads=heads_to_use,
                dataset_name=dataset_name,
                analysis_split=analysis_split,
                positions=positions,
                random_state=random_state,
            )
            kmeans_head_dict = None
            if use_kmeans_cache:
                kmeans_head_dict = self._load_kmeans_cache(pkl_path, meta_path, meta)
                if kmeans_head_dict is not None:
                    print(f"  ✓ 載入 K-means cache (stage{stage_idx}_block{block_idx}_nc{nc}_head)")
            if kmeans_head_dict is None:
                print("  對每個 (position, head) 做 K-means...")
                kmeans_head_dict, _ = self.run_kmeans_per_position_per_head(
                    features_heads, n_clusters=nc, positions=positions,
                    heads=heads_to_use, random_state=random_state,
                )
                if use_kmeans_cache:
                    self._save_kmeans_cache(kmeans_head_dict, pkl_path, meta_path, meta)
                    print(f"  ✓ 已存 K-means cache (stage{stage_idx}_block{block_idx}_nc{nc}_head)")

            print("  取得各 (position, head, cluster) 的代表圖...")
            representatives_head = self.get_representative_images_per_position_per_head(
                features_heads, images, kmeans_head_dict, stage_idx,
                k_nearest=k_nearest
            )

            if save_cluster_representatives:
                # 視覺化 A: 固定 position，比較所有 head（儲存所有位置）
                pos_to_save = sorted(list({p for (p, _) in kmeans_head_dict.keys()}))
                print(f"  儲存每個 position 的 head 比較圖 ({len(pos_to_save)} 個位置)...")
                for p in pos_to_save:
                    self.visualize_heads_for_position_clusters(
                        representatives_head, p, heads_to_use, nc, k_nearest=k_nearest,
                        save_path=per_pos_dir / f"repr_heads_pos{p}",
                        show=False, clusters_per_fig=clusters_per_fig
                    )

                # 視覺化 B: 固定 head，比較所有 position（儲存所有 head）
                print(f"  儲存每個 head 的 position 比較圖 ({len(heads_to_use)} 個 head)...")
                for h in heads_to_use:
                    self.visualize_positions_for_head_clusters(
                        representatives_head, h, pos_to_save, nc, k_nearest=k_nearest,
                        save_path=per_head_dir / f"repr_positions_head{h}",
                        show=False, clusters_per_fig=clusters_per_fig
                    )
            else:
                print("  略過 stage/block 的 per-head cluster 代表圖儲存（save_cluster_representatives=False）")

            print("  統計 head 的 cluster 使用分布與 head 間差異...")
            labels_per_head = self.assign_input_to_clusters_per_head(features_heads, kmeans_head_dict)
            summary = self.summarize_head_cluster_usage(
                labels_per_head, n_clusters=nc
            )
            self.save_head_cluster_summary(summary, stat_dir)
            usage_probs = np.asarray(summary["usage_probs"])
            head_indices = summary.get("head_indices", [])
            js_mat = np.asarray(summary["js_divergence"])
            if usage_probs.size > 0:
                top_clusters = np.argmax(usage_probs, axis=1).tolist()
                print("    每個 head 的主導 cluster:")
                for i, h in enumerate(head_indices):
                    print(f"      head {h}: cluster {top_clusters[i]} (p={usage_probs[i, top_clusters[i]]:.3f})")
            if js_mat.size > 0:
                print(f"    head 間平均 JS divergence: {float(js_mat.mean()):.4f}")
            print(f"    已存統計檔: {stat_dir}")

            all_results[(stage_idx, block_idx)] = {
                'features_heads': features_heads,
                'images': images,
                'kmeans_head_dict': kmeans_head_dict,
                'representatives_head': representatives_head,
                'head_summary': summary,
                'H': H, 'W': W,
                'n_clusters': nc,
                'num_heads': num_heads,
                'heads_used': heads_to_use,
                'mode': 'head',
            }

        if m_inference is not None and m_inference > 0:
            self._infer_and_save_m_images_head(
                all_results, m_inference, n_clusters, k_nearest,
                save_dir, inference_seed, n_clusters_per_stage,
                gradcam_top_k=gradcam_top_k,
                trace_block=trace_block,
                save_gradcam_trace=save_gradcam_trace,
                save_all_inference_repr=save_all_inference_repr,
            )
        return all_results

    def visualize_and_save_head_assignment(
        self, img, labels_single_head, H, W, heads, n_clusters, save_prefix, heads_per_fig=8
    ):
        """
        儲存 per-head cluster 指派視覺化。
        每張圖包含原圖 + 多個 head 的 HxW cluster map。
        labels_single_head: dict[(pos, head)] -> int
        """
        heads = list(heads)
        n_parts = (len(heads) + heads_per_fig - 1) // heads_per_fig
        img_np = self._image_to_numpy(img)
        cmap = plt.cm.get_cmap('tab10', n_clusters) if n_clusters <= 10 else 'viridis'
        vmax = n_clusters - 1 if n_clusters > 1 else 1

        for part_idx in range(max(n_parts, 1)):
            hs = heads[part_idx * heads_per_fig:(part_idx + 1) * heads_per_fig]
            if not hs:
                break
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
                    key = (p, h)
                    if key in labels_single_head:
                        grid[p // W, p % W] = float(labels_single_head[key])
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

    def save_head_labels_json(self, labels_single_head, out_path, H, W, heads):
        """儲存單張圖的 per-(pos, head) cluster 指派。"""
        payload = {
            "H": int(H),
            "W": int(W),
            "heads": [int(h) for h in heads],
            "labels": [
                {"pos": int(p), "head": int(h), "cluster": int(c)}
                for (p, h), c in sorted(labels_single_head.items())
            ],
        }
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _save_inference_repr_per_head(
        self, img, labels_single_head, representatives_head, k_nearest,
        inference_dir, img_idx, stage_idx, block_idx, heads=None,
    ):
        """
        對每個 (pos, head) 存「input patch + 該 head cluster 代表圖」單列 PNG。
        輸出至 inference/all_pos_head_trace/img{idx}/
        """
        out_dir = Path(inference_dir) / "all_pos_head_trace" / f"img{img_idx}"
        out_dir.mkdir(parents=True, exist_ok=True)
        pairs = sorted(labels_single_head.keys(), key=lambda x: (x[0], x[1]))
        if heads is not None:
            head_set = set(int(h) for h in heads)
            pairs = [(p, h) for (p, h) in pairs if int(h) in head_set]

        for pos, head in pairs:
            cluster = labels_single_head[(pos, head)]
            reps = representatives_head.get((pos, head, cluster), [])[:k_nearest]
            row = {
                "input_patch": self._stage_patch(img, stage_idx, pos),
                "reps": reps,
            }
            parent = self._parent_for_stage_position(stage_idx, pos)
            if parent is None:
                fname = f"s{stage_idx}_b{block_idx}_pos{pos}_head{head}_top00.png"
            else:
                parent_stage, parent_pos = parent
                fname = (
                    f"s{stage_idx}_b{block_idx}_pos{pos}_head{head}_"
                    f"parent_s{parent_stage}p{parent_pos}.png"
                )
            self._save_single_row_image(row, out_dir / fname, n_repr=k_nearest)

    def _cluster_records_for_position_per_head(
        self, labels_by_checkpoint, stage_idx, pos, heads, trace_block="last"
    ):
        """GradCAM JSON 用：回傳各 block 下每個 head 的 cluster 指派。"""
        records = []
        for block_idx in self._resolve_trace_block_indices(stage_idx, trace_block):
            labels_single = labels_by_checkpoint.get((stage_idx, block_idx), {})
            head_records = []
            for h in heads:
                key = (int(pos), int(h))
                if key in labels_single:
                    head_records.append({
                        "head": int(h),
                        "cluster": int(labels_single[key]),
                        "missing_kmeans": False,
                    })
                else:
                    head_records.append({
                        "head": int(h),
                        "cluster": None,
                        "missing_kmeans": True,
                    })
            records.append({
                "block": int(block_idx),
                "heads": head_records,
            })
        return records

    def build_gradcam_trace_payload_head(
        self, img_idx, gradcam_result, traces, labels_by_checkpoint,
        heads, trace_block="last"
    ):
        """組合 head 模式的 GradCAM top-k / 回溯 / per-head cluster labels。"""
        last_stage_idx = len(self.stage_resolutions) - 1
        heads = [int(h) for h in heads]
        payload = {
            "img_idx": int(img_idx),
            "mode": "head",
            "heads": heads,
            "target_class": int(gradcam_result["target_class"]),
            "pred_class": int(gradcam_result["pred_class"]),
            "last_stage": int(last_stage_idx),
            "last_stage_H": int(gradcam_result["H"]),
            "last_stage_W": int(gradcam_result["W"]),
            "trace_block": trace_block,
            "cam": gradcam_result["cam"].tolist(),
            "top_positions": [],
        }

        for top_item in gradcam_result["top_positions"]:
            last_pos = int(top_item["pos"])
            item_payload = {
                **top_item,
                "clusters": self._cluster_records_for_position_per_head(
                    labels_by_checkpoint, last_stage_idx, last_pos, heads, trace_block
                ),
                "sources_by_stage": {},
            }
            for stage_idx in range(last_stage_idx - 1, -1, -1):
                source_items = []
                for source in traces[last_pos].get(stage_idx, []):
                    pos = source["pos"]
                    source_payload = dict(source)
                    if source["is_pad"] or pos is None:
                        source_payload["clusters"] = []
                    else:
                        source_payload["clusters"] = self._cluster_records_for_position_per_head(
                            labels_by_checkpoint, stage_idx, int(pos), heads, trace_block
                        )
                    source_items.append(source_payload)
                item_payload["sources_by_stage"][str(stage_idx)] = source_items
            payload["top_positions"].append(item_payload)
        return payload

    def save_topk_trace_representative_pages_head(
        self, img, all_results, labels_by_checkpoint, gradcam_result,
        trace_dir, heads, trace_block="last", k_nearest=4,
    ):
        """
        GradCAM top-k 回溯：對每個 traced position × head 存 single-row 代表圖。
        命名：s{S}_b{B}_pos{P}_head{H}_top{R}.png 或 ..._parent_s{PS}p{PP}.png
        """
        trace_dir = Path(trace_dir)
        single_rows_dir = trace_dir / "single_rows"
        single_rows_dir.mkdir(parents=True, exist_ok=True)
        last_stage_idx = len(self.stage_resolutions) - 1
        heads = [int(h) for h in heads]

        def _save_pos_heads(stage, pos, *, rank=None, parent_stage=None, parent_pos=None):
            block_indices = self._resolve_trace_block_indices(stage, trace_block)
            block_idx = block_indices[0]
            key = (stage, block_idx)
            if key not in all_results or key not in labels_by_checkpoint:
                return
            labels_single = labels_by_checkpoint[key]
            representatives = all_results[key]["representatives_head"]
            for h in heads:
                cluster = labels_single.get((int(pos), int(h)))
                if cluster is None:
                    continue
                reps = representatives.get((int(pos), int(h), int(cluster)), [])[:k_nearest]
                if not reps:
                    continue
                row = {
                    "input_patch": self._stage_patch(img, stage, pos),
                    "reps": reps,
                }
                if rank is not None:
                    fname = f"s{stage}_b{block_idx}_pos{pos}_head{h}_top{rank:02d}.png"
                else:
                    fname = (
                        f"s{stage}_b{block_idx}_pos{pos}_head{h}_"
                        f"parent_s{parent_stage}p{parent_pos}.png"
                    )
                self._save_single_row_image(row, single_rows_dir / fname, n_repr=k_nearest)

        for top_item in gradcam_result["top_positions"]:
            rank = int(top_item["rank"])
            top_pos = int(top_item["pos"])
            _save_pos_heads(last_stage_idx, top_pos, rank=rank)

            queue = [(last_stage_idx, top_pos)]
            visited = {(last_stage_idx, top_pos)}
            while queue:
                parent_stage, parent_pos = queue.pop(0)
                if parent_stage <= 0:
                    continue
                child_stage = parent_stage - 1
                for child in self._children_for_parent(child_stage, parent_pos):
                    if child.get("is_pad") or child.get("pos") is None:
                        continue
                    child_pos = int(child["pos"])
                    node = (child_stage, child_pos)
                    if node in visited:
                        continue
                    visited.add(node)
                    _save_pos_heads(
                        child_stage, child_pos,
                        parent_stage=parent_stage, parent_pos=parent_pos,
                    )
                    queue.append(node)

    def _infer_and_save_m_images_head(
        self, all_results, m, n_clusters, k_nearest, save_dir, seed=42,
        n_clusters_per_stage=None, gradcam_top_k=10, trace_block="last",
        save_gradcam_trace=True, save_all_inference_repr=False,
    ):
        """
        head 模式推論：
          inference/img{i}_original.png
          inference/img{i}_s{s}_b{b}_head_assign_part{p}.png
          inference/img{i}_s{s}_b{b}_head_labels.json
          可選 all_pos_head_trace / GradCAM top-k per-head 代表圖
        """
        inference_dir = Path(save_dir) / "inference"
        inference_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== 隨機取 {m} 張圖推論 per-head cluster 指派 ===")
        sample_imgs, sample_gt_labels = self.sample_random_images(m, seed=seed)
        trace_jsonl_path = inference_dir / "gradcam_trace_clusters_head.jsonl"

        with torch.no_grad():
            imgs = sample_imgs.to(self.device)
            infer_checkpoints = get_all_features_and_heads_at_checkpoints(self.model, imgs)
            logits = self.model(imgs)
            sample_pred_labels = logits.argmax(dim=1).cpu()

        n_imgs = sample_imgs.shape[0]
        for img_idx in range(n_imgs):
            pct_img = 100 * (img_idx + 1) / n_imgs if n_imgs > 0 else 0
            print(f"\n  推論進度: 圖 {img_idx+1}/{n_imgs} ({pct_img:.0f}%)")
            img = sample_imgs[img_idx]
            img_np = self._image_to_numpy(img)

            gt = int(sample_gt_labels[img_idx].item())
            pred = int(sample_pred_labels[img_idx].item())
            gt_name = self.format_class_label(gt)
            pred_name = self.format_class_label(pred)
            correct = (gt == pred)
            title_color = "lime" if correct else "red"
            title_str = f"GT: {gt_name}  |  Pred: {pred_name}"

            fig, ax = plt.subplots(figsize=(4, 4.3))
            ax.imshow(img_np)
            ax.set_title(title_str, fontsize=10, color=title_color,
                         bbox=dict(facecolor='black', alpha=0.6, pad=3))
            ax.axis('off')
            plt.savefig(inference_dir / f"img{img_idx}_original.png",
                        bbox_inches='tight', pad_inches=0.1, dpi=150)
            plt.close(fig)

            labels_by_checkpoint = {}
            heads_by_checkpoint = {}
            for (stage_idx, block_idx), res in all_results.items():
                kmeans_head_dict = res['kmeans_head_dict']
                representatives_head = res['representatives_head']
                H, W = res['H'], res['W']
                heads_used = res.get('heads_used') or list(range(int(res.get('num_heads', 1))))
                nc = res.get('n_clusters') or (
                    n_clusters_per_stage.get(stage_idx, n_clusters) if n_clusters_per_stage else n_clusters
                )

                feat_heads = infer_checkpoints[(stage_idx, block_idx)][1][img_idx:img_idx + 1].cpu()
                labels = self.assign_input_to_clusters_per_head(feat_heads, kmeans_head_dict)
                labels_single = {(int(p), int(h)): int(arr[0]) for (p, h), arr in labels.items()}
                labels_by_checkpoint[(int(stage_idx), int(block_idx))] = labels_single
                heads_by_checkpoint[(int(stage_idx), int(block_idx))] = heads_used

                save_prefix = inference_dir / f"img{img_idx}_s{stage_idx}_b{block_idx}_head_assign"
                self.visualize_and_save_head_assignment(
                    sample_imgs[img_idx], labels_single, H, W, heads_used, nc,
                    save_prefix=save_prefix, heads_per_fig=8
                )
                self.save_head_labels_json(
                    labels_single,
                    inference_dir / f"img{img_idx}_s{stage_idx}_b{block_idx}_head_labels.json",
                    H, W, heads_used,
                )
                if save_all_inference_repr:
                    self._save_inference_repr_per_head(
                        sample_imgs[img_idx], labels_single, representatives_head,
                        k_nearest, inference_dir, img_idx, stage_idx, block_idx,
                        heads=heads_used,
                    )

            if save_gradcam_trace and gradcam_top_k is not None and gradcam_top_k > 0:
                # 用任一 checkpoint 的 heads（通常各 stage 相同）；取最後 stage
                last_stage_idx = len(self.stage_resolutions) - 1
                last_block = max(
                    b for (s, b) in labels_by_checkpoint.keys() if s == last_stage_idx
                )
                heads_used = heads_by_checkpoint[(last_stage_idx, last_block)]

                gradcam_result = self.compute_gradcam_topk(img, top_k=gradcam_top_k)
                traces = self.trace_last_stage_positions(gradcam_result["top_positions"])
                trace_payload = self.build_gradcam_trace_payload_head(
                    img_idx, gradcam_result, traces, labels_by_checkpoint,
                    heads=heads_used, trace_block=trace_block,
                )

                per_img_json = inference_dir / f"img{img_idx}_gradcam_trace_clusters_head.json"
                with open(per_img_json, "w", encoding="utf-8") as f:
                    json.dump(trace_payload, f, ensure_ascii=False, indent=2)

                mode = "w" if img_idx == 0 else "a"
                with open(trace_jsonl_path, mode, encoding="utf-8") as f:
                    f.write(json.dumps(trace_payload, ensure_ascii=False) + "\n")

                trace_dir = inference_dir / "gradcam_trace" / f"img{img_idx}"
                trace_dir.mkdir(parents=True, exist_ok=True)
                self.save_gradcam_topk_overview(
                    img, gradcam_result,
                    trace_dir / f"img{img_idx}_gradcam_top{gradcam_top_k}_overview.png"
                )
                self.save_gradcam_trace_summary(
                    gradcam_result, traces,
                    trace_dir / f"img{img_idx}_gradcam_top{gradcam_top_k}_trace_summary.png"
                )
                self.save_topk_trace_representative_pages_head(
                    img, all_results, labels_by_checkpoint, gradcam_result,
                    trace_dir,
                    heads=heads_used,
                    trace_block=trace_block,
                    k_nearest=k_nearest,
                )
                print(f"    已存 img{img_idx} 的 GradCAM top-{gradcam_top_k} per-head 回溯")

            msg = f"    已存 img{img_idx} 的原始圖與 per-head cluster 指派"
            if save_gradcam_trace and gradcam_top_k is not None and gradcam_top_k > 0:
                msg += "、GradCAM top-k per-head trace"
            if save_all_inference_repr:
                msg += "、所有 (pos, head) 代表圖"
            print(msg)

    def _infer_and_save_m_images(
        self, all_results, m, n_clusters, k_nearest, save_dir, seed=42,
        n_clusters_per_stage=None, gradcam_top_k=10, trace_block="last",
        save_gradcam_trace=True, trace_max_rows_per_fig=12,
        trace_expansions_per_fig=2, save_all_inference_repr=False
    ):
        """
        隨機取 m 張圖，對每個 (stage, block) 推論 cluster 指派並存檔。
        使用單次 forward 取得所有 checkpoint 特徵，避免重複計算。
        儲存: inference/img{i}_original.png, inference/img{i}_s{s}_b{b}.png
              inference/gradcam_trace/img{i}/... top-k trace 圖與代表圖。
              若 save_all_inference_repr=True，才額外儲存所有 position 的舊版 repr 圖。
        """
        inference_dir = save_dir / "inference"
        inference_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n=== 隨機取 {m} 張圖推論 cluster 指派 ===")
        sample_imgs, sample_gt_labels = self.sample_random_images(m, seed=seed)
        trace_jsonl_path = inference_dir / "gradcam_trace_clusters.jsonl"
        
        with torch.no_grad():
            imgs = sample_imgs.to(self.device)
            infer_checkpoints = get_all_features_at_checkpoints(self.model, imgs)
            logits = self.model(imgs)
            sample_pred_labels = logits.argmax(dim=1).cpu()
        
        n_imgs = sample_imgs.shape[0]
        for img_idx in range(n_imgs):
            pct_img = 100 * (img_idx + 1) / n_imgs if n_imgs > 0 else 0
            print(f"\n  推論進度: 圖 {img_idx+1}/{n_imgs} ({pct_img:.0f}%)")
            img = sample_imgs[img_idx]
            img_np = self._image_to_numpy(img)

            # 顯示 GT 和 Pred 的 label 名稱
            gt = int(sample_gt_labels[img_idx].item())
            pred = int(sample_pred_labels[img_idx].item())
            gt_name = self.format_class_label(gt)
            pred_name = self.format_class_label(pred)
            correct = (gt == pred)
            title_color = "lime" if correct else "red"
            title_str = f"GT: {gt_name}  |  Pred: {pred_name}"

            fig, ax = plt.subplots(figsize=(4, 4.3))
            ax.imshow(img_np)
            ax.set_title(title_str, fontsize=10, color=title_color,
                        bbox=dict(facecolor='black', alpha=0.6, pad=3))
            ax.axis('off')
            plt.savefig(inference_dir / f"img{img_idx}_original.png",
                        bbox_inches='tight', pad_inches=0.1, dpi=150)
            plt.close(fig)


            labels_by_checkpoint = {}
            for (stage_idx, block_idx), res in all_results.items():
                kmeans_dict = res['kmeans_dict']
                representatives = res['representatives']
                H, W = res['H'], res['W']
                nc = res.get('n_clusters') or (
                    n_clusters_per_stage.get(stage_idx, n_clusters) if n_clusters_per_stage else n_clusters
                )
                
                feat = infer_checkpoints[(stage_idx, block_idx)][0][img_idx:img_idx + 1].cpu()
                labels = self.assign_input_to_clusters(feat, kmeans_dict)
                labels_single = {int(p): int(labels[p][0]) for p in labels}
                labels_by_checkpoint[(int(stage_idx), int(block_idx))] = labels_single
                
                save_path = inference_dir / f"img{img_idx}_s{stage_idx}_b{block_idx}.png"
                self.visualize_and_save_cluster_assignment(
                    sample_imgs[img_idx], labels_single, H, W, nc,
                    save_path
                )
                if save_all_inference_repr:
                    self._save_inference_repr(
                        sample_imgs[img_idx], labels_single, representatives,
                        k_nearest, H, W, inference_dir, img_idx, stage_idx, block_idx
                    )

            if save_gradcam_trace and gradcam_top_k is not None and gradcam_top_k > 0:
                gradcam_result = self.compute_gradcam_topk(
                    img, top_k=gradcam_top_k
                )
                traces = self.trace_last_stage_positions(
                    gradcam_result["top_positions"]
                )
                trace_payload = self.build_gradcam_trace_payload(
                    img_idx, gradcam_result, traces, labels_by_checkpoint,
                    trace_block=trace_block
                )

                per_img_json = inference_dir / f"img{img_idx}_gradcam_trace_clusters.json"
                with open(per_img_json, "w", encoding="utf-8") as f:
                    json.dump(trace_payload, f, ensure_ascii=False, indent=2)

                mode = "w" if img_idx == 0 else "a"
                with open(trace_jsonl_path, mode, encoding="utf-8") as f:
                    f.write(json.dumps(trace_payload, ensure_ascii=False) + "\n")

                trace_dir = inference_dir / "gradcam_trace" / f"img{img_idx}"
                trace_dir.mkdir(parents=True, exist_ok=True)
                self.save_gradcam_topk_overview(
                    img, gradcam_result,
                    trace_dir / f"img{img_idx}_gradcam_top{gradcam_top_k}_overview.png"
                )
                self.save_gradcam_trace_summary(
                    gradcam_result, traces,
                    trace_dir / f"img{img_idx}_gradcam_top{gradcam_top_k}_trace_summary.png"
                )
                self.save_topk_trace_representative_pages(
                    img, all_results, labels_by_checkpoint, gradcam_result,
                    trace_dir,
                    trace_block=trace_block,
                    k_nearest=k_nearest,
                    max_rows_per_fig=trace_max_rows_per_fig,
                    expansions_per_fig=trace_expansions_per_fig
                )
                print(f"    已存 img{img_idx} 的 GradCAM top-{gradcam_top_k} 回溯 cluster JSON")
            msg = f"    已存 img{img_idx} 的原始圖與 cluster 指派圖"
            if save_gradcam_trace and gradcam_top_k is not None and gradcam_top_k > 0:
                msg += "、GradCAM top-k trace 圖"
            if save_all_inference_repr:
                msg += "、所有 position 代表圖"
            print(msg)

    def _save_inference_repr(
        self, img, labels_single, representatives, k_nearest,
        H, W, inference_dir, img_idx, stage_idx, block_idx
    ):
        """
        對每個 position 顯示其被分到的 cluster 的代表圖，並各自單獨存檔。
        輸出到 inference/all_pos_trace/img{idx}/，使用 GradCAM single_rows 的命名風格。
        """
        all_pos_dir = Path(inference_dir) / "all_pos_trace" / f"img{img_idx}"
        all_pos_dir.mkdir(parents=True, exist_ok=True)
        positions = sorted(labels_single.keys())

        for pos in positions:
            cluster = labels_single[pos]
            reps = representatives.get((pos, cluster), [])[:k_nearest]
            row = {
                "input_patch": self._stage_patch(img, stage_idx, pos),
                "reps": reps,
            }

            parent = self._parent_for_stage_position(stage_idx, pos)
            if parent is None:
                # 與 GradCAM trace 的 root row 命名一致；只有最後 stage 會沒有 parent。
                fname = f"s{stage_idx}_b{block_idx}_pos{pos}_top00.png"
            else:
                parent_stage, parent_pos = parent
                fname = (
                    f"s{stage_idx}_b{block_idx}_pos{pos}_"
                    f"parent_s{parent_stage}p{parent_pos}.png"
                )
            self._save_single_row_image(row, all_pos_dir / fname, n_repr=k_nearest)


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
    
    train_loader, test_loader = get_dataloader(
        dataset=dataset, root='./data/', batch_size=32,
        input_size=(img_size, img_size)
    )
    analysis_loader = test_loader if dataset == 'Caltech101' else train_loader
    display_mean = IMAGENET_MEAN if dataset == 'Caltech101' else None
    display_std = IMAGENET_STD if dataset == 'Caltech101' else None
    
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
    
    analyzer = ViTAnalyzer(
        model, analysis_loader, img_size=img_size,
        display_mean=display_mean, display_std=display_std,
        dataset_name=dataset
    )
    result = self.get_representative_images_per_position(
        features, images, kmeans_dict, stage_idx,
        k_nearest=k_nearest
    )
    
    # 示範：對單張圖做 cluster 指派
    sample_feat = result['features'][:1]
    labels = analyzer.assign_input_to_clusters(sample_feat, result['kmeans_dict'])
    pos_keys = sorted(labels.keys())[:10]
    print(f"範例：第一張圖在各位置的群聚標籤 (前 10 個位置): {[(p, labels[p][0]) for p in pos_keys]}")
    
    return analyzer, result


def run_dataset_analysis_all(
    model_path=None, dataset='Colored_MNIST', img_size=28, patch_size=1,
    max_samples=None, n_clusters=10, k_nearest=5, positions=None, save_dir=None,
    m_inference=None, inference_seed=42, clusters_per_fig=10,
    n_clusters_per_stage=None, model_args=None, data_root=None,
    save_cluster_representatives=True,
    mode="token", heads=None, analysis_batch_size=32,
    model_name=None, gradcam_top_k=10, trace_block="last",
    save_gradcam_trace=True, trace_max_rows_per_fig=12,
    trace_expansions_per_fig=2, save_all_inference_repr=False,
    analysis_split="auto",
    inference_split="test",
    use_kmeans_cache=True,
    random_state=42,
):
    """
    對每個 stage、每個 block、每個 position 存代表圖。
    若指定 m_inference，隨機取 m 張圖推論 cluster 指派並存檔。
    儲存至 save_dir/stage{s}_block{b}/repr_pos{pos}.png
    推論圖: save_dir/inference/img{i}_s{s}_b{b}.png

    Args:
        model_args: 若提供（如從 config），用於建立模型；否則用 patch_size 等參數
        mode: "token" 或 "head"
        heads: mode="head" 時可指定 head 索引列表
        analysis_split: "auto"、"train" 或 "test"。auto 會預設使用 train，
                        讓 K-means 以訓練集分群。
        inference_split: "train"、"test"、"auto" 或 "same_as_analysis"。
                        預設使用 test，讓測試集做推論；auto 也會解析成 test。
        use_kmeans_cache: 是否啟用 K-means cache
        random_state: K-means 隨機種子
    """
    import sys
    sys.path.insert(0, '.')
    from dataloader import get_dataloader
    _validate_mergingvit_model_args(model_args=model_args, model_name=model_name)
    
    _root = data_root if data_root is not None else str(Path(__file__).resolve().parent.parent / 'data')
    train_loader, test_loader = get_dataloader(
        dataset=dataset, root=_root, batch_size=analysis_batch_size,
        input_size=(img_size, img_size)
    )
    split = (analysis_split or "auto").lower()
    if split == "auto":
        split = "train"
    if split not in {"train", "test"}:
        raise ValueError(f"analysis_split 必須是 'auto'、'train' 或 'test'，目前收到: {analysis_split}")
    analysis_loader = train_loader if split == "train" else test_loader
    infer_split = (inference_split or "test").lower()
    if infer_split == "auto":
        infer_split = "test"
    if infer_split == "same_as_analysis":
        infer_split = split
    if infer_split not in {"train", "test"}:
        raise ValueError(
            "inference_split 必須是 'train'、'test' 或 'same_as_analysis'，"
            f"目前收到: {inference_split}"
        )
    inference_loader = train_loader if infer_split == "train" else test_loader
    print(f"K-means analysis split: {split}")
    print(f"Inference sampling split: {infer_split}")
    display_mean = IMAGENET_MEAN if dataset == 'Caltech101' else None
    display_std = IMAGENET_STD if dataset == 'Caltech101' else None
    
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
    
    analyzer = ViTAnalyzer(
        model, analysis_loader, img_size=img_size,
        display_mean=display_mean, display_std=display_std,
        dataset_name=dataset,
        inference_dataloader=inference_loader,
    )
    return analyzer.full_analysis_all_stages_blocks(
        max_samples=max_samples,
        n_clusters=n_clusters,
        k_nearest=k_nearest,
        positions=positions,
        save_dir=save_dir,
        m_inference=m_inference,
        inference_seed=inference_seed,
        clusters_per_fig=clusters_per_fig,
        n_clusters_per_stage=n_clusters_per_stage,
        save_cluster_representatives=save_cluster_representatives,
        mode=mode,
        heads=heads,
        gradcam_top_k=gradcam_top_k,
        trace_block=trace_block,
        save_gradcam_trace=save_gradcam_trace,
        trace_max_rows_per_fig=trace_max_rows_per_fig,
        trace_expansions_per_fig=trace_expansions_per_fig,
        save_all_inference_repr=save_all_inference_repr,
        use_kmeans_cache=use_kmeans_cache,
        model_path=model_path,
        dataset_name=dataset,
        analysis_split=split,
        random_state=random_state,
    )


# if __name__ == "__main__":
#     import config as cfg

#     # 從 config 讀取設定
#     model_name = cfg.config["model"]["name"]
#     model_args = cfg.config["model"]["args"]
#     ckpt_dir = cfg.config.get("kmeans_checkpoint_dir") or cfg.config["save_dir"]
#     _validate_mergingvit_model_args(model_args=model_args, model_name=model_name)
#     model_path = str(Path(cfg.config["root"]) / ckpt_dir / f"{model_name}_best.pth")
#     dataset = cfg.config["dataset"]
#     input_shape = cfg.config["input_shape"]
#     clusters_list = cfg.config.get("kmeans_clusters_per_stage", [30, 60, 120, 240])
#     n_clusters_per_stage = {i: v for i, v in enumerate(clusters_list)}

#     # 選項 A：單一 stage 分析（只畫中心位置）
#     # analyzer, result = run_colored_mnist_analysis(
#     #     model_path=model_path,
#     #     n_clusters=8, k_nearest=4, stage_idx=0
#     # )

#     # 選項 B：每個 stage、每個 block、每個 pos 都存代表圖，並隨機取 m 張圖推論
#     # cluster 數量依 config 的 kmeans_clusters_per_stage 設定
#     run_dataset_analysis_all(
#         model_path=model_path,
#         dataset=dataset,
#         img_size=input_shape[0],
#         patch_size=model_args.get("patch_size", 8),
#         max_samples=None,  # 可設 500 加速
#         n_clusters_per_stage=n_clusters_per_stage,
#         k_nearest=4,
#         clusters_per_fig=1,  # 每 10 個 cluster 一張圖
#         positions=None,  # None=全部位置；可傳 [0,1,105] 等子集加速
#         save_dir='plots/kmeans/Caltech101/',   # 預設 plots/kmeans_all_stages_blocks/
#         m_inference=3,   # 隨機取 m 張圖推論 cluster 指派並存檔
#         inference_seed=42,
#         save_cluster_representatives=False,  # 只存 inference 圖，不存每個 cluster 代表圖
#         model_args=model_args,  # 從 config 傳入，含 merge_size 等
#         data_root=str(Path(cfg.config["root"]) / "data"),
#         mode="token",
#         heads=None,
#         analysis_batch_size=cfg.config.get("batch_size", 4),
#         model_name=model_name,
#         gradcam_top_k=cfg.config.get("gradcam_trace_top_k", 5),
#         trace_block=cfg.config.get("gradcam_trace_block", "last"),
#         save_gradcam_trace=cfg.config.get("save_gradcam_trace", True),
#         trace_max_rows_per_fig=cfg.config.get("trace_max_rows_per_fig", 12),
#         trace_expansions_per_fig=cfg.config.get("trace_expansions_per_fig", 2),
#         save_all_inference_repr=cfg.config.get("save_all_inference_repr", False),
#     )
