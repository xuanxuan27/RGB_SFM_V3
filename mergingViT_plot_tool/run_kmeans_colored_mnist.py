#!/usr/bin/env python3
"""
以 Colored MNIST 執行 K-means 分析的主程式。
從專案根目錄執行: python mergingViT_plot_tool/run_kmeans_colored_mnist.py
"""
import sys
from pathlib import Path

# 確保可 import 專案模組
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mergingViT_plot_tool.Kmeans_analysis import run_colored_mnist_analysis

if __name__ == "__main__":
    # Colored MNIST: 28x28, patch_size=1
    # Stage 0: 28x28=784 個向量 (較慢)
    # Stage 1: 14x14=196 個向量 (符合範例)
    analyzer, result = run_colored_mnist_analysis(
        model_path=None,  # 可改為 'pth/Colored_MNIST/MergingVIT_best.pth'
        dataset='Colored_MNIST',
        img_size=28,
        patch_size=1,
        max_samples=500,
        n_clusters=8,
        k_nearest=4,
        stage_idx=1,  # 14x14=196 個向量
        positions=None,  # 可傳 [0,1,...,195] 或子集以加速
    )
