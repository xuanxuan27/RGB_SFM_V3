import torch
from file_tools import increment_path
from pathlib import Path
import os

project = "paper experiment"
name = 'HierarchicalViT_NonclassicFace'
group = "NonclassicFace"
tags = ['HierarchicalViT', 'NonclassicFace']
description = """
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
load_model_name = 'VIT_best'  # "RGB_SFMCNN_V2_best" 'only-rgb-25-channels', 'rgb-1-5-1-1-new_100_color', 'MergingVIT_best'

# arch = {
#     "name": 'RGB_SFMCNN_V2',
#     "need_calculate_status": False,
#     "args": {
#         "in_channels": 3,
#         "out_channels": 30,
#         "mode" : "rgb", #  'rgb', 'gray', or 'both'
#         "Conv2d_kernel": [[ (1, 1), (5, 5), (1, 1), (1, 1)],
#                           [ (5, 5),  (1, 1), (1, 1)]],
#         # SFM_methods: "alpha_mean" "max" "none"
#         "SFM_methods": [["alpha_mean", "alpha_mean", "alpha_mean", "alpha_mean"],
#                         ["alpha_mean", "alpha_mean", "alpha_mean", "alpha_mean"]],
#         "SFM_filters": [[  (1, 1), (2, 2),  (1, 3),  (1, 1)],
#                         [ (2, 2),  (1, 3),  (1, 1)]],
#         # 對應到畫圖時，該 channel 的形狀
#         "channels": [[(10, 10), (15, 15), (25, 25),  (35, 35)],
#                      [(7, 10), (15, 15), (35, 35)]],
#         "strides": [[1, 4, 1, 1],
#                     [4, 1, 1]],
#         "paddings": [[0, 0, 0, 0],
#                      [0, 0, 0]],
#         # color_filter : "new_10" "new_30" "new_100"  "old_30"
#         "color_filter" : "new_100",

#         # conv_method:  "cdist", "dot_product" "squared_cdist" "cosine" "none"
#         "conv_method" : [["none", "cosine", "cosine", "cosine"],
#                          ["cosine", "cosine", "cosine", "cosine"]],
#         # initial: "kaiming" "uniform"
#         "initial": [["none", "kaiming", "kaiming", "kaiming"],
#                     ["kaiming", "kaiming", "kaiming", "kaiming"]],
#         # rbfs: "triangle" "gauss" 'sigmoid' 'cReLU' 'cReLU_percent' 'regularization'
#         "rbfs": [[["triangle", 'cReLU_percent'], ['cReLU_percent'], ['cReLU_percent'], ['cReLU_percent']],
#                                  [['cReLU_percent'], ['cReLU_percent'], ['cReLU_percent'], ['cReLU_percent']]],
#         "activate_params": [[[1, 0.3], [-1, 0.4], [-1, 0.5], [-1, 0.5]],
#                             [[-1, 0.3], [-1, 0.4], [-1, 0.5], [-1, 0.5]]],
#         "fc_input": (1225) * 1 * 3,  # (rbg last channels 1225 + gray last channels 1225) * last layer shape
#         "device": device

#     }
# }

# arch = {
#     "name": 'RGB_SFMCNN_V3',
#     "need_calculate_status": False,
#     "args": {
#         "in_channels": 3,
#         "out_channels": 30,
#         "mode" : "gray", #  'rgb', 'gray', or 'both'
#         "Conv2d_kernel": [[ (1, 1), (5, 5), (1, 1), (1, 1)],
#                           [ (5, 5),  (1, 1), (1, 1)]],
#         # SFM_methods: "alpha_mean" "max" "none"
#         "SFM_methods": [["alpha_mean", "alpha_mean", "alpha_mean", "alpha_mean"],
#                         ["alpha_mean", "alpha_mean", "alpha_mean", "alpha_mean"]],
#         "SFM_filters": [[  (1, 1), (2, 2),  (1, 3),  (1, 1)],
#                         [ (2, 2),  (1, 3),  (1, 1)]],
#         # 對應到畫圖時，該 channel 的形狀
#         "channels": [[(10, 10), (15, 15), (25, 25),  (35, 35)],
#                      [(7, 10), (15, 15), (35, 35)]],
#         "strides": [[1, 4, 1, 1],
#                     [4, 1, 1]],
#         "paddings": [[0, 0, 0, 0],
#                      [0, 0, 0]],
#         # color_filter : "new_10" "new_30" "new_100"  "old_30"
#         "color_filter" : "new_100",

#         # conv_method:  "cdist", "dot_product" "squared_cdist" "cosine" "none"
#         "conv_method" : [["none", "cosine", "cosine", "cosine"],
#                          ["cosine", "cosine", "cosine", "cosine"]],
#         # initial: "kaiming" "uniform"
#         "initial": [["none", "kaiming", "kaiming", "kaiming"],
#                     ["kaiming", "kaiming", "kaiming", "kaiming"]],
#         # rbfs: "triangle" "gauss" 'sigmoid' 'cReLU' 'cReLU_percent' 'regularization'
#         "rbfs": [[["triangle", 'cReLU_percent'], ['cReLU_percent'], ['cReLU_percent'], ['cReLU_percent']],
#                                  [['cReLU_percent'], ['cReLU_percent'], ['cReLU_percent'], ['cReLU_percent']]],
#         "activate_params": [[[1, 0.3], [-1, 0.4], [-1, 0.5], [-1, 0.5]],
#                             [[-1, 0.3], [-1, 0.4], [-1, 0.5], [-1, 0.5]]],
#         "fc_input": (1225) * 1 * 3,  # (rbg last channels 1225 + gray last channels 1225) * last layer shape
#         "device": device

#     }
# }

# arch = {
#     "name": 'ResNet',
#     "need_calculate_status" : False,
#     "args":{
#         'layers':18,
#         'in_channels':3,
#         "out_channels": 8
#     }
# }

# arch = {
#     "name": 'AlexNet',
#     "need_calculate_status" : False,
#     "args":{
#         'in_channels':3,
#         "out_channels": 8,
#         "input_size": (28, 28)
#     }
# }
#

# arch = {
#     "name": 'DenseNet',
#     "need_calculate_status" : False,
#     "args":{
#         'in_channels':3,
#         "out_channels": 8
#     }
# }

# arch = {
#     "name": 'GoogLeNet',
# "need_calculate_status" : False,
#     "args":{
#         'in_channels':3,
#         "out_channels": 5
#     }
# }

# arch = {
#     "name": 'Swin_tiny',
#     "need_calculate_status": False,
#     "args": {
#         "in_channels": 3,
#         "out_channels": 8,
#         # "drop_rate": 0.0,
#         # "drop_path_rate": 0.05,
#     }
# }

# arch = {
#     "name": 'VIT',
#     "need_calculate_status": False,
#     "args": {
#         "in_channels": 3,
#         "num_classes": 101,
#         "model_name": "vit_tiny_patch16_224",
#         "pretrained": False,
#         "img_size": 224,
#         "drop_rate": 0.1,
#         "drop_path_rate": 0.3,
#         # 若 dataloader 輸出的大小不是 img_size，可設 True 讓模型內自動 resize。
#         "auto_resize": False,
#     }
# }

# arch = {
#     "name": 'PVTv2',
#     "need_calculate_status": False,
#     "args": {
#         "in_channels": 3,
#         "num_classes": 101,
#         "model_name": "pvt_v2_b0",  # PVTv2 系列最小參數版本
#         "pretrained": False,
#         "drop_rate": 0.1,
#         "drop_path_rate": 0.3,
#         # 原版 pvt_v2_b0 無法直接吃 28x28；小於 32 時會自動放大到 32x32。
#         "auto_resize": False,
#         "min_input_size": 32,
#     }
# }

arch = {
    "name": 'MergingViT',
    "need_calculate_status": False,
    "args": {
        "img_size": 224,
        "patch_size": 8,
        "in_chans": 3,
        "num_classes": 2,
        # "embed_dims": [64, 128, 256, 512],
        "embed_dims": [32, 64, 128, 256],
        "num_heads": [2, 4, 8, 16],
        "depths": [1, 1, 1, 1],
        # merge_size: 224/16=14x14 → [(2,2),(2,2),(2,2)] → 7×7 → 4×4 → 2×2
        "merge_size": [(2, 2), (2, 2), (2, 2)], # 28x28 -> 28x14 -> 14x14 -> 7x7 
        "drop_rate": 0.1,  # ViT 風格 dropout，緩解過擬合
        "drop_path_rate": 0.3,
    }
}

create_dir = False
if create_dir:
    save_dir = increment_path('runs/train/exp', exist_ok=False)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
else:
    save_dir = 'runs/train/exp'

print(save_dir)

# lr_scheduler = {
#     "name": "ReduceLROnPlateau",
#     "args": {
#         "patience": 10
#     }
# }

lr_scheduler = {
    "name": "CosineAnnealingLR",
    "args": {
        "T_max": 150,    # 對應 epoch 數
        "eta_min": 1e-6
    }
}

optimizer = {
    "name": "AdamW",
    "args": {
        "weight_decay": 0.01,
    }
}

config = {
    "device": device,
    "root": os.path.dirname(__file__),
    "save_dir": save_dir,
    "load_model_name": load_model_name,
    "model": arch,
    "plot_bar": True,
    "plot_CAM" : False,
    # "dataset":
    # 'Colored_MNIST', 'Colored_FashionMNIST', 'MultiColor_Shapes_Database'
    # "PathMNIST", "BloodMNIST", "CIFAR10"
    # "RetinaMNIST_224", "HeartCalcification_Color"
    # "Caltech101", "NonclassicFace"
    "dataset": 'NonclassicFace',
    "input_shape": (224, 224),
    "batch_size": 16,  # 28x28 較小，batch 可調大
    "epoch": 150,
    "early_stop": True,
    "patience": 30,  # How many epochs without progress, early stop
    "lr": 0.0003,
    "lr_scheduler": lr_scheduler,
    "optimizer": optimizer,
    'use_metric_based_loss': False,
    "loss_fn": 'CrossEntropyLoss',  # 'CustomLoss', # 'CustomLoss' "CrossEntropyLoss" 'MetricBaseLoss', 'LabelSmoothingCrossEntropy'
    "training_loss_fn": 'CrossEntropyLoss',
    "use_preprocessed_image": False,
    "confusion_matrix": "multiclass", # "binary" "multiclass"
    # Heart calcification detection
    "heart_calcification": {
        "grid_size": 45,  # Image cutting size
        "need_resize_height": True,  # Whether to resize based on image height
        "resize_height": 900,  # Resize size
        "threshold": 0.5,
        # For calcification point bounding box, determine if it's a calcification point, shrink the bounding box
        "enhance_method": 'none',
        # Data contrast enhancement method 'contrast' 'normalize' 'histogram_equalization' 'scale_and_offset' 'clahe' 'none'
        "contrast_factor": 1.5,  # Contrast factor, default is 1.0 (no change)
        "use_vessel_mask": False,  # Use vessel mask
        "use_min_count" : True, # 讓正負樣本數量平衡
        "augment_positive":  True,
        "augment_multiplier" : 10
    },

    # MergingViT K-means 分析畫圖
    "kmeans_clusters_per_stage": [32, 64, 128, 256], # [4, 8, 16, 32], [32, 64, 128, 256]
    "kmeans_checkpoint_dir": "runs/train/exp331",  # 畫圖時要載入的 model checkpoint 所在資料夾（與 save_dir 分開）
    "kmeans_save_dir": "plots/kmeans/nonclassicface_222222_head24816_head_padding_repr/",  # K-means 分析結果保存目錄
    "kmeans_use_cache": True,
    "kmeans_mode": "head", # Kmeans 模式，"token" 表示觀察所有 head 合併的結果，"head" 表示獨立觀察各個 head
    "kmeans_heads": None, # Kmeans 要觀察的 head index，None 表示觀察所有 head
    "m_inference": 4, # 要抽取的 inference 圖片數量，None 表示抽取所有圖片（使用 kmeans_inference_image_dir 指定特定資料夾時使用）
    "inference_seed": 42,  # 同一個 seed 會抽同一批 dataset index；換 seed 可抽新圖
    # 手動指定 inference 圖片資料夾（None=仍從 dataset train/test 抽圖）
    # 例: "data/kmeans_inference_custom"；支援子資料夾，依檔名排序
    # folder 模式：m_inference=None 取全部；有數字則取前 m 張
    "kmeans_inference_image_dir": None, #"data/kmeans_inference_custom", None 則從 train/test 抽圖
    "save_all_inference_repr": True,  # True 時輸出同一張 inference 圖的所有 patch repr
    "vlm_analysis": {
        # "gradcam_trace": 分析 GradCAM top-k trace/single_rows
        # "vlm_analysis": 分析 inference/vlm_analysis/img*/ 底下手動挑選的 patch trace
        "caption_source": "vlm_analysis",
        "input_dir": "plots/kmeans/caltech101_222222_padding_repr/inference/vlm_analysis",
        "model": "Qwen/Qwen2-VL-7B-Instruct",
        "max_new_tokens": 512,
        "stage_start": 3,
        "stage_end": 0,
    },
}