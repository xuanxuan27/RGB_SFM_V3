import torch

from torch.nn.modules.utils import _pair
from torch.nn.common_types import _size_2_t
from torch.nn import init
from torch import nn
from torch import Tensor
from torch.autograd import Variable

import torchvision.transforms
import torch.nn.functional as F
import torch
import math

from config import config, arch

'''
來自 2026 葉容瑄的碩士論文:
目標是把 gray 路徑改成使用 RGB 三個 filter 再送進高斯及後續模組
並且改成不使用 RGB 模組處理彩色圖片
'''

def get_feature_extraction_layers(model):
    """
    獲取模型中各層特徵圖的提取器（RGB 與灰階分支）

    參數:
        model: 要分析的模型

    回傳:
        (dict, dict): RGB 分支與灰階分支的特徵圖提取器字典
    """
    mode = arch['args']['mode']  # 模式
    use_gray = mode in ['gray', 'both']
    layers = extract_dual_branch_layers(model, branch_names=("RGB_convs", "Gray_convs"), CI_only=False,
                                        use_gray=use_gray)
    rgb_layers = layers.get("RGB_convs", {})
    gray_layers = layers.get("Gray_convs", None)
    return rgb_layers, gray_layers


def get_basic_target_layers(model, use_gray = True):
    """
    獲取用於 CI（Critical Input）分析所需的層（僅提取包含 activation 的部分）

    參數:
        model: 要分析的模型
        use_gray: 使否使用輪廓層

    回傳:
        (dict, dict): RGB 分支與灰階分支的 CI 分析層字典
    """
    layers = extract_dual_branch_layers(model, branch_names=("RGB_convs", "Gray_convs"), CI_only=True,
                                        use_gray=use_gray)
    rgb_layers = layers.get("RGB_convs", {})
    gray_layers = layers.get("Gray_convs", {})
    return rgb_layers, gray_layers


def extract_dual_branch_layers(model, branch_names=("RGB_conv", "Gray_conv"), CI_only=False, use_gray=True):
    """
    根據模型與分支名稱，自動提取 RGB 與 Gray 卷積層（可選用 Gray）。

    Args:
        model: 模型
        branch_names: 分支名稱前綴（預設為 RGB_conv, Gray_conv）
        CI_only: 是否僅提取 activation 層
        use_gray: 是否使用 Gray 分支

    Returns:
        Dict[str, Dict[str, nn.Sequential]]: {branch: {layer_name: layer}}
    """
    layer_map = {}

    rgb_convs = getattr(model, f"{branch_names[0]}", None)
    if rgb_convs:
        layer_map[branch_names[0]] = extract_branch_layers(rgb_convs, branch_names[0], CI_only=CI_only)

    if use_gray:
        gray_convs = getattr(model, f"{branch_names[1]}", None)
        if gray_convs:
            layer_map[branch_names[1]] = extract_branch_layers(gray_convs, branch_names[1], CI_only=CI_only)

    return layer_map


def extract_branch_layers(conv_blocks, branch_name, CI_only=False):
    """
    建立特定卷積分支（RGB 或 Gray）中各層的特徵提取模組。

    參數:
        conv_blocks: 該分支的 Sequential 卷積區塊（例如 model.RGB_convs）
        branch_name: 層的命名前綴，例如 "RGB_convs"
        CI_only: 若為 True，僅提取經過 activation 的層（供 CI 使用）

    回傳:
        dict: 層名對應的特徵提取模組
    """
    layer_extractors = {}

    for i in range(len(conv_blocks)):
        layer_prefix = f"{branch_name}_{i}"  # ex: RGB_convs_0

        # 取得前 i 層 + 第 i 層的前兩個模組（通常是 conv + activation）
        layer_extractors[f"{layer_prefix}"] = nn.Sequential(
            *(list(conv_blocks[:i])) + list([conv_blocks[i][:2]])
        )

        # 若為完整特徵提取用途，則加入更多細部節點
        if not CI_only:
            # 卷積層輸出（不含 activation）
            layer_extractors[f"{layer_prefix}_after_Conv"] = nn.Sequential(
                *(list(conv_blocks[:i])) + list([conv_blocks[i][:1]])
            )

            # activation 輸出（通常是 conv + activation）
            layer_extractors[f"{layer_prefix}_activation"] = nn.Sequential(
                *(list(conv_blocks[:i])) + list([conv_blocks[i][:2]])
            )

            # 若存在 SFM（空間特徵融合）則包含該層
            if len(conv_blocks[i]) >= 3:
                layer_extractors[f"{layer_prefix}_SFM"] = nn.Sequential(
                    *(list(conv_blocks[:i])) + list([conv_blocks[i][:3]])
                )

    return layer_extractors


'''
    主模型Part
'''
class RGB_SFMCNN_V3(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 Conv2d_kernel,
                 channels,
                 SFM_filters,
                 strides,
                 conv_method,
                 initial,
                 rbfs,
                 SFM_methods,
                 paddings,
                 fc_input,
                 device,
                 activate_params,
                 color_filter: str = "old",
                 mode='both') -> None:
        super().__init__()

        self.mode = mode.lower()
        assert self.mode in ['rgb', 'gray', 'both'], "mode must be 'rgb', 'gray', or 'both'"

        if self.mode in ['gray', 'both']:
            self.gray_transform = torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(),
                NormalizeToRange()
            ])

        # ========== RGB 分支 ==========
        if self.mode in ['rgb', 'both']:
            RGB_conv2d = self._make_RGBBlock(
                in_channels,
                channels[0][0],
                Conv2d_kernel[0][0],
                stride=strides[0][0],
                padding=paddings[0][0],
                rbfs=rbfs[0][0],
                initial='uniform',
                SFM_method=SFM_methods[0][0],
                SFM_filters=SFM_filters[0][0],
                color_filter=color_filter,
                device=device,
                activate_param=activate_params[0][0]
            )

            rgb_basicBlocks = []
            for i in range(1, len(Conv2d_kernel[0])):
                basicBlock = self._make_BasicBlock(
                    channels[0][i - 1],
                    channels[0][i],
                    Conv2d_kernel[0][i],
                    stride=strides[0][i],
                    padding=paddings[0][i],
                    filter=SFM_filters[0][i],
                    rbfs=rbfs[0][i],
                    SFM_method=SFM_methods[0][i],
                    initial=initial[0][i],
                    device=device,
                    activate_param=activate_params[0][i],
                    conv_method=conv_method[0][i]
                )
                rgb_basicBlocks.append(basicBlock)

            self.RGB_convs = nn.Sequential(
                RGB_conv2d,
                *rgb_basicBlocks
            )

        # ========== Gray 分支 ==========
        if self.mode in ['gray', 'both']:
            # 使用 RGB_Only_Conv2d 需要 3 個輸入通道（RGB）
            GRAY_conv2d = self._make_GrayBlock(
                3,  # 改為 3，因為 RGB_Only_Conv2d 需要 RGB 輸入
                channels[1][0],
                Conv2d_kernel[1][0],
                stride=strides[1][0],
                padding=paddings[1][0],
                rbfs=rbfs[1][0],
                initial=initial[1][0],
                SFM_method=SFM_methods[1][0],
                SFM_filters=SFM_filters[1][0],
                device=device,
                activate_param=activate_params[1][0],
                conv_method=conv_method[1][0]
            )

            gray_basicBlocks = []
            for i in range(1, len(Conv2d_kernel[1])):
                basicBlock = self._make_BasicBlock(
                    channels[1][i - 1],
                    channels[1][i],
                    Conv2d_kernel[1][i],
                    stride=strides[1][i],
                    padding=paddings[1][i],
                    filter=SFM_filters[1][i],
                    rbfs=rbfs[1][i],
                    SFM_method=SFM_methods[1][i],
                    initial=initial[1][i],
                    device=device,
                    activate_param=activate_params[1][i],
                    conv_method=conv_method[1][i]
                )
                gray_basicBlocks.append(basicBlock)

            self.Gray_convs = nn.Sequential(
                GRAY_conv2d,
                *gray_basicBlocks
            )

        # ========== 動態計算實際特徵大小 ==========
        # 使用 dummy input 計算實際輸出特徵大小
        try:
            # 從 config 獲取輸入尺寸，如果沒有則使用預設值 (224, 224)
            try:
                input_shape = config.get('input_shape', (224, 224))
                if isinstance(input_shape, (list, tuple)) and len(input_shape) == 2:
                    dummy_h, dummy_w = input_shape
                else:
                    dummy_h, dummy_w = 224, 224
            except:
                dummy_h, dummy_w = 224, 224
            
            with torch.no_grad():
                # 處理 device 參數（可能是字符串或 torch.device）
                if isinstance(device, str):
                    dummy_device = torch.device(device)
                else:
                    dummy_device = device
                dummy_input = torch.zeros(1, in_channels, dummy_h, dummy_w, device=dummy_device)
                features_list = []
                
                if self.mode in ['rgb', 'both']:
                    rgb_out = self.RGB_convs(dummy_input)
                    rgb_out = rgb_out.reshape(1, -1)
                    features_list.append(rgb_out)
                
                if self.mode in ['gray', 'both']:
                    # 使用 RGB_Only_Conv2d，直接使用 RGB 輸入，不需要 gray_transform
                    gray_out = self.Gray_convs(dummy_input)  # 直接使用 RGB 輸入
                    gray_out = gray_out.reshape(1, -1)
                    features_list.append(gray_out)
                
                if features_list:
                    concat_features = torch.cat(features_list, dim=-1)
                    actual_fc_input = concat_features.shape[1]
                else:
                    actual_fc_input = fc_input
            
            # 如果計算出的值與配置值不同，使用實際值並給出警告
            if actual_fc_input != fc_input:
                print(f"[警告] fc_input 計算不一致：配置值={fc_input}, 實際值={actual_fc_input}，將使用實際值")
                fc_input = actual_fc_input
        except Exception as e:
            print(f"[警告] 無法動態計算 fc_input，使用配置值 {fc_input}: {e}")

        # ========== 全連接層 ==========
        self.fc1 = nn.Sequential(
            nn.Linear(fc_input, out_channels)
        )

    def forward(self, x):
        features = []

        if self.mode in ['rgb', 'both']:
            rgb_output = self.RGB_convs(x)
            rgb_output = rgb_output.reshape(x.shape[0], -1)
            features.append(rgb_output)
            # print(f"rgb_output{rgb_output.shape}")

        if self.mode in ['gray', 'both']:
            # 使用 RGB_Only_Conv2d，直接使用 RGB 輸入，不需要 gray_transform
            gray_output = self.Gray_convs(x)  # 直接使用 RGB 輸入
            gray_output = gray_output.reshape(x.shape[0], -1)
            # print(f"gray_output{gray_output.shape}")
            features.append(gray_output)

        concat_output = torch.cat(features, dim=-1)
        # print(f"concat_output{concat_output.shape}")
        # print(f"self.fc1{self.fc1}")
        output = self.fc1(concat_output)
        return output

    '''
        第一層彩色卷積Block(不含空間合併)
    '''

    def _make_RGBBlock(self,
                       in_channel: int,
                       out_channels: tuple,
                       kernel_size: tuple,
                       stride: int = 1,
                       padding: int = 0,
                       initial: str = "kaiming",
                       rbfs=['gauss', 'cReLU_percent'],
                       activate_param=[0, 0],
                       color_filter: str = "old",
                       SFM_method: str = "alpha_mean",
                       SFM_filters: tuple = (1, 1),
                       device: str = "cuda"):
        # 基礎層
        layers = []

        # rgb 卷基層
        out_channel = out_channels[0] * out_channels[1]
        rgb_conv_layer = RGB_Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                             initial=initial, device=device, color_filter=color_filter)
        # 建立響應模組
        rbf_layer = make_rbfs(rbfs, activate_param, device, required_grad=True)
        # 建立空間合併模組
        sfm_layer = SFM(filter=SFM_filters, device=device, method=SFM_method)

        layers.append(rgb_conv_layer)
        layers.append(rbf_layer)
        layers.append(sfm_layer)

        return nn.Sequential(*layers)




    '''
        第一層灰階卷積Block(不含空間合併)
    '''

    def _make_GrayBlock(self,
                        in_channel : int,
                        out_channels: tuple,
                        kernel_size,
                        stride: int = 1,
                        padding: int = 0,
                        conv_method: str = "cdist",
                        initial: str = "kaiming",
                        rbfs=['gauss', 'cReLU_percent'],
                        activate_param=[0, 0],
                        SFM_method: str = "alpha_mean",
                        SFM_filters: tuple = (1, 1),
                        device: str = "cuda"):
        # 基礎層
        layers = []
        # gray 卷基層
        out_channel = out_channels[0] * out_channels[1]
        # 將 conv_method 映射到 method 參數
        # conv_method 可能是 "cdist", "dot_product", "squared_cdist", "cosine"
        # RGB_Only_Conv2d 的 method 是 "lab_distance", "convolution", "euclidean"
        method_mapping = {
            "cdist": "euclidean",  # cdist 類似歐幾里得距離
            "dot_product": "convolution",  # dot_product 就是標準卷積
            "squared_cdist": "euclidean",  # squared_cdist 也類似歐幾里得
            "cosine": "euclidean"  # cosine 暫時映射到 euclidean
        }
        method = method_mapping.get(conv_method, "convolution")  # 預設使用 convolution
        
        rgb_conv_layer = RGB_Only_Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                            initial=initial, device=device)
        # 建立響應模組
        rbf_layer = make_rbfs(rbfs, activate_param, device, required_grad=True)
        # 建立空間合併模組
        sfm_layer = SFM(filter=SFM_filters, device=device, method=SFM_method)

        layers.append(rgb_conv_layer)
        layers.append(rbf_layer)
        layers.append(sfm_layer)

        return nn.Sequential(*layers)

    def _make_BasicBlock(self,
                         in_channels: tuple,
                         out_channels: tuple,
                         kernel_size: tuple,
                         stride: int = 1,
                         padding: int = 0,
                         filter: tuple = (1, 1),
                         rbfs=['gauss', 'cReLU_percent'],
                         SFM_method: str ="alpha_mean",
                         initial: str = "kaiming",
                         device: str = "cuda",
                         activate_param=[0, 0],
                         conv_method: str = "cdist"):
        # 基礎層
        layers = []
        # rbf 卷基層
        in_channel = in_channels[0] * in_channels[1]
        out_channel = out_channels[0] * out_channels[1]
        rgf_conv_layer = RBF_Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                           initial=initial,  conv_method=conv_method, device=device)
        # 建立響應模組
        rbf_layer = make_rbfs(rbfs, activate_param, device, required_grad=True)

        layers.append(rgf_conv_layer)
        layers.append(rbf_layer)

        # 建立空間合併模組
        if SFM_method != "none":
            sfm_layer = SFM(filter=filter, device=device, method=SFM_method)
            layers.append(sfm_layer)

        return nn.Sequential(*layers)




'''
    RGB 卷積層
    return output shape = (batches, channels, height, width)
'''

import torch
import torch.nn as nn

class RGB_Conv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple,
                 stride: int = 1,
                 padding: int = 0,
                 initial: str = "kaiming",
                 color_filter: str = "new_30",
                 requires_grad: bool = False,
                 device=None,
                 dtype=None) -> None:
        super().__init__()


        if color_filter == "old_30":
            weights = [[185, 31, 87],
                       [208, 47, 72],
                       [221, 68, 59],
                       [233, 91, 35],
                       [230, 120, 0],
                       [244, 157, 0],
                       [241, 181, 0],
                       [238, 201, 0],
                       [210, 193, 0],
                       [168, 187, 0],
                       [88, 169, 29],
                       [0, 161, 90],
                       [0, 146, 110],
                       [0, 133, 127],
                       [0, 116, 136],
                       [0, 112, 155],
                       [0, 96, 156],
                       [0, 91, 165],
                       [26, 84, 165],
                       [83, 74, 160],
                       [112, 63, 150], [129, 55, 138],
                       [143, 46, 124], [173, 46, 108], [255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128], [0, 0, 0], [255, 255, 255]]
        elif color_filter == "new_30":
            weights = [[255, 255, 255], [219, 178, 187], [210, 144, 98], [230, 79, 56], [207, 62, 108], [130, 44, 28],
                       [91, 31, 58], [209, 215, 63], [194, 202, 119], [224, 148, 36], [105, 147, 29], [131, 104, 50],
                       [115, 233, 72], [189, 211, 189], [109, 215, 133], [72, 131, 77], [69, 81, 65], [77, 212, 193],
                       [101, 159, 190], [120, 142, 215], [121, 102, 215], [111, 42, 240], [75, 42, 185], [57, 41, 119],
                       [42, 46, 71], [216, 129, 199], [214, 67, 205], [147, 107, 128], [136, 48, 133], [0, 0, 0]]
        elif color_filter == "new_100":
            weights = [[255, 254, 255], [224, 171, 174], [208, 148, 123], [226, 122, 151], [236, 120, 28], [227, 116, 66], [224, 116, 91], [240, 97, 150], [182, 111, 146], [236, 60, 40], [199, 95, 95], [236, 50, 79], [140, 108, 101], [204, 35, 113], [199, 38, 71], [161, 74, 25], [153, 53, 87], [122, 71, 54], [149, 34, 24], [89, 63, 60], [96, 51, 9], [124, 10, 40], [79, 43, 31], [84, 11, 24], [236, 237, 213], [241, 227, 14], [154, 247, 30], [209, 219, 136], [217, 217, 48], [189, 220, 77], [239, 197, 153], [232, 185, 82], [238, 169, 27], [175, 162, 102], [181, 161, 75], [174, 165, 43], [211, 141, 61], [120, 154, 13], [121, 141, 53], [155, 113, 70], [145, 115, 25], [81, 86, 39], [36, 38, 28], [160, 234, 179], [76, 243, 84], [167, 230, 121], [105, 224, 101], [136, 218, 58], [82, 222, 137], [169, 187, 158], [48, 186, 56], [117, 177, 100], [114, 146, 110], [67, 150, 44], [90, 109, 94], [54, 113, 42], [33, 76, 45], [172, 234, 229], [35, 251, 196], [73, 214, 179], [98, 196, 202], [77, 167, 207], [50, 148, 117], [66, 114, 139], [47, 115, 116], [35, 55, 67], [155, 183, 231], [181, 162, 225], [166, 166, 176], [100, 155, 239], [50, 132, 239], [153, 106, 208], [124, 120, 186], [74, 93, 212], [98, 102, 143], [121, 69, 210], [68, 51, 240], [62, 76, 155], [54, 33, 183], [37, 58, 113], [28, 22, 136], [48, 19, 89], [33, 30, 63], [226, 185, 215], [242, 155, 229], [196, 134, 214], [234, 105, 227], [230, 88, 180], [228, 35, 229], [253, 0, 159], [185, 65, 213], [205, 39, 170], [146, 35, 247], [162, 60, 139], [104, 81, 105], [109, 49, 110], [112, 28, 134], [96, 24, 64], [45, 22, 36], [0, 0, 0]]
        elif color_filter == "new_10":
            weights = [[255, 254, 255], [194, 73, 71], [77, 58, 63], [200, 156, 76], [132, 206, 76], [124, 184, 150], [145, 147, 195], [102, 58, 189], [194, 74, 165], [0, 0, 0]]
        elif color_filter == "new_300":
            weights = [[255, 254, 255], [224, 203, 196], [250, 176, 106], [245, 170, 206], [245, 177, 131], [254, 168, 182], [216, 164, 166], [223, 163, 142], [255, 123, 52], [254, 124, 80], [251, 124, 105], [234, 128, 176], [254, 113, 177], [255, 123, 4], [213, 140, 176], [178, 158, 151], [218, 122, 65], [229, 110, 115], [224, 111, 138], [215, 122, 90], [211, 123, 114], [243, 96, 139], [221, 121, 35], [196, 133, 89], [252, 50, 57], [182, 111, 146], [237, 76, 25], [219, 94, 53], [254, 49, 29], [230, 79, 102], [221, 94, 21], [189, 110, 123], [203, 66, 133], [160, 105, 86], [185, 83, 132], [226, 32, 111], [199, 80, 65], [133, 115, 108], [218, 60, 42], [232, 26, 67], [229, 29, 89], [139, 92, 95], [198, 43, 54], [192, 46, 97], [179, 66, 75], [175, 67, 96], [178, 23, 63], [167, 50, 13], [146, 67, 61], [133, 77, 37], [151, 66, 9], [143, 37, 70], [133, 53, 26], [127, 54, 69], [159, 0, 71], [132, 16, 16], [130, 17, 38], [100, 52, 36], [65, 49, 44], [88, 39, 0], [112, 0, 45], [85, 11, 14], [68, 28, 33], [83, 12, 33], [255, 217, 160], [224, 226, 185], [204, 237, 0], [249, 223, 72], [176, 243, 0], [245, 224, 104], [187, 238, 131], [252, 223, 18], [206, 232, 159], [229, 231, 10], [144, 249, 0], [249, 217, 186], [198, 216, 117], [155, 228, 48], [178, 222, 87], [204, 216, 88], [216, 209, 145], [209, 215, 53], [223, 209, 118], [183, 222, 50], [215, 193, 36], [233, 185, 76], [236, 185, 39], [156, 207, 102], [255, 175, 42], [224, 186, 131], [160, 186, 115], [236, 161, 63], [213, 170, 90], [240, 135, 0], [179, 163, 44], [186, 157, 127], [125, 176, 73], [175, 163, 75], [192, 156, 102], [222, 146, 0], [162, 164, 126], [200, 155, 46], [155, 149, 88], [116, 161, 20], [141, 155, 23], [186, 118, 12], [167, 126, 6], [144, 134, 47], [183, 118, 50], [122, 141, 46], [149, 113, 34], [81, 132, 29], [119, 121, 85], [145, 113, 62], [88, 113, 47], [114, 106, 17], [96, 93, 35], [97, 74, 47], [78, 80, 46], [86, 61, 10], [52, 55, 23], [245, 255, 242], [115, 250, 130], [143, 245, 157], [75, 254, 99], [92, 254, 63], [46, 255, 129], [155, 244, 130], [121, 233, 46], [148, 228, 86], [198, 211, 195], [111, 233, 85], [181, 217, 170], [162, 223, 143], [150, 224, 169], [110, 229, 169], [56, 218, 127], [3, 222, 68], [112, 213, 128], [44, 222, 19], [60, 202, 86], [92, 181, 72], [56, 186, 36], [80, 182, 99], [90, 162, 111], [122, 156, 112], [138, 150, 136], [85, 166, 17], [88, 147, 73], [24, 137, 59], [21, 122, 9], [30, 104, 58], [67, 99, 58], [29, 90, 19], [70, 81, 68], [28, 60, 22], [22, 32, 21], [149, 240, 210], [0, 252, 209], [200, 227, 236], [127, 241, 235], [152, 219, 220], [84, 230, 194], [22, 231, 219], [37, 235, 168], [169, 218, 195], [129, 199, 254], [93, 210, 204], [45, 211, 229], [113, 204, 229], [170, 196, 205], [50, 190, 238], [102, 193, 164], [140, 166, 174], [45, 170, 247], [58, 182, 125], [122, 167, 198], [107, 173, 174], [0, 183, 149], [33, 163, 159], [62, 159, 183], [110, 157, 135], [79, 154, 207], [0, 135, 239], [4, 152, 97], [37, 133, 107], [62, 129, 130], [97, 123, 130], [12, 111, 161], [0, 116, 138], [55, 100, 80], [9, 87, 89], [33, 83, 110], [57, 82, 89], [32, 56, 63], [13, 60, 43], [7, 32, 39], [184, 190, 230], [180, 167, 239], [150, 176, 238], [146, 154, 247], [153, 160, 199], [108, 163, 247], [160, 130, 231], [135, 124, 192], [124, 131, 168], [105, 132, 191], [152, 106, 240], [82, 127, 239], [122, 118, 239], [54, 120, 199], [99, 111, 200], [33, 85, 255], [37, 100, 207], [38, 93, 231], [92, 81, 231], [89, 90, 207], [106, 96, 161], [96, 103, 139], [94, 70, 255], [108, 51, 215], [99, 75, 169], [110, 30, 239], [48, 43, 222], [69, 76, 110], [88, 51, 176], [38, 22, 246], [44, 78, 132], [31, 52, 161], [25, 42, 183], [31, 59, 139], [49, 45, 104], [54, 0, 168], [43, 51, 83], [37, 24, 110], [10, 0, 117], [4, 16, 45], [229, 219, 237], [238, 195, 221], [235, 171, 231], [255, 159, 231], [207, 157, 239], [195, 166, 214], [231, 146, 240], [203, 141, 199], [236, 117, 224], [167, 159, 175], [225, 129, 200], [247, 102, 248], [201, 133, 248], [176, 135, 160], [229, 99, 185], [208, 104, 232], [204, 86, 193], [232, 61, 170], [222, 39, 240], [242, 21, 194], [170, 104, 216], [250, 8, 171], [255, 0, 148], [233, 31, 217], [223, 64, 193], [157, 98, 154], [162, 77, 224], [197, 49, 201], [145, 100, 177], [167, 87, 177], [184, 54, 224], [221, 36, 133], [178, 85, 154], [156, 0, 255], [121, 94, 139], [151, 42, 231], [110, 102, 117], [180, 51, 140], [151, 83, 117], [198, 7, 141], [123, 61, 169], [117, 80, 103], [145, 41, 170], [156, 37, 147], [83, 75, 89], [148, 5, 112], [114, 29, 177], [154, 0, 91], [121, 55, 90], [122, 44, 133], [101, 59, 132], [116, 25, 98], [123, 22, 77], [85, 31, 84], [79, 17, 125], [92, 28, 64], [92, 10, 104], [53, 31, 70], [59, 20, 90], [63, 29, 51], [33, 27, 39], [64, 0, 40], [0, 0, 0]]



        kernel_h, kernel_w = kernel_size

        # 轉成 tensor 並正規化，shape: (30, 3)
        weight_tensor = torch.tensor(weights, dtype=dtype, device=device) / 255.0

        # 變成 shape: (30, 3, 1, 1)
        weight_tensor = weight_tensor.unsqueeze(-1).unsqueeze(-1)

        # 轉成 shape: (30, 1, 1, 3)
        weight_tensor = weight_tensor.permute(0, 2, 3, 1)

        # 擴展成 shape: (30, kernel_h, kernel_w, 3) -> (color_num, kernel_h, kernel_w, channel_num)
        weight_tensor = weight_tensor.repeat(1, kernel_h, kernel_w, 1)

        # 註冊為可訓練參數
        # requires_grad: 是否可訓練。如果否，則為固定色卡，只使用代表色；如果為 True，則可以訓練，色卡會改變。
        self.weight = nn.Parameter(weight_tensor, requires_grad=requires_grad)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.initial = initial
        self.use_average = True

    def forward(self, input_tensor):
        # input_tensor shape: (B, 3, H, W) -> (B, 3, 224, 224)
        unfold = torch.nn.Unfold(kernel_size=self.kernel_size, stride=self.stride)
        # input_unfolded shape: (B, 3*kernel_h*kernel_w, L)
        # L = ((H+padding*2-kernel_h+1)/stride+1)*((W+padding*2-kernel_w+1)/stride+1)
        # L = (224+0*2-1+1)*(224+0*2-1+1) = 224*224 = 50176
        input_unfolded = unfold(input_tensor) # [B, 3, 50176]

        height, width = int(input_unfolded.shape[2] ** 0.5), int(input_unfolded.shape[2] ** 0.5) # 224, 224
        # input_unfolded shape: (B, 3, kernel_h, kernel_w, height, width)
        input_unfolded = input_unfolded.view(-1, 3, self.kernel_size[0], self.kernel_size[1], height, width)
        input_unfolded = input_unfolded.permute(0, 4, 5, 1, 2, 3) # (B, height, width, 3, kernel_h, kernel_w)


        permute_weight = self.weight.permute(0, 3, 1, 2)
        # weights shape: (30, 3, kernel_h, kernel_w)
        distances = self.batched_LAB_distance(
            input_unfolded.unsqueeze(1),
            permute_weight.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        )
        return distances


    '''
        LAB Distance
    '''

    def batched_LAB_distance(self, windows_RGBcolor, weights_RGBcolor):
        # RGB 顏色數據從 [0, 1] 映射到 [0, 255]
        R_1, G_1, B_1 = (windows_RGBcolor[:, :, :, :, 0] * 255, windows_RGBcolor[:, :, :, :, 1] * 255,
                         windows_RGBcolor[:, :, :, :, 2] * 255)
        R_2, G_2, B_2 = weights_RGBcolor[:, :, :, :, 0] * 255, weights_RGBcolor[:, :, :, :, 1] * 255, weights_RGBcolor[:, :, :, :, 2] * 255

        # 計算 rmean 作為矩陣形式
        rmean = (R_1 + R_2) / 2

        # 計算 RGB 分量的差異
        R = R_1 - R_2
        G = G_1 - G_2
        B = B_1 - B_2

        # 計算加權的歐幾里得距離
        distance = torch.sqrt(
            (2 + rmean / 256) * (R ** 2) +
            4 * (G ** 2) +
            (2 + (255 - rmean) / 256) * (B ** 2) +
            1e-8
        ).sum(dim=4).sum(dim=4)

        distance = distance / (self.kernel_size[0] * self.kernel_size[1] * 768)

        return distance

    def color_similarity(self, input_patch, weight_kernel, method="delta_e"):
        """
        計算輸入圖像區塊與權重之間的顏色相似度
        input_patch: (1000, 6, 6, 3, 5, 5) -> 區塊圖像
        weight_kernel: (30, 3, 5, 5) -> 權重過濾器
        method: 選擇的顏色公式, 預設使用 Delta E

        返回: (1000, 30, 6, 6) 相似度張量
        """
        if method == "delta_e":
            # 假設 RGB 是近似 Lab 顏色空間 (不是真的 Lab 轉換)
            delta = (input_patch - weight_kernel) ** 2
            # print(f"delta {delta.shape}")
            # print(f"distance {torch.sqrt(delta.sum(dim=4).sum(dim=4).sum(dim=4))}")

            return torch.sqrt(delta.sum(dim=4).sum(dim=4).sum(dim=4)) / 10 # (1000, 30, 6, 6)

        elif method == "cosine":
            # Cosine 相似度 = dot(A, B) / (||A|| * ||B||)
            input_flat = input_patch.flatten(start_dim=3)  # (1000, 6, 6, 3*5*5)
            weight_flat = weight_kernel.flatten(start_dim=2)  # (30, 3*5*5)

            dot_product = (input_flat * weight_flat.unsqueeze(0).unsqueeze(2).unsqueeze(3)).sum(dim=4)
            norm_input = torch.norm(input_flat, dim=4)
            norm_weight = torch.norm(weight_flat, dim=2, keepdim=True)

            return dot_product / (norm_input * norm_weight)  # (1000, 30, 6, 6)

        else:
            raise ValueError("不支援的方法，請選擇 'delta_e' 或 'cosine'")


    # 使用卷積
    def _dot_product(self, input: Tensor) -> Tensor:
        # 使用卷積層進行計算
        result = F.conv2d(input, self.weight.view(self.out_channels, self.out_channels, *self.kernel_size),
                          stride=self.stride, padding=self.padding)
        return result

    def transform_weights(self):
        return self.weight

    def extra_repr(self) -> str:
        return f"initial = {self.initial}, weight shape = {self.weight.shape}, cal_dist = LAB"

    def rgb_to_hsv(self, RGB):
        r, g, b = RGB
        maxc = max(r, g, b)
        minc = min(r, g, b)
        v = maxc
        if minc == maxc:
            return 0.0, 0.0, v
        s = (maxc - minc) / maxc
        rc = (maxc - r) / (maxc - minc)
        gc = (maxc - g) / (maxc - minc)
        bc = (maxc - b) / (maxc - minc)
        if r == maxc:
            h = bc - gc
        elif g == maxc:
            h = 2.0 + rc - bc
        else:
            h = 4.0 + gc - rc
        h = (h / 6.0) % 1.0
        return h, s, v



'''
    RGB Only Conv2d
    return output shape = (batches, channels, height, width)
'''


class RGB_Only_Conv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple,
                 stride: int = 1,
                 padding: int = 0,
                 initial: str = "kaiming",
                 requires_grad: bool = False,
                 method: str = "lab_distance",
                 device=None,
                 dtype=None) -> None:
        """
        將灰階路徑改成使用 RGB 三個 filter 再送進後續模組
        
        Args:
            in_channels: 輸入通道數（應該是 3，對應 RGB）
            out_channels: 輸出通道數（總輸出通道數，會是 3 的倍數）
            kernel_size: 卷積核大小 (h, w)
            stride: 步長
            padding: 填充
            initial: 初始化方法（此處不使用，因為使用固定的 RGB filter）
            requires_grad: 權重是否可訓練
            method: 計算方式
                - "lab_distance": LAB 顏色距離（類似 RGB_Conv2d，目前預設）
                - "convolution": 標準卷積（類似 Gray_Conv2d 的 dot_product）
                - "euclidean": 歐幾里得距離
            device: 設備
            dtype: 數據類型
        """
        super().__init__()
        
        assert in_channels == 3, "RGB_Only_Conv2d 需要 3 個輸入通道（RGB）"
        
        kernel_h, kernel_w = kernel_size
        
        # 只有三個 filter: [255,0,0], [0,255,0], [0,0,255]
        weights = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        
        # 轉成 tensor 並正規化，shape: (3, 3)
        weight_tensor = torch.tensor(weights, dtype=dtype, device=device) / 255.0
        
        # 變成 shape: (3, 3, 1, 1)
        weight_tensor = weight_tensor.unsqueeze(-1).unsqueeze(-1)
        
        # 轉成 shape: (3, 1, 1, 3)
        weight_tensor = weight_tensor.permute(0, 2, 3, 1)
        
        # 擴展成 shape: (3, kernel_h, kernel_w, 3) -> (filter_num, kernel_h, kernel_w, channel_num)
        weight_tensor = weight_tensor.repeat(1, kernel_h, kernel_w, 1)
        
        # 註冊為可訓練參數
        self.weight = nn.Parameter(weight_tensor, requires_grad=requires_grad)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.initial = initial
        self.method = method
    
    def forward(self, input_tensor):
        """
        前向傳播
        根據 method 參數選擇不同的計算方式
        
        Args:
            input_tensor: (B, 3, H, W) RGB 影像
        
        Returns:
            output: (B, out_channels, H', W') 輸出
        """
        if self.method == "lab_distance":
            return self._forward_lab_distance(input_tensor)
        elif self.method == "convolution":
            return self._forward_convolution(input_tensor)
        elif self.method == "euclidean":
            return self._forward_euclidean(input_tensor)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _forward_lab_distance(self, input_tensor):
        """
        LAB 顏色距離（類似 RGB_Conv2d）
        """
        # input_tensor shape: (B, 3, H, W)
        unfold = torch.nn.Unfold(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        # input_unfolded shape: (B, 3*kernel_h*kernel_w, L)
        input_unfolded = unfold(input_tensor)
        
        height, width = int(input_unfolded.shape[2] ** 0.5), int(input_unfolded.shape[2] ** 0.5)
        # input_unfolded shape: (B, 3, kernel_h, kernel_w, height, width)
        input_unfolded = input_unfolded.view(-1, 3, self.kernel_size[0], self.kernel_size[1], height, width)
        input_unfolded = input_unfolded.permute(0, 4, 5, 1, 2, 3)  # (B, height, width, 3, kernel_h, kernel_w)
        
        # 類似 RGB_Conv2d，對完整的 RGB patch 計算與三個 filter 的 LAB 距離
        permute_weight = self.weight.permute(0, 3, 1, 2)  # (3, 3, kernel_h, kernel_w)
        # weights shape: (3, 3, kernel_h, kernel_w) - 三個 filter，每個都是完整的 RGB
        
        # 計算每個 RGB patch 與三個 filter 的距離
        distances = self.batched_LAB_distance(
            input_unfolded.unsqueeze(1),  # (B, 1, height, width, 3, kernel_h, kernel_w)
            permute_weight.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # (1, 3, 1, 1, 3, kernel_h, kernel_w)
        )
        # distances shape: (B, 3, height, width) - 每個位置對應三個 filter 的距離
        
        return self._expand_channels(distances)
    
    def _forward_convolution(self, input_tensor):
        """
        標準卷積（類似 Gray_Conv2d 的 dot_product）
        """
        # 將 weight 轉換成卷積格式: (3, 3, kernel_h, kernel_w)
        conv_weight = self.weight.permute(0, 3, 1, 2)  # (3, 3, kernel_h, kernel_w)
        
        # 使用標準卷積
        output = F.conv2d(input_tensor, conv_weight, stride=self.stride, padding=self.padding)
        # output shape: (B, 3, H', W')
        
        return self._expand_channels(output)
    
    def _forward_euclidean(self, input_tensor):
        """
        歐幾里得距離
        """
        # input_tensor shape: (B, 3, H, W)
        unfold = torch.nn.Unfold(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        input_unfolded = unfold(input_tensor)  # (B, 3*kernel_h*kernel_w, L)
        
        height, width = int(input_unfolded.shape[2] ** 0.5), int(input_unfolded.shape[2] ** 0.5)
        input_unfolded = input_unfolded.view(-1, 3, self.kernel_size[0], self.kernel_size[1], height, width)
        input_unfolded = input_unfolded.permute(0, 4, 5, 1, 2, 3)  # (B, height, width, 3, kernel_h, kernel_w)
        
        # 將 weight 轉換: (3, kernel_h, kernel_w, 3) -> (3, 3, kernel_h, kernel_w)
        permute_weight = self.weight.permute(0, 3, 1, 2)  # (3, 3, kernel_h, kernel_w)
        
        # 計算歐幾里得距離
        # 將 input 和 weight 都 flatten
        input_flat = input_unfolded.reshape(-1, 3 * self.kernel_size[0] * self.kernel_size[1])  # (B*H*W, 3*kernel_h*kernel_w)
        weight_flat = permute_weight.reshape(3, 3 * self.kernel_size[0] * self.kernel_size[1])  # (3, 3*kernel_h*kernel_w)
        
        # 計算距離: (B*H*W, 3)
        distances = torch.cdist(input_flat, weight_flat)  # (B*H*W, 3)
        
        # Reshape: (B, H, W, 3) -> (B, 3, H, W)
        distances = distances.reshape(input_tensor.shape[0], height, width, 3).permute(0, 3, 1, 2)
        
        return self._expand_channels(distances)
    
    def _expand_channels(self, output):
        """
        擴展輸出通道數以符合 out_channels
        """
        # output shape: (B, 3, H', W')
        # 如果 out_channels > 3，需要擴展輸出通道數
        filters_per_channel = self.out_channels // 3
        if filters_per_channel > 1:
            # 重複每個 filter 的結果
            # output: (B, 3, H', W') -> (B, 3, 1, H', W') -> (B, 3, filters_per_channel, H', W')
            output = output.unsqueeze(2).repeat(1, 1, filters_per_channel, 1, 1)
            # Reshape: (B, 3, filters_per_channel, H', W') -> (B, 3*filters_per_channel, H', W')
            output = output.reshape(output.shape[0], -1, output.shape[3], output.shape[4])
        
        # 如果 out_channels 不是 3 的倍數，補齊或截斷
        current_channels = output.shape[1]
        if current_channels < self.out_channels:
            # 補齊：重複最後的通道
            diff = self.out_channels - current_channels
            output = torch.cat([output, output[:, -diff:, :, :]], dim=1)
        elif current_channels > self.out_channels:
            # 截斷：只保留前 out_channels 個通道
            output = output[:, :self.out_channels, :, :]
        
        return output
    
    def batched_LAB_distance(self, windows_RGBcolor, weights_RGBcolor):
        """
        LAB Distance 計算（類似 RGB_Conv2d）
        """
        # RGB 顏色數據從 [0, 1] 映射到 [0, 255]
        R_1, G_1, B_1 = (windows_RGBcolor[:, :, :, :, 0] * 255, 
                         windows_RGBcolor[:, :, :, :, 1] * 255,
                         windows_RGBcolor[:, :, :, :, 2] * 255)
        R_2, G_2, B_2 = (weights_RGBcolor[:, :, :, :, 0] * 255, 
                         weights_RGBcolor[:, :, :, :, 1] * 255, 
                         weights_RGBcolor[:, :, :, :, 2] * 255)
        
        # 計算 rmean 作為矩陣形式
        rmean = (R_1 + R_2) / 2
        
        # 計算 RGB 分量的差異
        R = R_1 - R_2
        G = G_1 - G_2
        B = B_1 - B_2
        
        # 計算加權的歐幾里得距離
        distance = torch.sqrt(
            (2 + rmean / 256) * (R ** 2) +
            4 * (G ** 2) +
            (2 + (255 - rmean) / 256) * (B ** 2) +
            1e-8
        ).sum(dim=4).sum(dim=4)
        
        distance = distance / (self.kernel_size[0] * self.kernel_size[1] * 768)
        
        return distance
    
    def extra_repr(self) -> str:
        method_str = {
            "lab_distance": "LAB",
            "convolution": "Conv",
            "euclidean": "Euclidean"
        }.get(self.method, self.method)
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, method={method_str}, filters=[R,G,B]"


    # 使用餘弦相似度
    def _cosine(self, input: Tensor) -> Tensor:
        output_width = math.floor(
            (input.shape[-1] + self.padding * 2 - (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        output_height = math.floor(
            (input.shape[-2] + self.padding * 2 - (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        # Unfold output = (batch, output_width * output_height, C×∏(kernel_size))
        windows = F.unfold(input, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding).permute(0, 2, 1)

        # 計算 windows 和 weight 的 L2 範數
        windows_norm = torch.norm(windows, p=2, dim=2, keepdim=True)
        weight_norm = torch.norm(self.weight, p=2, dim=1, keepdim=True)


        # 計算點積
        dot_product = torch.matmul(windows, self.weight.t())


        # 計算餘弦相似度
        cosine = dot_product / (windows_norm * weight_norm.t() + 1e-8)
        # 調整維度順序並重塑
        result = cosine.permute(0, 2, 1)
        result = result.reshape(result.shape[0], result.shape[1], output_height, output_width)
        return result

'''
    Gray 卷積層
    return output shape = (batches, channels, height, width)
'''


class Gray_Conv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 initial: str = "kaiming",
                 conv_method: str = "cdist",
                 device=None,
                 dtype=None):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = padding
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.initial = initial
        self.conv_method = conv_method

        self.weight = torch.empty((out_channels, in_channels * self.kernel_size[0] * self.kernel_size[1]),
                                  **factory_kwargs)
        self.reset_parameters(initial)

    def reset_parameters(self, initial) -> None:
        if initial == "kaiming":
            # kaiming 初始化
            # bound  = sqrt(6/(1 + a^2 * fan))
            # fan = self.weight.size(1) * 1
            init.kaiming_uniform_(self.weight)
        elif initial == "uniform":
            init.uniform_(self.weight)
        else:
            raise "RBF_Conv2d initial error"

        # 将第一个输出通道的权重设置为0
        with torch.no_grad():
            self.weight[0].fill_(-1)
            self.weight[1].fill_(1)

        # self.weight = (self.weight - torch.min(self.weight)) / (torch.max(self.weight) - torch.min(self.weight))

        self.weight = nn.Parameter(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        if self.conv_method == "cdist":
            return self._cdist(input)
        elif self.conv_method == "dot_product":
            return self._dot_product(input)
        elif self.conv_method == "squared_cdist":
            return self._squared_cdist(input)
        elif self.conv_method == "cosine":
            return self._cosine(input)
        else:
            print(f"Can't find {self.conv_method} conv method")

    # 使用距離公式
    def _cdist(self, input: Tensor) -> Tensor:
        output_width = math.floor(
            (input.shape[-1] + self.padding * 2 - (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        output_height = math.floor(
            (input.shape[-2] + self.padding * 2 - (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        # Unfold output = (batch, output_width * output_height, C×∏(kernel_size))
        windows = F.unfold(input, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding).permute(0, 2,
                                                                                                                  1)
        result = torch.cdist(windows, self.weight).permute(0, 2, 1)

        result = result.reshape(result.shape[0], result.shape[1], output_height, output_width)

        return result

    # Weight 取平方的後，距離公式
    def _squared_cdist(self, input: Tensor) -> Tensor:
        output_width = math.floor(
            (input.shape[-1] + self.padding * 2 - (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        output_height = math.floor(
            (input.shape[-2] + self.padding * 2 - (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        # Unfold output = (batch, output_width * output_height, C×∏(kernel_size))
        windows = F.unfold(input, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding).permute(0, 2,
                                                                                                                  1)
        # 將 self.weight 取平方
        squared_weight = self.weight ** 2

        result = torch.cdist(windows, squared_weight).permute(0, 2, 1)

        result = result.reshape(result.shape[0], result.shape[1], output_height, output_width)

        return result

    # 使用餘弦相似度
    def _cosine(self, input: Tensor) -> Tensor:
        output_width = math.floor(
            (input.shape[-1] + self.padding * 2 - (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        output_height = math.floor(
            (input.shape[-2] + self.padding * 2 - (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        # Unfold output = (batch, output_width * output_height, C×∏(kernel_size))
        windows = F.unfold(input, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding).permute(0, 2, 1)

        # 計算 windows 和 weight 的 L2 範數
        windows_norm = torch.norm(windows, p=2, dim=2, keepdim=True)
        weight_norm = torch.norm(self.weight, p=2, dim=1, keepdim=True)


        # 計算點積
        dot_product = torch.matmul(windows, self.weight.t())


        # 計算餘弦相似度
        cosine = dot_product / (windows_norm * weight_norm.t() + 1e-8)
        # 調整維度順序並重塑
        result = cosine.permute(0, 2, 1)
        result = result.reshape(result.shape[0], result.shape[1], output_height, output_width)
        return result

    # 使用卷積
    def _dot_product(self, input: Tensor) -> Tensor:
        # 使用卷積層進行計算
        result = F.conv2d(input, self.weight.view(self.out_channels, self.in_channels, *self.kernel_size),
                          stride=self.stride, padding=self.padding)
        return result

    def extra_repr(self) -> str:
        return f"initial = {self.initial}, weight shape = {(self.out_channels, self.in_channels, *self.kernel_size)}"




'''
    RBF 卷積層
    return output shape = (batches, channels, height, width)
'''


class RBF_Conv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 initial: str = "kaiming",
                 conv_method: str = "cdist",
                 device=None,
                 dtype=None):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = padding
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.initial = initial
        self.conv_method = conv_method

        self.weight = torch.empty((out_channels, in_channels * self.kernel_size[0] * self.kernel_size[1]),
                                  **factory_kwargs)
        self.reset_parameters(initial)

    def reset_parameters(self, initial) -> None:
        if initial == "kaiming":
            # kaiming 初始化
            # bound  = sqrt(6/(1 + a^2 * fan))
            # fan = self.weight.size(1) * 1
            init.kaiming_uniform_(self.weight)
        elif initial == "uniform":
            init.uniform_(self.weight)
        else:
            raise "RBF_Conv2d initial error"

        self.weight = nn.Parameter(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        if self.conv_method == "cdist":
            return self._cdist(input)
        elif self.conv_method == "dot_product":
            return self._dot_product(input)
        elif self.conv_method == "squared_cdist":
            return self._squared_cdist(input)
        elif self.conv_method == "cosine":
            return self._cosine(input)
        else:
            print(f"Can't find {self.conv_method} conv method")


    # 使用距離公式
    def _cdist(self, input: Tensor) -> Tensor:
        output_width = math.floor(
            (input.shape[-1] + self.padding * 2 - (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        output_height = math.floor(
            (input.shape[-2] + self.padding * 2 - (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        # Unfold output = (batch, output_width * output_height, C×∏(kernel_size))
        windows = F.unfold(input, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding).permute(0, 2,
                                                                                                                  1)
        result = torch.cdist(windows, self.weight).permute(0, 2, 1)

        result = result.reshape(result.shape[0], result.shape[1], output_height, output_width)

        return result

    # Weight 取平方的後，距離公式
    def _squared_cdist(self,  input: Tensor) -> Tensor:
        output_width = math.floor(
            (input.shape[-1] + self.padding * 2 - (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        output_height = math.floor(
            (input.shape[-2] + self.padding * 2 - (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        # Unfold output = (batch, output_width * output_height, C×∏(kernel_size))
        windows = F.unfold(input, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding).permute(0, 2,
                                                                                                                  1)
        # 將 self.weight 取平方
        squared_weight = self.weight ** 2

        result = torch.cdist(windows, squared_weight).permute(0, 2, 1)

        result = result.reshape(result.shape[0], result.shape[1], output_height, output_width)

        return result


    # 使用卷積
    def _dot_product(self, input: Tensor) -> Tensor:
        # 使用卷積層進行計算
        result = F.conv2d(input, self.weight.view(self.out_channels, self.in_channels, *self.kernel_size),
                           stride=self.stride, padding=self.padding)
        return result

    # 使用餘弦相似度
    def _cosine(self, input: Tensor) -> Tensor:

        output_width = math.floor(
            (input.shape[-1] + self.padding * 2 - (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        output_height = math.floor(
            (input.shape[-2] + self.padding * 2 - (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        # Unfold output = (batch, output_width * output_height, C×∏(kernel_size))
        windows = F.unfold(input, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding).permute(0, 2,
                                                                                                                  1)

        # 計算 windows 和 weight 的 L2 範數(向量的歐基里德距離)
        windows_norm = torch.norm(windows, p=2, dim=2, keepdim=True)
        weight_norm = torch.norm(self.weight, p=2, dim=1, keepdim=True)

        # 計算點積
        dot_product = torch.matmul(windows, self.weight.t())

        # 計算餘弦相似度
        cosine = dot_product / (windows_norm * weight_norm.t() + 1e-8)

        # 調整維度順序並重塑
        result = cosine.permute(0, 2, 1)
        result = result.reshape(result.shape[0], result.shape[1], output_height, output_width)
        return result


    def extra_repr(self) -> str:
        return f"initial = {self.initial}, weight shape = {(self.out_channels, self.in_channels, *self.kernel_size)}"


'''
    前處理Part(自由取用)
'''


class Renormalize(object):
    def __call__(self, images):
        batch, channel = images.shape[0], images.shape[1]
        min_vals = images.view(batch, channel, -1).min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        max_vals = images.view(batch, channel, -1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        normalized_images = (images - min_vals) / (max_vals - min_vals)
        result = torch.where(max_vals == min_vals, torch.zeros_like(images), normalized_images)
        return result


class NormalizeToRange(object):
    def __call__(self, images):
        """
        将图像归一化到 [-1, 1] 区间

        Args:
            images: 输入图像张量

        Returns:
            归一化后的图像张量，范围在 [-1, 1] 之间
        """
        batch, channel = images.shape[0], images.shape[1]
        min_vals = images.view(batch, channel, -1).min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        max_vals = images.view(batch, channel, -1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)

        # 先归一化到 [0, 1]
        normalized_images = (images - min_vals) / (max_vals - min_vals + 1e-8)
        # 然后转换到 [-1, 1]
        result = normalized_images * 2 - 1

        # 处理特殊情况：当 max_vals == min_vals 时，返回零张量
        return torch.where(max_vals == min_vals, torch.zeros_like(images), result)


class AvoidAll0(object):
    def __call__(self, images):
        """
        将图像轉到 [-1, 1] 区间，避免全黑圖片都是 0

        Args:
            images: 输入图像张量

        Returns:
            归一化后的图像张量，范围在 [-1, 1] 之间
        """

        # 然后转换到 [-1, 1]
        result = images * 2 - 1

        # 处理特殊情况：当 max_vals == min_vals 时，返回零张量
        return result


'''
    響應過濾模組(輸入陣列)
'''
def make_rbfs(rbfs, activate_param, device, required_grad=True):
    # 建立響應模組
    rbf_layers = []
    for rbf in rbfs:
        print(rbf)
        rbf_layers.append(get_rbf(rbf, activate_param, device, required_grad=required_grad))
    return nn.Sequential(*rbf_layers)


'''
    響應過濾模組(所有可能性)
'''
def get_rbf(rbf, activate_param, device, required_grad = True):
    if rbf == "triangle":
        return triangle(w=activate_param[0], requires_grad=required_grad, device=device)
    elif rbf == "gauss":
        return gauss(std=activate_param[0], device=device)
    elif rbf == 'sigmoid':
        return Sigmoid()
    elif rbf == 'cReLU':
        return cReLU()
    elif rbf == 'cReLU_percent':
        return cReLU_percent(percent=activate_param[1])
    elif rbf == 'regularization':
        return MeanVarianceRegularization()
    else:
        raise ValueError(f"Unknown RBF type: {rbf}")


class triangle(nn.Module):
    def __init__(self, w: float, requires_grad: bool = True, device: str = "cuda"):
        super().__init__()
        self.w = torch.Tensor([w]).to(device)
        if requires_grad:
            self.w = nn.Parameter(self.w, requires_grad=True)

    def forward(self, d):
        w_tmp = self.w.to(d.device)
        d_clamped = torch.clamp(d, max=w_tmp)
        result = torch.abs(torch.ones_like(d) - torch.div(d_clamped, w_tmp))
        return result


    def extra_repr(self) -> str:
        return f"w = {self.w.item()}"

class gauss(nn.Module):
    def __init__(self, std, requires_grad: bool = True, device: str = "cuda"):
        super().__init__()
        self.std = torch.Tensor([std]).to(device)
        if requires_grad:
            self.std = nn.Parameter(self.std)

    def forward(self, d):
        std = self.std.to(d.device)
        result = torch.exp(torch.pow(d, 2) / (-2 * torch.pow(std, 2)))
        return result


    def extra_repr(self) -> str:
        return f"std={self.std.item()}"


class cReLU(nn.Module):
    def __init__(self,
                 bias: float = 0.7,
                 requires_grad: bool = True,
                 device: str = "cuda") -> None:
        super().__init__()
        self.bias = torch.tensor([bias]).to(device)
        if requires_grad:
            self.bias = nn.Parameter(self.bias, requires_grad=True)

    def forward(self, x):
        bias_tmp = self.bias.to(x.device)
        result = x * torch.ge(x, bias_tmp.repeat(x.shape[0]).view(-1, 1, 1, 1)).float()
        return result


    def extra_repr(self) -> str:
        return f"bias={self.bias.item()}"


class cReLU_percent(nn.Module):
    def __init__(self, percent: float = 0.5, device: str = "cuda") -> None:
        super().__init__()
        self.percent = torch.tensor([percent]).to(device)
        self.device = device

    def forward(self, x):
        percent = self.percent.to(x.device)

        batch_size, count, h, w = x.shape
        x_flatten = x.permute(0, 2, 3, 1).reshape(batch_size, h * w, count)

        k = math.ceil(percent.item() * count)
        top_k, _ = x_flatten.topk(k, dim=2, largest=True)
        threshold = top_k[:, :, -1].unsqueeze(2)

        x_filtered = torch.where(x_flatten >= threshold, x_flatten, torch.tensor(0.0, device=x.device))
        x_filtered = torch.clamp(x_filtered, min=0)

        result = x_filtered.reshape(batch_size, h, w, count).permute(0, 3, 1, 2)

        return result



    def extra_repr(self) -> str:
        return f"percent={self.percent.item()}"


class MeanVarianceRegularization(nn.Module):
    def __init__(self, target_range=(0, 1), epsilon=1e-8):
        super().__init__()
        self.target_min, self.target_max = target_range
        self.epsilon = epsilon

    def forward(self, x):
        # 計算每個 feature map 的最小值和最大值
        min_per_map = x.amin(dim=(0, 2, 3), keepdim=True)  # shape = [1, C, 1, 1] 
        max_per_map = x.amax(dim=(0, 2, 3), keepdim=True)  # shape = [1, C, 1, 1]

        # Min-max 正規化到目標範圍
        x_norm = (x - min_per_map) / (max_per_map - min_per_map + self.epsilon)
        x_final = x_norm * (self.target_max - self.target_min) + self.target_min
        
        return x_final


'''
    時序合併層
    parameters:
        filter: 合併的範圍
'''
class SFM(nn.Module):
    def __init__(self,
                 filter: _size_2_t,
                 alpha_max: float = 0.9,
                 alpha_min: float = 1.0,
                 device: str = "cuda",
                 method: str = "alpha_mean") -> None:
        super(SFM, self).__init__()
        self.filter = filter
        self.alpha = torch.linspace(start=alpha_min, end=alpha_max, steps=math.prod(self.filter),
                                    requires_grad=False).reshape(*self.filter)
        self.device = device
        self.method = method

    def forward(self, input: Tensor) -> Tensor:
        batch_num, channels, height, width = input.shape

        alpha_pows = self.alpha.to(input.device).repeat(input.shape[1], 1, 1)
        _, filter_h, filter_w = alpha_pows.shape

        unfolded_input = input.unfold(2, filter_h, filter_h).unfold(3, filter_w, filter_w)
        unfolded_input = unfolded_input.reshape(batch_num, channels, -1, filter_h * filter_w)

        if self.method == "max":
            output_width = math.floor((width - (filter_w - 1) - 1) / filter_w + 1)
            output_height = math.floor((height - (filter_h - 1) - 1) / filter_h + 1)
            output = unfolded_input.max(dim=-1).values.reshape(batch_num, channels, output_height, output_width)
        elif self.method == "alpha_mean":
            expanded_filter = alpha_pows.reshape(channels, 1, -1)
            expanded_filter = expanded_filter.repeat(batch_num, 1, 1, 1)
            result = unfolded_input * expanded_filter

            output_width = math.floor((width - (filter_w - 1) - 1) / filter_w + 1)
            output_height = math.floor((height - (filter_h - 1) - 1) / filter_h + 1)
            output = result.mean(dim=-1).reshape(batch_num, channels, output_height, output_width)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return output

    def extra_repr(self) -> str:
        return f"filter={self.filter}, method={self.method}, alpha={self.alpha.detach().numpy()}"



class Sigmoid(nn.Module):
    def __init__(self,
                 requires_grad: bool = True,
                 device: str = "cuda") -> None:
        super().__init__()
        self.requires_grad = requires_grad
        self.device = device

    def forward(self, x):
        return torch.sigmoid(x)

    def extra_repr(self) -> str:
        return "Sigmoid activation"


