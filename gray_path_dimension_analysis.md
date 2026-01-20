# Gray 路徑維度變化分析 (RGB_SFMCNN_V3)

## 概述
Gray 路徑使用 `RGB_Only_Conv2d` 處理 RGB 輸入，然後經過 RBF 響應模組和 SFM 空間合併模組。

## 配置參數（以 config.py 註釋配置為例）
- **輸入尺寸**: (28, 28) - Colored_MNIST
- **Conv2d_kernel[1]**: [(5, 5), (1, 1), (1, 1)]
- **channels[1]**: [(7, 10), (15, 15), (35, 35)]
- **strides[1]**: [4, 1, 1]
- **paddings[1]**: [0, 0, 0]
- **SFM_filters[1]**: [(2, 2), (1, 3), (1, 1)]

---

## 維度變化流程

### 輸入層
```
輸入: (B, 3, 28, 28)
```
- B: batch size
- 3: RGB 通道數
- 28×28: Colored_MNIST 圖像尺寸

---

### Layer 0: GrayBlock（第一層）

#### 1. RGB_Only_Conv2d
```python
# 配置
in_channels = 3
out_channels = channels[1][0][0] * channels[1][0][1] = 7 * 10 = 70
kernel_size = (5, 5)
stride = 4
padding = 0
```

**為什麼是 70？**
- 在 `_make_GrayBlock` 中（第374行）：
  ```python
  out_channel = out_channels[0] * out_channels[1]
  ```
- 其中 `out_channels` 參數來自配置中的 `channels[1][0] = (7, 10)`
- 因此：`out_channel = 7 * 10 = 70`
- 這個設計允許用兩個數字 `(7, 10)` 來表示通道數的"形狀"，最終通道數是兩者的乘積

**計算過程**:
- 輸入: `(B, 3, 28, 28)`
- 輸出空間尺寸計算:
  - `H' = floor((H + 2*padding - kernel_h) / stride + 1)`
  - `H' = floor((28 + 0 - 5) / 4 + 1) = floor(23/4 + 1) = floor(5.75 + 1) = 6`
  - `W' = 6` (相同計算)

**輸出**: `(B, 70, 6, 6)`

**說明**:
- `RGB_Only_Conv2d` 使用三個固定 filter: `[255,0,0]`, `[0,255,0]`, `[0,0,255]`
- 計算每個 RGB patch 與三個 filter 的距離（LAB/歐幾里得/卷積）
- 初始輸出是 `(B, 3, H', W')`（三個 filter 的響應）

**通道擴展過程**（`_expand_channels` 方法）:
1. 初始輸出: `(B, 3, H', W')`
2. 計算重複次數: `filters_per_channel = 70 // 3 = 23`
3. 重複每個 filter: `3 * 23 = 69` 通道 → `(B, 69, H', W')`
4. 補齊到目標通道數: `70 - 69 = 1`，重複最後一個通道 → `(B, 70, H', W')`

#### 2. RBF 層（響應模組）
```python
# 配置: cReLU_percent
```

**計算過程**:
- 輸入: `(B, 70, 6, 6)`
- RBF 層不改變空間維度，只對特徵值進行激活/過濾

**輸出**: `(B, 70, 6, 6)`

#### 3. SFM 層（空間合併模組）
```python
# 配置
SFM_filter = (2, 2)
SFM_method = "alpha_mean"
```

**計算過程**:
- 輸入: `(B, 70, 6, 6)`
- 輸出空間尺寸計算:
  - `H'' = floor((H - (filter_h - 1) - 1) / filter_w + 1)`
  - `H'' = floor((6 - (2-1) - 1) / 2 + 1) = floor(4/2 + 1) = 3`
  - `W'' = 3` (相同計算)

**輸出**: `(B, 70, 3, 3)`

---

### Layer 1: BasicBlock（第二層）

#### 1. RBF_Conv2d
```python
# 配置
in_channels = 7 * 10 = 70
out_channels = 15 * 15 = 225
kernel_size = (1, 1)
stride = 1
padding = 0
```

**計算過程**:
- 輸入: `(B, 70, 3, 3)`
- 輸出空間尺寸:
  - `H' = floor((3 + 0 - 1) / 1 + 1) = 3`
  - `W' = 3`

**輸出**: `(B, 225, 3, 3)`

#### 2. RBF 層
**輸出**: `(B, 225, 3, 3)` (空間維度不變)

#### 3. SFM 層
```python
# 配置
SFM_filter = (1, 3)
```

**計算過程**:
- 輸入: `(B, 225, 3, 3)`
- 輸出空間尺寸:
  - `H'' = floor((3 - (1-1) - 1) / 1 + 1) = floor(2/1 + 1) = 3`
  - `W'' = floor((3 - (3-1) - 1) / 3 + 1) = floor(0/3 + 1) = 1`

**輸出**: `(B, 225, 3, 1)`

---

### Layer 2: BasicBlock（第三層）

#### 1. RBF_Conv2d
```python
# 配置
in_channels = 15 * 15 = 225
out_channels = 35 * 35 = 1225
kernel_size = (1, 1)
stride = 1
padding = 0
```

**計算過程**:
- 輸入: `(B, 225, 3, 1)`
- 輸出空間尺寸:
  - `H' = floor((3 + 0 - 1) / 1 + 1) = 3`
  - `W' = floor((1 + 0 - 1) / 1 + 1) = 1`

**輸出**: `(B, 1225, 3, 1)`

#### 2. RBF 層
**輸出**: `(B, 1225, 3, 1)` (空間維度不變)

#### 3. SFM 層
```python
# 配置
SFM_filter = (1, 1)
```

**計算過程**:
- 輸入: `(B, 1225, 3, 1)`
- 輸出空間尺寸:
  - `H'' = floor((3 - (1-1) - 1) / 1 + 1) = 3`
  - `W'' = floor((1 - (1-1) - 1) / 1 + 1) = 1`

**輸出**: `(B, 1225, 3, 1)`

---

### 最終輸出

#### Reshape
```python
gray_output = gray_output.reshape(x.shape[0], -1)
```

**計算過程**:
- 輸入: `(B, 1225, 3, 1)`
- Flatten: `1225 * 3 * 1 = 3675`

**輸出**: `(B, 3675)`

---

## 維度變化總結表

| 層 | 模組 | 輸入形狀 | 輸出形狀 | 說明 |
|---|---|---|---|---|
| **輸入** | - | `(B, 3, 28, 28)` | - | RGB 輸入 |
| **Layer 0** | RGB_Only_Conv2d | `(B, 3, 28, 28)` | `(B, 70, 6, 6)` | kernel=(5,5), stride=4 |
| | RBF (cReLU_percent) | `(B, 70, 6, 6)` | `(B, 70, 6, 6)` | 空間維度不變 |
| | SFM (2,2) | `(B, 70, 6, 6)` | `(B, 70, 3, 3)` | 空間下採樣 |
| **Layer 1** | RBF_Conv2d | `(B, 70, 3, 3)` | `(B, 225, 3, 3)` | kernel=(1,1), stride=1 |
| | RBF (cReLU_percent) | `(B, 225, 3, 3)` | `(B, 225, 3, 3)` | 空間維度不變 |
| | SFM (1,3) | `(B, 225, 3, 3)` | `(B, 225, 3, 1)` | 寬度下採樣 |
| **Layer 2** | RBF_Conv2d | `(B, 225, 3, 1)` | `(B, 1225, 3, 1)` | kernel=(1,1), stride=1 |
| | RBF (cReLU_percent) | `(B, 1225, 3, 1)` | `(B, 1225, 3, 1)` | 空間維度不變 |
| | SFM (1,1) | `(B, 1225, 3, 1)` | `(B, 1225, 3, 1)` | 空間維度不變 |
| **輸出** | Reshape | `(B, 1225, 3, 1)` | `(B, 3675)` | Flatten |

---

## 關鍵公式

### 卷積輸出尺寸
```
H_out = floor((H_in + 2*padding - kernel_h) / stride + 1)
W_out = floor((W_in + 2*padding - kernel_w) / stride + 1)
```

### SFM 輸出尺寸
```
H_out = floor((H_in - (filter_h - 1) - 1) / filter_h + 1)
W_out = floor((W_in - (filter_w - 1) - 1) / filter_w + 1)

簡化後：
H_out = floor((H_in - filter_h) / filter_h + 1)
W_out = floor((W_in - filter_w) / filter_w + 1)
```

### 通道數計算
```
out_channels = channels[i][0] * channels[i][1]
```

---

## 注意事項

1. **RGB_Only_Conv2d** 使用三個固定 filter (R, G, B)，然後通過 `_expand_channels` 擴展到目標通道數
2. **RBF 層**（如 cReLU_percent）不改變空間維度，只對特徵值進行激活/過濾
3. **SFM 層**會根據 filter 大小進行空間下採樣
4. 最終輸出會 reshape 成 `(B, feature_dim)` 用於全連接層

---

## 驗證

根據上述分析，gray 路徑的最終特徵維度為 **3675**，這與動態計算的 `fc_input` 應該一致。

