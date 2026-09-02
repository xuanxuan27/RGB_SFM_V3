# MergingViT：階層式 Patch Merging 與 K-means 特徵分析

本專案說明 **MergingViT**（階層式 Vision Transformer）的訓練方式，以及對應的 **K-means / GradCAM / VLM** 特徵分析流程。

---

## 簡介

**MergingViT** 來自 2026 葉容瑄碩士論文實作：在每個 encoder stage 結束後合併鄰近 patch，觀察模型在不同解析度下學到的特徵。

整體實驗可分成三層：

1. **訓練 MergingViT**：以 `config.py` 指定架構與資料集，執行 `train.py`。
2. **K-means 分析**：對每個空間位置（或每個 attention head）的 token 做分群，取出群聚代表 patch，並用 GradCAM 從最後一層往前回溯重要位置。
3. **VLM 敘述與彙整**：將「輸入 patch + cluster 代表圖」送進視覺語言模型，產出 JSONL / CSV，便於人工觀察與論文整理。

一鍵跑完後兩步可用 `run_experiment_pipeline.py`。

---

## 模型架構（MergingViT）

定義於 [`models/MergingViT.py`](models/MergingViT.py)。

```
影像 → PatchEmbed → Stage 0 (Transformer) → FlexiblePatchMerging
                   → Stage 1 (Transformer) → FlexiblePatchMerging
                   → ...
                   → 最後 Stage (Transformer) → Global Average Pool → 分類頭
```

### 主要元件

| 模組 | 作用 |
|------|------|
| `PatchEmbed` | 用 `Conv2d` 將影像切成 patch，投影到 `embed_dims[0]` |
| `TransformerBlock` | Pre-norm Attention + MLP，含 DropPath |
| `FlexiblePatchMerging` | 把鄰近 `m_h × m_w` 個 patch 接到 channel，再用 Linear 降維到下一 stage 的 `2×dim` |
| `Attention` | 前向時會暫存 `individual_heads_output`、`pre_projection_features`，供 K-means 的 `head` / `token` 模式使用 |

### FlexiblePatchMerging 與 Padding

當目前解析度 `H, W` 無法被 `m_h, m_w` 整除時，會在**右側與下側**自動 zero-pad，再做合併。這會影響後續 stage 的 token 網格，K-means 代表圖也會以 padded 感受野裁切（超出原圖的部分顯示為黑邊）。

`merge_size` 支援：

- 單一數字：`2` → 每一層都用 `(2, 2)`
- 單一 tuple：`(2, 2)` → 每一層都用 `(2, 2)`
- per-stage 列表：`[(2, 2), (2, 2), (2, 2)]`（長度須等於 merge 次數 = stage 數 − 1）

範例（`img_size=224`, `patch_size=8`）：

- 初始 grid：`28 × 28`
- `merge_size=[(2,2),(2,2),(2,2)]`：`28×28 → 14×14 → 7×7 → 4×4`  
  （最後一次合併前 `7` 無法被 `2` 整除，會 pad 成 `8` 再除以 `2`）

---

## 安裝說明

```bash
# 1. 建立 Conda 環境
conda create --name SFM python=3.10
conda activate SFM

# 2. 安裝 PyTorch（請依 GPU 至 PyTorch 官網選對應版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 安裝其他套件
pip install -r requirements.txt
```

K-means 分析額外需要 `scikit-learn`（已列在 `requirements.txt`）。  
VLM caption 另需 Hugging Face `transformers`；若使用 Qwen2-VL / Qwen2.5-VL，還需 `qwen-vl-utils`：

```bash
pip install transformers qwen-vl-utils
# InternVL 量化載入（選用，單卡 16GB 建議 4bit）
pip install bitsandbytes
```

後續 Python 指令建議在 `SFM` 環境執行，例如：

```bash
conda run -n SFM python train.py
```

---

## 資料集準備

請至實驗室 NAS 下載資料集，並放入 `data/`。

MergingViT 實驗常用資料集（於 `config.py` 的 `dataset` 設定）：

| 名稱 | 說明 |
|------|------|
| `BloodMNIST` | 血細胞分類（MedMNIST） |
| `Caltech101` | 物體分類；dataloader 使用 ImageNet 正規化與較強 augmentation |
| `NonclassicFace` | 人臉相關自訂資料集 |
| `Colored_MNIST` / `Colored_Fashion_MNIST` 等 | 其餘資料集見 `dataloader/get_dataloader.py` |

自訂 inference 圖片可放在例如 `data/kmeans_inference_custom/`（支援子資料夾，依檔名排序）。

---

## 設定檔（`config.py`）

所有訓練與分析參數都從根目錄 [`config.py`](config.py) 讀取。請將 `model.name` 設為 `MergingViT`，並填好下方的架構與分析參數。

`train.py` 會把當次的 `config.py` 與模型定義複製到 `runs/train/exp*/`。分析時請把 `kmeans_checkpoint_dir` 指到該資料夾，pipeline 會優先用 **checkpoint 旁邊那份 config** 建模型，避免之後改了 `merge_size` 卻和權重對不上。

### MergingViT 架構參數（`config["model"]["args"]`）

| 參數 | 說明 |
|------|------|
| `img_size` | 輸入邊長，需與 `input_shape` 一致 |
| `patch_size` | 初始 patch 大小 |
| `in_chans` | 輸入通道數 |
| `num_classes` | 分類數，需與資料集一致 |
| `embed_dims` | 各 stage 的 channel 維度，例如 `[32, 64, 128, 256]` |
| `num_heads` | 各 stage 的 head 數，需整除對應 `embed_dims` |
| `depths` | 各 stage 的 Transformer block 數 |
| `merge_size` | 各次 patch merging 的 `(m_h, m_w)` |
| `drop_rate` / `drop_path_rate` | dropout 與 stochastic depth |

### K-means / VLM 分析參數

| 參數 | 說明 |
|------|------|
| `kmeans_checkpoint_dir` | 訓練 run 資料夾，內含 `{model_name}_best.pth` 與 `config.py` |
| `kmeans_save_dir` | 分析結果輸出目錄 |
| `kmeans_clusters_per_stage` | 各 stage 的 cluster 數，例如 `[32, 64, 128, 256]` |
| `kmeans_mode` | `"token"`：對完整 token 分群；`"head"`：對每個 `(position, head)` 獨立分群 |
| `kmeans_heads` | `head` 模式下要觀察的 head index 列表；`None` 表示全部 |
| `kmeans_use_cache` | 是否快取 K-means 結果（相同模型 / 資料集 / cluster 數可跳過重算） |
| `m_inference` | 要推論的圖片張數；`None` 在資料夾模式下表示全部 |
| `inference_seed` | 從 dataset 抽 inference 圖的隨機種子 |
| `kmeans_inference_image_dir` | 手動指定 inference 圖片資料夾；`None` 則從 test split 抽圖 |
| `save_all_inference_repr` | `True` 時額外輸出該圖所有 position 的 repr 列 |
| `vlm_analysis.caption_source` | `"gradcam_trace"` 或 `"vlm_analysis"`（見下方） |
| `vlm_analysis.input_dir` | `caption_source="vlm_analysis"` 時掃描的圖片根目錄 |
| `vlm_analysis.model` | Hugging Face 模型 id |
| `vlm_analysis.max_new_tokens` | VLM 生成長度 |
| `vlm_analysis.stage_start` / `stage_end` | 要 caption 的 stage 範圍（含兩端；可從大到小） |

---

## 模型訓練

於 `config.py` 設好 `model`（`name: MergingViT`）、`dataset`、`input_shape` 等後：

```bash
conda activate SFM

# 一般訓練
python train.py

# K-Fold 訓練
python train_kfold.py
```

訓練完成後，權重與設定會寫入：

```
runs/train/expXXX/
  MergingViT_best.pth
  MergingViT.py
  config.py
```

評估單次 checkpoint 可用 [`eval_checkpoint.py`](eval_checkpoint.py)。

---

## 實驗流程總覽

建議順序：

```
訓練 MergingViT
    ↓
Step 1  K-means 分群 + 代表圖 + GradCAM trace
    ↓  （可選：把感興趣的 patch 複製到 inference/vlm_analysis/imgN/）
Step 2  VLM 對 single-row 圖產生敘述（JSONL）
    ↓
Step 3  彙整成 CSV（可合併人工觀察）
```

一鍵執行（參數全部來自 `config.py`）：

```bash
# 跑完整三步
python run_experiment_pipeline.py

# 只跑 VLM + 彙整（K-means 已跑過）
python run_experiment_pipeline.py --skip-kmeans

# 只彙整 CSV
python run_experiment_pipeline.py --skip-kmeans --skip-caption

# 指定 dataset、caption 來源與 stage 範圍
python run_experiment_pipeline.py \
  --dataset BloodMNIST \
  --caption-source vlm_analysis \
  --caption-stage-start 3 \
  --caption-stage-end 0
```

常用 CLI：

| 參數 | 說明 |
|------|------|
| `--skip-kmeans` | 跳過 Step 1 |
| `--skip-caption` | 跳過 Step 2 |
| `--skip-summary` | 跳過 Step 3 |
| `--dataset` | 覆寫 `config["dataset"]`，同時影響 K-means 資料載入與 VLM prompt |
| `--caption-source` | `gradcam_trace` 或 `vlm_analysis` |
| `--caption-stage-start` / `--caption-stage-end` | 覆寫 VLM 的 stage 範圍 |
| `--output-jsonl` / `--output-csv` | 自訂輸出路徑（預設在 `kmeans_save_dir` 下） |
| `--manual-notes` | 人工觀察 JSONL（欄位：`rel_key`、`manual_note`） |

---

## Step 1：K-means 分析

實作於 [`mergingViT_plot_tool/Kmeans_analysis_padding_repr.py`](mergingViT_plot_tool/Kmeans_analysis_padding_repr.py)。  
一般請透過 `run_experiment_pipeline.py` 呼叫；該腳本會：

1. 從 `kmeans_checkpoint_dir` 載入 `MergingViT_best.pth` 與當時的 `config.py`
2. 用 **train split** 抽取每個 `(stage, 最後一個 block)`、merge 前的特徵
3. 對每個空間位置做 K-means（L2 正規化後的 cosine 空間）
4. 取各群聚中心最近的 `k_nearest=4` 張圖，裁出 **含 padding 的感受野 patch** 當代表圖
5. 對 inference 圖做 cluster 指派，並用 GradCAM 在最後 stage 取 top-k token，依 merge mapping 回溯到較細的 stage

### token 模式 vs head 模式

- **`token`**：對完整 token 向量（所有 head 拼接後）分群，觀察該位置整體特徵。
- **`head`**：對每個 `(position, head)` 獨立分群，距離用該 head 的向量；裁圖仍用該 position 的感受野。可搭配 `kmeans_heads` 只看特定 head。

特徵來源是 Attention 裡暫存的 `individual_heads_output`（head 模式）或 block 輸出 token（token 模式）。

### Padding 代表圖

- 依各 stage 的 merge 倍率計算 token 在原圖上的完整感受野
- 超出影像邊界的部分補黑，對應 `FlexiblePatchMerging` 的 zero-pad
- 讓同一 stage 所有 token 的代表圖尺寸一致，方便並列比較

### Inference 圖片來源

1. **資料夾模式**：設定 `kmeans_inference_image_dir`（例如 `data/kmeans_inference_custom`）  
   - `m_inference=None`：該資料夾全部圖片  
   - 有數字：依檔名排序取前 m 張  
   - Caltech101 會走與 test 相同的 Resize / CenterCrop / ImageNet normalize
2. **Dataset 模式**：未指定資料夾時，預設從 **test split** 依 `inference_seed` 抽 `m_inference` 張

### GradCAM Trace

對每張 inference 圖：

1. 在最後一個 stage 算 GradCAM，取出 top-k 重要 token
2. 依 padded merge mapping 往前回溯子 patch（padding 來源會標示，不再繼續展開）
3. 每個被追蹤到的位置輸出一列「input + 4 張 cluster 代表圖」

輸出目錄見下方「結果資料夾結構」。

---

## Step 2：VLM Caption

實作於 [`vit_analysis_vlm/run_caption_analysis.py`](vit_analysis_vlm/run_caption_analysis.py)。  
Pipeline 會直接呼叫其中的掃描、載入與推理函式。

### Caption 來源

| `caption_source` | 掃描路徑 | 用途 |
|------------------|----------|------|
| `gradcam_trace` | `{kmeans_save_dir}/inference/gradcam_trace/img{N}/single_rows/` | 自動對 GradCAM top-k 回溯結果產生敘述 |
| `vlm_analysis` | `vlm_analysis.input_dir`（預設 `{kmeans_save_dir}/inference/vlm_analysis/img{N}/`） | 對**手動挑選**的 patch 列產生敘述 |

手動挑選時，把 single-row PNG 複製到 `inference/vlm_analysis/img{N}/`，檔名需維持：

```
s{S}_b{B}_pos{P}_top{R}.png
s{S}_b{B}_pos{P}_parent_s{PS}p{PP}.png
```

### 內建 Prompt

未指定 `--prompt-file` 時，會依 dataset 與 stage 選 prompt：

- **BloodMNIST**：Stage 0–2 偏色塊 / 梯度；Stage 3 偏血球形態學（8 類）
- **Caltech101**：依影像清晰度動態分析色彩梯度或物體局部結構
- **其他資料集**：通用繁中描述（顏色、形狀、位置、跨樣本一致性）

### 獨立執行（不經 pipeline）

適合調整模型、掃描模式或先 dry-run：

```bash
# 先確認會掃到哪些圖
python vit_analysis_vlm/run_caption_analysis.py \
  --scan-mode experiment \
  --plots-root plots/kmeans/your_exp/inference/vlm_analysis \
  --dataset BloodMNIST \
  --stage-start 3 --stage-end 0 \
  --dry-run

# 實際跑 caption
python vit_analysis_vlm/run_caption_analysis.py \
  --scan-mode experiment \
  --plots-root plots/kmeans/your_exp/inference/vlm_analysis \
  --output-dir plots/kmeans/your_exp \
  --output-template "{dataset}/captions_img{img}_stage{stage}.jsonl" \
  --dataset BloodMNIST \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --skip-existing

# InternVL（單卡 16GB 建議 4bit）
python vit_analysis_vlm/run_caption_analysis.py \
  --model OpenGVLab/InternVL2-8B \
  --internvl-quant 4bit \
  --scan-mode gradcam-trace \
  --plots-root plots/kmeans/your_exp/inference/gradcam_trace
```

支援的 `--scan-mode`：

| 模式 | 預期檔名 |
|------|----------|
| `repr` | `stage{S}_block{B}/repr_pos{P}_part{K}.png` |
| `inference` | `img{I}_s{S}_b{B}.png` |
| `inference-repr-pos` | `img{I}_s{S}_b{B}_repr_pos{P}_cluster{C}.png` |
| `single-row` | `s{S}_b{B}_pos{P}_top{R}.png` 或 `..._parent_s{PS}p{PP}.png` |
| `gradcam-trace` | `img{I}/single_rows/` 下的 single-row PNG |
| `experiment` | `{dataset}/img{I}/` 下的 single-row PNG |

支援模型：Qwen2.5-VL、Qwen2-VL、InternVL2。首次執行會從 Hugging Face 下載權重。  
`--skip-existing` 會依 JSONL 裡已有的 `rel_key` 跳過，方便中斷後續跑。

---

## Step 3：彙整 CSV

Pipeline 讀取 Step 2 的 JSONL，可選合併 `--manual-notes`，輸出 UTF-8 BOM 的 CSV。

欄位：`stage`、`block`、`pos`、`parent_stage`、`parent_pos`、`top_rank`、`rel_key`、`vlm_caption`、`manual_note`

預設輸出路徑：

- `caption_source=gradcam_trace` → `{kmeans_save_dir}/vlm_captions.jsonl`、`vlm_summary.csv`
- `caption_source=vlm_analysis` → `{kmeans_save_dir}/vlm_analysis_captions.jsonl`、`vlm_analysis_summary.csv`

人工觀察 JSONL 範例：

```json
{"rel_key": "img0/single_rows/s3_b0_pos12_top00", "manual_note": "核邊緣清晰，像 lymphocyte"}
```

---

## 結果資料夾結構

以 `kmeans_save_dir = plots/kmeans/your_exp/` 為例：

```
plots/kmeans/your_exp/
├── kmeans_cache/                          # K-means .pkl 與對應 _meta.json
├── stage{S}_block{B}/                     # （可選）各位置 cluster 代表圖
├── inference/
│   ├── img{i}_original.png                # 原圖
│   ├── img{i}_s{s}_b{b}.png               # 該 stage 的 cluster map
│   ├── img{i}_gradcam_trace_clusters.json
│   ├── gradcam_trace_clusters.jsonl
│   ├── gradcam_trace/img{i}/
│   │   └── single_rows/
│   │       ├── s{S}_b{B}_pos{P}_top{R}.png
│   │       └── s{S}_b{B}_pos{P}_parent_s{PS}p{PP}.png
│   ├── vlm_analysis/img{i}/               # 手動挑選的 single-row（選用）
│   └── all_pos_trace/                     # save_all_inference_repr=True 時
├── vlm_captions.jsonl                     # 或 vlm_analysis_captions.jsonl
└── vlm_summary.csv                        # 或 vlm_analysis_summary.csv
```

`head` 模式時，single-row 檔名會多 `_head{H}`，JSON 則為 `img{i}_gradcam_trace_clusters_head.json`。

Single-row 圖的內容固定為一列：

```
[input patch] [repr1] [repr2] [repr3] [repr4]
```

最左是當前推論圖在該位置的感受野；右側四格是同一 K-means cluster 的代表樣本。

---

## 建議操作順序（實務）

1. 在 `config.py` 設好 MergingViT 架構與 `dataset`，跑 `python train.py`。
2. 將 `kmeans_checkpoint_dir` 改成該次 `runs/train/expXXX`，並設定 `kmeans_save_dir`、`kmeans_mode`、cluster 數、inference 來源。
3. 先跑 K-means（可暫時關掉 VLM 以節省顯存）：
   ```bash
   python run_experiment_pipeline.py --skip-caption --skip-summary
   ```
4. 檢查 `inference/gradcam_trace/img*/single_rows/`；若只要分析少數位置，複製到 `inference/vlm_analysis/img*/`。
5. 設定 `vlm_analysis.caption_source` 後跑 caption：
   ```bash
   python run_experiment_pipeline.py --skip-kmeans
   ```
   已存在的 `rel_key` 會自動跳過。
6. 用產生的 CSV / JSONL 做質性分析；論文圖表可用 `vit_analysis_vlm/generate_latex_export.py` 等輔助腳本。

注意：K-means 與 VLM 都很吃 GPU。若同一張卡要跑 VLM，建議先 `--skip-kmeans`，或把 K-means cache 算完再載入語言模型。

---

## 專案資料夾結構

| 資料夾 / 檔案 | 說明 |
|----------------|------|
| `models/MergingViT.py` | MergingViT 架構 |
| `mergingViT_plot_tool/Kmeans_analysis_padding_repr.py` | K-means、padding 代表圖、GradCAM trace |
| `mergingViT_plot_tool/Kmeans_analysis.py` | 較早版本的 K-means 工具（代表圖不含 padding 對齊） |
| `vit_analysis_vlm/run_caption_analysis.py` | VLM 批次敘述 |
| `run_experiment_pipeline.py` | K-means → VLM → CSV 一鍵流程 |
| `config.py` | 訓練與分析的單一設定來源 |
| `train.py` / `train_kfold.py` | 模型訓練 |
| `eval_checkpoint.py` | checkpoint 評估 |
| `data/` | 資料集 |
| `runs/train/` | 訓練 run（權重、當次 config） |
| `plots/kmeans/` | K-means / VLM 分析輸出 |
| `pth/` | 另存的 `.pth` 權重 |
| `dataloader/` | 資料載入 |
