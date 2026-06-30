#!/usr/bin/env python3
"""
generate_latex_export.py

從 VLM caption JSONL 生成論文用的 LaTeX 片段，並複製對應圖片至匯出目錄。

輸出結構：
  latex_export/
    figures/
      ch4/caltech101/img{k}/        ← 主文用圖（single row）
        s3_b0_pos2_top01.png
        s2_b0_pos4_parent_s3p2.png
        ...
      appendix/caltech101/img{k}/   ← 附錄用圖（完整 trace page）
        top01_pos2_trace_page01.png
        ...
    generated/
      selected_patches_img{k}.tex   ← 主文 LaTeX 片段
      appendix_trace_img{k}.tex     ← 附錄 LaTeX 片段

用法：
  python generate_latex_export.py

設定區塊在下方 CONFIG 裡調整。
"""

import json
import shutil
import textwrap
from pathlib import Path

# ===========================================================================
# CONFIG：在這裡調整路徑與選圖清單
# ===========================================================================

# JSONL 路徑（Linux 端）
JSONL_PATH = Path("/home/xuan/RGB_SFM_V3/plots/kmeans/caltech101_222222/vlm_analysis_captions.jsonl")

# Kmeans VLM analysis 圖片根目錄（底下為 img{idx}/s*_b*_pos*.png）
KMEANS_ROOT = Path("/home/xuan/RGB_SFM_V3/plots/kmeans/caltech101_222222/inference/vlm_analysis")

# 匯出根目錄
EXPORT_ROOT = Path("/home/xuan/RGB_SFM_V3/latex_export")

# 資料集名稱（用於路徑）
DATASET = "caltech101_222222"

# --------------------------------------------------
# 主文選圖清單
# 格式：每個 entry 是一個 dict，包含：
#   img        : int，影像編號
#   rel_key    : str，對應 JSONL 的 rel_key
#   label      : str，LaTeX 圖片的簡短說明（人工填寫）
#   manual_note: str，人工觀察補充（可留空）
# --------------------------------------------------
SELECTED_PATCHES = [
    # img0 飛機：stage 3 非 GradCAM top-k 的候選區域
    {
        "img": 0,
        "rel_key": "img0/s3_b0_pos9_top00",
        "label": "Stage 3 pos 9",
        "manual_note": "代表圖均可辨識飛機機身側面結構，呈現白色弧形機身、等距排列的橢圓形舷窗及下方地面紋理。輸入區塊同樣涵蓋機身本體，並可見引擎與起落架等局部結構，與代表圖的主體特徵高度吻合，推測此群聚對應飛機類別的機身局部區域。",
    },
    # stage 3 pos9 對應的 stage 2 children
    {
        "img": 0,
        "rel_key": "img0/s2_b0_pos30_parent_s3p9",
        "label": "Stage 2 pos 30（stage 3 pos 9 的來源 patch）",
        "manual_note": "代表圖均呈現水平分層結構，上半部可見等距排列的矩形窗格，下方伴隨深色帶狀底部與地面紋理。輸入區塊同樣具有相似的水平紋理與上方窗格排列，與代表圖高度吻合。推測此群聚對應火車車廂側面、建築物、飛機機身側面等局部結構。",
    },
    {
        "img": 0,
        "rel_key": "img0/s2_b0_pos31_parent_s3p9",
        "label": "Stage 2 pos 31（stage 3 pos 9 的來源 patch）",
        "manual_note": "代表圖均呈現水平分層的色彩結構，上下區域明度與色調存在明顯差異，為群聚的主要共同特徵。然而各代表圖之間色調與局部形狀差異較大。輸入區塊同樣具有水平分層特徵，與代表圖的梯度方向一致，推測此群聚對應場景中具有水平邊界結構的區域，但語意指向較為模糊。",
    },
    {
        "img": 0,
        "rel_key": "img0/s2_b0_pos37_parent_s3p9",
        "label": "Stage 2 pos 37（stage 3 pos 9 的來源 patch）",
        "manual_note": "代表圖以大面積色塊、水平分層為主，根據顏色可推測為土地、草皮等自然物體。輸入圖片則以小面積色塊、垂直分層為主，可觀察到帶有圓弧狀的物體，背景為咖啡色土地。推測此群聚對應自然景觀、戶外環境、草皮、土地等語意指向。",
    },
    {
        "img": 0,
        "rel_key": "img0/s2_b0_pos38_parent_s3p9",
        "label": "Stage 2 pos 38（stage 3 pos 9 的來源 patch）",
        "manual_note": "部分代表圖依稀可見圓形輪廓，且主要物體集中於畫面中央，具有一定的結構共性。輸入區塊色調與局部形狀與代表圖關聯性較低，契合度有限。推測此群聚對應具有圓形結構的物體局部，如車輪、機械設備或建築構件等。",
    },
    # stage 2 pos 9 對應的 stage 1 children
    {
        "img": 0,
        "rel_key": "img0/s1_b0_pos116_parent_s2p30",
        "label": "Stage 1 pos 116（stage 2 pos 30 的來源 patch）",
        "manual_note": "代表圖色調以灰白色為主，與輸入圖片不一致，但 repr 1、3、4  可觀察到水平與斜向分層的特性與輸入圖片接近。",
    },
    {
        "img": 0,
        "rel_key": "img0/s1_b0_pos117_parent_s2p30",
        "label": "Stage 1 pos 117（stage 2 pos 30 的來源 patch）",
        "manual_note": "該群聚代表圖色調差異顯著，但均呈現上下較深、中間明度較高的共同分布特徵。輸入區塊為灰綠調、上淺下深之漸層，與代表圖的明度分布方向不一致。",
    },
    {
        "img": 0,
        "rel_key": "img0/s1_b0_pos130_parent_s2p30",
        "label": "Stage 1 pos 130（stage 2 pos 30 的來源 patch）",
        "manual_note": "代表圖均呈現明顯的水平分層色彩分布，上下區域明度與色調存在明顯差異。輸入區塊同樣具有水平分層特徵，與代表圖的梯度方向一致。",
    },
    {
        "img": 0,
        "rel_key": "img0/s1_b0_pos131_parent_s2p30",
        "label": "Stage 1 pos 131（stage 2 pos 30 的來源 patch）",
        "manual_note": "代表圖整體明度較高，色調以灰白為主，部分代表圖帶有不規則的局部明暗變化，但缺乏一致的梯度方向或結構性分布。輸入圖片則有明顯水平顏色分層，與 repr 4 特徵接近。",
    },
    # stage 1 pos 116 對應的 stage 0 children
    {
        "img": 0,
        "rel_key": "img0/s0_b0_pos456_parent_s1p116",
        "label": "Stage 0 pos 456（stage 1 pos 116 的來源 patch）",
        "manual_note": "涵蓋黃褐、粉紅至深紅棕及深灰等差異顯著的色調，缺乏一致的色彩或梯度方向。",
    },
    {
        "img": 0,
        "rel_key": "img0/s0_b0_pos457_parent_s1p116",
        "label": "Stage 0 pos 457（stage 1 pos 116 的來源 patch）",
        "manual_note": "代表圖以低彩度色塊為主，多數呈現由上至下明度遞減的水平漸層。輸入圖片為灰綠調，上亮下暗，與 repr2～repr4 的漸層方向一致。",
    },
    {
        "img": 0,
        "rel_key": "img0/s0_b0_pos484_parent_s1p116",
        "label": "Stage 0 pos 484（stage 1 pos 116 的來源 patch）",
        "manual_note": "各代表圖均呈現均勻的低明度、低彩度色塊，色調集中於深灰至灰綠範圍，缺乏明顯的亮暗梯度或色彩邊界。",
    },
    {
        "img": 0,
        "rel_key": "img0/s0_b0_pos485_parent_s1p116",
        "label": "Stage 0 pos 485（stage 1 pos 116 的來源 patch）",
        "manual_note": "代表圖色調分佈較分散，且均質性不一致。輸入圖片色調與 repr 1、2 較接近，水平漸層則與 repr 4 一致。",
    },
]

# --------------------------------------------------
# 附錄完整 trace 清單
# 格式：每個 entry 是一個 dict，包含：
#   img        : int，影像編號
#   top_rank   : int，GradCAM top-k 排名（例如 1）
#   pos        : int，stage 3 的 pos（例如 2）
#   n_pages    : int，trace page 總頁數（例如 11）
#   caption    : str，附錄 figure 的說明文字
# trace page 檔名格式：top{rank:02d}_pos{pos}_trace_page{page:02d}.png
# --------------------------------------------------
APPENDIX_TRACES = [
    # {
    #     "img": 4,
    #     "top_rank": 1,
    #     "pos": 2,
    #     "n_pages": 11,
    #     "caption": (
    #         "img4（椅子）top-1 patch（stage~3 pos2）之完整跨 stage 來源回溯視覺化，"
    #         "共 11 頁，涵蓋 Stage 3 至 Stage 0 的所有來源位置與群聚代表樣本。"
    #         "每一橫列左側為輸入 patch，右側四格為所屬群聚的代表樣本（repr1--repr4）。"
    #     ),
    # },
]

# ===========================================================================
# 工具函式
# ===========================================================================

def load_jsonl(path: Path) -> dict[str, dict]:
    """讀取 JSONL，以 rel_key 為鍵建立查詢字典。"""
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = obj.get("rel_key", "")
            if key:
                records[key] = obj
    return records


def escape_latex(text: str) -> str:
    """對 LaTeX 特殊字元進行簡單跳脫（用於 caption 文字）。"""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&",  r"\&"),
        ("%",  r"\%"),
        ("$",  r"\$"),
        ("#",  r"\#"),
        ("_",  r"\_"),
        ("{",  r"\{"),
        ("}",  r"\}"),
        ("~",  r"\textasciitilde{}"),
        ("^",  r"\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def extract_summary(caption: str) -> str:
    """從 VLM caption 中擷取「總結：」後的一句話摘要。"""
    for line in caption.split("\n"):
        line = line.strip()
        if line.startswith("總結："):
            return line[3:].strip()
    # 若找不到總結標記，取最後一個非空行
    lines = [l.strip() for l in caption.split("\n") if l.strip()]
    return lines[-1] if lines else caption


def extract_steps_until_top2(caption: str) -> str:
    """
    擷取 VLM caption 中「Top-1（最可能）預測」冒號後，到「Top-2」前的內容。
    """
    text = caption.strip()
    if not text:
        return ""

    top1_markers = (
        "**Top-1（最可能）預測**：",
        "**Top-1（最可能）預測**:",
        "Top-1（最可能）預測：",
        "Top-1（最可能）預測:",
        "**Top-1 預測**：",
        "**Top-1 預測**:",
        "Top-1 預測：",
        "Top-1 預測:",
        "**Top-1**：",
        "**Top-1**:",
        "Top-1：",
        "Top-1:",
    )
    start = -1
    for marker in top1_markers:
        start = text.find(marker)
        if start >= 0:
            start += len(marker)
            break
    if start < 0:
        return extract_summary(caption)

    top2_markers = ("- **Top-2", "**Top-2", "Top-2", "- **Top 2", "**Top 2", "Top 2")
    end_candidates = [
        idx for marker in top2_markers
        if (idx := text.find(marker, start)) >= 0
    ]
    end = min(end_candidates) if end_candidates else len(text)
    return text[start:end].strip()


def copy_file(src: Path, dst: Path) -> bool:
    """複製檔案，目標目錄不存在時自動建立。回傳是否成功。"""
    if not src.exists():
        print(f"  [WARN] 找不到來源檔案：{src}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


# ===========================================================================
# 主文片段生成
# ===========================================================================

def generate_selected_patches_tex(
    img_idx: int,
    patches: list[dict],
    records: dict[str, dict],
    figures_rel_base: str,   # LaTeX 相對路徑前綴，例如 figures/ch4/caltech101/img4
    export_figures_dir: Path, # 實際複製目標目錄
    kmeans_root: Path,
) -> str:
    """
    生成主文用的 LaTeX 片段。
    每個 patch 輸出一個 figure 環境，包含：
      - single row 圖片
      - VLM 摘要（總結句）
      - 人工觀察補充（若有）
    每個 patch 各自輸出為單欄 figure。
    """
    lines = []
    lines.append(f"% ---- 主文 case study：img{img_idx} ----")
    lines.append(f"% 自動生成，請勿直接修改此區塊")
    lines.append("")

    # 分組：vlm_analysis JSONL 的 rel_key 格式為 img{idx}/s{stage}_...
    s3_patches = [p for p in patches if p["rel_key"].startswith(f"img{img_idx}/s3_")]
    s2_patches = [p for p in patches if p["rel_key"].startswith(f"img{img_idx}/s2_")]
    s1_patches = [p for p in patches if p["rel_key"].startswith(f"img{img_idx}/s1_")]
    s0_patches = [p for p in patches if p["rel_key"].startswith(f"img{img_idx}/s0_")]

    def append_single_patch_figure(patch: dict) -> None:
        rel_key = patch["rel_key"]
        rec = records.get(rel_key)
        fname = rel_key.split("/")[-1] + ".png"  # e.g. s3_b0_pos2_top01.png

        # 複製圖片
        src = kmeans_root / f"img{img_idx}" / fname
        dst = export_figures_dir / fname
        copy_file(src, dst)

        img_path = f"{figures_rel_base}/{fname}"
        label_tex = escape_latex(patch["label"])

        analysis_text = ""
        if rec:
            analysis_text = escape_latex(extract_steps_until_top2(rec["caption"]))
        manual = escape_latex(patch.get("manual_note", ""))

        fig_label = rel_key.replace("/", "-").replace("_", "-")

        lines.append(r"\begin{figure}[htbp]")
        lines.append(r"  \centering")
        lines.append(f"  \\includegraphics[width=0.92\\linewidth]{{{img_path}}}")
        lines.append(f"  \\caption{{{label_tex}}}")
        lines.append(f"  \\label{{fig:{fig_label}}}")
        if analysis_text or manual:
            lines.append(r"  \vspace{0.5em}")
            lines.append(r"  \begin{minipage}{0.92\linewidth}")
            if analysis_text:
                lines.append(f"    \\textbf{{VLM 分析：}}{analysis_text}")
            if analysis_text and manual:
                lines.append("")
            if manual:
                lines.append(f"    \\textbf{{人工觀察：}}{manual}")
            lines.append(r"  \end{minipage}")
        lines.append(r"\end{figure}")
        lines.append("")

    # Stage 3/2/1/0：依 SELECTED_PATCHES 順序分 stage 輸出。
    for patch in s3_patches + s2_patches + s1_patches + s0_patches:
        append_single_patch_figure(patch)

    return "\n".join(lines)


# ===========================================================================
# 附錄片段生成
# ===========================================================================

def generate_appendix_trace_tex(
    trace: dict,
    figures_rel_base: str,   # LaTeX 相對路徑前綴，例如 figures/appendix/caltech101/img4
    export_figures_dir: Path,
    kmeans_root: Path,
) -> str:
    """
    生成附錄用的 LaTeX 片段。
    把所有 trace page 圖片依序排列，每張佔整行。
    """
    img_idx   = trace["img"]
    top_rank  = trace["top_rank"]
    pos       = trace["pos"]
    n_pages   = trace["n_pages"]
    caption   = escape_latex(trace["caption"])

    lines = []
    lines.append(f"% ---- 附錄完整 trace：img{img_idx} top{top_rank:02d} pos{pos} ----")
    lines.append(f"% 自動生成，請勿直接修改此區塊")
    lines.append("")

    # 所有 trace page 串成一個 figure 群（用 figure* 或 longtable 視排版需求調整）
    # 這裡用連續的 figure 讓 LaTeX 自動換頁
    for page in range(1, n_pages + 1):
        fname = f"top{top_rank:02d}_pos{pos}_trace_page{page:02d}.png"
        src = kmeans_root / f"img{img_idx}" / fname
        dst = export_figures_dir / fname
        copy_file(src, dst)

        img_path = f"{figures_rel_base}/{fname}"
        page_label = f"appendix-img{img_idx}-top{top_rank:02d}-pos{pos}-page{page:02d}"

        lines.append(r"\begin{figure}[p]")
        lines.append(r"  \centering")
        lines.append(f"  \\includegraphics[width=\\linewidth]{{{img_path}}}")
        if page == 1:
            # 只有第一頁放完整 caption 和 label
            lines.append(f"  \\caption{{{caption}}}")
            lines.append(f"  \\label{{fig:{page_label}}}")
        else:
            lines.append(
                f"  \\caption{{（續）img{img_idx} top-{top_rank} pos{pos} "
                f"跨 stage 來源回溯，第 {page} 頁／共 {n_pages} 頁。}}"
            )
            lines.append(f"  \\label{{fig:{page_label}}}")
        lines.append(r"\end{figure}")
        lines.append("")

    return "\n".join(lines)


# ===========================================================================
# 主程式
# ===========================================================================

def main():
    print("載入 JSONL...")
    records = load_jsonl(JSONL_PATH)
    print(f"  共 {len(records)} 筆記錄")

    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    generated_dir = EXPORT_ROOT / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    # 按 img_idx 分組主文選圖
    imgs_in_selected = sorted(set(p["img"] for p in SELECTED_PATCHES))

    for img_idx in imgs_in_selected:
        patches = [p for p in SELECTED_PATCHES if p["img"] == img_idx]

        figures_dir = EXPORT_ROOT / "figures" / "ch4" / DATASET / f"img{img_idx}"
        figures_rel = f"figures/ch4/{DATASET}/img{img_idx}"

        print(f"\n生成主文片段：img{img_idx}（{len(patches)} 個 patch）")
        tex = generate_selected_patches_tex(
            img_idx=img_idx,
            patches=patches,
            records=records,
            figures_rel_base=figures_rel,
            export_figures_dir=figures_dir,
            kmeans_root=KMEANS_ROOT,
        )

        out_path = generated_dir / f"selected_patches_img{img_idx}.tex"
        out_path.write_text(tex, encoding="utf-8")
        print(f"  → {out_path}")

    # 附錄完整 trace
    for trace in APPENDIX_TRACES:
        img_idx  = trace["img"]
        top_rank = trace["top_rank"]
        pos      = trace["pos"]

        figures_dir = EXPORT_ROOT / "figures" / "appendix" / DATASET / f"img{img_idx}"
        figures_rel = f"figures/appendix/{DATASET}/img{img_idx}"

        print(f"\n生成附錄片段：img{img_idx} top{top_rank:02d} pos{pos}（{trace['n_pages']} 頁）")
        tex = generate_appendix_trace_tex(
            trace=trace,
            figures_rel_base=figures_rel,
            export_figures_dir=figures_dir,
            kmeans_root=KMEANS_ROOT,
        )

        out_path = generated_dir / f"appendix_trace_img{img_idx}_top{top_rank:02d}_pos{pos}.tex"
        out_path.write_text(tex, encoding="utf-8")
        print(f"  → {out_path}")

    print("\n完成。請將 latex_export/ 整個複製至 Windows 論文目錄。")
    print("論文中使用方式：")
    print("  \\input{generated/selected_patches_img0}")


if __name__ == "__main__":
    main()