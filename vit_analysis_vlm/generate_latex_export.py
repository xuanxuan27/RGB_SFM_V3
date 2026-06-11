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
JSONL_PATH = Path("/home/xuan/RGB_SFM_V3/plots/kmeans/Caltech101/vlm_captions_caltech101.jsonl")

# Kmeans 輸出根目錄（含 inference/gradcam_trace/）
KMEANS_ROOT = Path("/home/xuan/RGB_SFM_V3/plots/kmeans/Caltech101/inference/gradcam_trace/")

# 匯出根目錄
EXPORT_ROOT = Path("/home/xuan/RGB_SFM_V3/latex_export")

# 資料集名稱（用於路徑）
DATASET = "caltech101"

# --------------------------------------------------
# 主文選圖清單
# 格式：每個 entry 是一個 dict，包含：
#   img        : int，影像編號
#   rel_key    : str，對應 JSONL 的 rel_key
#   label      : str，LaTeX 圖片的簡短說明（人工填寫）
#   manual_note: str，人工觀察補充（可留空）
# --------------------------------------------------
SELECTED_PATCHES = [
    # img4 椅子：stage 3 top1
    {
        "img": 4,
        "rel_key": "img4/single_rows/s3_b0_pos2_top01",
        "label": "Stage~3 top-1 patch（pos2，椅背垂直木條區域）",
        "manual_note": "",
    },
    # img4 椅子：stage 2 四個 source patch
    {
        "img": 4,
        "rel_key": "img4/single_rows/s2_b0_pos4_parent_s3p2",
        "label": "Stage~2 pos4（top-1 的左上來源，木條與邊框交界）",
        "manual_note": "",
    },
    {
        "img": 4,
        "rel_key": "img4/single_rows/s2_b0_pos5_parent_s3p2",
        "label": "Stage~2 pos5（top-1 的右上來源，純垂直木條）",
        "manual_note": "",
    },
    {
        "img": 4,
        "rel_key": "img4/single_rows/s2_b0_pos11_parent_s3p2",
        "label": "Stage~2 pos11（top-1 的左下來源，木材質感色塊）",
        "manual_note": "",
    },
    {
        "img": 4,
        "rel_key": "img4/single_rows/s2_b0_pos12_parent_s3p2",
        "label": "Stage~2 pos12（top-1 的右下來源，細垂直條紋）",
        "manual_note": "",
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
    {
        "img": 4,
        "top_rank": 1,
        "pos": 2,
        "n_pages": 11,
        "caption": (
            "img4（椅子）top-1 patch（stage~3 pos2）之完整跨 stage 來源回溯視覺化，"
            "共 11 頁，涵蓋 stage~3 至 stage~0 的所有來源位置與群聚代表樣本。"
            "每一橫列左側為輸入 patch，右側四格為所屬群聚的代表樣本（repr1--repr4）。"
        ),
    },
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
    最後以 subfigure 環境把 stage 2 的四張合成一個 2×2 的組合圖。
    """
    lines = []
    lines.append(f"% ---- 主文 case study：img{img_idx} ----")
    lines.append(f"% 自動生成，請勿直接修改此區塊")
    lines.append("")

    # 分組：stage 3 和 stage 2 分開排
    s3_patches = [p for p in patches if p["rel_key"].startswith(f"img{img_idx}/single_rows/s3_")]
    s2_patches = [p for p in patches if p["rel_key"].startswith(f"img{img_idx}/single_rows/s2_")]

    # Stage 3：單張大圖
    for patch in s3_patches:
        rel_key = patch["rel_key"]
        rec = records.get(rel_key)
        fname = rel_key.split("/")[-1] + ".png"  # e.g. s3_b0_pos2_top01.png

        # 複製圖片
        src = kmeans_root / f"img{img_idx}" / "single_rows" / fname
        dst = export_figures_dir / fname
        copy_file(src, dst)

        img_path = f"{figures_rel_base}/{fname}"
        label_tex = escape_latex(patch["label"])

        summary = ""
        if rec:
            summary = escape_latex(extract_summary(rec["caption"]))
        manual = escape_latex(patch.get("manual_note", ""))

        caption_parts = [label_tex]
        if summary:
            caption_parts.append(f"\\textbf{{VLM 描述：}}{summary}")
        if manual:
            caption_parts.append(f"\\textbf{{人工觀察：}}{manual}")
        caption_tex = "\\newline ".join(caption_parts)

        fig_label = rel_key.replace("/", "-").replace("_", "-")

        lines.append(r"\begin{figure}[htbp]")
        lines.append(r"  \centering")
        lines.append(f"  \\includegraphics[width=0.92\\linewidth]{{{img_path}}}")
        lines.append(f"  \\caption{{{caption_tex}}}")
        lines.append(f"  \\label{{fig:{fig_label}}}")
        lines.append(r"\end{figure}")
        lines.append("")

    # Stage 2：四張組合成 2×2 subfigure
    if s2_patches:
        lines.append(r"\begin{figure}[htbp]")
        lines.append(r"  \centering")

        for i, patch in enumerate(s2_patches):
            rel_key = patch["rel_key"]
            rec = records.get(rel_key)
            fname = rel_key.split("/")[-1] + ".png"

            src = kmeans_root / f"img{img_idx}" / "single_rows" / fname
            dst = export_figures_dir / fname
            copy_file(src, dst)

            img_path = f"{figures_rel_base}/{fname}"
            label_tex = escape_latex(patch["label"])

            summary = ""
            if rec:
                summary = escape_latex(extract_summary(rec["caption"]))
            manual = escape_latex(patch.get("manual_note", ""))

            subcaption_parts = [label_tex]
            if summary:
                subcaption_parts.append(summary)
            if manual:
                subcaption_parts.append(f"人工：{manual}")
            subcaption_tex = "；".join(subcaption_parts)

            fig_label = rel_key.replace("/", "-").replace("_", "-")

            # 2×2：每兩張換行
            lines.append(r"  \begin{subfigure}[b]{0.48\linewidth}")
            lines.append(r"    \centering")
            lines.append(f"    \\includegraphics[width=\\linewidth]{{{img_path}}}")
            lines.append(f"    \\caption{{{subcaption_tex}}}")
            lines.append(f"    \\label{{fig:{fig_label}}}")
            lines.append(r"  \end{subfigure}")
            if i % 2 == 1 and i < len(s2_patches) - 1:
                lines.append(r"  \\[6pt]")
            elif i % 2 == 0:
                lines.append(r"  \hfill")

        lines.append(
            f"  \\caption{{img{img_idx} top-1 patch（stage~3 pos2）"
            r"之四個 stage~2 來源位置群聚代表樣本。"
            r"每張子圖左側為輸入 patch，右側四格為所屬群聚的代表樣本。}}"
        )
        lines.append(f"  \\label{{fig:img{img_idx}-s2-sources}}")
        lines.append(r"\end{figure}")
        lines.append("")

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
    print("  \\input{generated/selected_patches_img4}")
    print("  \\input{generated/appendix_trace_img4_top01_pos2}")


if __name__ == "__main__":
    main()