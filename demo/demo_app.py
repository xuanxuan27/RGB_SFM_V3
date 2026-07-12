#!/usr/bin/env python3
"""
demo_app.py — MergingViT 語義追溯 Trace Explorer

只讀 demo_manifest.json 與磁碟上的 PNG，不載入模型、不跑推論。
口試現場零 GPU 依賴。

用法:
  streamlit run demo/demo_app.py -- --manifest demo/demo_manifest.json

或設定環境變數:
  DEMO_MANIFEST=demo/demo_manifest.json streamlit run demo/demo_app.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import streamlit as st

# --- 論文定位用的固定文案（口試時的定調語） -------------------------------
FRAMING = {
    "gradcam": "Grad-CAM 僅用於**優先級排序**：決定哪些 token 值得檢視，"
               "不構成任何語義主張。",
    "kmeans": "K-means 提供**候選分組**：以 L2-normalized embedding 做等價於 "
              "cosine 的分群，是結構鷹架，不是語義結論。",
    "vlm": "VLM caption 是**輔助可讀標註**（auxiliary annotation），"
           "用於讓語義可被人工檢驗，不作為獨立驗證證據。",
}

# 各 stage 的粒度標籤（中性描述，不作語義宣稱）
STAGE_HINT = {
    0: "最細粒度",
    1: "低階組合",
    2: "中階組合",
    3: "最抽象",
}

# ---------------------------------------------------------------------------
# Stage-level 人工分析
#
# 這裡承載的是論文中「跨 stage 語義演化趨勢」的整體論證，
# 而非對單一 patch 的個別標註 —— 你的人工分析本來就是 stage-level 的。
#
# TODO(Sharon): 用論文對應章節的敘述替換以下佔位文字。
#               每則建議 1–3 句，直接引用或改寫論文段落，並標注出處（§x.y）。
#               若某層尚未撰寫，留空字串即可，UI 會自動隱藏該區塊。
# ---------------------------------------------------------------------------
STAGE_NOTE: dict[int, str] = {
    3: "（論文 §_._）【佔位】Stage 3 的整體觀察：cluster 代表圖呈現何種一致性？"
       "是否接近物件層級的語義？請以論文原文替換。",
    2: "（論文 §_._）【佔位】Stage 2 的整體觀察：是否開始出現部件層級的分化？",
    1: "（論文 §_._）【佔位】Stage 1 的整體觀察：語義是否退化為紋理 / 局部結構？",
    0: "（論文 §_._）【佔位】Stage 0 的整體觀察：是否僅剩顏色與邊緣等低階描述？",
}

# 整體結論（顯示於頁尾）。同樣以論文原文替換。
# TODO(Sharon): 若實際 caption 不支持此趨勢，請刪除此段，
#               不要讓介面替論文做未經驗證的宣稱。
CLOSING_NOTE = (
    "（論文 §_._）【佔位】跨 stage 的整體結論。"
    "例如：語義隨 stage 由抽象收斂至局部；此過程在無 CLS token 的階層式架構下依然可被完整追蹤。"
)


# ---------------------------------------------------------------------------
# 載入
# ---------------------------------------------------------------------------

def parse_cli() -> str:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--manifest", default=os.environ.get("DEMO_MANIFEST",
                                                          "demo/demo_manifest.json"))
    args, _ = ap.parse_known_args(sys.argv[1:])
    return args.manifest


@st.cache_data(show_spinner=False)
def load_manifest(path: str) -> dict:
    p = Path(path)
    if not p.is_file():
        st.error(f"找不到 manifest：{p.resolve()}\n\n"
                 "請先執行 `python demo/build_manifest.py --kmeans-save-dir ...`")
        st.stop()
    return json.loads(p.read_text(encoding="utf-8"))


def asset(manifest: dict, rel: str | None) -> Path | None:
    """manifest 內的路徑是相對於 assets_root。"""
    if not rel:
        return None
    p = Path(rel)
    if not p.is_absolute():
        p = Path(manifest["assets_root"]) / p
    return p if p.is_file() else None


# ---------------------------------------------------------------------------
# 節點渲染
# ---------------------------------------------------------------------------

def _caption_coverage(node: dict) -> tuple[int, int]:
    """回傳 (節點總數, 有 caption 的節點數)。"""
    total = 1
    with_cap = 1 if (node.get("caption") or "").strip() else 0
    for c in node.get("children", []):
        t, w = _caption_coverage(c)
        total += t
        with_cap += w
    return total, with_cap


def node_label(node: dict) -> str:
    cl = node.get("cluster")
    cl_s = f"cluster {cl}" if cl is not None else "cluster —"
    n_child = len(node.get("children", []))
    suffix = f" · {n_child} 個子 patch" if n_child else " · 葉節點"
    return f"Stage {node['stage']} · pos {node['pos']} · {cl_s}{suffix}"


def _annotation_block(label: str, body: str, accent: str, bg: str) -> None:
    st.markdown(
        f"<div style='border-left:3px solid {accent};padding:6px 10px;"
        f"background:{bg};border-radius:4px;margin:6px 0;'>"
        f"<b style='color:{accent};font-size:.8em;letter-spacing:.03em;'>{label}</b>"
        f"<br>{body}</div>",
        unsafe_allow_html=True,
    )


def render_node(node: dict, manifest: dict, depth: int, max_depth: int,
                show_caption: bool, show_note: bool,
                seen_stages: set[int] | None = None) -> None:
    """
    遞迴渲染一個節點。深度 0 為 top-k 根節點。

    seen_stages 用來讓 stage-level 的人工分析在同一條展開路徑上
    只顯示一次（否則四個兄弟 patch 會重複四遍同一段論述）。
    """
    if seen_stages is None:
        seen_stages = set()

    img_path = asset(manifest, node.get("image"))

    if img_path:
        st.image(str(img_path), use_container_width=True)
    else:
        st.info("此節點無代表圖（可能為 padding 位置）")

    stage = node["stage"]
    st.caption(f"**Stage {stage}** — {STAGE_HINT.get(stage, '')}"
               f" · cluster {node.get('cluster', '—')} · pos {node['pos']}")

    if node.get("shared_ancestor"):
        st.caption("↩︎ 此節點為多個 top-k patch 的共同祖先，已在他處展開。")

    # --- VLM：node-level 輔助標註 ---
    if show_caption:
        cap = node.get("caption") or "—（此節點未標註；VLM 為輔助層，不需窮盡）"
        _annotation_block("VLM · AUXILIARY ANNOTATION（NODE-LEVEL）",
                          cap, "#0f766e", "rgba(20,184,166,.07)")

    # --- 人工分析暫時不顯示 ---
    # if show_note and stage not in seen_stages:
    #     stage_note = STAGE_NOTE.get(stage, "").strip()
    #     if stage_note:
    #         _annotation_block(f"MANUAL ANALYSIS · STAGE {stage}（STAGE-LEVEL）",
    #                           stage_note, "#1e3a5f", "rgba(30,58,95,.07)")
    #         seen_stages = seen_stages | {stage}
    #
    # if show_note:
    #     note = node.get("manual_note")
    #     if note:
    #         _annotation_block("MANUAL ANALYSIS（NODE-LEVEL）",
    #                           note, "#1e3a5f", "rgba(30,58,95,.12)")

    children = node.get("children", [])
    if not children or depth >= max_depth:
        return

    st.markdown(f"**↓ 展開至 Stage {stage - 1}（{len(children)} 個子 patch）**")

    # 兄弟節點共用同一份 seen_stages（用 list 包裝以便跨迴圈累積），
    # 確保 stage-level 論述在該層只印一次，而非每個兄弟各印一遍。
    shared = set(seen_stages)
    for child in children:
        with st.expander(node_label(child), expanded=(depth == 0)):
            before = set(shared)
            render_node(child, manifest, depth + 1, max_depth,
                        show_caption, show_note, seen_stages=shared)
            shared |= before | {child["stage"]}


# ---------------------------------------------------------------------------
# 主畫面
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="MergingViT Trace Explorer",
                       layout="wide", initial_sidebar_state="expanded")
    manifest = load_manifest(parse_cli())

    st.markdown(
        "<h1 style='margin-bottom:0'>MergingViT · Semantic Trace Explorer</h1>"
        "<p style='color:#64748b;margin-top:4px'>"
        "追蹤單一 token 從最抽象的 Stage 3 回溯至 Stage 0 的語義演化</p>",
        unsafe_allow_html=True,
    )

    # --- Sidebar ---
    with st.sidebar:
        st.header("控制面板")

        imgs = manifest["images"]
        idx = st.selectbox(
            "推論圖片",
            options=list(range(len(imgs))),
            format_func=lambda i: f"img{imgs[i]['img_idx']}",
        )
        entry = imgs[idx]

        ranks = [r["rank"] for r in entry["roots"]]
        rank = st.selectbox("Grad-CAM Top-k patch", ranks,
                            format_func=lambda r: f"Top {r}")
        root = next(r for r in entry["roots"] if r["rank"] == rank)

        max_depth = st.slider("展開深度", 1, entry["last_stage"],
                              entry["last_stage"])
        show_caption = st.checkbox("顯示 VLM caption", value=True)
        show_note = False

        st.divider()
        total, with_cap = _caption_coverage(root["node"])
        pct = (100 * with_cap / total) if total else 0
        st.caption(f"**此樹 caption 覆蓋率**：{with_cap}/{total}（{pct:.0f}%）")
        if with_cap == 0:
            st.warning(
                "此樹沒有任何 VLM caption。請確認 `--captions` 指向正確檔案："
                "`vlm_captions.jsonl` 或 `vlm_analysis_captions.jsonl`。",
                icon="⚠️",
            )

        st.divider()
        st.caption("**方法定位**")
        for k in ("gradcam", "kmeans", "vlm"):
            st.caption(FRAMING[k])

    # --- Row 1: 原圖 / Grad-CAM overview ---
    st.subheader("1 · 輸入與 Grad-CAM 優先級")
    st.caption(FRAMING["gradcam"])

    panels = [
        ("**原始輸入**", entry.get("original"), "無原圖"),
        ("**Grad-CAM Top-k**", entry.get("overview"), "無 overview"),
    ]
    for col, (title, rel, empty_msg) in zip(st.columns(len(panels)), panels):
        with col:
            st.markdown(title)
            p = asset(manifest, rel)
            if p:
                st.image(str(p), use_container_width=True)
            else:
                st.info(empty_msg)

    m1, m2, m3 = st.columns(3)
    m1.metric("Target class", entry.get("target_class", "—"))
    m2.metric("Pred class", entry.get("pred_class", "—"))
    m3.metric("目前追溯", f"Top {rank} · pos {root['pos']} "
                          f"(score {root['score']:.3f})")

    st.divider()

    # --- Row 2: Trace tree ---
    st.subheader(f"2 · 語義追溯樹 — Top {rank}, position {root['pos']}")
    st.caption(
        "由 Stage 3 向下展開。每一層的 cluster 代表圖說明「該位置的特徵最接近哪一類」；"
        "VLM caption 提供可檢視的輔助語義標註。"
    )
    st.caption(FRAMING["kmeans"] + "　" + FRAMING["vlm"])

    render_node(root["node"], manifest, depth=0, max_depth=max_depth,
                show_caption=show_caption, show_note=show_note)

    st.divider()
    if CLOSING_NOTE.strip():
        st.caption(CLOSING_NOTE)


if __name__ == "__main__":
    main()