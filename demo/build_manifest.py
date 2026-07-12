#!/usr/bin/env python3
"""
build_manifest.py — 將 K-means / Grad-CAM trace / VLM caption 的輸出
彙整成單一 demo_manifest.json，供 Streamlit demo 讀取。

Demo app 只讀 manifest + 圖片檔，不載入模型、不跑推論。

輸入（皆由 run_experiment_pipeline.py 產生）:
  {kmeans_save_dir}/inference/
      img{I}_original.png
      img{I}_gradcam_trace_clusters.json
      gradcam_trace/img{I}/
          img{I}_gradcam_top{K}_overview.png
          img{I}_gradcam_top{K}_trace_summary.png
          single_rows/
              s{S}_b{B}_pos{P}_top{R}.png              <- top-k 節點本身
              s{S}_b{B}_pos{P}_parent_s{PS}p{PP}.png   <- 子 patch
  {kmeans_save_dir}/vlm_captions.jsonl        (可選)
  manual_notes.jsonl                          (可選, 欄位: rel_key + manual_note)

用法:
  python demo/build_manifest.py \
      --kmeans-save-dir plots/kmeans/Caltech101 \
      --captions plots/kmeans/Caltech101/vlm_captions.jsonl \
      --manual-notes demo/manual_notes.jsonl \
      --out demo/demo_manifest.json

輸出的 manifest 結構:
{
  "images": [
    {
      "img_idx": 0,
      "original": "<path>",
      "overview": "<path>",
      "trace_summary": "<path>",
      "target_class": 12, "pred_class": 12,
      "last_stage": 3, "last_stage_H": 4, "last_stage_W": 4,
      "roots": [
        {   # 一個 top-k patch = 一棵樹
          "rank": 1, "pos": 5, "score": 0.93,
          "node": { ...遞迴 node... }
        }
      ]
    }
  ]
}

node = {
  "stage", "block", "pos", "row", "col", "is_pad",
  "cluster", "image", "rel_key", "caption", "manual_note",
  "children": [node, ...]
}
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

TOP_PATTERN = re.compile(r"^s(\d+)_b(\d+)_pos(\d+)_top(\d+)\.png$", re.IGNORECASE)
CHILD_PATTERN = re.compile(
    r"^s(\d+)_b(\d+)_pos(\d+)_parent_s(\d+)p(\d+)\.png$", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# 讀取輔助資料
# ---------------------------------------------------------------------------

def load_jsonl_by_relkey(path: Path | None, field: str) -> dict[str, str]:
    """讀 JSONL，回傳 {rel_key: field 值}。rel_key 可能含 img{N}/ 前綴。"""
    out: dict[str, str] = {}
    if path is None or not Path(path).is_file():
        return out
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = obj.get("rel_key")
            val = obj.get(field)
            if key and val is not None:
                out[str(key)] = str(val)
    return out


def lookup(table: dict[str, str], *candidates: str) -> str:
    """
    caption JSONL 的 rel_key 前綴不一致（有的含 img{N}/、single_rows/，有的沒有）。
    依序嘗試多個候選 key，並額外嘗試「後綴比對」作為 fallback。
    """
    for cand in candidates:
        if cand in table:
            return table[cand]
    for cand in candidates:
        for key, val in table.items():
            if key.endswith(cand):
                return val
    return ""


def annotation_candidates(img_idx: int, rel_key: str) -> list[str]:
    """
    產生 caption / manual note 的候選 key。

    Grad-CAM trace 的 root 檔名使用 top01/top02...，但 vlm_analysis
    手動挑選圖常會把同一個 stage/pos 存成 top00，因此 root 節點需要額外
    嘗試 top00 命名。
    """
    candidates = [
        f"img{img_idx}/single_rows/{rel_key}",
        f"single_rows/{rel_key}",
        f"img{img_idx}/{rel_key}",
        rel_key,
    ]

    m = TOP_PATTERN.match(f"{rel_key}.png")
    if m:
        s, b, p, _r = (int(m.group(i)) for i in range(1, 5))
        top00_key = f"s{s}_b{b}_pos{p}_top00"
        candidates.extend([
            f"img{img_idx}/single_rows/{top00_key}",
            f"single_rows/{top00_key}",
            f"img{img_idx}/{top00_key}",
            top00_key,
        ])

    return candidates


# ---------------------------------------------------------------------------
# 從 trace JSON 重建 stage 幾何
# ---------------------------------------------------------------------------

def infer_stage_grids(payload: dict, merge_size: int) -> dict[int, tuple[int, int]]:
    """
    由最後 stage 的 (H, W) 往回推每個 stage 的 padded 網格大小。

    merge 是 padded 的：stage s 的 padded_H = next_H * m_h。
    因此回推時直接乘 merge_size 即可得到 padded 網格，
    這正好對應 single_rows 檔名裡 pos 的編碼基準（含 padding 的 W）。

    注意：真實（未 padding）的 H/W 可能小於 padded 值，但 pos 的計算
    在 _children_for_parent 用的是 info["W"]（真實 W），
    在 padding 位置則 pos=None。因此我們只用這個網格做 parent 推導的
    「候選」，真正的 parent 關係以檔名為準（見 build_tree）。
    """
    last = int(payload["last_stage"])
    grids: dict[int, tuple[int, int]] = {
        last: (int(payload["last_stage_H"]), int(payload["last_stage_W"]))
    }
    h, w = grids[last]
    for s in range(last - 1, -1, -1):
        h, w = h * merge_size, w * merge_size
        grids[s] = (h, w)
    return grids


# ---------------------------------------------------------------------------
# 掃描 single_rows
# ---------------------------------------------------------------------------

def scan_single_rows(single_rows_dir: Path) -> tuple[dict, dict]:
    """
    回傳:
      tops:     {(stage, pos): {"rank", "block", "path"}}
      children: {(parent_stage, parent_pos): [{"stage","pos","block","path"}, ...]}
    """
    tops: dict[tuple[int, int], dict] = {}
    children: dict[tuple[int, int], list[dict]] = {}

    if not single_rows_dir.is_dir():
        return tops, children

    for png in sorted(single_rows_dir.glob("*.png")):
        m = TOP_PATTERN.match(png.name)
        if m:
            s, b, p, r = (int(m.group(i)) for i in range(1, 5))
            tops[(s, p)] = {"rank": r, "block": b, "path": png}
            continue

        m = CHILD_PATTERN.match(png.name)
        if m:
            s, b, p, ps, pp = (int(m.group(i)) for i in range(1, 6))
            children.setdefault((ps, pp), []).append(
                {"stage": s, "pos": p, "block": b, "path": png}
            )

    for lst in children.values():
        lst.sort(key=lambda d: d["pos"])
    return tops, children


# ---------------------------------------------------------------------------
# cluster 查表
# ---------------------------------------------------------------------------

def build_cluster_lookup(payload: dict) -> dict[tuple[int, int], int | None]:
    """
    從 trace JSON 建 {(stage, pos): cluster}。
    JSON 的 clusters 是 list（trace_block="all" 時可能多筆），取第一筆。
    """
    table: dict[tuple[int, int], int | None] = {}

    def _first_cluster(recs: list) -> int | None:
        if not recs:
            return None
        return recs[0].get("cluster")

    last_stage = int(payload["last_stage"])
    for item in payload.get("top_positions", []):
        table[(last_stage, int(item["pos"]))] = _first_cluster(item.get("clusters", []))
        for stage_str, sources in item.get("sources_by_stage", {}).items():
            stage = int(stage_str)
            for src in sources:
                if src.get("pos") is None:
                    continue
                table[(stage, int(src["pos"]))] = _first_cluster(src.get("clusters", []))
    return table


# ---------------------------------------------------------------------------
# 遞迴建樹
# ---------------------------------------------------------------------------

def build_tree(
    stage: int,
    pos: int,
    block: int,
    image_path: Path | None,
    *,
    children_map: dict,
    cluster_table: dict,
    captions: dict,
    notes: dict,
    img_idx: int,
    assets_root: Path,
    seen: set,
) -> dict[str, Any]:
    """
    以 (stage, pos) 為節點遞迴展開。子節點來自 single_rows 檔名記錄的
    parent 關係，因此不需要重算 merge 幾何 —— 檔名即事實來源。

    seen 防止共享祖先造成的無窮遞迴（不同 top-k 的子樹可能重疊）。
    """
    key = (stage, pos)
    rel_key = _rel_key_for(stage, block, pos, children_map, image_path)

    node: dict[str, Any] = {
        "stage": stage,
        "pos": pos,
        "block": block,
        "cluster": cluster_table.get(key),
        "image": _relpath(image_path, assets_root) if image_path else None,
        "rel_key": rel_key,
        "caption": lookup(captions, *annotation_candidates(img_idx, rel_key)),
        "manual_note": lookup(notes, *annotation_candidates(img_idx, rel_key)),
        "children": [],
    }

    if key in seen:
        node["shared_ancestor"] = True
        return node
    seen = seen | {key}

    for child in children_map.get(key, []):
        node["children"].append(
            build_tree(
                child["stage"],
                child["pos"],
                child["block"],
                child["path"],
                children_map=children_map,
                cluster_table=cluster_table,
                captions=captions,
                notes=notes,
                img_idx=img_idx,
                assets_root=assets_root,
                seen=seen,
            )
        )
    node["children"].sort(key=lambda n: n["pos"])
    return node


def _rel_key_for(stage, block, pos, children_map, image_path) -> str:
    """由檔名還原 rel_key（不含副檔名），與 run_caption_analysis 的命名一致。"""
    if image_path is None:
        return f"s{stage}_b{block}_pos{pos}"
    return Path(image_path).stem


def _relpath(p: Path | None, root: Path) -> str | None:
    if p is None:
        return None
    try:
        return str(Path(p).resolve().relative_to(root.resolve()))
    except ValueError:
        return str(Path(p).resolve())


# ---------------------------------------------------------------------------
# 單張圖
# ---------------------------------------------------------------------------

def build_image_entry(
    img_idx: int,
    inference_dir: Path,
    captions: dict,
    notes: dict,
    assets_root: Path,
) -> dict[str, Any] | None:
    trace_json = inference_dir / f"img{img_idx}_gradcam_trace_clusters.json"
    if not trace_json.is_file():
        return None

    payload = json.loads(trace_json.read_text(encoding="utf-8"))
    trace_dir = inference_dir / "gradcam_trace" / f"img{img_idx}"
    tops, children_map = scan_single_rows(trace_dir / "single_rows")
    cluster_table = build_cluster_lookup(payload)

    last_stage = int(payload["last_stage"])

    roots = []
    for item in sorted(payload.get("top_positions", []), key=lambda d: d["rank"]):
        pos = int(item["pos"])
        meta = tops.get((last_stage, pos))
        block = meta["block"] if meta else 0
        image_path = meta["path"] if meta else None
        node = build_tree(
            last_stage, pos, block, image_path,
            children_map=children_map,
            cluster_table=cluster_table,
            captions=captions,
            notes=notes,
            img_idx=img_idx,
            assets_root=assets_root,
            seen=set(),
        )
        roots.append({
            "rank": int(item["rank"]),
            "pos": pos,
            "score": float(item.get("score", 0.0)),
            "row": int(item.get("row", 0)),
            "col": int(item.get("col", 0)),
            "node": node,
        })

    def _find(pattern: str) -> str | None:
        hits = sorted(trace_dir.glob(pattern))
        return _relpath(hits[0], assets_root) if hits else None

    return {
        "img_idx": img_idx,
        "original": _relpath(inference_dir / f"img{img_idx}_original.png", assets_root)
        if (inference_dir / f"img{img_idx}_original.png").is_file() else None,
        "overview": _find(f"img{img_idx}_gradcam_top*_overview.png"),
        "trace_summary": _find(f"img{img_idx}_gradcam_top*_trace_summary.png"),
        "target_class": payload.get("target_class"),
        "pred_class": payload.get("pred_class"),
        "last_stage": last_stage,
        "last_stage_H": payload.get("last_stage_H"),
        "last_stage_W": payload.get("last_stage_W"),
        "cam": payload.get("cam"),
        "roots": roots,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Build demo manifest from pipeline outputs")
    ap.add_argument("--kmeans-save-dir", type=Path, required=True)
    ap.add_argument("--captions", type=Path, default=None,
                    help="VLM caption JSONL（含 rel_key / caption）")
    ap.add_argument("--manual-notes", type=Path, default=None,
                    help="人工觀察 JSONL（含 rel_key / manual_note）")
    ap.add_argument("--out", type=Path, default=Path("demo/demo_manifest.json"))
    ap.add_argument("--assets-root", type=Path, default=None,
                    help="manifest 內圖片路徑的相對基準；預設為 kmeans-save-dir")
    args = ap.parse_args()

    inference_dir = args.kmeans_save_dir / "inference"
    if not inference_dir.is_dir():
        raise SystemExit(f"找不到 inference 目錄: {inference_dir.resolve()}")

    assets_root = args.assets_root or args.kmeans_save_dir

    captions = load_jsonl_by_relkey(args.captions, "caption")
    notes = load_jsonl_by_relkey(args.manual_notes, "manual_note")
    print(f"載入 caption {len(captions)} 筆、manual note {len(notes)} 筆")

    img_indices = sorted(
        int(m.group(1))
        for p in inference_dir.glob("img*_gradcam_trace_clusters.json")
        if (m := re.match(r"^img(\d+)_gradcam_trace_clusters\.json$", p.name))
    )
    if not img_indices:
        raise SystemExit(f"在 {inference_dir} 找不到 img*_gradcam_trace_clusters.json")

    images = []
    for idx in img_indices:
        entry = build_image_entry(idx, inference_dir, captions, notes, assets_root)
        if entry is None:
            continue
        n_nodes = sum(_count(r["node"]) for r in entry["roots"])
        print(f"  img{idx}: {len(entry['roots'])} 棵樹, {n_nodes} 個節點")
        images.append(entry)

    manifest = {
        "assets_root": str(assets_root.resolve()),
        "n_images": len(images),
        "images": images,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n✓ 輸出 manifest: {args.out.resolve()}")


def _count(node: dict) -> int:
    return 1 + sum(_count(c) for c in node.get("children", []))


if __name__ == "__main__":
    main()
