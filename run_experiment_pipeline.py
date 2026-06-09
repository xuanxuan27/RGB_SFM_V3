#!/usr/bin/env python3
"""
run_experiment_pipeline.py — K-means 分析 → VLM caption → 彙整 CSV 的完整 pipeline。

用法：
  python run_experiment_pipeline.py                        # 跑全部三個 step
  python run_experiment_pipeline.py --skip-kmeans          # 跳過 K-means，直接跑 VLM
  python run_experiment_pipeline.py --skip-caption         # 跳過 VLM，直接彙整
  python run_experiment_pipeline.py --skip-summary         # 跳過彙整
  python run_experiment_pipeline.py --skip-kmeans --skip-caption  # 只彙整

所有參數從 config.py 讀取，不另外定義一套參數。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
import re
import sys
from pathlib import Path

# 確保可以 import 同層的模組
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---------------------------------------------------------------------------
# Step 1：K-means 分析
# ---------------------------------------------------------------------------

def run_step_kmeans(cfg: dict) -> Path:
    """
    執行 K-means 分析，含 single_rows 存圖與 K-means cache。
    回傳 gradcam_trace 根目錄路徑。
    """
    from mergingViT_plot_tool.Kmeans_analysis import run_dataset_analysis_all

    model_name   = cfg["model"]["name"]
    model_args   = cfg["model"]["args"]
    ckpt_dir     = cfg.get("kmeans_checkpoint_dir") or cfg["save_dir"]
    model_path   = str(Path(cfg["root"]) / ckpt_dir / f"{model_name}_best.pth")
    dataset      = cfg["dataset"]
    input_shape  = cfg["input_shape"]
    clusters_list = cfg.get("kmeans_clusters_per_stage", [30, 60, 120, 240])
    n_clusters_per_stage = {i: v for i, v in enumerate(clusters_list)}
    save_dir     = cfg.get("kmeans_save_dir", "plots/kmeans/Caltech101/")
    data_root    = str(Path(cfg["root"]) / "data")

    print("\n" + "=" * 60)
    print("Step 1：K-means 分析")
    print(f"  model_path : {model_path}")
    print(f"  dataset    : {dataset}")
    print(f"  save_dir   : {save_dir}")
    print("=" * 60)

    run_dataset_analysis_all(
        model_path=model_path,
        dataset=dataset,
        img_size=input_shape[0],
        patch_size=model_args.get("patch_size", 8),
        max_samples=None,
        n_clusters_per_stage=n_clusters_per_stage,
        k_nearest=4,
        clusters_per_fig=1,
        positions=None,
        save_dir=save_dir,
        m_inference=cfg.get("m_inference", 5),
        inference_seed=cfg.get("inference_seed", 42),
        save_cluster_representatives=False,
        model_args=model_args,
        data_root=data_root,
        mode="token",
        heads=None,
        analysis_batch_size=cfg.get("batch_size", 4),
        model_name=model_name,
        gradcam_top_k=cfg.get("gradcam_trace_top_k", 5),
        trace_block=cfg.get("gradcam_trace_block", "last"),
        save_gradcam_trace=cfg.get("save_gradcam_trace", True),
        trace_max_rows_per_fig=cfg.get("trace_max_rows_per_fig", 12),
        trace_expansions_per_fig=cfg.get("trace_expansions_per_fig", 2),
        save_all_inference_repr=cfg.get("save_all_inference_repr", False),
    )

    gradcam_trace_dir = Path(save_dir) / "inference" / "gradcam_trace"
    print(f"\n✓ Step 1 完成，gradcam_trace 目錄：{gradcam_trace_dir.resolve()}")
    return gradcam_trace_dir


# ---------------------------------------------------------------------------
# Step 2：VLM Caption
# ---------------------------------------------------------------------------

def _discover_gradcam_single_row_records(rca, gradcam_trace_dir: Path):
    """
    掃描 GradCAM trace 輸出的 single_rows。

    新流程會輸出到 gradcam_trace/img{N}/single_rows；也保留支援舊的
    gradcam_trace/single_rows 目錄。
    """
    gradcam_trace_dir = Path(gradcam_trace_dir)

    if gradcam_trace_dir.name == "single_rows":
        candidate_roots = [gradcam_trace_dir.parent]
    elif (gradcam_trace_dir / "single_rows").is_dir():
        candidate_roots = [gradcam_trace_dir]
    else:
        candidate_roots = [
            p for p in sorted(gradcam_trace_dir.glob("img*"))
            if p.is_dir() and (p / "single_rows").is_dir()
        ]

    records = []
    img_dir_pattern = re.compile(r"img(\d+)$")
    for root in candidate_roots:
        root_records = rca.discover_single_row_images(root)
        match = img_dir_pattern.match(root.name)
        if match:
            img_idx = int(match.group(1))
            for rec in root_records:
                rec.img = img_idx
                rec.rel_key = f"{root.name}/{rec.rel_key}"
        records.extend(root_records)

    return records


def run_step_caption(cfg: dict, gradcam_trace_dir: Path, output_jsonl: Path) -> None:
    """
    對 gradcam_trace 目錄裡的 single_rows 圖跑 VLM，輸出 JSONL。
    已存在的 rel_key 自動跳過（--skip-existing）。
    """
    from vit_analysis_vlm import run_caption_analysis as rca

    vlm_cfg      = cfg.get("vlm_analysis", {})
    model_id     = vlm_cfg.get("model", "Qwen/Qwen2-VL-7B-Instruct")
    max_new_tokens = vlm_cfg.get("max_new_tokens", 512)
    stage_start  = vlm_cfg.get("stage_start", 3)
    stage_end    = vlm_cfg.get("stage_end", 0)

    print("\n" + "=" * 60)
    print("Step 2：VLM Caption")
    print(f"  gradcam_trace: {gradcam_trace_dir.resolve()}")
    print(f"  output_jsonl: {output_jsonl.resolve()}")
    print(f"  model       : {model_id}")
    print("=" * 60)

    if not gradcam_trace_dir.is_dir():
        print("  ✗ 找不到 gradcam_trace 目錄，跳過 Step 2。")
        return

    records = _discover_gradcam_single_row_records(rca, gradcam_trace_dir)
    records = rca.filter_and_sort_records_by_stage(
        records, stage_start=stage_start, stage_end=stage_end
    )

    if not records:
        print("  ✗ 找不到符合 stage 範圍的圖片，跳過 Step 2。")
        return

    prompt = rca.build_single_row_prompt()

    family = rca.detect_model_family(model_id)
    print(f"  載入模型 {model_id}（family: {family}）…")
    processor, model, model_family = rca.load_model(model_id)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    done_keys = rca.load_done_rel_keys(output_jsonl)
    print(f"  已完成：{len(done_keys)} 筆，本次需處理：{len(records)} 筆")

    with output_jsonl.open("a", encoding="utf-8") as out_f:
        for i, rec in enumerate(records):
            if rec.rel_key in done_keys:
                print(f"  [{i+1}/{len(records)}] 略過（已存在）{rec.rel_key}")
                continue
            print(f"  [{i+1}/{len(records)}] {rec.rel_key} …", flush=True)
            try:
                text = rca.caption_one_image(
                    processor=processor,
                    model=model,
                    model_family=model_family,
                    image_path=rec.image_path,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                )
            except Exception as e:
                text = f"[ERROR] {e!s}"
                print(f"       失敗: {e}", flush=True)

            rec_data = asdict(rec) if is_dataclass(rec) else vars(rec)
            row = {
                **rec_data,
                "caption": text,
                "prompt": prompt,
                "model": model_id,
                "model_family": model_family,
                "max_new_tokens": max_new_tokens,
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()

    print(f"\n✓ Step 2 完成，輸出：{output_jsonl.resolve()}")


# ---------------------------------------------------------------------------
# Step 3：彙整 CSV
# ---------------------------------------------------------------------------

def run_step_summary(output_jsonl: Path, manual_notes_path: Path | None, out_csv: Path) -> None:
    """
    讀取 JSONL，合併人工觀察（若有），輸出彙整 CSV。
    CSV 欄位：stage / block / pos / parent_stage / parent_pos / top_rank /
              vlm_caption / manual_note
    """
    import csv

    print("\n" + "=" * 60)
    print("Step 3：彙整 CSV")
    print(f"  jsonl    : {output_jsonl.resolve()}")
    print(f"  out_csv  : {out_csv.resolve()}")
    print("=" * 60)

    if not output_jsonl.is_file():
        print("  ✗ 找不到 JSONL，跳過 Step 3。")
        return

    # 讀取人工觀察（若有）：格式同為 JSONL，key 為 rel_key，含 manual_note 欄位
    manual: dict[str, str] = {}
    if manual_notes_path and manual_notes_path.is_file():
        with manual_notes_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    k = obj.get("rel_key")
                    note = obj.get("manual_note", "")
                    if k:
                        manual[k] = note
                except json.JSONDecodeError:
                    continue
        print(f"  載入人工觀察：{len(manual)} 筆")

    rows = []
    with output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append({
                "stage":        obj.get("stage", ""),
                "block":        obj.get("block", ""),
                "pos":          obj.get("pos", ""),
                "parent_stage": obj.get("parent_stage", ""),
                "parent_pos":   obj.get("parent_pos", ""),
                "top_rank":     obj.get("top_rank", ""),
                "rel_key":      obj.get("rel_key", ""),
                "vlm_caption":  obj.get("caption", ""),
                "manual_note":  manual.get(obj.get("rel_key", ""), ""),
            })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["stage", "block", "pos", "parent_stage", "parent_pos",
                  "top_rank", "rel_key", "vlm_caption", "manual_note"]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Step 3 完成，共 {len(rows)} 筆，輸出：{out_csv.resolve()}")


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="K-means → VLM caption → 彙整 CSV pipeline")
    p.add_argument("--skip-kmeans",  action="store_true", help="跳過 Step 1（K-means 分析）")
    p.add_argument("--skip-caption", action="store_true", help="跳過 Step 2（VLM caption）")
    p.add_argument("--skip-summary", action="store_true", help="跳過 Step 3（彙整 CSV）")
    p.add_argument(
        "--manual-notes",
        type=Path,
        default=None,
        help="人工觀察 JSONL 路徑（欄位：rel_key + manual_note），可選",
    )
    p.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="VLM caption 輸出 JSONL 路徑（預設從 config 的 kmeans_save_dir 推算）",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="彙整 CSV 輸出路徑（預設從 config 的 kmeans_save_dir 推算）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import config as cfg_module
    cfg = cfg_module.config

    save_dir = Path(cfg.get("kmeans_save_dir", "plots/kmeans/Caltech101/"))
    output_jsonl = args.output_jsonl or (save_dir / "vlm_captions.jsonl")
    output_csv   = args.output_csv   or (save_dir / "vlm_summary.csv")

    gradcam_trace_dir = save_dir / "inference" / "gradcam_trace"

    # Step 1
    if not args.skip_kmeans:
        gradcam_trace_dir = run_step_kmeans(cfg)
    else:
        print("\nStep 1：跳過（--skip-kmeans）")

    # Step 2
    if not args.skip_caption:
        run_step_caption(cfg, gradcam_trace_dir, output_jsonl)
    else:
        print("\nStep 2：跳過（--skip-caption）")

    # Step 3
    if not args.skip_summary:
        run_step_summary(output_jsonl, args.manual_notes, output_csv)
    else:
        print("\nStep 3：跳過（--skip-summary）")

    print("\n✓ Pipeline 完成。")


if __name__ == "__main__":
    main()