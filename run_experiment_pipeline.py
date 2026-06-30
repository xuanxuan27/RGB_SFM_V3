#!/usr/bin/env python3
"""
run_experiment_pipeline.py — K-means 分析 → VLM caption → 彙整 CSV 的完整 pipeline。

用法：
  python run_experiment_pipeline.py                        # 跑全部三個 step
  python run_experiment_pipeline.py --skip-kmeans          # 跳過 K-means，直接跑 VLM
  python run_experiment_pipeline.py --skip-caption         # 跳過 VLM，直接彙整
  python run_experiment_pipeline.py --skip-summary         # 跳過彙整
  python run_experiment_pipeline.py --skip-kmeans --skip-caption  # 只彙整
  python run_experiment_pipeline.py --caption-stage-start 3 --caption-stage-end 1

所有參數從 config.py 讀取，不另外定義一套參數。
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, is_dataclass
import importlib.util
import io
import json
import re
import sys
from pathlib import Path

# 確保可以 import 同層的模組
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---------------------------------------------------------------------------
# Step 1：K-means 分析
# ---------------------------------------------------------------------------

def _resolve_project_path(project_root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project_root / path


def _load_run_config(checkpoint_dir: str | Path, project_root: Path) -> dict | None:
    """
    從訓練 run 目錄讀取當時保存的 config.py。

    train.py 會把根目錄的 config.py 複製到 runs/train/exp*/config.py；
    用這份 config 建模型，才能避免目前 config.py 改過 merge_size 後和 checkpoint 不一致。
    """
    config_path = _resolve_project_path(project_root, checkpoint_dir) / "config.py"
    if not config_path.is_file():
        return None

    module_name = f"_kmeans_run_config_{abs(hash(config_path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)

    run_config = getattr(module, "config", None)
    return run_config if isinstance(run_config, dict) else None


def run_step_kmeans(cfg: dict) -> Path:
    """
    執行 K-means 分析，含 single_rows 存圖與 K-means cache。
    回傳 gradcam_trace 根目錄路徑。
    """
    from mergingViT_plot_tool.Kmeans_analysis import run_dataset_analysis_all

    project_root = Path(cfg["root"])
    ckpt_dir     = cfg.get("kmeans_checkpoint_dir") or cfg["save_dir"]
    run_cfg      = _load_run_config(ckpt_dir, project_root)
    model_cfg    = run_cfg.get("model", cfg["model"]) if run_cfg else cfg["model"]
    model_name   = model_cfg["name"]
    model_args   = model_cfg["args"]
    model_path   = str(_resolve_project_path(project_root, ckpt_dir) / f"{model_name}_best.pth")
    dataset      = cfg["dataset"]
    input_shape  = cfg["input_shape"]
    clusters_list = cfg.get("kmeans_clusters_per_stage", [30, 60, 120, 240])
    n_clusters_per_stage = {i: v for i, v in enumerate(clusters_list)}
    save_dir     = cfg.get("kmeans_save_dir", "plots/kmeans/Caltech101/")
    data_root    = str(Path(cfg["root"]) / "data")

    print("\n" + "=" * 60)
    print("Step 1：K-means 分析")
    print(f"  model_path : {model_path}")
    if run_cfg:
        print(f"  model_args : 從 {(_resolve_project_path(project_root, ckpt_dir) / 'config.py').resolve()} 讀取")
        print(f"  merge_size : {model_args.get('merge_size')}")
    else:
        print("  model_args : 從目前 config.py 讀取")
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
        m_inference=cfg.get("m_inference", 3),
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
        analysis_split=cfg.get("kmeans_dataset_split", "auto"),
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


def _discover_vlm_analysis_records(rca, vlm_analysis_dir: Path):
    """
    掃描 inference/vlm_analysis/img{N}/ 內的 single-row PNG。
    這些圖片通常由 all_pos_trace 中挑選並複製而來，檔名沿用
    s{S}_b{B}_pos{P}_parent_s{PS}p{PP}.png 或 s{S}_b{B}_pos{P}_top{R}.png。
    """
    vlm_analysis_dir = Path(vlm_analysis_dir)
    img_dir_pattern = re.compile(r"img(\d+)$")
    candidate_roots = [
        p for p in sorted(vlm_analysis_dir.glob("img*"))
        if p.is_dir() and img_dir_pattern.match(p.name)
    ]

    records = []
    for root in candidate_roots:
        root_records = rca.discover_single_row_images(root)
        img_idx = int(img_dir_pattern.match(root.name).group(1))
        for rec in root_records:
            rec.img = img_idx
            rec.rel_key = f"{root.name}/{rec.rel_key}"
        records.extend(root_records)

    return records


def run_step_caption(
    cfg: dict,
    gradcam_trace_dir: Path,
    output_jsonl: Path,
    caption_stage_start: int | None = None,
    caption_stage_end: int | None = None,
    dataset_name: str | None = None,
    caption_source_override: str | None = None,
) -> None:
    """
    對 gradcam_trace 目錄裡的 single_rows 圖跑 VLM，輸出 JSONL。
    已存在的 rel_key 自動跳過（--skip-existing）。
    """
    from vit_analysis_vlm import run_caption_analysis as rca

    vlm_cfg      = cfg.get("vlm_analysis", {})
    model_id     = vlm_cfg.get("model", "Qwen/Qwen2-VL-7B-Instruct")
    max_new_tokens = vlm_cfg.get("max_new_tokens", 512)
    stage_start  = caption_stage_start if caption_stage_start is not None else vlm_cfg.get("stage_start", 3)
    stage_end    = caption_stage_end if caption_stage_end is not None else vlm_cfg.get("stage_end", 0)
    prompt_dataset = dataset_name or vlm_cfg.get("dataset") or cfg.get("dataset")
    caption_source = caption_source_override or vlm_cfg.get("caption_source", "gradcam_trace")
    save_dir = Path(cfg.get("kmeans_save_dir", "plots/kmeans/Caltech101/"))
    vlm_analysis_dir = Path(vlm_cfg.get("input_dir", save_dir / "inference" / "vlm_analysis"))

    print("\n" + "=" * 60)
    print("Step 2：VLM Caption")
    print(f"  source      : {caption_source}")
    print(f"  gradcam_trace: {gradcam_trace_dir.resolve()}")
    print(f"  vlm_analysis: {vlm_analysis_dir.resolve()}")
    print(f"  output_jsonl: {output_jsonl.resolve()}")
    print(f"  dataset    : {prompt_dataset or '(generic)'}")
    print(f"  model       : {model_id}")
    print(f"  stages      : {stage_start} -> {stage_end}")
    print("=" * 60)

    if caption_source == "gradcam_trace":
        if not gradcam_trace_dir.is_dir():
            print("  ✗ 找不到 gradcam_trace 目錄，跳過 Step 2。")
            return
        records = _discover_gradcam_single_row_records(rca, gradcam_trace_dir)
    elif caption_source == "vlm_analysis":
        if not vlm_analysis_dir.is_dir():
            print("  ✗ 找不到 vlm_analysis 目錄，跳過 Step 2。")
            return
        records = _discover_vlm_analysis_records(rca, vlm_analysis_dir)
    else:
        raise ValueError("vlm_analysis.caption_source 必須是 'gradcam_trace' 或 'vlm_analysis'")

    records = rca.filter_and_sort_records_by_stage(
        records, stage_start=stage_start, stage_end=stage_end
    )

    if not records:
        print("  ✗ 找不到符合 stage 範圍的圖片，跳過 Step 2。")
        return

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
            prompt = rca.build_single_row_prompt(prompt_dataset, stage=rec.stage)
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
                "dataset": prompt_dataset,
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
    p.add_argument(
        "--caption-stage-start",
        type=int,
        default=None,
        help="VLM caption 起始 stage；預設讀 config['vlm_analysis']['stage_start']，再預設為 3",
    )
    p.add_argument(
        "--caption-stage-end",
        type=int,
        default=None,
        help="VLM caption 結束 stage；預設讀 config['vlm_analysis']['stage_end']，再預設為 0",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="覆寫 config['dataset']，並用於 K-means 與 VLM prompt 選擇",
    )
    p.add_argument(
        "--caption-source",
        choices=("gradcam_trace", "vlm_analysis"),
        default=None,
        help="覆寫 config['vlm_analysis']['caption_source']",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import config as cfg_module
    cfg = dict(cfg_module.config)
    if args.dataset:
        cfg["dataset"] = args.dataset

    save_dir = Path(cfg.get("kmeans_save_dir", "plots/kmeans/Caltech101/"))
    vlm_cfg = cfg.get("vlm_analysis", {})
    caption_source = args.caption_source or vlm_cfg.get("caption_source", "gradcam_trace")
    default_stem = "vlm_analysis" if caption_source == "vlm_analysis" else "vlm"
    output_jsonl = args.output_jsonl or (save_dir / f"{default_stem}_captions.jsonl")
    output_csv   = args.output_csv   or (save_dir / f"{default_stem}_summary.csv")

    gradcam_trace_dir = save_dir / "inference" / "gradcam_trace"

    # Step 1
    if not args.skip_kmeans:
        gradcam_trace_dir = run_step_kmeans(cfg)
    else:
        print("\nStep 1：跳過（--skip-kmeans）")

    # Step 2
    if not args.skip_caption:
        run_step_caption(
            cfg,
            gradcam_trace_dir,
            output_jsonl,
            caption_stage_start=args.caption_stage_start,
            caption_stage_end=args.caption_stage_end,
            dataset_name=args.dataset,
            caption_source_override=args.caption_source,
        )
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