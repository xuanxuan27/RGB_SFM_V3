#!/usr/bin/env python3
"""
依 plots/vlm_analysis/stage{S}_block{B}/repr_pos{P}_part{K}.png 批次送進 Qwen2-VL，
將每張 cluster 代表圖的敘述另存為 JSONL。

用法（專案根目錄）:
  python vit_analysis_vlm/run_caption_analysis.py
  python vit_analysis_vlm/run_caption_analysis.py --plots-root /path/to/plots/vlm_analysis --limit 3

若尚未載過模型，首次會從 Hugging Face 下載。
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

# 專案根（vit_analysis_vlm/ 的上一層）
_ROOT = Path(__file__).resolve().parent.parent

DIR_PATTERN = re.compile(r"^stage(\d+)_block(\d+)$")
FILE_PATTERN_ORIGINAL = re.compile(r"^repr_pos(\d+)_part(\d+)\.png$", re.IGNORECASE)
FILE_PATTERN_SPLIT_CLUSTER = re.compile(
    r"^repr_pos(\d+)_part(\d+)_cluster(\d+)\.png$", re.IGNORECASE
)


@dataclass
class PlotRecord:
    image_path: str
    stage: int
    block: int
    pos: int
    part: int
    cluster: int | None
    rel_key: str


def discover_repr_images(plots_root: Path, *, input_structure: str) -> list[PlotRecord]:
    """
    掃描 stage*_block* 子資料夾內符合命名的 png。

    input_structure:
    - original: repr_pos{pos}_part{part}.png
    - split_cluster: repr_pos{pos}_part{part}_cluster{c}.png
    - auto: 先嘗試 split_cluster，若找不到再用 original
    """
    plots_root = plots_root.resolve()
    if not plots_root.is_dir():
        raise FileNotFoundError(f"找不到圖片根目錄: {plots_root}")

    if input_structure not in {"original", "split_cluster", "auto"}:
        raise ValueError(f"未知 input_structure: {input_structure}")

    records: list[PlotRecord] = []
    for sub in sorted(plots_root.iterdir()):
        if not sub.is_dir():
            continue
        dm = DIR_PATTERN.match(sub.name)
        if not dm:
            continue
        stage, block = int(dm.group(1)), int(dm.group(2))
        patterns: list[tuple[str, re.Pattern[str]]] = []
        if input_structure == "original":
            patterns = [(r"repr_pos*_part*.png", FILE_PATTERN_ORIGINAL)]
        elif input_structure == "split_cluster":
            patterns = [(r"repr_pos*_part*_cluster*.png", FILE_PATTERN_SPLIT_CLUSTER)]
        else:  # auto
            patterns = [
                (r"repr_pos*_part*_cluster*.png", FILE_PATTERN_SPLIT_CLUSTER),
                (r"repr_pos*_part*.png", FILE_PATTERN_ORIGINAL),
            ]

        matched_any = False
        for glob_pat, file_pat in patterns:
            for png in sorted(sub.glob(glob_pat)):
                fm = file_pat.match(png.name)
                if not fm:
                    continue
                pos, part = int(fm.group(1)), int(fm.group(2))
                cluster: int | None = None
                if file_pat is FILE_PATTERN_SPLIT_CLUSTER:
                    cluster = int(fm.group(3))
                rel = f"stage{stage}_block{block}/repr_pos{pos}_part{part}"
                if cluster is not None:
                    rel = f"{rel}_cluster{cluster}"
                records.append(
                    PlotRecord(
                        image_path=str(png.resolve()),
                        stage=stage,
                        block=block,
                        pos=pos,
                        part=part,
                        cluster=cluster,
                        rel_key=rel,
                    )
                )
                matched_any = True
            if matched_any:
                break

    records.sort(
        key=lambda r: (
            r.stage,
            r.block,
            r.pos,
            r.part,
            -1 if r.cluster is None else r.cluster,
            r.image_path,
        )
    )
    return records


def build_default_prompt() -> str:
    return (
        "這張圖包含多張來自同一個 K-means cluster 的 image patch，"
        "這些 patch 是從 ViT attention 特徵空間中聚類後，最接近群心的代表樣本。\n\n"
        "請依以下幾個面向分析這些 patch 的共同視覺特徵：\n"
        "1. 顏色與對比：主要色調、前景背景關係\n"
        "2. 形狀與結構：幾何特徵、筆畫方向、邊緣特性\n"
        "3. 位置線索：特徵集中在 patch 的哪個區域（上/下/左/右/中）\n"
        "4. 跨樣本一致性：這幾張 patch 之間相似在哪、差異在哪\n\n"
        "最後用一句話總結：這個 cluster 最可能在捕捉圖像中的什麼局部視覺結構。\n"
        "請用繁體中文回答。"
    )


def caption_one_image(
    *,
    processor,
    model,
    image_uri: str,
    prompt: str,
    max_new_tokens: int,
) -> str:
    # Lazy import so --dry-run can work without torch/transformers installed.
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_uri},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    out = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return out[0].strip() if out else ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="對 cluster 代表圖批次跑 Qwen2-VL 敘述")
    p.add_argument(
        "--plots-root",
        type=Path,
        default=_ROOT / "plots" / "vlm_analysis",
        help="內含 stage{S}_block{B} 子資料夾的目錄（預設: <repo>/plots/vlm_analysis）",
    )
    p.add_argument(
        "--input-structure",
        type=str,
        default="original",
        choices=["original", "split_cluster", "auto"],
        help=(
            "輸入圖片命名結構（預設 original）。"
            " original=repr_pos{pos}_part{part}.png；"
            " split_cluster=repr_pos{pos}_part{part}_cluster{c}.png；"
            " auto=先找 split_cluster，沒有再找 original。"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "plots" / "vlm_analysis",
        help="輸出 JSONL 目錄（預設: <repo>/plots/vlm_analysis）",
    )
    p.add_argument(
        "--output-template",
        type=str,
        default="captions_stage{stage}.jsonl",
        help="每個 stage 的輸出檔名樣板，需包含 {stage}（預設: captions_stage{stage}.jsonl）",
    )
    p.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="Hugging Face 模型 id 或本機路徑",
    )
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只處理前 N 張圖（除錯用）",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只列出將處理的檔案，不載入模型",
    )
    p.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="自訂 prompt 文字檔（UTF-8），若未指定則用內建繁中說明",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="若 output JSONL 中已有相同 rel_key，則跳過（簡易續跑）",
    )
    p.add_argument(
        "--stage-start",
        type=int,
        default=3,
        help="起始 stage（預設 3）",
    )
    p.add_argument(
        "--stage-end",
        type=int,
        default=1,
        help="結束 stage（預設 1）",
    )
    return p.parse_args()


def load_done_rel_keys(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.is_file():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                k = obj.get("rel_key") or obj.get("meta", {}).get("rel_key")
                if k:
                    done.add(str(k))
            except json.JSONDecodeError:
                continue
    return done


def stage_range_inclusive(start: int, end: int) -> list[int]:
    step = -1 if start >= end else 1
    return list(range(start, end + step, step))


def filter_and_sort_records_by_stage(
    records: Iterable[PlotRecord], stage_start: int, stage_end: int
) -> list[PlotRecord]:
    stages = set(stage_range_inclusive(stage_start, stage_end))
    reverse = stage_start > stage_end
    out = [r for r in records if r.stage in stages]
    out.sort(
        key=lambda r: (
            -r.stage if reverse else r.stage,
            r.block,
            r.pos,
            r.part,
            -1 if r.cluster is None else r.cluster,
            r.image_path,
        )
    )
    return out


def main() -> None:
    args = parse_args()
    if "{stage}" not in args.output_template:
        raise ValueError("--output-template 必須包含 {stage} 佔位符")

    records = discover_repr_images(args.plots_root, input_structure=args.input_structure)
    records = filter_and_sort_records_by_stage(
        records, stage_start=args.stage_start, stage_end=args.stage_end
    )
    if args.limit is not None:
        records = records[: args.limit]

    if args.prompt_file is not None:
        prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    else:
        prompt = build_default_prompt()

    if args.dry_run:
        print(f"plots_root={args.plots_root.resolve()}")
        print(f"input_structure={args.input_structure}")
        print(
            f"stage 範圍: {args.stage_start} -> {args.stage_end}（含）"
            f"，共 {len(records)} 張，prompt 長度 {len(prompt)} 字元"
        )
        for r in records:
            print(r.rel_key, "->", r.image_path)
        return

    if not records:
        print(
            f"在 {args.plots_root.resolve()} 底下找不到符合 stage 範圍 "
            f"{args.stage_start}->{args.stage_end} 的 repr_pos*_part*.png，結束。"
        )
        return

    # Lazy import: only needed when actually running captioning.
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    print(f"載入模型 {args.model} …")
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    stage_to_records: dict[int, list[PlotRecord]] = {}
    for rec in records:
        stage_to_records.setdefault(rec.stage, []).append(rec)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written_files: list[Path] = []

    for stage, stage_records in stage_to_records.items():
        out_path = args.output_dir / args.output_template.format(stage=stage)
        done_keys = load_done_rel_keys(out_path) if args.skip_existing else set()
        written_files.append(out_path)

        print(f"\n=== Stage {stage}：{len(stage_records)} 張，輸出 -> {out_path.resolve()} ===")
        with out_path.open("a", encoding="utf-8") as out_f:
            for i, rec in enumerate(stage_records):
                if rec.rel_key in done_keys:
                    print(f"[{i+1}/{len(stage_records)}] 略過（已存在） {rec.rel_key}")
                    continue
                uri = Path(rec.image_path).as_uri()
                print(f"[{i+1}/{len(stage_records)}] {rec.rel_key} …", flush=True)
                try:
                    text = caption_one_image(
                        processor=processor,
                        model=model,
                        image_uri=uri,
                        prompt=prompt,
                        max_new_tokens=args.max_new_tokens,
                    )
                except Exception as e:
                    text = f"[ERROR] {e!s}"
                    print(f"     失敗: {e}", flush=True)

                row = {
                    **asdict(rec),
                    "caption": text,
                    "prompt": prompt,
                    "model": args.model,
                    "max_new_tokens": args.max_new_tokens,
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()

    print("\n完成，結果輸出檔案：")
    for p in written_files:
        print(" -", p.resolve())


if __name__ == "__main__":
    main()