#!/usr/bin/env python3
"""
run_merge_sweep.py — 自動跑多組 merge_size 組合的訓練實驗。

原理：
  每個 run 前，把 config.py 替換成對應 merge 設定的版本，
  再用 subprocess 跑 train.py。跑完後還原 config.py。
  這樣 train.py 完全不需要修改。

用法：
  python run_merge_sweep.py                    # 跑全部組合
  python run_merge_sweep.py --dry-run          # 只印出計畫，不實際訓練
  python run_merge_sweep.py --start-from 2     # 從第 2 組開始（斷點續跑）
  python run_merge_sweep.py --only 0 3         # 只跑第 0 和第 3 組
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================
# ★ 在這裡設定所有要跑的 merge 組合
# ============================================================
# 格式：每個 entry 是一個 dict，包含：
#   name       : wandb run 名稱（會存進 config['name']）
#   group      : wandb group（方便在 wandb UI 一起比較）
#   merge_size : 傳給 MergingViT 的 merge_size，支援
#                  - int：所有 stage 用同一個值，e.g. 2
#                  - list of (h,w) tuples：per-stage 設定
#
# 其他參數（embed_dims、depths、drop_rate 等）維持 config.py 原本的設定。
# 如果某組需要額外覆寫其他參數，加進 extra_model_args 或 extra_config。

MERGE_EXPERIMENTS = [
    {
        "name": "MergingViT_Colored_MNIST_SF221212",
        "group": "Colored_MNIST_sweep",
        "merge_size": [(2, 2), (1, 2), (1, 2)],
        "extra_model_args": {},
        "extra_config": {},
    },
    {
        "name": "MergingViT_Colored_MNIST_SF222121",
        "group": "Colored_MNIST_sweep",
        "merge_size": [(2, 2), (2, 1), (2, 1)],
        "extra_model_args": {},
        "extra_config": {},
    },
    {
        "name": "MergingViT_Colored_MNIST_SF221441",
        "group": "Colored_MNIST_sweep",
        "merge_size": [(2, 2), (1, 4), (4, 1)],
        "extra_model_args": {},
        "extra_config": {},
    },
    {
        "name": "MergingViT_Colored_MNIST_SF224114",
        "group": "Colored_MNIST_sweep",
        "merge_size": [(2, 2), (4, 1), (1, 4)],
        "extra_model_args": {},
        "extra_config": {},
    },
    {
        "name": "MergingViT_Colored_MNIST_SF411422",
        "group": "Colored_MNIST_sweep",
        "merge_size": [(4, 1), (1, 4), (2, 2)],
        "extra_model_args": {},
        "extra_config": {},
    },
    {
        "name": "MergingViT_Colored_MNIST_SF144122",
        "group": "Colored_MNIST_sweep",
        "merge_size": [(1, 4), (4, 1), (2, 2)],
        "extra_model_args": {},
        "extra_config": {},
    },
]
# ============================================================


def fmt_duration(seconds: float) -> str:
    """把秒數格式化成 1h 23m 45s。"""
    return str(timedelta(seconds=int(seconds)))


def build_config_py(original_config_path: Path, exp: dict) -> str:
    """
    讀取原始 config.py，把 merge_size、name 等欄位替換成本次實驗的設定，
    回傳修改後的字串（不寫檔，讓呼叫端決定）。
    """
    original = original_config_path.read_text(encoding="utf-8")

    merge_size = exp["merge_size"]
    extra_model = exp.get("extra_model_args", {})
    extra_cfg = exp.get("extra_config", {})

    lines = original.splitlines()
    new_lines = []

    for line in lines:
        stripped = line.strip()

        # 替換 name（wandb run 名稱）
        if stripped.startswith("name = ") and "MergingViT" in stripped:
            new_lines.append(f"name = '{exp['name']}'")
            continue

        # 替換 group
        if stripped.startswith("group = "):
            new_lines.append(f"group = '{exp['group']}'")
            continue

        # 替換 merge_size
        if '"merge_size"' in stripped and ":" in stripped:
            indent = len(line) - len(line.lstrip())
            new_lines.append(" " * indent + f'"merge_size": {merge_size},')
            continue

        new_lines.append(line)

    result = "\n".join(new_lines)

    # 處理 extra_model_args（如果有需要覆寫的其他 model args）
    for key, val in extra_model.items():
        # 簡單字串替換：找到 "key": ... 這一行並替換
        import re
        pattern = rf'("  {key}"\s*:\s*)[^\n,]+'
        replacement = rf'\g<1>{repr(val)}'
        result = re.sub(pattern, replacement, result)

    return result


def find_new_exp_dir(runs_dir: Path, known_dirs: set[Path]) -> Path | None:
    """
    訓練結束後，在 runs/train/ 找出這次新建的 exp 資料夾。
    做法：比對訓練前後的資料夾集合，取差集。
    """
    if not runs_dir.exists():
        return None
    current = set(runs_dir.iterdir())
    new_dirs = current - known_dirs
    if not new_dirs:
        return None
    # 取最新建立的（以防萬一有多個）
    return max(new_dirs, key=lambda p: p.stat().st_ctime)


def run_experiment(
    exp: dict,
    idx: int,
    total: int,
    project_dir: Path,
    dry_run: bool = False,
) -> tuple[bool, str | None]:
    """
    執行單一組實驗。回傳 (是否成功, exp 資料夾名稱)。
    """
    config_path = project_dir / "config.py"
    backup_path = project_dir / "config.py.sweep_backup"
    runs_dir = project_dir / "runs" / "train"

    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  [{idx + 1}/{total}] {exp['name']}")
    print(f"  merge_size : {exp['merge_size']}")
    print(f"  group      : {exp['group']}")
    print(sep)

    if dry_run:
        print("  [dry-run] 略過實際訓練")
        return True, None

    # 1. 備份 config.py；記錄訓練前的 exp 資料夾清單
    shutil.copy2(config_path, backup_path)
    existing_dirs = set(runs_dir.iterdir()) if runs_dir.exists() else set()

    try:
        # 2. 寫入本次實驗的 config.py
        new_config = build_config_py(config_path, exp)
        config_path.write_text(new_config, encoding="utf-8")

        # 3. 執行 train.py
        start_time = time.time()
        print(f"  開始時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        result = subprocess.run(
            [sys.executable, "train.py"],
            cwd=str(project_dir),
        )

        elapsed = time.time() - start_time
        print(f"\n  結束時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  耗時：{fmt_duration(elapsed)}")

        # 4. 找出這次建立的 exp 資料夾
        exp_dir = find_new_exp_dir(runs_dir, existing_dirs)
        exp_dir_name = exp_dir.name if exp_dir else "unknown"
        if exp_dir:
            print(f"  checkpoint : runs/train/{exp_dir_name}/")

        if result.returncode != 0:
            print(f"  ✗ train.py 回傳錯誤碼 {result.returncode}")
            return False, exp_dir_name

        print(f"  ✓ 完成")
        return True, exp_dir_name

    except KeyboardInterrupt:
        print("\n  中斷！還原 config.py ...")
        raise

    finally:
        # 5. 無論成功/失敗，還原 config.py
        shutil.copy2(backup_path, config_path)
        backup_path.unlink(missing_ok=True)


def write_sweep_log(log_path: Path, records: list[dict]) -> None:
    """
    把每組實驗的結果寫成 CSV，欄位：
      index | name | merge_size | exp_dir | status | start_time | duration
    """
    fieldnames = ["index", "name", "merge_size", "exp_dir", "status", "start_time", "duration"]
    with log_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"\n  ✓ Sweep log 已存至：{log_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="自動跑多組 merge_size 訓練實驗")
    p.add_argument(
        "--dry-run", action="store_true",
        help="只印出實驗計畫，不實際執行訓練"
    )
    p.add_argument(
        "--start-from", type=int, default=0, metavar="N",
        help="從第 N 組（0-indexed）開始，用於斷點續跑"
    )
    p.add_argument(
        "--only", type=int, nargs="+", metavar="N",
        help="只跑指定的組合索引，e.g. --only 0 2 4"
    )
    p.add_argument(
        "--project-dir", type=Path, default=None,
        help="專案根目錄（預設為此腳本所在位置）"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    project_dir = args.project_dir or Path(__file__).resolve().parent

    # 決定要跑哪些組合
    if args.only:
        indices = args.only
    else:
        indices = list(range(args.start_from, len(MERGE_EXPERIMENTS)))

    experiments = [(i, MERGE_EXPERIMENTS[i]) for i in indices]
    total = len(MERGE_EXPERIMENTS)

    # 印出計畫
    print("\n" + "=" * 65)
    print("  Merge Sweep 計畫")
    print("=" * 65)
    for i, exp in experiments:
        print(f"  [{i}] {exp['name']:30s}  merge={exp['merge_size']}")
    print(f"\n  共 {len(experiments)} 組，預估 {len(experiments) * 2}+ 小時")
    print("=" * 65)

    if args.dry_run:
        print("\n[dry-run 模式] 不會實際執行訓練。\n")

    failed = []
    records = []
    sweep_start = time.time()

    try:
        for i, exp in experiments:
            start_time = datetime.now()
            t0 = time.time()

            success, exp_dir_name = run_experiment(
                exp=exp,
                idx=i,
                total=total,
                project_dir=project_dir,
                dry_run=args.dry_run,
            )

            records.append({
                "index": i,
                "name": exp["name"],
                "merge_size": str(exp["merge_size"]),
                "exp_dir": exp_dir_name or ("dry-run" if args.dry_run else "unknown"),
                "status": "success" if success else "failed",
                "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration": fmt_duration(time.time() - t0),
            })

            if not success:
                failed.append((i, exp["name"]))
                print(f"  ⚠ [{i}] {exp['name']} 失敗，繼續下一組...")

    except KeyboardInterrupt:
        print("\n\n使用者中斷。")

    finally: 
        # 寫 sweep log
        if not args.dry_run and records:
            log_path = project_dir / f"sweep_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            write_sweep_log(log_path, records)

    # 最終摘要
    total_elapsed = time.time() - sweep_start
    print("\n" + "=" * 65)
    print("  Sweep 完成摘要")
    print("=" * 65)
    print(f"  總耗時 : {fmt_duration(total_elapsed)}")
    for r in records:
        status_icon = "✓" if r["status"] == "success" else "✗"
        print(f"  {status_icon} [{r['index']}] {r['name']:30s}  → {r['exp_dir']}  ({r['duration']})")
    if failed:
        print(f"\n  失敗組 : {len(failed)} 組")
        for i, name in failed:
            print(f"    [{i}] {name}")
    else:
        print("\n  全部成功 ✓")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()