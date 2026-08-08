#!/usr/bin/env python3
"""依序對各資料集跑 K-means 分析（與 Kmeans_analysis.py / run_experiment_pipeline.py 配合）"""
from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "overnight_logs"
PLOTS_ROOT = ROOT / "plots" / "kmeans" / "scalable"
LOG_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_ROOT.mkdir(parents=True, exist_ok=True)

# (dataset, checkpoint exp 目錄名, stage0~3 各自的 cluster 數量)
EXPERIMENTS = [
    ("Colored_MNIST", "exp325", [16, 32, 64, 128]),
    ("Colored_FashionMNIST", "exp326", [16, 32, 64, 128]),
    ("BloodMNIST", "exp327", [16, 32, 64, 128]),
    ("Caltech101", "exp330", [32, 64, 128, 256]),
]


def _resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _load_run_config(checkpoint_dir: str | Path) -> dict | None:
    """從 runs/train/exp*/config.py 讀取訓練當下的設定。"""
    config_path = _resolve_project_path(checkpoint_dir) / "config.py"
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


class _Tee:
    """同時寫入多個輸出串流。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


_NOISY_LINE_RE = re.compile(
    r"(收集特徵: batch|K-means 位置進度:|代表圖進度:|per-head 代表圖進度:)"
)


def _keep_for_log(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    if "\r" in line:
        return False
    if _NOISY_LINE_RE.search(line):
        return False
    if line.startswith("Using downloaded and verified file:"):
        return False
    if line.startswith("<class 'dataloader."):
        return False
    return True


class _FilteredLogStream:
    """只把非噪音訊息寫入 log，保留終端完整輸出。"""

    def __init__(self, stream):
        self.stream = stream
        self._buf = ""

    def write(self, data):
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if _keep_for_log(line):
                self.stream.write(line.rstrip() + "\n")
                self.stream.flush()

    def flush(self):
        if self._buf:
            tail = self._buf.strip()
            if _keep_for_log(tail):
                self.stream.write(tail + "\n")
                self.stream.flush()
            self._buf = ""
        self.stream.flush()


def _clusters_per_stage_dict(clusters: list[int]) -> dict[int, int]:
    return {i: int(nc) for i, nc in enumerate(clusters)}


def _clusters_slug(clusters: list[int]) -> str:
    return "_".join(str(nc) for nc in clusters)


def _build_save_dir(dataset: str, exp_id: str, clusters_per_stage: list[int]) -> Path:
    dataset_slug = dataset.lower()
    return PLOTS_ROOT / dataset_slug / exp_id / f"nc{_clusters_slug(clusters_per_stage)}"


def run_one_kmeans(
    dataset: str,
    exp_id: str,
    clusters_per_stage: list[int],
    defaults: dict,
) -> dict:
    checkpoint_dir = f"runs/train/{exp_id}"
    run_cfg = _load_run_config(checkpoint_dir)
    if run_cfg is None:
        raise FileNotFoundError(f"找不到 {checkpoint_dir}/config.py")

    model_cfg = run_cfg.get("model", defaults["model"])
    model_name = model_cfg["name"]
    model_args = model_cfg["args"]
    model_path = str(_resolve_project_path(checkpoint_dir) / f"{model_name}_best.pth")
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"找不到 checkpoint: {model_path}")

    input_shape = run_cfg.get("input_shape", defaults.get("input_shape", (28, 28)))
    save_dir = _build_save_dir(dataset, exp_id, clusters_per_stage)
    n_clusters_per_stage = _clusters_per_stage_dict(clusters_per_stage)

    from mergingViT_plot_tool.Kmeans_analysis_padding_repr import run_dataset_analysis_all

    print("\n" + "=" * 60)
    print(f"K-means: {dataset} / {exp_id}")
    print(f"  model_path : {model_path}")
    print(f"  save_dir   : {save_dir}")
    print(f"  stages nc  : {clusters_per_stage}")
    print("=" * 60)

    start = time.time()
    run_dataset_analysis_all(
        model_path=model_path,
        dataset=dataset,
        img_size=input_shape[0],
        patch_size=model_args.get("patch_size", 2),
        max_samples=None,
        n_clusters=clusters_per_stage[0],
        n_clusters_per_stage=n_clusters_per_stage,
        k_nearest=defaults.get("k_nearest", 4),
        clusters_per_fig=defaults.get("clusters_per_fig", 1),
        positions=None,
        save_dir=str(save_dir),
        m_inference=defaults.get("m_inference", 2),
        inference_seed=defaults.get("inference_seed", 42),
        save_cluster_representatives=defaults.get("save_cluster_representatives", False),
        model_args=model_args,
        data_root=str(ROOT / "data"),
        mode=defaults.get("kmeans_mode", "token"),
        heads=None,
        analysis_batch_size=run_cfg.get("batch_size", defaults.get("batch_size", 32)),
        model_name=model_name,
        gradcam_top_k=defaults.get("gradcam_trace_top_k", 5),
        trace_block=defaults.get("gradcam_trace_block", "last"),
        save_gradcam_trace=defaults.get("save_gradcam_trace", True),
        trace_max_rows_per_fig=defaults.get("trace_max_rows_per_fig", 12),
        trace_expansions_per_fig=defaults.get("trace_expansions_per_fig", 2),
        save_all_inference_repr=defaults.get("save_all_inference_repr", True),
        analysis_split="train",
        inference_split="test",
        use_kmeans_cache=defaults.get("kmeans_use_cache", True),
    )
    elapsed = time.time() - start

    gradcam_trace_dir = save_dir / "inference" / "gradcam_trace"
    return {
        "dataset": dataset,
        "exp_id": exp_id,
        "clusters_per_stage": str(clusters_per_stage),
        "checkpoint_dir": checkpoint_dir,
        "model_path": model_path,
        "save_dir": str(save_dir),
        "gradcam_trace_dir": str(gradcam_trace_dir),
        "elapsed_min": round(elapsed / 60, 1),
        "returncode": 0,
    }


def _iter_jobs(
    experiments: list[tuple[str, str, list[int]]],
    only_datasets: set[str] | None,
):
    for dataset, exp_id, clusters_per_stage in experiments:
        if only_datasets and dataset not in only_datasets:
            continue
        yield dataset, exp_id, clusters_per_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="依序對各資料集跑 K-means（stage 0~3 各自設定 cluster 數）"
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="只跑指定資料集，可重複指定多次",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="從第 N 個 job 開始（0-based，斷點續跑）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只列出將執行的 job，不實際跑 K-means",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(ROOT))

    import config as cfg_module

    defaults = dict(cfg_module.config)
    only_datasets = set(args.datasets) if args.datasets else None

    jobs = list(_iter_jobs(EXPERIMENTS, only_datasets))
    if args.start_from > 0:
        jobs = jobs[args.start_from:]

    if not jobs:
        print("沒有符合條件的 job。")
        return

    print(f"共 {len(jobs)} 個 K-means job：")
    for i, (dataset, exp_id, clusters_per_stage) in enumerate(jobs):
        save_dir = _build_save_dir(dataset, exp_id, clusters_per_stage)
        print(
            f"  [{i}] {dataset} {exp_id} "
            f"stages={clusters_per_stage} -> {save_dir}"
        )

    if args.dry_run:
        print("\n--dry-run：結束。")
        return

    summary_rows: list[dict] = []
    for dataset, exp_id, clusters_per_stage in jobs:
        nc_slug = _clusters_slug(clusters_per_stage)
        log_file = LOG_DIR / f"{dataset}_kmeans_{nc_slug}_{datetime.now():%Y%m%d_%H%M}.log"
        print(f"\nlog → {log_file}")

        header = (
            f"# dataset={dataset} exp_id={exp_id} "
            f"clusters_per_stage={clusters_per_stage}\n"
            f"# save_dir={_build_save_dir(dataset, exp_id, clusters_per_stage)}\n"
        )

        with open(log_file, "w", encoding="utf-8") as log_fp:
            log_fp.write(header)
            log_fp.flush()
            old_stdout = sys.stdout
            sys.stdout = _Tee(old_stdout, _FilteredLogStream(log_fp))
            try:
                row = run_one_kmeans(dataset, exp_id, clusters_per_stage, defaults)
                row["error"] = ""
            except Exception as exc:
                row = {
                    "dataset": dataset,
                    "exp_id": exp_id,
                    "clusters_per_stage": str(clusters_per_stage),
                    "checkpoint_dir": f"runs/train/{exp_id}",
                    "model_path": "N/A",
                    "save_dir": str(_build_save_dir(dataset, exp_id, clusters_per_stage)),
                    "gradcam_trace_dir": "N/A",
                    "elapsed_min": 0.0,
                    "returncode": 1,
                    "error": str(exc),
                }
                print(f"✗ 失敗: {exc}")
            finally:
                sys.stdout = old_stdout
                log_fp.flush()
                footer = (
                    f"# returncode={row['returncode']} elapsed_min={row['elapsed_min']}\n"
                )
                if row.get("error"):
                    footer += f"# error={row['error']}\n"
                log_fp.write(footer)
                log_fp.flush()

        row["log_file"] = str(log_file)

        summary_rows.append(row)
        status = "✓" if row["returncode"] == 0 else "✗"
        print(
            f"{status} {dataset} {exp_id} stages={clusters_per_stage} "
            f"耗時 {row['elapsed_min']} 分鐘"
        )

    summary_path = LOG_DIR / "kmeans_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n全部完成，彙總見 {summary_path}")


if __name__ == "__main__":
    main()
