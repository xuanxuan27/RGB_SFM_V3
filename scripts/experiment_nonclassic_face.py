#!/usr/bin/env python3
"""
依序用不同模型訓練 NonclassicFace（FaceVsNonFace）。

會沿用當前 config.py 的訓練超參數（lr、epoch、batch_size…），
只覆寫 name / tags / dataset / model（arch）。
"""
from __future__ import annotations

import csv
import os
import pty
import re
import select
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.py"
LOG_DIR = ROOT / "overnight_logs"
RUNS_DIR = ROOT / "runs" / "train"
LOG_DIR.mkdir(exist_ok=True)

DATASET = "NonclassicFace"
NUM_CLASSES = 2

# (wandb / tags 顯示名稱, config arch["name"])
MODELS = [
    ("DenseNet", "DenseNet"),
    ("ResNet", "ResNet"),
    ("ViT", "VIT"),
    ("PVTv2", "PVTv2"),
    ("Swin_tiny", "Swin_tiny"),
]


def build_arch_block(arch_name: str) -> str:
    """依模型名稱產生 config.py 用的 arch 區塊（num_classes / out_channels = 2）。"""
    if arch_name == "DenseNet":
        return """arch = {
    "name": 'DenseNet',
    "need_calculate_status" : False,
    "args":{
        'in_channels':3,
        "out_channels": 2
    }
}"""
    if arch_name == "ResNet":
        return """arch = {
    "name": 'ResNet',
    "need_calculate_status" : False,
    "args":{
        'layers':18,
        'in_channels':3,
        "out_channels": 2
    }
}"""
    if arch_name == "VIT":
        return """arch = {
    "name": 'VIT',
    "need_calculate_status": False,
    "args": {
        "in_channels": 3,
        "num_classes": 2,
        "model_name": "vit_tiny_patch16_224",
        "pretrained": False,
        "img_size": 224,
        "drop_rate": 0.1,
        "drop_path_rate": 0.3,
        "auto_resize": False,
    }
}"""
    if arch_name == "PVTv2":
        return """arch = {
    "name": 'PVTv2',
    "need_calculate_status": False,
    "args": {
        "in_channels": 3,
        "num_classes": 2,
        "model_name": "pvt_v2_b0",
        "pretrained": False,
        "drop_rate": 0.1,
        "drop_path_rate": 0.3,
        "auto_resize": False,
        "min_input_size": 32,
    }
}"""
    if arch_name == "Swin_tiny":
        return """arch = {
    "name": 'Swin_tiny',
    "need_calculate_status": False,
    "args": {
        "in_channels": 3,
        "out_channels": 2,
    }
}"""
    raise ValueError(f"未知模型: {arch_name}")


def replace_active_arch(text: str, new_arch_block: str) -> str:
    """替換 config.py 中未註解的 arch = { ... } 區塊。"""
    lines = text.splitlines(keepends=True)
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^arch\s*=\s*\{", line):
            start = i
            break
    if start is None:
        raise RuntimeError("找不到 active arch = {...} 區塊")

    depth = 0
    end = None
    for j in range(start, len(lines)):
        depth += lines[j].count("{") - lines[j].count("}")
        if depth == 0:
            end = j
            break
    if end is None:
        raise RuntimeError("arch 區塊大括號不平衡")

    block = new_arch_block if new_arch_block.endswith("\n") else new_arch_block + "\n"
    return "".join(lines[:start]) + block + "".join(lines[end + 1 :])


def patch_config(model_label: str, arch_name: str) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    text = re.sub(
        r"^name\s*=\s*'[^']*'",
        f"name = '{model_label}_NonclassicFace'",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^tags\s*=\s*\[[^\]]*\]",
        f"tags = ['{model_label}', 'NonclassicFace']",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r'^group\s*=\s*["\'][^"\']*["\']',
        'group = "NonclassicFace"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r'"dataset":\s*\'[^\']*\'',
        f'"dataset": \'{DATASET}\'',
        text,
    )
    text = re.sub(
        r"^load_model_name\s*=\s*'[^']*'",
        f"load_model_name = '{arch_name}_best'",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text = replace_active_arch(text, build_arch_block(arch_name))
    CONFIG_PATH.write_text(text, encoding="utf-8")


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_PROGRESS_RE = re.compile(
    r"^(?:"
    r"Loss:\s|"
    r"-{5,}EPOCH\s+\d+|"
    r"Test Loss:\s|"
    r".*\d+%\|"
    r")"
)


def normalize_visible(raw: str) -> str:
    text = _ANSI_RE.sub("", raw)
    if "\r" in text:
        if text.endswith("\r") and text.count("\r") == 1:
            text = text[:-1]
        else:
            text = text.split("\r")[-1]
    return text.rstrip("\n")


def keep_for_log(raw: str) -> bool:
    line = normalize_visible(raw).strip()
    if not line:
        return False
    if _PROGRESS_RE.match(line):
        return False
    return True


def list_exp_dirs() -> set[str]:
    if not RUNS_DIR.is_dir():
        return set()
    return {p.name for p in RUNS_DIR.glob("exp*") if p.is_dir()}


def detect_new_save_dir(before: set[str]) -> str:
    after = list_exp_dirs()
    created = after - before
    if not created:
        return "N/A"

    def exp_key(name: str) -> int:
        m = re.search(r"exp(\d+)$", name)
        return int(m.group(1)) if m else -1

    newest = max(created, key=exp_key)
    return f"runs/train/{newest}"


def parse_save_dir(log_text: str, fallback: str = "N/A") -> str:
    m = re.search(r"^save_dir:\s*(runs/train/exp\d*)", log_text, re.MULTILINE)
    if m:
        return m.group(1)
    matches = re.findall(r"runs/train/exp\d+", log_text)
    return matches[-1] if matches else fallback


def parse_best_acc(log_text: str) -> str:
    acc_matches = re.findall(r"[Bb]est\s+epoch:.*?val_acc:\s*(\d+\.\d+)", log_text)
    if acc_matches:
        return acc_matches[-1]
    m = re.findall(r"^Valid:\s*\n\s*Accuracy:\s*([0-9.]+)", log_text, re.MULTILINE)
    if m:
        return m[-1]
    acc_matches = re.findall(r"val_acc:\s*(\d+\.\d+)", log_text)
    return acc_matches[-1] if acc_matches else "N/A"


def extract_final_summary(log_text: str) -> str:
    lines = log_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("Train:"):
            start = i
    if start is None:
        return ""

    end = len(lines)
    for i in range(start, len(lines)):
        if lines[i].startswith("checkpoint keys:"):
            end = i + 1
            break
    block = "\n".join(lines[start:end]).strip()
    return block + "\n" if block else ""


def parse_metrics(log_text: str) -> dict:
    metrics = {
        "test_acc": "N/A",
        "precision": "N/A",
        "recall": "N/A",
        "f1": "N/A",
        "balanced_acc": "N/A",
        "train_acc": "N/A",
        "valid_acc": "N/A",
    }
    m = re.search(
        r"Train:\s*\n\s*Accuracy:\s*([0-9.]+).*?"
        r"Valid:\s*\n\s*Accuracy:\s*([0-9.]+)",
        log_text,
        re.DOTALL,
    )
    if m:
        metrics["train_acc"] = m.group(1)
        metrics["valid_acc"] = m.group(2)

    blocks = list(
        re.finditer(
            r"Metrics \(macro-average\):\s*\n"
            r"Accuracy\s*:\s*([0-9.]+)\s*\n"
            r"Precision\s*:\s*([0-9.]+)\s*\n"
            r"Recall\s*:\s*([0-9.]+)\s*\n"
            r"F1-score\s*:\s*([0-9.]+)\s*\n"
            r"Balanced Acc\.\s*:\s*([0-9.]+)",
            log_text,
        )
    )
    if blocks:
        b = blocks[-1]
        metrics["test_acc"] = b.group(1)
        metrics["precision"] = b.group(2)
        metrics["recall"] = b.group(3)
        metrics["f1"] = b.group(4)
        metrics["balanced_acc"] = b.group(5)
    return metrics


def strip_wandb_noise(log_text: str) -> str:
    lines = log_text.splitlines()
    out = []
    skip_history = False
    for line in lines:
        if line.startswith("wandb: Run history:"):
            skip_history = True
            continue
        if skip_history:
            if line.startswith("wandb: Run summary:"):
                skip_history = False
                out.append(line)
            elif line.startswith("wandb:") and line.strip() == "wandb:":
                continue
            elif line.startswith("wandb:"):
                continue
            else:
                skip_history = False
                out.append(line)
            continue
        if line.startswith("wandb: Using wandb-core"):
            continue
        if line.startswith("wandb: Currently logged in"):
            continue
        if line.startswith("wandb: Tracking run"):
            continue
        if line.startswith("wandb: Run data is saved"):
            continue
        if line.startswith("wandb: Run `wandb offline`"):
            continue
        if line.startswith("wandb: Syncing run"):
            continue
        if line.startswith("wandb: WARNING"):
            continue
        if line.startswith("wandb: Synced "):
            continue
        if line.startswith("wandb: Find logs at:"):
            continue
        out.append(line)
    return "\n".join(out).rstrip() + "\n"


def _flush_log_lines(buf: bytes, log_fp) -> bytes:
    while b"\n" in buf:
        line_b, buf = buf.split(b"\n", 1)
        line = line_b.decode("utf-8", errors="ignore")
        if keep_for_log(line):
            log_fp.write(normalize_visible(line) + "\n")
            log_fp.flush()
    if b"\r" in buf:
        buf = buf.split(b"\r")[-1]
    return buf


def run_train_tee(log_fp) -> int:
    master_fd, slave_fd = pty.openpty()
    proc = subprocess.Popen(
        ["conda", "run", "-n", "SFM", "--no-capture-output", "python", "-u", "train.py"],
        cwd=ROOT,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
    )
    os.close(slave_fd)

    buf = b""
    try:
        while True:
            finished = proc.poll() is not None
            timeout = 0.0 if finished else 0.1
            r, _, _ = select.select([master_fd], [], [], timeout)
            if r:
                try:
                    chunk = os.read(master_fd, 4096)
                except OSError:
                    chunk = b""
                if chunk:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                    buf += chunk
                    buf = _flush_log_lines(buf, log_fp)
                elif finished:
                    break
            elif finished:
                break
        if buf and keep_for_log(buf.decode("utf-8", errors="ignore")):
            log_fp.write(normalize_visible(buf.decode("utf-8", errors="ignore")) + "\n")
            log_fp.flush()
    finally:
        os.close(master_fd)

    return proc.wait()


def main():
    backup_path = CONFIG_PATH.with_suffix(".py.bak_nonclassic_face")
    shutil.copyfile(CONFIG_PATH, backup_path)
    print(f"已備份 config.py → {backup_path.name}")

    summary_rows = []
    try:
        for model_label, arch_name in MODELS:
            print(
                f"\n{'=' * 60}\n"
                f"開始訓練: {model_label} on {DATASET}\n"
                f"name='{model_label}_NonclassicFace', "
                f"tags=['{model_label}', 'NonclassicFace']\n"
                f"{'=' * 60}"
            )
            patch_config(model_label, arch_name)

            log_file = LOG_DIR / f"{model_label}_{DATASET}_{datetime.now():%Y%m%d_%H%M}.log"
            print(f"log → {log_file}（已過濾 tqdm / epoch loss）")
            before_exps = list_exp_dirs()
            start = time.time()
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(
                    f"# model={model_label} arch={arch_name} dataset={DATASET} "
                    f"num_classes={NUM_CLASSES}\n"
                )
                f.flush()
                result_code = run_train_tee(f)
            elapsed = time.time() - start

            fallback_save = detect_new_save_dir(before_exps)
            log_text = log_file.read_text(encoding="utf-8", errors="ignore")
            body = re.sub(r"^# model=.*\n", "", log_text, count=1)
            body = strip_wandb_noise(body)
            save_dir = parse_save_dir(body, fallback=fallback_save)
            best_acc = parse_best_acc(body)
            metrics = parse_metrics(body)
            if best_acc == "N/A" and metrics["valid_acc"] != "N/A":
                best_acc = metrics["valid_acc"]
            final_summary = extract_final_summary(body)

            header = (
                f"# model={model_label} arch={arch_name} dataset={DATASET} "
                f"num_classes={NUM_CLASSES}\n"
                f"# save_dir={save_dir}\n"
                f"# best_acc={best_acc} test_acc={metrics['test_acc']} "
                f"precision={metrics['precision']} recall={metrics['recall']} "
                f"f1={metrics['f1']} balanced_acc={metrics['balanced_acc']}\n"
                f"# returncode={result_code} elapsed_min={round(elapsed / 60, 1)}\n"
            )
            summary_section = ""
            if final_summary:
                summary_section = (
                    "# ===== FINAL SUMMARY =====\n"
                    f"{final_summary}"
                    "# ===== END SUMMARY =====\n\n"
                )
            log_file.write_text(header + summary_section + body, encoding="utf-8")

            summary_rows.append(
                {
                    "model": model_label,
                    "arch": arch_name,
                    "dataset": DATASET,
                    "save_dir": save_dir,
                    "elapsed_min": round(elapsed / 60, 1),
                    "best_acc": best_acc,
                    "train_acc": metrics["train_acc"],
                    "valid_acc": metrics["valid_acc"],
                    "test_acc": metrics["test_acc"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "balanced_acc": metrics["balanced_acc"],
                    "returncode": result_code,
                    "log_file": str(log_file),
                }
            )
            print(
                f"✓ {model_label} 完成，{save_dir}，耗時 {elapsed / 60:.1f} 分鐘，"
                f"best_acc={best_acc}, test_acc={metrics['test_acc']}, "
                f"recall={metrics['recall']}, f1={metrics['f1']}"
            )
    finally:
        shutil.copyfile(backup_path, CONFIG_PATH)
        print(f"\n已還原 config.py（來自 {backup_path.name}）")

    if summary_rows:
        summary_csv = LOG_DIR / "summary_nonclassic_face.csv"
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n全部完成，彙總見 {summary_csv.relative_to(ROOT)}")
    else:
        print("\n沒有任何實驗結果可彙總")


if __name__ == "__main__":
    main()
