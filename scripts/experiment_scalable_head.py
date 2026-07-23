"""依序重跑各資料集（與 train.py / config.py 配合）"""
import os, re, subprocess, time, csv, sys, pty, select
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.py"
LOG_DIR = ROOT / "overnight_logs"
RUNS_DIR = ROOT / "runs" / "train"
LOG_DIR.mkdir(exist_ok=True)

# (dataset 名稱, num_classes, input_shape, patch_size, batch_size)
# Caltech101: 224x224 / patch 8 / batch 16；其餘: 28x28 / patch 2 / batch 256
DATASETS = [
    # ("Colored_MNIST", 30, (28, 28), 2, 256),
    # ("Colored_FashionMNIST", 30, (28, 28), 2, 256),
    # ("BloodMNIST", 8, (28, 28), 2, 256),
    ("Caltech101", 101, (224, 224), 8, 16),
]

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# tqdm batch 進度、EPOCH 標題、每 epoch 的 Test Loss/Acc
_PROGRESS_RE = re.compile(
    r"^(?:"
    r"Loss:\s|"                          # tqdm: Loss: x, Accuracy: y
    r"-{5,}EPOCH\s+\d+|"                 # ------------------------------EPOCH 0--
    r"Test Loss:\s|"                     # 每 epoch valid 結果
    r".*\d+%\|"                          # 其他進度條
    r")"
)

def patch_config(dataset_name: str, num_classes: int, input_shape: tuple, patch_size: int, batch_size: int):
    h, w = input_shape
    text = CONFIG_PATH.read_text(encoding="utf-8")
    text = re.sub(r"^name\s*=\s*'[^']*'", f"name = 'HierarchicalViT_{dataset_name}'", text, count=1, flags=re.MULTILINE)
    text = re.sub(
        r"^tags\s*=\s*\[[^\]]*\]",
        f"tags = ['HierarchicalViT', '{dataset_name}']",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text = re.sub(r'"dataset":\s*\'[^\']*\'', f'"dataset": \'{dataset_name}\'', text)
    text = re.sub(r'"num_classes":\s*\d+', f'"num_classes": {num_classes}', text)
    text = re.sub(r'"input_shape":\s*\(\s*\d+\s*,\s*\d+\s*\)', f'"input_shape": ({h}, {w})', text)
    text = re.sub(r'"img_size":\s*\d+', f'"img_size": {h}', text)
    text = re.sub(r'"patch_size":\s*\d+', f'"patch_size": {patch_size}', text)
    text = re.sub(r'"batch_size":\s*\d+', f'"batch_size": {batch_size}', text)
    CONFIG_PATH.write_text(text, encoding="utf-8")

def normalize_visible(raw: str) -> str:
    """
    去掉 ANSI；處理 PTY 的 CRLF（列尾多一個 \\r）與 tqdm 的 \\r 覆寫。
    注意：不可對「只有尾端 \\r」的列做 split('\\r')[-1]，否則會變成空字串。
    """
    text = _ANSI_RE.sub("", raw)
    if "\r" in text:
        # 純 CRLF 殘留：'content\r' → 'content'
        if text.endswith("\r") and text.count("\r") == 1:
            text = text[:-1]
        else:
            # tqdm 原地更新：'old\rnew' → 'new'
            text = text.split("\r")[-1]
    return text.rstrip("\n")

def keep_for_log(raw: str) -> bool:
    """log 只留架構 / 資料集 / exp / Best epoch / 最終 summary / metrics。"""
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
    """從 train.py 輸出解析本次 save_dir（例如 runs/train/exp246）。"""
    m = re.search(r"^save_dir:\s*(runs/train/exp\d*)", log_text, re.MULTILINE)
    if m:
        return m.group(1)
    matches = re.findall(r"runs/train/exp\d+", log_text)
    if matches:
        return matches[-1]
    return fallback

def parse_best_acc(log_text: str) -> str:
    # train.py: "✓ Best epoch: {e}, val_acc: {valid_acc:.4f}"
    acc_matches = re.findall(r"[Bb]est\s+epoch:.*?val_acc:\s*(\d+\.\d+)", log_text)
    if acc_matches:
        return acc_matches[-1]
    # 最終 Valid / Test summary
    m = re.findall(r"^Valid:\s*\n\s*Accuracy:\s*([0-9.]+)", log_text, re.MULTILINE)
    if m:
        return m[-1]
    acc_matches = re.findall(r"val_acc:\s*(\d+\.\d+)", log_text)
    return acc_matches[-1] if acc_matches else "N/A"

def extract_final_summary(log_text: str) -> str:
    """
    抽出最終 Train/Valid/Test/Metrics 區塊（從第一個 Train: 到 Test 2 之後的 metrics）。
    方便放在 log 開頭，不用翻過整份模型架構。
    """
    lines = log_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("Train:"):
            start = i
    if start is None:
        return ""

    end = len(lines)
    # 優先切到 checkpoint keys；否則保留到檔尾的 metrics / Test 區塊
    for i in range(start, len(lines)):
        if lines[i].startswith("checkpoint keys:"):
            end = i + 1
            break
    block = "\n".join(lines[start:end]).strip()
    return block + "\n" if block else ""

def parse_metrics(log_text: str) -> dict:
    """從最後一次 Metrics (macro-average) 區塊解析 precision/recall/f1 等。"""
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

    # 取最後一個完整 metrics block（通常是 Test 2）
    blocks = list(re.finditer(
        r"Metrics \(macro-average\):\s*\n"
        r"Accuracy\s*:\s*([0-9.]+)\s*\n"
        r"Precision\s*:\s*([0-9.]+)\s*\n"
        r"Recall\s*:\s*([0-9.]+)\s*\n"
        r"F1-score\s*:\s*([0-9.]+)\s*\n"
        r"Balanced Acc\.\s*:\s*([0-9.]+)",
        log_text,
    ))
    if blocks:
        b = blocks[-1]
        metrics["test_acc"] = b.group(1)
        metrics["precision"] = b.group(2)
        metrics["recall"] = b.group(3)
        metrics["f1"] = b.group(4)
        metrics["balanced_acc"] = b.group(5)
    return metrics

def strip_wandb_noise(log_text: str) -> str:
    """去掉 wandb Run history / 同步雜訊，保留 Run summary 與連結。"""
    lines = log_text.splitlines()
    out = []
    skip_history = False
    for line in lines:
        if line.startswith("wandb: Run history:"):
            skip_history = True
            continue
        if skip_history:
            # history 區塊之後遇到空 wandb: 或 Run summary 結束
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
        # 開頭冗長登入提示可留 URL、略過 Using/logged in/Tracking 等
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
    """從 buffer 切出完整列（以 \\n），過濾後寫 log，回傳殘段。"""
    while b"\n" in buf:
        line_b, buf = buf.split(b"\n", 1)
        line = line_b.decode("utf-8", errors="ignore")
        if keep_for_log(line):
            log_fp.write(normalize_visible(line) + "\n")
            log_fp.flush()
    # 只保留 tqdm \\r 覆寫的最後一段，避免 buffer 膨脹
    if b"\r" in buf:
        buf = buf.split(b"\r")[-1]
    return buf

def run_train_tee(log_fp) -> int:
    """
    用 PTY 跑 train.py：tqdm 會當成真實終端，進度條原地更新（\\r），
    同時把非進度列寫入精簡 log。
    """
    master_fd, slave_fd = pty.openpty()
    proc = subprocess.Popen(
        ["conda", "run", "-n", "SFM", "--no-capture-output",
         "python", "-u", "train.py"],
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
        # 殘段若有可留內容也寫入
        if buf and keep_for_log(buf.decode("utf-8", errors="ignore")):
            log_fp.write(normalize_visible(buf.decode("utf-8", errors="ignore")) + "\n")
            log_fp.flush()
    finally:
        os.close(master_fd)

    return proc.wait()

def main():
    summary_rows = []
    for dataset_name, num_classes, input_shape, patch_size, batch_size in DATASETS:
        print(f"\n{'='*60}\n開始訓練: {dataset_name} "
              f"(shape={input_shape}, patch={patch_size}, batch={batch_size})\n{'='*60}")
        patch_config(dataset_name, num_classes, input_shape, patch_size, batch_size)

        log_file = LOG_DIR / f"{dataset_name}_{datetime.now():%Y%m%d_%H%M}.log"
        print(f"log → {log_file}（已過濾 tqdm / epoch loss）")
        before_exps = list_exp_dirs()
        start = time.time()
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"# dataset={dataset_name} num_classes={num_classes} "
                    f"input_shape={input_shape} patch_size={patch_size} "
                    f"batch_size={batch_size}\n")
            f.flush()
            result_code = run_train_tee(f)
        elapsed = time.time() - start

        fallback_save = detect_new_save_dir(before_exps)
        log_text = log_file.read_text(encoding="utf-8", errors="ignore")
        body = re.sub(r"^# dataset=.*\n", "", log_text, count=1)
        body = strip_wandb_noise(body)
        save_dir = parse_save_dir(body, fallback=fallback_save)
        best_acc = parse_best_acc(body)
        metrics = parse_metrics(body)
        if best_acc == "N/A" and metrics["valid_acc"] != "N/A":
            best_acc = metrics["valid_acc"]
        final_summary = extract_final_summary(body)

        header = (
            f"# dataset={dataset_name} num_classes={num_classes} "
            f"input_shape={input_shape} patch_size={patch_size} "
            f"batch_size={batch_size}\n"
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

        summary_rows.append({
            "dataset": dataset_name,
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
        })
        print(f"✓ {dataset_name} 完成，{save_dir}，耗時 {elapsed/60:.1f} 分鐘，"
              f"best_acc={best_acc}, test_acc={metrics['test_acc']}, "
              f"recall={metrics['recall']}, f1={metrics['f1']}")

    with open(LOG_DIR / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print("\n全部完成，彙總見 overnight_logs/summary.csv")

if __name__ == "__main__":
    main()
