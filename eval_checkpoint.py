"""
用來直接跑 checkpoint 的評估，不用跑完整的 train.py
可印出 accuracy, precision, recall, f1-score, balanced accuracy 等 metrics

Usage:
python eval_checkpoint.py

Edit EXP_DIR to specify the experiment folder.
"""

import importlib.util
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, balanced_accuracy_score)
from tqdm.autonotebook import tqdm

import models
from dataloader import get_dataloader
from loss.loss_function import get_loss_function

# ─── 設定這裡 ────────────────────────────────────────────
EXP_DIR = "runs/train/exp106"
# ─────────────────────────────────────────────────────────


def load_config_from_dir(exp_dir: str):
    config_path = Path(exp_dir) / 'config.py'
    spec = importlib.util.spec_from_file_location('exp_config', config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.config, mod.arch


def print_multiclass_metrics(targets, preds):
    targets = np.asarray(targets)
    preds   = np.asarray(preds)
    acc  = accuracy_score(targets, preds)
    prec = precision_score(targets, preds, average='macro', zero_division=0)
    rec  = recall_score(targets, preds, average='macro', zero_division=0)
    f1   = f1_score(targets, preds, average='macro', zero_division=0)
    bal  = balanced_accuracy_score(targets, preds)
    print(f"\nAccuracy      : {acc:.4f}")
    print(f"Precision     : {prec:.4f} (macro)")
    print(f"Recall        : {rec:.4f} (macro)")
    print(f"F1-score      : {f1:.4f} (macro)")
    print(f"Balanced Acc. : {bal:.4f}\n")


def run_eval(exp_dir: str):
    config, arch = load_config_from_dir(exp_dir)
    device = config['device']

    # 強制用專案根目錄，不用 exp 資料夾的 root
    project_root = str(Path(__file__).resolve().parent)
    config['root'] = project_root

    # 找 checkpoint
    ckpt_path = Path(exp_dir) / f"{arch['name']}_best.pth"
    if not ckpt_path.exists():
        ckpt_path = Path(exp_dir) / 'best_epoch.pth'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{exp_dir}")

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model_weights', ckpt)
    print(f"✓ 載入 checkpoint: {ckpt_path}")
    if 'best_epoch' in ckpt:
        print(f"  best_epoch={ckpt['best_epoch']}")
        print(f"  train_acc={ckpt.get('train_acc', '?'):.4f}, train_loss={ckpt.get('train_loss', '?'):.4f}")
        print(f"  val_acc={ckpt.get('valid_acc', '?'):.4f},   val_loss={ckpt.get('valid_loss', '?'):.4f}")

    # 建立模型
    model = getattr(getattr(models, arch['name']), arch['name'])(**dict(arch['args']))
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    # Dataloader
    _, test_loader = get_dataloader(
        dataset=config['dataset'],
        root=config['root'] + '/data/',
        batch_size=config['batch_size'],
        input_size=config['input_shape'],
    )

    loss_fn = get_loss_function(config['loss_fn'])

    all_targets, all_preds = [], []
    losses, correct, size = 0, 0, 0

    with torch.no_grad():
        for batch, (X, y) in tqdm(enumerate(test_loader), total=len(test_loader)):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            losses += loss.item()
            size += len(X)

            targets = y if y.dim() == 1 else y.argmax(1)
            preds   = pred.argmax(1)
            correct += (preds == targets).float().sum().item()
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    print(f"\nTest Accuracy : {correct/size:.4f}")
    print(f"Test Avg Loss : {losses/(batch+1):.6f}")
    print_multiclass_metrics(all_targets, all_preds)


if __name__ == '__main__':
    run_eval(EXP_DIR)