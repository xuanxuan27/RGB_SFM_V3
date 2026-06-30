"""
save_bloodmnist_examples.py
每個類別各取 N 張範例圖，存成獨立 PNG 與一張整合對照表。

用法（在 /home/xuan/RGB_SFM_V3/ 下執行）：
    python save_bloodmnist_examples.py
    python save_bloodmnist_examples.py --n 3 --out plots/bloodmnist_examples
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader

# 讓 import dataloader 找得到
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataloader import get_dataloader

CLASS_NAMES = {
    0: "嗜鹼性球",
    1: "嗜酸性粒細胞",
    2: "成紅血球",
    3: "未成熟粒細胞",
    4: "淋巴細胞",
    5: "單核細胞",
    6: "中性粒細胞",
    7: "血小板",
}
DISPLAY_CLASS_NAMES = {
    0: "Basophil",
    1: "Eosinophil",
    2: "Erythroblast",
    3: "Immature granulocytes",
    4: "Lymphocyte",
    5: "Monocyte",
    6: "Neutrophil",
    7: "Platelet",
}
NUM_CLASSES = 8


def label_to_class(label) -> int:
    """支援 scalar label 與 one-hot label。"""
    if hasattr(label, "detach"):
        label = label.detach().cpu().numpy()

    arr = np.asarray(label).squeeze()
    if arr.size == 1:
        return int(arr.item())
    return int(arr.argmax())


def collect_examples(loader: DataLoader, n_per_class: int) -> dict[int, list[np.ndarray]]:
    """從 dataloader 收集每個類別各 n 張原始像素圖（uint8, HWC）。"""
    collected: dict[int, list[np.ndarray]] = {c: [] for c in range(NUM_CLASSES)}
    needed = set(range(NUM_CLASSES))

    for imgs, labels in loader:
        for img, label in zip(imgs, labels):
            c = label_to_class(label)
            if c in needed and len(collected[c]) < n_per_class:
                # img: Tensor CHW, float [0,1] 或 [-1,1] — 還原到 uint8
                arr = img.detach().cpu().permute(1, 2, 0).numpy()
                arr = np.clip(arr, 0.0, 1.0)
                arr = (arr * 255).astype(np.uint8)
                collected[c].append(arr)
                if len(collected[c]) >= n_per_class:
                    needed.discard(c)
        if not needed:
            break

    return collected


def save_individual(collected: dict, out_dir: Path) -> None:
    """每張圖單獨存 PNG。"""
    ind_dir = out_dir / "individual"
    ind_dir.mkdir(parents=True, exist_ok=True)
    for c, imgs in collected.items():
        for i, arr in enumerate(imgs):
            path = ind_dir / f"class{c}_{CLASS_NAMES[c]}_ex{i}.png"
            plt.imsave(str(path), arr)
    print(f"  個別圖片已存至 {ind_dir}/")


def save_grid(collected: dict, out_dir: Path, n_per_class: int) -> None:
    """
    輸出類似學長論文表格的對照圖：
    欄 = Label index / Label / Example，列 = 類別。
    """
    index_w = 1.4
    label_w = 2.7
    example_w = max(3.2, n_per_class * 1.05)
    row_h = 1.05
    header_h = 0.85
    total_w = index_w + label_w + example_w
    total_h = header_h + NUM_CLASSES * row_h

    fig, ax = plt.subplots(figsize=(total_w, total_h))
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    def draw_cell(x: float, y: float, width: float, height: float, linewidth: float = 1.2) -> None:
        ax.add_patch(
            mpatches.Rectangle(
                (x, y),
                width,
                height,
                facecolor="white",
                edgecolor="black",
                linewidth=linewidth,
            )
        )

    columns = [
        ("Label\nindex", 0.0, index_w),
        ("Label", index_w, label_w),
        ("Example", index_w + label_w, example_w),
    ]

    header_y = total_h - header_h
    for title, x, width in columns:
        draw_cell(x, header_y, width, header_h, linewidth=1.4)
        ax.text(
            x + width / 2,
            header_y + header_h / 2,
            title,
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            fontfamily="serif",
        )

    for r, c in enumerate(range(NUM_CLASSES)):
        y = total_h - header_h - (r + 1) * row_h
        draw_cell(0.0, y, index_w, row_h)
        draw_cell(index_w, y, label_w, row_h)
        draw_cell(index_w + label_w, y, example_w, row_h)

        ax.text(
            index_w / 2,
            y + row_h / 2,
            str(c),
            ha="center",
            va="center",
            fontsize=14,
            fontfamily="serif",
        )
        ax.text(
            index_w + label_w / 2,
            y + row_h / 2,
            DISPLAY_CLASS_NAMES[c],
            ha="center",
            va="center",
            fontsize=13,
            fontfamily="sans-serif",
        )

        imgs = collected[c]
        if not imgs:
            continue

        img_size = min(row_h * 0.72, example_w / max(n_per_class + 0.7, 1))
        gap = (example_w - img_size * n_per_class) / (n_per_class + 1)
        img_y0 = y + (row_h - img_size) / 2

        for i, arr in enumerate(imgs[:n_per_class]):
            img_x0 = index_w + label_w + gap + i * (img_size + gap)
            ax.imshow(
                arr,
                extent=(img_x0, img_x0 + img_size, img_y0, img_y0 + img_size),
                interpolation="nearest",
                zorder=2,
            )

    plt.tight_layout()
    grid_path = out_dir / "bloodmnist_grid.png"
    fig.savefig(str(grid_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  對照表已存至 {grid_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3, help="每類別取幾張（預設 3）")
    parser.add_argument("--out", type=str, default="plots/bloodmnist_examples")
    parser.add_argument("--img-size", type=int, default=28)
    parser.add_argument("--data-root", type=str, default="./data")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"載入 BloodMNIST（{args.img_size}×{args.img_size}）…")
    train_loader, _ = get_dataloader(
        dataset="BloodMNIST",
        root=args.data_root,
        batch_size=64,
        input_size=(args.img_size, args.img_size),
    )

    print(f"每類別各取 {args.n} 張…")
    collected = collect_examples(train_loader, args.n)

    save_individual(collected, out_dir)
    save_grid(collected, out_dir, args.n)

    print("完成！")


if __name__ == "__main__":
    main()