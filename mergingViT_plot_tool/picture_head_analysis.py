"""
從 inference 產生的 *_head_labels.json 做離散層級的 head 分析（不需重跑模型）。

輸出圖表至 mergingViT_plot_tool/plot_head_analysis/<json_stem>/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 專案根目錄
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_head_labels_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    H, W = int(data["H"]), int(data["W"])
    num_heads = int(data["num_heads"])
    n_pos = H * W
    rows = data["labels"]
    if len(rows) != n_pos * num_heads:
        raise ValueError(
            f"labels 筆數 {len(rows)} != H*W*num_heads = {n_pos * num_heads} ({path})"
        )
    mat = np.zeros((n_pos, num_heads), dtype=np.int64)
    for item in rows:
        p, h = int(item["pos"]), int(item["head"])
        mat[p, h] = int(item["cluster"])
    return {
        "H": H,
        "W": W,
        "num_heads": num_heads,
        "mat": mat,
        "path": path,
    }


def head_head_agreement(mat: np.ndarray) -> np.ndarray:
    """mat: [n_pos, num_heads] -> agreement[i,j] = 位置平均一致率"""
    Hh = mat.shape[1]
    out = np.zeros((Hh, Hh), dtype=np.float64)
    for i in range(Hh):
        for j in range(Hh):
            out[i, j] = float(np.mean(mat[:, i] == mat[:, j]))
    return out


def unique_clusters_per_position(mat: np.ndarray) -> np.ndarray:
    """每個 position 上各 head 所指派的 cluster 有幾種不同值。"""
    return np.array([len(np.unique(mat[p])) for p in range(mat.shape[0])])


def neighbor_same_rate_2d(grid: np.ndarray, H: int, W: int) -> float:
    """
    grid: [H, W] cluster ids。
    只比較右鄰與下鄰，每條無向邊只算一次。
    """
    same = 0
    total = 0
    for r in range(H):
        for c in range(W):
            if c + 1 < W:
                total += 1
                if grid[r, c] == grid[r, c + 1]:
                    same += 1
            if r + 1 < H:
                total += 1
                if grid[r, c] == grid[r + 1, c]:
                    same += 1
    return float(same / total) if total else 0.0


def discrete_entropy(labels: np.ndarray) -> float:
    """自然對數熵；labels 為一維整數。"""
    _, counts = np.unique(labels, return_counts=True)
    p = counts.astype(np.float64) / counts.sum()
    return float(-np.sum(p * np.log(p + 1e-20)))


def analyze_one(
    payload: dict,
    out_dir: Path,
    dpi: int = 150,
) -> dict:
    H, W = payload["H"], payload["W"]
    num_heads = payload["num_heads"]
    mat = payload["mat"]
    stem = payload["path"].stem

    out_dir = out_dir / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="notebook")
    summary: dict = {
        "source_json": str(payload["path"]),
        "H": H,
        "W": W,
        "num_heads": num_heads,
    }

    # 1) Head × Head 一致率熱圖
    agree = head_head_agreement(mat)
    summary["head_head_agreement_mean_offdiag"] = float(
        (np.sum(agree) - np.trace(agree)) / (num_heads * (num_heads - 1))
    )

    fig, ax = plt.subplots(figsize=(max(5, num_heads * 0.6), max(4, num_heads * 0.55)))
    sns.heatmap(
        agree,
        vmin=0,
        vmax=1,
        cmap="viridis",
        square=True,
        annot=num_heads <= 12,
        fmt=".2f",
        cbar_kws={"label": "P(same cluster)"},
        ax=ax,
    )
    ax.set_xlabel("head j")
    ax.set_ylabel("head i")
    ax.set_title(f"{stem}\nHead–head cluster agreement (same K-means label)")
    plt.tight_layout()
    p_agree = out_dir / "head_head_agreement.png"
    fig.savefig(p_agree, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # 2) 每個 position 上「幾種不同 cluster」
    nuniq = unique_clusters_per_position(mat)
    summary["unique_clusters_per_pos_mean"] = float(np.mean(nuniq))
    summary["unique_clusters_per_pos_std"] = float(np.std(nuniq))
    summary["frac_pos_all_heads_same_cluster"] = float(np.mean(nuniq == 1))

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(0.5, num_heads + 1.5, 1)
    ax.hist(nuniq, bins=bins, edgecolor="black", align="mid")
    ax.set_xlabel("Number of distinct cluster ids across heads (per spatial position)")
    ax.set_ylabel("Count (positions)")
    ax.set_title(f"{stem}\nCross-head label diversity per position")
    ax.set_xticks(range(1, num_heads + 1))
    plt.tight_layout()
    p_div = out_dir / "position_label_diversity.png"
    fig.savefig(p_div, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # 3) 每個 head：空間鄰居一致率
    neigh_rates = []
    for h in range(num_heads):
        grid = mat[:, h].reshape(H, W)
        neigh_rates.append(neighbor_same_rate_2d(grid, H, W))
    summary["neighbor_same_rate_per_head"] = neigh_rates

    fig, ax = plt.subplots(figsize=(max(6, num_heads * 0.45), 4))
    x = np.arange(num_heads)
    ax.bar(x, neigh_rates, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xlabel("head")
    ax.set_ylabel("Fraction of edges with same cluster")
    ax.set_ylim(0, 1)
    ax.set_title(f"{stem}\nSpatial coherence (right/down neighbors)")
    plt.tight_layout()
    p_neigh = out_dir / "spatial_neighbor_agreement_per_head.png"
    fig.savefig(p_neigh, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # 4) 每個 head 的 cluster 使用熵（跨位置）
    entropies = [discrete_entropy(mat[:, h]) for h in range(num_heads)]
    summary["cluster_usage_entropy_per_head"] = entropies

    fig, ax = plt.subplots(figsize=(max(6, num_heads * 0.45), 4))
    ax.bar(x, entropies, color="seagreen", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xlabel("head")
    ax.set_ylabel("Entropy (nat) of cluster ids over positions")
    ax.set_title(f"{stem}\nHow spread out cluster assignments are per head")
    plt.tight_layout()
    p_ent = out_dir / "cluster_entropy_per_head.png"
    fig.savefig(p_ent, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def collect_json_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if not input_path.name.endswith(".json"):
            raise ValueError(f"預期 .json 檔案: {input_path}")
        return [input_path]
    if input_path.is_dir():
        paths = sorted(input_path.glob("*head_labels.json"))
        if not paths:
            paths = sorted(input_path.glob("*.json"))
        if not paths:
            raise FileNotFoundError(f"目錄內找不到 *head_labels.json 或 .json: {input_path}")
        return paths
    raise FileNotFoundError(input_path)


def main():
    default_out = Path(__file__).resolve().parent / "plot_head_analysis"
    parser = argparse.ArgumentParser(description="分析 *_head_labels.json 並輸出圖表")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="單一 *_head_labels.json 或含多個 json 的目錄（預設 glob *head_labels.json）",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_out,
        help=f"輸出根目錄（預設: {default_out}）",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    paths = collect_json_paths(args.input.resolve())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for p in paths:
        print(f"分析: {p}")
        payload = load_head_labels_json(p)
        summ = analyze_one(payload, args.out_dir, dpi=args.dpi)
        all_summaries.append(summ)
        print(f"  -> {args.out_dir / p.stem}")

    if len(all_summaries) > 1:
        batch_path = args.out_dir / "batch_summary.json"
        with open(batch_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, ensure_ascii=False, indent=2)
        print(f"已寫入彙總: {batch_path}")


if __name__ == "__main__":
    main()
