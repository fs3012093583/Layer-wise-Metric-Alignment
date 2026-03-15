from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def plot_training_curves(history: Sequence[dict], save_path: str) -> None:
    _ensure_parent(save_path)
    epochs = [row["epoch"] for row in history]
    loss = [row.get("loss_total", np.nan) for row in history]
    val_acc = [row.get("val_acc", np.nan) for row in history]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(epochs, loss, label="Loss", color="#c0392b", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="#c0392b")
    ax1.tick_params(axis="y", labelcolor="#c0392b")

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_acc, label="Val Acc", color="#1f618d", linewidth=2)
    ax2.set_ylabel("Val Accuracy", color="#1f618d")
    ax2.tick_params(axis="y", labelcolor="#1f618d")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_accuracy_stairs(values: Iterable[float], save_path: str) -> None:
    _ensure_parent(save_path)
    vals = list(values)
    x = np.arange(1, len(vals) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.step(x, vals, where="post", linewidth=2.2, color="#117a65")
    ax.set_xlabel("Stage / Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    title: str = "t-SNE Feature Map",
    max_points: int = 4000,
    random_state: int = 42,
) -> None:
    _ensure_parent(save_path)
    if len(features) > max_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(features), size=max_points, replace=False)
        features = features[idx]
        labels = labels[idx]

    emb = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca").fit_transform(features)

    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=labels, s=7, cmap="tab20", alpha=0.85)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=240)
    plt.close(fig)
