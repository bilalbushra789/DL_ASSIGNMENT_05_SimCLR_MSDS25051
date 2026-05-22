"""Visualization helpers for Assignment 5."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def denormalize_cifar10(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616], device=tensor.device).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)


def save_augmentation_grid(originals, view1s, view2s, out_path: str | Path, max_rows: int = 10) -> None:
    """Save a grid: Original | View 1 | View 2."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = min(max_rows, len(originals))
    fig, axes = plt.subplots(rows, 3, figsize=(6, 2 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    for r in range(rows):
        imgs = [originals[r], view1s[r], view2s[r]]
        titles = ["Original", "View 1", "View 2"]
        for c in range(3):
            img = imgs[c]
            if isinstance(img, torch.Tensor):
                if img.ndim == 3 and img.shape[0] == 3:
                    img = denormalize_cifar10(img.detach().cpu()).permute(1, 2, 0).numpy()
            axes[r, c].imshow(img)
            axes[r, c].set_title(titles[c])
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_2d_feature_plot(
    features: np.ndarray,
    labels: np.ndarray,
    out_path: str | Path,
    method: str = "pca",
    title: str = "Feature Visualization",
    seed: int = 2026,
) -> None:
    """Save PCA or t-SNE visualization of features.

    Labels should be used only for coloring the plot, not for SSL training.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    method = method.lower()
    if method == "pca":
        coords = PCA(n_components=2, random_state=seed).fit_transform(features)
    elif method in {"tsne", "t-sne"}:
        coords = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30, random_state=seed).fit_transform(features)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=8, alpha=0.75)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(scatter, ax=ax, ticks=range(10))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
