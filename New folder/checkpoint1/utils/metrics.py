"""Basic metric helpers for Assignment 5."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def top1_accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def compute_accuracy(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def compute_confusion_matrix(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(10)))


def per_class_accuracy(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> dict[str, float]:
    cm = compute_confusion_matrix(y_true, y_pred)
    result: dict[str, float] = {}
    for i, cls_name in enumerate(CIFAR10_CLASSES):
        denom = cm[i].sum()
        result[cls_name] = float(cm[i, i] / denom) if denom else 0.0
    return result


def save_confusion_matrix(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    out_path: str | Path,
    title: str = "Confusion Matrix",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cm = compute_confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CIFAR10_CLASSES)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
