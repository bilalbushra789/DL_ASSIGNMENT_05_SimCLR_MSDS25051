"""
Task 2: Understanding Augmentations
Deep Learning Spring 2026 - Assignment 5 SimCLR
Checkpoint 1

Implements the SimCLR two-view transform and visualises augmentations.
Saves:
  results/augmentation_examples.png
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from utils.seed import set_seed
from utils.dataset_splits import get_cifar10_subset, TwoViewDataset
from utils.visualization import save_augmentation_grid

import matplotlib.pyplot as plt
import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT   = os.path.join(BASE_DIR, "data")
SPLITS_DIR  = os.path.join(BASE_DIR, "splits")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 2026
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


# ─── SimCLR Augmentation Pipeline (as specified in assignment) ────────────────
simclr_transform = T.Compose([
    T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])


# ─── Two-View Transform (as required by assignment) ───────────────────────────
class TwoViewTransform:
    """Applies the same stochastic transform twice to produce two different views."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2


# ─── Helpers ──────────────────────────────────────────────────────────────────
def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised CIFAR-10 CHW tensor to HWC numpy in [0, 1]."""
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std  = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    img = torch.clamp(tensor * std + mean, 0.0, 1.0)
    return img.permute(1, 2, 0).numpy()


def save_augmentation_examples(originals, view1s, view2s, out_path, max_rows=10):
    """
    Save a grid of Original | View 1 | View 2 for each image.
    originals: list of raw PIL images (or tensors before normalisation)
    view1s, view2s: list of normalised tensors (C, H, W)
    """
    rows = min(max_rows, len(originals))
    fig, axes = plt.subplots(rows, 3, figsize=(7, 2.2 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, 0)

    col_titles = ["Original", "Augmented View 1", "Augmented View 2"]
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=10, fontweight="bold", pad=6)

    for r in range(rows):
        # Original — PIL image → numpy
        orig = np.array(originals[r])          # (H, W, 3), uint8
        v1   = denormalize(view1s[r])
        v2   = denormalize(view2s[r])

        axes[r, 0].imshow(orig)
        axes[r, 1].imshow(v1)
        axes[r, 2].imshow(v2)

        for c in range(3):
            axes[r, c].axis("off")

    fig.suptitle("SimCLR Augmentation Examples: Original | View 1 | View 2",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    set_seed(SEED)

    # Load dataset without any transform to get raw PIL images for display
    raw_dataset = get_cifar10_subset(
        data_root=DATA_ROOT,
        split_file=os.path.join(SPLITS_DIR, "train_labeled_10percent.txt"),
        train=True,
        transform=None,          # raw PIL
        download=True,
    )

    # Apply two-view transform separately so we can keep the raw PIL too
    two_view = TwoViewTransform(simclr_transform)

    NUM_EXAMPLES = 10

    originals, view1s, view2s = [], [], []
    for i in range(NUM_EXAMPLES):
        pil_img, _ = raw_dataset[i]
        v1, v2 = two_view(pil_img)
        originals.append(pil_img)
        view1s.append(v1)
        view2s.append(v2)

    save_augmentation_examples(
        originals, view1s, view2s,
        out_path=os.path.join(RESULTS_DIR, "augmentation_examples.png"),
        max_rows=NUM_EXAMPLES,
    )

    # ── Demonstration: TwoViewDataset wrapping ────────────────────────────────
    # This is also provided by the starter dataset_splits.py
    ssl_base = get_cifar10_subset(
        data_root=DATA_ROOT,
        split_file=os.path.join(SPLITS_DIR, "train_ssl_unlabeled.txt"),
        train=True,
        transform=None,
        download=False,
    )
    two_view_dataset = TwoViewDataset(ssl_base, two_view)
    sample_v1, sample_v2, _ = two_view_dataset[0]
    print(f"TwoViewDataset demo — view1 shape: {sample_v1.shape}, view2 shape: {sample_v2.shape}")
    print(f"SSL unlabeled dataset size: {len(two_view_dataset)}")

    # ── Conceptual questions (answers in report) ──────────────────────────────
    print("\n--- Augmentation Task Q&A (for report) ---")
    print("Q1. Are the two augmented views identical?")
    print("    No. Each call to the stochastic transform produces a different random crop,")
    print("    flip, colour jitter, and optional grayscale.")
    print()
    print("Q2. Do they still represent the same object?")
    print("    Yes. The augmentations are designed to preserve the semantic content.")
    print()
    print("Q3. Why should SimCLR treat them as a positive pair?")
    print("    Both views are derived from the same source image. Their underlying")
    print("    visual content is identical, so their feature representations should")
    print("    be similar; SimCLR uses this as a free supervisory signal.")
    print()
    print("Q4. What if augmentations are too weak?")
    print("    Views will look nearly identical. The model can trivially match them")
    print("    without learning semantically meaningful features.")
    print()
    print("Q5. What if augmentations are too strong?")
    print("    Views may lose their semantic content (e.g., extreme crops that show")
    print("    different objects). The positive pair assumption breaks and the model")
    print("    receives a noisy training signal.")


if __name__ == "__main__":
    main()
