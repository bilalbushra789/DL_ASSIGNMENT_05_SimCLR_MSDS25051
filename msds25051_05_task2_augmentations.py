"""
Task 2: Understanding Augmentations
=====================================
Implements the SimCLR augmentation pipeline and TwoViewTransform.
Visualizes 10 examples: Original | Augmented View 1 | Augmented View 2
Output:
    results/augmentation_examples.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.seed import set_seed

# ── Paths ──────────────────────────────────────────────────────────────────
SPLIT_DIR  = "splits"
DATA_DIR   = "./data"
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

SEED = 2026


# ══════════════════════════════════════════════════════════════════════════
#  SimCLR Augmentation Pipeline  (exactly as specified in the assignment)
# ══════════════════════════════════════════════════════════════════════════
simclr_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4,
                           saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2470, 0.2435, 0.2616)),
])


# ══════════════════════════════════════════════════════════════════════════
#  Two-View Wrapper  (must be implemented yourself per assignment rules)
# ══════════════════════════════════════════════════════════════════════════
class TwoViewTransform:
    """
    Applies the same stochastic transform twice to produce two
    differently-augmented views of the same image.
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2


# ══════════════════════════════════════════════════════════════════════════
#  Helper: un-normalise a tensor for display
# ══════════════════════════════════════════════════════════════════════════
MEAN = np.array([0.4914, 0.4822, 0.4465])
STD  = np.array([0.2470, 0.2435, 0.2616])

def denormalize(tensor):
    """Convert a normalised CHW tensor to a displayable HWC numpy array."""
    img = tensor.permute(1, 2, 0).cpu().numpy()   # CHW -> HWC
    img = img * STD + MEAN                          # undo normalise
    return np.clip(img, 0, 1)


# ══════════════════════════════════════════════════════════════════════════
#  Visualisation
# ══════════════════════════════════════════════════════════════════════════
def visualize_augmentations(num_examples=10,
                             save_path="results/augmentation_examples.png"):
    """
    Shows `num_examples` rows, each containing:
        Original Image | Augmented View 1 | Augmented View 2
    """
    # Load raw (PIL) images — no transform, so we can apply manually
    raw_dataset = datasets.CIFAR10(root=DATA_DIR, train=True,
                                    download=True, transform=None)

    # Load split indices for the labeled set (any split works for visualisation)
    split_file = os.path.join(SPLIT_DIR, "train_labeled_10percent.txt")
    if os.path.exists(split_file):
        with open(split_file) as f:
            indices = [int(l.strip()) for l in f if l.strip()]
    else:
        # Fallback: first N images
        indices = list(range(num_examples * 5))

    # Pick evenly-spaced samples
    step     = max(1, len(indices) // num_examples)
    selected = [indices[i * step] for i in range(num_examples)]

    CIFAR10_CLASSES = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    fig, axes = plt.subplots(num_examples, 3,
                              figsize=(7, num_examples * 2.2))
    fig.suptitle("SimCLR Augmentation Examples\n"
                 "Original  |  Augmented View 1  |  Augmented View 2",
                 fontsize=13, y=1.01)

    col_titles = ["Original", "Augmented View 1", "Augmented View 2"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")

    for row, idx in enumerate(selected):
        pil_img, label = raw_dataset[idx]
        class_name     = CIFAR10_CLASSES[label]

        view1, view2   = TwoViewTransform(simclr_transform)(pil_img)

        # Original (convert PIL -> numpy)
        axes[row, 0].imshow(pil_img)
        axes[row, 0].set_ylabel(class_name, fontsize=8, rotation=0,
                                 labelpad=40, va="center")

        # View 1
        axes[row, 1].imshow(denormalize(view1))

        # View 2
        axes[row, 2].imshow(denormalize(view2))

        for col in range(3):
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Augmentation examples saved to {save_path}")


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    set_seed(SEED)

    print("\n[Task 2] Augmentation pipeline and two-view transform\n")
    print("SimCLR transform pipeline:")
    print("  1. RandomResizedCrop(32, scale=(0.2, 1.0))")
    print("  2. RandomHorizontalFlip(p=0.5)")
    print("  3. ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)")
    print("  4. RandomGrayscale(p=0.2)")
    print("  5. ToTensor()")
    print("  6. Normalize(CIFAR-10 mean/std)\n")

    visualize_augmentations(num_examples=10,
                             save_path=os.path.join(RESULT_DIR,
                                                     "augmentation_examples.png"))

    # ── Conceptual answers ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Answers to Augmentation Questions:")
    print("=" * 60)
    print("""
Q1. Are the two augmented views identical?
    No. Each view is produced by independently sampling random
    crop positions, flip decisions, colour jitter magnitudes,
    and grayscale probabilities, so the two views differ.

Q2. Do they still represent the same object?
    Yes. All transforms are content-preserving; they change
    appearance (colour, crop region, orientation) but not
    the object identity.

Q3. Why should SimCLR treat them as a positive pair?
    Because they originate from the same image. The model
    should learn representations that are invariant to these
    appearance changes, so both views should map to similar
    points in feature space.

Q4. What could go wrong if augmentations are too weak?
    Views would look almost identical. The contrastive task
    becomes trivially easy (the model matches near-identical
    patches rather than learning semantic invariances),
    leading to poor generalisation.

Q5. What could go wrong if augmentations are too strong?
    Views may no longer visually represent the same object
    (e.g. entirely different crops, complete colour destruction).
    The model gets conflicting signals and cannot form
    meaningful positive pairs, degrading representation quality.
""")
    print("[Done] Task 2 complete.")


if __name__ == "__main__":
    main()
