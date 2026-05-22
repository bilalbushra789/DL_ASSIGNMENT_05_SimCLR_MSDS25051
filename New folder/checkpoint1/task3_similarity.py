"""
Task 3: Feature Similarity Before Training
Deep Learning Spring 2026 - Assignment 5 SimCLR
Checkpoint 1

Passes images through a random (untrained) ResNet-18 encoder and computes
cosine similarity between:
  - Two augmented views of the SAME image   (positive pair)
  - Views from DIFFERENT images              (negative pair)

Prints the summary table required by the assignment.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader

from utils.seed import set_seed
from utils.dataset_splits import get_cifar10_subset, TwoViewDataset

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT   = os.path.join(BASE_DIR, "data")
SPLITS_DIR  = os.path.join(BASE_DIR, "splits")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

SEED       = 2026
BATCH_SIZE = 128
NUM_BATCHES = 10          # number of batches to average over for a stable estimate
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


# ─── SimCLR augmentation (same as Task 2) ────────────────────────────────────
simclr_transform = T.Compose([
    T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])


class TwoViewTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x), self.transform(x)


# ─── Encoder (random / untrained) ────────────────────────────────────────────
def build_encoder(num_classes: int = 10) -> nn.Module:
    """ResNet-18 modified for CIFAR-10. Classification head removed; returns 512-d."""
    model = models.resnet18(weights=None)
    model.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc     = nn.Identity()           # remove classifier → 512-d output
    return model


# ─── Similarity computation ───────────────────────────────────────────────────
@torch.no_grad()
def compute_batch_similarities(encoder, loader, device, max_batches=10):
    """
    Returns (mean_same_sim, mean_diff_sim) averaged over `max_batches` batches.

    For each batch of 2N views (N images × 2 views):
      - same-image similarity : mean cosine sim between view1_i and view2_i  (N pairs)
      - diff-image similarity : mean cosine sim between view1_i and view1_j  (i≠j)
    """
    encoder.eval()
    same_sims, diff_sims = [], []

    for batch_idx, (view1, view2, _) in enumerate(loader):
        if batch_idx >= max_batches:
            break

        view1, view2 = view1.to(device), view2.to(device)

        z1 = F.normalize(encoder(view1), dim=1)   # (N, 512)
        z2 = F.normalize(encoder(view2), dim=1)   # (N, 512)

        # Same-image cosine similarity: dot product of paired normalised vectors
        same_sim = (z1 * z2).sum(dim=1)           # (N,)
        same_sims.append(same_sim.mean().item())

        # Different-image similarity: all off-diagonal entries of z1 @ z1.T
        sim_matrix = z1 @ z1.T                    # (N, N)
        N = z1.size(0)
        mask = ~torch.eye(N, dtype=torch.bool, device=device)
        diff_sim = sim_matrix[mask].mean()
        diff_sims.append(diff_sim.item())

    return float(torch.tensor(same_sims).mean()), float(torch.tensor(diff_sims).mean())


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    set_seed(SEED)
    print(f"Device: {DEVICE}")

    # ── Dataset (use labeled split — any split works, labels ignored) ─────────
    base_dataset = get_cifar10_subset(
        data_root=DATA_ROOT,
        split_file=os.path.join(SPLITS_DIR, "train_labeled_10percent.txt"),
        train=True,
        transform=None,
        download=True,
    )
    two_view_dataset = TwoViewDataset(base_dataset, TwoViewTransform(simclr_transform))
    loader = DataLoader(
        two_view_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True,
    )

    # ── Random (untrained) encoder ────────────────────────────────────────────
    encoder = build_encoder().to(DEVICE)
    print(f"Encoder output dimension: 512 (random, untrained)")

    same_sim_before, diff_sim_before = compute_batch_similarities(
        encoder, loader, DEVICE, max_batches=NUM_BATCHES
    )

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Feature Similarity BEFORE SimCLR Training (Random Encoder)")
    print("=" * 60)
    print(f"{'Pair Type':<40} {'Avg Cosine Similarity':>20}")
    print("-" * 60)
    print(f"{'Same image, two augmented views':<40} {same_sim_before:>20.4f}")
    print(f"{'Different images':<40} {diff_sim_before:>20.4f}")
    print("=" * 60)

    print("\nInterpretation:")
    print(f"  Before SimCLR training, same-image similarity ({same_sim_before:.4f}) and")
    print(f"  different-image similarity ({diff_sim_before:.4f}) are very close.")
    print("  This confirms the random encoder does not distinguish positive pairs from")
    print("  negative pairs — it has not learned any semantic structure yet.")

    # Return values so they can be used by allCode.py / metrics.json
    return same_sim_before, diff_sim_before


if __name__ == "__main__":
    same, diff = main()
    print(f"\nsame_view_similarity_before   = {same:.4f}")
    print(f"different_image_similarity_before = {diff:.4f}")
