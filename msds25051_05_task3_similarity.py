"""
Deep Learning - Spring 2026
Assignment 5 - Task 3: Feature Similarity Before Training

Requirements:
  - Use random untrained ResNet-18 encoder (NO pretrained weights)
  - Pass two augmented views through encoder
  - Compute cosine similarity matrix (2N x 2N)
  - Compute average cosine similarity:
      * Same image, two augmented views (positive pairs)
      * Different images (negative pairs)
  - Save: results/similarity_matrix_before_training.png
  - Answer 3 questions about the similarity matrix
  - Random Seed: 2026

"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# 1.  REPRODUCIBILITY
# ─────────────────────────────────────────
SEED = 2026

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# ─────────────────────────────────────────
# 2.  PATHS & SETTINGS
# ─────────────────────────────────────────
SPLITS_DIR  = "splits"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8        # Small batch for visualization & table (matches assignment spec)

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────
# 3.  AUGMENTATION PIPELINE  (same as Task 2)
# ─────────────────────────────────────────
simclr_transform = T.Compose([
    T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD)
])

# ─────────────────────────────────────────
# 4.  TWO-VIEW TRANSFORM  (from Task 2)
# ─────────────────────────────────────────
class TwoViewTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2

# ─────────────────────────────────────────
# 5.  DATASET
# ─────────────────────────────────────────
def load_indices(filepath):
    with open(filepath, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]

class CIFAR10Split(Dataset):
    def __init__(self, root, indices, transform=None, train=True):
        self.base_dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=True, transform=None
        )
        self.indices   = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx       = self.indices[idx]
        img_pil, label = self.base_dataset[real_idx]
        if self.transform:
            result = self.transform(img_pil)
            return result[0], result[1], label   # view1, view2, label
        return img_pil, label

# ─────────────────────────────────────────
# 6.  LOAD UNLABELED SPLIT
# ─────────────────────────────────────────
ssl_indices = load_indices(os.path.join(SPLITS_DIR, "train_ssl_unlabeled.txt"))
print(f"SSL unlabeled samples: {len(ssl_indices)}")

two_view_tf = TwoViewTransform(simclr_transform)

ssl_dataset = CIFAR10Split(
    root="data",
    indices=ssl_indices,
    transform=two_view_tf,
    train=True
)

# DataLoader with batch size = 8
ssl_loader = DataLoader(
    ssl_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

# ─────────────────────────────────────────
# 7.  ENCODER  — ResNet-18 modified for CIFAR-10
#     weights=None → RANDOM, untrained
# ─────────────────────────────────────────
def build_cifar_resnet18():
    """
    ResNet-18 modified for CIFAR-10:
      conv1  : 3x3, stride 1, padding 1
      maxpool: removed (Identity)
      fc     : removed (Identity) → 512-dim output
    weights=None → completely random, no pretraining
    """
    model = models.resnet18(weights=None)
    model.conv1  = nn.Conv2d(3, 64, kernel_size=3,
                             stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Identity()
    return model

encoder = build_cifar_resnet18().to(DEVICE)
encoder.eval()

total_params = sum(p.numel() for p in encoder.parameters())
print(f"Encoder parameters: {total_params:,}  (random, untrained)\n")

# ─────────────────────────────────────────
# 8.  EXTRACT ONE BATCH  (N = 8)
# ─────────────────────────────────────────
set_seed(SEED)   # fix seed for reproducibility

data_iter = iter(ssl_loader)
view1_batch, view2_batch, _ = next(data_iter)

view1_batch = view1_batch.to(DEVICE)   # (8, 3, 32, 32)
view2_batch = view2_batch.to(DEVICE)   # (8, 3, 32, 32)

N = view1_batch.size(0)   # N = 8
print(f"Batch size N = {N}")
print(f"Total views  = 2N = {2*N}")

# ─────────────────────────────────────────
# 9.  FORWARD PASS THROUGH RANDOM ENCODER
# ─────────────────────────────────────────
with torch.no_grad():
    z1 = encoder(view1_batch)   # (N, 512)
    z2 = encoder(view2_batch)   # (N, 512)

print(f"Feature shape per view: {z1.shape}")

# ─────────────────────────────────────────
# 10.  L2-NORMALIZE → COSINE SIM = DOT PRODUCT
# ─────────────────────────────────────────
z1_norm = F.normalize(z1, dim=1)   # (N, 512)
z2_norm = F.normalize(z2, dim=1)   # (N, 512)

# Stack: rows 0..N-1 = View1, rows N..2N-1 = View2
all_features = torch.cat([z1_norm, z2_norm], dim=0)   # (2N, 512)
print(f"Combined feature matrix: {all_features.shape}")

# ─────────────────────────────────────────
# 11.  COSINE SIMILARITY MATRIX  (2N x 2N)
# ─────────────────────────────────────────
sim_matrix    = torch.mm(all_features, all_features.T)   # (2N, 2N)
sim_matrix_np = sim_matrix.cpu().numpy()

print(f"Similarity matrix shape: {sim_matrix_np.shape}")

# ─────────────────────────────────────────
# 12.  COMPUTE AVERAGE SIMILARITIES
# ─────────────────────────────────────────
pos_sims = []
neg_sims = []
two_N    = 2 * N

for i in range(two_N):
    for j in range(two_N):
        if i == j:
            continue   # skip diagonal (self-similarity)

        is_positive = (
            (i < N and j == i + N) or   # view1_i  ↔ view2_i
            (i >= N and j == i - N)      # view2_i  ↔ view1_i
        )

        if is_positive:
            pos_sims.append(sim_matrix_np[i, j])
        else:
            neg_sims.append(sim_matrix_np[i, j])

avg_pos_sim = float(np.mean(pos_sims))
avg_neg_sim = float(np.mean(neg_sims))

# ─────────────────────────────────────────
# 13.  PRINT RESULTS
# ─────────────────────────────────────────
print(f"\n{'='*60}")
print("FEATURE SIMILARITY — BEFORE SimCLR TRAINING")
print(f"{'='*60}")
print(f"Pair Type                              | Avg Cosine Similarity")
print(f"---------------------------------------|----------------------")
print(f"Same image, two augmented views        | {avg_pos_sim:.4f}")
print(f"Different images                       | {avg_neg_sim:.4f}")
print(f"{'='*60}")
print(f"\nInterpretation:")
print(f"  Before training, the random encoder has NO concept of which")
print(f"  views belong together. Positive and negative pair similarities")
print(f"  are close to each other — showing the encoder is uninformative.")

# ─────────────────────────────────────────
# 14.  POSITIVE PAIR TABLE 
#      For N=8 images, View2 indices = 8..15
# ─────────────────────────────────────────
print(f"\nPositive Pair Table (batch of {N} images):")
print(f"{'Original Image':<16} {'View 1 Index':<16} {'View 2 Index':<16} {'Positive Pair'}")
print("-" * 65)
for i in range(N):
    print(f"{'image ' + str(i):<16} {i:<16} {i + N:<16} {'yes'}")

# ─────────────────────────────────────────
# 15.  VISUALIZE SIMILARITY MATRIX
#      2N x 2N heatmap with positive pairs marked
# ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))

im = ax.imshow(sim_matrix_np, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Yellow dashed lines dividing View1 / View2 blocks
ax.axhline(y=N - 0.5, color='white', linewidth=1.5, linestyle='--')
ax.axvline(x=N - 0.5, color='white', linewidth=1.5, linestyle='--')

# Block labels on axes
ticks = list(range(two_N))
labels_v1 = [f"v1_{i}" for i in range(N)]
labels_v2 = [f"v2_{i}" for i in range(N)]
all_labels = labels_v1 + labels_v2

ax.set_xticks(ticks)
ax.set_xticklabels(all_labels, rotation=90, fontsize=7)
ax.set_yticks(ticks)
ax.set_yticklabels(all_labels, fontsize=7)

# Mark positive pairs with green boxes (matching assignment expected output)
for i in range(N):
    j = i + N   # positive pair: view1_i → view2_i
    # top-right block: (row=i, col=i+N)
    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                fill=False, edgecolor='lime',
                                linewidth=2.0))
    # bottom-left block: (row=i+N, col=i)
    ax.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1,
                                fill=False, edgecolor='lime',
                                linewidth=2.0))

ax.set_title(
    f"Cosine Similarity Matrix — Before SimCLR Training\n"
    f"(Green boxes = positive pairs | 2N={two_N} views)",
    fontsize=12, fontweight='bold'
)
ax.set_xlabel("View index", fontsize=10)
ax.set_ylabel("View index", fontsize=10)

plt.tight_layout()
save_path = os.path.join(RESULTS_DIR, "similarity_matrix_before_training.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved → {save_path}")

# ─────────────────────────────────────────
# 16.  REPORT TABLE
# ─────────────────────────────────────────
print(f"\nBefore SimCLR Training:")
print(f"  Same image (two augmented views) avg cosine similarity : {avg_pos_sim:.4f}")
print(f"  Different images avg cosine similarity                 : {avg_neg_sim:.4f}")

# ─────────────────────────────────────────
# 17.  REPORT ANSWERS — 3 Questions
# ─────────────────────────────────────────
print(f"\n{'='*60}")
print("TASK 3 — REPORT ANSWERS")
print(f"{'='*60}")
print("""
Q1. Why is the diagonal ignored?
    The diagonal entry sim[i, i] is the similarity of a view with
    itself, which is always 1.0. Including it gives no useful signal
    and would inflate scores. SimCLR explicitly masks the diagonal
    when computing the NT-Xent loss.

Q2. Where are the positive pairs located?
    For N images producing 2N views (indices 0..N-1 = View1,
    N..2N-1 = View2), the positive pair of image i is located at
    positions (i, i+N) and (i+N, i) — in the top-right and
    bottom-left off-diagonal blocks of the 2N x 2N matrix.

Q3. Why are all other entries treated as negatives?
    SimCLR assumes any two different images are unlikely to be the
    same class, especially with large batches. All views that are
    not the designated positive pair of a given anchor are treated
    as negatives. NT-Xent loss pushes them apart, forcing the
    encoder to learn discriminative features.
""")

# ─────────────────────────────────────────
# 18.  FINAL SUMMARY
# ─────────────────────────────────────────
print(f"{'='*60}")
print("TASK 3 SUMMARY")
print(f"{'='*60}")
print(f"Encoder           : ResNet-18, random weights (weights=None)")
print(f"Batch size N      : {N}")
print(f"Similarity matrix : {two_N} x {two_N}")
print(f"Avg sim (same img): {avg_pos_sim:.4f}  ← positive pairs")
print(f"Avg sim (diff img): {avg_neg_sim:.4f}  ← negative pairs")
print(f"Output saved      : {save_path}")
print(f"{'='*60}")
print(f"\nKey Observation:")
print(f"  Positive sim ({avg_pos_sim:.4f}) ≈ Negative sim ({avg_neg_sim:.4f})")
print(f"  → Random encoder does NOT distinguish same-image pairs.")
print(f"  → SimCLR training should INCREASE positive similarity")
print(f"     and DECREASE negative similarity.")
