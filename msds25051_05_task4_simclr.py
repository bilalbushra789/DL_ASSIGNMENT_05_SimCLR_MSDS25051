"""
msds25051_05_task4_simclr.py
Deep Learning Spring 2026 - Assignment 5: SimCLR
Checkpoint 2

Implements:
  Task 4.1 - Encoder (ResNet-18 CIFAR-10) + Projection Head
  Task 4.2 - Positive and Negative Pair Construction
  Task 4.3 - Cosine Similarity Matrix + Heatmap
  Task 4.4 - NT-Xent Contrastive Loss (from scratch)
  Task 3   - Feature similarity before training
"""

# ─────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────
import os
import sys
import random
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.datasets import CIFAR10

# ─────────────────────────────────────────────────────────────
# PATHS  
# ─────────────────────────────────────────────────────────────
if os.path.exists("/content/drive/MyDrive/DL_05"):
    # Google Colab
    DRIVE_BASE  = "/content/drive/MyDrive/DL_05"
    DATA_ROOT   = f"{DRIVE_BASE}/data"
    SPLITS_DIR  = f"{DRIVE_BASE}/splits"
    RESULTS_DIR = "/content/results"
    GRAPHS_DIR  = "/content/graphs"
    MODELS_DIR  = "/content/models"
else:
    # Local
    BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT   = os.path.join(BASE_DIR, "data")
    SPLITS_DIR  = os.path.join(BASE_DIR, "splits")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    GRAPHS_DIR  = os.path.join(BASE_DIR, "graphs")
    MODELS_DIR  = os.path.join(BASE_DIR, "models")

for d in [RESULTS_DIR, GRAPHS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
SEED        = 2026
BATCH_SIZE  = 128
TEMPERATURE = 0.5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CIFAR_MEAN  = (0.4914, 0.4822, 0.4465)
CIFAR_STD   = (0.2470, 0.2435, 0.2616)

# ─────────────────────────────────────────────────────────────
# SEED
# ─────────────────────────────────────────────────────────────
def set_seed(seed=2026):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ─────────────────────────────────────────────────────────────
# DATASET UTILITIES
# ─────────────────────────────────────────────────────────────
def read_split_indices(path):
    lines = [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines()]
    return [int(l) for l in lines if l]

def get_cifar10_subset(data_root, split_file, train,
                        transform=None, download=False):
    ds = CIFAR10(root=str(data_root), train=train,
                 transform=transform, download=download)
    return Subset(ds, read_split_indices(split_file))

class TwoViewDataset(Dataset):
    """Returns (view1, view2, label) — label ignored during SSL."""
    def __init__(self, base_dataset, two_view_transform):
        self.base_dataset = base_dataset
        self.two_view_transform = two_view_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, target = self.base_dataset[idx]
        v1, v2 = self.two_view_transform(image)
        return v1, v2, target

class TwoViewTransform:
    """Applies the same stochastic transform twice → two different views."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x), self.transform(x)

# ─────────────────────────────────────────────────────────────
# SIMCLR AUGMENTATION PIPELINE  (as specified in assignment)
# ─────────────────────────────────────────────────────────────
simclr_transform = T.Compose([
    T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
])

# ─────────────────────────────────────────────────────────────
# TASK 4.1 — ENCODER
# ResNet-18 modified for CIFAR-10 (32x32 images)
# ─────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    """
    ResNet-18 backbone modified for CIFAR-10:
      - conv1 : 3x3, stride=1, padding=1  (no large-scale downsampling)
      - maxpool : removed (replaced with Identity)
      - fc : removed → outputs 512-d feature vector h
    """
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=None)
        # Modify first conv for 32x32 input
        base.conv1  = nn.Conv2d(3, 64, kernel_size=3,
                                stride=1, padding=1, bias=False)
        # Remove max-pool (would kill spatial info on small images)
        base.maxpool = nn.Identity()
        # Strip the final FC layer — we want 512-d features
        self.backbone = nn.Sequential(*list(base.children())[:-1])

    def forward(self, x):
        x = self.backbone(x)        # (B, 512, 1, 1)
        x = x.flatten(start_dim=1) # (B, 512)
        return x


# ─────────────────────────────────────────────────────────────
# TASK 4.1 — PROJECTION HEAD
# ─────────────────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    """
    MLP projection head:
      Linear(512 -> 256) -> ReLU -> Linear(256 -> 128)
    Maps 512-d encoder output to 128-d space for contrastive loss.
    """
    def __init__(self, in_dim=512, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────
# FULL SIMCLR MODEL
# ─────────────────────────────────────────────────────────────
class SimCLR(nn.Module):
    """
    Combines Encoder + ProjectionHead.
    Returns (h, z):
      h : 512-d representation  (used for downstream tasks)
      z : 128-d projection      (used for contrastive loss only)
    """
    def __init__(self):
        super().__init__()
        self.encoder   = Encoder()
        self.projector = ProjectionHead()

    def forward(self, x):
        h = self.encoder(x)    # (B, 512)
        z = self.projector(h)  # (B, 128)
        return h, z


# ─────────────────────────────────────────────────────────────
# TASK 4.3 — COSINE SIMILARITY MATRIX
# ─────────────────────────────────────────────────────────────
def compute_similarity_matrix(z):
    """
    Compute (2N x 2N) cosine similarity matrix.

    Args:
        z : (2N, dim) tensor of L2-normalised projections
    Returns:
        sim : (2N, 2N) cosine similarity matrix
    """
    z   = F.normalize(z, dim=1)
    sim = z @ z.T   # (2N, 2N)
    return sim


# ─────────────────────────────────────────────────────────────
# TASK 4.4 — NT-Xent LOSS 
# ─────────────────────────────────────────────────────────────
class NTXentLoss(nn.Module):
    """
    Normalised Temperature-scaled Cross Entropy Loss.

    For a batch of N original images → 2N augmented views:
      Positive pair  : (view1_i, view2_i) — same source image
      Negative pairs : all other 2N-2 views

    loss(i,j) = -log[ exp(sim(zi,zj) / tau) /
                      sum_{k != i} exp(sim(zi,zk) / tau) ]

    Final loss = mean over all 2N anchor terms.
    tau = temperature (default 0.5)
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.tau = temperature

    def forward(self, z1, z2):
        """
        Args:
            z1 : (N, dim) projections from view 1
            z2 : (N, dim) projections from view 2
        Returns:
            scalar loss
        """
        N = z1.size(0)

        # 1. Concatenate and L2-normalise → (2N, dim)
        z = torch.cat([z1, z2], dim=0)
        z = F.normalize(z, dim=1)

        # 2. Full cosine similarity matrix scaled by temperature → (2N, 2N)
        sim = (z @ z.T) / self.tau

        # 3. Build positive-pair label for each row:
        #    row i (in view1 block) → positive at i+N
        #    row i (in view2 block) → positive at i-N
        labels = torch.cat([
            torch.arange(N, 2 * N, device=z.device),
            torch.arange(0, N,     device=z.device),
        ])  # (2N,)

        # 4. Mask diagonal (self-similarity) → set to -inf
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float('-inf'))

        # 5. Cross-entropy: for each row, maximise similarity to the positive
        loss = F.cross_entropy(sim, labels)
        return loss


# ─────────────────────────────────────────────────────────────
# VISUALISE SIMILARITY MATRIX HEATMAP
# ─────────────────────────────────────────────────────────────
def plot_similarity_matrix(sim_np, N, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim_np, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("View index")
    ax.set_ylabel("View index")

    # Highlight positive pairs with white boxes
    for i in range(N):
        # view1_i <-> view2_i  (top-right block)
        ax.add_patch(plt.Rectangle(
            (i + N - 0.5, i - 0.5), 1, 1,
            fill=False, edgecolor='white', lw=2))
        # view2_i <-> view1_i  (bottom-left block)
        ax.add_patch(plt.Rectangle(
            (i - 0.5, i + N - 0.5), 1, 1,
            fill=False, edgecolor='white', lw=2))

    # Dividing lines between view1 / view2 blocks
    ax.axhline(N - 0.5, color='yellow', lw=1.5, linestyle='--', alpha=0.7)
    ax.axvline(N - 0.5, color='yellow', lw=1.5, linestyle='--', alpha=0.7)
    ax.text(N / 2 - 0.5, -1.5, "View 1", ha='center', fontsize=9, color='yellow')
    ax.text(N + N / 2 - 0.5, -1.5, "View 2", ha='center', fontsize=9, color='yellow')

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.show()
    print(f"Saved → {out_path}")


# ─────────────────────────────────────────────────────────────
# POSITIVE / NEGATIVE PAIR TABLE  (report requirement Task 4.2)
# ─────────────────────────────────────────────────────────────
def print_pair_table(N, num_show=4):
    print("\nPositive Pair Table (for report — Task 4.2):")
    print(f"{'Original Image':<16} {'View 1 Index':<14} "
          f"{'View 2 Index':<14} {'Positive Pair'}")
    print("-" * 58)
    for i in range(num_show):
        print(f"  image {i:<10} {i:<14} {i + N:<14} yes")
    print(f"\n  Total images in batch : {N}")
    print(f"  Total views  (2N)     : {2 * N}")
    print(f"  Total positive pairs  : {N}")
    print(f"  Total negative pairs per anchor: {2*N - 2}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    set_seed(SEED)
    print(f"Device     : {DEVICE}")
    print(f"Seed       : {SEED}")
    print(f"Temperature: {TEMPERATURE}")
    print()

    # ── Build model ───────────────────────────────────────────
    model = SimCLR().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    enc_params   = sum(p.numel() for p in model.encoder.parameters())
    proj_params  = sum(p.numel() for p in model.projector.parameters())
    print(f"Encoder parameters    : {enc_params:,}")
    print(f"Projector parameters  : {proj_params:,}")
    print(f"Total parameters      : {total_params:,}")

    # ── Quick shape verification ───────────────────────────────
    _x = torch.randn(4, 3, 32, 32).to(DEVICE)
    _h, _z = model(_x)
    print(f"\nShape check:")
    print(f"  Input  x : {_x.shape}")
    print(f"  Encoder h: {_h.shape}  (should be [4, 512])")
    print(f"  Projector z: {_z.shape}  (should be [4, 128])")
    assert _h.shape == (4, 512), "Encoder output shape wrong!"
    assert _z.shape == (4, 128), "Projector output shape wrong!"
    del _x, _h, _z
    print("  Shape check passed ✓")

    # ── NT-Xent loss sanity check ──────────────────────────────
    criterion = NTXentLoss(temperature=TEMPERATURE)
    _z1 = torch.randn(8, 128).to(DEVICE)
    _z2 = torch.randn(8, 128).to(DEVICE)
    _loss = criterion(_z1, _z2)
    expected = torch.log(torch.tensor(15.0)).item()  # log(2N-1)
    print(f"\nNT-Xent loss sanity check:")
    print(f"  Loss on random projections  : {_loss.item():.4f}")
    print(f"  Expected ≈ log(2N-1) = log(15): {expected:.4f}")
    print(f"  NT-Xent Loss implemented ✓")
    del _z1, _z2, _loss

    # ── Load data ─────────────────────────────────────────────
    print(f"\nLoading dataset from {SPLITS_DIR} ...")
    base_ds = get_cifar10_subset(
        DATA_ROOT,
        os.path.join(SPLITS_DIR, "train_labeled_10percent.txt"),
        train=True, transform=None, download=True,
    )
    two_view_ds = TwoViewDataset(base_ds, TwoViewTransform(simclr_transform))
    loader = DataLoader(two_view_ds, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=2, pin_memory=True)
    print(f"Dataset size: {len(two_view_ds)}")

    # ── Similarity matrix BEFORE training ─────────────────────
    model.eval()
    SMALL_N = 16   # use 16 images → 32 views for readable heatmap

    small_loader = DataLoader(two_view_ds, batch_size=SMALL_N,
                              shuffle=True, num_workers=2)
    v1_batch, v2_batch, _ = next(iter(small_loader))

    with torch.no_grad():
        _, z1 = model(v1_batch.to(DEVICE))
        _, z2 = model(v2_batch.to(DEVICE))

    # (2N, 128) combined projections
    z_all   = torch.cat([z1, z2], dim=0)
    sim_mat = compute_similarity_matrix(z_all)
    sim_np  = sim_mat.cpu().numpy()

    # Plot heatmap
    plot_similarity_matrix(
        sim_np, N=SMALL_N,
        out_path=os.path.join(RESULTS_DIR, "similarity_matrix_before_training.png"),
        title="Cosine Similarity Matrix BEFORE SimCLR Training\n"
              f"(2N={2*SMALL_N} views: first {SMALL_N}=view1, last {SMALL_N}=view2)\n"
              "White boxes = positive pairs | Yellow line = view1/view2 boundary",
    )

    # ── Compute average similarities ──────────────────────────
    z1_n = F.normalize(z1, dim=1)
    z2_n = F.normalize(z2, dim=1)

    # Same-image (positive pairs)
    same_sim = (z1_n * z2_n).sum(dim=1).mean().item()

    # Different-image (off-diagonal of z1 @ z1.T)
    cross     = z1_n @ z1_n.T
    off_diag  = ~torch.eye(SMALL_N, dtype=torch.bool, device=DEVICE)
    diff_sim  = cross[off_diag].mean().item()

    print("\n" + "=" * 55)
    print("Feature Similarity BEFORE SimCLR Training")
    print("=" * 55)
    print(f"  Same image  (positive pairs)  : {same_sim:.4f}")
    print(f"  Different images (negatives)  : {diff_sim:.4f}")
    print("=" * 55)
    print("\nInterpretation:")
    print("  Same-image and different-image similarities are close.")
    print("  The random encoder has no sense of which views belong together.")
    print("  SimCLR training will push same-image sim UP, diff-image sim DOWN.")

    # ── Positive/Negative pair table ──────────────────────────
    print_pair_table(N=SMALL_N, num_show=4)

    # ── Similarity matrix Q&A (for report) ────────────────────
    print("""
Similarity Matrix Q&A (Task 4.3 — for report):

Q1. Why is the diagonal ignored?
    The diagonal represents each view's similarity with itself = 1.0.
    This is trivially the highest value and provides no useful learning
    signal; including it would dominate the denominator of NT-Xent loss.

Q2. Where are the positive pairs located?
    For a 2N x 2N matrix where the first N rows/cols are view1 and
    the last N are view2, positive pairs are at positions (i, i+N)
    and (i+N, i) — the off-diagonal blocks of the top-right and
    bottom-left quadrants. These are highlighted in white in the heatmap.

Q3. Why are all other entries treated as negatives?
    SimCLR assumes any two different images (even same class) should
    have dissimilar representations compared to the two views of the
    SAME image. This large set of negatives (2N-2 per anchor) makes
    the task harder and forces the model to learn more discriminative
    features.
""")

    # ── Update metrics.json ───────────────────────────────────
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        metrics = {
            "student_name": "bilal bushra",
            "roll_number" : "MSDS25051",
            "seed"        : 2026,
            "batch_size"  : BATCH_SIZE,
            "simclr_epochs"        : 50,
            "linear_probe_epochs"  : 20,
            "finetuning_epochs"    : 20,
            "learning_rate"        : 0.0003,
            "temperature"          : TEMPERATURE,
            "supervised_10percent_test_acc"    : 0.0,
            "random_linear_probe_test_acc"     : 0.0,
            "simclr_linear_probe_test_acc"     : 0.0,
            "simclr_finetune_test_acc"         : 0.0,
            "same_view_similarity_before"      : 0.0,
            "different_image_similarity_before": 0.0,
            "same_view_similarity_after"       : 0.0,
            "different_image_similarity_after" : 0.0,
            "github_repo_url"              : "",
            "first_commit_date"            : "",
            "last_commit_before_deadline"  : "",
            "number_of_meaningful_commits" : 0,
        }

    metrics["same_view_similarity_before"]       = round(same_sim, 4)
    metrics["different_image_similarity_before"] = round(diff_sim, 4)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nUpdated metrics.json → {metrics_path}")

    # ── Copy to Drive if on Colab ──────────────────────────────
    if os.path.exists("/content/drive/MyDrive/DL_05"):
        import shutil
        DRIVE_BASE = "/content/drive/MyDrive/DL_05"
        for folder, local in [("results", RESULTS_DIR), ("graphs", GRAPHS_DIR)]:
            dst = f"{DRIVE_BASE}/{folder}"
            os.makedirs(dst, exist_ok=True)
            for fname in os.listdir(local):
                shutil.copy2(f"{local}/{fname}", f"{dst}/{fname}")
                print(f"Saved to Drive → {dst}/{fname}")

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("CHECKPOINT 2 COMPLETE")
    print("=" * 55)
    print(f"  Encoder output dim      : 512")
    print(f"  Projector output dim    : 128")
    print(f"  Temperature (tau)       : {TEMPERATURE}")
    print(f"  Same-view sim (before)  : {same_sim:.4f}")
    print(f"  Diff-image sim (before) : {diff_sim:.4f}")
    print()
    outputs = [
        os.path.join(RESULTS_DIR, "similarity_matrix_before_training.png"),
        os.path.join(RESULTS_DIR, "metrics.json"),
    ]
    for p in outputs:
        status = "✓" if os.path.exists(p) else "✗ MISSING"
        print(f"  [{status}] {p}")

    return same_sim, diff_sim


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
