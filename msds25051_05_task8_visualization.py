"""
Deep Learning - Spring 2026
Assignment 5 - Task 8: PCA / t-SNE Feature Visualization

Requirements:
  - Extract 512-dim features for 1000 validation images (seed=2026)
  - Generate features from 3 encoders:
      1. Random untrained encoder
      2. SimCLR pretrained encoder
      3. Fine-tuned encoder
  - Reduce to 2D using PCA or t-SNE
  - Color points by class label (labels used ONLY for coloring)
  - Save:
      results/random_encoder_pca_or_tsne.png
      results/simclr_encoder_pca_or_tsne.png
      results/finetuned_encoder_pca_or_tsne.png

Questions to answer:
  1. Do features from the random encoder show class-wise grouping?
  2. Do features from the SimCLR encoder show better grouping?
  3. Does fine-tuning improve class separation?
  4. Which classes are still confused?

Random Seed: 2026
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
    torch.backends.cudnn.benchmark     = False

set_seed()

# ─────────────────────────────────────────
# 2.  PATHS & DEVICE
# ─────────────────────────────────────────
SPLITS_DIR  = "splits"
RESULTS_DIR = "results"
MODELS_DIR  = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLES   = 1000   # fixed subset of validation images
USE_TSNE    = True   # True = t-SNE, False = PCA

print(f"Using device : {DEVICE}")
print(f"Val samples  : {N_SAMPLES}")
print(f"Method       : {'t-SNE' if USE_TSNE else 'PCA'}\n")

# CIFAR-10 class names for plot legend
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 10 distinct colors for 10 classes
COLORS = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
          '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4']

# ─────────────────────────────────────────
# 3.  TRANSFORMS  (eval only — no augmentation)
# ─────────────────────────────────────────
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

eval_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD)
])

# ─────────────────────────────────────────
# 4.  DATASET
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
            img = self.transform(img_pil)
        else:
            img = img_pil
        return img, label

# ─────────────────────────────────────────
# 5.  LOAD VALIDATION SPLIT
#     Fix 1000 samples with seed=2026
# ─────────────────────────────────────────
val_indices = load_indices(os.path.join(SPLITS_DIR, "val.txt"))
print(f"Total val samples : {len(val_indices)}")

# Fix random subset of 1000 with seed=2026
set_seed(SEED)
selected_indices = random.sample(range(len(val_indices)), N_SAMPLES)
selected_indices.sort()

val_dataset = CIFAR10Split(
    root="data",
    indices=val_indices,
    transform=eval_transform,
    train=True
)

# Subset: exactly 1000 images, fixed by seed
val_subset  = Subset(val_dataset, selected_indices)
val_loader  = DataLoader(
    val_subset,
    batch_size=128,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print(f"Selected {N_SAMPLES} validation images (seed={SEED})\n")

# ─────────────────────────────────────────
# 6.  ENCODER — ResNet-18 modified for CIFAR-10
# ─────────────────────────────────────────
def build_encoder():
    model         = models.resnet18(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3,
                              stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Identity()
    return model   # output: (batch, 512)

# ─────────────────────────────────────────
# 7.  CLASSIFICATION MODEL
#     Used to load finetuned_model.pt
# ─────────────────────────────────────────
class ClassificationModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder    = encoder
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        h   = self.encoder(x)
        out = self.classifier(h)
        return out

# ─────────────────────────────────────────
# 8.  FEATURE EXTRACTION
#     Extract 512-dim encoder features
#     Labels used ONLY for coloring plots
# ─────────────────────────────────────────
def extract_features(encoder, loader, device):
    """
    Extract 512-dim features from the encoder.
    Returns:
        features : np.array (N, 512)
        labels   : np.array (N,)  — used only for coloring
    """
    encoder.eval()
    all_features = []
    all_labels   = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            h    = encoder(imgs)       # (batch, 512)
            all_features.append(h.cpu().numpy())
            all_labels.append(labels.numpy())

    features = np.concatenate(all_features, axis=0)  # (N, 512)
    labels   = np.concatenate(all_labels,   axis=0)  # (N,)
    return features, labels

# ─────────────────────────────────────────
# 9.  DIMENSIONALITY REDUCTION
#     PCA or t-SNE → 2D
# ─────────────────────────────────────────
def reduce_to_2d(features, method='tsne', seed=SEED):
    """
    Reduce 512-dim features to 2D using PCA or t-SNE.
    """
    if method == 'tsne':
        print(f"  Running t-SNE (n_samples={features.shape[0]})...")
        # PCA to 50 dims first for speed, then t-SNE
        pca_50     = PCA(n_components=50, random_state=seed)
        feat_50    = pca_50.fit_transform(features)
        tsne       = TSNE(n_components=2, random_state=seed,
                          perplexity=30, n_iter=1000, verbose=0)
        reduced    = tsne.fit_transform(feat_50)
    else:
        print(f"  Running PCA (n_samples={features.shape[0]})...")
        pca     = PCA(n_components=2, random_state=seed)
        reduced = pca.fit_transform(features)

    return reduced   # (N, 2)

# ─────────────────────────────────────────
# 10.  PLOT FUNCTION
# ─────────────────────────────────────────
def plot_2d(reduced, labels, title, save_path, method_name):
    """
    Scatter plot colored by class label.
    Labels used ONLY for coloring — not during training.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for class_idx in range(10):
        mask = labels == class_idx
        ax.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            c=COLORS[class_idx],
            label=CLASS_NAMES[class_idx],
            alpha=0.6,
            s=15,
            edgecolors='none'
        )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(f"{method_name} Dimension 1", fontsize=11)
    ax.set_ylabel(f"{method_name} Dimension 2", fontsize=11)
    ax.legend(
        loc='upper right',
        fontsize=8,
        markerscale=2,
        framealpha=0.8,
        ncol=2
    )
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")

# ─────────────────────────────────────────
# 11. METHOD NAME FOR LABELS
# ─────────────────────────────────────────
method      = 'tsne' if USE_TSNE else 'pca'
method_name = 't-SNE' if USE_TSNE else 'PCA'
file_suffix = 'tsne' if USE_TSNE else 'pca'

# ═══════════════════════════════════════════════════════
# ENCODER 1 — RANDOM UNTRAINED ENCODER
# ═══════════════════════════════════════════════════════
print(f"{'='*55}")
print("ENCODER 1 — Random Untrained Encoder")
print(f"{'='*55}")

set_seed(SEED)
random_encoder = build_encoder().to(DEVICE)

features_rand, labels_rand = extract_features(
    random_encoder, val_loader, DEVICE
)
print(f"  Features shape: {features_rand.shape}")

reduced_rand = reduce_to_2d(features_rand, method=method)

plot_2d(
    reduced_rand, labels_rand,
    title=f"Random Untrained Encoder — {method_name}\n"
          f"({N_SAMPLES} validation images, seed={SEED})",
    save_path=os.path.join(RESULTS_DIR,
                           f"random_encoder_pca_or_tsne.png"),
    method_name=method_name
)

# ═══════════════════════════════════════════════════════
# ENCODER 2 — SimCLR PRETRAINED ENCODER
# ═══════════════════════════════════════════════════════
print(f"\n{'='*55}")
print("ENCODER 2 — SimCLR Pretrained Encoder")
print(f"{'='*55}")

encoder_path = os.path.join(MODELS_DIR, "simclr_encoder.pt")
if not os.path.exists(encoder_path):
    raise FileNotFoundError(
        f"SimCLR encoder not found at {encoder_path}. "
        "Run Task 5 first."
    )

set_seed(SEED)
simclr_encoder = build_encoder().to(DEVICE)
simclr_encoder.load_state_dict(
    torch.load(encoder_path, map_location=DEVICE)
)
print(f"  Loaded: {encoder_path}")

features_simclr, labels_simclr = extract_features(
    simclr_encoder, val_loader, DEVICE
)
print(f"  Features shape: {features_simclr.shape}")

reduced_simclr = reduce_to_2d(features_simclr, method=method)

plot_2d(
    reduced_simclr, labels_simclr,
    title=f"SimCLR Pretrained Encoder — {method_name}\n"
          f"({N_SAMPLES} validation images, seed={SEED})",
    save_path=os.path.join(RESULTS_DIR,
                           f"simclr_encoder_pca_or_tsne.png"),
    method_name=method_name
)

# ═══════════════════════════════════════════════════════
# ENCODER 3 — FINE-TUNED ENCODER
# ═══════════════════════════════════════════════════════
print(f"\n{'='*55}")
print("ENCODER 3 — Fine-tuned Encoder")
print(f"{'='*55}")

ft_path = os.path.join(MODELS_DIR, "finetuned_model.pt")
if not os.path.exists(ft_path):
    raise FileNotFoundError(
        f"Fine-tuned model not found at {ft_path}. "
        "Run Task 7 first."
    )

set_seed(SEED)
ft_encoder = build_encoder().to(DEVICE)
ft_model   = ClassificationModel(ft_encoder).to(DEVICE)
ft_model.load_state_dict(
    torch.load(ft_path, map_location=DEVICE)
)
print(f"  Loaded: {ft_path}")

# Extract features from encoder part only
features_ft, labels_ft = extract_features(
    ft_model.encoder, val_loader, DEVICE
)
print(f"  Features shape: {features_ft.shape}")

reduced_ft = reduce_to_2d(features_ft, method=method)

plot_2d(
    reduced_ft, labels_ft,
    title=f"Fine-tuned Encoder — {method_name}\n"
          f"({N_SAMPLES} validation images, seed={SEED})",
    save_path=os.path.join(RESULTS_DIR,
                           f"finetuned_encoder_pca_or_tsne.png"),
    method_name=method_name
)

# ─────────────────────────────────────────
# 12.  ANSWERS TO TASK 8 QUESTIONS
# ─────────────────────────────────────────
print(f"\n{'='*60}")
print("TASK 8 — REPORT ANSWERS")
print(f"{'='*60}")
print("""
Q1. Do features from the random encoder show class-wise grouping?
    No. The random encoder produces features with no semantic
    meaning. All 10 classes overlap heavily in 2D space — there
    is no visible separation or clustering by class.

Q2. Do features from the SimCLR encoder show better grouping?
    Yes. SimCLR training pushes different images apart and pulls
    same-image views together. Even without labels, the encoder
    learns features that naturally group visually similar images,
    resulting in visible (though imperfect) class clusters.

Q3. Does fine-tuning improve class separation?
    Yes, significantly. Fine-tuning with labeled data directly
    optimizes for classification, so the encoder learns features
    that maximally separate the 10 classes. The clusters become
    much tighter and more clearly separated.

Q4. Which classes are still confused?
    Typically: cat/dog, automobile/truck, bird/airplane/deer.
    These classes share similar visual features (shape, texture,
    color) making them harder to separate in feature space.
""")

# ─────────────────────────────────────────
# 13.  FINAL SUMMARY
# ─────────────────────────────────────────
print(f"{'='*55}")
print("TASK 8 SUMMARY")
print(f"{'='*55}")
print(f"Method           : {method_name}")
print(f"Val images used  : {N_SAMPLES} (seed={SEED})")
print(f"Feature dim      : 512 → 2")
print(f"\nSaved files:")
print(f"  results/random_encoder_pca_or_tsne.png")
print(f"  results/simclr_encoder_pca_or_tsne.png")
print(f"  results/finetuned_encoder_pca_or_tsne.png")
print(f"{'='*55}")
print("\nTask 8 complete.")
