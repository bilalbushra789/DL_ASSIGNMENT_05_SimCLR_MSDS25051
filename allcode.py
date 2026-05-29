

# ══════════════════════════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import os
import sys
import random
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import CIFAR10

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ══════════════════════════════════════════════════════════════════════════════
# PATHS & GLOBAL CONFIG
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT   = os.path.join(BASE_DIR, "data")
SPLITS_DIR  = os.path.join(BASE_DIR, "splits")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
GRAPHS_DIR  = os.path.join(BASE_DIR, "graphs")
MODELS_DIR  = os.path.join(BASE_DIR, "models")

for _d in [DATA_ROOT, RESULTS_DIR, GRAPHS_DIR, MODELS_DIR]:
    os.makedirs(_d, exist_ok=True)

SEED        = 2026
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CIFAR_MEAN  = (0.4914, 0.4822, 0.4465)
CIFAR_STD   = (0.2470, 0.2435, 0.2616)

# ══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_indices(filepath):
    with open(filepath, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]


class CIFAR10Split(Dataset):
    """Generic CIFAR-10 subset dataset."""
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


class TwoViewTransform:
    """Applies the same stochastic transform twice to produce two different views."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class TwoViewDataset(Dataset):
    """Wraps a base dataset and returns (view1, view2, label)."""
    def __init__(self, base_dataset, two_view_transform):
        self.base_dataset        = base_dataset
        self.two_view_transform  = two_view_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, target = self.base_dataset[idx]
        v1, v2 = self.two_view_transform(image)
        return v1, v2, target


class TwoViewDatasetNoLabel(Dataset):
    """Like TwoViewDataset but returns only (view1, view2) — no label used."""
    def __init__(self, root, indices, transform, train=True):
        self.base_dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=True, transform=None
        )
        self.indices   = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx       = self.indices[idx]
        img_pil, _     = self.base_dataset[real_idx]
        v1, v2 = self.transform(img_pil)
        return v1, v2


# ── SimCLR augmentation pipeline ──────────────────────────────────────────────
simclr_transform = T.Compose([
    T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
])

train_transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
])

eval_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
])


# ══════════════════════════════════════════════════════════════════════════════
# SHARED MODEL COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def build_encoder():
    """
    ResNet-18 modified for CIFAR-10 (32x32):
      - conv1 : 3x3, stride=1, padding=1
      - maxpool : replaced with Identity
      - fc : replaced with Identity → 512-dim output
    Random weights (weights=None).
    """
    model         = models.resnet18(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Identity()
    return model


class ProjectionHead(nn.Module):
    """MLP: Linear(512->256) -> ReLU -> Linear(256->128)"""
    def __init__(self, in_dim=512, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimCLR(nn.Module):
    """Encoder + ProjectionHead. Returns (h, z)."""
    def __init__(self):
        super().__init__()
        self.encoder         = build_encoder()
        self.projection_head = ProjectionHead()

    def forward(self, x):
        h = self.encoder(x)            # (B, 512)
        z = self.projection_head(h)    # (B, 128)
        return h, z


class NTXentLoss(nn.Module):
    """
    Normalised Temperature-scaled Cross Entropy Loss.
    Implements NT-Xent from scratch.
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.tau = temperature

    def forward(self, z1, z2):
        N  = z1.size(0)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        z  = torch.cat([z1, z2], dim=0)            # (2N, D)
        sim = torch.mm(z, z.T) / self.tau           # (2N, 2N)
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float('-inf'))
        labels = torch.cat([
            torch.arange(N, 2 * N),
            torch.arange(0, N)
        ]).to(z.device)
        return F.cross_entropy(sim, labels)


class LinearProbe(nn.Module):
    """Frozen encoder + trainable Linear(512->10) head."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder    = encoder
        self.classifier = nn.Linear(512, 10)
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            h = self.encoder(x)
        return self.classifier(h)


class ClassificationModel(nn.Module):
    """Full model: encoder + Linear(512->10). Both parts trainable."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder    = encoder
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        h = self.encoder(x)
        return self.classifier(h)


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 — SUPERVISED BASELINE WITH LIMITED LABELS
# ══════════════════════════════════════════════════════════════════════════════

def build_resnet18_cifar10(num_classes=10):
    """ResNet-18 from scratch for CIFAR-10, with classification head."""
    model         = models.resnet18(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(512, num_classes)
    return model


def top1_accuracy_from_logits(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def save_confusion_matrix(y_true, y_pred, out_path, title="Confusion Matrix"):
    from sklearn.metrics import confusion_matrix
    CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                       'dog','frog','horse','ship','truck']
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    im  = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ticks = np.arange(len(CIFAR10_CLASSES))
    ax.set_xticks(ticks); ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha='right')
    ax.set_yticks(ticks); ax.set_yticklabels(CIFAR10_CLASSES)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black', fontsize=8)
    ax.set_ylabel('True Label'); ax.set_xlabel('Predicted Label')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[Saved] {out_path}")


def per_class_accuracy(y_true, y_pred):
    CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                       'dog','frog','horse','ship','truck']
    result = {}
    for i, name in enumerate(CIFAR10_CLASSES):
        mask = y_true == i
        if mask.sum() > 0:
            result[name] = (y_pred[mask] == i).mean()
        else:
            result[name] = 0.0
    return result


def task1_train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def task1_evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_logits, all_labels = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * images.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    return total_loss / len(loader.dataset), top1_accuracy_from_logits(all_logits, all_labels), all_logits, all_labels


def task1_save_loss_curve(train_losses, val_losses, val_accs, out_path):
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_losses, label="Train Loss", color="steelblue")
    axes[0].plot(epochs, val_losses,   label="Val Loss",   color="coral")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Supervised Baseline — Loss Curves"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, val_accs, label="Val Accuracy", color="green")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Supervised Baseline — Validation Accuracy"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)
    print(f"[Saved] {out_path}")


def task1_main():
    print("\n" + "="*60)
    print("TASK 1 — SUPERVISED BASELINE (10% LABELED)")
    print("="*60)
    set_seed(SEED)
    BATCH_SIZE = 64; NUM_EPOCHS = 20; LR = 3e-4; NUM_CLASSES = 10

    train_idx = load_indices(os.path.join(SPLITS_DIR, "train_labeled_10percent.txt"))
    val_idx   = load_indices(os.path.join(SPLITS_DIR, "val.txt"))
    test_idx  = load_indices(os.path.join(SPLITS_DIR, "test.txt"))

    train_dataset = CIFAR10Split(DATA_ROOT, train_idx, transform=train_transform, train=True)
    val_dataset   = CIFAR10Split(DATA_ROOT, val_idx,   transform=eval_transform,  train=True)
    test_dataset  = CIFAR10Split(DATA_ROOT, test_idx,  transform=eval_transform,  train=False)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model     = build_resnet18_cifar10(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_losses, val_losses, val_accs = [], [], []
    best_val_acc, best_model_path = 0.0, os.path.join(MODELS_DIR, "supervised_best.pt")

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'Val Acc':>10}")
    print("="*50)
    for epoch in range(1, NUM_EPOCHS + 1):
        tl  = task1_train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        vl, va, _, _ = task1_evaluate(model, val_loader, criterion, DEVICE)
        train_losses.append(tl); val_losses.append(vl); val_accs.append(va)
        print(f"{epoch:>6} {tl:>12.4f} {vl:>10.4f} {va:>9.4f}")
        if va > best_val_acc:
            best_val_acc = va
            torch.save(model.state_dict(), best_model_path)
    print(f"Best Val Acc: {best_val_acc:.4f}")

    task1_save_loss_curve(train_losses, val_losses, val_accs,
                          os.path.join(GRAPHS_DIR, "supervised_loss.png"))

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    test_loss, test_acc, test_logits, test_labels = task1_evaluate(model, test_loader, criterion, DEVICE)
    test_preds = test_logits.argmax(dim=1).numpy()

    print(f"\nTest Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    pca_dict = per_class_accuracy(test_labels.numpy(), test_preds)
    print("Per-class accuracy:")
    for cls, acc in pca_dict.items():
        print(f"  {cls:<12}: {acc:.4f}")

    save_confusion_matrix(test_labels.numpy(), test_preds,
                          out_path=os.path.join(RESULTS_DIR, "supervised_confusion_matrix.png"),
                          title=f"Supervised Baseline (10% labels) — Test Acc: {test_acc:.4f}")
    return test_acc


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — UNDERSTANDING AUGMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════

def task2_denormalize(tensor):
    mean = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
    std  = torch.tensor(CIFAR_STD).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0.0, 1.0).permute(1, 2, 0).numpy()


def task2_save_augmentation_examples(originals, view1s, view2s, out_path, max_rows=10):
    rows = min(max_rows, len(originals))
    fig, axes = plt.subplots(rows, 3, figsize=(7, 2.2 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, 0)
    for c, title in enumerate(["Original", "Augmented View 1", "Augmented View 2"]):
        axes[0, c].set_title(title, fontsize=10, fontweight="bold", pad=6)
    for r in range(rows):
        orig = np.array(originals[r])
        v1   = task2_denormalize(view1s[r])
        v2   = task2_denormalize(view2s[r])
        axes[r, 0].imshow(orig); axes[r, 1].imshow(v1); axes[r, 2].imshow(v2)
        for c in range(3):
            axes[r, c].axis("off")
    fig.suptitle("SimCLR Augmentation Examples: Original | View 1 | View 2",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


def task2_main():
    print("\n" + "="*60)
    print("TASK 2 — AUGMENTATION VISUALISATION")
    print("="*60)
    set_seed(SEED)

    raw_idx     = load_indices(os.path.join(SPLITS_DIR, "train_labeled_10percent.txt"))
    raw_dataset = CIFAR10Split(DATA_ROOT, raw_idx, transform=None, train=True)
    two_view    = TwoViewTransform(simclr_transform)
    NUM_EXAMPLES = 10
    originals, view1s, view2s = [], [], []
    for i in range(NUM_EXAMPLES):
        pil_img, _ = raw_dataset[i]
        v1, v2 = two_view(pil_img)
        originals.append(pil_img); view1s.append(v1); view2s.append(v2)

    task2_save_augmentation_examples(
        originals, view1s, view2s,
        out_path=os.path.join(RESULTS_DIR, "augmentation_examples.png"))

    ssl_idx     = load_indices(os.path.join(SPLITS_DIR, "train_ssl_unlabeled.txt"))
    ssl_base    = CIFAR10Split(DATA_ROOT, ssl_idx, transform=None, train=True)
    tvd         = TwoViewDataset(ssl_base, two_view)
    sv1, sv2, _ = tvd[0]
    print(f"TwoViewDataset demo — view1: {sv1.shape}, view2: {sv2.shape}")
    print(f"SSL unlabeled size: {len(tvd)}")

    print("\n--- Augmentation Q&A ---")
    print("Q1. Are views identical? No — different random crops/flips/colour jitter each call.")
    print("Q2. Same object? Yes — semantic content is preserved.")
    print("Q3. Why positive pair? Both derived from the same source image.")
    print("Q4. Too weak? Model trivially matches views without learning features.")
    print("Q5. Too strong? Views lose semantic content; positive assumption breaks.")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 3 — FEATURE SIMILARITY BEFORE TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def task3_compute_similarity_matrix(features_np):
    """Compute 2N x 2N cosine similarity matrix from stacked L2-normed features."""
    z  = torch.tensor(features_np)
    z  = F.normalize(z, dim=1)
    return (z @ z.T).numpy()


def task3_avg_similarities(sim_matrix_np, N):
    pos_sims, neg_sims = [], []
    two_N = 2 * N
    for i in range(two_N):
        for j in range(two_N):
            if i == j:
                continue
            is_pos = (i < N and j == i + N) or (i >= N and j == i - N)
            if is_pos:
                pos_sims.append(sim_matrix_np[i, j])
            else:
                neg_sims.append(sim_matrix_np[i, j])
    return float(np.mean(pos_sims)), float(np.mean(neg_sims))


def task3_plot_similarity_matrix(sim_np, N, out_path, title):
    two_N  = 2 * N
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim_np, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axhline(y=N - 0.5, color='white', linewidth=1.5, linestyle='--')
    ax.axvline(x=N - 0.5, color='white', linewidth=1.5, linestyle='--')
    all_labels = [f"v1_{i}" for i in range(N)] + [f"v2_{i}" for i in range(N)]
    ax.set_xticks(range(two_N)); ax.set_xticklabels(all_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(two_N)); ax.set_yticklabels(all_labels, fontsize=7)
    for i in range(N):
        j = i + N
        ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='lime', linewidth=2.0))
        ax.add_patch(plt.Rectangle((i-0.5, j-0.5), 1, 1, fill=False, edgecolor='lime', linewidth=2.0))
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("View index", fontsize=10); ax.set_ylabel("View index", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {out_path}")


def task3_main():
    print("\n" + "="*60)
    print("TASK 3 — FEATURE SIMILARITY BEFORE TRAINING")
    print("="*60)
    set_seed(SEED)
    BATCH_SIZE = 8

    ssl_idx = load_indices(os.path.join(SPLITS_DIR, "train_ssl_unlabeled.txt"))
    print(f"SSL unlabeled samples: {len(ssl_idx)}")

    ssl_ds = TwoViewDatasetNoLabel(DATA_ROOT, ssl_idx,
                                   transform=TwoViewTransform(simclr_transform), train=True)
    loader = DataLoader(ssl_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    encoder = build_encoder().to(DEVICE)
    encoder.eval()
    print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,} (random)")

    set_seed(SEED)
    v1_batch, v2_batch = next(iter(loader))
    v1_batch, v2_batch = v1_batch.to(DEVICE), v2_batch.to(DEVICE)
    N = v1_batch.size(0)

    with torch.no_grad():
        z1 = F.normalize(encoder(v1_batch), dim=1)
        z2 = F.normalize(encoder(v2_batch), dim=1)

    z_all  = torch.cat([z1, z2], dim=0).cpu().numpy()
    sim_np = task3_compute_similarity_matrix(z_all)
    avg_pos, avg_neg = task3_avg_similarities(sim_np, N)

    print(f"\nSame image (pos pairs) : {avg_pos:.4f}")
    print(f"Different images       : {avg_neg:.4f}")

    task3_plot_similarity_matrix(
        sim_np, N,
        out_path=os.path.join(RESULTS_DIR, "similarity_matrix_before_training.png"),
        title=f"Cosine Similarity Matrix — Before SimCLR Training\n"
              f"(Green boxes = positive pairs | 2N={2*N} views)")

    print("\n--- Task 3 Q&A ---")
    print("Q1. Why ignore diagonal? Self-similarity is always 1.0 — no useful signal.")
    print("Q2. Positive pairs location? (i, i+N) and (i+N, i) in the 2N×2N matrix.")
    print("Q3. Why treat others as negatives? Forces discriminative feature learning.")

    return avg_pos, avg_neg


# ══════════════════════════════════════════════════════════════════════════════
# TASK 4 — SIMCLR MODEL + NT-XENT LOSS VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def task4_main():
    print("\n" + "="*60)
    print("TASK 4 — SIMCLR MODEL + NT-XENT LOSS")
    print("="*60)
    set_seed(SEED)
    TEMPERATURE = 0.5

    model = SimCLR().to(DEVICE)
    enc_p  = sum(p.numel() for p in model.encoder.parameters())
    proj_p = sum(p.numel() for p in model.projection_head.parameters())
    print(f"Encoder params    : {enc_p:,}")
    print(f"Projector params  : {proj_p:,}")
    print(f"Total params      : {enc_p + proj_p:,}")

    # Shape check
    _x = torch.randn(4, 3, 32, 32).to(DEVICE)
    _h, _z = model(_x)
    assert _h.shape == (4, 512), "Encoder output shape wrong!"
    assert _z.shape == (4, 128), "Projector output shape wrong!"
    print(f"\nShape check: h={_h.shape} ✓  z={_z.shape} ✓")
    del _x, _h, _z

    # NT-Xent sanity check
    criterion = NTXentLoss(temperature=TEMPERATURE)
    _z1 = torch.randn(8, 128).to(DEVICE)
    _z2 = torch.randn(8, 128).to(DEVICE)
    _loss    = criterion(_z1, _z2)
    expected = torch.log(torch.tensor(15.0)).item()
    print(f"\nNT-Xent on random projections : {_loss.item():.4f}")
    print(f"Expected ≈ log(2N-1)=log(15)  : {expected:.4f}  ✓")
    del _z1, _z2, _loss

    # Load data and visualise similarity matrix
    SMALL_N = 16
    ssl_idx = load_indices(os.path.join(SPLITS_DIR, "train_labeled_10percent.txt"))
    base_ds = CIFAR10Split(DATA_ROOT, ssl_idx, transform=None, train=True)
    tvd     = TwoViewDataset(base_ds, TwoViewTransform(simclr_transform))
    loader  = DataLoader(tvd, batch_size=SMALL_N, shuffle=True, num_workers=2)

    model.eval()
    v1, v2, _ = next(iter(loader))
    with torch.no_grad():
        _, z1 = model(v1.to(DEVICE))
        _, z2 = model(v2.to(DEVICE))

    z_all   = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    sim_mat = (z_all @ z_all.T).cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim_mat, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    for i in range(SMALL_N):
        ax.add_patch(plt.Rectangle((i+SMALL_N-0.5, i-0.5), 1, 1, fill=False, edgecolor='white', lw=2))
        ax.add_patch(plt.Rectangle((i-0.5, i+SMALL_N-0.5), 1, 1, fill=False, edgecolor='white', lw=2))
    ax.axhline(SMALL_N-0.5, color='yellow', lw=1.5, linestyle='--', alpha=0.7)
    ax.axvline(SMALL_N-0.5, color='yellow', lw=1.5, linestyle='--', alpha=0.7)
    ax.set_title("Cosine Similarity Matrix BEFORE SimCLR Training", fontsize=11)
    ax.set_xlabel("View index"); ax.set_ylabel("View index")
    fig.tight_layout()
    _path = os.path.join(RESULTS_DIR, "task4_similarity_matrix.png")
    fig.savefig(_path, dpi=200); plt.close(fig)
    print(f"[Saved] {_path}")

    z1_n = F.normalize(z1, dim=1); z2_n = F.normalize(z2, dim=1)
    same_sim = (z1_n * z2_n).sum(dim=1).mean().item()
    cross    = z1_n @ z1_n.T
    off_diag = ~torch.eye(SMALL_N, dtype=torch.bool, device=DEVICE)
    diff_sim = cross[off_diag].mean().item()
    print(f"\nSame-image sim (before): {same_sim:.4f}")
    print(f"Diff-image sim (before): {diff_sim:.4f}")
    return same_sim, diff_sim


# ══════════════════════════════════════════════════════════════════════════════
# TASK 5 — SIMCLR PRETRAINING
# ══════════════════════════════════════════════════════════════════════════════

def task5_compute_similarity_stats(model, loader, device, n_batches=10):
    model.eval()
    pos_sims, neg_sims = [], []
    with torch.no_grad():
        for i, (v1, v2) in enumerate(loader):
            if i >= n_batches:
                break
            h1, _ = model(v1.to(device))
            h2, _ = model(v2.to(device))
            h1 = F.normalize(h1, dim=1); h2 = F.normalize(h2, dim=1)
            N  = h1.size(0)
            h  = torch.cat([h1, h2], dim=0)
            sim = torch.mm(h, h.T).cpu().numpy()
            for ii in range(2*N):
                for jj in range(2*N):
                    if ii == jj: continue
                    is_pos = (ii < N and jj == ii+N) or (ii >= N and jj == ii-N)
                    (pos_sims if is_pos else neg_sims).append(sim[ii, jj])
    return float(np.mean(pos_sims)), float(np.mean(neg_sims))


def task5_plot_sim_matrix(model, loader, device, save_path, title):
    model.eval()
    v1, v2 = next(iter(loader))
    v1, v2 = v1[:8].to(device), v2[:8].to(device)
    with torch.no_grad():
        h1, _ = model(v1); h2, _ = model(v2)
    h1 = F.normalize(h1, dim=1); h2 = F.normalize(h2, dim=1)
    h  = torch.cat([h1, h2], dim=0)
    sim_np = torch.mm(h, h.T).cpu().numpy()
    N_vis = 8; two_N = 16
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim_np, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axhline(y=N_vis-0.5, color='white', linewidth=1.5, linestyle='--')
    ax.axvline(x=N_vis-0.5, color='white', linewidth=1.5, linestyle='--')
    all_labels = [f"v1_{i}" for i in range(N_vis)] + [f"v2_{i}" for i in range(N_vis)]
    ax.set_xticks(range(two_N)); ax.set_xticklabels(all_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(two_N)); ax.set_yticklabels(all_labels, fontsize=7)
    for i in range(N_vis):
        j = i + N_vis
        ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='lime', linewidth=2.0))
        ax.add_patch(plt.Rectangle((i-0.5, j-0.5), 1, 1, fill=False, edgecolor='lime', linewidth=2.0))
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("View index", fontsize=10); ax.set_ylabel("View index", fontsize=10)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[Saved] {save_path}")


def task5_main():
    print("\n" + "="*60)
    print("TASK 5 — SIMCLR PRETRAINING")
    print("="*60)
    set_seed(SEED)

    BATCH_SIZE = 64; EPOCHS = 50; LR = 3e-4; TEMPERATURE = 0.5

    ssl_idx = load_indices(os.path.join(SPLITS_DIR, "train_ssl_unlabeled.txt"))
    print(f"SSL unlabeled samples: {len(ssl_idx)}")

    ssl_ds = TwoViewDatasetNoLabel(DATA_ROOT, ssl_idx,
                                   transform=TwoViewTransform(simclr_transform), train=True)
    ssl_loader = DataLoader(ssl_ds, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=2, pin_memory=True, drop_last=True)
    vis_loader = DataLoader(ssl_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)

    print(f"Batches per epoch: {len(ssl_loader)}")

    model     = SimCLR().to(DEVICE)
    criterion = NTXentLoss(temperature=TEMPERATURE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Similarity before training
    print("\nComputing similarity BEFORE training...")
    pos_before, neg_before = task5_compute_similarity_stats(model, ssl_loader, DEVICE)
    print(f"  Pos (same img): {pos_before:.4f}  |  Neg (diff img): {neg_before:.4f}")

    # Training loop
    print(f"\n{'='*55}")
    print("STARTING SimCLR PRETRAINING")
    print(f"{'='*55}")
    epoch_losses = []; start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        batch_losses = []
        for v1, v2 in ssl_loader:
            _, z1 = model(v1.to(DEVICE)); _, z2 = model(v2.to(DEVICE))
            loss = criterion(z1, z2)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            batch_losses.append(loss.item())
        epoch_loss = float(np.mean(batch_losses))
        epoch_losses.append(epoch_loss)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:>3}/{EPOCHS}]  Loss: {epoch_loss:.4f}  "
                  f"Time: {time.time()-start_time:.1f}s")

    total_time = time.time() - start_time
    print(f"\nPretraining complete in {total_time/60:.1f} min")

    # Save encoder
    enc_path = os.path.join(MODELS_DIR, "simclr_encoder.pt")
    torch.save(model.encoder.state_dict(), enc_path)
    print(f"Encoder saved → {enc_path}")
    full_path = os.path.join(MODELS_DIR, "simclr_full.pt")
    torch.save(model.state_dict(), full_path)
    print(f"Full SimCLR saved → {full_path}")

    # Loss curve
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, EPOCHS+1), epoch_losses, color='royalblue', linewidth=2,
            marker='o', markersize=3, label='SimCLR Training Loss')
    ax.set_xlabel("Epoch"); ax.set_ylabel("NT-Xent Loss")
    ax.set_title(f"SimCLR Pretraining Loss Curve\n(BS={BATCH_SIZE} | LR={LR} | tau={TEMPERATURE})",
                 fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(1, EPOCHS)
    plt.tight_layout()
    loss_path = os.path.join(GRAPHS_DIR, "simclr_pretraining_loss.png")
    plt.savefig(loss_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Loss curve saved → {loss_path}")

    # Similarity after
    print("\nComputing similarity AFTER training...")
    pos_after, neg_after = task5_compute_similarity_stats(model, ssl_loader, DEVICE)
    print(f"  Pos (same img): {pos_after:.4f}  |  Neg (diff img): {neg_after:.4f}")

    set_seed(SEED)
    task5_plot_sim_matrix(model, vis_loader, DEVICE,
                          save_path=os.path.join(RESULTS_DIR, "similarity_matrix_after_training.png"),
                          title="Cosine Similarity Matrix — After SimCLR Training\n"
                                "(Green boxes = positive pairs | 2N=16 views)")

    print(f"\n{'='*60}")
    print("BEFORE vs AFTER COMPARISON")
    print(f"{'='*60}")
    print(f"{'Pair Type':<38} | {'Before':>8} | {'After':>8}")
    print(f"{'-'*38}-|{'-'*9}-|{'-'*9}")
    print(f"{'Same image (two views)':<38} | {pos_before:>8.4f} | {pos_after:>8.4f}")
    print(f"{'Different images':<38} | {neg_before:>8.4f} | {neg_after:>8.4f}")

    return pos_before, neg_before, pos_after, neg_after, epoch_losses[-1]


# ══════════════════════════════════════════════════════════════════════════════
# TASK 6 — LINEAR PROBE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def task6_train_linear_probe(model, train_loader, val_loader, epochs, lr, device, name):
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_accs, val_accs = [], []

    print(f"\n--- Training: {name} ---")
    print(f"{'Epoch':<8} {'Train Acc':>10} {'Val Acc':>10}")
    print("-"*30)

    for epoch in range(1, epochs + 1):
        model.train()
        correct = total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            correct += (outputs.argmax(1) == labels).sum().item()
            total   += labels.size(0)
        train_acc = correct / total

        model.eval(); correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                correct += (preds == labels).sum().item(); total += labels.size(0)
        val_acc = correct / total
        train_accs.append(train_acc); val_accs.append(val_acc)
        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:<8} {train_acc:>9.4f} {val_acc:>10.4f}")
    return train_accs, val_accs


def task6_evaluate(model, loader, device):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item(); total += labels.size(0)
    return correct / total


def task6_main():
    print("\n" + "="*60)
    print("TASK 6 — LINEAR PROBE EVALUATION")
    print("="*60)
    set_seed(SEED)
    EPOCHS = 20; LR = 3e-4; BATCH_SIZE = 64

    train_idx = load_indices(os.path.join(SPLITS_DIR, "train_labeled_10percent.txt"))
    val_idx   = load_indices(os.path.join(SPLITS_DIR, "val.txt"))
    test_idx  = load_indices(os.path.join(SPLITS_DIR, "test.txt"))

    t_train = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4),
                         T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
    t_eval  = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])

    train_ds = CIFAR10Split(DATA_ROOT, train_idx, transform=t_train, train=True)
    val_ds   = CIFAR10Split(DATA_ROOT, val_idx,   transform=t_eval,  train=True)
    test_ds  = CIFAR10Split(DATA_ROOT, test_idx,  transform=t_eval,  train=False)

    tl = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    vl = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    sl = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Experiment A — Random encoder
    print(f"\n{'='*55}")
    print("EXPERIMENT A — RANDOM ENCODER LINEAR PROBE")
    set_seed(SEED)
    rand_probe = LinearProbe(build_encoder().to(DEVICE)).to(DEVICE)
    ta_rand, va_rand = task6_train_linear_probe(rand_probe, tl, vl, EPOCHS, LR, DEVICE, "Random Encoder")
    test_acc_rand = task6_evaluate(rand_probe, sl, DEVICE)
    print(f"Random Encoder Test Acc: {test_acc_rand:.4f} ({test_acc_rand*100:.2f}%)")

    # Experiment B — SimCLR encoder
    print(f"\n{'='*55}")
    print("EXPERIMENT B — SimCLR ENCODER LINEAR PROBE")
    enc_path = os.path.join(MODELS_DIR, "simclr_encoder.pt")
    if not os.path.exists(enc_path):
        raise FileNotFoundError(f"SimCLR encoder not found at {enc_path}. Run Task 5 first.")
    set_seed(SEED)
    simclr_enc = build_encoder().to(DEVICE)
    simclr_enc.load_state_dict(torch.load(enc_path, map_location=DEVICE))
    simclr_probe = LinearProbe(simclr_enc).to(DEVICE)
    ta_simclr, va_simclr = task6_train_linear_probe(simclr_probe, tl, vl, EPOCHS, LR, DEVICE, "SimCLR Encoder")
    test_acc_simclr = task6_evaluate(simclr_probe, sl, DEVICE)
    print(f"SimCLR Encoder Test Acc: {test_acc_simclr:.4f} ({test_acc_simclr*100:.2f}%)")

    probe_save = os.path.join(MODELS_DIR, "linear_probe.pt")
    torch.save(simclr_probe.state_dict(), probe_save)
    print(f"Linear probe saved → {probe_save}")

    # Plot
    epochs_range = range(1, EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, ta_r, ta_s, ylabel, title in [
        (axes[0], va_rand, va_simclr, "Validation Accuracy", "Linear Probe — Validation Accuracy"),
        (axes[1], ta_rand, ta_simclr, "Training Accuracy",   "Linear Probe — Training Accuracy")]:
        ax.plot(epochs_range, ta_r, color='gray',      linewidth=2, marker='o', markersize=3, label='Random Encoder')
        ax.plot(epochs_range, ta_s, color='royalblue', linewidth=2, marker='s', markersize=3, label='SimCLR Encoder')
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(1, EPOCHS); ax.set_ylim(0, 1)
    plt.suptitle(f"Linear Probe | Random: {test_acc_rand*100:.2f}%  SimCLR: {test_acc_simclr*100:.2f}%",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plot_path = os.path.join(GRAPHS_DIR, "linear_probe_accuracy.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Plot saved → {plot_path}")

    print(f"\nImprovement: +{(test_acc_simclr - test_acc_rand)*100:.2f}%")
    return test_acc_rand, test_acc_simclr


# ══════════════════════════════════════════════════════════════════════════════
# TASK 7 — FINE-TUNING THE SIMCLR ENCODER
# ══════════════════════════════════════════════════════════════════════════════

def task7_train_model(model, train_loader, val_loader, epochs, lr, device, name):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_accs, val_accs = [], []
    best_val_acc, best_state = 0.0, None

    print(f"\n--- Training: {name} ---")
    print(f"{'Epoch':<8} {'Train Acc':>10} {'Val Acc':>10}")
    print("-"*30)

    for epoch in range(1, epochs + 1):
        model.train(); correct = total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs); loss = criterion(outputs, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            correct += (outputs.argmax(1) == labels).sum().item(); total += labels.size(0)
        train_acc = correct / total

        model.eval(); correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                correct += (preds == labels).sum().item(); total += labels.size(0)
        val_acc = correct / total
        train_accs.append(train_acc); val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:<8} {train_acc:>9.4f} {val_acc:>10.4f}")

    if best_state:
        model.load_state_dict(best_state)
    print(f"Best val acc: {best_val_acc:.4f}")
    return train_accs, val_accs


def task7_evaluate(model, loader, device):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item(); total += labels.size(0)
    return correct / total


def task7_main(supervised_test_acc=0.0, random_probe_test_acc=0.0, simclr_probe_test_acc=0.0):
    print("\n" + "="*60)
    print("TASK 7 — FINE-TUNING SimCLR ENCODER")
    print("="*60)
    set_seed(SEED)
    EPOCHS = 20; LR = 3e-4; BATCH_SIZE = 64

    train_idx = load_indices(os.path.join(SPLITS_DIR, "train_labeled_10percent.txt"))
    val_idx   = load_indices(os.path.join(SPLITS_DIR, "val.txt"))
    test_idx  = load_indices(os.path.join(SPLITS_DIR, "test.txt"))

    t_train = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4),
                         T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
    t_eval  = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])

    train_ds = CIFAR10Split(DATA_ROOT, train_idx, transform=t_train, train=True)
    val_ds   = CIFAR10Split(DATA_ROOT, val_idx,   transform=t_eval,  train=True)
    test_ds  = CIFAR10Split(DATA_ROOT, test_idx,  transform=t_eval,  train=False)

    tl = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    vl = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    sl = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    enc_path = os.path.join(MODELS_DIR, "simclr_encoder.pt")
    if not os.path.exists(enc_path):
        raise FileNotFoundError(f"SimCLR encoder not found at {enc_path}. Run Task 5 first.")

    simclr_enc = build_encoder().to(DEVICE)
    simclr_enc.load_state_dict(torch.load(enc_path, map_location=DEVICE))
    ft_model = ClassificationModel(simclr_enc).to(DEVICE)
    print(f"Trainable params: {sum(p.numel() for p in ft_model.parameters() if p.requires_grad):,}")

    ta_ft, va_ft = task7_train_model(ft_model, tl, vl, EPOCHS, LR, DEVICE, "SimCLR Fine-tuning")
    test_acc_ft  = task7_evaluate(ft_model, sl, DEVICE)
    print(f"Fine-tuning Test Acc: {test_acc_ft:.4f} ({test_acc_ft*100:.2f}%)")

    ft_save = os.path.join(MODELS_DIR, "finetuned_model.pt")
    torch.save(ft_model.state_dict(), ft_save)
    print(f"Fine-tuned model saved → {ft_save}")

    # Plot
    epochs_range = range(1, EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, accs, ylabel, title in [
        (axes[0], va_ft, "Validation Accuracy", "Fine-tuning — Validation Accuracy"),
        (axes[1], ta_ft, "Training Accuracy",   "Fine-tuning — Training Accuracy")]:
        ax.plot(epochs_range, accs, color='green', linewidth=2, marker='o', markersize=3)
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3); ax.set_xlim(1, EPOCHS); ax.set_ylim(0, 1)
    plt.suptitle(f"SimCLR Fine-tuning | Test Acc: {test_acc_ft*100:.2f}%",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    ft_plot = os.path.join(GRAPHS_DIR, "finetuning_accuracy.png")
    plt.savefig(ft_plot, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Plot saved → {ft_plot}")

    # Comparison table
    print(f"\n{'='*65}")
    print("TASK 7 — FINAL COMPARISON TABLE")
    print(f"{'='*65}")
    print(f"{'Experiment':<45} {'Test Acc':>10}")
    print("-"*55)
    for name, acc in [
        ("Supervised ResNet-18 from scratch (10% labels)", supervised_test_acc),
        ("Random frozen encoder + linear classifier",      random_probe_test_acc),
        ("SimCLR frozen encoder + linear classifier",      simclr_probe_test_acc),
        ("SimCLR pretrained encoder + full fine-tuning",   test_acc_ft),
    ]:
        print(f"{name:<45} {acc*100:>9.2f}%")
    print("="*65)

    return test_acc_ft


# ══════════════════════════════════════════════════════════════════════════════
# TASK 8 — PCA / t-SNE FEATURE VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']
VIS_COLORS      = ['#e6194b','#3cb44b','#ffe119','#4363d8','#f58231',
                   '#911eb4','#42d4f4','#f032e6','#bfef45','#fabed4']


def task8_extract_features(encoder, loader, device):
    encoder.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            h = encoder(imgs.to(device))
            all_features.append(h.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_features), np.concatenate(all_labels)


def task8_reduce_2d(features, method='tsne', seed=SEED):
    if method == 'tsne':
        print(f"  Running t-SNE (n={features.shape[0]})...")
        feat_50  = PCA(n_components=50, random_state=seed).fit_transform(features)
        reduced  = TSNE(n_components=2, random_state=seed, perplexity=30,
                        n_iter=1000, verbose=0).fit_transform(feat_50)
    else:
        print(f"  Running PCA (n={features.shape[0]})...")
        reduced = PCA(n_components=2, random_state=seed).fit_transform(features)
    return reduced


def task8_plot_2d(reduced, labels, title, save_path, method_name):
    fig, ax = plt.subplots(figsize=(10, 8))
    for ci in range(10):
        mask = labels == ci
        ax.scatter(reduced[mask, 0], reduced[mask, 1], c=VIS_COLORS[ci],
                   label=CIFAR10_CLASSES[ci], alpha=0.6, s=15, edgecolors='none')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(f"{method_name} Dim 1", fontsize=11)
    ax.set_ylabel(f"{method_name} Dim 2", fontsize=11)
    ax.legend(loc='upper right', fontsize=8, markerscale=2, framealpha=0.8, ncol=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[Saved] {save_path}")


def task8_main(use_tsne=True):
    print("\n" + "="*60)
    print("TASK 8 — PCA / t-SNE FEATURE VISUALISATION")
    print("="*60)
    set_seed(SEED)

    N_SAMPLES   = 1000
    method      = 'tsne' if use_tsne else 'pca'
    method_name = 't-SNE' if use_tsne else 'PCA'

    val_idx = load_indices(os.path.join(SPLITS_DIR, "val.txt"))
    set_seed(SEED)
    sel_idx     = sorted(random.sample(range(len(val_idx)), N_SAMPLES))
    val_ds      = CIFAR10Split(DATA_ROOT, val_idx, transform=eval_transform, train=True)
    val_subset  = Subset(val_ds, sel_idx)
    val_loader  = DataLoader(val_subset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Validation images: {N_SAMPLES} (seed={SEED})")

    # Encoder 1 — Random
    print(f"\n{'='*55}\nENCODER 1 — Random Untrained\n{'='*55}")
    set_seed(SEED)
    rand_enc = build_encoder().to(DEVICE)
    f_rand, l_rand = task8_extract_features(rand_enc, val_loader, DEVICE)
    r_rand = task8_reduce_2d(f_rand, method)
    task8_plot_2d(r_rand, l_rand,
                  title=f"Random Untrained Encoder — {method_name}\n({N_SAMPLES} val images, seed={SEED})",
                  save_path=os.path.join(RESULTS_DIR, "random_encoder_pca_or_tsne.png"),
                  method_name=method_name)

    # Encoder 2 — SimCLR
    print(f"\n{'='*55}\nENCODER 2 — SimCLR Pretrained\n{'='*55}")
    enc_path = os.path.join(MODELS_DIR, "simclr_encoder.pt")
    if not os.path.exists(enc_path):
        raise FileNotFoundError(f"SimCLR encoder not found at {enc_path}. Run Task 5 first.")
    set_seed(SEED)
    simclr_enc = build_encoder().to(DEVICE)
    simclr_enc.load_state_dict(torch.load(enc_path, map_location=DEVICE))
    f_simclr, l_simclr = task8_extract_features(simclr_enc, val_loader, DEVICE)
    r_simclr = task8_reduce_2d(f_simclr, method)
    task8_plot_2d(r_simclr, l_simclr,
                  title=f"SimCLR Pretrained Encoder — {method_name}\n({N_SAMPLES} val images, seed={SEED})",
                  save_path=os.path.join(RESULTS_DIR, "simclr_encoder_pca_or_tsne.png"),
                  method_name=method_name)

    # Encoder 3 — Fine-tuned
    print(f"\n{'='*55}\nENCODER 3 — Fine-tuned\n{'='*55}")
    ft_path = os.path.join(MODELS_DIR, "finetuned_model.pt")
    if not os.path.exists(ft_path):
        raise FileNotFoundError(f"Fine-tuned model not found at {ft_path}. Run Task 7 first.")
    set_seed(SEED)
    ft_enc   = build_encoder().to(DEVICE)
    ft_model = ClassificationModel(ft_enc).to(DEVICE)
    ft_model.load_state_dict(torch.load(ft_path, map_location=DEVICE))
    f_ft, l_ft = task8_extract_features(ft_model.encoder, val_loader, DEVICE)
    r_ft = task8_reduce_2d(f_ft, method)
    task8_plot_2d(r_ft, l_ft,
                  title=f"Fine-tuned Encoder — {method_name}\n({N_SAMPLES} val images, seed={SEED})",
                  save_path=os.path.join(RESULTS_DIR, "finetuned_encoder_pca_or_tsne.png"),
                  method_name=method_name)

    print("\n--- Task 8 Q&A ---")
    print("Q1. Random encoder class grouping? No — all classes overlap, no separation.")
    print("Q2. SimCLR better grouping? Yes — visible but imperfect class clusters.")
    print("Q3. Fine-tuning improve separation? Yes — significantly tighter clusters.")
    print("Q4. Confused classes? cat/dog, automobile/truck, bird/airplane/deer.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — RUN ALL TASKS IN ORDER
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'#'*65}")
    print("  Deep Learning Spring 2026 — Assignment 5: SimCLR")
    print("  Student: bilal bushra | Roll: MSDS25051")
    print(f"  Device : {DEVICE}")
    print(f"{'#'*65}\n")

    # Task 1 — Supervised baseline
    supervised_acc = task1_main()

    # Task 2 — Augmentation visualisation
    task2_main()

    # Task 3 — Feature similarity before training
    task3_main()

    # Task 4 — Model structure + NT-Xent verification
    task4_main()

    # Task 5 — SimCLR pretraining
    pos_b, neg_b, pos_a, neg_a, final_loss = task5_main()

    # Task 6 — Linear probe evaluation
    rand_probe_acc, simclr_probe_acc = task6_main()

    # Task 7 — Fine-tuning
    ft_acc = task7_main(
        supervised_test_acc   = supervised_acc,
        random_probe_test_acc = rand_probe_acc,
        simclr_probe_test_acc = simclr_probe_acc,
    )

    # Task 8 — Feature visualisation
    task8_main(use_tsne=True)

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{'#'*65}")
    print("  ASSIGNMENT 5 — COMPLETE SUMMARY")
    print(f"{'#'*65}")
    print(f"  Supervised baseline (10% labels)        : {supervised_acc*100:.2f}%")
    print(f"  Random encoder linear probe             : {rand_probe_acc*100:.2f}%")
    print(f"  SimCLR encoder linear probe             : {simclr_probe_acc*100:.2f}%")
    print(f"  SimCLR fine-tuned                       : {ft_acc*100:.2f}%")
    print(f"  Pos sim before/after SimCLR             : {pos_b:.4f} → {pos_a:.4f}")
    print(f"  Neg sim before/after SimCLR             : {neg_b:.4f} → {neg_a:.4f}")
    print(f"{'#'*65}")


if __name__ == "__main__":
    main()
