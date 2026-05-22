"""
Task 1: Supervised Baseline with Limited Labels
Deep Learning Spring 2026 - Assignment 5 SimCLR
Checkpoint 1

Trains a ResNet-18 from scratch using only 10% labeled CIFAR-10 split.
Saves:
  graphs/supervised_loss.png
  results/supervised_confusion_matrix.png
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.models as models

from utils.seed import set_seed
from utils.dataset_splits import get_cifar10_subset
from utils.metrics import (
    top1_accuracy_from_logits,
    save_confusion_matrix,
    per_class_accuracy,
)

import matplotlib.pyplot as plt
import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.path.join(BASE_DIR, "data")
SPLITS_DIR = os.path.join(BASE_DIR, "splits")
GRAPHS_DIR = os.path.join(BASE_DIR, "graphs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR  = os.path.join(BASE_DIR, "models")

for d in [DATA_ROOT, GRAPHS_DIR, RESULTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Hyperparameters ──────────────────────────────────────────────────────────
SEED        = 2026
BATCH_SIZE  = 64
NUM_EPOCHS  = 20       # as specified in assignment for supervised baseline
LR          = 3e-4
NUM_CLASSES = 10
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Transforms ───────────────────────────────────────────────────────────────
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

train_transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])

eval_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])


# ─── Model ────────────────────────────────────────────────────────────────────
def build_resnet18_cifar10(num_classes: int = 10) -> nn.Module:
    """ResNet-18 modified for CIFAR-10 (32x32 images).
    
    Changes from ImageNet version:
      - conv1: 3x3 kernel, stride 1, padding 1 (instead of 7x7, stride 2)
      - Remove maxpool layer
      - Replace fc with Linear(512 -> num_classes)
    """
    model = models.resnet18(weights=None)
    # Modify first conv for 32x32 input
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove maxpool
    model.maxpool = nn.Identity()
    # Replace classifier head
    model.fc = nn.Linear(512, num_classes)
    return model


# ─── Training / Evaluation helpers ────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    avg_loss = total_loss / len(loader.dataset)
    acc = top1_accuracy_from_logits(all_logits, all_labels)
    return avg_loss, acc, all_logits, all_labels


def save_loss_curve(train_losses, val_losses, val_accs, out_path):
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_losses, label="Train Loss", color="steelblue")
    axes[0].plot(epochs, val_losses, label="Val Loss", color="coral")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Supervised Baseline — Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_accs, label="Val Accuracy", color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Supervised Baseline — Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[Saved] {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    set_seed(SEED)
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = get_cifar10_subset(
        data_root=DATA_ROOT,
        split_file=os.path.join(SPLITS_DIR, "train_labeled_10percent.txt"),
        train=True,
        transform=train_transform,
        download=True,
    )
    val_dataset = get_cifar10_subset(
        data_root=DATA_ROOT,
        split_file=os.path.join(SPLITS_DIR, "val.txt"),
        train=True,
        transform=eval_transform,
        download=False,
    )
    test_dataset = get_cifar10_subset(
        data_root=DATA_ROOT,
        split_file=os.path.join(SPLITS_DIR, "test.txt"),
        train=False,
        transform=eval_transform,
        download=False,
    )

    print(f"Train samples : {len(train_dataset)}")
    print(f"Val   samples : {len(val_dataset)}")
    print(f"Test  samples : {len(test_dataset)}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # ── Model, loss, optimizer ────────────────────────────────────────────────
    model = build_resnet18_cifar10(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # ── Training loop ─────────────────────────────────────────────────────────
    train_losses, val_losses, val_accs = [], [], []
    best_val_acc = 0.0
    best_model_path = os.path.join(MODELS_DIR, "supervised_best.pt")

    print("\n" + "=" * 60)
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'Val Acc':>10}")
    print("=" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"{epoch:>6} {train_loss:>12.4f} {val_loss:>10.4f} {val_acc:>9.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    print("=" * 60)
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

    # ── Save loss / accuracy curves ───────────────────────────────────────────
    save_loss_curve(
        train_losses, val_losses, val_accs,
        os.path.join(GRAPHS_DIR, "supervised_loss.png"),
    )

    # ── Test evaluation with best model ──────────────────────────────────────
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    test_loss, test_acc, test_logits, test_labels = evaluate(
        model, test_loader, criterion, DEVICE
    )
    test_preds = test_logits.argmax(dim=1).numpy()
    test_labels_np = test_labels.numpy()

    print(f"\nTest Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")

    # Per-class accuracy
    pca_dict = per_class_accuracy(test_labels_np, test_preds)
    print("\nPer-class accuracy on test set:")
    for cls, acc in pca_dict.items():
        print(f"  {cls:<12}: {acc:.4f}")

    # ── Save confusion matrix ─────────────────────────────────────────────────
    save_confusion_matrix(
        test_labels_np, test_preds,
        out_path=os.path.join(RESULTS_DIR, "supervised_confusion_matrix.png"),
        title=f"Supervised Baseline (10% labels) — Test Accuracy: {test_acc:.4f}",
    )

    print(f"\n[Summary]")
    print(f"  Model: ResNet-18 from scratch, 10% labeled CIFAR-10")
    print(f"  Train samples : 5000")
    print(f"  Val   samples : 5000")
    print(f"  Test  samples : 10000")
    print(f"  Test Accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")

    return test_acc


if __name__ == "__main__":
    main()
