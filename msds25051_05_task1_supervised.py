"""
Task 1: Supervised Baseline with Limited Labels
================================================
Trains ResNet-18 from scratch using only the fixed 10% labeled split.
Outputs:
    graphs/supervised_loss.png
    results/supervised_confusion_matrix.png
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.seed import set_seed
from utils.dataset_splits import get_cifar10_with_transform, load_split_indices
from utils.metrics import (compute_accuracy, plot_confusion_matrix,
                            save_metrics, load_metrics)

# ── Paths ──────────────────────────────────────────────────────────────────
SPLIT_DIR   = "splits"
DATA_DIR    = "./data"
GRAPH_DIR   = "graphs"
RESULT_DIR  = "results"
MODEL_DIR   = "models"
os.makedirs(GRAPH_DIR,  exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────
SEED        = 2026
BATCH_SIZE  = 64
EPOCHS      = 30
LR          = 3e-4
NUM_CLASSES = 10
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Transforms ─────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2470, 0.2435, 0.2616)),
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2470, 0.2435, 0.2616)),
])


# ── Model ──────────────────────────────────────────────────────────────────
def build_resnet18_cifar(num_classes=10):
    """
    ResNet-18 modified for CIFAR-10:
      - conv1: 3x3, stride=1, padding=1  (instead of 7x7, stride=2)
      - maxpool removed
      - final fc: 512 -> num_classes
    """
    model = models.resnet18(weights=None)
    # Replace first conv
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                             stride=1, padding=1, bias=False)
    # Remove maxpool
    model.maxpool = nn.Identity()
    # Replace classifier head
    model.fc = nn.Linear(512, num_classes)
    return model


# ── Training & Evaluation helpers ─────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss    += loss.item() * images.size(0)
        preds          = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total         += images.size(0)

    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        total_loss    += loss.item() * images.size(0)
        preds          = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total         += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (total_loss / total,
            total_correct / total,
            all_preds,
            all_labels)


# ── Plot loss curves ───────────────────────────────────────────────────────
def plot_loss_curves(train_losses, val_losses, save_path):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", marker="o", markersize=3)
    plt.plot(epochs, val_losses,   label="Val Loss",   marker="s", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Supervised Baseline — Training & Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] Loss curve saved to {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    set_seed(SEED)
    print(f"\n[Device] Using: {DEVICE}")

    # ── Datasets & Loaders ────────────────────────────────────────────────
    train_dataset = get_cifar10_with_transform(train_transform,
                                               split_dir=SPLIT_DIR,
                                               data_dir=DATA_DIR,
                                               split="labeled")
    val_dataset   = get_cifar10_with_transform(eval_transform,
                                               split_dir=SPLIT_DIR,
                                               data_dir=DATA_DIR,
                                               split="val")
    test_dataset  = get_cifar10_with_transform(eval_transform,
                                               split_dir=SPLIT_DIR,
                                               data_dir=DATA_DIR,
                                               split="test")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"[Data] Train labeled: {len(train_dataset)} | "
          f"Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # ── Model, Loss, Optimizer ────────────────────────────────────────────
    model     = build_resnet18_cifar(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Training Loop ─────────────────────────────────────────────────────
    train_losses, val_losses = [], []
    best_val_acc = 0.0

    print(f"\n[Train] Starting supervised baseline for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc          = train_one_epoch(model, train_loader,
                                                   criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _  = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        print(f"  Epoch [{epoch:02d}/{EPOCHS}] "
              f"Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, "supervised_best.pt"))

    # ── Final Test Evaluation ─────────────────────────────────────────────
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "supervised_best.pt"),
                                     map_location=DEVICE))
    _, test_acc, test_preds, test_labels = evaluate(model, test_loader,
                                                     criterion, DEVICE)
    print(f"\n[Result] Best Val Acc: {best_val_acc:.4f}")
    print(f"[Result] Final Test Accuracy: {test_acc:.4f}")

    # ── Save Outputs ──────────────────────────────────────────────────────
    plot_loss_curves(train_losses, val_losses,
                     os.path.join(GRAPH_DIR, "supervised_loss.png"))

    plot_confusion_matrix(test_labels, test_preds,
                          save_path=os.path.join(RESULT_DIR,
                                                  "supervised_confusion_matrix.png"),
                          title="Supervised Baseline — Confusion Matrix (Test Set)")

    # Update/create metrics.json
    metrics_path = os.path.join(RESULT_DIR, "metrics.json")
    try:
        metrics = load_metrics(metrics_path)
    except FileNotFoundError:
        metrics = {}
    metrics["supervised_10percent_test_acc"] = round(test_acc, 4)
    save_metrics(metrics, metrics_path)

    print("\n[Done] Checkpoint 1 — Task 1 complete.")
    print(f"  graphs/supervised_loss.png")
    print(f"  results/supervised_confusion_matrix.png")
    print(f"  results/metrics.json  (supervised_10percent_test_acc updated)")


if __name__ == "__main__":
    main()
