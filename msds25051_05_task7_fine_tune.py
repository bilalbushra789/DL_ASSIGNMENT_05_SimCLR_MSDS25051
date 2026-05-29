"""
Deep Learning - Spring 2026
Assignment 5 - Task 7: Fine-tuning the SimCLR Encoder

Requirements:
  - Initialize encoder with SimCLR pretrained weights
  - Train FULL model end-to-end (encoder + classifier)
  - Use only 10% labeled training data
  - Use: train_labeled_10percent.txt, val.txt, test.txt
  - Fine-tuning epochs : 20
  - Optimizer          : Adam
  - LR                 : 3e-4
  - Random Seed        : 2026
  - Save: graphs/finetuning_accuracy.png
  - Save: models/finetuned_model.pt

Final comparison table (Task 7 spec):
  Supervised ResNet-18 from scratch using 10% labels     ___
  Random frozen encoder + linear classifier              ___
  SimCLR frozen encoder + linear classifier              ___
  SimCLR pretrained encoder + full fine-tuning           ___
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
    torch.backends.cudnn.benchmark     = False

set_seed()

# ─────────────────────────────────────────
# 2.  PATHS & DEVICE
# ─────────────────────────────────────────
SPLITS_DIR  = "splits"
RESULTS_DIR = "results"
GRAPHS_DIR  = "graphs"
MODELS_DIR  = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR,  exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────
# 3.  FIXED SETTINGS  (Section 6)
# ─────────────────────────────────────────
EPOCHS     = 20
LR         = 3e-4
BATCH_SIZE = 64

print(f"\n{'='*55}")
print("FIXED SETTINGS — FINE-TUNING")
print(f"{'='*55}")
print(f"Epochs     : {EPOCHS}")
print(f"LR         : {LR}")
print(f"Batch size : {BATCH_SIZE}")
print(f"Optimizer  : Adam")
print(f"Seed       : {SEED}")
print(f"{'='*55}\n")

# ─────────────────────────────────────────
# 4.  TRANSFORMS
# ─────────────────────────────────────────
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

train_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomCrop(32, padding=4),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD)
])

eval_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD)
])

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
            img = self.transform(img_pil)
        else:
            img = img_pil
        return img, label

# ─────────────────────────────────────────
# 6.  LOAD SPLITS
# ─────────────────────────────────────────
train_indices = load_indices(os.path.join(SPLITS_DIR, "train_labeled_10percent.txt"))
val_indices   = load_indices(os.path.join(SPLITS_DIR, "val.txt"))
test_indices  = load_indices(os.path.join(SPLITS_DIR, "test.txt"))

print(f"Labeled train samples : {len(train_indices)}")
print(f"Val samples           : {len(val_indices)}")
print(f"Test samples          : {len(test_indices)}\n")

train_dataset = CIFAR10Split("data", train_indices, transform=train_transform, train=True)
val_dataset   = CIFAR10Split("data", val_indices,   transform=eval_transform,  train=True)
test_dataset  = CIFAR10Split("data", test_indices,  transform=eval_transform,  train=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)

# ─────────────────────────────────────────
# 7.  ENCODER — ResNet-18 modified for CIFAR-10
# ─────────────────────────────────────────
def build_encoder():
    model         = models.resnet18(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3,
                              stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Identity()
    return model   # output: (batch, 512)

# ─────────────────────────────────────────
# 8.  FULL CLASSIFICATION MODEL
#     Encoder + Linear(512->10)
#     For fine-tuning: BOTH encoder and classifier trained
#     For linear probe: encoder frozen, only classifier trained
# ─────────────────────────────────────────
class ClassificationModel(nn.Module):
    """
    Full model: encoder + classification head.
    Used for:
      - Supervised baseline (Task 1) — train from scratch
      - Fine-tuning (Task 7) — initialize with SimCLR weights,
        train everything end-to-end
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder    = encoder
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        h   = self.encoder(x)      # (batch, 512)
        out = self.classifier(h)   # (batch, 10)
        return out

# ─────────────────────────────────────────
# 9.  TRAINING FUNCTION (full model)
# ─────────────────────────────────────────
def train_model(model, train_loader, val_loader,
                epochs, lr, device, name):
    """
    Train the full model end-to-end.
    Both encoder and classifier parameters are updated.
    """
    # Optimize ALL parameters — encoder + classifier
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_accs = []
    val_accs   = []

    print(f"\n--- Training: {name} ---")
    print(f"{'Epoch':<8} {'Train Acc':>10} {'Val Acc':>10}")
    print("-" * 30)

    best_val_acc  = 0.0
    best_state    = None

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        correct = 0
        total   = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs      = model(imgs)
            loss         = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds    = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        train_acc = correct / total

        # ── Validate ──
        model.eval()
        correct = 0
        total   = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs      = model(imgs)
                preds        = outputs.argmax(dim=1)
                correct     += (preds == labels).sum().item()
                total       += labels.size(0)

        val_acc = correct / total

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:<8} {train_acc:>9.4f} {val_acc:>10.4f}")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"Best val acc: {best_val_acc:.4f}")
    return train_accs, val_accs

# ─────────────────────────────────────────
# 10.  EVALUATION FUNCTION
# ─────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs      = model(imgs)
            preds        = outputs.argmax(dim=1)
            correct     += (preds == labels).sum().item()
            total       += labels.size(0)

    return correct / total

# ═══════════════════════════════════════════════════════
# FINE-TUNING — SimCLR pretrained encoder + full training
# Initialize encoder with SimCLR weights
# Train ENTIRE model end-to-end
# ═══════════════════════════════════════════════════════
print(f"\n{'='*55}")
print("FINE-TUNING — SimCLR Pretrained Encoder")
print(f"{'='*55}")
print("Encoder     : SimCLR pretrained weights")
print("Training    : Full end-to-end (encoder + classifier)")
print("Labels      : 10% labeled split")

# Load SimCLR pretrained encoder
encoder_path = os.path.join(MODELS_DIR, "simclr_encoder.pt")
if not os.path.exists(encoder_path):
    raise FileNotFoundError(
        f"SimCLR encoder not found at {encoder_path}. "
        "Please run Task 5 first."
    )

set_seed(SEED)
simclr_encoder = build_encoder().to(DEVICE)
simclr_encoder.load_state_dict(
    torch.load(encoder_path, map_location=DEVICE)
)
print(f"Loaded SimCLR encoder from: {encoder_path}")

# Build full model — encoder NOT frozen
finetune_model = ClassificationModel(simclr_encoder).to(DEVICE)

total_params    = sum(p.numel() for p in finetune_model.parameters())
trainable_params = sum(p.numel() for p in finetune_model.parameters()
                       if p.requires_grad)
print(f"Total parameters     : {total_params:,}")
print(f"Trainable parameters : {trainable_params:,}  (ALL — encoder + classifier)")

train_accs_ft, val_accs_ft = train_model(
    finetune_model, train_loader, val_loader,
    EPOCHS, LR, DEVICE,
    "SimCLR Fine-tuning"
)

test_acc_ft = evaluate(finetune_model, test_loader, DEVICE)
print(f"\nSimCLR Fine-tuning — Test Accuracy: {test_acc_ft:.4f} "
      f"({test_acc_ft*100:.2f}%)")

# ─────────────────────────────────────────
# 11.  SAVE FINE-TUNED MODEL
# ─────────────────────────────────────────
ft_save = os.path.join(MODELS_DIR, "finetuned_model.pt")
torch.save(finetune_model.state_dict(), ft_save)
print(f"\nFine-tuned model saved → {ft_save}")

# ─────────────────────────────────────────
# 12.  LOAD PREVIOUS RESULTS
#      from Task 1 (supervised) and Task 6 (linear probes)
#      Enter your actual values from those tasks here
# ─────────────────────────────────────────
# ── Fill these from your Task 1 and Task 6 outputs ──
supervised_test_acc   = 0.0    # ← replace with Task 1 test accuracy
random_probe_test_acc = 0.2537 # ← from Task 6 output
simclr_probe_test_acc = 0.7316 # ← from Task 6 output

# ─────────────────────────────────────────
# 13.  PLOT — Fine-tuning accuracy curve
# ─────────────────────────────────────────
epochs_range = range(1, EPOCHS + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Left: Validation Accuracy ──
axes[0].plot(epochs_range, val_accs_ft,
             color='green', linewidth=2, marker='o',
             markersize=3, label='SimCLR Fine-tuning')
axes[0].set_xlabel("Epoch", fontsize=12)
axes[0].set_ylabel("Validation Accuracy", fontsize=12)
axes[0].set_title("Fine-tuning — Validation Accuracy",
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(1, EPOCHS)
axes[0].set_ylim(0, 1)

# ── Right: Training Accuracy ──
axes[1].plot(epochs_range, train_accs_ft,
             color='green', linewidth=2, marker='o',
             markersize=3, label='SimCLR Fine-tuning')
axes[1].set_xlabel("Epoch", fontsize=12)
axes[1].set_ylabel("Training Accuracy", fontsize=12)
axes[1].set_title("Fine-tuning — Training Accuracy",
                  fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(1, EPOCHS)
axes[1].set_ylim(0, 1)

plt.suptitle(
    f"SimCLR Fine-tuning  |  Full End-to-End  |  10% Labels\n"
    f"Fine-tuning Test Acc: {test_acc_ft*100:.2f}%",
    fontsize=12, fontweight='bold'
)
plt.tight_layout()

ft_plot_path = os.path.join(GRAPHS_DIR, "finetuning_accuracy.png")
plt.savefig(ft_plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot saved → {ft_plot_path}")

# ─────────────────────────────────────────
# 14.  FINAL COMPARISON TABLE  (Task 7 spec)
# ─────────────────────────────────────────
print(f"\n{'='*65}")
print("TASK 7 — FINAL COMPARISON TABLE")
print(f"{'='*65}")
print(f"{'Experiment':<45} {'Test Acc':>10}")
print(f"{'-'*55}")
print(f"{'Supervised ResNet-18 from scratch (10% labels)':<45} "
      f"{supervised_test_acc*100:>9.2f}%")
print(f"{'Random frozen encoder + linear classifier':<45} "
      f"{random_probe_test_acc*100:>9.2f}%")
print(f"{'SimCLR frozen encoder + linear classifier':<45} "
      f"{simclr_probe_test_acc*100:>9.2f}%")
print(f"{'SimCLR pretrained encoder + full fine-tuning':<45} "
      f"{test_acc_ft*100:>9.2f}%")
print(f"{'='*65}")

# ─────────────────────────────────────────
# 15.  FINAL SUMMARY
# ─────────────────────────────────────────
print(f"\n{'='*55}")
print("TASK 7 SUMMARY")
print(f"{'='*55}")
print(f"Encoder init     : SimCLR pretrained weights")
print(f"Encoder frozen?  : NO — full end-to-end training")
print(f"Trainable parts  : Encoder + Linear(512->10)")
print(f"Epochs           : {EPOCHS}")
print(f"LR               : {LR}")
print(f"Optimizer        : Adam")
print(f"Fine-tune acc    : {test_acc_ft*100:.2f}%")
print(f"\nSaved files:")
print(f"  {ft_save}")
print(f"  {ft_plot_path}")
print(f"{'='*55}")
print("\nTask 7 complete")