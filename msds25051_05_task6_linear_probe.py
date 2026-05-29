"""
Deep Learning - Spring 2026
Assignment 5 - Task 6: Linear Probe Evaluation

Sub-tasks:
  Experiment A: Random Encoder Linear Probe
    - Randomly initialized frozen ResNet-18 encoder
    - Train only Linear(512 -> 10)

  Experiment B: SimCLR Encoder Linear Probe
    - SimCLR pretrained frozen encoder
    - Train only Linear(512 -> 10)

Rules:
  - Encoder is FROZEN in both experiments
  - Only linear classifier is trained
  - Use: train_labeled_10percent.txt, val.txt, test.txt
  - Linear probing epochs : 20
  - Optimizer             : Adam
  - LR                    : 3e-4
  - Random Seed           : 2026
  - Save: graphs/linear_probe_accuracy.png
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
print("FIXED SETTINGS — LINEAR PROBING")
print(f"{'='*55}")
print(f"Epochs     : {EPOCHS}")
print(f"LR         : {LR}")
print(f"Batch size : {BATCH_SIZE}")
print(f"Optimizer  : Adam")
print(f"Seed       : {SEED}")
print(f"{'='*55}\n")

# ─────────────────────────────────────────
# 4.  TRANSFORMS
#     Standard eval transform — NO augmentation
#     for labeled classification
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
#     train_labeled_10percent, val, test
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
# 8.  LINEAR PROBE MODEL
#     Frozen encoder + trainable Linear(512->10)
# ─────────────────────────────────────────
class LinearProbe(nn.Module):
    """
    Frozen encoder + single linear classification head.
    Only the linear head is trained.
    Encoder gradients are completely disabled.
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder    = encoder
        self.classifier = nn.Linear(512, 10)

        # Freeze encoder — no gradients
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            h = self.encoder(x)   # (batch, 512) — frozen
        out = self.classifier(h)  # (batch, 10)  — trained
        return out

# ─────────────────────────────────────────
# 9.  TRAINING FUNCTION
# ─────────────────────────────────────────
def train_linear_probe(model, train_loader, val_loader,
                       epochs, lr, device, name):
    """
    Train only the linear classifier head.
    Encoder is frozen — only classifier parameters are updated.
    """
    # Only optimize classifier parameters
    optimizer = torch.optim.Adam(
        model.classifier.parameters(), lr=lr
    )
    criterion = nn.CrossEntropyLoss()

    train_accs = []
    val_accs   = []

    print(f"\n--- Training: {name} ---")
    print(f"{'Epoch':<8} {'Train Acc':>10} {'Val Acc':>10}")
    print("-" * 30)

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

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:<8} {train_acc:>9.4f} {val_acc:>10.4f}")

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
# EXPERIMENT A — RANDOM ENCODER LINEAR PROBE
# Randomly initialized frozen encoder + Linear(512->10)
# ═══════════════════════════════════════════════════════
print(f"\n{'='*55}")
print("EXPERIMENT A — RANDOM ENCODER LINEAR PROBE")
print(f"{'='*55}")

set_seed(SEED)
random_encoder      = build_encoder().to(DEVICE)
random_probe_model  = LinearProbe(random_encoder).to(DEVICE)

# Verify encoder is frozen
trainable = sum(p.numel() for p in random_probe_model.parameters()
                if p.requires_grad)
print(f"Trainable parameters : {trainable:,}  (classifier only)")

train_accs_rand, val_accs_rand = train_linear_probe(
    random_probe_model, train_loader, val_loader,
    EPOCHS, LR, DEVICE, "Random Encoder"
)

test_acc_rand = evaluate(random_probe_model, test_loader, DEVICE)
print(f"\nRandom Encoder Linear Probe — Test Accuracy: {test_acc_rand:.4f} "
      f"({test_acc_rand*100:.2f}%)")

# ═══════════════════════════════════════════════════════
# EXPERIMENT B — SimCLR ENCODER LINEAR PROBE
# SimCLR pretrained frozen encoder + Linear(512->10)
# ═══════════════════════════════════════════════════════
print(f"\n{'='*55}")
print("EXPERIMENT B — SimCLR ENCODER LINEAR PROBE")
print(f"{'='*55}")

# Load SimCLR pretrained encoder weights
encoder_path = os.path.join(MODELS_DIR, "simclr_encoder.pt")
if not os.path.exists(encoder_path):
    raise FileNotFoundError(
        f"SimCLR encoder not found at {encoder_path}. "
        "Please run Task 5 (rollNumber_05_task5_pretraining.py) first."
    )

set_seed(SEED)
simclr_encoder = build_encoder().to(DEVICE)
simclr_encoder.load_state_dict(torch.load(encoder_path, map_location=DEVICE))
print(f"Loaded SimCLR encoder from: {encoder_path}")

simclr_probe_model = LinearProbe(simclr_encoder).to(DEVICE)

trainable = sum(p.numel() for p in simclr_probe_model.parameters()
                if p.requires_grad)
print(f"Trainable parameters : {trainable:,}  (classifier only)")

train_accs_simclr, val_accs_simclr = train_linear_probe(
    simclr_probe_model, train_loader, val_loader,
    EPOCHS, LR, DEVICE, "SimCLR Encoder"
)

test_acc_simclr = evaluate(simclr_probe_model, test_loader, DEVICE)
print(f"\nSimCLR Encoder Linear Probe — Test Accuracy: {test_acc_simclr:.4f} "
      f"({test_acc_simclr*100:.2f}%)")

# ─────────────────────────────────────────
# 11.  SAVE LINEAR PROBE MODEL
# ─────────────────────────────────────────
probe_save = os.path.join(MODELS_DIR, "linear_probe.pt")
torch.save(simclr_probe_model.state_dict(), probe_save)
print(f"\nLinear probe model saved → {probe_save}")

# ─────────────────────────────────────────
# 12.  PLOT ACCURACY CURVES
#      Both experiments on same plot
# ─────────────────────────────────────────
epochs_range = range(1, EPOCHS + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Left: Validation Accuracy Comparison ──
axes[0].plot(epochs_range, val_accs_rand,
             color='gray', linewidth=2, marker='o',
             markersize=3, label='Random Encoder')
axes[0].plot(epochs_range, val_accs_simclr,
             color='royalblue', linewidth=2, marker='s',
             markersize=3, label='SimCLR Encoder')
axes[0].set_xlabel("Epoch", fontsize=12)
axes[0].set_ylabel("Validation Accuracy", fontsize=12)
axes[0].set_title("Linear Probe — Validation Accuracy", fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(1, EPOCHS)
axes[0].set_ylim(0, 1)

# ── Right: Training Accuracy Comparison ──
axes[1].plot(epochs_range, train_accs_rand,
             color='gray', linewidth=2, marker='o',
             markersize=3, label='Random Encoder')
axes[1].plot(epochs_range, train_accs_simclr,
             color='royalblue', linewidth=2, marker='s',
             markersize=3, label='SimCLR Encoder')
axes[1].set_xlabel("Epoch", fontsize=12)
axes[1].set_ylabel("Training Accuracy", fontsize=12)
axes[1].set_title("Linear Probe — Training Accuracy", fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(1, EPOCHS)
axes[1].set_ylim(0, 1)

plt.suptitle(
    f"Linear Probe Evaluation  |  Frozen Encoder  |  10% Labels\n"
    f"Random Test Acc: {test_acc_rand*100:.2f}%   "
    f"SimCLR Test Acc: {test_acc_simclr*100:.2f}%",
    fontsize=12, fontweight='bold'
)
plt.tight_layout()

plot_path = os.path.join(GRAPHS_DIR, "linear_probe_accuracy.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot saved → {plot_path}")

# ─────────────────────────────────────────
# 13.  RESULTS TABLE
# ─────────────────────────────────────────
improvement = (test_acc_simclr - test_acc_rand) * 100

print(f"\n{'='*65}")
print("TASK 6 RESULTS TABLE")
print(f"{'='*65}")
print(f"{'Model':<35} {'Encoder':<20} {'Trainable Part':<18} {'Test Acc':>8}")
print(f"{'-'*65}")
print(f"{'Random Linear Probe':<35} {'Random frozen':<20} {'Linear only':<18} "
      f"{test_acc_rand*100:>7.2f}%")
print(f"{'SimCLR Linear Probe':<35} {'SimCLR frozen':<20} {'Linear only':<18} "
      f"{test_acc_simclr*100:>7.2f}%")
print(f"{'='*65}")
print(f"\nSimCLR improvement over random: +{improvement:.2f}%")

# ─────────────────────────────────────────
# 14.  FINAL SUMMARY
# ─────────────────────────────────────────
print(f"\n{'='*55}")
print("TASK 6 SUMMARY")
print(f"{'='*55}")
print(f"Encoder frozen?              : YES (both experiments)")
print(f"Trainable part               : Linear(512->10) only")
print(f"Epochs                       : {EPOCHS}")
print(f"LR                           : {LR}")
print(f"Optimizer                    : Adam")
print(f"Labels used during probing?  : YES (10% labeled split)")
print(f"Random encoder test acc      : {test_acc_rand*100:.2f}%")
print(f"SimCLR encoder test acc      : {test_acc_simclr*100:.2f}%")
print(f"Improvement                  : +{improvement:.2f}%")
print(f"\nSaved files:")
print(f"  {plot_path}")
print(f"  {probe_save}")
print(f"{'='*55}")
print("\nTask 6 complete. Proceed to Task 7 (Fine-tuning).")
