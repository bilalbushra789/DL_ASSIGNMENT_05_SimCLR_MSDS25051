"""
Deep Learning - Spring 2026
Assignment 5 - Task 5: SimCLR Pretraining

Requirements:
  - Train SimCLR on unlabeled CIFAR-10 (train_ssl_unlabeled.txt)
  - Labels must NOT be used during training
  - Save: graphs/simclr_pretraining_loss.png
  - Save: results/similarity_matrix_after_training.png
  - Compute feature similarity before vs after training
  - Save trained encoder: models/simclr_encoder.pt
  - Random Seed : 2026

Fixed Training Settings (Section 6):
  - Batch size  : 64 (use 32 if GPU cannot support 64)
  - Epochs      : 50
  - Optimizer   : Adam
  - LR          : 3e-4
  - Temperature : 0.5
"""

import os
import random
import time
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

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

# ─────────────────────────────────────────
# 3.  FIXED TRAINING SETTINGS  
# ─────────────────────────────────────────
BATCH_SIZE  = 64       
EPOCHS      = 50
LR          = 3e-4
TEMPERATURE = 0.5

print(f"\n{'='*55}")
print("FIXED TRAINING SETTINGS")
print(f"{'='*55}")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Epochs       : {EPOCHS}")
print(f"Optimizer    : Adam")
print(f"LR           : {LR}")
print(f"Temperature  : {TEMPERATURE}")
print(f"Seed         : {SEED}")
print(f"{'='*55}\n")

# ─────────────────────────────────────────
# 4.  AUGMENTATION PIPELINE  
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
# 5.  TWO-VIEW TRANSFORM
# ─────────────────────────────────────────
class TwoViewTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2

# ─────────────────────────────────────────
# 6.  DATASET
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
            return result[0], result[1]   # view1, view2 — NO label
        return img_pil

# ─────────────────────────────────────────
# 7.  LOAD UNLABELED SSL SPLIT
# ─────────────────────────────────────────
ssl_indices = load_indices(os.path.join(SPLITS_DIR, "train_ssl_unlabeled.txt"))
print(f"SSL unlabeled samples : {len(ssl_indices)}")

ssl_dataset = CIFAR10Split(
    root="data",
    indices=ssl_indices,
    transform=TwoViewTransform(simclr_transform),
    train=True
)

ssl_loader = DataLoader(
    ssl_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True    # drop last incomplete batch for stable NT-Xent
)

print(f"Batches per epoch     : {len(ssl_loader)}\n")

# ─────────────────────────────────────────
# 8.  MODEL: ENCODER + PROJECTION HEAD
#     
# ─────────────────────────────────────────
def build_encoder():
    """ResNet-18 modified for CIFAR-10 — random weights."""
    model = models.resnet18(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3,
                              stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Identity()
    return model   # output: (batch, 512)

class ProjectionHead(nn.Module):
    """Linear(512->256) -> ReLU -> Linear(256->128)"""
    def __init__(self, in_dim=512, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder         = build_encoder()
        self.projection_head = ProjectionHead()

    def forward(self, x):
        h = self.encoder(x)          # (batch, 512)
        z = self.projection_head(h)  # (batch, 128)
        return h, z

# ─────────────────────────────────────────
# 9.  NT-Xent LOSS  
# ─────────────────────────────────────────
class NTXentLoss(nn.Module):
    """
    NT-Xent loss implemented from scratch.
    For positive pair (i, j):
      L(i,j) = -log [ exp(sim(z_i,z_j)/tau) /
                      sum_{k!=i} exp(sim(z_i,z_k)/tau) ]
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.tau = temperature

    def forward(self, z1, z2):
        N  = z1.size(0)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        z  = torch.cat([z1, z2], dim=0)          # (2N, D)

        sim = torch.mm(z, z.T) / self.tau         # (2N, 2N)

        # mask diagonal
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float('-inf'))

        # positive labels: view i → i+N, view i+N → i
        labels = torch.cat([
            torch.arange(N, 2 * N),
            torch.arange(0, N)
        ]).to(z.device)

        loss = F.cross_entropy(sim, labels)
        return loss

# ─────────────────────────────────────────
# 10.  COMPUTE SIMILARITY STATS
#      Used before AND after training
# ─────────────────────────────────────────
def compute_similarity_stats(model, loader, device, n_batches=10):
    """
    Compute average cosine similarity for:
      - positive pairs  (same image, two views)
      - negative pairs  (different images)
    Uses encoder features (512-dim), NOT projection.
    """
    model.eval()
    pos_sims = []
    neg_sims = []

    with torch.no_grad():
        for i, (v1, v2) in enumerate(loader):
            if i >= n_batches:
                break
            v1, v2 = v1.to(device), v2.to(device)
            h1, _  = model(v1)   # (N, 512)
            h2, _  = model(v2)

            h1 = F.normalize(h1, dim=1)
            h2 = F.normalize(h2, dim=1)

            N    = h1.size(0)
            h    = torch.cat([h1, h2], dim=0)        # (2N, 512)
            sim  = torch.mm(h, h.T).cpu().numpy()    # (2N, 2N)

            for ii in range(2 * N):
                for jj in range(2 * N):
                    if ii == jj:
                        continue
                    is_pos = (ii < N and jj == ii + N) or \
                             (ii >= N and jj == ii - N)
                    if is_pos:
                        pos_sims.append(sim[ii, jj])
                    else:
                        neg_sims.append(sim[ii, jj])

    return float(np.mean(pos_sims)), float(np.mean(neg_sims))

# ─────────────────────────────────────────
# 11.  SIMILARITY MATRIX HEATMAP
#      Used before AND after training
# ─────────────────────────────────────────
def plot_similarity_matrix(model, loader, device, save_path, title):
    """
    Plot 16x16 cosine similarity matrix (N=8 batch)
    with green boxes on positive pairs.
    """
    model.eval()
    v1, v2 = next(iter(loader))
    v1, v2 = v1[:8].to(device), v2[:8].to(device)   # N=8

    with torch.no_grad():
        h1, _ = model(v1)
        h2, _ = model(v2)

    h1 = F.normalize(h1, dim=1)
    h2 = F.normalize(h2, dim=1)
    h  = torch.cat([h1, h2], dim=0)             # (16, 512)

    sim_np = torch.mm(h, h.T).cpu().numpy()     # (16, 16)
    N_vis  = 8
    two_N  = 16

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim_np, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.axhline(y=N_vis - 0.5, color='white', linewidth=1.5, linestyle='--')
    ax.axvline(x=N_vis - 0.5, color='white', linewidth=1.5, linestyle='--')

    ticks      = list(range(two_N))
    all_labels = [f"v1_{i}" for i in range(N_vis)] + \
                 [f"v2_{i}" for i in range(N_vis)]
    ax.set_xticks(ticks); ax.set_xticklabels(all_labels, rotation=90, fontsize=7)
    ax.set_yticks(ticks); ax.set_yticklabels(all_labels, fontsize=7)

    # Green boxes = positive pairs
    for i in range(N_vis):
        j = i + N_vis
        ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                    fill=False, edgecolor='lime', linewidth=2.0))
        ax.add_patch(plt.Rectangle((i-0.5, j-0.5), 1, 1,
                                    fill=False, edgecolor='lime', linewidth=2.0))

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("View index", fontsize=10)
    ax.set_ylabel("View index", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {save_path}")

# ─────────────────────────────────────────
# 12.  BEFORE-TRAINING SIMILARITY
# ─────────────────────────────────────────
print("Computing similarity BEFORE training...")
model = SimCLR().to(DEVICE)

# Small loader for visualization (N=8)
vis_loader = DataLoader(
    ssl_dataset, batch_size=8, shuffle=True,
    num_workers=2, pin_memory=True
)

pos_before, neg_before = compute_similarity_stats(model, ssl_loader, DEVICE)
print(f"  Same image (pos pairs) : {pos_before:.4f}")
print(f"  Different images       : {neg_before:.4f}")

# ─────────────────────────────────────────
# 13.  SIMCLR PRETRAINING LOOP
# ─────────────────────────────────────────
criterion = NTXentLoss(temperature=TEMPERATURE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(f"\n{'='*55}")
print("STARTING SimCLR PRETRAINING")
print(f"{'='*55}")

epoch_losses = []
start_time   = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    batch_losses = []

    for view1, view2 in ssl_loader:
        view1 = view1.to(DEVICE)
        view2 = view2.to(DEVICE)

        # Forward pass — get projections (128-dim)
        _, z1 = model(view1)
        _, z2 = model(view2)

        # NT-Xent contrastive loss
        loss = criterion(z1, z2)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())

    epoch_loss = float(np.mean(batch_losses))
    epoch_losses.append(epoch_loss)

    # Print every 5 epochs
    if epoch % 5 == 0 or epoch == 1:
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch:>3}/{EPOCHS}]  "
              f"Loss: {epoch_loss:.4f}  "
              f"Time: {elapsed:.1f}s")

total_time = time.time() - start_time
print(f"\nPretraining complete! Total time: {total_time:.1f}s "
      f"({total_time/60:.1f} min)")

# ─────────────────────────────────────────
# 14.  SAVE TRAINED ENCODER
# ─────────────────────────────────────────
encoder_save = os.path.join(MODELS_DIR, "simclr_encoder.pt")
torch.save(model.encoder.state_dict(), encoder_save)
print(f"\nEncoder saved → {encoder_save}")

# Also save full SimCLR model (encoder + projection head)
simclr_full_save = os.path.join(MODELS_DIR, "simclr_full.pt")
torch.save(model.state_dict(), simclr_full_save)
print(f"Full SimCLR saved → {simclr_full_save}")

# ─────────────────────────────────────────
# 15.  PLOT LOSS CURVE
# ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(1, EPOCHS + 1), epoch_losses,
        color='royalblue', linewidth=2, marker='o',
        markersize=3, label='SimCLR Training Loss')
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("NT-Xent Loss", fontsize=12)
ax.set_title("SimCLR Pretraining Loss Curve\n"
             f"(CIFAR-10 unlabeled | BS={BATCH_SIZE} | "
             f"LR={LR} | tau={TEMPERATURE})",
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(1, EPOCHS)

# Annotate start and end loss
ax.annotate(f"Start: {epoch_losses[0]:.3f}",
            xy=(1, epoch_losses[0]),
            xytext=(5, epoch_losses[0] + 0.1),
            fontsize=9, color='green')
ax.annotate(f"End: {epoch_losses[-1]:.3f}",
            xy=(EPOCHS, epoch_losses[-1]),
            xytext=(EPOCHS - 10, epoch_losses[-1] + 0.1),
            fontsize=9, color='red')

plt.tight_layout()
loss_plot_path = os.path.join(GRAPHS_DIR, "simclr_pretraining_loss.png")
plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Loss curve saved → {loss_plot_path}")

# ─────────────────────────────────────────
# 16.  AFTER-TRAINING SIMILARITY
# ─────────────────────────────────────────
print("\nComputing similarity AFTER training...")
pos_after, neg_after = compute_similarity_stats(model, ssl_loader, DEVICE)
print(f"  Same image (pos pairs) : {pos_after:.4f}")
print(f"  Different images       : {neg_after:.4f}")

# ─────────────────────────────────────────
# 17.  SIMILARITY MATRIX — AFTER TRAINING
# ─────────────────────────────────────────
set_seed(SEED)
plot_similarity_matrix(
    model, vis_loader, DEVICE,
    save_path=os.path.join(RESULTS_DIR, "similarity_matrix_after_training.png"),
    title="Cosine Similarity Matrix — After SimCLR Training\n"
          "(Green boxes = positive pairs | 2N=16 views)"
)

# ─────────────────────────────────────────
# 18.  COMPARISON TABLE
# ─────────────────────────────────────────
print(f"\n{'='*60}")
print("FEATURE SIMILARITY — BEFORE vs AFTER SimCLR TRAINING")
print(f"{'='*60}")
print(f"{'Pair Type':<38} | {'Before':>8} | {'After':>8}")
print(f"{'-'*38}-|{'-'*9}-|{'-'*9}")
print(f"{'Same image, two augmented views':<38} | {pos_before:>8.4f} | {pos_after:>8.4f}")
print(f"{'Different images':<38} | {neg_before:>8.4f} | {neg_after:>8.4f}")
print(f"{'='*60}")

gap_before = pos_before - neg_before
gap_after  = pos_after  - neg_after
print(f"\nPositive-Negative gap BEFORE : {gap_before:.4f}")
print(f"Positive-Negative gap AFTER  : {gap_after:.4f}")

if gap_after > gap_before:
    print("SimCLR INCREASED the positive-negative gap. Training successful!")
else:
    print("Gap did not increase — try more epochs or check implementation.")

# ─────────────────────────────────────────
# 19.  FINAL SUMMARY
# ─────────────────────────────────────────
print(f"\n{'='*60}")
print("TASK 5 SUMMARY")
print(f"{'='*60}")
print(f"Encoder         : ResNet-18 CIFAR-10 modified")
print(f"Batch size      : {BATCH_SIZE}")
print(f"Epochs          : {EPOCHS}")
print(f"LR              : {LR}")
print(f"Temperature     : {TEMPERATURE}")
print(f"Training time   : {total_time/60:.1f} min")
print(f"Final loss      : {epoch_losses[-1]:.4f}")
print(f"Pos sim before  : {pos_before:.4f}")
print(f"Pos sim after   : {pos_after:.4f}")
print(f"Neg sim before  : {neg_before:.4f}")
print(f"Neg sim after   : {neg_after:.4f}")
print(f"\nSaved files:")
print(f"  {encoder_save}")
print(f"  {simclr_full_save}")
print(f"  {loss_plot_path}")
print(f"  results/similarity_matrix_after_training.png")
print(f"{'='*60}")
print("\nTask 5 complete")
