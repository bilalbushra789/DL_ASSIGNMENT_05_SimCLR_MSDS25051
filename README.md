# Assignment 5 SimCLR — Checkpoint 1

## Structure
```
checkpoint1/
├── splits/
│   ├── train_labeled_10percent.txt   # 5 000 indices
│   ├── val.txt                        # 5 000 indices
│   ├── test.txt                       # 10 000 indices
│   └── train_ssl_unlabeled.txt        # 45 000 indices
├── utils/
│   ├── seed.py
│   ├── dataset_splits.py
│   ├── metrics.py
│   └── visualization.py
├── task1_supervised.py
├── task2_augmentations.py
├── task3_similarity.py
└── allCode.py
```

## Tasks in Checkpoint 1

| File | Task |
|---|---|
| `task1_supervised.py` | Supervised ResNet-18 baseline, 10% labels |
| `task2_augmentations.py` | SimCLR augmentation pipeline + visualisation |
| `task3_similarity.py` | Cosine similarity before SimCLR training |
| `allCode.py` | Runs all three tasks in sequence |

## How to run

### Individual tasks
```bash
python task1_supervised.py
python task2_augmentations.py
python task3_similarity.py
```

## Expected outputs
```
graphs/supervised_loss.png
results/supervised_confusion_matrix.png
results/augmentation_examples.png
models/supervised_best.pt
```

## Dataset splits used
| Split | Samples |
|---|---|
| train_labeled_10percent | 5 000 |
| val | 5 000 |
| test | 10 000 |
| train_ssl_unlabeled | 45 000 |

## Settings
| Setting | Value |
|---|---|
| Seed | 2026 |
| Batch size | 64 |
| Supervised epochs | 20 |
| Optimizer | Adam |
| Learning rate | 3e-4 |
| Encoder | ResNet-18 CIFAR-10 modified |

## ResNet-18 CIFAR-10 modifications
- `conv1`: 3×3 kernel, stride 1, padding 1 (instead of 7×7 stride 2)
- `maxpool`: replaced with `nn.Identity()`
- `fc`: Linear(512 → 10) for supervised; Linear(512 → 512) identity for encoder-only tasks

## Requirements
```
torch
torchvision
scikit-learn
matplotlib
numpy
```
Install:
```bash
pip install torch torchvision scikit-learn matplotlib numpy
```

# Checkpoint 2 — Day 6
## SimCLR Core Components

### What Was Implemented
- Encoder (ResNet-18 modified for CIFAR-10)
- Projection Head
- Positive and Negative Pair Construction
- Cosine Similarity Matrix
- NT-Xent Loss

---

## Files Added This Checkpoint

```
msds25051_05_task4_simclr.py
results/
  similarity_matrix_before_training.png
```

---

## Task 4.1 — Encoder and Projection Head

### Encoder
ResNet-18 modified for CIFAR-10 (weights=None — random, NOT pretrained):
- `conv1` → 3×3 convolution, stride=1, padding=1
- `maxpool` → removed (Identity)
- `fc` → removed (Identity)
- Output: **512-dimensional feature vector**

```python
model = torchvision.models.resnet18(weights=None)
model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.fc      = nn.Identity()
```

### Projection Head
Used ONLY during SimCLR pretraining. Discarded after training.
```
Linear(512 → 256) → ReLU → Linear(256 → 128)
```

| Module | Parameters |
|---|---|
| Encoder (ResNet-18 CIFAR-10) | 11,168,832 |
| Projection Head (512→256→128) | 164,224 |
| Total | 11,333,056 |

---

## Task 4.2 — Positive and Negative Pair Construction

For a batch of N original images, TwoViewTransform produces 2N augmented views:
- Views `0 .. N-1` = View 1 of each image
- Views `N .. 2N-1` = View 2 of each image

**Positive pair:** `(i, i+N)` — two augmented views of the same image  
**Negative pairs:** all other `2N-2` views per anchor — labels never used

**Positive Pair Table**

| Original Image | View 1 Index | View 2 Index | Positive Pair |
|---|---|---|---|
| image 0 | 0 | 4 | yes |
| image 1 | 1 | 5 | yes |
| image 2 | 2 | 6 | yes |
| image 3 | 3 | 7 | yes |

---

## Task 4.3 — Cosine Similarity Matrix

- Computed cosine similarity matrix of size **2N × 2N**
- Used N=8 small batch → **16 × 16** matrix for visualization
- L2-normalized features → cosine similarity = dot product
- Visualized as heatmap with green boxes marking positive pairs

**Output:** `results/similarity_matrix_before_training.png`

### Questions

**Q1. Why is the diagonal ignored?**  
`sim[i,i]` is always 1.0 (a view compared with itself). It carries no
learning signal and would dominate the loss if included. SimCLR
explicitly masks the diagonal.

**Q2. Where are the positive pairs located?**  
At positions `(i, i+N)` and `(i+N, i)` — the anti-diagonal of the
top-right and bottom-left quadrants of the 2N×2N matrix.

**Q3. Why are all other entries treated as negatives?**  
With a large enough batch, two randomly sampled images are very
unlikely to be the same class. Treating all non-positive pairs as
negatives forces the encoder to learn discriminative features that
separate different images.

### Similarity Values (Random Encoder — Before Training)

| Pair Type | Avg Cosine Similarity |
|---|---|
| Same image, two augmented views | 0.9839 |
| Different images | 0.9791 |

Both values are nearly equal (~0.98), confirming the random encoder
has no concept of which views belong together.

---

## Task 4.4 — NT-Xent Contrastive Loss

Implemented entirely from scratch. No library used (no lightly,
solo-learn, pytorch-metric-learning, etc.).

**Formula for positive pair (i, j):**
```
L(i,j) = -log [ exp(sim(z_i, z_j) / tau) /
                sum_{k != i} exp(sim(z_i, z_k) / tau) ]
```
where `sim(a,b)` = cosine similarity, `tau = 0.5`

**Implementation (6 steps):**
1. L2-normalize z1 and z2
2. Concatenate → 2N views
3. Compute 2N×2N similarity matrix, divide by tau
4. Mask diagonal with `-inf` (exclude self-similarity)
5. Build positive labels: `[N, N+1, ..., 2N-1, 0, 1, ..., N-1]`
6. Apply `F.cross_entropy(sim, labels)`

**Verification:**
- For a random encoder, NT-Xent loss ≈ `log(2N-1)`
- With N=8: `log(15)` ≈ **2.7081**
- Observed loss: **2.6993** confirms correct implementation

---

## How to Run

```bash
python msds25051_05_task4_simclr.py
```

**Requirements:** `torch`, `torchvision`, `numpy`, `matplotlib`  
**splits/** folder must contain `train_ssl_unlabeled.txt`

---

## Notes
- `weights=None` confirmed — encoder is completely random, not pretrained
- NT-Xent loss built only with `torch` and `torch.nn.functional`
- No SimCLR libraries used
- Labels not used anywhere in Task 4

# Checkpoint 3 — Day 9
## SimCLR Pretraining

### What Was Implemented
- SimCLR pretraining loop (unlabeled data, no labels used)
- NT-Xent loss curve generation
- Feature similarity before vs after SimCLR training
- Similarity matrix after training

---


## Task 5 — SimCLR Pretraining

### Dataset
- Split used: `train_ssl_unlabeled.txt`
- Total samples: **45,000**
- Labels: **NOT used** anywhere during pretraining
- Batches per epoch: **703** (drop_last=True for stable NT-Xent)

---

### Fixed Training Settings (Section 6)

| Setting | Value Used |
|---|---|
| Dataset | CIFAR-10 unlabeled |
| Encoder | ResNet-18 modified for CIFAR-10 |
| Image size | 32 × 32 |
| Batch size | 64 |
| Epochs | 50 |
| Optimizer | Adam |
| Learning rate | 3e-4 |
| Temperature (tau) | 0.5 |
| Projection dimension | 128 |
| Random seed | 2026 |
| Labels used? | NO |
| GPU | CUDA |
| Approximate training time | ~82.3 min |

---

### Training Loop

- For each batch: load view1 and view2 (no labels)
- Forward pass through encoder → projection head → 128-dim z1, z2
- Compute NT-Xent loss on z1, z2
- Backpropagate and update with Adam optimizer
- Record loss per epoch

---

### Loss Curve

| Epoch | Loss |
|-------|------|
| 1     | 3.8478 |
| 5     | 3.4306 |
| 10    | 3.3432 |
| 15    | 3.3053 |
| 20    | 3.2801 |
| 25    | 3.2613 |
| 30    | 3.2457 |
| 35    | 3.2337 |
| 40    | 3.2220 |
| 45    | 3.2124 |
| 50    | 3.2057 |

Loss decreased steadily from **3.8478 → 3.2057** over 50 epochs.

**Output:** `graphs/simclr_pretraining_loss.png`

---

### Feature Similarity — Before vs After SimCLR Training

| Pair Type | Before SimCLR | After SimCLR |
|---|---|---|
| Same image, two augmented views | 0.9890 | 0.9141 |
| Different images | 0.9855 | 0.3211 |
| **Positive-Negative Gap** | **0.0035** | **0.5930** |

**Output:** `results/similarity_matrix_after_training.png`

---

### Analysis

**Before training:**
- Positive similarity (0.9890) ≈ Negative similarity (0.9855)
- Gap = **0.0035** — random encoder cannot distinguish same vs different images

**After training:**
- Negative similarity dropped sharply: 0.9855 → **0.3211**
- Positive similarity stayed high: 0.9890 → **0.9141**
- Gap = **0.5930** — 169× larger than before training

SimCLR successfully learned to bring two augmented views of the same
image closer together while pushing views from different images apart.
Training was **successful**.

---

## How to Run

```bash
python msds25051_05_task5_pretraining.py
```

**Requirements:** `torch`, `torchvision`, `numpy`, `matplotlib`  
**splits/** folder must contain `train_ssl_unlabeled.txt`  
**models/** folder will be created automatically

---

## Notes
- `drop_last=True` in DataLoader ensures every batch is exactly N=64
- Similarity computed on encoder features (512-dim), not projection (128-dim)
- Encoder weights saved to `models/simclr_encoder.pt` for Tasks 6 and 7
- Labels never loaded or used at any point during this checkpoint
