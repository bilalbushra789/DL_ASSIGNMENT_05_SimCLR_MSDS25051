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
```

## Tasks in Checkpoint 1

| File | Task |
|---|---|
| `task1_supervised.py` | Supervised ResNet-18 baseline, 10% labels |
| `task2_augmentations.py` | SimCLR augmentation pipeline + visualisation |


## How to run

### Individual tasks
```bash
python task1_supervised.py
python task2_augmentations.py

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
- `fc`: Linear(512 → 10) for supervised; Linear identity for encoder-only tasks

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


# Checkpoint 4 — Day 12
## Linear Probing, Fine-tuning, Visualization, and Final Report

### What Was Implemented
- Linear probing (random encoder + SimCLR encoder)
- Fine-tuning (SimCLR pretrained encoder, full end-to-end)
- PCA/t-SNE feature visualization 
- metrics.json generation
- test_predictions.csv generation
- Final report

---

## Files Added This Checkpoint

```
msds25051_05_task6_linear_probe.py
msds25051_05_task7_fine_tune.py
msds25051_05_task8_pca.py
generate_final_outputs.py
models/
  linear_probe.pt
  finetuned_model.pt
graphs/
  linear_probe_accuracy.png
  finetuning_accuracy.png
results/
  random_encoder_pca_or_tsne.png
  simclr_encoder_pca_or_tsne.png
  finetuned_encoder_pca_or_tsne.png
  metrics.json
  test_predictions.csv
Report.pdf
```

---

## Task 6 — Linear Probe Evaluation

### Setup
- Encoder: **completely frozen** in both experiments
- Trainable part: `Linear(512 -> 10)` only (5,130 parameters)
- Splits: `train_labeled_10percent.txt`, `val.txt`, `test.txt`
- Epochs: 20, Optimizer: Adam, LR: 3e-4

### Results

| Model | Encoder | Trainable Part | Test Accuracy |
|---|---|---|---|
| Random Linear Probe | Random frozen | Linear only | 25.37% |
| SimCLR Linear Probe | SimCLR frozen | Linear only | **73.16%** |

**SimCLR improvement over random: +47.79%**

This gap — achieved with the same frozen architecture and same linear head — directly proves SimCLR learned genuinely useful visual representations entirely without labels.

**Output:** `graphs/linear_probe_accuracy.png`

---

## Task 7 — Fine-tuning the SimCLR Encoder

### Setup
- Encoder initialized with SimCLR pretrained weights
- Full model trained end-to-end (encoder + classifier, all parameters)
- Splits: `train_labeled_10percent.txt`, `val.txt`, `test.txt`
- Epochs: 20, Optimizer: Adam, LR: 3e-4


**Output:** `graphs/finetuning_accuracy.png`

---

## Task 8 — PCA / t-SNE Feature Visualization

### Setup
- Method: t-SNE (PCA to 50 dims first, then t-SNE to 2D)
- Validation images: 1,000 fixed (seed=2026)
- Labels used: only for coloring, NOT during training

### Three Encoders

| Encoder | Observation |
|---|---|
| Random untrained | All 10 classes fully mixed — no grouping |
| SimCLR pretrained | Visible partial clustering by class |
| Fine-tuned | Clear tight clusters per class |

### Questions Answered

**Q1. Random encoder grouping?** No — all classes overlap, no semantic structure.

**Q2. SimCLR encoder better grouping?** Yes — visible clusters emerge without any labels.

**Q3. Fine-tuning improve separation?** Yes, significantly — tight compact clusters for all classes.

**Q4. Confused classes?** cat/dog, bird/airplane/deer, automobile/truck.

**Outputs:**
- `results/random_encoder_pca_or_tsne.png`
- `results/simclr_encoder_pca_or_tsne.png`
- `results/finetuned_encoder_pca_or_tsne.png`

---

## metrics.json

```json
{
    "student_name": "bilal_bushra",
    "roll_number": "msds25051",
    "seed": 2026,
    "batch_size": 64,
    "simclr_epochs": 50,
    "linear_probe_epochs": 20,
    "finetuning_epochs": 20,
    "learning_rate": 0.0003,
    "temperature": 0.5,
    "supervised_10percent_test_acc": 0.7483,
    "random_linear_probe_test_acc": 0.2537,
    "simclr_linear_probe_test_acc": 0.7316,
    "simclr_finetune_test_acc": 0.8196,
    "same_view_similarity_before": 0.9890,
    "different_image_similarity_before": 0.9855,
    "same_view_similarity_after": 0.9141,
    "different_image_similarity_after": 0.3211
}
```

**Output:** `results/metrics.json`

---

## test_predictions.csv

Format: `image_index, true_label, predicted_label, prob_class_0, ..., prob_class_9`

Generated from the fine-tuned model on 10,000 test images.
Fine-tuned model test accuracy: **81.96%**

**Output:** `results/test_predictions.csv`

---


## How to Run

```bash
# Task 6 — Linear probing
python msds25051_05_task6_linear_probe.py

# Task 7 — Fine-tuning
python msds25051_05_task7_fine_tune.py

# Task 8 — t-SNE visualization
python msds25051_05_task8_pca.py

```

**Requirements:** `torch`, `torchvision`, `sklearn`, `numpy`, `matplotlib`
**models/simclr_encoder.pt** must exist 

---

