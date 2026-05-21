# DL Assignment 5 — SimCLR | Checkpoint 1

## Checkpoint 1 Tasks (Day 3)

| Task | File | Output |
|------|------|--------|
| Fixed dataset split loading | `utils/dataset_splits.py` | — |
| Supervised baseline training | `msds25051_05_task1_supervised.py` | `graphs/supervised_loss.png`, `results/supervised_confusion_matrix.png` |
| Augmentation pipeline + two-view transform + visualization | `msds25051_05_task2_augmentations.py` | `results/augmentation_examples.png` |

---

## Setup

```bash
pip install -r requirements.txt
```

## Step 1 — Get split files

 `splits/`: 
```
splits/
  train_labeled_10percent.txt  5000
  train_ssl_unlabeled.txt      45000
  val.txt                      5000
  test.txt                     5000
```


```bash
python generate_splits.py
```
> **Replace these with TA-provided files before final submission.**

---

## Step 2 — Run Task 1: Supervised Baseline

```bash
python rollNumber_05_task1_supervised.py
```

**Outputs:**
- `graphs/supervised_loss.png`
- `results/supervised_confusion_matrix.png`
- `results/metrics.json` (updated with `supervised_10percent_test_acc`)
- `models/supervised_best.pt`

---

## Step 3 — Run Task 2: Augmentation Visualization

```bash
python rollNumber_05_task2_augmentations.py
```

**Output:**
- `results/augmentation_examples.png`

---

## Project Structure

```
checkpoint1/
├── rollNumber_05_task1_supervised.py   ← Task 1: supervised baseline
├── rollNumber_05_task2_augmentations.py← Task 2: augmentation + two-view
├── generate_splits.py                  ← helper to generate placeholder splits
├── requirements.txt
├── README.md
├── utils/
│   ├── __init__.py
│   ├── seed.py                         ← set_seed(2026)
│   ├── dataset_splits.py               ← split file loaders
│   └── metrics.py                      ← accuracy, confusion matrix, metrics.json
├── splits/                             ← put TA split files here
├── data/                               ← CIFAR-10 auto-downloaded here
├── graphs/
│   └── supervised_loss.png
├── results/
│   ├── augmentation_examples.png
│   ├── supervised_confusion_matrix.png
│   └── metrics.json
└── models/
    └── supervised_best.pt
```

---

## Key Design Decisions

- **ResNet-18 for CIFAR-10**: `conv1` replaced with 3×3, stride=1, padding=1; `maxpool` replaced with `Identity()`.
- **Random seed**: `2026` applied to Python, NumPy, and PyTorch.
- **No pre-trained weights**: `weights=None` passed to `resnet18()`.
- **Split loading**: strictly uses the provided `.txt` files; `random_split()` is never used.
- **SimCLR transform**: exactly as specified in the assignment (RandomResizedCrop, HorizontalFlip, ColorJitter, RandomGrayscale, Normalize).
- **TwoViewTransform**: hand-implemented wrapper; no external SimCLR library used.
