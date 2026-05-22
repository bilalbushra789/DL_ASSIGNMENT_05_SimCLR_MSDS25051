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
