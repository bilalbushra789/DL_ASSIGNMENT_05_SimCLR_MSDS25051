"""Utilities for loading fixed split files.

This file intentionally does not implement SimCLR. It only helps students load
CIFAR-10 subsets from the instructor-provided split files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10


def read_split_indices(path: str | Path) -> list[int]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [int(line) for line in lines if line]


def get_cifar10_subset(
    data_root: str | Path,
    split_file: str | Path,
    train: bool,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
) -> Dataset:
    """Return a CIFAR-10 subset according to a fixed split file.

    Args:
        data_root: CIFAR-10 data root.
        split_file: Path to a txt file containing integer indices.
        train: Use CIFAR-10 official train split if True, official test split if False.
        transform: Optional image transform.
        target_transform: Optional label transform.
        download: Download CIFAR-10 if not present.
    """
    dataset = CIFAR10(
        root=str(data_root),
        train=train,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )
    indices = read_split_indices(split_file)
    return Subset(dataset, indices)


class TwoViewDataset(Dataset):
    """Wrap a dataset so it returns two augmented views and the original target.

    For SimCLR pretraining, students should ignore the target in the training loop.
    This wrapper is provided as a simple data utility, not as a SimCLR implementation.
    """

    def __init__(self, base_dataset: Dataset, two_view_transform: Callable):
        self.base_dataset = base_dataset
        self.two_view_transform = two_view_transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        image, target = self.base_dataset[idx]
        view1, view2 = self.two_view_transform(image)
        return view1, view2, target
