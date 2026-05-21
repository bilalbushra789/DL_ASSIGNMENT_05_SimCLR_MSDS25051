import os
from torch.utils.data import Subset
from torchvision import datasets

def load_split_indices(filepath):
    with open(filepath) as f:
        indices = [int(l.strip()) for l in f if l.strip()]
    print(f"Loaded {len(indices)} indices from {filepath}")
    return indices

def get_cifar10_with_transform(transform, split_dir="splits",
                                data_dir="./data", split="labeled"):
    is_train = split in ("labeled", "ssl")
    full_dataset = datasets.CIFAR10(root=data_dir, train=is_train,
                                     download=True, transform=transform)
    split_file_map = {
        "labeled": "train_labeled_10percent.txt",
        "ssl":     "train_ssl_unlabeled.txt",
        "val":     "val.txt",
        "test":    "test.txt",
    }
    idx = load_split_indices(os.path.join(split_dir, split_file_map[split]))
    return Subset(full_dataset, idx)
