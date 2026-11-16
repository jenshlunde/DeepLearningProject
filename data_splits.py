"""Shared deterministic dataset split helpers."""

from typing import List, Tuple

import torch
from torch.utils.data import random_split

# A shared helper method for splitting the training dataset into traning and validation by 90/10 rule
def jigsaw_train_validation_indices(
    dataset_size: int, seed: int
) -> Tuple[List[int], List[int]]:
    train_size = int(0.9 * dataset_size)
    validation_size = dataset_size - train_size
    train_subset, validation_subset = random_split(
        range(dataset_size),
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(seed),
    )
    return list(train_subset.indices), list(validation_subset.indices)
