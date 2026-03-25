"""
class_deletion.py — Fixed
Fix #2: Replaced sample-by-sample iteration with vectorised torch.where.

Original bug:
    for i in range(len(dataset)):
        if dataset[i][1] != class_to_delete:   # 50,000 individual __getitem__ calls
            remaining.append(i)

This triggered 50,000 image decode operations just to read labels.
torchvision datasets expose .targets directly — no image loading needed.

Fix: use torch.where on the targets tensor — one vectorised operation.
"""

from torch.utils.data import Subset
import torch


def class_deletion(dataset, class_to_delete):
    """
    Remove all samples of a given class from the dataset.

    Args:
        dataset        : torchvision dataset with a .targets attribute
        class_to_delete: integer class label to remove

    Returns:
        remaining_dataset : Subset excluding deleted class
        deleted_dataset   : Subset containing only deleted class
    """
    # FIX #2: vectorised label lookup — no image loading, no Python loop
    targets = torch.tensor(dataset.targets)

    remaining_indices = torch.where(targets != class_to_delete)[0].tolist()
    deleted_indices   = torch.where(targets == class_to_delete)[0].tolist()

    print(f"[class_deletion] Class {class_to_delete}: "
          f"{len(deleted_indices)} samples deleted, "
          f"{len(remaining_indices)} remaining.")

    remaining_dataset = Subset(dataset, remaining_indices)
    deleted_dataset   = Subset(dataset, deleted_indices)

    return remaining_dataset, deleted_dataset