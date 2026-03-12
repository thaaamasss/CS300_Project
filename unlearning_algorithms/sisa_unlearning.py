"""
sisa_unlearning.py — Fixed
--------------------------
True SISA unlearning: retrain only the shards that contain deleted samples.

Interface matches all other unlearning functions:
    sisa_unlearning(model, remaining_dataset, deleted_dataset,
                    num_classes, input_channels, input_size, ...) -> model

Two cases handled:
  Case A — model is a SISAEnsemble with shard_indices:
      True SISA unlearning — identify affected shards, retrain only those.

  Case B — model is a plain CNNModel (e.g. best learning algo was Adam):
      Fall back to fine-tuning on remaining_dataset.
      Logs a clear warning so the limitation is visible in output.
      This matches the original project's behaviour for non-SISA best models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import copy
import time


def sisa_unlearning(model, remaining_dataset, deleted_dataset,
                    num_classes, input_channels, input_size,
                    num_epochs=10, batch_size=64, lr=0.001, device=None):
    """
    SISA unlearning with same interface as all other unlearning functions.

    Args:
        model             : trained model (SISAEnsemble or plain CNNModel)
        remaining_dataset : dataset after deletion
        deleted_dataset   : samples to forget
        num_classes       : number of output classes
        input_channels    : input channels (1 or 3)
        input_size        : spatial input size (28 or 32)
        num_epochs        : epochs for retraining affected shards
        batch_size        : dataloader batch size
        lr                : Adam learning rate
        device            : torch.device

    Returns:
        updated model (cpu) — same type as input model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------------------------------------------------------
    # Case A: True SISA unlearning — SISAEnsemble with shard_indices
    # ------------------------------------------------------------------
    if hasattr(model, 'shard_indices') and model.shard_indices is not None:
        return _true_sisa_unlearning(
            model, remaining_dataset, deleted_dataset,
            num_epochs=num_epochs, batch_size=batch_size,
            lr=lr, device=device
        )

    # ------------------------------------------------------------------
    # Case B: Plain CNNModel — fall back to fine-tuning on remaining data
    # Best learning algorithm was not SISA, so no shard structure exists.
    # ------------------------------------------------------------------
    print("[SISA Unlearning] WARNING: model has no shard_indices "
          "(best learning algo was not SISA).")
    print("[SISA Unlearning] Falling back to fine-tuning on remaining data.")

    return _fallback_finetune(
        model, remaining_dataset,
        num_epochs=num_epochs, batch_size=batch_size,
        lr=lr, device=device
    )


# ----------------------------------------------------------------------
# TRUE SISA UNLEARNING (Case A)
# ----------------------------------------------------------------------

def _true_sisa_unlearning(sisa_model, remaining_dataset, deleted_dataset,
                           num_epochs, batch_size, lr, device):
    """Retrain only shards that contain deleted samples."""

    # Build set of deleted indices relative to the original full dataset
    # deleted_dataset is a Subset — recover its original indices
    if hasattr(deleted_dataset, 'indices'):
        deleted_set = set(deleted_dataset.indices)
    else:
        # fallback: use range if indices not available
        deleted_set = set(range(len(deleted_dataset)))

    num_shards = len(sisa_model.shard_models)

    # Identify affected shards
    affected_shards = []
    for shard_id in range(num_shards):
        shard_idx_set = set(sisa_model.shard_indices[shard_id])
        if shard_idx_set & deleted_set:
            affected_shards.append(shard_id)

    print(f"[SISA Unlearning] Total shards     : {num_shards}")
    print(f"[SISA Unlearning] Deleted samples  : {len(deleted_set)}")
    print(f"[SISA Unlearning] Affected shards  : {affected_shards}")
    print(f"[SISA Unlearning] Untouched shards : "
          f"{[i for i in range(num_shards) if i not in affected_shards]}")

    if not affected_shards:
        print("[SISA Unlearning] No affected shards. Nothing to retrain.")
        return sisa_model.cpu()

    criterion = nn.CrossEntropyLoss()

    # Need the full original dataset to rebuild shard subsets
    # remaining_dataset + deleted_dataset together = original train data
    # We use remaining_dataset indices to filter each shard
    if hasattr(remaining_dataset, 'indices'):
        remaining_set = set(remaining_dataset.indices)
    else:
        remaining_set = None   # can't filter, retrain on all remaining

    for shard_id in affected_shards:
        print(f"\n[SISA Unlearning] Retraining shard {shard_id} from scratch...")

        original_shard_indices = sisa_model.shard_indices[shard_id]

        # Remove deleted indices from this shard
        if remaining_set is not None:
            remaining_shard_indices = [
                idx for idx in original_shard_indices if idx in remaining_set
            ]
        else:
            remaining_shard_indices = [
                idx for idx in original_shard_indices if idx not in deleted_set
            ]

        print(f"  Shard {shard_id}: {len(original_shard_indices)} -> "
              f"{len(remaining_shard_indices)} samples after deletion")

        if len(remaining_shard_indices) == 0:
            print(f"  WARNING: Shard {shard_id} empty after deletion. Reinitialising.")
            sisa_model.shard_models[shard_id].apply(_reset_weights)
            continue

        # Need the base dataset to build a Subset — get it from remaining_dataset
        base_dataset = _get_base_dataset(remaining_dataset)
        shard_subset = Subset(base_dataset, remaining_shard_indices)
        shard_loader = DataLoader(shard_subset, batch_size=batch_size,
                                  shuffle=True, num_workers=2, pin_memory=True)

        # Reset shard to fresh random weights — true retraining from scratch
        sisa_model.shard_models[shard_id].apply(_reset_weights)
        shard_model = sisa_model.shard_models[shard_id].to(device)

        optimizer = optim.Adam(shard_model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        shard_model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct = total = 0
            for inputs, labels in shard_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = shard_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total   += labels.size(0)
            scheduler.step()
            acc = 100.0 * correct / total
            print(f"  Epoch [{epoch+1}/{num_epochs}]  "
                  f"Loss: {epoch_loss/len(shard_loader):.4f}  Acc: {acc:.2f}%")

        sisa_model.shard_models[shard_id] = shard_model.cpu()

    print(f"\n[SISA Unlearning] Done. Retrained {len(affected_shards)}/{num_shards} shards.")
    return sisa_model.cpu()


# ----------------------------------------------------------------------
# FALLBACK FINE-TUNE (Case B)
# ----------------------------------------------------------------------

def _fallback_finetune(model, remaining_dataset, num_epochs, batch_size, lr, device):
    """Fine-tune on remaining data when no shard structure is available."""

    unlearn_model = copy.deepcopy(model).to(device)
    remaining_loader = DataLoader(remaining_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=2, pin_memory=True)
    optimizer = optim.Adam(unlearn_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc   = 0.0
    best_state = copy.deepcopy(unlearn_model.state_dict())

    for epoch in range(num_epochs):
        unlearn_model.train()
        epoch_loss = 0.0
        correct = total = 0
        for inputs, labels in remaining_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = unlearn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)
        acc = correct / total
        scheduler.step()
        if acc > best_acc:
            best_acc   = acc
            best_state = copy.deepcopy(unlearn_model.state_dict())
        print(f"  [SISA Fallback] Epoch [{epoch+1}/{num_epochs}]  "
              f"Loss: {epoch_loss/len(remaining_loader):.4f}  Acc: {acc:.4f}")

    unlearn_model.load_state_dict(best_state)
    return unlearn_model.cpu()


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------

def _reset_weights(module):
    """Reset all learnable parameters to default initialisation."""
    if hasattr(module, 'reset_parameters'):
        module.reset_parameters()


def _get_base_dataset(dataset):
    """Unwrap Subset layers to get the base torchvision dataset."""
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset
