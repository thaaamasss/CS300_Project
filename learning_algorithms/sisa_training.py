"""
CORRECTED SISA Unlearning Implementation
-----------------------------------------
Original bug: Fine-tuned the ENTIRE SISAEnsemble on remaining_dataset.
              This is NOT true SISA unlearning — it behaves like fine-tuning.

True SISA unlearning:
    1. Identify WHICH shards contain deleted samples
    2. Retrain ONLY those shards from scratch on their shard data minus deleted samples
    3. Leave all other shards completely untouched
    4. Ensemble inference (logit averaging) remains the same

This gives a provable guarantee: deleted samples never touch the retrained shard,
and untouched shards are identical to before (no collateral damage).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import copy
import time


def sisa_unlearning(sisa_model, full_dataset, deleted_indices, num_epochs=10,
                    batch_size=64, lr=0.001, device=None):
    """
    True SISA unlearning: retrain only the shards that contain deleted samples.

    Args:
        sisa_model     : trained SISAEnsemble (has .shard_models and .shard_indices)
        full_dataset   : the original full training dataset (before any deletion)
        deleted_indices: list/set of indices (into full_dataset) that must be forgotten
        num_epochs     : epochs to retrain each affected shard (default matches training)
        batch_size     : dataloader batch size
        lr             : learning rate for retraining (same as original training lr)
        device         : torch.device (auto-detected if None)

    Returns:
        dict with keys:
            'model'           : updated SISAEnsemble (affected shards retrained)
            'affected_shards' : list of shard indices that were retrained
            'time'            : total wall-clock time in seconds
    """

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -------------------------------------------------------------------------
    # STEP 1: Validate that the model carries shard assignment metadata
    # -------------------------------------------------------------------------
    # SISAEnsemble must store which original dataset indices belong to each shard.
    # The original sisa_training.py does NOT save this — see note at bottom.
    # We assume the fixed SISAEnsemble stores self.shard_indices as a list of lists.

    if not hasattr(sisa_model, 'shard_indices') or sisa_model.shard_indices is None:
        raise AttributeError(
            "SISAEnsemble must store shard_indices (list of lists of dataset indices). "
            "The original implementation does not save this — see sisa_training_fixed.py "
            "for the corrected training code that saves shard assignments."
        )

    deleted_set = set(deleted_indices)
    num_shards = len(sisa_model.shard_models)

    # -------------------------------------------------------------------------
    # STEP 2: Identify which shards contain at least one deleted sample
    # -------------------------------------------------------------------------
    affected_shards = []
    for shard_id in range(num_shards):
        shard_idx_set = set(sisa_model.shard_indices[shard_id])
        if shard_idx_set & deleted_set:          # intersection is non-empty
            affected_shards.append(shard_id)

    print(f"[SISA Unlearning] Total shards      : {num_shards}")
    print(f"[SISA Unlearning] Deleted samples   : {len(deleted_set)}")
    print(f"[SISA Unlearning] Affected shards   : {affected_shards}")
    print(f"[SISA Unlearning] Untouched shards  : "
          f"{[i for i in range(num_shards) if i not in affected_shards]}")

    if not affected_shards:
        print("[SISA Unlearning] No affected shards found. Nothing to retrain.")
        return {
            'model': sisa_model,
            'affected_shards': [],
            'time': 0.0
        }

    # -------------------------------------------------------------------------
    # STEP 3: Retrain each affected shard from scratch on (shard_data - deleted)
    # -------------------------------------------------------------------------
    start_time = time.time()

    for shard_id in affected_shards:
        print(f"\n[SISA Unlearning] Retraining shard {shard_id} from scratch...")

        # Build the new shard dataset: original shard indices minus deleted ones
        original_shard_indices = sisa_model.shard_indices[shard_id]
        remaining_shard_indices = [
            idx for idx in original_shard_indices
            if idx not in deleted_set
        ]

        if len(remaining_shard_indices) == 0:
            print(f"  WARNING: Shard {shard_id} has no remaining samples after deletion. "
                  f"Reinitialising with random weights.")
            # Re-initialise model with fresh random weights (no training data left)
            sisa_model.shard_models[shard_id].apply(_reset_weights)
            continue

        print(f"  Shard {shard_id}: {len(original_shard_indices)} → "
              f"{len(remaining_shard_indices)} samples "
              f"(removed {len(original_shard_indices) - len(remaining_shard_indices)})")

        # Create DataLoader for this shard's remaining data
        shard_subset = Subset(full_dataset, remaining_shard_indices)
        shard_loader = DataLoader(shard_subset, batch_size=batch_size,
                                  shuffle=True, num_workers=2, pin_memory=True)

        # Reset shard model to fresh random weights — TRUE retraining from scratch
        # This is the key difference from fine-tuning: we do NOT use the old weights
        sisa_model.shard_models[shard_id].apply(_reset_weights)
        shard_model = sisa_model.shard_models[shard_id].to(device)

        optimizer = optim.Adam(shard_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        shard_model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

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
                total += labels.size(0)

            acc = 100.0 * correct / total
            print(f"  Epoch [{epoch+1}/{num_epochs}]  "
                  f"Loss: {epoch_loss/len(shard_loader):.4f}  Acc: {acc:.2f}%")

        # Put the retrained shard back in the ensemble (in-place, already a reference)
        sisa_model.shard_models[shard_id] = shard_model.cpu()

    elapsed = time.time() - start_time
    print(f"\n[SISA Unlearning] Done. Retrained {len(affected_shards)}/{num_shards} shards "
          f"in {elapsed:.2f}s")
    print(f"[SISA Unlearning] Untouched shards carry their original weights — "
          f"no collateral damage.")

    return {
        'model': sisa_model,
        'affected_shards': affected_shards,
        'time': elapsed
    }


# =============================================================================
# HELPER: Reset model weights to fresh random initialisation
# =============================================================================

def _reset_weights(module):
    """
    Recursively reset all learnable weights to their default initialisation.
    Called via model.apply(_reset_weights) to reinitialise a shard from scratch.
    """
    if hasattr(module, 'reset_parameters'):
        module.reset_parameters()


# =============================================================================
# FIXED SISA TRAINING — saves shard_indices so unlearning can use them
# =============================================================================

def sisa_training_fixed(model_class, dataset, num_shards=5, num_epochs=10,
                         batch_size=64, lr=0.001, device=None, **model_kwargs):
    """
    Fixed SISA training that saves shard_indices on the ensemble model.

    The original sisa_training.py discards shard assignments after training,
    making true SISA unlearning impossible. This version attaches shard_indices
    to the SISAEnsemble so unlearning can identify affected shards.

    Args:
        model_class  : the CNN model class (e.g. CNNModel)
        dataset      : full training dataset
        num_shards   : number of shards (default 5, same as original)
        num_epochs   : epochs per shard
        batch_size   : dataloader batch size
        lr           : Adam learning rate
        device       : torch.device
        **model_kwargs: passed to model_class constructor (e.g. num_classes, in_channels)

    Returns:
        SISAEnsemble with .shard_indices attribute populated
    """
    from models.architectures.sisa_model import SISAEnsemble

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n = len(dataset)
    shard_size = n // num_shards          # tail samples discarded (matches original)
    criterion = nn.CrossEntropyLoss()

    # Build shard index assignments BEFORE training so they can be saved
    all_indices = list(range(n))
    shard_indices = [
        all_indices[i * shard_size: (i + 1) * shard_size]
        for i in range(num_shards)
    ]

    shard_models = []

    for shard_id in range(num_shards):
        print(f"\n[SISA Training] Training shard {shard_id+1}/{num_shards} "
              f"({len(shard_indices[shard_id])} samples)...")

        shard_subset = Subset(dataset, shard_indices[shard_id])
        shard_loader = DataLoader(shard_subset, batch_size=batch_size,
                                  shuffle=True, num_workers=2, pin_memory=True)

        model = model_class(**model_kwargs).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in shard_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            acc = 100.0 * correct / total
            print(f"  Shard {shard_id} | Epoch [{epoch+1}/{num_epochs}] "
                  f"Loss: {epoch_loss/len(shard_loader):.4f}  Acc: {acc:.2f}%")

        shard_models.append(model.cpu())

    # Build the ensemble
    ensemble = SISAEnsemble(shard_models)

    # KEY FIX: attach shard assignments so unlearning can use them
    ensemble.shard_indices = shard_indices

    print(f"\n[SISA Training] Complete. Ensemble of {num_shards} shards ready.")
    print(f"[SISA Training] shard_indices saved on model — unlearning will work correctly.")

    return ensemble


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    """
    Minimal usage example showing how training and unlearning connect.

    Assumes:
        - CNNModel at models/architectures/cnn_model.py
        - SISAEnsemble at models/architectures/sisa_model.py
        - torchvision MNIST available
    """
    import torchvision
    import torchvision.transforms as transforms
    from models.architectures.cnn_model import CNNModel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )

    # --- Step 1: Train with fixed SISA (saves shard_indices) ---
    ensemble = sisa_training_fixed(
        model_class=CNNModel,
        dataset=train_dataset,
        num_shards=5,
        num_epochs=10,
        batch_size=64,
        lr=0.001,
        device=device,
        num_classes=10,
        in_channels=1
    )

    # --- Step 2: Define which samples to delete (e.g. random 500) ---
    import random
    deleted_indices = random.sample(range(len(train_dataset)), 500)

    # --- Step 3: True SISA unlearning — only affected shards retrained ---
    result = sisa_unlearning(
        sisa_model=ensemble,
        full_dataset=train_dataset,
        deleted_indices=deleted_indices,
        num_epochs=10,
        batch_size=64,
        lr=0.001,
        device=device
    )

    print(f"\nAffected shards retrained : {result['affected_shards']}")
    print(f"Unlearning time           : {result['time']:.2f}s")
    print(f"Untouched shards          : "
          f"{[i for i in range(5) if i not in result['affected_shards']]}")

    # Expected output with 500 random deletions across 5 shards of 12000 each:
    # Almost certainly all 5 shards are affected (each shard likely contains ~100 deleted)
    # But with targeted/class deletion, often only 1-2 shards are affected → big speedup
