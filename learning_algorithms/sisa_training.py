"""
sisa_training.py — Fixed
Fixes applied:
  #3  Remainder samples no longer silently discarded — distributed across shards
  #3  shard_indices saved on the ensemble so SISA unlearning can work correctly
      (prerequisite for the corrected sisa_unlearning.py)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import time


def sisa_training(dataset, num_classes, input_channels, input_size, num_shards=5,
                  num_epochs=10, batch_size=64, lr=0.001, device=None):
    """
    SISA Training: train an ensemble of shard models independently.

    Returns:
        (SISAEnsemble, final_avg_loss)  — matches the (model, loss) contract
        The ensemble has .shard_indices attached for use by sisa_unlearning.
    """
    from models.architectures.cnn_model import CNNModel
    from models.architectures.sisa_model import SISAEnsemble

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n = len(dataset)
    shard_size = n // num_shards
    remainder  = n % num_shards          # FIX #3: capture remainder

    all_indices = list(range(n))

    # Build base shards
    shard_indices = [
        all_indices[i * shard_size: (i + 1) * shard_size]
        for i in range(num_shards)
    ]

    # FIX #3: distribute remainder samples across first `remainder` shards
    # instead of silently discarding them
    for i in range(remainder):
        extra_idx = num_shards * shard_size + i
        shard_indices[i].append(all_indices[extra_idx])

    if remainder > 0:
        print(f"[SISA] Distributed {remainder} remainder sample(s) "
              f"across first {remainder} shard(s) (no data discarded).")

    criterion = nn.CrossEntropyLoss()
    shard_models = []
    all_losses = []

    for shard_id in range(num_shards):
        print(f"\n[SISA] Training shard {shard_id + 1}/{num_shards} "
              f"({len(shard_indices[shard_id])} samples)...")

        shard_subset = Subset(dataset, shard_indices[shard_id])
        shard_loader = DataLoader(shard_subset, batch_size=batch_size,
                                  shuffle=True, num_workers=2, pin_memory=True)

        model = CNNModel(input_channels=input_channels, num_classes=num_classes, input_size=input_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # FIX #7: cosine annealing LR scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        model.train()
        final_loss = 0.0
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct = total = 0
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
            final_loss = epoch_loss / len(shard_loader)
            scheduler.step()   # FIX #7
            acc = 100.0 * correct / total
            print(f"  Shard {shard_id} | Epoch [{epoch+1}/{num_epochs}] "
                  f"Loss: {final_loss:.4f}  Acc: {acc:.2f}%")

        all_losses.append(final_loss)
        shard_models.append(model.cpu())

    # Pass shard_indices into constructor — stored natively, survives deepcopy
    ensemble = SISAEnsemble(shard_models, shard_indices=shard_indices)

    avg_loss = sum(all_losses) / len(all_losses)
    print(f"\n[SISA] Training complete. Average shard loss: {avg_loss:.4f}")

    return ensemble, avg_loss   # FIX #1: return (model, loss) not just model
