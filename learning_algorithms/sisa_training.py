import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import copy

from utils.config import NUM_SLICES


def sisa_training(dataset, num_classes, input_channels, input_size,
                  num_shards=5, num_slices=None, num_epochs=10,
                  batch_size=64, lr=0.001, device=None):
    from models.architectures.cnn_model import CNNModel
    from models.architectures.sisa_model import SISAEnsemble

    if num_slices is None:
        num_slices = NUM_SLICES

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n = len(dataset)
    shard_size = n // num_shards
    remainder = n % num_shards
    all_indices = list(range(n))

    # Build shard index lists
    shard_indices = [
        all_indices[i * shard_size: (i + 1) * shard_size]
        for i in range(num_shards)
    ]
    # Distribute remainder samples
    for i in range(remainder):
        shard_indices[i].append(all_indices[num_shards * shard_size + i])

    if remainder > 0:
        print(f"[SISA] Distributed {remainder} remainder sample(s) "
              f"across first {remainder} shard(s).")

    print(f"[SISA] {num_shards} shards × {num_slices} slice(s) = "
          f"{num_shards * num_slices} training segments.")

    criterion = nn.CrossEntropyLoss()
    shard_models = []
    shard_checkpoints = []   # list of lists: shard_checkpoints[shard][slice] = (cumulative_indices, state_dict)
    all_losses = []

    for shard_id in range(num_shards):
        print(f"\n[SISA] === Shard {shard_id + 1}/{num_shards} "
              f"({len(shard_indices[shard_id])} samples) ===")

        shard_idx = shard_indices[shard_id]
        slice_size = len(shard_idx) // num_slices
        slice_remainder = len(shard_idx) % num_slices

        # Build cumulative slice index lists
        # slice k trains on indices 0..k*slice_size (cumulative)
        slice_boundaries = []
        for s in range(num_slices):
            end = (s + 1) * slice_size + (slice_remainder if s == num_slices - 1 else 0)
            slice_boundaries.append(end)

        model = CNNModel(
            input_channels=input_channels,
            num_classes=num_classes,
            input_size=input_size
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        # Scheduler spans all slices × epochs
        total_steps = num_slices * num_epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        this_shard_checkpoints = []
        final_loss = 0.0
        cumulative_end = 0

        for slice_id in range(num_slices):
            cumulative_end = slice_boundaries[slice_id]
            cumulative_indices = shard_idx[:cumulative_end]

            print(f"  Slice {slice_id + 1}/{num_slices}: "
                  f"training on {len(cumulative_indices)} cumulative samples...")

            shard_subset = Subset(dataset, cumulative_indices)
            shard_loader = DataLoader(
                shard_subset, batch_size=batch_size,
                shuffle=True, num_workers=2, pin_memory=True
            )

            model.train()
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
                scheduler.step()
                acc = 100.0 * correct / total
                print(f"    Epoch [{epoch+1}/{num_epochs}]  "
                      f"Loss: {final_loss:.4f}  Acc: {acc:.2f}%")

            # Save checkpoint after this slice
            checkpoint = {
                'slice_id': slice_id,
                'cumulative_indices': cumulative_indices,
                'state_dict': copy.deepcopy(model.state_dict()),
                'optimizer_state': copy.deepcopy(optimizer.state_dict()),
            }
            this_shard_checkpoints.append(checkpoint)
            print(f"  Slice {slice_id + 1} checkpoint saved "
                  f"({len(cumulative_indices)} samples seen so far).")

        all_losses.append(final_loss)
        shard_models.append(model.cpu())
        shard_checkpoints.append(this_shard_checkpoints)

    ensemble = SISAEnsemble(
        shard_models,
        shard_indices=shard_indices,
        shard_checkpoints=shard_checkpoints,
        num_slices=num_slices
    )

    avg_loss = sum(all_losses) / len(all_losses)
    print(f"\n[SISA] Training complete. Average shard loss: {avg_loss:.4f}")
    print(f"[SISA] Checkpoints stored: {num_shards} shards × "
          f"{num_slices} slices = {num_shards * num_slices} total.")

    return ensemble, avg_loss