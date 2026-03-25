import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import copy


def sisa_unlearning(model, remaining_dataset, deleted_dataset,
                    num_classes, input_channels, input_size,
                    num_epochs=10, batch_size=64, lr=0.001, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Case A / B: SISAEnsemble
    if hasattr(model, 'shard_indices') and model.shard_indices is not None:
        return _sisa_unlearn(
            model, remaining_dataset, deleted_dataset,
            num_epochs=num_epochs, batch_size=batch_size,
            lr=lr, device=device
        )

    # Case C: plain CNNModel
    print("[SISA Unlearning] WARNING: no shard_indices found. "
          "Falling back to fine-tuning.")
    return _fallback_finetune(
        model, remaining_dataset,
        num_epochs=num_epochs, batch_size=batch_size,
        lr=lr, device=device
    )


# ----------------------------------------------------------------------
# MAIN SISA UNLEARNING (Cases A and B)
# ----------------------------------------------------------------------

def _sisa_unlearn(sisa_model, remaining_dataset, deleted_dataset,
                  num_epochs, batch_size, lr, device):

    if hasattr(deleted_dataset, 'indices'):
        deleted_set = set(deleted_dataset.indices)
    else:
        deleted_set = set(range(len(deleted_dataset)))

    if hasattr(remaining_dataset, 'indices'):
        remaining_set = set(remaining_dataset.indices)
    else:
        remaining_set = None

    num_shards = len(sisa_model.shard_models)
    has_checkpoints = (sisa_model.shard_checkpoints is not None
                       and len(sisa_model.shard_checkpoints) == num_shards)
    num_slices = sisa_model.num_slices if hasattr(sisa_model, 'num_slices') else 1

    # Identify affected shards
    affected_shards = [
        sid for sid in range(num_shards)
        if set(sisa_model.shard_indices[sid]) & deleted_set
    ]

    print(f"[SISA Unlearning] Shards: {num_shards}  Slices/shard: {num_slices}")
    print(f"[SISA Unlearning] Deleted samples  : {len(deleted_set)}")
    print(f"[SISA Unlearning] Affected shards  : {affected_shards}")
    print(f"[SISA Unlearning] Untouched shards : "
          f"{[i for i in range(num_shards) if i not in affected_shards]}")
    print(f"[SISA Unlearning] Slice checkpoints: {'yes' if has_checkpoints else 'no — full shard retrain'}")

    if not affected_shards:
        print("[SISA Unlearning] Nothing to retrain.")
        return sisa_model.cpu()

    criterion = nn.CrossEntropyLoss()
    base_dataset = _get_base_dataset(remaining_dataset)

    for shard_id in affected_shards:
        print(f"\n[SISA Unlearning] Processing shard {shard_id}...")

        original_indices = sisa_model.shard_indices[shard_id]

        # Remove deleted indices from this shard
        if remaining_set is not None:
            remaining_shard_indices = [i for i in original_indices if i in remaining_set]
        else:
            remaining_shard_indices = [i for i in original_indices if i not in deleted_set]

        print(f"  Shard {shard_id}: {len(original_indices)} → "
              f"{len(remaining_shard_indices)} samples after deletion")

        if len(remaining_shard_indices) == 0:
            print(f"  WARNING: shard {shard_id} empty after deletion. Reinitialising.")
            sisa_model.shard_models[shard_id].apply(_reset_weights)
            continue

        if has_checkpoints and num_slices > 1:
            # ----- Case A: slice-aware unlearning -----
            _retrain_from_affected_slice(
                sisa_model=sisa_model,
                shard_id=shard_id,
                deleted_set=deleted_set,
                remaining_shard_indices=remaining_shard_indices,
                base_dataset=base_dataset,
                num_epochs=num_epochs,
                batch_size=batch_size,
                lr=lr,
                device=device,
                criterion=criterion
            )
        else:
            # ----- Case B: no slices — full shard retrain -----
            print(f"  No slice checkpoints. Retraining shard {shard_id} from scratch...")
            _retrain_shard_from_scratch(
                sisa_model=sisa_model,
                shard_id=shard_id,
                remaining_indices=remaining_shard_indices,
                base_dataset=base_dataset,
                num_epochs=num_epochs,
                batch_size=batch_size,
                lr=lr,
                device=device,
                criterion=criterion
            )

    print(f"\n[SISA Unlearning] Done. "
          f"Retrained {len(affected_shards)}/{num_shards} shards.")
    return sisa_model.cpu()


# ----------------------------------------------------------------------
# SLICE-AWARE RETRAIN (Case A)
# ----------------------------------------------------------------------

def _retrain_from_affected_slice(sisa_model, shard_id, deleted_set,
                                  remaining_shard_indices, base_dataset,
                                  num_epochs, batch_size, lr, device, criterion):
    checkpoints = sisa_model.shard_checkpoints[shard_id]
    num_slices = len(checkpoints)

    # Find first affected slice — the earliest slice whose cumulative
    # indices contain at least one deleted sample
    affected_slice = None
    for s, ckpt in enumerate(checkpoints):
        if set(ckpt['cumulative_indices']) & deleted_set:
            affected_slice = s
            break

    if affected_slice is None:
        print(f"  Shard {shard_id}: no deleted sample found in any slice. Skipping.")
        return

    print(f"  First affected slice: {affected_slice + 1}/{num_slices}")

    # Load checkpoint from slice BEFORE the affected one (or fresh init if affected_slice=0)
    shard_model = sisa_model.shard_models[shard_id]

    if affected_slice == 0:
        print(f"  Deletion in first slice → reinitialising from scratch.")
        shard_model.apply(_reset_weights)
        start_slice = 0
        # No previous optimizer state — fresh optimizer
        optimizer = optim.Adam(shard_model.parameters(), lr=lr)
    else:
        prev_ckpt = checkpoints[affected_slice - 1]
        print(f"  Restoring checkpoint from slice {affected_slice} "
              f"(trained on {len(prev_ckpt['cumulative_indices'])} samples).")
        shard_model.load_state_dict(prev_ckpt['state_dict'])
        optimizer = optim.Adam(shard_model.parameters(), lr=lr)
        optimizer.load_state_dict(prev_ckpt['optimizer_state'])
        start_slice = affected_slice

    shard_model = shard_model.to(device)
    remaining_set_local = set(remaining_shard_indices)

    # Retrain from affected_slice onward
    # Rebuild slice boundaries from the original checkpoints, filtering out deleted
    total_retrain_epochs = (num_slices - start_slice) * num_epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_retrain_epochs, 1)
    )

    for s in range(start_slice, num_slices):
        # Cumulative indices for this slice, minus deleted
        raw_cumulative = checkpoints[s]['cumulative_indices']
        clean_cumulative = [i for i in raw_cumulative if i in remaining_set_local]

        print(f"  Slice {s + 1}/{num_slices}: "
              f"{len(raw_cumulative)} → {len(clean_cumulative)} samples (deleted removed)")

        if len(clean_cumulative) == 0:
            print(f"  Slice {s + 1} empty after deletion, skipping.")
            continue

        shard_subset = Subset(base_dataset, clean_cumulative)
        shard_loader = DataLoader(shard_subset, batch_size=batch_size,
                                  shuffle=True, num_workers=2, pin_memory=True)

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
                total += labels.size(0)
            scheduler.step()
            acc = 100.0 * correct / total
            print(f"    Slice {s+1} Epoch [{epoch+1}/{num_epochs}]  "
                  f"Loss: {epoch_loss/len(shard_loader):.4f}  Acc: {acc:.2f}%")

        # Update checkpoint for this slice (updated state after deletion)
        sisa_model.shard_checkpoints[shard_id][s]['state_dict'] = \
            copy.deepcopy(shard_model.state_dict())
        sisa_model.shard_checkpoints[shard_id][s]['cumulative_indices'] = \
            clean_cumulative

    sisa_model.shard_models[shard_id] = shard_model.cpu()
    sisa_model.shard_indices[shard_id] = remaining_shard_indices
    print(f"  Shard {shard_id} slice-aware retrain complete. "
          f"Retrained slices {start_slice+1}–{num_slices}.")


# ----------------------------------------------------------------------
# FULL SHARD RETRAIN (Case B)
# ----------------------------------------------------------------------

def _retrain_shard_from_scratch(sisa_model, shard_id, remaining_indices,
                                 base_dataset, num_epochs, batch_size,
                                 lr, device, criterion):
    sisa_model.shard_models[shard_id].apply(_reset_weights)
    shard_model = sisa_model.shard_models[shard_id].to(device)
    optimizer = optim.Adam(shard_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    shard_subset = Subset(base_dataset, remaining_indices)
    shard_loader = DataLoader(shard_subset, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)

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
            total += labels.size(0)
        scheduler.step()
        acc = 100.0 * correct / total
        print(f"  Epoch [{epoch+1}/{num_epochs}]  "
              f"Loss: {epoch_loss/len(shard_loader):.4f}  Acc: {acc:.2f}%")

    sisa_model.shard_models[shard_id] = shard_model.cpu()
    sisa_model.shard_indices[shard_id] = remaining_indices


# ----------------------------------------------------------------------
# FALLBACK FINE-TUNE (Case C)
# ----------------------------------------------------------------------

def _fallback_finetune(model, remaining_dataset, num_epochs, batch_size, lr, device):
    unlearn_model = copy.deepcopy(model).to(device)
    remaining_loader = DataLoader(remaining_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=2, pin_memory=True)
    optimizer = optim.Adam(unlearn_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
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
            total += labels.size(0)
        acc = correct / total
        scheduler.step()
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(unlearn_model.state_dict())
        print(f"  [SISA Fallback] Epoch [{epoch+1}/{num_epochs}]  "
              f"Loss: {epoch_loss/len(remaining_loader):.4f}  Acc: {acc:.4f}")

    unlearn_model.load_state_dict(best_state)
    return unlearn_model.cpu()


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------

def _reset_weights(module):
    if hasattr(module, 'reset_parameters'):
        module.reset_parameters()

def _get_base_dataset(dataset):
    from torch.utils.data import Subset
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset