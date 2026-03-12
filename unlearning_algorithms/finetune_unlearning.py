"""
finetune_unlearning.py — Fixed
Fixes applied:
  #7  Added CosineAnnealingLR scheduler
  #9  Returns best-accuracy checkpoint, not final epoch weights

Fine-tuning unlearning: continue training the existing model on remaining data.
Passive forgetting — no explicit forgetting signal.
Does NOT provide a provable privacy guarantee.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy


def finetune_unlearning(model, remaining_dataset, deleted_dataset,
                        num_classes, in_channels, num_epochs=5,
                        batch_size=64, lr=0.0005, device=None):
    """
    Fine-tune the existing model on remaining_dataset only.
    deleted_dataset is accepted but intentionally never used.

    Returns:
        best model (by training accuracy) after fine-tuning
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Deep copy — do not modify the original model
    unlearn_model = copy.deepcopy(model).to(device)

    remaining_loader = DataLoader(remaining_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=2, pin_memory=True)

    optimizer = optim.Adam(unlearn_model.parameters(), lr=lr)

    # FIX #7: scheduler over fine-tuning epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    criterion = nn.CrossEntropyLoss()

    # FIX #9: best-checkpoint tracking
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
        scheduler.step()   # FIX #7

        # FIX #9: save best checkpoint
        if acc > best_acc:
            best_acc   = acc
            best_state = copy.deepcopy(unlearn_model.state_dict())

        print(f"  [FineTuning] Epoch [{epoch+1}/{num_epochs}]  "
              f"Loss: {epoch_loss/len(remaining_loader):.4f}  Acc: {acc:.4f}")

    # FIX #9: restore best weights
    unlearn_model.load_state_dict(best_state)
    print(f"  [FineTuning] Best epoch accuracy: {best_acc:.4f}")

    return unlearn_model.cpu()
