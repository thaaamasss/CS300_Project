"""
retraining_unlearning.py — Fixed
Fixes applied:
  #9  Returns best-accuracy checkpoint, not final epoch weights

Gold standard unlearning: retrain from scratch on remaining data only.
Deleted samples never touch the new model — provable privacy guarantee.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy


def retraining_unlearning(model, remaining_dataset, deleted_dataset,
                          num_classes, in_channels, num_epochs=10,
                          batch_size=64, lr=0.001, device=None):
    """
    Retrain a fresh model from scratch on remaining_dataset only.
    deleted_dataset is accepted but intentionally never used.

    Returns:
        best model (by training accuracy) trained on remaining data only
    """
    from models.architectures.cnn_model import CNNModel

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Fresh model — old weights discarded entirely
    new_model = CNNModel(num_classes=num_classes, in_channels=in_channels).to(device)

    remaining_loader = DataLoader(remaining_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=2, pin_memory=True)

    optimizer = optim.Adam(new_model.parameters(), lr=lr)

    # FIX #7: cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    criterion = nn.CrossEntropyLoss()

    # FIX #9: best-checkpoint tracking
    best_acc   = 0.0
    best_state = None

    for epoch in range(num_epochs):
        new_model.train()
        epoch_loss = 0.0
        correct = total = 0

        for inputs, labels in remaining_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = new_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

        acc = correct / total
        scheduler.step()   # FIX #7

        # FIX #9: save best epoch
        if acc > best_acc:
            best_acc   = acc
            best_state = copy.deepcopy(new_model.state_dict())

        print(f"  [Retraining] Epoch [{epoch+1}/{num_epochs}]  "
              f"Loss: {epoch_loss/len(remaining_loader):.4f}  Acc: {acc:.4f}")

    # FIX #9: restore best weights
    new_model.load_state_dict(best_state)
    print(f"  [Retraining] Best epoch accuracy: {best_acc:.4f}")

    return new_model.cpu()
