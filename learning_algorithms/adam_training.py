"""
adam_training.py — Fixed
Fixes applied:
  #1  Returns (model, final_loss) instead of just model
  #7  Added CosineAnnealingLR scheduler
  #9  Saves and returns best-accuracy checkpoint
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy


def adam_training(dataset, num_classes, input_channels, input_size, num_epochs=10,
                  batch_size=64, lr=0.001, device=None):
    """
    Train a CNNModel with Adam + cosine annealing.

    Returns:
        (best_model, final_loss) tuple
    """
    from models.architectures.cnn_model import CNNModel

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=True)

    model = CNNModel(input_channels=input_channels, num_classes=num_classes, input_size=input_size).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # FIX #7: cosine annealing LR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    criterion = nn.CrossEntropyLoss()

    # FIX #9: track best model checkpoint
    best_acc   = 0.0
    best_state = None
    final_loss = 0.0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = total = 0

        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

        final_loss = epoch_loss / len(loader)
        acc = correct / total
        scheduler.step()   # FIX #7

        # FIX #9: save best checkpoint
        if acc > best_acc:
            best_acc   = acc
            best_state = copy.deepcopy(model.state_dict())

        print(f"  [Adam] Epoch [{epoch+1}/{num_epochs}]  "
              f"Loss: {final_loss:.4f}  Acc: {acc:.4f}  LR: {scheduler.get_last_lr()[0]:.6f}")

    # FIX #9: restore best weights
    model.load_state_dict(best_state)

    return model.cpu(), final_loss   # FIX #1
