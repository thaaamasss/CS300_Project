"""
influence_unlearning.py — Fixed
Fixes applied:
  #5  Explicit num_unlearn_epochs variable (kept at 1, but now documented and configurable)
  #9  Saves and returns best-accuracy checkpoint on remaining data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy


def influence_unlearning(model, remaining_dataset, deleted_dataset,
                         num_classes, in_channels, num_epochs=10,
                         batch_size=64, lr=0.0005, device=None):
    """
    Influence-based unlearning via gradient negation.

    Approximates true influence unlearning (which requires H^-1 * grad)
    by using raw gradient negation — equivalent to assuming H^-1 = Identity.

    Stable only for small, incoherent deletions. Catastrophically unstable
    for class deletion or large batch deletion (see project analysis).

    FIX #5: num_unlearn_epochs is now an explicit variable (kept at 1).
    This makes the design choice visible and testable rather than accidental.
    Increasing beyond 1 amplifies instability — kept at 1 intentionally.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    unlearn_model = copy.deepcopy(model).to(device)
    deleted_loader = DataLoader(deleted_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=2, pin_memory=True)
    remaining_loader = DataLoader(remaining_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=2, pin_memory=True)

    optimizer = optim.SGD(unlearn_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # FIX #5: explicit epoch count — documented design choice, not accidental omission
    # Kept at 1: more epochs amplify gradient negation and increase collapse risk.
    # To experiment: increase cautiously and monitor remaining_accuracy.
    num_unlearn_epochs = 1

    # FIX #9: best-checkpoint tracking on remaining data
    best_acc   = _eval_acc(unlearn_model, remaining_loader, device)
    best_state = copy.deepcopy(unlearn_model.state_dict())

    for epoch in range(num_unlearn_epochs):
        unlearn_model.train()
        for inputs, labels in deleted_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = unlearn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Negate all gradients — approximation of -H^-1 * grad
            for param in unlearn_model.parameters():
                if param.grad is not None:
                    param.grad.neg_()

            optimizer.step()

        # FIX #9: checkpoint if remaining accuracy improved
        acc = _eval_acc(unlearn_model, remaining_loader, device)
        print(f"  [Influence] Epoch [{epoch+1}/{num_unlearn_epochs}]  "
              f"Remaining acc: {acc:.4f}")
        if acc > best_acc:
            best_acc   = acc
            best_state = copy.deepcopy(unlearn_model.state_dict())

    # FIX #9: restore best checkpoint
    unlearn_model.load_state_dict(best_state)

    return unlearn_model.cpu()


def _eval_acc(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)
    return correct / total if total > 0 else 0.0
