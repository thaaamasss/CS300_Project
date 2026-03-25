import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy


def finetune_unlearning(model, remaining_dataset, deleted_dataset,
                        num_classes, input_channels, input_size, num_epochs=5,
                        batch_size=64, lr=0.0005, device=None,
                        freeze_backbone=True):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    unlearn_model = copy.deepcopy(model).to(device)

    # Per-layer freezing
    if freeze_backbone:
        _freeze_backbone(unlearn_model)
        trainable_params = [p for p in unlearn_model.parameters() if p.requires_grad]
        frozen_params = [p for p in unlearn_model.parameters() if not p.requires_grad]
        print(f"  [FineTuning] Backbone frozen. "
              f"Trainable params: {sum(p.numel() for p in trainable_params):,}  "
              f"Frozen params: {sum(p.numel() for p in frozen_params):,}")
    else:
        trainable_params = list(unlearn_model.parameters())
        print(f"  [FineTuning] All layers trainable "
              f"({sum(p.numel() for p in trainable_params):,} params).")

    remaining_loader = DataLoader(remaining_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=2, pin_memory=True)

    optimizer = optim.Adam(
        [p for p in unlearn_model.parameters() if p.requires_grad], lr=lr
    )
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

        print(f"  [FineTuning] Epoch [{epoch+1}/{num_epochs}]  "
              f"Loss: {epoch_loss/len(remaining_loader):.4f}  Acc: {acc:.4f}")

    unlearn_model.load_state_dict(best_state)
    print(f"  [FineTuning] Best epoch accuracy: {best_acc:.4f}")

    return unlearn_model.cpu()


# ----------------------------------------------------------------------
# HELPER
# ----------------------------------------------------------------------

def _freeze_backbone(model):
    frozen_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            for param in module.parameters():
                param.requires_grad = False
            frozen_layers.append(name)
    if frozen_layers:
        print(f"  [FineTuning] Frozen layers: {frozen_layers}")
    else:
        print("  [FineTuning] WARNING: no Conv2d layers found to freeze.")