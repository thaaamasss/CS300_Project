"""
influence_unlearning.py — FIM Diagonal Hessian Implementation
--------------------------------------------------------------
Previous implementation assumed H^-1 = Identity (raw gradient negation).
This version approximates H^-1 using the diagonal of the Fisher Information Matrix (FIM).

Theory:
    True influence unlearning update:
        theta_new = theta - H^-1 * grad_L(deleted, theta)

    H^-1 is intractable (708 GB for this model). FIM diagonal approximation:
        H ≈ diag(FIM)
        H^-1 ≈ 1 / diag(FIM)   (element-wise reciprocal)

    FIM diagonal for parameter theta_i:
        F_ii = E[ (d/d_theta_i  log p(y|x, theta))^2 ]
             ≈ (1/N) * sum over dataset of (grad_i of log_softmax)^2

    This gives a per-parameter scaling factor. Parameters with high curvature
    (large FIM diagonal) get smaller updates — the model is more "certain" about
    those weights. Parameters with low curvature get larger updates.

    This is strictly better than H^-1 = Identity because:
    - It respects the geometry of the loss landscape
    - It prevents large updates to high-curvature (shared/important) parameters
    - It reduces catastrophic collapse on class deletion

Why FIM diagonal and not full FIM:
    Full FIM = N x N matrix = 708 GB for this model. Diagonal = N floats = 1.6 MB.

Steps:
    1. Compute FIM diagonal on remaining_dataset (captures current loss geometry)
    2. Compute gradient of loss on deleted_dataset
    3. Scale gradient by 1/FIM_diag (the H^-1 approximation)
    4. Apply the scaled update: theta = theta - scaled_grad
    5. Fine-tune briefly on remaining_dataset to recover any collateral damage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import copy


def influence_unlearning(model, remaining_dataset, deleted_dataset,
                         num_classes, input_channels, input_size, num_epochs=10,
                         batch_size=64, lr=0.0005, device=None,
                         fim_samples=2000, fim_damping=1e-3,
                         recovery_epochs=2):
    """
    Influence-based unlearning with FIM diagonal Hessian approximation.

    Args:
        model             : trained model to unlearn from
        remaining_dataset : dataset after deletion (used for FIM + recovery)
        deleted_dataset   : samples to forget
        num_classes       : number of output classes
        in_channels       : input channels (1 for grayscale, 3 for RGB)
        num_epochs        : not used for the influence step, kept for API consistency
        batch_size        : dataloader batch size
        lr                : learning rate for recovery fine-tuning
        device            : torch.device
        fim_samples       : number of samples to use for FIM estimation
                            (2000 is enough for a stable diagonal estimate)
        fim_damping       : damping constant added to FIM diagonal to prevent
                            division by near-zero values (lambda in (F + lambda*I)^-1)
        recovery_epochs   : short fine-tuning on remaining data after influence step
                            to recover any collateral damage from the update

    Returns:
        unlearned model (cpu)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    unlearn_model = copy.deepcopy(model).to(device)
    unlearn_model.eval()

    # ------------------------------------------------------------------
    # STEP 1: Compute FIM diagonal on remaining_dataset
    # ------------------------------------------------------------------
    print("[Influence-FIM] Step 1: Computing FIM diagonal on remaining data...")

    fim_diagonal = _compute_fim_diagonal(
        unlearn_model, remaining_dataset,
        num_samples=fim_samples, batch_size=batch_size, device=device
    )

    # Add damping: (FIM + lambda*I)^-1 — prevents exploding updates where FIM ~ 0
    fim_inv_diagonal = 1.0 / (fim_diagonal + fim_damping)

    fim_mean   = fim_diagonal.mean().item()
    fim_min    = fim_diagonal.min().item()
    fim_max    = fim_diagonal.max().item()
    print(f"  FIM diagonal — mean: {fim_mean:.6f}  min: {fim_min:.6f}  max: {fim_max:.6f}")
    print(f"  Damping: {fim_damping}  |  H^-1 scale range: "
          f"[{fim_inv_diagonal.min().item():.2f}, {fim_inv_diagonal.max().item():.2f}]")

    # ------------------------------------------------------------------
    # STEP 2: Compute gradient of loss on deleted_dataset
    # ------------------------------------------------------------------
    print("\n[Influence-FIM] Step 2: Computing deletion gradient...")

    deleted_loader = DataLoader(deleted_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=2, pin_memory=True)
    criterion = nn.CrossEntropyLoss()

    unlearn_model.zero_grad()
    unlearn_model.train()

    total_loss = torch.tensor(0.0, device=device)
    num_deleted = 0

    for inputs, labels in deleted_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = unlearn_model(inputs)
        loss = criterion(outputs, labels)
        total_loss = total_loss + loss * inputs.size(0)
        num_deleted += inputs.size(0)

    # Average loss over all deleted samples
    avg_loss = total_loss / num_deleted
    avg_loss.backward()

    # Collect flat gradient vector across all parameters
    deletion_grad = torch.cat([
        p.grad.flatten() if p.grad is not None else torch.zeros(p.numel(), device=device)
        for p in unlearn_model.parameters()
    ])
    print(f"  Deletion gradient norm (raw):          {deletion_grad.norm().item():.6f}")

    # ------------------------------------------------------------------
    # STEP 3: Scale gradient by H^-1 (FIM diagonal inverse)
    # ------------------------------------------------------------------
    print("\n[Influence-FIM] Step 3: Applying H^-1 scaling...")

    scaled_grad = fim_inv_diagonal * deletion_grad

    print(f"  Scaled gradient norm (after H^-1):     {scaled_grad.norm().item():.6f}")
    print(f"  Scale ratio (scaled/raw):              "
          f"{(scaled_grad.norm() / (deletion_grad.norm() + 1e-10)).item():.4f}")

    # ------------------------------------------------------------------
    # STEP 4: Apply the influence update — theta = theta - H^-1 * grad
    # ------------------------------------------------------------------
    print("\n[Influence-FIM] Step 4: Applying influence parameter update...")

    unlearn_model.zero_grad()
    offset = 0
    with torch.no_grad():
        for param in unlearn_model.parameters():
            numel = param.numel()
            param_update = scaled_grad[offset: offset + numel].view(param.shape)
            param.data -= param_update      # subtract: remove the influence of deleted samples
            offset += numel

    # ------------------------------------------------------------------
    # STEP 5: Short recovery fine-tuning on remaining data
    # Influence step can introduce small perturbations to shared features.
    # A brief fine-tune on remaining data corrects these without re-learning deleted samples.
    # ------------------------------------------------------------------
    if recovery_epochs > 0:
        print(f"\n[Influence-FIM] Step 5: Recovery fine-tuning "
              f"({recovery_epochs} epochs on remaining data)...")

        remaining_loader = DataLoader(remaining_dataset, batch_size=batch_size,
                                      shuffle=True, num_workers=2, pin_memory=True)
        optimizer = optim.Adam(unlearn_model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=recovery_epochs)

        best_acc   = 0.0
        best_state = copy.deepcopy(unlearn_model.state_dict())

        for epoch in range(recovery_epochs):
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

            print(f"  Recovery Epoch [{epoch+1}/{recovery_epochs}]  "
                  f"Loss: {epoch_loss/len(remaining_loader):.4f}  Acc: {acc:.4f}")

        unlearn_model.load_state_dict(best_state)
        print(f"  Recovery best accuracy: {best_acc:.4f}")

    print("\n[Influence-FIM] Unlearning complete.")
    return unlearn_model.cpu()


# ----------------------------------------------------------------------
# FIM DIAGONAL COMPUTATION
# ----------------------------------------------------------------------

def _compute_fim_diagonal(model, dataset, num_samples, batch_size, device):
    """
    Compute the diagonal of the Fisher Information Matrix.

    FIM_ii = E[ (d/d_theta_i  log p(y|x, theta))^2 ]

    Estimated as:
        (1/N) * sum_{x,y in dataset} (grad_i of log p(y|x))^2

    This uses the empirical FIM — gradients are computed at the true labels
    (not sampled from the model's predicted distribution). Empirical FIM is
    standard practice and equivalent to the true FIM for well-trained models.

    Args:
        model       : model to compute FIM for (should be in eval mode)
        dataset     : dataset to estimate FIM on (remaining_dataset)
        num_samples : number of samples to use (subset for efficiency)
        batch_size  : batch size for dataloader
        device      : torch.device

    Returns:
        fim_diag: 1D tensor of shape (num_params,) — the FIM diagonal
    """
    from torch.utils.data import Subset
    import random

    # Subsample for efficiency — 2000 samples gives stable estimate
    n = len(dataset)
    if num_samples < n:
        indices = random.sample(range(n), num_samples)
        subset = Subset(dataset, indices)
    else:
        subset = dataset

    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    # Count total parameters
    num_params = sum(p.numel() for p in model.parameters())
    fim_diag = torch.zeros(num_params, device=device)

    model.eval()
    num_processed = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        for i in range(inputs.size(0)):
            # Per-sample gradient computation
            model.zero_grad()

            output = model(inputs[i].unsqueeze(0))
            log_prob = F.log_softmax(output, dim=1)[0, labels[i]]

            # Compute gradient of log p(y|x) w.r.t. all parameters
            grads = torch.autograd.grad(log_prob, model.parameters(),
                                        retain_graph=False, create_graph=False)

            # Accumulate squared gradients into FIM diagonal
            flat_grad = torch.cat([g.flatten() for g in grads])
            fim_diag += flat_grad.pow(2)
            num_processed += 1

    # Normalise by number of samples
    fim_diag /= num_processed

    print(f"  FIM diagonal computed over {num_processed} samples, "
          f"{num_params:,} parameters.")

    return fim_diag
