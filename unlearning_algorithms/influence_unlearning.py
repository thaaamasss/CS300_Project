import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import copy
import random


def influence_unlearning(model, remaining_dataset, deleted_dataset,
                         num_classes, input_channels, input_size, num_epochs=10,
                         batch_size=64, lr=1e-4, device=None,
                         cg_iterations=20, cg_damping=1e-2,
                         cg_samples=1000, recovery_epochs=3):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    unlearn_model = copy.deepcopy(model).to(device)

    # ------------------------------------------------------------------
    # STEP 1: Compute gradient of loss on deleted_dataset (the RHS vector b)
    # ------------------------------------------------------------------
    print("[Influence-CG] Step 1: Computing deletion gradient (b = grad_deleted)...")

    deleted_loader = DataLoader(deleted_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=2, pin_memory=True)
    criterion = nn.CrossEntropyLoss()

    unlearn_model.train()
    unlearn_model.zero_grad()

    total_loss = torch.tensor(0.0, device=device)
    num_deleted = 0

    for inputs, labels in deleted_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = unlearn_model(inputs)
        loss = criterion(outputs, labels)
        total_loss = total_loss + loss * inputs.size(0)
        num_deleted += inputs.size(0)

    avg_loss = total_loss / num_deleted
    avg_loss.backward()

    # b: flat gradient vector — the vector we want to left-multiply by H^-1
    b = torch.cat([
        p.grad.detach().flatten() if p.grad is not None
        else torch.zeros(p.numel(), device=device)
        for p in unlearn_model.parameters()
    ]).clone()

    print(f"  Deletion gradient norm: {b.norm().item():.6f}  ({num_deleted} deleted samples)")
    unlearn_model.zero_grad()

    # ------------------------------------------------------------------
    # STEP 2: Conjugate Gradient — solve H*x = b
    # ------------------------------------------------------------------
    print(f"\n[Influence-CG] Step 2: Running CG ({cg_iterations} iterations, "
          f"damping={cg_damping}, samples={cg_samples})...")

    # Subsample remaining_dataset for HVP estimation
    n = len(remaining_dataset)
    hvp_indices = random.sample(range(n), min(cg_samples, n))
    hvp_subset = Subset(remaining_dataset, hvp_indices)

    x = _conjugate_gradient(
        model=unlearn_model,
        b=b,
        dataset=hvp_subset,
        batch_size=batch_size,
        damping=cg_damping,
        num_iterations=cg_iterations,
        device=device
    )

    print(f"  CG solution norm: {x.norm().item():.6f}")
    print(f"  Scale ratio (CG/raw): {(x.norm() / (b.norm() + 1e-10)).item():.4f}")

    # ------------------------------------------------------------------
    # STEP 3: Apply influence update — theta = theta - lr * H^-1 * grad
    # ------------------------------------------------------------------
    print(f"\n[Influence-CG] Step 3: Applying influence update (lr={lr})...")

    unlearn_model.zero_grad()
    offset = 0
    with torch.no_grad():
        for param in unlearn_model.parameters():
            numel = param.numel()
            param.data -= lr * x[offset: offset + numel].view(param.shape)
            offset += numel

    # ------------------------------------------------------------------
    # STEP 4: Recovery fine-tuning
    # ------------------------------------------------------------------
    if recovery_epochs > 0:
        print(f"\n[Influence-CG] Step 4: Recovery fine-tuning "
              f"({recovery_epochs} epochs on remaining data)...")

        remaining_loader = DataLoader(remaining_dataset, batch_size=batch_size,
                                      shuffle=True, num_workers=2, pin_memory=True)
        optimizer = optim.Adam(unlearn_model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=recovery_epochs)

        best_acc = 0.0
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
                total += labels.size(0)

            acc = correct / total
            scheduler.step()

            if acc > best_acc:
                best_acc = acc
                best_state = copy.deepcopy(unlearn_model.state_dict())

            print(f"  Recovery Epoch [{epoch+1}/{recovery_epochs}]  "
                  f"Loss: {epoch_loss/len(remaining_loader):.4f}  Acc: {acc:.4f}")

        unlearn_model.load_state_dict(best_state)
        print(f"  Recovery best accuracy: {best_acc:.4f}")

    print("\n[Influence-CG] Unlearning complete.")
    return unlearn_model.cpu()


# ----------------------------------------------------------------------
# CONJUGATE GRADIENT SOLVER
# ----------------------------------------------------------------------

def _conjugate_gradient(model, b, dataset, batch_size, damping,
                        num_iterations, device):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    r_dot_r = torch.dot(r, r)

    for i in range(num_iterations):
        # Compute H*p via HVP
        Hp = _hessian_vector_product(model, p, dataset, batch_size, device)
        # Add damping: (H + lambda*I) * p
        Hp = Hp + damping * p

        pHp = torch.dot(p, Hp)

        # Avoid division by zero
        if pHp.abs().item() < 1e-12:
            print(f"  CG early stop at iteration {i+1} (pHp too small)")
            break

        alpha = r_dot_r / pHp
        x = x + alpha * p
        r = r - alpha * Hp

        r_dot_r_new = torch.dot(r, r)
        residual = r_dot_r_new.sqrt().item()

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  CG iter {i+1:3d}  residual: {residual:.6f}")

        # Convergence check
        if residual < 1e-6:
            print(f"  CG converged at iteration {i+1}")
            break

        beta = r_dot_r_new / r_dot_r
        p = r + beta * p
        r_dot_r = r_dot_r_new

    return x


def _hessian_vector_product(model, v, dataset, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    criterion = nn.CrossEntropyLoss()

    num_params = sum(p.numel() for p in model.parameters())
    Hv = torch.zeros(num_params, device=device)
    num_samples = 0

    model.train()

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        batch_size_actual = inputs.size(0)

        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # First-order gradients with create_graph=True (needed for second-order)
        grads = torch.autograd.grad(loss, model.parameters(),
                                    create_graph=True, retain_graph=True)
        flat_grad = torch.cat([g.flatten() for g in grads])

        # Dot product of gradient with v (scalar)
        grad_v = torch.dot(flat_grad, v)

        # Second-order: gradient of (grad · v) w.r.t. params = H * v
        grads2 = torch.autograd.grad(grad_v, model.parameters(),
                                     retain_graph=False)
        flat_hv = torch.cat([g.detach().flatten() for g in grads2])

        Hv += flat_hv * batch_size_actual
        num_samples += batch_size_actual

    Hv /= num_samples
    return Hv