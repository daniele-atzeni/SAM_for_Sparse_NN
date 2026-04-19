import torch
from torch import nn
import numpy as np

from src.pyhessian import hessian


import torch
from torch import nn
import numpy as np


@torch.no_grad()
def masked_grad_norm(model):
    norms = []
    for p in model.parameters():
        if p.grad is None:
            continue

        if hasattr(p, "weight_mask"):
            g = p.grad * p.weight_mask
        else:
            g = p.grad

        norms.append(torch.norm(g, p=2))

    return torch.norm(torch.stack(norms), p=2)


@torch.no_grad()
def sam_perturbation(model, rho, pruned: bool = False):
    if pruned:
        grad_norm = masked_grad_norm(model)
    else:
        grad_norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2)
                for p in model.parameters()
                if p.grad is not None
            ]),
            p=2
        )

    scale = rho / (grad_norm + 1e-12)

    for p in model.parameters():
        if p.grad is None:
            continue

        if pruned and hasattr(p, "weight_mask"):
            p.add_(p.grad * p.weight_mask * scale)
        else:
            p.add_(p.grad * scale)

    return scale

@torch.no_grad()
def sam_restore(model, scale, pruned: bool = False):
    for p in model.parameters():
        if p.grad is None:
            continue

        if pruned and hasattr(p, "weight_mask"):
            p.sub_(p.grad * p.weight_mask * scale)
        else:
            p.sub_(p.grad * scale)

@torch.no_grad()
def random_perturbation_fixed_norm(model, std: float, pruned: bool = False):
    noises = []
    norms = []

    for p in model.parameters():
        if not p.requires_grad:
            noises.append(None)
            continue

        noise = torch.randn_like(p)

        if pruned and hasattr(p, "weight_mask"):
            noise = noise * p.weight_mask

        noises.append(noise)
        norms.append(torch.norm(noise, p=2))

    # global L2 norm across all parameters
    global_norm = torch.norm(torch.stack(norms), p=2)
    scale = std / (global_norm + 1e-12)

    for p, noise in zip(model.parameters(), noises):
        if noise is not None:
            p.add_(noise * scale)

    return noises, scale


@torch.no_grad()
def random_restore_fixed_norm(model, noises, scale):
    for p, noise in zip(model.parameters(), noises):
        if noise is not None:
            p.sub_(noise * scale)


def _build_param_masks(model, device):
    """List of masks aligned with model.parameters() order (ones where no pruning)."""
    masks = []
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            mask_name = name.replace('weight', 'weight_mask').replace('bias', 'bias_mask')
            if hasattr(module, mask_name):
                masks.append(getattr(module, mask_name).to(device))
            else:
                masks.append(torch.ones_like(param, device=device))
    return masks


@torch.no_grad()
def weight_distribution_metrics(model) -> dict:
    """L1/L2 ratio, Gini coefficient, pruned weight norm (Propositions 3 & 6).

    Only considers weight tensors (not biases or BN affine params) so that the
    metrics reflect the pruned coordinate structure.
    """
    active_w, pruned_w = [], []
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if 'weight' not in name:
                continue
            if hasattr(module, 'weight_mask'):
                mask = module.weight_mask.bool()
                w = param.detach()
                active_w.append(w[mask].flatten())
                pruned_w.append(w[~mask].flatten())
            else:
                active_w.append(param.detach().flatten())

    active = torch.cat(active_w) if active_w else torch.zeros(1)
    pruned = torch.cat(pruned_w) if pruned_w else torch.zeros(1)

    l1 = active.abs().sum().item()
    l2 = active.norm(2).item()
    l1_l2_ratio = l1 / (l2 + 1e-12)

    # Gini coefficient — higher means more weight concentrated on fewer params (SAM prediction)
    w_np = np.sort(active.abs().cpu().numpy())
    n = len(w_np)
    if n > 0 and w_np.sum() > 0:
        idx = np.arange(1, n + 1)
        gini = float((2 * (idx * w_np).sum() / (n * w_np.sum())) - (n + 1) / n)
    else:
        gini = 0.0

    return {
        "l1_l2_ratio": l1_l2_ratio,
        "gini": gini,
        "pruned_weight_norm": pruned.norm(2).item(),
        "active_weight_count": int(active.numel()),
    }


def post_pruning_metrics(model, device, data_loader, criterion) -> dict:
    """Masked gradient norm + weight distribution immediately after pruning.

    Computes ||P_m^T ∇L(θ^(m))|| and ||(1-m)⊙θ*|| (Proposition 3) plus the
    weight distribution metrics for Proposition 6.  Call this before any
    retraining after a pruning step.
    """
    model.eval()

    data, target = next(iter(data_loader))
    data, target = data.to(device), target.to(device)

    model.zero_grad()
    output = model(data)
    loss = criterion(output, target).mean()
    loss.backward()

    norms = []
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if param.grad is None:
                continue
            mask_name = name.replace('weight', 'weight_mask').replace('bias', 'bias_mask')
            if hasattr(module, mask_name):
                g = param.grad * getattr(module, mask_name)
            else:
                g = param.grad
            norms.append(g.norm(2))

    grad_norm = torch.norm(torch.stack(norms), 2).item() if norms else 0.0
    model.zero_grad()

    dist = weight_distribution_metrics(model)
    return {"masked_grad_norm": grad_norm, **dist}


def eigenvector_alignment(model, device, data_loader, n_batch: int = 1) -> float:
    """||P_m^T v1||^2: mass of the full Hessian top eigenvector on active coords.

    Computes the top eigenvector of the *full* (unmasked) Hessian and measures
    what fraction of its squared norm lies on the active (unpruned) coordinates.
    Close to 1 means the dominant curvature direction is preserved by the mask.
    """
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()

    v1 = None
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        cuda = device.type == 'cuda' if isinstance(device, torch.device) else ('cuda' in device)
        hessian_comp = hessian(model, loss_fn, data=(inputs, targets), cuda=cuda)
        _, batch_eigvecs = hessian_comp.eigenvalues(top_n=1)
        v1 = batch_eigvecs[0]
        if batch_idx == n_batch - 1:
            break

    if v1 is None:
        return 0.0

    masks = _build_param_masks(model, v1[0].device)
    sq_norm = sum((v1_i * m).pow(2).sum().item() for v1_i, m in zip(v1, masks))
    return float(sq_norm)


def hessian_flatness(
        device: str | torch.device,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        n_eigs: int,
        pruned: bool = False,
        n_batch: int = 1,
        ) -> dict:

    # set the loss function
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()

    trace = None
    eigenvals = None
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        cuda = device.type=='cuda' if isinstance(device, torch.device) else ('cuda' in device)
        hessian_comp = hessian(model, loss_fn, data=(inputs, targets), cuda=cuda)


        print(f"Computing Hessian trace...")
        if not pruned:
            batch_trace = np.mean(hessian_comp.trace())
        else:
            batch_trace = np.mean(hessian_comp.pruned_trace())
        print(f"Trace: {batch_trace:.6f}")

        print(f"Computing top {n_eigs} eigenvalues...")
        if not pruned:
            batch_eigenvals, _ = hessian_comp.eigenvalues(top_n=n_eigs)
        else:
            batch_eigenvals, _ = hessian_comp.pruned_eigenvalues(top_n=n_eigs)

        print(f"Batch {batch_idx}: Trace: {batch_trace:.6f}, eigval: " + " ".join([
            f"({idx}) {eigval:.6f}" for idx, eigval in enumerate(batch_eigenvals)
        ]))

        if trace is None:
          trace = batch_trace
        else:
          trace += batch_trace
        if eigenvals is None:
            eigenvals = batch_eigenvals
        else:
            eigenvals = [e1 + e2 for e1, e2 in zip(eigenvals, batch_eigenvals)]

        if batch_idx == n_batch - 1:
            break

    trace = trace / (batch_idx + 1)
    eigenvals = [eigval / (batch_idx + 1) for eigval in eigenvals]

    # Count active parameters via module-level mask access (parameter-level .weight_mask
    # does not exist for torch.nn.utils.prune — masks live on the module as buffers).
    count_total = 0   # all active params (including biases / BN affine)
    count_weights = 0  # active weight-only params (the pruned coordinate subspace)
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            mask_name = name.replace('weight', 'weight_mask').replace('bias', 'bias_mask')
            if pruned and hasattr(module, mask_name):
                active = int(getattr(module, mask_name).sum().item())
            else:
                active = param.numel()
            count_total += active
            if 'weight' in name and 'bias' not in name:
                count_weights += active

    trace_per_param = trace / (count_total + 1e-8)
    # tr(H_m)/k where k = active weight params only (Propositions 1 & 2)
    trace_per_active_weight = trace / (count_weights + 1e-8)

    return_dict = {
        "trace": trace,
        "trace_per_param": trace_per_param,
        "trace_per_active_weight": trace_per_active_weight,
    }
    for idx, eigval in enumerate(eigenvals):
        return_dict[f"eigval_{idx}"] = eigval
    return return_dict


def evaluate(
    model: nn.Module,
    device: str | torch.device,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    rho: float = 0.5,
    std: float = 20,
    pruned: bool = False,
    n_batch: int = 1,
    evaluate_flatness: bool = True,
    eval_batches: int = None
    ) -> dict:

    model.eval()

    # perturbation-based metrics
    batch_loss = []
    batch_sam_loss = []
    batch_corrects = []
    batch_random_loss = []
    batch_random_sgd = []
    n_samples = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if eval_batches is not None and batch_idx >= eval_batches:
            break
        data, target = data.to(device), target.to(device)

        # ---- first forward + backward ----
        output = model(data)
        loss = criterion(output, target).mean()
        loss.backward()

        # ---- perturb weights ----
        scale = sam_perturbation(model, rho, pruned)

        # ---- second forward (sharpness-aware loss) ----
        with torch.no_grad():
            output_perturbed = model(data)
            sam_loss = criterion(output_perturbed, target).mean()

        # ---- restore weights & clean grads ----
        sam_restore(model, scale, pruned)
        model.zero_grad()

        # append batch metrics
        n_samples += data.size(0)
        batch_loss.append(loss.item() * data.size(0))
        batch_sam_loss.append((sam_loss.item() - loss.item()) * data.size(0))
        batch_corrects.append(
            output.argmax(dim=1).eq(target).sum().item()
        )

        # ---- random fixed-norm perturbation (5 repeats) ----
        random_loss_diffs = []

        with torch.no_grad():
            for _ in range(20):
                noises, scale = random_perturbation_fixed_norm(
                    model, std=std, pruned=pruned
                )

                output_rand = model(data)
                rand_loss = criterion(output_rand, target).mean()
                random_loss_diffs.append(rand_loss.item() - loss.item())

                random_restore_fixed_norm(model, noises, scale)

        avg_random_loss = np.mean(random_loss_diffs)
        batch_random_loss.append(avg_random_loss * data.size(0))
        std_random_loss = np.std(random_loss_diffs)
        batch_random_sgd.append(std_random_loss * data.size(0))


    metrics = {
        "Loss": sum(batch_loss) / n_samples,
        "SAM Loss": sum(batch_sam_loss) / n_samples,
        "Accuracy": sum(batch_corrects) / n_samples,
        "Random Loss": sum(batch_random_loss) / n_samples,
        "Random std": sum(batch_random_sgd) / n_samples,
        }

    if evaluate_flatness:
        # hessian-based metrics
        hessian_metrics = hessian_flatness(
            device=device,
            model=model,
            data_loader=test_loader,
            n_eigs=10,
            pruned=pruned,
            n_batch=n_batch
        )
        metrics.update(hessian_metrics)

    return metrics
