import torch
from torch import nn
import numpy as np

from src.pyhessian import hessian


def _get_mask(module, name):
    """Return the pruning mask for a named parameter, or None if not pruned.

    Handles both original ('weight') and post-prune ('weight_orig') names,
    since torch.nn.utils.prune renames the parameter to weight_orig while
    storing the mask as the weight_mask buffer on the module.
    """
    base = name.replace('_orig', '')
    mask_name = base.replace('weight', 'weight_mask').replace('bias', 'bias_mask')
    return getattr(module, mask_name, None)


def _build_param_masks(model, device):
    """Masks aligned with model.parameters() order; ones where not pruned."""
    masks = []
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            mask = _get_mask(module, name)
            masks.append(mask.to(device) if mask is not None else torch.ones_like(param, device=device))
    return masks


@torch.no_grad()
def masked_grad_norm(model):
    norms = []
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if param.grad is None:
                continue
            mask = _get_mask(module, name)
            g = param.grad * mask if mask is not None else param.grad
            norms.append(g.norm(2))
    return torch.norm(torch.stack(norms), 2) if norms else torch.tensor(0.0)


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


@torch.no_grad()
def weight_distribution_metrics(model) -> dict:
    """L1/L2 ratio, Gini coefficient, pruned weight norm (Propositions 3 & 6).

    Only considers weight tensors (not biases or BN affine params) so that the
    metrics reflect the pruned coordinate structure.
    """
    l1, l2_sq, pruned_sq, count = 0.0, 0.0, 0.0, 0
    active_abs = []  # collected for Gini (requires sorted flat array)

    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if 'weight' not in name:
                continue
            mask = _get_mask(module, name)
            w = param.detach()
            if mask is not None:
                mb = mask.bool()
                w_active = w[mb]
                pruned_sq += w[~mb].pow(2).sum().item()
            else:
                w_active = w.flatten()
            l1 += w_active.abs().sum().item()
            l2_sq += w_active.pow(2).sum().item()
            count += w_active.numel()
            active_abs.append(w_active.abs().cpu())

    l1_l2_ratio = l1 / (l2_sq ** 0.5 + 1e-12)

    if count > 0 and l1 > 0:
        w_np = np.sort(torch.cat(active_abs).numpy())
        gini = float(2 * np.dot(np.arange(1, count + 1), w_np) / (count * w_np.sum()) - (count + 1) / count)
    else:
        gini = 0.0

    return {
        "l1_l2_ratio": l1_l2_ratio,
        "gini": gini,
        "pruned_weight_norm": pruned_sq ** 0.5,
        "active_weight_count": count,
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
    criterion(model(data), target).mean().backward()
    grad_norm = masked_grad_norm(model).item()
    model.zero_grad()

    return {"masked_grad_norm": grad_norm, **weight_distribution_metrics(model)}


def eigenvector_alignment(model, device, data_loader, n_batch: int = 1) -> float:
    """||P_m^T v1||^2: mass of the full Hessian top eigenvector on active coords.

    Computes the top eigenvector of the *full* (unmasked) Hessian and measures
    what fraction of its squared norm lies on the active (unpruned) coordinates.
    Close to 1 means the dominant curvature direction is preserved by the mask.
    """
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    model.zero_grad()

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
    return float(sum((v1_i * m).pow(2).sum().item() for v1_i, m in zip(v1, masks)))


def hessian_flatness(
        device: str | torch.device,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        n_eigs: int,
        pruned: bool = False,
        n_batch: int = 1,
        ) -> dict:

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

    count_total = 0
    count_weights = 0
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            mask = _get_mask(module, name) if pruned else None
            active = int(mask.sum().item()) if mask is not None else param.numel()
            count_total += active
            if 'weight' in name and 'bias' not in name:
                count_weights += active

    trace_per_param = trace / (count_total + 1e-8)
    trace_per_active_weight = trace / (count_weights + 1e-8)  # tr(H_m)/k, Propositions 1 & 2

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

        output = model(data)
        loss = criterion(output, target).mean()
        loss.backward()

        scale = sam_perturbation(model, rho, pruned)

        with torch.no_grad():
            output_perturbed = model(data)
            sam_loss = criterion(output_perturbed, target).mean()

        sam_restore(model, scale, pruned)
        model.zero_grad()

        n_samples += data.size(0)
        batch_loss.append(loss.item() * data.size(0))
        batch_sam_loss.append((sam_loss.item() - loss.item()) * data.size(0))
        batch_corrects.append(
            output.argmax(dim=1).eq(target).sum().item()
        )

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
