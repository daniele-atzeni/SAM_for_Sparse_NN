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


    # compute trace for non-pruned parameters
    count = 0
    for p in model.parameters():
        if pruned and hasattr(p, "weight_mask"):
            count += p.weight_mask.sum().item()
        else:
            count += p.numel()
    trace_per_param = trace / count
    return_dict ={"trace": trace, "trace_per_param": trace_per_param}
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
    n_batch: int = 1
    ) -> dict:

    model.eval()

    # perturbation-based metrics
    batch_loss = []
    batch_sam_loss = []
    batch_corrects = []
    batch_random_loss = []
    batch_random_sgd = []
    for data, target in test_loader:
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
        "Loss": sum(batch_loss) / len(test_loader.dataset),
        "SAM Loss": sum(batch_sam_loss) / len(test_loader.dataset),
        "Accuracy": sum(batch_corrects) / len(test_loader.dataset),
        "Random Loss": sum(batch_random_loss) / len(test_loader.dataset),
        "Random std": sum(batch_random_sgd) / len(test_loader.dataset),
        }

    # hessian-based metrics
    hessian_metrics = hessian_flatness(
        device=device,
        model=model,
        data_loader=test_loader,
        n_eigs=10,
        pruned=True,
        n_batch=n_batch
    )
    metrics.update(hessian_metrics)

    return metrics