"""
Jacobian-based spectral diagnostics for the three-factor decomposition
(Proposition 3.2) and stochastic estimates for CNNs.

Computes:
  - σ²_max(J)  via power iteration on J^T J
  - s̄_J = tr(J^T J) / p  via Hutchinson estimation
  - Rs, Rδ, η  (the full three-factor decomposition)
  - Stochastic Jacobian spectral estimates for CNNs (where full J is infeasible)
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Helper: flatten model parameters / masks into single vectors
# ---------------------------------------------------------------------------

def _flatten_params(model: nn.Module) -> torch.Tensor:
    """Flatten all requires_grad parameters into a single vector."""
    return torch.cat([p.detach().flatten() for p in model.parameters() if p.requires_grad])


def _get_pruning_mask_flat(model: nn.Module) -> torch.Tensor:
    """Return a flat binary mask aligned with model.parameters() order.
    Ones where not pruned (or no pruning applied)."""
    masks = []
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            base = name.replace('_orig', '')
            mask_name = base.replace('weight', 'weight_mask').replace('bias', 'bias_mask')
            mask = getattr(module, mask_name, None)
            if mask is not None:
                masks.append(mask.detach().flatten())
            else:
                masks.append(torch.ones(param.numel(), device=param.device))
    return torch.cat(masks)


# ---------------------------------------------------------------------------
# Full Jacobian computation (MLPs only — O(nC * p) memory)
# ---------------------------------------------------------------------------

def compute_full_jacobian(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: Optional[int] = None,
) -> torch.Tensor:
    """Compute the full output Jacobian J = ∂f/∂θ ∈ R^{nC × p}.

    Only feasible for small models (MLPs). Returns J on CPU.
    """
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    p_total = sum(p.numel() for p in params)

    rows = []
    n_collected = 0
    for data, _ in data_loader:
        data = data.to(device)
        output = model(data)  # (batch, C)
        batch_size, C = output.shape

        for i in range(batch_size):
            for c in range(C):
                model.zero_grad()
                output_ic = model(data[i:i+1])  # re-forward single sample
                output_ic[0, c].backward(retain_graph=(c < C - 1))
                row = torch.cat([p.grad.detach().flatten() for p in params])
                rows.append(row.cpu())
                model.zero_grad()

            n_collected += 1
            if max_samples is not None and n_collected >= max_samples:
                break
        if max_samples is not None and n_collected >= max_samples:
            break

    J = torch.stack(rows)  # (nC, p)
    return J


def exact_spectral_diagnostics(J: torch.Tensor) -> dict:
    """From a full Jacobian, compute σ²_max, s̄_J, and singular values."""
    p = J.shape[1]
    # SVD (or just singular values)
    S = torch.linalg.svdvals(J.float())
    sigma2_max = (S[0] ** 2).item()
    sbar_J = (S ** 2).sum().item() / p  # tr(J^T J) / p
    return {
        "sigma2_max": sigma2_max,
        "sbar_J": sbar_J,
        "singular_values": S.numpy(),
    }


# ---------------------------------------------------------------------------
# Stochastic Jacobian spectral estimates (works for CNNs)
# ---------------------------------------------------------------------------

def _jvp(model, data, target, params, v_flat, criterion=None):
    """Compute J @ v via forward-mode AD (finite differences fallback).

    Uses the identity: J v = d/dt f(θ + t*v)|_{t=0}
    Approximated with central finite differences for robustness.
    """
    eps = 1e-4
    # Save original params
    orig = [p.data.clone() for p in params]

    # Perturb forward
    offset = 0
    for p in params:
        n = p.numel()
        p.data.add_(v_flat[offset:offset+n].view_as(p), alpha=eps)
        offset += n
    with torch.no_grad():
        f_plus = model(data).detach()

    # Perturb backward
    offset = 0
    for p, o in zip(params, orig):
        p.data.copy_(o)
        n = p.numel()
        p.data.add_(v_flat[offset:offset+n].view_as(p), alpha=-eps)
        offset += n
    with torch.no_grad():
        f_minus = model(data).detach()

    # Restore
    for p, o in zip(params, orig):
        p.data.copy_(o)

    Jv = (f_plus - f_minus) / (2 * eps)
    return Jv.flatten()


def _jtv(model, data, params, u_flat, n_outputs):
    """Compute J^T @ u via backward pass.

    u_flat is a vector in output space (nC,).
    """
    model.zero_grad()
    output = model(data)
    # Reshape u to match output
    u = u_flat.view_as(output)
    output.backward(gradient=u)
    Jtu = torch.cat([p.grad.detach().flatten() for p in params])
    model.zero_grad()
    return Jtu


def stochastic_sigma2_max(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_iters: int = 50,
    n_batches: int = 1,
) -> float:
    """Estimate σ²_max(J) via power iteration on J^T J.

    Uses stochastic Jacobian-vector products — works for any model size.
    """
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    p_total = sum(p.numel() for p in params)

    # Collect data batches
    batches = []
    for i, (data, target) in enumerate(data_loader):
        batches.append((data.to(device), target.to(device)))
        if i + 1 >= n_batches:
            break

    # Random initial vector
    v = torch.randn(p_total, device=device)
    v = v / v.norm()

    eigenvalue = None
    for it in range(n_iters):
        # Compute J^T J v = J^T (J v) over batches
        JtJv = torch.zeros(p_total, device=device)
        for data, target in batches:
            n = data.shape[0]
            C = model(data[:1]).shape[1]
            Jv = _jvp(model, data, target, params, v)
            JtJv_batch = _jtv(model, data, params, Jv, n * C)
            JtJv += JtJv_batch

        new_eigenvalue = v.dot(JtJv).item()
        v = JtJv / (JtJv.norm() + 1e-12)

        if eigenvalue is not None and abs(new_eigenvalue - eigenvalue) / (abs(eigenvalue) + 1e-6) < 1e-4:
            eigenvalue = new_eigenvalue
            break
        eigenvalue = new_eigenvalue

    return eigenvalue


def stochastic_trace_JtJ(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_samples: int = 20,
    n_batches: int = 1,
) -> float:
    """Estimate tr(J^T J) via Hutchinson: E[z^T J^T J z] = tr(J^T J).

    Uses Rademacher random vectors.
    """
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    p_total = sum(p.numel() for p in params)

    batches = []
    for i, (data, target) in enumerate(data_loader):
        batches.append((data.to(device), target.to(device)))
        if i + 1 >= n_batches:
            break

    estimates = []
    for _ in range(n_samples):
        z = torch.randint(0, 2, (p_total,), device=device).float() * 2 - 1

        # Compute ||Jz||^2 = z^T J^T J z
        norm_sq = 0.0
        for data, target in batches:
            Jz = _jvp(model, data, target, params, z)
            norm_sq += Jz.dot(Jz).item()

        estimates.append(norm_sq)

    return np.mean(estimates)


def stochastic_spectral_diagnostics(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_power_iters: int = 50,
    n_hutchinson: int = 20,
    n_batches: int = 1,
) -> dict:
    """Stochastic estimates of σ²_max and s̄_J for large models."""
    params = [p for p in model.parameters() if p.requires_grad]
    p_total = sum(p.numel() for p in params)

    sigma2_max = stochastic_sigma2_max(model, data_loader, device,
                                        n_iters=n_power_iters, n_batches=n_batches)
    tr_JtJ = stochastic_trace_JtJ(model, data_loader, device,
                                    n_samples=n_hutchinson, n_batches=n_batches)
    sbar_J = tr_JtJ / p_total

    return {
        "sigma2_max": sigma2_max,
        "sbar_J": sbar_J,
        "tr_JtJ": tr_JtJ,
        "p": p_total,
    }


# ---------------------------------------------------------------------------
# Three-factor decomposition
# ---------------------------------------------------------------------------

def compute_pruning_perturbation(
    model: nn.Module,
    dense_state_dict: dict,
    device: torch.device,
) -> torch.Tensor:
    """Compute δ = θ^(m) - θ* as a flat vector.

    After PyTorch pruning, weight_orig retains original values and the
    effective weight is weight_orig * weight_mask.  So:
        δ = (θ* ⊙ m) - θ* = -(1 - m) ⊙ θ*
    """
    delta_parts = []
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            # Get the effective (masked) parameter value
            base = name.replace('_orig', '')
            mask_name = base.replace('weight', 'weight_mask').replace('bias', 'bias_mask')
            mask = getattr(module, mask_name, None)

            if mask is not None:
                effective_param = param.detach() * mask.to(device)
            else:
                effective_param = param.detach()

            # Get the corresponding dense parameter
            # Try multiple name formats
            dense_name = base
            # Build full name by finding the module prefix
            full_prefix = ""
            for mod_name, mod in model.named_modules():
                if mod is module:
                    full_prefix = mod_name + "." if mod_name else ""
                    break

            dense_key = full_prefix + dense_name
            if dense_key in dense_state_dict:
                dense_param = dense_state_dict[dense_key].to(device)
            elif full_prefix + name in dense_state_dict:
                dense_param = dense_state_dict[full_prefix + name].to(device)
            else:
                dense_param = torch.zeros_like(param)

            delta = effective_param - dense_param
            delta_parts.append(delta.flatten())

    return torch.cat(delta_parts)


def three_factor_decomposition(
    sbar_J_sam: float,
    sbar_J_sgd: float,
    delta_sam: torch.Tensor,
    delta_sgd: torch.Tensor,
    J_sam: Optional[torch.Tensor] = None,
    J_sgd: Optional[torch.Tensor] = None,
) -> dict:
    """Compute the three-factor decomposition (Proposition 3.2).

    Args:
        sbar_J_sam, sbar_J_sgd: mean per-parameter output sensitivity
        delta_sam, delta_sgd: pruning perturbation vectors (flat)
        J_sam, J_sgd: full Jacobians (optional, for exact η computation)

    Returns:
        dict with Rs, Rdelta, eta_sam, eta_sgd, eta_ratio, predicted, actual
    """
    Rs = sbar_J_sam / (sbar_J_sgd + 1e-12)
    delta_sgd_sq = (delta_sgd.norm() ** 2).item()
    delta_sam_sq = (delta_sam.norm() ** 2).item()
    Rdelta = delta_sam_sq / (delta_sgd_sq + 1e-12)

    result = {
        "Rs": Rs,
        "Rdelta": Rdelta,
        "Rs_Rdelta": Rs * Rdelta,
    }

    if J_sam is not None and J_sgd is not None:
        # Exact computation of ||Jδ||² and η
        Jd_sam = (J_sam.to(delta_sam.device) @ delta_sam).norm() ** 2
        Jd_sgd = (J_sgd.to(delta_sgd.device) @ delta_sgd).norm() ** 2

        eta_sam = Jd_sam.item() / (sbar_J_sam * (delta_sam.norm() ** 2).item() + 1e-12)
        eta_sgd = Jd_sgd.item() / (sbar_J_sgd * (delta_sgd.norm() ** 2).item() + 1e-12)

        actual_ratio = Jd_sam.item() / (Jd_sgd.item() + 1e-12)

        result.update({
            "eta_sam": eta_sam,
            "eta_sgd": eta_sgd,
            "eta_ratio": eta_sam / (eta_sgd + 1e-12),
            "actual_ratio": actual_ratio,
            "Jdelta_sam_sq": Jd_sam.item(),
            "Jdelta_sgd_sq": Jd_sgd.item(),
        })

    return result


def compute_delta_from_pruning(
    model: nn.Module,
    sparsity: float,
    device: torch.device,
) -> tuple[torch.Tensor, nn.Module]:
    """Apply magnitude pruning at given sparsity and return the perturbation δ.

    Returns delta_flat. The model is modified in-place with pruning hooks.
    """
    # Save dense state dict before pruning (clean parameter names)
    dense_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    # Apply pruning
    parameters_to_prune = [
        (module, 'weight')
        for _, module in model.named_modules()
        if isinstance(module, (nn.Linear, nn.Conv2d))
    ]
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )

    # Compute δ = θ^(m) - θ* = -(1-m) ⊙ θ*
    delta = compute_pruning_perturbation(model, dense_state_dict, device)

    return delta


# ---------------------------------------------------------------------------
# Full decomposition pipeline for a pair of models
# ---------------------------------------------------------------------------

def full_decomposition_mlp(
    model_sam: nn.Module,
    model_sgd: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    sparsities: list[float],
    max_samples: Optional[int] = None,
) -> list[dict]:
    """Run the full three-factor decomposition for MLPs at multiple sparsities.

    This computes exact Jacobians and all decomposition terms including η.
    Models should be dense (unpruned) when passed in.

    Returns a list of dicts, one per sparsity level.
    """
    import copy

    # Compute full Jacobians for both models
    print("Computing full Jacobian for SAM model...")
    J_sam = compute_full_jacobian(model_sam, data_loader, device, max_samples=max_samples)
    print(f"  J_sam shape: {J_sam.shape}")

    print("Computing full Jacobian for SGD model...")
    J_sgd = compute_full_jacobian(model_sgd, data_loader, device, max_samples=max_samples)
    print(f"  J_sgd shape: {J_sgd.shape}")

    # Spectral diagnostics
    spec_sam = exact_spectral_diagnostics(J_sam)
    spec_sgd = exact_spectral_diagnostics(J_sgd)

    Rs = spec_sam["sbar_J"] / spec_sgd["sbar_J"]
    print(f"Rs = {Rs:.4f}")
    print(f"  SAM: σ²_max={spec_sam['sigma2_max']:.2f}, s̄_J={spec_sam['sbar_J']:.4f}")
    print(f"  SGD: σ²_max={spec_sgd['sigma2_max']:.2f}, s̄_J={spec_sgd['sbar_J']:.4f}")

    results = []
    for s in sparsities:
        print(f"\nSparsity {s}:")
        # Deep copy models for each sparsity
        m_sam = copy.deepcopy(model_sam).to(device)
        m_sgd = copy.deepcopy(model_sgd).to(device)

        # Compute pruning perturbations
        delta_sam = compute_delta_from_pruning(m_sam, s, device)
        delta_sgd = compute_delta_from_pruning(m_sgd, s, device)

        # Three-factor decomposition with exact η
        decomp = three_factor_decomposition(
            sbar_J_sam=spec_sam["sbar_J"],
            sbar_J_sgd=spec_sgd["sbar_J"],
            delta_sam=delta_sam.cpu(),
            delta_sgd=delta_sgd.cpu(),
            J_sam=J_sam,
            J_sgd=J_sgd,
        )
        decomp["sparsity"] = s
        decomp["sigma2_max_sam"] = spec_sam["sigma2_max"]
        decomp["sigma2_max_sgd"] = spec_sgd["sigma2_max"]
        decomp["sbar_J_sam"] = spec_sam["sbar_J"]
        decomp["sbar_J_sgd"] = spec_sgd["sbar_J"]

        print(f"  Rs={decomp['Rs']:.4f}, Rδ={decomp['Rdelta']:.4f}, "
              f"η_SAM={decomp.get('eta_sam', 'N/A'):.4f}, "
              f"η_SGD={decomp.get('eta_sgd', 'N/A'):.4f}")
        print(f"  Rs·Rδ={decomp['Rs_Rdelta']:.4f}, "
              f"Rs·Rδ·(η_SAM/η_SGD)={decomp['Rs_Rdelta'] * decomp.get('eta_ratio', 1):.4f}, "
              f"Actual={decomp.get('actual_ratio', 'N/A'):.4f}")

        # Cleanup pruned models
        del m_sam, m_sgd
        results.append(decomp)

    return results


def full_decomposition_cnn(
    model_sam: nn.Module,
    model_sgd: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    sparsities: list[float],
    n_power_iters: int = 50,
    n_hutchinson: int = 20,
    n_batches: int = 3,
) -> list[dict]:
    """Run the three-factor decomposition for CNNs using stochastic estimates.

    Computes Rs and Rδ via stochastic Jacobian spectral estimates.
    η is not computed (would require full J), but Rs and Rδ are reported.
    """
    import copy

    print("Computing stochastic spectral diagnostics for SAM model...")
    spec_sam = stochastic_spectral_diagnostics(
        model_sam, data_loader, device,
        n_power_iters=n_power_iters, n_hutchinson=n_hutchinson, n_batches=n_batches
    )
    print(f"  SAM: σ²_max≈{spec_sam['sigma2_max']:.2f}, s̄_J≈{spec_sam['sbar_J']:.6f}")

    print("Computing stochastic spectral diagnostics for SGD model...")
    spec_sgd = stochastic_spectral_diagnostics(
        model_sgd, data_loader, device,
        n_power_iters=n_power_iters, n_hutchinson=n_hutchinson, n_batches=n_batches
    )
    print(f"  SGD: σ²_max≈{spec_sgd['sigma2_max']:.2f}, s̄_J≈{spec_sgd['sbar_J']:.6f}")

    Rs = spec_sam["sbar_J"] / spec_sgd["sbar_J"]
    print(f"Rs ≈ {Rs:.4f}")

    results = []
    for s in sparsities:
        print(f"\nSparsity {s}:")
        m_sam = copy.deepcopy(model_sam).to(device)
        m_sgd = copy.deepcopy(model_sgd).to(device)

        delta_sam = compute_delta_from_pruning(m_sam, s, device)
        delta_sgd = compute_delta_from_pruning(m_sgd, s, device)

        Rdelta = (delta_sam.norm() ** 2).item() / (delta_sgd.norm() ** 2).item()

        decomp = {
            "sparsity": s,
            "Rs": Rs,
            "Rdelta": Rdelta,
            "Rs_Rdelta": Rs * Rdelta,
            "sigma2_max_sam": spec_sam["sigma2_max"],
            "sigma2_max_sgd": spec_sgd["sigma2_max"],
            "sbar_J_sam": spec_sam["sbar_J"],
            "sbar_J_sgd": spec_sgd["sbar_J"],
            "delta_sam_norm2": (delta_sam.norm() ** 2).item(),
            "delta_sgd_norm2": (delta_sgd.norm() ** 2).item(),
        }
        print(f"  Rs={Rs:.4f}, Rδ={Rdelta:.4f}, Rs·Rδ={Rs*Rdelta:.4f}")

        del m_sam, m_sgd
        results.append(decomp)

    return results