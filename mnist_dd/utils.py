import copy

import torch.nn as nn
import torch
from config import VAR, D, K

def glorot_init(model):
    nn.init.xavier_uniform_(model.hidden.weight)
    nn.init.zeros_(model.hidden.bias)
    nn.init.xavier_uniform_(model.output.weight)
    nn.init.zeros_(model.output.bias)

def _fill_normal(view, std):
    """Fill a slice of a parameter with N(0, std**2).

    Assignment from randn_like rather than view.normal_(0, std): an in-place
    normal_ on a SLICE of an MPS tensor is silently a no-op (the slice keeps
    whatever nn.Linear's default init put there -- U(-1/sqrt(fan_in),
    1/sqrt(fan_in)), a visibly different scale), so on a laptop the "new"
    hidden units were never actually getting VAR-scaled weights. Slice
    assignment goes through copy_ and behaves the same on cpu/mps/cuda."""
    view[:] = torch.randn_like(view) * std

def reuse_weights(small_model, big_model, H1):
    std = VAR**0.5
    with torch.no_grad():
        big_model.hidden.weight[:H1] = small_model.hidden.weight
        big_model.hidden.bias[:H1] = small_model.hidden.bias
        _fill_normal(big_model.hidden.weight[H1:], std)
        _fill_normal(big_model.hidden.bias[H1:], std)

        big_model.output.weight[:, :H1] = small_model.output.weight
        _fill_normal(big_model.output.weight[:, H1:], std)
        big_model.output.bias[:] = small_model.output.bias

def glorot_target_norm(weight):
    """Expected Frobenius norm of a fresh xavier_uniform_ draw for this shape."""
    fan_out, fan_in = weight.shape
    return (weight.numel() * 2.0 / (fan_in + fan_out)) ** 0.5

def normalize_to_glorot_norm(model):
    """Rescale a reused model to the norm a fresh Glorot init would have, so it
    keeps the DIRECTIONS it inherited but starts at the same scale as a fresh
    one. Leaves the initial classifier exactly intact -- only its scale changes.

    Per-layer rather than one global scalar: a trained-then-reused model has
    drifted from Glorot's hidden/output norm ratio (measured: up to +24% along
    the chain), so a single factor would fix the total norm while leaving the
    per-layer norms 18-37% off the no-reuse run -- a second difference in an
    experiment meant to isolate one.

    Biases are SCALED, not zeroed. ReLU is positively homogeneous, so scaling
    hidden by a1 (weight and bias) and output's weight by a2 with its bias by
    a1*a2 maps f -> a1*a2*f: a positive scaling of the logits, leaving every
    argmax untouched. Zeroing the biases instead would match glorot_init's own
    zero-bias convention and make the norm match exact rather than +0.02%, but
    it is NOT a pure rescaling -- measured on a trained chain step it flips
    11% of the initial predictions, discarding exactly the directional
    information this preset exists to preserve. The biases are only ~0.05% of
    the squared norm, so keeping them costs +0.02% on init_weight_norm, far
    below the seed-to-seed scatter."""
    with torch.no_grad():
        a_hidden = glorot_target_norm(model.hidden.weight) / model.hidden.weight.norm()
        a_output = glorot_target_norm(model.output.weight) / model.output.weight.norm()
        model.hidden.weight.mul_(a_hidden)
        model.hidden.bias.mul_(a_hidden)
        model.output.weight.mul_(a_output)
        model.output.bias.mul_(a_hidden * a_output)

def weight_norms(model):
    """L2 norms of the model's parameters as they are right now -- call before
    training for the initialization's norm, after for the trained model's.

    "weight_norm" is over every parameter (both layers' weights AND biases),
    i.e. the norm of the full parameter vector; the per-layer entries are the
    Frobenius norms of the two weight matrices alone, to show which layer any
    growth is happening in. Not normalized by parameter count -- divide by
    num_params(H)**0.5 downstream if you want a per-parameter scale."""
    total_sq = sum(p.detach().pow(2).sum() for p in model.parameters())
    return {
        "weight_norm": total_sq.sqrt().item(),
        "hidden_weight_norm": model.hidden.weight.detach().norm().item(),
        "output_weight_norm": model.output.weight.detach().norm().item(),
    }

def num_params(H):
    return (D + 1) * H + (H + 1) * K

def balance_mlp(model, eps=1e-12):
    """
    Rescale each hidden unit of a 1-hidden-layer ReLU MLP to its minimum-norm,
    function-preserving representative (incoming norm == outgoing norm).
    """
    with torch.no_grad():
        W_1 = model.hidden.weight
        b_1 = model.hidden.bias
        W_2 = model.output.weight
        p = torch.sqrt((W_1**2).sum(dim=1) + b_1**2)
        q = W_2.norm(dim=0)
        alive = (p > eps) & (q > eps)
        c = torch.ones_like(p)
        c[alive] = torch.sqrt(q[alive] / p[alive])

        W_1.mul_(c.unsqueeze(1))
        b_1.mul_(c)
        W_2.mul_((1.0 / c).unsqueeze(0))

def min_achievable_norm(model):
    """
    Smallest total weight norm any function-preserving rescaling can reach,
    attained by balance_mlp. Minimises per hidden unit, so it sits at or below
    the single-factor orbit's floor of sqrt(2*sqrt(A*B) + C).
    """
    with torch.no_grad():
        p = torch.sqrt((model.hidden.weight**2).sum(dim=1) + model.hidden.bias**2)
        q = model.output.weight.norm(dim=0)
        return torch.sqrt(2 * (p * q).sum() + (model.output.bias**2).sum()).item()

def rescale_to_norm(model, target):
    """
    Scale a 1-hidden-layer ReLU MLP to a total weight norm of `target` without
    changing the function it computes. Of the two solutions, takes the
    hidden-heavy one. Raises if `target` is below min_achievable_norm(model).
    """
    balance_mlp(model)   # canonical gauge, so the result depends only on the function
    with torch.no_grad():
        A = (model.hidden.weight**2).sum() + (model.hidden.bias**2).sum()
        B = (model.output.weight**2).sum()
        C = (model.output.bias**2).sum()

        disc = (target**2 - C)**2 - 4 * A * B
        if disc < 0:
            raise ValueError(f"target norm {target:.4f} is below the minimum "
                             f"achievable {min_achievable_norm(model):.4f}")
        t = torch.sqrt(((target**2 - C) + torch.sqrt(disc)) / (2 * A))

        model.hidden.weight.mul_(t)
        model.hidden.bias.mul_(t)
        model.output.weight.mul_(1.0 / t)

def check_function_preserved(model, transform=balance_mlp, n=64, rtol=1e-4, verbose=True):
    device = next(model.parameters()).device
    probe = copy.deepcopy(model)
    x = torch.randn(n, model.hidden.in_features, device=device)

    with torch.no_grad():
        before = model(x)
        transform(probe)
        after = probe(x)

    max_diff = (before - after).abs().max().item()
    scale = before.abs().max().item()
    rel_diff = max_diff / scale if scale > 0 else max_diff
    if verbose:
        print(f"max output difference after {transform.__name__}: "
              f"{max_diff:.2e} absolute, {rel_diff:.2e} relative")
    assert rel_diff <= rtol, (
        f"{transform.__name__} changed the function: {rel_diff:.2e} relative "
        f"> rtol={rtol:.0e} -- check tensor axes!")
    return rel_diff