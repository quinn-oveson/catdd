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