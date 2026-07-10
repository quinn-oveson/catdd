import torch.nn as nn
import torch
from config import STD, D, K

def glorot_init(model):
    nn.init.xavier_uniform_(model.hidden.weight)
    nn.init.zeros_(model.hidden.bias)
    nn.init.xavier_uniform_(model.output.weight)
    nn.init.zeros_(model.output.bias)

def reuse_weights(small_model, big_model, H1):
    with torch.no_grad():
        big_model.hidden.weight[:H1] = small_model.hidden.weight
        big_model.hidden.bias[:H1] = small_model.hidden.bias
        big_model.hidden.weight[H1:].normal_(0, STD**0.5)
        big_model.hidden.bias[H1:].normal_(0, STD**0.5)

        big_model.output.weight[:, :H1] = small_model.output.weight
        big_model.output.weight[:, H1:].normal_(0, STD**0.5)
        big_model.output.bias[:] = small_model.output.bias

def num_params(H):
    return (D + 1) * H + (H + 1) * K