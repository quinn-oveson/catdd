import torch.nn as nn
import config

D = config.D
K = config.K

class MLP(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.flatten = nn.Flatten()
        # H is the width of the hidden layer
        self.hidden = nn.Linear(D, H)
        self.relu = nn.ReLU()
        self.output = nn.Linear(H, K)
    
    def forward(self, x):
        x = self.flatten(x)
        return self.output(self.relu(self.hidden(x)))