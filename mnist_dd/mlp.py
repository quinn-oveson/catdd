import torch.nn as nn

# CONSTANTS
# d = size of input (number of pixels in one MNIST image)
d = 784
# K = number of classes (number of possible digits)
K = 10

class MLP(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.flatten = nn.Flatten()
        # H is the width of the hidden layer
        self.hidden = nn.Linear(d, H)
        self.relu = nn.ReLU()
        self.output = nn.Linear(H, K)
    
    def forward(self, x):
        x = self.flatten(x)
        return self.output(self.relu(self.hidden(x)))