import torch.nn as nn
import torch
import config
import numpy as np

LR = config.LR
MOMENTUM = config.MOMENTUM
MAX_EPOCHS = config.MAX_EPOCHS
DECAY_INTERVAL = config.DECAY_INTERVAL
GAMMA = config.GAMMA

def run_epoch(model, X, y_onehot, criterion, optimizer, batch_size = None):
    optimizer.zero_grad()
    loss = criterion(model(X), y_onehot)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_model(model, X, y_onehot, is_underparam):
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=DECAY_INTERVAL, gamma=GAMMA)
    losses = []
    for epoch in range(MAX_EPOCHS):
        train_error = run_epoch(model, X, y_onehot, criterion, optimizer)
        losses.append(train_error)
        if is_underparam:
            scheduler.step()
            if train_error == 0:
                break
    return model, losses

def evaluate(model, X, y_onehot, y_labels):
    outputs = model(X)
    preds = model(X).argmax(dim=1)
    # return zero-one loss and MSE loss
    return (preds != y_labels).float().mean().item(), np.mean((outputs - y_onehot).detach().numpy()**2)