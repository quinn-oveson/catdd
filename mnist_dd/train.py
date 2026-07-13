import torch.nn as nn
import torch
import config
import numpy as np
from config import LR, MOMENTUM, MAX_EPOCHS, DECAY_INTERVAL, GAMMA


def run_epoch(model, X, y_onehot, y_labels, criterion, optimizer, batch_size = None):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y_onehot)
    loss.backward()
    optimizer.step()
    preds = outputs.argmax(dim=1)
    train_error = (preds != y_labels).float().mean().item()
    return loss.item(), train_error

def train_model(model, X, y_onehot, y_labels, is_underparam):
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=DECAY_INTERVAL, gamma=GAMMA)
    losses = []
    for epoch in range(MAX_EPOCHS):
        mse_loss, train_error = run_epoch(model, X, y_onehot, y_labels, criterion, optimizer)
        losses.append(mse_loss)
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