import torch.nn as nn
import torch
import config

LR = config.LR
MOMENTUM = config.MOMENTUM
MAX_EPOCHS = config.MAX_EPOCHS
DECAY_INTERVAL = config.DECAY_INTERVAL
GAMMA = config.GAMMA

def run_epoch(model, X, y_onehot, criterion, optimizer):
    optimizer.zero_grad()
    loss = criterion(model(X), y_onehot)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_model(model, X, y_onehot, is_underparam):
    stopped = False
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=DECAY_INTERVAL, gamma=GAMMA)
    for epoch in range(MAX_EPOCHS):
        train_error = run_epoch(model, X, y_onehot, criterion, optimizer)
        if is_underparam:
            scheduler.step()
            if train_error == 0:
                stopped = True
                break
    return model, stopped