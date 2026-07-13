import torch.nn as nn
import torch
from config import LR, MOMENTUM, MAX_EPOCHS, DECAY_INTERVAL, GAMMA, EARLY_STOP_CHECK_INTERVAL


def run_epoch(model, X, y_onehot, y_labels, criterion, optimizer, compute_error=False):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y_onehot)
    loss.backward()
    optimizer.step()
    train_error = None
    if compute_error:
        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            train_error = (preds != y_labels).float().mean()
    return loss.detach(), train_error

def train_model(model, X, y_onehot, y_labels, is_underparam):
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=DECAY_INTERVAL, gamma=GAMMA)
    losses = torch.empty(MAX_EPOCHS, device=X.device)
    n_epochs_run = MAX_EPOCHS
    for epoch in range(MAX_EPOCHS):
        check_now = is_underparam and (
            epoch % EARLY_STOP_CHECK_INTERVAL == 0
        )
        loss, train_error = run_epoch(
            model, X, y_onehot, y_labels, criterion, optimizer, compute_error=check_now
        )
        losses[epoch] = loss
        if is_underparam:
            scheduler.step()
            if check_now and train_error.item() == 0:
                n_epochs_run = epoch + 1
                break
    return model, losses[:n_epochs_run].cpu().numpy()

def evaluate(model, X, y_onehot, y_labels):
    with torch.no_grad():
        outputs = model(X)
        preds = outputs.argmax(dim=1)
        zero_one = (preds != y_labels).float().mean()
        mse = ((outputs - y_onehot) ** 2).mean()
    # return zero-one loss and MSE loss
    return zero_one.item(), mse.item()