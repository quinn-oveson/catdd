import torch.nn as nn
import torch
from config import LR, MOMENTUM, MAX_EPOCHS, DECAY_INTERVAL, GAMMA, EARLY_STOP_CHECK_INTERVAL, LOSS_FUNC, BATCH_SIZE
from sweep_config import DECAY_UNDERPARAM, DECAY_OVERPARAM, STOP_UNDERPARAM, STOP_OVERPARAM


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

def train_model(model, X, y_onehot, y_labels, is_underparam, lr=LR, batch_size=BATCH_SIZE, loss_func=LOSS_FUNC):
    criterion = loss_func
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=DECAY_INTERVAL, gamma=GAMMA)
    losses = torch.empty(MAX_EPOCHS, device=X.device)
    n_epochs_run = MAX_EPOCHS

    n_train = X.shape[0]
    batch_size = batch_size if batch_size is not None else n_train

    stop_enabled = STOP_UNDERPARAM if is_underparam else STOP_OVERPARAM
    decay_enabled = DECAY_UNDERPARAM if is_underparam else DECAY_OVERPARAM

    for epoch in range(MAX_EPOCHS):
        check_now = stop_enabled and (epoch % EARLY_STOP_CHECK_INTERVAL == 0)
        perm = torch.randperm(n_train, device=X.device)
        epoch_loss = torch.zeros((), device=X.device)
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            loss, _ = run_epoch(
                model, X[idx], y_onehot[idx], y_labels[idx], criterion, optimizer, compute_error=False
            )
            epoch_loss += loss * idx.shape[0]
        losses[epoch] = epoch_loss / n_train

        if decay_enabled:
            scheduler.step()

        if check_now:
            with torch.no_grad():
                preds = model(X).argmax(dim=1)
                train_error = (preds != y_labels).float().mean()
            if train_error.item() == 0:
                n_epochs_run = epoch + 1
                break
    return model, losses[:n_epochs_run].cpu().numpy()

def evaluate(model, X, y_onehot, y_labels):
    with torch.no_grad():
        outputs = model(X)
        preds = outputs.argmax(dim=1)
        zero_one = (preds != y_labels).float().mean()
        mse = ((outputs - y_onehot) ** 2).mean()
        ce = nn.functional.cross_entropy(outputs, y_onehot)
    # return zero-one loss, MSE loss, and cross-entropy loss
    return zero_one.item(), mse.item(), ce.item()