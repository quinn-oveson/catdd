from mlp import MLP
from utils import glorot_init, reuse_weights, num_params
from data import load_mnist_subset, onehot
from train import train_model, evaluate
from config import H_VALS, SEEDS, N_TRIALS, K, N_TRAIN
import numpy as np
import torch 

train_zeroone = np.zeros((N_TRIALS, len(H_VALS)))
test_zeroone = np.zeros((N_TRIALS, len(H_VALS)))
train_MSE = np.zeros((N_TRIALS, len(H_VALS)))
test_MSE = np.zeros((N_TRIALS, len(H_VALS)))

for i in range(N_TRIALS):
    seed = SEEDS[i]
    X_train, y_train, X_test, y_test = load_mnist_subset(n = N_TRAIN, seed = seed)
    y_train_onehot = onehot(y_train)
    y_test_onehot = onehot(y_test)
    smaller_model = None
    H1 = None
    for j in range(len(H_VALS)):
        H = int(H_VALS[j])
        model = MLP(H)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_train, y_train_onehot = X_train.to(device), y_train_onehot.to(device)
        model = model.to(device)
        is_underparam = num_params(H) < K*N_TRAIN
        if j == 0 or not is_underparam:
            glorot_init(model)
        else:
            reuse_weights(smaller_model, model, H_prev)
            assert torch.allclose(model.hidden.weight[:H_prev], smaller_model.hidden.weight)
            print(f"H={H}: new units std ≈ {model.hidden.weight[H_prev:].std().item():.4f}")  # should be ~0.1

        train_model(model, X_train, y_train_onehot, y_train, is_underparam)
        train_zeroone[i, j], train_MSE[i, j] = evaluate(model, X_train, y_train_onehot, y_train)
        test_zeroone[i, j], test_MSE[i, j] = evaluate(model, X_test, y_test_onehot, y_test)
        smaller_model = model
        H_prev = H

avg_train_zeroone = np.mean(train_zeroone, axis=0)
avg_test_zeroone = np.mean(test_zeroone, axis=0)
avg_train_MSE = np.mean(train_MSE, axis=0)
avg_test_MSE = np.mean(test_MSE, axis=0)

if __name__ == "__main__":
    print(avg_train_zeroone, avg_test_zeroone, avg_train_MSE, avg_test_MSE)