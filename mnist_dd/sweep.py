from mlp import MLP
from utils import glorot_init, reuse_weights, num_params
from data import load_mnist_subset, onehot
from train import train_model, evaluate
from config import H_VALS, SEEDS, N_TRIALS, K, N_TRAIN, REUSE_WEIGHTS_UNDERPARAM, CONFIG_LABEL, REUSE_WEIGHTS_OVERPARAM
import numpy as np
import pandas as pd
import torch
import os

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

train_zeroone = np.zeros((N_TRIALS, len(H_VALS)))
test_zeroone = np.zeros((N_TRIALS, len(H_VALS)))
train_MSE = np.zeros((N_TRIALS, len(H_VALS)))
test_MSE = np.zeros((N_TRIALS, len(H_VALS)))
train_CE = np.zeros((N_TRIALS, len(H_VALS)))
test_CE = np.zeros((N_TRIALS, len(H_VALS)))

for i in range(N_TRIALS):
    seed = SEEDS[i]
    X_train, y_train, X_test, y_test = load_mnist_subset(n = N_TRAIN, seed = seed)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)
    y_train_onehot = onehot(y_train)
    y_test_onehot = onehot(y_test)
    smaller_model = None
    H1 = None
    for j in range(len(H_VALS)):
        H = int(H_VALS[j])
        print(f"Working on H={H}, trial #{i+1}")
        model = MLP(H).to(device)
        is_underparam = num_params(H) < K*N_TRAIN
        if j == 0 or (is_underparam and not REUSE_WEIGHTS_UNDERPARAM) or ((not is_underparam) and not REUSE_WEIGHTS_OVERPARAM):
            glorot_init(model)
        else:
            reuse_weights(smaller_model, model, H_prev)
        train_model(model, X_train, y_train_onehot, y_train, is_underparam)
        train_zeroone[i, j], train_MSE[i, j], train_CE[i, j] = evaluate(model, X_train, y_train_onehot, y_train)
        test_zeroone[i, j], test_MSE[i, j], test_CE[i, j] = evaluate(model, X_test, y_test_onehot, y_test)
        smaller_model = model
        H_prev = H

avg_train_zeroone = np.mean(train_zeroone, axis=0)
avg_test_zeroone = np.mean(test_zeroone, axis=0)
avg_train_MSE = np.mean(train_MSE, axis=0)
avg_test_MSE = np.mean(test_MSE, axis=0)
avg_train_CE = np.mean(train_CE, axis=0)
avg_test_CE = np.mean(test_CE, axis=0)

if __name__ == "__main__":
    print(avg_train_zeroone, avg_test_zeroone, avg_train_MSE, avg_test_MSE, avg_train_CE, avg_test_CE)

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{CONFIG_LABEL}.csv")

    # rows = H values, columns = the 6 averaged error metrics
    results_df = pd.DataFrame(
        {
            "train_zeroone": avg_train_zeroone,
            "test_zeroone": avg_test_zeroone,
            "train_MSE": avg_train_MSE,
            "test_MSE": avg_test_MSE,
            "train_CE": avg_train_CE,
            "test_CE": avg_test_CE,
        },
        index=pd.Index([int(h) for h in H_VALS], name="H"),
    )
    results_df.to_csv(results_path, float_format="%.6f")
    print(f"Results written to {results_path}")