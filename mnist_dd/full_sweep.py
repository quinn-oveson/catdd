"""Stage 2 full sweep: run the full Belkin H_VALS curve for the (lr, batch_size)
candidates identified by the Stage 1 probe (see probe.py / plot_probe.py /
results/probe_summary.csv).

Same task_id / decode_task_id pattern as probe.py: one task = one (lr,
batch_size, seed) cell, trained sequentially over every H in config.H_VALS
(the digitized values from belkin_figure3.png -- see extract_belkin_markers.py
-- not just the small-H probe subset). Meant to run identically two ways:
  - locally, looped over all task_ids (see run_full_sweep_local.py)
  - as one SLURM array task, with --task_id set from $SLURM_ARRAY_TASK_ID
    (see slurm/full_sweep_array.sbatch)

Sanity-check the whole mapping for free before spending any compute:
    for i in 0 1 2 ... ; do python full_sweep.py --task_id $i --dry_run; done
"""
import argparse
import os

import pandas as pd
import torch

from config import N_TRAIN, K, REUSE_WEIGHTS_UNDERPARAM, REUSE_WEIGHTS_OVERPARAM, H_VALS
from data import load_mnist_subset, onehot
from mlp import MLP
from train import train_model, evaluate
from utils import glorot_init, reuse_weights, num_params

# Candidates carried over from the Stage 1 probe (see results/probe_summary.csv /
# results/probe_vs_belkin.png) -- both at batch_size=32, the two lr's that best
# matched Belkin's small-H targets.
LR_GRID = [0.001, 0.0005]
BATCH_SIZE_GRID = [32]
SEEDS = list(range(5))  # Belkin averages over 5 trials

N_LR = len(LR_GRID)
N_BS = len(BATCH_SIZE_GRID)
N_SEEDS = len(SEEDS)
TOTAL_TASKS = N_LR * N_BS * N_SEEDS

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "full_sweep")


def decode_task_id(task_id):
    if not (0 <= task_id < TOTAL_TASKS):
        raise ValueError(f"task_id must be in [0, {TOTAL_TASKS}), got {task_id}")
    seed_idx = task_id % N_SEEDS
    bs_idx = (task_id // N_SEEDS) % N_BS
    lr_idx = task_id // (N_SEEDS * N_BS)
    return LR_GRID[lr_idx], BATCH_SIZE_GRID[bs_idx], SEEDS[seed_idx]


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def run_full_task(task_id, output_dir=RESULTS_DIR, full_batch=False):
    lr, batch_size, seed = decode_task_id(task_id)
    if full_batch:
        # For fast local sanity checks only -- batch_size=32 means 4000/32=125
        # SGD steps/epoch, which is painfully slow on a laptop CPU/MPS. The
        # cluster run (slurm/full_sweep_array.sbatch) never passes --full_batch,
        # so it always uses the real batch_size=32 candidate from the grid.
        batch_size = N_TRAIN
    device = get_device()

    X_train, y_train, X_test, y_test = load_mnist_subset(n=N_TRAIN, seed=seed)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)
    y_train_onehot = onehot(y_train)
    y_test_onehot = onehot(y_test)

    rows = []
    smaller_model = None
    H_prev = None
    for j, H in enumerate(H_VALS):
        H = int(H)
        model = MLP(H).to(device)
        is_underparam = num_params(H) < K * N_TRAIN
        if j == 0 or (is_underparam and not REUSE_WEIGHTS_UNDERPARAM) or ((not is_underparam) and not REUSE_WEIGHTS_OVERPARAM):
            glorot_init(model)
        else:
            reuse_weights(smaller_model, model, H_prev)

        train_model(model, X_train, y_train_onehot, y_train, is_underparam, lr=lr, batch_size=batch_size)
        train_zeroone, train_mse, train_ce = evaluate(model, X_train, y_train_onehot, y_train)
        test_zeroone, test_mse, test_ce = evaluate(model, X_test, y_test_onehot, y_test)

        rows.append({
            "lr": lr, "batch_size": batch_size, "seed": seed, "H": H,
            "train_zeroone": train_zeroone, "test_zeroone": test_zeroone,
            "train_MSE": train_mse, "test_MSE": test_mse,
            "train_CE": train_ce, "test_CE": test_ce,
        })
        smaller_model = model
        H_prev = H

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"task{task_id:03d}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False, float_format="%.6f")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one Stage 2 full-sweep cell (all H_VALS).")
    parser.add_argument("--task_id", type=int, required=True,
                         help=f"Index in [0, {TOTAL_TASKS}) identifying one (lr, batch_size, seed) cell.")
    parser.add_argument("--output_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--dry_run", action="store_true",
                         help="Print the (lr, batch_size, seed) this task_id maps to, without training.")
    parser.add_argument("--full_batch", action="store_true",
                         help="Override to full-batch training (batch_size=N_TRAIN) for fast local "
                              "testing. The cluster run should NOT use this -- batch_size=32 is the "
                              "actual candidate from the Stage 1 probe.")
    args = parser.parse_args()

    lr, batch_size, seed = decode_task_id(args.task_id)
    if args.full_batch:
        batch_size = N_TRAIN
    if args.dry_run:
        print(f"task_id={args.task_id} -> lr={lr}, batch_size={batch_size}, seed={seed}")
    else:
        print(f"task_id={args.task_id}: lr={lr}, batch_size={batch_size}, seed={seed}, H_VALS={H_VALS}")
        out_path = run_full_task(args.task_id, args.output_dir, full_batch=args.full_batch)
        print(f"Wrote {out_path}")
