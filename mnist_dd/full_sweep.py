"""Stage 2 full sweep: run the full Belkin H_VALS curve for the (lr, batch_size)
candidates identified by the Stage 1 probe (see probe.py / plot_probe.py /
results/probe_summary.csv).

Belkin's weight-reuse scheme (config.REUSE_WEIGHTS_UNDERPARAM=True,
REUSE_WEIGHTS_OVERPARAM=False) only chains together the underparameterized H's
below the interpolation threshold -- each overparameterized H is always
trained from a fresh Glorot init, independent of every other H (including
each other). So there are two kinds of task, split by config.H_VALS via the
num_params(H) < K*N_TRAIN threshold:
  - "chain" tasks: one per (lr, batch_size, seed), training every
    underparameterized H sequentially (weight reuse requires the order).
  - "independent" tasks: one per (lr, batch_size, seed, H) for each
    overparameterized H -- these have no dependency on anything else and are
    fully parallelizable, unlike a naive "train all 23 H's in one task" design
    which would serialize the most expensive models (up to H=1000) behind
    each other for no reason.

task_id < N_CHAIN_TASKS decodes to a chain task; task_id >= N_CHAIN_TASKS
decodes to an independent single-H task. Meant to run identically two ways:
  - locally, looped over all task_ids (see run_full_sweep_local.py)
  - as one SLURM array task, with --task_id set from $SLURM_ARRAY_TASK_ID
    (see slurm/full_sweep_array.sbatch)

Sanity-check the whole mapping for free before spending any compute:
    for i in 0 1 2 ... ; do python full_sweep.py --task_id $i --dry_run; done
"""
import argparse
import os
import time
from collections import namedtuple

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

UNDERPARAM_H_VALS = [int(h) for h in H_VALS if num_params(h) < K * N_TRAIN]
OVERPARAM_H_VALS = [int(h) for h in H_VALS if num_params(h) >= K * N_TRAIN]

N_CHAIN_TASKS = N_LR * N_BS * N_SEEDS
N_INDEPENDENT_TASKS = N_LR * N_BS * N_SEEDS * len(OVERPARAM_H_VALS)
TOTAL_TASKS = N_CHAIN_TASKS + N_INDEPENDENT_TASKS

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "full_sweep")

TaskSpec = namedtuple("TaskSpec", ["kind", "lr", "batch_size", "seed", "H"])


def decode_task_id(task_id):
    if not (0 <= task_id < TOTAL_TASKS):
        raise ValueError(f"task_id must be in [0, {TOTAL_TASKS}), got {task_id}")

    if task_id < N_CHAIN_TASKS:
        seed_idx = task_id % N_SEEDS
        bs_idx = (task_id // N_SEEDS) % N_BS
        lr_idx = task_id // (N_SEEDS * N_BS)
        return TaskSpec("chain", LR_GRID[lr_idx], BATCH_SIZE_GRID[bs_idx], SEEDS[seed_idx], None)

    idx = task_id - N_CHAIN_TASKS
    n_overparam = len(OVERPARAM_H_VALS)
    h_idx = idx % n_overparam
    idx2 = idx // n_overparam
    seed_idx = idx2 % N_SEEDS
    bs_idx = (idx2 // N_SEEDS) % N_BS
    lr_idx = idx2 // (N_SEEDS * N_BS)
    return TaskSpec("independent", LR_GRID[lr_idx], BATCH_SIZE_GRID[bs_idx], SEEDS[seed_idx], OVERPARAM_H_VALS[h_idx])


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def run_full_task(task_id, output_dir=RESULTS_DIR, full_batch=False):
    spec = decode_task_id(task_id)
    lr, batch_size, seed = spec.lr, spec.batch_size, spec.seed
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

    if spec.kind == "chain":
        smaller_model = None
        H_prev = None
        for j, H in enumerate(UNDERPARAM_H_VALS):
            t0 = time.time()
            model = MLP(H).to(device)
            # Every H here is underparameterized by construction, so Belkin's rule
            # reduces to: Glorot for the first (smallest) network, reuse after that.
            if j == 0 or not REUSE_WEIGHTS_UNDERPARAM:
                glorot_init(model)
            else:
                reuse_weights(smaller_model, model, H_prev)

            train_model(model, X_train, y_train_onehot, y_train, True, lr=lr, batch_size=batch_size)
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
            print(f"  [{j + 1}/{len(UNDERPARAM_H_VALS)}] H={H} done in {time.time() - t0:.1f}s "
                  f"(test_zeroone={test_zeroone:.4f})", flush=True)

    else:  # "independent" -- a single overparameterized H, always fresh Glorot init
        t0 = time.time()
        H = spec.H
        model = MLP(H).to(device)
        if REUSE_WEIGHTS_OVERPARAM:
            raise NotImplementedError(
                "REUSE_WEIGHTS_OVERPARAM=True would reintroduce a dependency on the "
                "previous H's weights, breaking the independent-task parallelization "
                "this script relies on for overparameterized H's."
            )
        glorot_init(model)

        train_model(model, X_train, y_train_onehot, y_train, False, lr=lr, batch_size=batch_size)
        train_zeroone, train_mse, train_ce = evaluate(model, X_train, y_train_onehot, y_train)
        test_zeroone, test_mse, test_ce = evaluate(model, X_test, y_test_onehot, y_test)

        rows.append({
            "lr": lr, "batch_size": batch_size, "seed": seed, "H": H,
            "train_zeroone": train_zeroone, "test_zeroone": test_zeroone,
            "train_MSE": train_mse, "test_MSE": test_mse,
            "train_CE": train_ce, "test_CE": test_ce,
        })
        print(f"  H={H} done in {time.time() - t0:.1f}s (test_zeroone={test_zeroone:.4f})", flush=True)

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"Peak GPU memory allocated this task: {peak_mb:.1f} MB", flush=True)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"task{task_id:03d}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False, float_format="%.6f")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one Stage 2 full-sweep task (a weight-reuse chain or one independent overparam H).")
    parser.add_argument("--task_id", type=int, required=True,
                         help=f"Index in [0, {TOTAL_TASKS}): [0, {N_CHAIN_TASKS}) are chain tasks, "
                              f"[{N_CHAIN_TASKS}, {TOTAL_TASKS}) are independent overparam-H tasks.")
    parser.add_argument("--output_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--dry_run", action="store_true",
                         help="Print what this task_id maps to, without training.")
    parser.add_argument("--full_batch", action="store_true",
                         help="Override to full-batch training (batch_size=N_TRAIN) for fast local "
                              "testing. The cluster run should NOT use this -- batch_size=32 is the "
                              "actual candidate from the Stage 1 probe.")
    args = parser.parse_args()

    spec = decode_task_id(args.task_id)
    lr, batch_size, seed = spec.lr, spec.batch_size, spec.seed
    if args.full_batch:
        batch_size = N_TRAIN
    if args.dry_run:
        h_desc = f"H={spec.H}" if spec.H is not None else f"H_chain={UNDERPARAM_H_VALS}"
        print(f"task_id={args.task_id} -> kind={spec.kind}, lr={lr}, batch_size={batch_size}, seed={seed}, {h_desc}")
    else:
        print(f"task_id={args.task_id}: kind={spec.kind}, lr={lr}, batch_size={batch_size}, seed={seed}", flush=True)
        out_path = run_full_task(args.task_id, args.output_dir, full_batch=args.full_batch)
        print(f"Wrote {out_path}")
