"""Sweep-behavior configuration: the (lr, batch_size, seed) grid for Stage 2/3
full_sweep.py, and how weight-reuse / LR-decay / early-stopping behave in the
underparameterized vs. overparameterized regions (consumed by train.py,
full_sweep.py, probe.py, sweep.py). Kept separate from config.py, which holds
physical/model constants (D, K, N_TRAIN, H_VALS, LR, MAX_EPOCHS, ...)
describing the MLP/MNIST problem itself rather than how we search over or
parallelize training runs.

BELKIN_CONFIG = True: one-command reproduction of Belkin et al. 2019 Fig 3.
Collapses the grid to the single (lr, batch_size) candidate identified by the
Stage 1 probe sweep (see probe.py / results/probe_summary.csv) as the best
fit against Belkin's digitized curve, and locks all six behavior flags plus
LOSS_FUNC to Belkin's exact setup below -- so `python full_sweep.py`
reproduces the paper's plot without re-running the Stage 1 search.

BELKIN_CONFIG = False: the grid/flags below apply as literally written (edit
freely) for exploratory sweeps beyond Belkin's own setup.
"""
import torch.nn as nn

BELKIN_CONFIG = False

# Stage 1 probe's best-fit (lr, batch_size) candidate against Belkin's Fig 3
# (see results/probe_summary.csv / results/probe_vs_belkin.png).
BELKIN_LR = 0.0005
BELKIN_BATCH_SIZE = 32

# --- Grid to sweep over lr / batch_size / seed (Stage 2/3, full_sweep.py) ---
# Ignored (overridden below) when BELKIN_CONFIG=True.
# CE loss search: same LR range as the Stage 1 MSE probe (softmax+CE's
# output-layer gradient, softmax-onehot, is the same order of magnitude as
# MSE's 2*(output-onehot), so there's no reason to expect CE to need a
# different LR scale) crossed with a batch-size range from the MSE candidate
# (32) up through much larger batches (fewer, bigger SGD steps/epoch) to
# check sensitivity in that direction too. Kept >= 32 -- smaller batches
# just add wall-clock (more steps/epoch) without a distinct enough gradient-
# noise regime to be worth the extra grid points here.
LR_GRID = [0.0005, 0.001, 0.005, 0.01, 0.05]
BATCH_SIZE_GRID = [32, 64, 128, 256, 400, 800]
SEEDS = list(range(5))  # Belkin averages over 5 trials

# --- Behavior flags, split underparam/overparam, one consistent naming
# convention for all six. Ignored (overridden below) when BELKIN_CONFIG=True.
# Reuse into a given H is governed by THAT H's own region flag (not the
# previous H's) -- so if both flags are True, reuse runs continuously
# straight through the interpolation threshold; if only the overparam flag
# is True, the first overparam H still reuses from whatever model came right
# before it, even if that was an independently-trained underparam H. ---

# reuse the smaller model's weights (utils.reuse_weights) instead of a fresh
# Glorot init when moving to the next H
REUSE_WEIGHTS_UNDERPARAM = False   # Belkin = True
REUSE_WEIGHTS_OVERPARAM = False   # Belkin = False
# step the LR scheduler (config.DECAY_INTERVAL / config.GAMMA)
DECAY_UNDERPARAM = True    # Belkin = True
DECAY_OVERPARAM = True    # Belkin = False
# stop early once train classification error hits 0 (checked every
# config.EARLY_STOP_CHECK_INTERVAL epochs)
STOP_UNDERPARAM = True    # Belkin = True
STOP_OVERPARAM = True    # Belkin = False

# loss function used for training (Belkin = nn.MSELoss()). Swap for
# nn.CrossEntropyLoss() to train against cross-entropy instead -- the
# one-hot targets already used everywhere (train.py/full_sweep.py/probe.py)
# work as CrossEntropyLoss's soft-label target as-is, no other code changes
# needed.
LOSS_FUNC = nn.CrossEntropyLoss()    # Belkin = nn.MSELoss()

if BELKIN_CONFIG:
    LR_GRID = [BELKIN_LR]
    BATCH_SIZE_GRID = [BELKIN_BATCH_SIZE]
    REUSE_WEIGHTS_UNDERPARAM = True
    REUSE_WEIGHTS_OVERPARAM = False
    DECAY_UNDERPARAM = True
    DECAY_OVERPARAM = False
    STOP_UNDERPARAM = True
    STOP_OVERPARAM = False
    LOSS_FUNC = nn.MSELoss()
