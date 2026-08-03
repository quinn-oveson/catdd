"""Sweep-behavior configuration: the (lr, batch_size, seed) grid for Stage 2/3
full_sweep.py, and how weight-reuse / LR-decay / early-stopping behave in the
underparameterized vs. overparameterized regions (consumed by train.py,
full_sweep.py, probe.py, sweep.py). Kept separate from config.py, which holds
physical/model constants (D, K, N_TRAIN, H_VALS, LR, MAX_EPOCHS, ...)
describing the MLP/MNIST problem itself rather than how we search over or
parallelize training runs.

Which configuration applies is picked by the CATDD_SWEEP environment variable,
naming one of the PRESETS below -- NOT by editing this file:

    CATDD_SWEEP=belkin python full_sweep.py --task_id 0
    sbatch --export=ALL,CATDD_SWEEP=belkin_noreuse ... slurm/full_sweep_array.sbatch

An environment variable rather than a CLI flag because this module is imported
at module-load time by train.py, full_sweep.py, probe.py, sweep.py and by
slurm/full_sweep_array.sbatch's job-renaming subshell -- there is no single
entry point to thread a flag through.

Selecting by name rather than by editing matters on the cluster: each compute
node re-imports this module when its array task actually STARTS, so editing
these values while a second array is still queued would silently change what
that queued job runs. Two presets can be in the queue simultaneously and stay
independent, and each writes to its own results/full_sweep/<name>/ directory
and results/full_sweep_summary_<name>.csv (see full_sweep.RESULTS_DIR /
aggregate_full_sweep.SUMMARY_PATH).

Presets:
  belkin          One-command reproduction of Belkin et al. 2019 Fig 3.
                  Collapses the grid to the single (lr, batch_size) candidate
                  identified by the Stage 1 probe sweep (see probe.py /
                  results/probe_summary.csv) as the best fit against Belkin's
                  digitized curve, and locks all six behavior flags plus
                  LOSS_FUNC to Belkin's exact setup.
  belkin_noreuse  Identical to `belkin` in every respect EXCEPT
                  REUSE_WEIGHTS_UNDERPARAM, which is False -- so no model at
                  any width inherits weights from a smaller one. Paired with
                  `belkin` (same seeds, hence the same MNIST subsets; same lr,
                  batch size, loss, decay and early-stopping behavior) it
                  isolates weight reuse as the single independent variable,
                  for testing whether the double-descent spike at the
                  interpolation threshold tracks weight norm.
  belkin_reuse_normalized
                  Identical to `belkin` -- reuse and all -- except each reused
                  model is rescaled to a fresh init's norm before training
                  (per weight matrix, biases zeroed; see
                  utils.normalize_to_glorot_norm). Keeps the directions reuse
                  carries, drops the inflated scale, so its init_weight_norm
                  matches `belkin_noreuse` exactly. The third arm that
                  separates "large norm" from "reused directions": if the
                  spike vanishes here, norm was the cause; if it survives,
                  norm was only a correlate of whatever else reuse carries.
  belkin_wd<v>    Identical to `belkin` -- reuse on, weights untouched -- except
                  SGD carries a WEIGHT_DECAY of <v> (see WEIGHT_DECAY_ARMS
                  below for the values actually registered). Tests whether
                  penalizing the norm throughout training, rather than only
                  rescaling it once at init the way
                  `belkin_reuse_normalized` does, improves generalization.
                  Note the decay COMPOUNDS along the reuse chain: the shrunken
                  model trained at one H is the one reused into the next, so
                  this suppresses the chain's norm growth cumulatively rather
                  than per-model.
  custom          Exploratory sweeps beyond Belkin's own setup (currently the
                  cross-entropy lr x batch_size grid search). Edit freely --
                  this is the one preset meant to be changed in place.

Note on the two unrelated things called "decay": DECAY_UNDERPARAM/
DECAY_OVERPARAM are the LEARNING-RATE schedule (config.DECAY_INTERVAL /
config.GAMMA), and have nothing to do with WEIGHT_DECAY, which is SGD's L2
penalty on the parameters. A preset can have either, both or neither.
"""
import os

import torch.nn as nn

# Stage 1 probe's best-fit (lr, batch_size) candidate against Belkin's Fig 3
# (see results/probe_summary.csv / results/probe_vs_belkin.png).
BELKIN_LR = 0.0005
BELKIN_BATCH_SIZE = 32

# Belkin's exact setup. Behavior flags are split underparam/overparam, one
# consistent naming convention for all six:
#   REUSE_WEIGHTS_*  reuse the smaller model's weights (utils.reuse_weights)
#                    instead of a fresh Glorot init when moving to the next H
#   DECAY_*          step the LR scheduler (config.DECAY_INTERVAL / config.GAMMA)
#   STOP_*           stop early once train classification error hits 0 (checked
#                    every config.EARLY_STOP_CHECK_INTERVAL epochs)
# WEIGHT_DECAY is NOT split by region -- it is one scalar handed to
# torch.optim.SGD (train.train_model), applied at every H. 0.0 disables it,
# which is Belkin's own setup.
# Reuse into a given H is governed by THAT H's own region flag (not the
# previous H's) -- so if both reuse flags are True, reuse runs continuously
# straight through the interpolation threshold; if only the overparam flag is
# True, the first overparam H still reuses from whatever model came right
# before it, even if that was an independently-trained underparam H.
_BELKIN = dict(
    LR_GRID=[BELKIN_LR],
    BATCH_SIZE_GRID=[BELKIN_BATCH_SIZE],
    SEEDS=list(range(5)),  # Belkin averages over 5 trials
    REUSE_WEIGHTS_UNDERPARAM=True,
    REUSE_WEIGHTS_OVERPARAM=False,
    NORMALIZE_REUSED_WEIGHTS=False,
    DECAY_UNDERPARAM=True,
    DECAY_OVERPARAM=False,
    STOP_UNDERPARAM=True,
    STOP_OVERPARAM=False,
    WEIGHT_DECAY=0.0,
    LOSS_FUNC=nn.MSELoss(),
)

# The weight-norm experiment's control: Belkin's setup with reuse turned off on
# the underparam side too (the overparam side is already False in Belkin's own
# setup), so every H trains from a fresh Glorot init. Spread from _BELKIN
# rather than spelled out again, so the two presets provably differ in this one
# flag and nothing else -- editing _BELKIN moves both together.
_BELKIN_NOREUSE = {**_BELKIN, "REUSE_WEIGHTS_UNDERPARAM": False}

# The third arm, separating "large norm" from "reused directions". Reuse is ON
# exactly as in `belkin` -- same chain, same inherited weights -- but each
# reused model is rescaled to a fresh init's norm before training
# (utils.normalize_to_glorot_norm), so it keeps the directions it inherited
# without the inflated scale. Its init_weight_norm curve should lie exactly on
# `belkin_noreuse`'s, while its weights point where `belkin`'s do.
#
# Reading the result: if the double-descent spike DISAPPEARS here, the norm was
# the culprit. If it SURVIVES, the spike comes from something else reuse
# carries -- the directions / the particular basin -- and norm was a correlate.
_BELKIN_REUSE_NORMALIZED = {**_BELKIN, "NORMALIZE_REUSED_WEIGHTS": True}

# The weight-decay arms: Belkin's setup untouched -- reuse on, reused weights
# used exactly as inherited -- with SGD's L2 penalty turned on. Where
# `belkin_reuse_normalized` intervenes on the norm ONCE, at each H's init,
# this penalizes it continuously through all 6000 epochs, and because each H's
# trained (shrunken) model is what the next H reuses, the effect compounds
# along the chain rather than acting per-model.
#
# One preset per value rather than a WEIGHT_DECAY_GRID axis: a new grid axis
# would have to be threaded through decode_task_id/TOTAL_TASKS,
# slurm_array_specs, aggregate_full_sweep's (lr, batch_size) pivot index and
# every plot script's row lookup. A preset per value costs the same compute,
# is submitted the same way (one array each, in parallel), and each value
# lands in its own results/full_sweep/<name>/ and summary CSV for free.
#
# Values chosen against this setup's effective step size. Momentum 0.95 makes
# the effective lr ~lr/(1-0.95) = 0.01, and a full run is 6000 epochs x
# 4000/32 = 125 steps = 750k steps, so ignoring the data gradient the weights
# shrink by ~exp(-7500*wd) over a run. That approximation was checked against a
# short local run (300 full-batch steps, wd=0.01: predicted 3.0% shrink,
# measured 2.8%), so the exponent is trustworthy as an upper bound on the
# decay's strength -- the data gradient pushes back, so the real norm settles
# at some equilibrium above it.
#
# On that scale: 1e-5 -> exp(-0.075), a light touch; 1e-4 -> exp(-0.75), the
# interesting middle where the decay is comparable to the chain's norm growth
# (measured under `belkin`: 3.7 at H=4 rising to 23.1 at H=50); 1e-3 ->
# exp(-7.5), almost certainly over-regularized to the point of underfitting.
# 1e-3 is kept deliberately as the upper bracket -- an arm that is clearly too
# strong bounds the useful range, and a scan that only goes up to the value
# that happens to work can't show that. Widen or trim this dict to change the
# scan -- adding an entry adds a preset, no other file needs to know.
WEIGHT_DECAY_ARMS = {
    "belkin_wd1e-5": 1e-5,
    "belkin_wd1e-4": 1e-4,
    "belkin_wd1e-3": 1e-3,
}

# NOTE: the lr/batch-size GRIDS below belong to `custom` only. Neither Belkin
# preset above has a grid -- both are pinned to the single (BELKIN_LR,
# BELKIN_BATCH_SIZE) candidate.
#
# Cross-entropy grid search: same LR range as the Stage 1 MSE probe (softmax+
# CE's output-layer gradient, softmax-onehot, is the same order of magnitude as
# MSE's 2*(output-onehot), so there's no reason to expect CE to need a
# different LR scale) crossed with a batch-size range from the MSE candidate
# (32) up through much larger batches (fewer, bigger SGD steps/epoch) to check
# sensitivity in that direction too. Kept >= 32 -- smaller batches just add
# wall-clock (more steps/epoch) without a distinct enough gradient-noise regime
# to be worth the extra grid points here. The one-hot targets used everywhere
# (train.py/full_sweep.py/probe.py) work as CrossEntropyLoss's soft-label
# target as-is.
_CUSTOM = dict(
    LR_GRID=[0.0005, 0.001, 0.005, 0.01, 0.05],
    BATCH_SIZE_GRID=[32, 64, 128, 256, 400, 800],
    SEEDS=list(range(5)),
    REUSE_WEIGHTS_UNDERPARAM=False,
    REUSE_WEIGHTS_OVERPARAM=False,
    NORMALIZE_REUSED_WEIGHTS=False,
    DECAY_UNDERPARAM=True,
    DECAY_OVERPARAM=True,
    STOP_UNDERPARAM=True,
    STOP_OVERPARAM=True,
    WEIGHT_DECAY=0.0,
    LOSS_FUNC=nn.CrossEntropyLoss(),
)

PRESETS = {
    "belkin": _BELKIN,
    "belkin_noreuse": _BELKIN_NOREUSE,
    "belkin_reuse_normalized": _BELKIN_REUSE_NORMALIZED,
    "custom": _CUSTOM,
    **{name: {**_BELKIN, "WEIGHT_DECAY": wd} for name, wd in WEIGHT_DECAY_ARMS.items()},
}

SWEEP_NAME = os.environ.get("CATDD_SWEEP", "custom")
if SWEEP_NAME not in PRESETS:
    raise SystemExit(
        f"CATDD_SWEEP={SWEEP_NAME!r} is not a known sweep preset. "
        f"Valid names: {', '.join(sorted(PRESETS))}."
    )

_preset = PRESETS[SWEEP_NAME]

LR_GRID = _preset["LR_GRID"]
BATCH_SIZE_GRID = _preset["BATCH_SIZE_GRID"]
SEEDS = _preset["SEEDS"]
REUSE_WEIGHTS_UNDERPARAM = _preset["REUSE_WEIGHTS_UNDERPARAM"]
REUSE_WEIGHTS_OVERPARAM = _preset["REUSE_WEIGHTS_OVERPARAM"]
NORMALIZE_REUSED_WEIGHTS = _preset["NORMALIZE_REUSED_WEIGHTS"]
DECAY_UNDERPARAM = _preset["DECAY_UNDERPARAM"]
DECAY_OVERPARAM = _preset["DECAY_OVERPARAM"]
STOP_UNDERPARAM = _preset["STOP_UNDERPARAM"]
STOP_OVERPARAM = _preset["STOP_OVERPARAM"]
WEIGHT_DECAY = _preset["WEIGHT_DECAY"]
LOSS_FUNC = _preset["LOSS_FUNC"]
