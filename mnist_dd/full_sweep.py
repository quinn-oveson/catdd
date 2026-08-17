"""Stage 2 full sweep: run the full Belkin H_VALS curve for the (lr, batch_size)
candidates in sweep_config.LR_GRID/BATCH_SIZE_GRID (see sweep_config.py -- the
preset named by $CATDD_SWEEP decides those, and `belkin`/`belkin_noreuse`
collapse them to the single best-fit candidate from the Stage 1 probe).

Each H's row records train/test zero-one, MSE and CE loss, how many epochs it
ran, and the L2 norm of its weights both at initialization and after training
(utils.weight_norms) -- the norms are what makes a `belkin` vs
`belkin_noreuse` pair able to say whether the double-descent spike tracks
weight norm.

Task splitting is derived from sweep_config.REUSE_WEIGHTS_UNDERPARAM/
REUSE_WEIGHTS_OVERPARAM via a per-H rule: H_VALS (sorted ascending) is
partitioned into maximal "segments" -- runs of consecutive H's where each H
after the first reuses weights from the H immediately before it, exactly
matching probe.py/sweep.py's own reuse-vs-glorot rule (reuse into a given H
is governed by THAT H's own under/overparam region flag, not the previous
H's). Each segment is trained by one task, sequentially within the segment
(weight reuse requires the order); different segments have no dependency on
each other and are fully parallelizable.

This means:
  - REUSE_WEIGHTS_UNDERPARAM=True, REUSE_WEIGHTS_OVERPARAM=False (Belkin's
    own setup, the `belkin` preset): one segment covering all underparameterized
    H's (a "chain" task) plus one singleton segment per overparameterized H
    (an "independent" task each) -- the interpolation threshold breaks the
    chain because the first overparameterized H's own flag is False.
  - Both True: a single segment spans ALL of H_VALS -- one fully serial
    chain per (lr, batch_size, seed), with weight reuse running continuously
    straight through the interpolation threshold.
  - Both False (the `belkin_noreuse` and `custom` presets): every H is its own
    singleton segment -- fully parallel.
  - REUSE_WEIGHTS_UNDERPARAM=False, REUSE_WEIGHTS_OVERPARAM=True: every
    underparameterized H is independent EXCEPT the last one, which gets
    folded into the same segment as the overparameterized chain (since the
    first overparameterized H's own flag is True, it extends rather than
    breaks that segment) -- so the first overparameterized H reuses from the
    last underparameterized H's trained model, without needing any
    cross-task checkpointing.

task_id decodes to one segment for one (lr, batch_size, seed) cell. Meant to
run identically two ways:
  - locally, looped over all task_ids (see run_full_sweep_local.py)
  - as one SLURM array task, with --task_id set from $SLURM_ARRAY_TASK_ID
    (see slurm/full_sweep_array.sbatch)

Sanity-check the whole mapping for free before spending any compute (every
command below needs the same CATDD_SWEEP as the run it's describing, since the
preset decides both the grid and the segments):
    for i in 0 1 2 ... ; do CATDD_SWEEP=belkin python full_sweep.py --task_id $i --dry_run; done
Check recommended SLURM --mem/--time per segment (see resource_tier() below)
with:
    CATDD_SWEEP=belkin python -c "from full_sweep import SEGMENTS, resource_tier; [print(i, len(s), resource_tier(s)) for i, s in enumerate(SEGMENTS)]"
"""
import argparse
import os
import sys
import time
from collections import namedtuple

import pandas as pd
import torch

from config import N_TRAIN, K
# H_VALS comes from the preset, not config: the belkin_h_* arms resample the
# widths themselves (see sweep_config.py).
from sweep_config import (SWEEP_NAME, H_VALS, LR_GRID, BATCH_SIZE_GRID, SEEDS,
                          REUSE_WEIGHTS_UNDERPARAM, REUSE_WEIGHTS_OVERPARAM,
                          NORMALIZE_REUSED_WEIGHTS, BALANCE_INIT,
                          INIT_NORM_TARGETS, RESCALE_OUTPUT_HEAVY, LOSS_FUNC)
from data import load_mnist_subset, onehot
from mlp import MLP
from train import train_model, evaluate
from utils import (glorot_init, reuse_weights, num_params, weight_norms,
                   normalize_to_glorot_norm, rescale_to_norm, balance_mlp)

N_LR = len(LR_GRID)
N_BS = len(BATCH_SIZE_GRID)
N_SEEDS = len(SEEDS)

RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
# Per-preset, so two presets' task CSVs never collide -- both can be in the
# SLURM queue at the same time (see sweep_config.py's docstring). Anything
# needing results/ itself should use RESULTS_ROOT rather than walking up from
# RESULTS_DIR, which is two levels deep.
RESULTS_DIR = os.path.join(RESULTS_ROOT, "full_sweep", SWEEP_NAME)

TaskSpec = namedtuple("TaskSpec", ["lr", "batch_size", "seed", "H_list"])


def _reuses_from_prev(H):
    """Whether H reuses weights from the H immediately before it (in sorted
    H_VALS order), governed by H's OWN region flag -- same rule as
    probe.py/sweep.py's inline glorot_init/reuse_weights branch."""
    is_underparam = num_params(H) < K * N_TRAIN
    return REUSE_WEIGHTS_UNDERPARAM if is_underparam else REUSE_WEIGHTS_OVERPARAM


def _build_segments():
    segments = [[H_VALS[0]]]  # first H overall has no predecessor -- always fresh Glorot
    for H in H_VALS[1:]:
        if _reuses_from_prev(H):
            segments[-1].append(H)
        else:
            segments.append([H])
    return segments


SEGMENTS = _build_segments()
N_SEGMENTS = len(SEGMENTS)
TOTAL_TASKS = N_LR * N_BS * N_SEEDS * N_SEGMENTS


def resource_tier(H_list):
    """Rough (mem_gb, time_hms) resource recommendation for a segment, from
    memory/runtime measured on BYU FSL: a single H trains one model (~4G,
    ~1h, generous); a full H_VALS-length chain trains every model in one
    process without releasing GPU memory between them (~32G, ~12h, safe);
    anything in between -- a partial chain -- measured ~16G/~6h (generous).
    These are tiered from specific measured runs, not an exact formula for
    arbitrary lengths -- recalibrate with `sacct` if H_VALS or the grid
    changes substantially enough to move a segment far from these anchors.
    """
    if len(H_list) == 1:
        return 4, "01:00:00"
    if len(H_list) == len(H_VALS):
        return 32, "12:00:00"
    return 16, "06:00:00"


def _compact_ranges(ids):
    """Collapse a sorted list of ints into SLURM --array syntax: [1,2,3,5] -> '1-3,5'."""
    parts = []
    start = prev = ids[0]
    for i in ids[1:]:
        if i == prev + 1:
            prev = i
            continue
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = i
    parts.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(parts)


def slurm_array_specs():
    """Group every task_id by its resource tier, most expensive tier first.

    Returns [(array_spec, mem_gb, time_hms, n_tasks), ...], where array_spec is a
    compacted SLURM --array string. One entry per tier present = one array to
    submit, since --mem/--time are per-submission and the tiers have very
    different cost profiles.

    Derived from SEGMENTS rather than written by hand, because the task_ids of a
    given tier are NOT a contiguous block: segment index is the fastest-varying
    axis in decode_task_id, so e.g. the belkin preset's 15-H chain lands on
    task_ids 0,9,18,27,36 (one per seed), not on 0 alone. A hand-written range
    gets that wrong and silently under-provisions the chain tasks.
    """
    by_tier = {}
    for task_id in range(TOTAL_TASKS):
        tier = resource_tier(SEGMENTS[task_id % N_SEGMENTS])
        by_tier.setdefault(tier, []).append(task_id)
    return [(_compact_ranges(ids), mem_gb, time_hms, len(ids))
            for (mem_gb, time_hms), ids in sorted(by_tier.items(), reverse=True)]


def decode_task_id(task_id):
    if not (0 <= task_id < TOTAL_TASKS):
        raise ValueError(f"task_id must be in [0, {TOTAL_TASKS}), got {task_id}")

    seg_idx = task_id % N_SEGMENTS
    idx2 = task_id // N_SEGMENTS
    seed_idx = idx2 % N_SEEDS
    bs_idx = (idx2 // N_SEEDS) % N_BS
    lr_idx = idx2 // (N_SEEDS * N_BS)
    return TaskSpec(LR_GRID[lr_idx], BATCH_SIZE_GRID[bs_idx], SEEDS[seed_idx], SEGMENTS[seg_idx])


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
        # so it always uses the real batch_size candidate from the grid.
        batch_size = N_TRAIN
    device = get_device()

    # data.py seeds its own generator for the subset; this fixes the weight
    # inits and batch order, which otherwise come from OS entropy.
    torch.manual_seed(seed)

    X_train, y_train, X_test, y_test = load_mnist_subset(n=N_TRAIN, seed=seed)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)
    y_train_onehot = onehot(y_train)
    y_test_onehot = onehot(y_test)

    rows = []
    smaller_model = None
    H_prev = None
    for j, H in enumerate(spec.H_list):
        t0 = time.time()
        is_underparam = num_params(H) < K * N_TRAIN
        model = MLP(H).to(device)
        if j == 0:
            glorot_init(model)
        else:
            reuse_weights(smaller_model, model, H_prev)
            if NORMALIZE_REUSED_WEIGHTS:
                # Keep the inherited directions, drop the inherited scale --
                # only meaningful on a reused model, hence inside this branch.
                normalize_to_glorot_norm(model)

        # Outside the branch: both set the init norm without touching the
        # function, so they apply to reused and fresh inits alike.
        if BALANCE_INIT:
            balance_mlp(model)
        if INIT_NORM_TARGETS is not None:
            rescale_to_norm(model, INIT_NORM_TARGETS[H],
                            output_heavy=RESCALE_OUTPUT_HEAVY)

        # Before the first SGD step: with weight reuse this carries the
        # previous (trained, larger-norm) model's weights, without it it's
        # Glorot scale -- the two are what the reuse-vs-weight-norm comparison
        # is actually contrasting, so record it, not just the trained norm.
        init_weight_norm = weight_norms(model)["weight_norm"]

        _, epoch_losses = train_model(model, X_train, y_train_onehot, y_train, is_underparam, lr=lr, batch_size=batch_size, loss_func=LOSS_FUNC)
        train_zeroone, train_mse, train_ce = evaluate(model, X_train, y_train_onehot, y_train)
        test_zeroone, test_mse, test_ce = evaluate(model, X_test, y_test_onehot, y_test)

        rows.append({
            "lr": lr, "batch_size": batch_size, "seed": seed, "H": H,
            "train_zeroone": train_zeroone, "test_zeroone": test_zeroone,
            "train_MSE": train_mse, "test_MSE": test_mse,
            "train_CE": train_ce, "test_CE": test_ce,
            "n_epochs": len(epoch_losses),
            "init_weight_norm": init_weight_norm,
            **weight_norms(model),
        })
        smaller_model = model
        H_prev = H
        print(f"  [{j + 1}/{len(spec.H_list)}] H={H} done in {time.time() - t0:.1f}s "
              f"(test_zeroone={test_zeroone:.4f})", flush=True)

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"Peak GPU memory allocated this task: {peak_mb:.1f} MB", flush=True)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"task{task_id:03d}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False, float_format="%.6f")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one Stage 2 full-sweep task (one weight-reuse segment).")
    parser.add_argument("--task_id", type=int, default=None,
                         help=f"Index in [0, {TOTAL_TASKS}), one per (lr, batch_size, seed, segment) -- "
                              f"see full_sweep.SEGMENTS for the current segment list.")
    parser.add_argument("--output_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--dry_run", action="store_true",
                         help="Print what this task_id maps to, without training.")
    parser.add_argument("--full_batch", action="store_true",
                         help="Override to full-batch training (batch_size=N_TRAIN) for fast local "
                              "testing. The cluster run should NOT use this -- the batch_size grid "
                              "value is the actual candidate to use.")
    parser.add_argument("--print_array_specs", action="store_true",
                         help="Print one 'array_spec mem_gb time n_tasks' line per resource tier for "
                              "this preset, most expensive first, and exit -- the --array/--mem/--time "
                              "to submit. Machine-readable (whitespace-separated); consumed by "
                              "slurm/submit_weight_norm_experiment.sh.")
    args = parser.parse_args()

    if args.print_array_specs:
        for array_spec, mem_gb, time_hms, n_tasks in slurm_array_specs():
            print(f"{array_spec} {mem_gb} {time_hms} {n_tasks}")
        raise SystemExit(0)

    if args.task_id is None:
        parser.error("--task_id is required (unless --print_array_specs is given)")

    spec = decode_task_id(args.task_id)
    lr, batch_size, seed = spec.lr, spec.batch_size, spec.seed
    if args.full_batch:
        batch_size = N_TRAIN
    kind = "chain" if len(spec.H_list) > 1 else "independent"
    if args.dry_run:
        print(f"[{SWEEP_NAME}] task_id={args.task_id} -> kind={kind}, lr={lr}, batch_size={batch_size}, seed={seed}, H_list={spec.H_list}")
    else:
        print(f"[{SWEEP_NAME}] task_id={args.task_id}: kind={kind}, lr={lr}, batch_size={batch_size}, seed={seed}, H_list={spec.H_list}", flush=True)
        out_path = run_full_task(args.task_id, args.output_dir, full_batch=args.full_batch)
        print(f"Wrote {out_path}")

        # Skip interpreter shutdown: observed 2026-08-03 on m13h/m13l, tasks
        # wrote their CSV then hung in CUDA teardown, holding a GPU until the
        # wall clock killed them. The file is already on disk here.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
