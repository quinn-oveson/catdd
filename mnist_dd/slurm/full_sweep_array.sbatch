#!/bin/bash --login
# SLURM array job for the Stage 2 full sweep (full_sweep.py).
#
# Two kinds of array task, both decoded from $SLURM_ARRAY_TASK_ID by
# full_sweep.decode_task_id() -- see that module's docstring for why:
#   - task_id in [0, 10): a "chain" task -- one per (lr, batch_size, seed),
#     training every underparameterized H sequentially (weight reuse requires
#     the order).
#   - task_id in [10, 90): an "independent" task -- one per
#     (lr, batch_size, seed, H) for each of the 8 overparameterized H's
#     (52,55,60,70,100,200,300,1000). Each trains a single fresh-Glorot model
#     with no dependency on anything else, so these are fully parallelizable
#     instead of serializing the most expensive models (up to H=1000) behind
#     each other for no reason.
#
# Chain tasks are the priority: they're the ones we expect to take longer (up
# to 15 sequential models each, vs. 1 for an independent task), so submit them
# as their own array first at normal priority, and submit the independent
# tasks as a SEPARATE array afterward with a higher --nice value -- this
# explicitly deprioritizes the independent job whenever both are competing
# for the same limited concurrent-node budget, rather than hoping submission
# order alone gets that ordering right. See step 3 at the bottom of this file
# for the exact two-submission commands. This split also lets each array get
# its own --time budget instead of one placeholder covering both very
# different cost profiles.
#
# Before running this for real, do the ONE-TIME setup below (same env as
# probe_array.sbatch -- skip if you already have the `catdd` mamba env from
# that), and run the small calibration test described at the bottom of this
# file first.

#SBATCH --time=12:00:00           # PLACEHOLDER -- override with --time= at submission
                                   # time for each of the two arrays separately (see
                                   # step 3 at the bottom), since chain and independent
                                   # tasks have very different cost profiles (a chain
                                   # task always does the same ~15 underparam H's; an
                                   # independent task's cost depends entirely on which
                                   # single H it drew, from H=52 up to H=1000). With
                                   # batch_size=32 that's 4000/32=125 steps/epoch, and
                                   # (for overparameterized H, no early stopping)
                                   # always 6000 epochs =~ 750K SGD steps for the
                                   # single most expensive independent task (H=1000).
                                   # DO NOT submit either array until you've run the
                                   # calibration test below on BOTH a chain task and
                                   # the H=1000 task, replaced this with real measured
                                   # numbers (with margin) per array, and checked them
                                   # against your partition's max walltime (e.g.
                                   # `sinfo` or cluster docs).
#SBATCH --mem=4G                  # H up to 1000 is still a tiny model (~800K params,
                                   # batch_size=32) -- 4G should be plenty, but bump
                                   # this if a task OOMs.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --array=0-9           # PLACEHOLDER, always override with --array= at
                               # submission time -- defaults to just the 10 chain
                               # tasks (2 lr x 1 batch_size x 5 seeds) since those
                               # are the priority group. The 80 independent tasks
                               # (2 lr x 1 batch_size x 5 seeds x 8 overparameterized
                               # H's, task_id 10-89) get submitted as a separate,
                               # deprioritized array -- see step 3 below.
#SBATCH -J catdd_full_sweep
#SBATCH --output=slurm_logs/full_sweep_%A_%a.out
#SBATCH --error=slurm_logs/full_sweep_%A_%a.err

# mkdir -p slurm_logs before submitting -- SLURM will not create it for you.

mamba activate catdd          # Same env as probe_array.sbatch -- see that file's
                               # header comment for the one-time setup instructions
                               # (mamba env creation, and the CUDA-12x-not-13x /
                               # Pascal gotcha for installing pytorch on marylou9g).

# Before first submission: copy your already-downloaded mnist_data/ folder from
# your laptop to this same relative location on the cluster (e.g. via rsync/scp),
# if you haven't already for the probe run. Compute nodes are not confirmed to
# have outbound internet access, so don't rely on data.py's auto-download here.

cd "$SLURM_SUBMIT_DIR"   # mnist_dd/ directory you ran sbatch from
                          # (NOT "$(dirname "$0")" -- SLURM copies the script
                          # into /var/spool/slurmd/... before running it, so
                          # $0 does not point back to your submission directory)

# #SBATCH -J above only sets the name at submission time, before
# $SLURM_ARRAY_TASK_ID exists, so every task would otherwise show up in
# squeue as identical "catdd_full_sweep" rows. Rename this task's own job at
# runtime instead, decoding via full_sweep.decode_task_id() (the same source
# of truth full_sweep.py itself uses) rather than duplicating the grid
# definition here where it could drift out of sync.
CELL_NAME=$(python -c "
from full_sweep import decode_task_id
spec = decode_task_id(${SLURM_ARRAY_TASK_ID})
h_part = f'_H{spec.H}' if spec.H is not None else '_chain'
print(f'{spec.kind}_lr{spec.lr}_bs{spec.batch_size}_seed{spec.seed}{h_part}')
")
scontrol update JobId="$SLURM_JOB_ID" JobName="$CELL_NAME"

python full_sweep.py --task_id "$SLURM_ARRAY_TASK_ID"

# --- Recommended rollout, don't jump straight to the full array ---
# 1. Calibration test on ONE task of EACH kind -- their cost profiles are
#    unrelated, so timing one tells you nothing about the other:
#      mkdir -p slurm_logs
#      sbatch --array=0 --time=12:00:00 slurm/full_sweep_array.sbatch    # a chain task
#      sbatch --array=17 --time=12:00:00 slurm/full_sweep_array.sbatch   # task_id 17 =
#                                                                        # lr=0.001,seed=0,H=1000,
#                                                                        # the single most
#                                                                        # expensive independent task
#    Check progress with: squeue -u $USER   and inspect slurm_logs/*.out after.
# 2. Once both succeed, note each kind's actual runtime (with margin) -- you'll
#    use these as the two --time values below. Optionally submit a few more
#    tasks of each kind (e.g. --array=1,5,9 and --array=20,50,89) to confirm
#    consistency across different lr/seed/H combinations before committing to
#    the full arrays.
# 3. Submit the real thing as TWO SEPARATE arrays, chain first at normal
#    priority, independent second with a higher --nice so it's explicitly
#    deprioritized whenever both are competing for the same limited
#    concurrent-node budget (replace <CHAIN_TIME> / <INDEP_TIME> with the
#    measured numbers from step 2):
#      sbatch --array=0-9   --time=06:00:00  slurm/full_sweep_array.sbatch
#      sbatch --array=10-89 --time=00:30:00 --nice slurm/full_sweep_array.sbatch
