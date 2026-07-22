#!/bin/bash --login
# SLURM array job for the Stage 2 full sweep (full_sweep.py).
#
# One array task = one (lr, batch_size, seed) cell, decoded deterministically
# from $SLURM_ARRAY_TASK_ID by full_sweep.decode_task_id(). Before running this
# for real, do the ONE-TIME setup below (same env as probe_array.sbatch -- skip
# if you already have the `catdd` mamba env from that), and run the small
# calibration test described at the bottom of this file first.

#SBATCH --time=12:00:00           # PLACEHOLDER -- this is NOT based on a measured
                                   # number, unlike probe_array.sbatch's estimate.
                                   # This task is a fundamentally different shape
                                   # than the probe: it trains the FULL H_VALS
                                   # curve (23 models per task), and 8 of those H's
                                   # (52,55,60,70,100,200,300,1000) are
                                   # over-parameterized -- no early stopping, always
                                   # runs the fixed 6000 epochs. Several of the
                                   # under-parameterized H's may also fail to reach
                                   # zero train error and run all 6000 epochs too.
                                   # With batch_size=32 that's 4000/32=125 steps/epoch,
                                   # so worst case ~23*6000*125 =~ 17M SGD steps in a
                                   # single task. DO NOT submit the full --array=0-9
                                   # until you've run the calibration test below and
                                   # replaced this with a real measured number (with
                                   # margin) -- and check it against your partition's
                                   # max walltime (e.g. `sinfo` or cluster docs).
#SBATCH --mem=4G                  # H up to 1000 is still a tiny model (~800K params,
                                   # batch_size=32) -- 4G should be plenty, but bump
                                   # this if a task OOMs.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --array=0-9           # 10 total cells (2 lr x 1 batch_size x 5 seeds).
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
# runtime instead, decoding lr/batch_size/seed via full_sweep.decode_task_id()
# (the same source of truth full_sweep.py itself uses) rather than duplicating
# the grid definition here where it could drift out of sync.
CELL_NAME=$(python -c "
from full_sweep import decode_task_id
lr, bs, seed = decode_task_id(${SLURM_ARRAY_TASK_ID})
print(f'lr{lr}_bs{bs}_seed{seed}')
")
scontrol update JobId="$SLURM_JOB_ID" JobName="$CELL_NAME"

python full_sweep.py --task_id "$SLURM_ARRAY_TASK_ID"

# --- Recommended rollout, don't jump straight to the full array ---
# 1. Calibration test on a SINGLE cell first (this task is much more expensive
#    than the probe's, so don't even start with 3 -- see how long just one
#    takes before committing more of your allocation):
#      mkdir -p slurm_logs
#      sbatch --array=0 --time=12:00:00 slurm/full_sweep_array.sbatch
#    Check progress with: squeue -u $USER   and inspect slurm_logs/*.out after.
#    If you want an earlier read before it finishes entirely, the per-H timing
#    is visible incrementally in the .out file as each H completes (add a
#    print in full_sweep.run_full_task's loop if you want per-H timestamps).
# 2. Once that succeeds, compare its actual runtime to the --time above,
#    replace the placeholder with a real number (with margin), and submit a
#    couple more cells (e.g. --array=0,5,9) to confirm consistency across
#    different lr/seed combinations before committing to the full array.
# 3. Only then submit the real thing:
#      sbatch slurm/full_sweep_array.sbatch
