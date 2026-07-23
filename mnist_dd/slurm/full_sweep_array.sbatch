#!/bin/bash --login
# SLURM array job for the Stage 2 full sweep (full_sweep.py).
#
# Every array task decodes $SLURM_ARRAY_TASK_ID via full_sweep.decode_task_id()
# into one weight-reuse "segment" -- a run of one or more H's trained
# sequentially, for one (lr, batch_size, seed) cell -- see that module's
# docstring for exactly how segments are derived from sweep_config's
# REUSE_WEIGHTS_UNDERPARAM/REUSE_WEIGHTS_OVERPARAM flags. Segment sizes are
# NOT fixed at "underparam block first, overparam block second" in every flag
# combination (a segment can span the interpolation threshold), so don't
# assume a hardcoded task_id split -- run this before submitting anything to
# see the actual current segment sizes, where the expensive multi-H ones
# land, and each segment's recommended --mem/--time (full_sweep.resource_tier(),
# see below for where its numbers come from):
#   python -c "
#   from full_sweep import SEGMENTS, resource_tier
#   for i, s in enumerate(SEGMENTS):
#       mem, t = resource_tier(s)
#       print(i, len(s), s, f'mem={mem}G time={t}')
#   "
# (segment index = task_id % len(SEGMENTS); under today's default
# (sweep_config.BELKIN_CONFIG=True, REUSE_WEIGHTS_UNDERPARAM=True,
# REUSE_WEIGHTS_OVERPARAM=False) that's 1 lr x 1 batch_size x 5 seeds x 9
# segments = 45 tasks: segment 0 is the 15-H underparam chain, segments 1-8
# are the 8 overparameterized H's (52,55,60,70,100,200,300,1000) each as
# their own 1-H segment.)
#
# Memory scales with segment length: run_full_task() trains every H in a
# segment inside one long-lived process without resetting CUDA's peak-memory
# tracking between them, so a longer segment's peak allocation reflects the
# accumulated models/optimizer state, not just the largest single model.
# Measured on BYU FSL, three tiers (see full_sweep.resource_tier()):
#   - 1-H segment (independent):        ~4G,  ~1h  (generous)
#   - partial chain (today's 15-H one): ~16G, ~6h  (generous)
#   - full chain (all len(H_VALS)):     ~32G, ~12h (safe)
# A single shared --mem/--time can't be both cheap enough to not waste
# allocation on 1-H tasks and large enough to not OOM/timeout chain tasks, so
# --mem and --time are BOTH meant to be overridden per array submission (see
# step 3), one array per resource tier actually present among today's
# SEGMENTS.
#
# Larger segments are also the priority: they take longer than a 1-H segment
# (up to 15+ sequential models vs. 1), so submit them as their own array
# first at normal priority, and submit the 1-H segments as a SEPARATE array
# afterward with a higher --nice value -- this explicitly deprioritizes that
# job whenever both are competing for the same limited concurrent-node
# budget, rather than hoping submission order alone gets that ordering
# right. See step 3 at the bottom of this file for the exact submission
# commands. This split also lets each array get its own --mem/--time budget
# instead of one placeholder covering very different cost profiles.
#
# Before running this for real, do the ONE-TIME setup below (same env as
# probe_array.sbatch -- skip if you already have the `catdd` mamba env from
# that), and run the small calibration test described at the bottom of this
# file first.

#SBATCH --time=12:00:00           # PLACEHOLDER -- override with --time= at submission
                                   # time for each of the two arrays separately (see
                                   # step 3 at the bottom), since a multi-H segment and
                                   # a 1-H segment have very different cost profiles (a
                                   # multi-H segment always does the same ~15
                                   # underparam H's under today's default flags; a 1-H
                                   # segment's cost depends entirely on which single H
                                   # it drew, from H=52 up to H=1000). With
                                   # batch_size=32 that's 4000/32=125 steps/epoch, and
                                   # (for overparameterized H, no early stopping)
                                   # always 6000 epochs =~ 750K SGD steps for the
                                   # single most expensive 1-H segment (H=1000).
                                   # DO NOT submit either array until you've run the
                                   # calibration test below on BOTH a multi-H segment
                                   # and the H=1000 task, replaced this with real
                                   # measured numbers (with margin) per array, and
                                   # checked them against your partition's max
                                   # walltime (e.g. `sinfo` or cluster docs).
#SBATCH --mem=32G                 # PLACEHOLDER -- fail-safe default set to the
                                   # largest tier (a full H_VALS-length chain, see
                                   # the header comment above), so an unoverridden
                                   # submission wastes allocation rather than OOMing.
                                   # Override with --mem= per array submission to
                                   # match that array's actual tier (see step 3).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --array=0-44          # PLACEHOLDER, always override with --array= at
                               # submission time -- defaults to the full 45-task
                               # range under today's default flags (1 lr x 1
                               # batch_size x 5 seeds x 9 segments, see the header
                               # comment above for how to recompute this if
                               # sweep_config's flags/grid ever change).
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
kind = 'chain' if len(spec.H_list) > 1 else 'independent'
h_part = f'_H{spec.H_list[0]}-{spec.H_list[-1]}' if len(spec.H_list) > 1 else f'_H{spec.H_list[0]}'
print(f'{kind}_lr{spec.lr}_bs{spec.batch_size}_seed{spec.seed}{h_part}')
")
scontrol update JobId="$SLURM_JOB_ID" JobName="$CELL_NAME"

python full_sweep.py --task_id "$SLURM_ARRAY_TASK_ID"

# --- Recommended rollout, don't jump straight to the full array ---
# These task_ids and tiers are for today's default flags (sweep_config.
# BELKIN_CONFIG=True, REUSE_WEIGHTS_UNDERPARAM=True, REUSE_WEIGHTS_OVERPARAM=
# False) -- re-verify with the SEGMENTS/resource_tier one-liner in the header
# comment above before trusting them if you've changed sweep_config.py (e.g.
# turning on REUSE_WEIGHTS_OVERPARAM adds a third, full-chain tier at
# --mem=32G --time=12:00:00 -- submit that array too, at the highest
# priority since it's the most expensive tier).
# 1. Calibration test on ONE task of EACH tier present -- their cost profiles
#    are unrelated, so timing one tells you nothing about the other:
#      mkdir -p slurm_logs
#      sbatch --array=0 --time=06:00:00 --mem=16G slurm/full_sweep_array.sbatch   # task_id 0 =
#                                                                                 # the 15-H underparam chain (partial-chain tier)
#      sbatch --array=8 --time=01:00:00 --mem=4G  slurm/full_sweep_array.sbatch   # task_id 8 =
#                                                                                 # seed=0, H=1000, the single
#                                                                                 # most expensive 1-H segment (1-H tier)
#    Check progress with: squeue -u $USER   and inspect slurm_logs/*.out after.
#    Check ACTUAL memory usage with `sacct -j <jobid> --format=JobID,MaxRSS,Elapsed`
#    after each completes -- resource_tier()'s numbers are generous/safe
#    estimates from a specific prior measurement, not a guarantee for your run.
# 2. Once both succeed, note each tier's actual runtime/memory (with margin)
#    -- you'll use these as the --time/--mem values below. Optionally submit
#    a few more tasks of each tier (e.g. --array=1,2,3,4 and
#    --array=16,24,32,40) to confirm consistency across different seed/H
#    combinations before committing to the full arrays.
# 3. Submit the real thing as one array PER TIER present, largest/most
#    expensive tier first at normal priority, smaller tiers after with a
#    higher --nice so they're explicitly deprioritized whenever multiple are
#    competing for the same limited concurrent-node budget (replace
#    <CHAIN_TIME>/<CHAIN_MEM> etc. with the measured numbers from step 2 --
#    resource_tier()'s defaults shown here are a starting point):
#      sbatch --array=0    --time=06:00:00 --mem=16G          slurm/full_sweep_array.sbatch   # partial-chain tier
#      sbatch --array=1-44 --time=01:00:00 --mem=4G  --nice   slurm/full_sweep_array.sbatch   # 1-H tier
