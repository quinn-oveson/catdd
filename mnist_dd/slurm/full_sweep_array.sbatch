#!/bin/bash --login
# SLURM array job for the Stage 2 full sweep (full_sweep.py).
#
# WHICH SWEEP THIS RUNS IS NOT IN THIS FILE. Every submission must name a
# sweep_config.py preset via the CATDD_SWEEP environment variable:
#   sbatch --export=ALL,CATDD_SWEEP=belkin          ... slurm/full_sweep_array.sbatch
#   sbatch --export=ALL,CATDD_SWEEP=belkin_noreuse  ... slurm/full_sweep_array.sbatch
# The preset decides the grid, the behavior flags, the number of tasks AND the
# output directory (results/full_sweep/<preset>/), so two presets can be in the
# queue at the same time without colliding. This is deliberately an env var and
# not an edit to sweep_config.py: each compute node re-imports that module when
# its array task actually STARTS, so editing it while a second array is still
# queued would silently change what that queued job runs.
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
#   CATDD_SWEEP=belkin python -c "
#   from full_sweep import SEGMENTS, TOTAL_TASKS, resource_tier
#   print(TOTAL_TASKS, 'tasks')
#   for i, s in enumerate(SEGMENTS):
#       mem, t = resource_tier(s)
#       print(i, len(s), s, f'mem={mem}G time={t}')
#   "
# Segment index = task_id % len(SEGMENTS), i.e. the segment is the FASTEST-
# varying axis, so a given segment's tasks are spread across the array at a
# stride of len(SEGMENTS) -- they are NOT a contiguous block at the front.
# For the two presets this experiment uses:
#   CATDD_SWEEP=belkin -- 1 lr x 1 batch_size x 5 seeds x 9 segments = 45
#     tasks. Segment 0 is the 15-H underparam chain (H=4..50, weight reuse) and
#     segments 1-8 are the 8 overparameterized H's (52,55,60,70,100,200,300,
#     1000) each as their own 1-H segment. So the FIVE chain tasks (one per
#     seed) are task_ids 0,9,18,27,36 -- all five need the chain tier.
#   CATDD_SWEEP=belkin_noreuse -- no reuse anywhere, so all 23 H's are their
#     own 1-H segment: 1 x 1 x 5 seeds x 23 segments = 115 tasks, every one of
#     them the cheap tier, fully parallel.
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
                               # submission time -- 45 is only the CATDD_SWEEP=belkin
                               # task count (1 lr x 1 batch_size x 5 seeds x 9
                               # segments); belkin_noreuse has 115. See the header
                               # comment above for how to recompute this per preset.
#SBATCH -J catdd_full_sweep
#SBATCH --output=slurm_logs/full_sweep_%A_%a.out
#SBATCH --error=slurm_logs/full_sweep_%A_%a.err

# mkdir -p slurm_logs before submitting -- SLURM will not create it for you.

# Fail immediately rather than falling back to sweep_config's default preset --
# an unnamed submission would quietly run the wrong experiment (and, for the
# `custom` default, a 3450-task grid) into the wrong results directory.
: "${CATDD_SWEEP:?not set -- submit with --export=ALL,CATDD_SWEEP=belkin (or belkin_noreuse)}"

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
# The preset goes in the name too -- two presets' arrays can be running at the
# same time and are otherwise indistinguishable in squeue.
CELL_NAME=$(python -c "
from full_sweep import decode_task_id
from sweep_config import SWEEP_NAME
spec = decode_task_id(${SLURM_ARRAY_TASK_ID})
kind = 'chain' if len(spec.H_list) > 1 else 'independent'
h_part = f'_H{spec.H_list[0]}-{spec.H_list[-1]}' if len(spec.H_list) > 1 else f'_H{spec.H_list[0]}'
print(f'{SWEEP_NAME}_{kind}_lr{spec.lr}_bs{spec.batch_size}_seed{spec.seed}{h_part}')
")
scontrol update JobId="$SLURM_JOB_ID" JobName="$CELL_NAME"

python full_sweep.py --task_id "$SLURM_ARRAY_TASK_ID"

# ============================================================================
# NORMAL PATH -- the whole weight-norm experiment in one command:
#
#     ./slurm/submit_weight_norm_experiment.sh
#
# That submits both presets (belkin and belkin_noreuse), one array per resource
# tier with the correct --array/--mem/--time derived from
# full_sweep.slurm_array_specs(), plus a finalize job depending on all of them
# that aggregates and plots. Preview it without submitting anything with
# --dry-run. Everything below is the manual fallback -- for a single preset, a
# partial re-run of failed task_ids, or a preset the launcher doesn't cover.
# ============================================================================
#
# --- Manual rollout, don't jump straight to the full array ---
# DON'T hand-write --array ranges: the task_ids of a tier are not a contiguous
# block (segment index is the fastest-varying axis, so belkin's 15-H chain is
# task_ids 0,9,18,27,36 -- one per seed, NOT just 0, and putting 9,18,27,36 in
# the cheap array would OOM/time-out them). Ask for the ranges instead -- one
# "array_spec mem_gb time n_tasks" line per tier, most expensive first:
#     CATDD_SWEEP=<preset> python full_sweep.py --print_array_specs
# This covers any preset, including ones the launcher doesn't know about (e.g.
# turning on REUSE_WEIGHTS_OVERPARAM adds a full-chain tier at 32G/12:00:00).
# 1. Calibration test on ONE task of EACH tier present -- their cost profiles
#    are unrelated, so timing one tells you nothing about the other:
#      mkdir -p slurm_logs
#      sbatch --export=ALL,CATDD_SWEEP=belkin --array=0 --time=06:00:00 --mem=16G slurm/full_sweep_array.sbatch
#                                                                  # task_id 0 = the 15-H underparam chain (chain tier)
#      sbatch --export=ALL,CATDD_SWEEP=belkin --array=8 --time=01:00:00 --mem=4G  slurm/full_sweep_array.sbatch
#                                                                  # task_id 8 = seed=0, H=1000, the single
#                                                                  # most expensive 1-H segment (1-H tier)
#    Check progress with: squeue -u $USER   and inspect slurm_logs/*.out after.
#    Check ACTUAL memory usage with `sacct -j <jobid> --format=JobID,MaxRSS,Elapsed`
#    after each completes -- resource_tier()'s numbers are generous/safe
#    estimates from a specific prior measurement, not a guarantee for your run.
#    belkin_noreuse needs no separate calibration: every one of its tasks is a
#    1-H segment, the same tier as task_id 8 above.
# 2. Once both succeed, note each tier's actual runtime/memory (with margin)
#    -- you'll use these as the --time/--mem values below. Optionally submit
#    a few more tasks of each tier (e.g. --array=1,2,3,4 and
#    --array=16,24,32,40) to confirm consistency across different seed/H
#    combinations before committing to the full arrays.
# 3. Submit one array PER TIER, largest/most expensive tier first at normal
#    priority, smaller tiers after with a higher --nice so they're explicitly
#    deprioritized whenever multiple are competing for the same limited
#    concurrent-node budget. Take each tier's --array/--mem/--time straight
#    from --print_array_specs (above), overriding --mem/--time with the
#    measured numbers from step 2 if they differ:
#      CATDD_SWEEP=belkin python full_sweep.py --print_array_specs
#      # then, one sbatch per printed line, e.g. for the first (most expensive):
#      sbatch --export=ALL,CATDD_SWEEP=belkin --array=<spec> \
#             --time=<time> --mem=<mem>G slurm/full_sweep_array.sbatch
#      # and every subsequent line with --nice appended.
#    This is exactly what submit_weight_norm_experiment.sh automates.
# 4. Aggregate each preset separately (the env var picks both the input
#    directory and the output filename), then compare them -- or just run
#    `sbatch slurm/finalize_weight_norm.sbatch`, which does all four commands:
#      CATDD_SWEEP=belkin          python aggregate_full_sweep.py
#      CATDD_SWEEP=belkin_noreuse  python aggregate_full_sweep.py
#      python plot_weight_norm_comparison.py \
#        --summary_path1 results/full_sweep_summary_belkin.csv \
#        --sweep_name1 "Belkin weight reuse" \
#        --summary_path2 results/full_sweep_summary_belkin_noreuse.csv \
#        --sweep_name2 "No weight reuse"
