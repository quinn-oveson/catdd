#!/bin/bash
# One command to run the whole weight-norm experiment on the cluster.
#
#   ./slurm/submit_weight_norm_experiment.sh              # submit everything
#   ./slurm/submit_weight_norm_experiment.sh --dry-run    # print, submit nothing
#
# Submits the two sweep presets that differ only in weight reuse (see
# sweep_config.py -- `belkin` reuses the previous smaller model's weights at
# every underparameterized H, `belkin_noreuse` starts every H from a fresh
# Glorot init; everything else, including lr=0.0005, batch_size=32 and seeds
# 0-4, is identical), then submits a post-processing job that runs only if
# every training task succeeded.
#
# Four jobs total:
#   1. belkin, chain tier      -- the multi-H weight-reuse segments
#   2. belkin, 1-H tier        -- the overparameterized H's, --nice
#   3. belkin_noreuse, 1-H tier -- all 23 H's are independent here, --nice
#   4. finalize                -- aggregate + plot, --dependency=afterok on 1-3
#
# The --array/--mem/--time for each are NOT written here. They come from
# full_sweep.slurm_array_specs(), which derives them from the preset's actual
# SEGMENTS -- the task_ids of a tier are not a contiguous block (segment index
# is the fastest-varying axis), so hand-written ranges get it wrong and
# under-provision the expensive tasks. Inspect what will be submitted with:
#   CATDD_SWEEP=belkin python full_sweep.py --print_array_specs
#
# Optional environment variables:
#   EXCLUDE_NODES   passed through as --exclude=. Empty by default. Node
#                   m13l-2-2 had an uncorrectable ECC fault during the July
#                   2026 run (tasks failed there in ~30s while succeeding
#                   everywhere else) and that --exclude was never committed --
#                   so if you see that signature again:
#                     EXCLUDE_NODES=m13l-2-2 ./slurm/submit_weight_norm_experiment.sh

set -euo pipefail

DRY_RUN=false
case "${1:-}" in
    --dry-run) DRY_RUN=true ;;
    "") ;;
    *) echo "Unknown argument: $1" >&2; echo "Usage: $0 [--dry-run]" >&2; exit 2 ;;
esac

# A wrapper script is NOT copied to /var/spool/slurmd the way an sbatch script
# is, so $BASH_SOURCE is trustworthy here (unlike inside the .sbatch files --
# see the $0 gotcha in full_sweep_array.sbatch). sbatch must be invoked from
# mnist_dd/, because the .sbatch files cd to $SLURM_SUBMIT_DIR.
MNIST_DD="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MNIST_DD"

ARRAY_SBATCH="slurm/full_sweep_array.sbatch"
FINALIZE_SBATCH="slurm/finalize_weight_norm.sbatch"
PRESETS=(belkin belkin_noreuse)

EXCLUDE_ARG=()
if [[ -n "${EXCLUDE_NODES:-}" ]]; then
    EXCLUDE_ARG=(--exclude="$EXCLUDE_NODES")
fi

# --- Pre-flight: everything that can fail cheaply, before anything is queued ---

for f in "$ARRAY_SBATCH" "$FINALIZE_SBATCH"; do
    [[ -f "$f" ]] || { echo "ERROR: $MNIST_DD/$f not found." >&2; exit 1; }
done

# Run this on the FSL login node, not the laptop. Skipped for --dry-run, which
# submits nothing -- previewing the plan from the laptop before ssh'ing is fine.
if [[ "$DRY_RUN" == false ]] && ! command -v sbatch > /dev/null; then
    echo "ERROR: sbatch not found -- this has to run on the cluster login node." >&2
    echo "    ssh <netid>@ssh.rc.byu.edu" >&2
    exit 1
fi

# Resolve every preset's tiers ONCE, here, and reuse the output below for both
# the summary table and the submission loop.
#
# Worth the care: each call is a fresh `python full_sweep.py`, which imports
# torch, and on a shared cluster filesystem a cold torch import costs tens of
# seconds -- vastly more than the ~1s it costs on a laptop. Calling it per-loop
# instead of caching made the whole script noticeably slow on the login node.
#
# This also doubles as the env check. full_sweep.py imports torch, so if the
# catdd env isn't active here (not just inside the jobs) this fails now, with
# the actual error shown -- rather than silently yielding zero tiers, submitting
# zero arrays, and falling over later on a malformed empty --dependency.
SPECS=()
for preset in "${PRESETS[@]}"; do
    if ! out=$(CATDD_SWEEP="$preset" python full_sweep.py --print_array_specs 2>&1); then
        echo "ERROR: could not compute array specs for preset '$preset':" >&2
        echo "$out" | sed 's/^/    /' >&2
        echo "  If that's a ModuleNotFoundError, activate the env first:" >&2
        echo "    module load miniforge3 && mamba activate catdd" >&2
        echo "  (module loads do not persist across SSH sessions -- redo them in each new shell)" >&2
        exit 1
    fi
    if [[ -z "$out" ]]; then
        echo "ERROR: preset '$preset' produced no array specs -- nothing to submit." >&2
        exit 1
    fi
    SPECS+=("$out")
done

# SLURM will not create the log directory, and silently fails the job if it's missing.
mkdir -p slurm_logs

# Compute nodes are not confirmed to have outbound internet, so data.py's
# auto-download can't be relied on -- a missing dataset would fail all 160 tasks
# one at a time.
if [[ ! -d mnist_data ]]; then
    echo "ERROR: $MNIST_DD/mnist_data not found." >&2
    echo "  Copy it up from your laptop first, e.g.:" >&2
    echo "    rsync -av mnist_data/ <netid>@ssh.rc.byu.edu:$MNIST_DD/mnist_data/" >&2
    exit 1
fi

# aggregate_full_sweep.py globs the WHOLE results/full_sweep/<preset>/ directory,
# so leftover task CSVs from an earlier run would be silently folded into the new
# summary. Refuse rather than mix -- and note that CSVs written before the
# reuse_weights fix are not the same experiment, since that fix changed the
# random draw sequence for the reused models' new units.
DIRTY=false
for preset in "${PRESETS[@]}"; do
    dir="results/full_sweep/$preset"
    if compgen -G "$dir/task*.csv" > /dev/null; then
        n=$(find "$dir" -maxdepth 1 -name 'task*.csv' | wc -l | tr -d ' ')
        echo "ERROR: $dir already holds $n task*.csv from a previous run." >&2
        DIRTY=true
    fi
done
if [[ "$DIRTY" == true ]]; then
    echo >&2
    echo "  Aggregation would mix those into the new summary. Archive them:" >&2
    echo "    for p in ${PRESETS[*]}; do" >&2
    echo "      [ -d results/full_sweep/\$p ] && mv results/full_sweep/\$p results/full_sweep/\$p.bak-\$(date +%Y%m%d-%H%M%S)" >&2
    echo "    done" >&2
    echo "  ...or delete them if you don't need them, then re-run this script." >&2
    exit 1
fi

# --- Show the plan before queueing anything ---

echo "Submitting the weight-norm experiment from $MNIST_DD"
[[ -n "${EXCLUDE_NODES:-}" ]] && echo "Excluding nodes: $EXCLUDE_NODES"
echo
printf '%-16s %-32s %6s %10s %7s\n' PRESET ARRAY MEM TIME TASKS
for i in "${!PRESETS[@]}"; do
    while read -r array_spec mem_gb time_hms n_tasks; do
        printf '%-16s %-32s %5sG %10s %7s\n' "${PRESETS[$i]}" "$array_spec" "$mem_gb" "$time_hms" "$n_tasks"
    done <<< "${SPECS[$i]}"
done
echo

# --- Submit, most expensive tier first ---
# slurm_array_specs() already returns tiers in descending cost order, so the
# first tier of the first preset goes in at normal priority and everything after
# it gets --nice: cheaper/shorter tiers are explicitly deprioritized whenever
# they compete with an expensive one for the same concurrent-node budget.

JOB_IDS=()
FIRST=true
for i in "${!PRESETS[@]}"; do
    preset="${PRESETS[$i]}"
    while read -r array_spec mem_gb time_hms n_tasks; do
        nice_arg=()
        if [[ "$FIRST" == true ]]; then FIRST=false; else nice_arg=(--nice); fi

        # ${arr[@]+"${arr[@]}"} rather than "${arr[@]}": under `set -u`, bash 3.2
        # (macOS, where --dry-run gets tested) treats expanding an EMPTY array as
        # an unbound variable. Linux bash 4+ doesn't, but this works on both.
        cmd=(sbatch --parsable
             --export=ALL,CATDD_SWEEP="$preset"
             --array="$array_spec"
             --mem="${mem_gb}G"
             --time="$time_hms"
             ${nice_arg[@]+"${nice_arg[@]}"} ${EXCLUDE_ARG[@]+"${EXCLUDE_ARG[@]}"}
             "$ARRAY_SBATCH")

        if [[ "$DRY_RUN" == true ]]; then
            echo "${cmd[*]}"
        else
            job_id=$("${cmd[@]}")
            JOB_IDS+=("$job_id")
            echo "Submitted $preset ${array_spec} (${n_tasks} tasks, ${mem_gb}G, ${time_hms}) -> job $job_id"
        fi
    done <<< "${SPECS[$i]}"
done

# --- Post-processing, gated on every training task succeeding ---

if [[ "$DRY_RUN" == true ]]; then
    echo "sbatch --parsable --dependency=afterok:<job1>:<job2>:<job3> ${EXCLUDE_ARG[*]:-} $FINALIZE_SBATCH"
    echo
    echo "(--dry-run: nothing was submitted)"
    exit 0
fi

if [[ ${#JOB_IDS[@]} -eq 0 ]]; then
    echo "ERROR: no training arrays were submitted -- refusing to submit the finalize job." >&2
    echo "  Check: CATDD_SWEEP=belkin python full_sweep.py --print_array_specs" >&2
    exit 1
fi

dependency=$(IFS=:; echo "${JOB_IDS[*]}")
finalize_id=$(sbatch --parsable --dependency=afterok:"$dependency" ${EXCLUDE_ARG[@]+"${EXCLUDE_ARG[@]}"} "$FINALIZE_SBATCH")
echo "Submitted finalize (afterok:${dependency}) -> job $finalize_id"

cat <<EOF

Monitor with:
  squeue -u \$USER
  sacct -j ${JOB_IDS[0]} --format=JobID,JobName%40,MaxRSS,Elapsed,State

On success, job $finalize_id writes:
  results/full_sweep_summary_belkin.csv
  results/full_sweep_summary_belkin_noreuse.csv
  results/full_sweep_weight_norm_comparison.png
  results/full_sweep_init_weight_norm_comparison.png

If ANY training task fails, afterok is never satisfied and job $finalize_id sits
in the queue forever with reason DependencyNeverSatisfied. In that case:
  scancel $finalize_id
  # re-run just the failed task_ids, e.g.:
  #   sbatch --export=ALL,CATDD_SWEEP=belkin --array=<failed ids> --mem=<G> --time=<hms> $ARRAY_SBATCH
  # then do the post-processing by hand:
  sbatch $FINALIZE_SBATCH
EOF
