#!/bin/bash --login
# Post-processing for the weight-norm experiment: aggregate both presets and
# draw the comparison figures.
#
# Not meant to be submitted by hand -- slurm/submit_weight_norm_experiment.sh
# submits this last, with --dependency=afterok on all three training arrays, so
# it runs only once every task in every array has succeeded. (Running it by
# hand is fine too, once the task CSVs are all there; it's cheap and idempotent.)
#
# CPU-only and short -- this reads CSVs and draws two PNGs, no training, no GPU.

#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -J catdd_weight_norm_finalize
#SBATCH --output=slurm_logs/finalize_weight_norm_%j.out
#SBATCH --error=slurm_logs/finalize_weight_norm_%j.err

set -euo pipefail

mamba activate catdd    # same env as full_sweep_array.sbatch

cd "$SLURM_SUBMIT_DIR"  # mnist_dd/ -- NOT "$(dirname "$0")", SLURM copies this
                         # script to /var/spool/slurmd before running it

# The env var picks both the input directory (results/full_sweep/<preset>/) and
# the output filename (results/full_sweep_summary_<preset>.csv), so each preset
# needs its own invocation.
CATDD_SWEEP=belkin         python aggregate_full_sweep.py
CATDD_SWEEP=belkin_noreuse python aggregate_full_sweep.py

# Both norms: weight_norm is what training grew, init_weight_norm is what weight
# reuse carried in before the first SGD step. Together they separate "started
# big" from "grew big", which is the whole point of the belkin/belkin_noreuse
# pair -- one alone can't distinguish those.
for metric in weight_norm init_weight_norm; do
    python plot_weight_norm_comparison.py \
        --summary_path1 results/full_sweep_summary_belkin.csv \
        --sweep_name1 "Belkin weight reuse" \
        --summary_path2 results/full_sweep_summary_belkin_noreuse.csv \
        --sweep_name2 "No weight reuse" \
        --norm_metric "$metric" \
        --out_path "results/full_sweep_${metric}_comparison.png"
done

echo "Done. Copy these back to your laptop to look at them:"
echo "  results/full_sweep_summary_belkin.csv"
echo "  results/full_sweep_summary_belkin_noreuse.csv"
echo "  results/full_sweep_weight_norm_comparison.png"
echo "  results/full_sweep_init_weight_norm_comparison.png"
