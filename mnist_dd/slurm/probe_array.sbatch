#!/bin/bash
# SLURM array job for the Stage 1 probe sweep (probe.py).
#
# One array task = one (lr, batch_size, seed) cell, decoded deterministically
# from $SLURM_ARRAY_TASK_ID by probe.decode_task_id(). Before running this
# for real, do the ONE-TIME setup below, and run the small calibration test
# described at the bottom of this file first.

#SBATCH --time=03:00:00          # Each task trains 3 models (H=4,5,8) sequentially.
                                  # Locally measured worst case per MODEL: ~23 min
                                  # (batch_size=25), so a task could take up to ~70 min
                                  # in the worst case. This budget gives >2x margin on
                                  # top of that. GPU kernel-launch overhead can behave
                                  # differently than laptop CPU/MPS for tiny models with
                                  # many small steps, so re-check against the calibration
                                  # run below before trusting it blindly. Finishing early
                                  # is free -- SLURM releases the GPU as soon as the task
                                  # exits -- the only cost of over-requesting is slightly
                                  # worse backfill scheduling (may queue a bit longer).
#SBATCH --mem=8G
#SBATCH --gpus=1
#SBATCH --array=0-244%20         # 245 total cells (7 lr x 7 batch_size x 5 seeds).
                                  # %20 caps 20 running concurrently -- a conservative
                                  # starting guess, not a known hard limit. FSL has 210
                                  # non-preemptible GPUs (marylou9g alone has 160x P100)
                                  # plus ~157 preemptible ones that are low-risk to use
                                  # here since any one cell now costs up to ~70 min if
                                  # evicted, not ~24 min. Raise this once you've seen how
                                  # it actually schedules.
#SBATCH -J catdd_probe
#SBATCH --output=slurm_logs/probe_%A_%a.out
#SBATCH --error=slurm_logs/probe_%A_%a.err

# mkdir -p slurm_logs before submitting -- SLURM will not create it for you.

module load miniforge3
mamba activate catdd          # ONE-TIME SETUP (run interactively before first submission):
                               #   module load miniforge3
                               #   mamba create -n catdd python=3.10
                               #   mamba activate catdd
                               #   mamba install numpy pandas
                               #   Then follow https://rc.byu.edu/wiki/?page=PyTorch to
                               #   install the GPU build of pytorch/torchvision.
                               #   DO NOT just match nvidia-smi's reported CUDA version --
                               #   that's only the driver's ceiling. marylou9g (our main
                               #   GPU pool, 160x P100) is Pascal / compute capability 6.0,
                               #   and CUDA-13-targeted builds (e.g. cuda130) have DROPPED
                               #   Pascal support entirely -- they'll install and even
                               #   launch fine, then die on first .cuda() call with
                               #   "no kernel image is available for execution on the
                               #   device". Use a cuda12x build instead (e.g. cuda126),
                               #   which still includes sm_60 kernels and is backward-
                               #   compatible with newer drivers. Confirmed working combo:
                               #   pytorch=2.6.0=cuda126_generic_py310_hd77d226_204 +
                               #   torchvision=0.21.0=cuda126_py310_h4459643_1
                               #   (torchvision version must match the pytorch minor
                               #   version -- 2.6.x pairs with 0.21.x, 2.5.x with 0.20.x,
                               #   etc. -- let mamba's solver reject bad pairings rather
                               #   than trusting hand-picked build strings blindly).

# Before first submission: copy your already-downloaded mnist_data/ folder from
# your laptop to this same relative location on the cluster (e.g. via rsync/scp).
# Compute nodes are not confirmed to have outbound internet access, so don't
# rely on data.py's auto-download working here.

cd "$SLURM_SUBMIT_DIR"   # mnist_dd/ directory you ran sbatch from
                          # (NOT "$(dirname "$0")" -- SLURM copies the script
                          # into /var/spool/slurmd/... before running it, so
                          # $0 does not point back to your submission directory)
python probe.py --task_id "$SLURM_ARRAY_TASK_ID"

# --- Recommended rollout, don't jump straight to the full array ---
# 1. Calibration test on 3 known cells (worst-case batch=25, mid full-batch,
#    and the last index) to confirm the env/data/code actually works on real
#    cluster hardware, and to get a real timing number:
#      mkdir -p slurm_logs
#      sbatch --array=0,100,244 --time=01:30:00 slurm/probe_array.sbatch
#    Check results with: squeue -u $USER   and inspect slurm_logs/*.out after.
# 2. Once those 3 succeed, compare their actual runtime to the --time above,
#    adjust if needed, then submit the real thing:
#      sbatch slurm/probe_array.sbatch
