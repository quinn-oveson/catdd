"""Local (non-cluster) driver for the Stage 2 full sweep.

Runs every (lr, batch_size, seed) cell from full_sweep.py's grid sequentially,
each as its own subprocess invocation of `full_sweep.py --task_id N` -- the
exact same code path a SLURM array task will run later, so anything broken
here would have been broken there too.

This is much more expensive than run_stage1_local.py: each task now trains
the full H_VALS curve (23 models, several deep into the overparameterized
regime with a fixed 6000 epochs and no early stopping), not just 3 small-H
probe models. Consider running a single task_id locally first to get a feel
for timing before looping over all of them.

Always runs with --full_batch: batch_size=32 means 4000/32=125 SGD steps per
epoch, which is far too slow for a laptop CPU/MPS. batch_size=32 is only
meant to run on the cluster (slurm/full_sweep_array.sbatch does NOT pass
--full_batch) -- this local driver is for exercising the pipeline / catching
bugs cheaply, not for producing the real results.

A failure on one task_id does not stop the rest; failed ids are logged so you
can inspect and retry just those.
"""
import os
import subprocess
import sys

from full_sweep import RESULTS_DIR, TOTAL_TASKS, decode_task_id

FULL_SWEEP_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_sweep.py")
FAILED_LOG = os.path.join(RESULTS_DIR, "failed_tasks.txt")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    failed = []

    for task_id in range(TOTAL_TASKS):
        lr, batch_size, seed = decode_task_id(task_id)
        print(f"[{task_id + 1}/{TOTAL_TASKS}] lr={lr}, batch_size={batch_size}, seed={seed} (running --full_batch locally)")
        result = subprocess.run([sys.executable, FULL_SWEEP_SCRIPT, "--task_id", str(task_id), "--full_batch"])
        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode})")
            failed.append(task_id)

    if failed:
        with open(FAILED_LOG, "w") as f:
            f.write("\n".join(str(t) for t in failed))
        print(f"\n{len(failed)} task(s) failed: {failed}")
        print(f"Failed task ids written to {FAILED_LOG}")
    else:
        print("\nAll tasks completed successfully.")


if __name__ == "__main__":
    main()
