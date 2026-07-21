"""Local (non-cluster) driver for the Stage 1 probe sweep.

Runs every (lr, batch_size, seed) cell from probe.py's grid sequentially, each
as its own subprocess invocation of `probe.py --task_id N` -- the exact same
code path a SLURM array task will run later, so anything broken here would
have been broken there too.

A failure on one task_id does not stop the rest; failed ids are logged so you
can inspect and retry just those.
"""
import os
import subprocess
import sys

from probe import RESULTS_DIR, TOTAL_TASKS, decode_task_id

PROBE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe.py")
FAILED_LOG = os.path.join(RESULTS_DIR, "failed_tasks.txt")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    failed = []

    for task_id in range(TOTAL_TASKS):
        lr, batch_size, seed = decode_task_id(task_id)
        print(f"[{task_id + 1}/{TOTAL_TASKS}] lr={lr}, batch_size={batch_size}, seed={seed}")
        result = subprocess.run([sys.executable, PROBE_SCRIPT, "--task_id", str(task_id)])
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
