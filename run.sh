#!/usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:50:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
runner="${STUDY_PYTHON:-python}"
if [[ -z "${STUDY_PYTHON:-}" && -x /local_nvme/gpu-release-validation/env-torch291-cu128/bin/python ]]; then
    runner=/local_nvme/gpu-release-validation/env-torch291-cu128/bin/python
fi
if [[ "$1" == "all" ]]; then
    export STUDY_RESULTS="${STUDY_RESULTS:-$PWD/replication}"
    "$runner" data_io.py
    "$runner" study.py check
    "$runner" study.py tune --shard 0
    "$runner" study.py tune --shard 1
    "$runner" study.py select
    "$runner" study.py main --seeds 7 19 43
    "$runner" study.py native --seeds 19
    "$runner" study.py probe
    "$runner" early.py --rounds 5
    exec "$runner" step_matching.py
fi
if [[ "$1" == "step-matching" ]]; then
    shift
    exec "$runner" -u step_matching.py "$@"
fi
if [[ "$1" == "early" ]]; then
    shift
    exec "$runner" -u early.py "$@"
fi
exec "$runner" -u study.py "$@"
