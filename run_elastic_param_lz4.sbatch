#!/bin/bash
#SBATCH -p scc-cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=elastic
#SBATCH --output=elastic_%j.out
#SBATCH --error=elastic_%j.err

set -euo pipefail

NODES="${1:-4}"
TASKS_PER_NODE="${2:-2}"
CPUS_PER_TASK="${3:-8}"
GEN="${4:-0}"
EXCLUSIVE="${5:-false}"

LABEL="${NODES}x${TASKS_PER_NODE}x${CPUS_PER_TASK}_${SLURM_JOB_ID}"
JOB_DIR="${FWMP_BASE_OUTPUT_DIR}/job_${SLURM_JOB_ID}"

module purge
module load gcc
module load openmpi
module load python

cd ~/fwmp
source .venv/bin/activate
cd src

# -----------------------------
# threading
# -----------------------------
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=false

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export FWMP_DS="${FWMP_DS:-1}"

if [ -z "${FWMP_BASE_OUTPUT_DIR:-}" ]; then
    echo "ERROR: FWMP_BASE_OUTPUT_DIR not set"
    exit 1
fi

echo "----------------------------------------"
echo "job_id=$SLURM_JOB_ID"
echo "label=$LABEL"
echo "output=$FWMP_BASE_OUTPUT_DIR"
echo "FWMP_DS=$FWMP_DS"
echo "----------------------------------------"

mkdir -p "$JOB_DIR"
export FWMP_BASE_OUTPUT_DIR="$JOB_DIR"

srun --distribution=block:block --cpu-bind=cores \
    python -u elastic_lz4.py

rm -rf "$JOB_DIR"
echo "cleanup done"
