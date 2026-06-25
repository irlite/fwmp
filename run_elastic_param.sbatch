#!/bin/bash
#SBATCH -p scc-cpu
#SBATCH -N 4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
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

module purge
module load gcc
module load openmpi
module load python
module load scorep

cd ~/fwmp
source .venv/bin/activate

cd src

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=false
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export SCOREP_ENABLE_TRACING=1
export SCOREP_PROFILING_MAX_CALLPATH_DEPTH=200
export SCOREP_EXPERIMENT_DIRECTORY="../scorep_results/${LABEL}"
export PYTHONUNBUFFERED=1

echo "job_id=$SLURM_JOB_ID"
echo "label=$LABEL"
echo "nodes=$NODES"
echo "tasks_per_node=$TASKS_PER_NODE"
echo "cpus_per_task=$CPUS_PER_TASK"
echo "exclusive=$EXCLUSIVE"

srun --distribution=block:block --cpu-bind=cores python -m scorep elastic.py

cd ..

SRC_DIR="output/job_${SLURM_JOB_ID}"

if [ -d "$SRC_DIR" ]; then
    rm -rf "$SRC_DIR"
fi
