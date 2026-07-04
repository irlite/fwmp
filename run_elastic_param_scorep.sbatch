#!/bin/bash
#SBATCH -p scc-cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --job-name=elastic_scorep
#SBATCH --output=elastic_scorep_%j.out
#SBATCH --error=elastic_scorep_%j.err

set -euo pipefail

NODES="${1:-4}"
TASKS_PER_NODE="${2:-2}"
CPUS_PER_TASK="${3:-8}"
GEN="${4:-0}"
EXCLUSIVE="${5:-false}"

JOB_DIR="${FWMP_BASE_OUTPUT_DIR}/job_${SLURM_JOB_ID}"
LABEL="${NODES}x${TASKS_PER_NODE}x${CPUS_PER_TASK}_${SLURM_JOB_ID}"

module purge
module load gcc
module load openmpi
module load python
module load scorep

cd ~/fwmp
source .venv/bin/activate
cd src

# -----------------------------
# OpenMP / threading control
# -----------------------------
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=false

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# -----------------------------
# FWMP config
# -----------------------------
export FWMP_DS="${FWMP_DS:-1}"

# -----------------------------
# Score-P output directory
# -----------------------------
export SCOREP_EXPERIMENT_DIRECTORY="${FWMP_BASE_OUTPUT_DIR}/scorep/job_${SLURM_JOB_ID}"
mkdir -p "$SCOREP_EXPERIMENT_DIRECTORY"

export SCOREP_ENABLE_PROFILING=1
export SCOREP_ENABLE_TRACING=1
export SCOREP_TOTAL_MEMORY=256M
export SCOREP_MPI_ENABLE_GROUPS=ALL

echo "----------------------------------------"
echo "job_id=$SLURM_JOB_ID"
echo "label=$LABEL"
echo "FWMP_DS=$FWMP_DS"
echo "output=$FWMP_BASE_OUTPUT_DIR"
echo "scorep=$SCOREP_EXPERIMENT_DIRECTORY"
echo "----------------------------------------"

# -----------------------------
# job working directory
# -----------------------------
mkdir -p "$JOB_DIR"
export FWMP_BASE_OUTPUT_DIR="$JOB_DIR"

# -----------------------------
# RUN (IMPORTANT PART)
# -----------------------------
srun --distribution=block:block --cpu-bind=cores \
    python -m scorep --mpp=mpi --thread=omp elastic.py

# -----------------------------
# cleanup
# -----------------------------
rm -rf "$JOB_DIR"
