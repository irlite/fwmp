#!/bin/bash
#SBATCH --partition=scc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=07:00:00
#SBATCH --job-name=elastic_seq_tiled
#SBATCH --output=/mnt/vast-standard/home/maxim.barnstorf/u27934/fwmp/opt_fast_drive/seq_tiled/logs/elastic_seq_tiled_%j.out
#SBATCH --error=/mnt/vast-standard/home/maxim.barnstorf/u27934/fwmp/opt_fast_drive/seq_tiled/logs/elastic_seq_tiled_%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT="$HOME/fwmp"
OUTPUT_ROOT="$PROJECT_ROOT/opt_fast_drive/seq_tiled"
LOG_DIR="$OUTPUT_ROOT/logs"
JOB_OUTPUT_DIR="$OUTPUT_ROOT/job_${SLURM_JOB_ID}"

# ---------------------------------------------------------------------------
# FWMP configuration
# ---------------------------------------------------------------------------

export FWMP_NITER=50000
export FWMP_FRAME_STRIDE=100
export FWMP_DIRECT_VZ_SOURCE=0
export FWMP_DS=1
export FWMP_TILE_Z=8
export FWMP_TILE_X=256
export FWMP_BASE_OUTPUT_DIR="$JOB_OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Thread limits
# ---------------------------------------------------------------------------

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_DYNAMIC=false
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export BLOSC_NTHREADS="$SLURM_CPUS_PER_TASK"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

module purge
module load gcc
module load python

source "$PROJECT_ROOT/.venv/bin/activate"

mkdir -p "$JOB_OUTPUT_DIR"

cd "$PROJECT_ROOT/src"

# ---------------------------------------------------------------------------
# Job information
# ---------------------------------------------------------------------------

echo "job_id=$SLURM_JOB_ID"
echo "hostname=$(hostname)"
echo "project_root=$PROJECT_ROOT"
echo "output=$FWMP_BASE_OUTPUT_DIR"
echo "log_dir=$LOG_DIR"
echo "n_iterations=$FWMP_NITER"
echo "frame_stride=$FWMP_FRAME_STRIDE"
echo "direct_vz_source=$FWMP_DIRECT_VZ_SOURCE"
echo "ds=$FWMP_DS"
echo "tile_z=$FWMP_TILE_Z"
echo "tile_x=$FWMP_TILE_X"
echo "cpus_per_task=$SLURM_CPUS_PER_TASK"
echo "started=$(date --iso-8601=seconds)"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

srun \
    --ntasks=1 \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    --cpu-bind=cores \
    /usr/bin/time -v \
    python -u elastic_lz4_seq_tiled.py

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

rm -rf -- "$JOB_OUTPUT_DIR"

echo "finished=$(date --iso-8601=seconds)"
echo "temporary output removed: $JOB_OUTPUT_DIR"
echo "logs written directly to: $LOG_DIR"
