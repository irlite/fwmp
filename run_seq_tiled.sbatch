#!/bin/bash
#SBATCH -p scc-cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --job-name=elastic_seq_tiled
#SBATCH --output=elastic_seq_tiled_%j.out
#SBATCH --error=elastic_seq_tiled_%j.err
#SBATCH --exclude=c0011

set -euo pipefail

PROJECT_ROOT="$HOME/fwmp"
OUTPUT_ROOT="$PROJECT_ROOT/opt_fast_drive/seq_tiled"

export FWMP_NITER=50000
export FWMP_FRAME_STRIDE=100
export FWMP_DIRECT_VZ_SOURCE=0
export FWMP_DS=1
export FWMP_TILE_Z=8
export FWMP_TILE_X=256
export FWMP_BASE_OUTPUT_DIR="$OUTPUT_ROOT/job_${SLURM_JOB_ID}"

export OMP_NUM_THREADS=1
export OMP_DYNAMIC=false
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLOSC_NTHREADS=1

module purge
module load gcc
module load python

cd "$PROJECT_ROOT"
source .venv/bin/activate
cd src

LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"
mkdir -p "$FWMP_BASE_OUTPUT_DIR"

echo "job_id=$SLURM_JOB_ID"
echo "output=$FWMP_BASE_OUTPUT_DIR"
echo "n_iterations=$FWMP_NITER"
echo "frame_stride=$FWMP_FRAME_STRIDE"
echo "direct_vz_source=$FWMP_DIRECT_VZ_SOURCE"
echo "ds=$FWMP_DS"
echo "tile_z=$FWMP_TILE_Z"
echo "tile_x=$FWMP_TILE_X"

srun --ntasks=1 --cpus-per-task=1 --cpu-bind=cores \
    /usr/bin/time -v python -u elastic_lz4_seq_tiled.py

rm -rf "$FWMP_BASE_OUTPUT_DIR"

mv "elastic_seq_tiled_${SLURM_JOB_ID}.out" "$LOG_DIR/"
mv "elastic_seq_tiled_${SLURM_JOB_ID}.err" "$LOG_DIR/"

echo "logs saved to $LOG_DIR"
