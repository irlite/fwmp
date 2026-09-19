#!/bin/bash
#SBATCH -p scc-cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=07:00:00
#SBATCH --job-name=elastic_seq
#SBATCH --output=elastic_seq_%j.out
#SBATCH --error=elastic_seq_%j.err
set -euo pipefail

LABEL="seq_${SLURM_JOB_ID}"
FWMP_BASE_OUTPUT_DIR="/user/maxim.barnstorf/u27934/opt_fast_drive/gen91"
JOB_DIR="${FWMP_BASE_OUTPUT_DIR}/job_${SLURM_JOB_ID}"

module purge
module load gcc
module load python
cd ~/fwmp
source .venv/bin/activate
cd src

export FWMP_DS="${FWMP_DS:-1}"
export BLOSC_NTHREADS=1

echo "----------------------------------------"
echo "job_id=$SLURM_JOB_ID"
echo "label=$LABEL"
echo "output=$JOB_DIR"
echo "FWMP_DS=$FWMP_DS"
echo "----------------------------------------"

mkdir -p "$JOB_DIR"
export FWMP_BASE_OUTPUT_DIR="$JOB_DIR"

/usr/bin/time -v python -u elastic_lz4_seq.py

rm -rf "$JOB_DIR"
echo "cleanup done"
