#!/bin/bash
#SBATCH -p scc-cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --job-name=elastic_seq
#SBATCH --output=elastic_seq_%j.out
#SBATCH --error=elastic_seq_%j.err

set -e

module purge
module load gcc
module load python

cd ~/fwmp
source .venv/bin/activate
cd src

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=false
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -u elastic_seq.py
