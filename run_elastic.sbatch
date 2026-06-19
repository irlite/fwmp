#!/bin/bash
#SBATCH -p scc-cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --job-name=elastic
#SBATCH --output=elastic_%j.out
#SBATCH --error=elastic_%j.err

set -e

module purge
module load gcc
module load openmpi
module load python

cd ~/fwmp
source .venv/bin/activate
cd src

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export SCOREP_ENABLE_TRACING=1

export SCOREP_EXPERIMENT_DIRECTORY=scorep_trace_run1

srun --distribution=block:block --cpu-bind=cores python -m scorep -u elastic.py
