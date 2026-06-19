#!/bin/bash
#SBATCH -p scc-cpu
#SBATCH -N 1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=elastic-single-threaded
#SBATCH --output=logs/elastic-single-threaded.out
#SBATCH --error=logs/elastic-single-threaded.err

set -e

module purge
module load gcc
module load python
module load openmpi
module load scorep

source .venv/bin/activate
cd src
export SCOREP_ENABLE_TRACING=1
export SCOREP_EXPERIMENT_DIRECTORY=scorep_trace_run1
export LD_LIBRARY_PATH=/sw/rev/25.04/rome_mofed_cuda80_rocky8/linux-rocky8-zen/gcc-8.5.0/gcc-13.2.0-g6zzl2jiubdyluoqoc5tkcjiyxwabtsh/lib64:$LD_LIBRARY_PATH
python -m scorep -u elastic_single_threaded.py
