#!/bin/bash
#SBATCH -p scc-cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node64
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --job-name=elastic
#SBATCH --output=logs/elastic_1_16_4_%j.out
##SBATCH --error=logs/elastic_%j.err

#set -e

module purge
module load gcc
module load openmpi
module load python
module load scorep

source .venv/bin/activate
pip install mpi4py
cd ../src
# gcc -shared -fPIC -O3 elastic_kernels.c -o libelastic_kernels.so
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export SCOREP_ENABLE_TRACING=1
export SCOREP_PROFILING_MAX_CALLPATH_DEPTH=150

export SCOREP_EXPERIMENT_DIRECTORY="../scorep_results/1_64_1"
export PYTHONUNBUFFERED=1
#srun --distribution=block:block --cpu-bind=cores python -m scorep -u elastic.py
# srun --distribution=block:block --cpu-bind=cores --export=ALL pytho188n -m scorep  --mpp=mpi --thread=omp elastic.py
srun --distribution=block:block --cpu-bind=cores --export=ALL \
  bash -c 'echo $LD_LIBRARY_PATH; python -m scorep --mpp=mpi --thread=omp elastic.py'


SRC_DIR="/user/utkarsh.pathak/u27935/.project/dir.project/utkarsh/output/job_${SLURM_JOB_ID}"

if [ -d "$SRC_DIR" ]; then
    rm -rf "$SRC_DIR"
fi