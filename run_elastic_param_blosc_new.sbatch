#!/bin/bash
#SBATCH -p scc-cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --job-name=elastic
#SBATCH --output=elastic_%j.out
#SBATCH --error=elastic_%j.err
#SBATCH --exclude=c0011

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
module load scorep
module load python

cd ~/fwmp
source .venv/bin/activate
cd src

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=false
export OMP_WAIT_POLICY=active

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export BLOSC_NTHREADS="$SLURM_CPUS_PER_TASK"

if [ -z "${FWMP_BASE_OUTPUT_DIR:-}" ]; then
    echo "ERROR: FWMP_BASE_OUTPUT_DIR not set"
    exit 1
fi

export FWMP_SCALE_MODE="${FWMP_SCALE_MODE:-none}"
export FWMP_BASE_CORES="${FWMP_BASE_CORES:-1}"
export FWMP_BASE_DS="${FWMP_BASE_DS:-39}"
export FWMP_TOTAL_CORES="${FWMP_TOTAL_CORES:-$((NODES * TASKS_PER_NODE * CPUS_PER_TASK))}"
export FWMP_DS="${FWMP_DS:-1}"

export FWMP_SCOREP="${FWMP_SCOREP:-0}"

SCOREP_DIR="${FWMP_BASE_OUTPUT_DIR}/scorep_${LABEL}"

RUN_CMD=(python -u elastic_tiled.py)

if [ "$FWMP_SCOREP" = "1" ]; then
    mkdir -p "$SCOREP_DIR"

    export SCOREP_ENABLE_TRACING=true
    export SCOREP_ENABLE_PROFILING=false
    export SCOREP_EXPERIMENT_DIRECTORY="$SCOREP_DIR"
    export SCOREP_TOTAL_MEMORY=4G

    export SCOREP_METRIC_PLUGINS=resource_usage_plugin
    export SCOREP_METRIC_RESOURCE_USAGE_PLUGIN=VmRSS,VmHWM,VmPeak

    export SCOREP_MPI_ENABLE_GROUPS=ALL
    export SCOREP_OPENMP_ENABLE=true

    RUN_CMD=(python -m scorep --mpp=mpi --thread=omp elastic_tiled.py)
fi

echo "----------------------------------------"
echo "job_id=$SLURM_JOB_ID"
echo "label=$LABEL"
echo "output=$FWMP_BASE_OUTPUT_DIR"
echo "scale_mode=$FWMP_SCALE_MODE"
echo "base_cores=$FWMP_BASE_CORES"
echo "base_ds=$FWMP_BASE_DS"
echo "total_cores=$FWMP_TOTAL_CORES"
echo "scorep=$FWMP_SCOREP"
if [ "$FWMP_SCOREP" = "1" ]; then
    echo "scorep_dir=$SCOREP_DIR"
fi
echo "----------------------------------------"

mkdir -p "$JOB_DIR"
export FWMP_BASE_OUTPUT_DIR="$JOB_DIR"

srun --distribution=block:block --cpu-bind=cores \
    /usr/bin/time -v "${RUN_CMD[@]}"

if [ "$FWMP_SCOREP" = "1" ]; then
    echo "----------------------------------------"
    echo "Score-P trace: $SCOREP_DIR/traces.otf2"

    if command -v otf2-print >/dev/null 2>&1; then
        otf2-print "$SCOREP_DIR/traces.otf2" 2>/dev/null \
            | grep VmHWM | awk '{print $NF}' | sort -n | tail -1 \
            | xargs -I{} echo "max VmHWM (bytes) = {}"
    else
        echo "otf2-print not available"
    fi

    echo "----------------------------------------"
fi

rm -rf "$JOB_DIR"
echo "cleanup done"
