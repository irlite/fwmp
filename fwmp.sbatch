#!/bin/bash
#
# Build and run the elastic wave simulation under SLURM.
#
#   sbatch fwmp.sbatch
#   sbatch fwmp.sbatch --niter 2000 --compression blosc
#   sbatch -N 4 --ntasks-per-node=2 fwmp.sbatch --stage-local --write-mode async
#
#   FWMP_WORK_DIR=/scratch/$USER/fwmp   where output goes, default is here
#   FWMP_SETUP=1    build .venv, install dependencies, compile the kernels
#   FWMP_BUILD=1    compile the kernels and exit, nothing else
#   FWMP_SCOREP=1   run under Score-P, traces land next to the output
#   FWMP_KERNEL=v3  which kernel optimisation level to run, v1 to v4
#   FWMP_BLOCK_ROWS, FWMP_BLOCK_COLUMNS   tile shape for v3 and v4
#


#SBATCH --partition=scc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=fwmp
#SBATCH --output=fwmp_%j.out
#SBATCH --error=fwmp_%j.err

set -euo pipefail

# Under sbatch, $0 points at a spool copy, so prefer the submit directory.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$PROJECT_ROOT"

JOB_ID="${SLURM_JOB_ID:-noslurm}"


# DEFAULT_WORK_DIR=""
DEFAULT_WORK_DIR="/user/utkarsh.pathak/u27935/.project/dir.project/utkarsh/september"

WORK_DIR="${FWMP_WORK_DIR:-${DEFAULT_WORK_DIR:-$PROJECT_ROOT}}"
OUTPUT_DIRECTORY="${FWMP_OUTPUT_DIR:-$WORK_DIR/output/job_$JOB_ID}"

if [ "$WORK_DIR" = "$PROJECT_ROOT" ]; then
    echo "note: writing output next to the code. Set DEFAULT_WORK_DIR in this" \
         "script, or pass --export=ALL,FWMP_WORK_DIR=... to sbatch."
fi
USE_SCOREP="${FWMP_SCOREP:-0}"

module purge
module load gcc openmpi python
if [ "$USE_SCOREP" != "0" ]; then
    module load scorep
fi

# ---------------------------------------------------------------------------
# Building the kernels
#
# Runs with FWMP_BUILD=1 or as part of FWMP_SETUP=1, and never otherwise. Jobs
# just use whatever is already there, so a sweep does not recompile once per
# job or race to write the same file.
# ---------------------------------------------------------------------------

build_kernels() {
    local compiler suffix source level library
    local block_rows="${FWMP_BLOCK_ROWS:-32}"
    local block_columns="${FWMP_BLOCK_COLUMNS:-512}"

    if [ "$USE_SCOREP" != "0" ]; then
        # Score-P has to wrap the compiler to see inside the kernels at all.
        compiler="scorep-gcc --openmp"
        suffix="_scorep"
    else
        compiler="gcc"
        suffix=""
    fi

    for source in elastic_kernels_v*.c; do
        # elastic_kernels_v3_blocked.c -> v3
        level="${source#elastic_kernels_}"
        level="${level%%_*}"
        library="libelastic_kernels_${level}${suffix}.so"

        echo "compiling $source -> $library"
        $compiler -O3 -fopenmp -march=native -fPIC -shared \
            -DBLOCK_ROWS="$block_rows" -DBLOCK_COLUMNS="$block_columns" \
            -o "$library" "$source"
    done
}

if [ "${FWMP_BUILD:-0}" != "0" ]; then
    build_kernels
    echo "done. Kernels are in $PROJECT_ROOT"
    exit 0
fi

# ---------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------

if [ "${FWMP_SETUP:-0}" != "0" ]; then
    echo "building .venv"
    # --system-site-packages so a cluster-provided mpi4py, already built
    # against the local MPI, is visible from inside the venv.
    python -m venv --system-site-packages .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip

    # mpi4py is the one package that has to match the MPI srun uses, and it is
    # the one that breaks. Recent pip wheels are prebuilt against a generic ABI
    # and dlopen libmpi.so at import time, which fails the moment the cluster's
    # MPI is not on the loader path. So: keep whatever the system already
    # provides if it imports, and otherwise compile against the module's mpicc
    # instead of downloading a wheel that will not match.
    if python -c "from mpi4py import MPI" 2>/dev/null; then
        echo "mpi4py already works, leaving it alone"
    else
        if ! command -v mpicc >/dev/null; then
            echo "no mpicc on PATH. Load an MPI module that provides one." >&2
            exit 1
        fi
        echo "compiling mpi4py against $(command -v mpicc)"
        MPICC="$(command -v mpicc)" python -m pip install \
            --no-binary=mpi4py --no-cache-dir mpi4py
    fi

    python -m pip install numpy h5py segyio hdf5plugin

    # Prove it works now, rather than finding out from every rank of a queued
    # job an hour later.
    python -c "from mpi4py import MPI; print('mpi4py ok:', MPI.Get_library_version().splitlines()[0])"

    build_kernels
else
    if [ ! -f .venv/bin/activate ]; then
        echo "no .venv here. Run once with FWMP_SETUP=1 first." >&2
        exit 1
    fi
    source .venv/bin/activate
fi

# ---------------------------------------------------------------------------
# C kernels
# ---------------------------------------------------------------------------

# Which optimisation level this job runs. FWMP_KERNEL=v3 to compare.
KERNEL_LEVEL="${FWMP_KERNEL:-v2}"

# Instrumented and plain builds get separate names so both can sit there at
# once and you never have to rebuild when switching between them.
if [ "$USE_SCOREP" != "0" ]; then
    KERNEL_SUFFIX="_scorep"
else
    KERNEL_SUFFIX=""
fi

SELECTED_KERNEL="$PROJECT_ROOT/libelastic_kernels_${KERNEL_LEVEL}${KERNEL_SUFFIX}.so"

if [ ! -f "$SELECTED_KERNEL" ]; then
    echo "missing $SELECTED_KERNEL" >&2
    if [ "$USE_SCOREP" != "0" ]; then
        echo "Build the instrumented kernels: FWMP_SCOREP=1 FWMP_BUILD=1 sbatch fwmp.sbatch" >&2
    else
        echo "Build the kernels: FWMP_BUILD=1 sbatch fwmp.sbatch" >&2
    fi
    exit 1
fi

# ---------------------------------------------------------------------------
# Threading
# ---------------------------------------------------------------------------

# One OpenMP team per rank, filling that rank's cores and staying put. Without
# the binding, threads drift across sockets and the timings turn to noise.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=false

# NumPy's backends would otherwise start their own threads and fight the kernels
# for cores. There is no heavy linear algebra here, so one thread each is fine.
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

# The user's own --output-dir, if they passed one, comes later in "$@" and wins.
LAUNCH_COMMAND=(
    python -u fwmp.py
    --output-dir "$OUTPUT_DIRECTORY"
    --kernel-lib "$SELECTED_KERNEL"
    "$@"
)

if [ "$USE_SCOREP" != "0" ]; then
    export SCOREP_EXPERIMENT_DIRECTORY="$OUTPUT_DIRECTORY/scorep"
    export SCOREP_ENABLE_PROFILING=1
    export SCOREP_ENABLE_TRACING=1
    export SCOREP_TOTAL_MEMORY=256M
    export SCOREP_MPI_ENABLE_GROUPS=ALL
    mkdir -p "$SCOREP_EXPERIMENT_DIRECTORY"

    LAUNCH_COMMAND=(
        python -u -m scorep --mpp=mpi --thread=omp fwmp.py
        --output-dir "$OUTPUT_DIRECTORY"
        --kernel-lib "$SELECTED_KERNEL"
        "$@"
    )
fi

echo "----------------------------------------"
echo "job      $JOB_ID"
echo "layout   ${SLURM_JOB_NUM_NODES:-1} node(s)" \
     "x ${SLURM_NTASKS_PER_NODE:-1} rank(s)" \
     "x ${SLURM_CPUS_PER_TASK:-1} thread(s)"
echo "kernel   $KERNEL_LEVEL"
echo "work     $WORK_DIR"
echo "output   $OUTPUT_DIRECTORY"
echo "args     $*"
echo "----------------------------------------"

srun --distribution=block:block --cpu-bind=cores "${LAUNCH_COMMAND[@]}"
