#!/bin/bash
#
# Run the tiled kernel (src/elastic_tiled.py), optionally under Score-P.
#
#   sbatch --export=ALL,FWMP_BUILD=1 fwmp_tiled.sbatch       compile, then exit
#   sbatch fwmp_tiled.sbatch                                 run
#   sbatch --export=ALL,FWMP_SCOREP=1 fwmp_tiled.sbatch      run under Score-P
#
# This is deliberately separate from fwmp.sbatch. The tiled kernel is not a
# fifth optimisation level of the same code: it exports three fused entry
# points instead of two and overlaps the halo exchange with computation, so it
# needs its own driver, which is src/elastic_tiled.py.
#
# Environment variables:
#   FWMP_BUILD=1    compile the tiled kernels and exit
#   FWMP_SCOREP=1   build with scorep-gcc, run under Score-P
#   FWMP_TILE_ROWS  tile height in rows, default 64
#   FWMP_NITER      steps, default 500 under Score-P, 50000 otherwise
#   FWMP_MARCH      compiler target, default native

#SBATCH --partition=scc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:15:00
#SBATCH --job-name=fwmp_tiled
#SBATCH --output=fwmp_tiled_%j.out
#SBATCH --error=fwmp_tiled_%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$PROJECT_ROOT"

SRC_DIR="$PROJECT_ROOT/src"
JOB_ID="${SLURM_JOB_ID:-noslurm}"
USE_SCOREP="${FWMP_SCOREP:-0}"

# Same setting as fwmp.sbatch. Edit it here, or export FWMP_WORK_DIR.
DEFAULT_WORK_DIR="/user/utkarsh.pathak/u27935/.project/dir.project/utkarsh/september"

WORK_DIR="${FWMP_WORK_DIR:-${DEFAULT_WORK_DIR:-$PROJECT_ROOT}}"
OUTPUT_DIRECTORY="${FWMP_OUTPUT_DIR:-$WORK_DIR/output_tiled/job_$JOB_ID}"

module purge
module load gcc
module load openmpi
module load python
if [ "$USE_SCOREP" != "0" ]; then
    module load scorep
fi

# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

# Instrumented and plain builds get separate names, so both can sit there and
# switching between them needs no rebuild. elastic_tiled.py picks one with
# FWMP_KERNEL_LIB, which needs the one-line change described in the README.
if [ "$USE_SCOREP" != "0" ]; then
    KERNEL_SUFFIX="_scorep"
else
    KERNEL_SUFFIX=""
fi

TILED_LIBRARY="libelastic_kernels_tiled${KERNEL_SUFFIX}.so"

if [ "${FWMP_BUILD:-0}" != "0" ]; then
    MARCH="${FWMP_MARCH:-native}"

    if [ "$USE_SCOREP" != "0" ]; then
        COMPILER="scorep-gcc --openmp"
    else
        COMPILER="gcc"
    fi

    echo "compiling src/elastic_kernels_tiled.c -> src/$TILED_LIBRARY"
    $COMPILER -O3 -fopenmp -march="$MARCH" -fPIC -shared \
        -o "$SRC_DIR/$TILED_LIBRARY" "$SRC_DIR/elastic_kernels_tiled.c"

    echo "done."
    exit 0
fi

if [ ! -f "$SRC_DIR/$TILED_LIBRARY" ]; then
    echo "missing $SRC_DIR/$TILED_LIBRARY" >&2
    if [ "$USE_SCOREP" != "0" ]; then
        echo "Build it: sbatch --export=ALL,FWMP_BUILD=1,FWMP_SCOREP=1 fwmp_tiled.sbatch" >&2
    else
        echo "Build it: sbatch --export=ALL,FWMP_BUILD=1 fwmp_tiled.sbatch" >&2
    fi
    exit 1
fi

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

if [ ! -f .venv/bin/activate ]; then
    echo "no .venv in $PROJECT_ROOT" >&2
    exit 1
fi
source .venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=false

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLOSC_NTHREADS=1

export FWMP_DS="${FWMP_DS:-1}"
export FWMP_DIRECT_VZ_SOURCE="${FWMP_DIRECT_VZ_SOURCE:-0}"
export FWMP_KERNEL_LIB="$TILED_LIBRARY"

# Tile height. Unset, the C picks 4 * OMP_NUM_THREADS clamped to 16..256, so it
# would change with every thread count and you could not tell tiling effects
# apart from thread scaling. Pin it for a sweep.
export FWMP_TILE_ROWS="${FWMP_TILE_ROWS:-64}"

# Instrumented runs emit a lot of region events per step, so keep them short by
# default. An uninstrumented run gets the production step count.
if [ "$USE_SCOREP" != "0" ]; then
    export FWMP_NITER="${FWMP_NITER:-500}"
    export FWMP_FRAME_STRIDE="${FWMP_FRAME_STRIDE:-500}"
else
    export FWMP_NITER="${FWMP_NITER:-50000}"
    export FWMP_FRAME_STRIDE="${FWMP_FRAME_STRIDE:-100}"
fi

mkdir -p "$OUTPUT_DIRECTORY"
export FWMP_BASE_OUTPUT_DIR="$OUTPUT_DIRECTORY"

LAUNCH=(python -u elastic_tiled.py)

if [ "$USE_SCOREP" != "0" ]; then
    export SCOREP_EXPERIMENT_DIRECTORY="$OUTPUT_DIRECTORY/scorep"
    export SCOREP_ENABLE_PROFILING="${SCOREP_ENABLE_PROFILING:-1}"
    export SCOREP_ENABLE_TRACING="${SCOREP_ENABLE_TRACING:-1}"
    export SCOREP_TOTAL_MEMORY="${SCOREP_TOTAL_MEMORY:-256M}"
    export SCOREP_MPI_ENABLE_GROUPS=ALL
    mkdir -p "$SCOREP_EXPERIMENT_DIRECTORY"

    LAUNCH=(python -u -m scorep --mpp=mpi --thread=omp elastic_tiled.py)
fi

NODES="${SLURM_JOB_NUM_NODES:-1}"
TASKS_PER_NODE="${SLURM_NTASKS_PER_NODE:-1}"
RANKS="${SLURM_NTASKS:-1}"

echo "----------------------------------------"
echo "job        $JOB_ID"
echo "layout     ${NODES}n x ${TASKS_PER_NODE}r x ${OMP_NUM_THREADS}t"
echo "kernel     $TILED_LIBRARY"
echo "tile rows  $FWMP_TILE_ROWS"
echo "niter      $FWMP_NITER (stride $FWMP_FRAME_STRIDE)"
echo "scorep     $USE_SCOREP"
echo "work       $WORK_DIR"
echo "output     $OUTPUT_DIRECTORY"
echo "----------------------------------------"

cd "$SRC_DIR"

# Wall clock around srun. elastic_tiled.py does not report its own timings, so
# this is the whole job including startup and I/O, NOT the time loop. Do not
# compare it against loop_seconds from fwmp.py.
START=$(date +%s.%N)
srun --distribution=block:block --cpu-bind=cores "${LAUNCH[@]}"
END=$(date +%s.%N)

WALL=$(awk "BEGIN {printf \"%.4f\", $END - $START}")

{
    echo "job_id,nodes,ranks,threads_per_rank,tile_rows,niter,frame_stride,downsample,scorep,wall_seconds"
    echo "$JOB_ID,$NODES,$RANKS,$OMP_NUM_THREADS,$FWMP_TILE_ROWS,$FWMP_NITER,$FWMP_FRAME_STRIDE,$FWMP_DS,$USE_SCOREP,$WALL"
} > "$OUTPUT_DIRECTORY/timing.csv"

echo "wall ${WALL}s, timing in $OUTPUT_DIRECTORY/timing.csv"
