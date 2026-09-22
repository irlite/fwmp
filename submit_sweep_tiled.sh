#!/bin/bash
#
# Submit a scaling sweep of the tiled kernel, then collect the results.
#
#   ./submit_sweep_tiled.sh              submit every config below
#   ./submit_sweep_tiled.sh --dry-run    print what would be submitted
#   ./submit_sweep_tiled.sh --collect DIR
#
# FWMP_SCOREP=1 runs the sweep instrumented.
# FWMP_TILE_ROWS pins the tile height; one sweep uses one value.
#
# Each configuration is NODES TASKS_PER_NODE CPUS_PER_TASK, and every job runs
# the same problem for the same number of steps, so the only thing changing is
# how the work is divided.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# What every job in the sweep runs. Keep --niter high enough that startup does
# not dominate, and --frame-stride high so the timing measures compute rather
# than I/O.
# elastic_tiled.py is env-driven, not argparse, so these go through --export.
# One frame keeps I/O out of the measurement.
SWEEP_NITER="${FWMP_NITER:-5000}"
SWEEP_FRAME_STRIDE="${FWMP_FRAME_STRIDE:-5000}"

# Cores per node, used to reject configurations that cannot be scheduled.
# Detected from the partition if sinfo can tell us, override if it guesses
# wrong.
PARTITION="${FWMP_PARTITION:-scc-cpu}"
CORES_PER_NODE="${FWMP_CORES_PER_NODE:-0}"

CONFIGS=(
    # Fixed 64 cores on one node: where is the best split between ranks and
    # threads? This is the most useful group, and usually the most surprising.
    "1  1 64"
    "1  2 32"
    "1  4 16"
    "1  8  8"
    "1 16  4"
    "1 32  2"
    "1 64  1"

    # One rank, growing threads: pure OpenMP scaling, no MPI in the picture.
    "1  1  1"
    "1  1  2"
    "1  1  4"
    "1  1  8"
    "1  1 16"
    "1  1 32"

    # One thread per rank, growing ranks: pure MPI scaling on one node.
    "1  2  1"
    "1  4  1"
    "1  8  1"
    "1 16  1"
    "1 32  1"

    # Across nodes at 2 ranks x 8 threads each: does it keep scaling once the
    # halo exchange has to cross the network?
    "1  2  8"
    "2  2  8"
    "4  2  8"
    "8  2  8"

    # Half-loaded node, from your original list.
    "1  4  8"
)

# ---------------------------------------------------------------------------
# Collect mode
# ---------------------------------------------------------------------------

if [ "${1:-}" = "--collect" ]; then
    SWEEP_DIR="${2:-}"
    if [ -z "$SWEEP_DIR" ] || [ ! -d "$SWEEP_DIR" ]; then
        echo "usage: $0 --collect <sweep directory>" >&2
        exit 1
    fi
    # Strip every trailing slash, so labels and the results path stay clean
    # when the directory is tab-completed.
    while [ "${SWEEP_DIR%/}" != "$SWEEP_DIR" ]; do
        SWEEP_DIR="${SWEEP_DIR%/}"
    done

    RESULTS="$SWEEP_DIR/results.csv"
    : > "$RESULTS"

    # Find results at any depth, so this works whether you point it at one
    # sweep or at the parent holding several. The label is the path relative to
    # wherever you pointed it, which keeps rows from different sweeps apart.
    found=0
    while IFS= read -r TIMING; do
        LABEL="${TIMING#"$SWEEP_DIR"/}"
        LABEL="${LABEL%/timing.csv}"

        if [ "$found" -eq 0 ]; then
            echo "label,$(head -1 "$TIMING")" >> "$RESULTS"
        fi
        echo "$LABEL,$(tail -1 "$TIMING")" >> "$RESULTS"
        found=$((found + 1))
    done < <(find "$SWEEP_DIR" -type f -name timing.csv | sort)

    # A run directory is named like its configuration. One without a
    # timing.csv is still queued, or it failed.
    missing=0
    while IFS= read -r RUN_DIR; do
        [ -f "$RUN_DIR/timing.csv" ] && continue
        echo "no result yet: ${RUN_DIR#"$SWEEP_DIR"/}" >&2
        missing=$((missing + 1))
    done < <(find "$SWEEP_DIR" -mindepth 1 -type d -name "[0-9]*x[0-9]*x[0-9]*" | sort)

    if [ "$found" -eq 0 ]; then
        rm -f "$RESULTS"
        echo "no finished runs under $SWEEP_DIR" >&2
        exit 1
    fi

    echo "collected $found run(s) into $RESULTS"
    [ "$missing" -gt 0 ] && echo "$missing run(s) still pending or failed" >&2
    exit 0
fi

# ---------------------------------------------------------------------------
# Submit mode
# ---------------------------------------------------------------------------

DRY_RUN="${1:-}"

if [ -n "$DRY_RUN" ] && [ "$DRY_RUN" != "--dry-run" ]; then
    echo "usage: $0 [--dry-run]" >&2
    echo "       $0 --collect <sweep directory>" >&2
    exit 1
fi

# Read DEFAULT_WORK_DIR out of fwmp_tiled.sbatch so there is one place to set it.
DEFAULT_WORK_DIR="$(sed -n 's/^DEFAULT_WORK_DIR="\(.*\)"$/\1/p' fwmp_tiled.sbatch | head -1)"
USE_SCOREP="${FWMP_SCOREP:-0}"
TILE_ROWS="${FWMP_TILE_ROWS:-64}"

WORK_DIR="${FWMP_WORK_DIR:-${DEFAULT_WORK_DIR:-$PROJECT_ROOT}}"
SWEEP_ID="$(date +%Y%m%d_%H%M%S)_tiled${TILE_ROWS}"
SWEEP_DIR="$WORK_DIR/sweeps_tiled/sweep_$SWEEP_ID"

# Ask SLURM how big a node is, rather than hardcoding it.
if [ "$CORES_PER_NODE" -eq 0 ]; then
    CORES_PER_NODE="$(sinfo -h -p "$PARTITION" -o "%c" 2>/dev/null | sort -rn | head -1 || true)"
    CORES_PER_NODE="${CORES_PER_NODE:-0}"
fi

if [ "$CORES_PER_NODE" -eq 0 ]; then
    echo "could not work out cores per node for partition $PARTITION."
    echo "Set FWMP_CORES_PER_NODE to skip the check."
else
    echo "partition $PARTITION has $CORES_PER_NODE cores per node"
fi

echo "sweep    $SWEEP_DIR"
echo "tile     $TILE_ROWS rows"
echo "scorep   $USE_SCOREP"
echo "niter    $SWEEP_NITER (stride $SWEEP_FRAME_STRIDE)"
echo

if [ "$USE_SCOREP" != "0" ]; then
    echo "Score-P is on. It slows every run and writes a trace per job, so the"
    echo "timings below are not comparable with an uninstrumented sweep. Cut"
    echo "CONFIGS down to the few you actually want to profile."
    echo
fi

# Every job needs the kernel already built. Check once here rather than
# submitting two dozen jobs that each fail the same way.
if [ "$USE_SCOREP" != "0" ]; then
    NEEDED_KERNEL="src/libelastic_kernels_tiled_scorep.so"
    BUILD_HINT="sbatch --export=ALL,FWMP_BUILD=1,FWMP_SCOREP=1 fwmp_tiled.sbatch"
else
    NEEDED_KERNEL="src/libelastic_kernels_tiled.so"
    BUILD_HINT="sbatch --export=ALL,FWMP_BUILD=1 fwmp_tiled.sbatch"
fi

if [ ! -f "$NEEDED_KERNEL" ]; then
    echo "missing $NEEDED_KERNEL" >&2
    echo "Build it first:  $BUILD_HINT" >&2
    exit 1
fi

mkdir -p "$SWEEP_DIR"

submitted=0
skipped=0
seen_labels=""

for CONFIG in "${CONFIGS[@]}"; do
    read -r NODES TASKS_PER_NODE CPUS_PER_TASK <<< "$CONFIG"

    LABEL="${NODES}x${TASKS_PER_NODE}x${CPUS_PER_TASK}"

    # The same config listed twice would submit two jobs writing to one
    # directory, and the second would overwrite the first.
    case " $seen_labels " in
        *" $LABEL "*)
            echo "skip $LABEL: duplicate entry in CONFIGS"
            skipped=$((skipped + 1))
            continue
            ;;
    esac
    seen_labels="$seen_labels $LABEL"

    CORES_NEEDED=$((TASKS_PER_NODE * CPUS_PER_TASK))
    if [ "$CORES_PER_NODE" -gt 0 ] && [ "$CORES_NEEDED" -gt "$CORES_PER_NODE" ]; then
        echo "skip $LABEL: needs $CORES_NEEDED cores per node, partition has $CORES_PER_NODE"
        skipped=$((skipped + 1))
        continue
    fi

    RUN_DIR="$SWEEP_DIR/$LABEL"

    if [ "$DRY_RUN" = "--dry-run" ]; then
        echo "would submit $LABEL -> $RUN_DIR"
        submitted=$((submitted + 1))
        continue
    fi

    mkdir -p "$RUN_DIR"

    JOB=$(sbatch --parsable \
        --nodes="$NODES" \
        --ntasks-per-node="$TASKS_PER_NODE" \
        --cpus-per-task="$CPUS_PER_TASK" \
        --job-name="tiled_$LABEL" \
        --output="$RUN_DIR/slurm.out" \
        --error="$RUN_DIR/slurm.err" \
        --export="ALL,FWMP_OUTPUT_DIR=$RUN_DIR,FWMP_TILE_ROWS=$TILE_ROWS,FWMP_SCOREP=$USE_SCOREP,FWMP_NITER=$SWEEP_NITER,FWMP_FRAME_STRIDE=$SWEEP_FRAME_STRIDE" \
        fwmp_tiled.sbatch)

    echo "submitted $LABEL as job $JOB"
    submitted=$((submitted + 1))
done

echo
echo "$submitted job(s) submitted, $skipped skipped"
echo
echo "when they finish:"
echo "  $0 --collect $SWEEP_DIR"
