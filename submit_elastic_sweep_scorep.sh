#!/bin/bash

set -euo pipefail

GEN=3
WEAK_SCALING=false
EXCLUSIVE=false

BASE_CORES=4
BASE_DS=8
STRONG_DS=20

BASE_OUT="/user/maxim.barnstorf/u27934/.project/dir.project/maxim"
GEN_DIR="${BASE_OUT}/gen${GEN}"
mkdir -p "$GEN_DIR"

SCOREP_BASE_DIR="${BASE_OUT}/scorep/gen${GEN}"
mkdir -p "$SCOREP_BASE_DIR"

CONFIGS=(
  "1 1 1"
  "1 1 2"
  "1 4 8"
  "1 12 8"
  "1 1 8"
  "1 2 8"
  "2 2 8"
  "4 2 8"
  "8 2 8"
  "1 1 64"
  "1 2 32"
  "1 4 16"
  "1 8 8"
  "1 16 4"
  "1 32 2"
  "1 64 1"
)

PREV_JOB=""

for CFG in "${CONFIGS[@]}"; do
    read -r NODES TASKS_PER_NODE CPUS_PER_TASK <<< "$CFG"

    LABEL="${NODES}x${TASKS_PER_NODE}x${CPUS_PER_TASK}"
    TOTAL_CORES=$((NODES * TASKS_PER_NODE * CPUS_PER_TASK))

    if [ "$WEAK_SCALING" = true ]; then
        DS=$(python3 - <<EOF
import math
base_cores = $BASE_CORES
base_ds = $BASE_DS
cores = $TOTAL_CORES
print(max(1, round(base_ds * (base_cores / cores) ** (1/3))))
EOF
)
    else
        DS=$STRONG_DS
    fi

    FWMP_BASE_OUTPUT_DIR="${BASE_OUT}/tmp/gen${GEN}"
    SCOREP_EXPERIMENT_DIR="${SCOREP_BASE_DIR}"

    mkdir -p "$FWMP_BASE_OUTPUT_DIR"
    mkdir -p "$SCOREP_EXPERIMENT_DIR"

    echo "Submitting ${LABEL}"
    echo "  cores = ${TOTAL_CORES}"
    echo "  DS    = ${DS}"
    echo "  scorep= ${SCOREP_EXPERIMENT_DIR}"

    SBATCH_ARGS=(
        -N "$NODES"
        --ntasks-per-node="$TASKS_PER_NODE"
        --cpus-per-task="$CPUS_PER_TASK"
        --output="${BASE_OUT}/gen${GEN}/${LABEL}_%j.out"
        --error="${BASE_OUT}/gen${GEN}/${LABEL}_%j.err"
    )

    if [ "$EXCLUSIVE" = true ]; then
        SBATCH_ARGS+=(--exclusive)
    fi

    if [ -n "$PREV_JOB" ]; then
        DEP_FLAG="--dependency=afterok:$PREV_JOB"
    else
        DEP_FLAG=""
    fi

    JOB_ID=$(sbatch --parsable \
        $DEP_FLAG \
        --export=ALL,\
FWMP_DS=$DS,\
FWMP_BASE_OUTPUT_DIR=$FWMP_BASE_OUTPUT_DIR,\
SCOREP_EXPERIMENT_DIR=$SCOREP_EXPERIMENT_DIR,\
SCOREP_ENABLE_TRACING=1,\
SCOREP_ENABLE_PROFILING=1,\
SCOREP_TOTAL_MEMORY=256M,\
SCOREP_MPI_ENABLE_GROUPS=ALL \
        "${SBATCH_ARGS[@]}" \
        run_elastic_param_scorep.sbatch \
        "$NODES" \
        "$TASKS_PER_NODE" \
        "$CPUS_PER_TASK" \
        "$GEN" \
        "$EXCLUSIVE"
    )

    echo "submitted ${LABEL}, job_id=${JOB_ID}"

    PREV_JOB="$JOB_ID"
done
