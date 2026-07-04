#!/bin/bash
set -euo pipefail

GEN=5
WEAK_SCALING=false
EXCLUSIVE=false

BASE_CORES=4
BASE_DS=13
STRONG_DS=10

CONFIGS=(
  "1 1 32"
)


CONFIGSSS=(
  "1 1 1"
  "1 2 8"
  "1 4 8"
  "1 8 8"
  "1 12 8"

  "1 2 4"
  "1 4 2"
  "1 8 1"

  "1 1 64"
  "1 2 32"
  "1 4 8"
  "1 8 8"
  "1 16 4"
  "1 32 2"
  "1 64 1"
  "4 2 8"
  "8 2 8"
)

CONFIGSS=(
  "1 4 16"
  "2 2 8"
  "1 8 8"
  "1 12 8"
  "1 1 64"
  "1 2 32"
  "1 4 8"
  "1 2 8"
  "1 1 8"
  "4 2 8"
  "1 16 4"
  "1 1 1"
  "1 1 2"
  "8 2 8"
)

BASE_OUT="/user/utkarsh.pathak/u27935/fwmp/opt_fast_drive"
mkdir -p "${BASE_OUT}"
mkdir -p "${BASE_OUT}/gen${GEN}/logs"

PREV_JOB=""

for CFG in "${CONFIGS[@]}"; do
    read -r NODES TASKS_PER_NODE CPUS_PER_TASK <<< "$CFG"

    LABEL="${NODES}x${TASKS_PER_NODE}x${CPUS_PER_TASK}"
    TOTAL_CORES=$((NODES * TASKS_PER_NODE * CPUS_PER_TASK))

    # scaling
    if [ "$WEAK_SCALING" = true ]; then
        DS=$(python3 - <<EOF
import math
base_ds = $BASE_DS
cores = $TOTAL_CORES
print(max(1, round(base_ds * (1 / cores) ** (1/3))))
EOF
)
    else
        DS=$STRONG_DS
    fi

    FWMP_BASE_OUTPUT_DIR="${BASE_OUT}/gen${GEN}/output/${LABEL}"
    mkdir -p "${FWMP_BASE_OUTPUT_DIR}"

    echo "Submitting ${LABEL}"
    echo "  cores  = ${TOTAL_CORES}"
    echo "  DS     = ${DS}"

    SBATCH_ARGS=(
        -N "$NODES"
        --ntasks-per-node="$TASKS_PER_NODE"
        --cpus-per-task="$CPUS_PER_TASK"
        --output="${BASE_OUT}/gen${GEN}/logs/${LABEL}_%j.out"
        --error="${BASE_OUT}/gen${GEN}/logs/${LABEL}_%j.err"
    )

    # dependency logic
    if [ -n "$PREV_JOB" ]; then
        DEP_FLAG="--dependency=afterok:$PREV_JOB"
    else
        DEP_FLAG=""
    fi

    JOB_ID=$(sbatch --parsable \
        $DEP_FLAG \
        --export=ALL,FWMP_DS=$DS,FWMP_BASE_OUTPUT_DIR=$FWMP_BASE_OUTPUT_DIR,FWMP_SCOREP=1 \
        "${SBATCH_ARGS[@]}" \
        run_elastic_param_blosc.sbatch \
        "$NODES" \
        "$TASKS_PER_NODE" \
        "$CPUS_PER_TASK" \
        "$GEN"
    )

    echo "submitted ${LABEL}, job_id=${JOB_ID}"

    PREV_JOB="$JOB_ID"
done