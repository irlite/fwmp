#!/bin/bash

set -euo pipefail

GEN=2
EXCLUSIVE=false
WEAK_SCALING=true

CONFIGS=(
  "1 2 7"
  "1 8 7"
)

mkdir -p "previous_runs/gen${GEN}"

DEPENDENCY=""

BASE_CORES=1
BASE_DS=23

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

ds = round(base_ds * math.sqrt(base_cores / cores))
print(max(1, ds))
EOF
)
    else
        DS=1
    fi

    echo "Submitting ${LABEL}"
    echo "  total cores = ${TOTAL_CORES}"
    echo "  FWMP_DS     = ${DS}"

    SBATCH_ARGS=(
        -N "$NODES"
        --ntasks-per-node="$TASKS_PER_NODE"
        --cpus-per-task="$CPUS_PER_TASK"
        --output="previous_runs/gen${GEN}/${LABEL}_%j.out"
        --error="previous_runs/gen${GEN}/${LABEL}_%j.err"
    )

    if [ "$EXCLUSIVE" = true ]; then
        SBATCH_ARGS+=(--exclusive)
    fi

    if [ -n "$DEPENDENCY" ]; then
        SBATCH_ARGS+=(--dependency="afterany:${DEPENDENCY}")
    fi

    SUBMIT_OUTPUT=$(
        sbatch \
            --export=ALL,FWMP_DS=$DS \
            "${SBATCH_ARGS[@]}" \
            run_elastic_param.sbatch \
            "$NODES" \
            "$TASKS_PER_NODE" \
            "$CPUS_PER_TASK" \
            "$GEN" \
            "$EXCLUSIVE"
    )

    JOB_ID=$(echo "$SUBMIT_OUTPUT" | awk '{print $4}')

    echo "submitted ${LABEL}, job_id=${JOB_ID}"

    if [ "$EXCLUSIVE" = false ]; then
        DEPENDENCY="$JOB_ID"
    fi
done
