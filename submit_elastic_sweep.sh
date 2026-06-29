#!/bin/bash

set -euo pipefail

GEN=4
EXCLUSIVE=false

CONFIGS=(
  "1 1 8"
  "1 8 12"
  "2 2 8"
)

mkdir -p "previous_runs/gen${GEN}"

DEPENDENCY=""

for CFG in "${CONFIGS[@]}"; do
    read -r NODES TASKS_PER_NODE CPUS_PER_TASK <<< "$CFG"

    LABEL="${NODES}x${TASKS_PER_NODE}x${CPUS_PER_TASK}"

    SBATCH_ARGS=(
        -N "$NODES"
        --ntasks-per-node="$TASKS_PER_NODE"
        --cpus-per-task="$CPUS_PER_TASK"
        --output="previous_runs/gen${GEN}/${LABEL}_%j.out"
    )

    if [ "$EXCLUSIVE" = true ]; then
        SBATCH_ARGS+=(--exclusive)
    fi

    if [ -n "$DEPENDENCY" ]; then
        SBATCH_ARGS+=(--dependency="afterany:${DEPENDENCY}")
    fi

    SUBMIT_OUTPUT=$(sbatch "${SBATCH_ARGS[@]}" run_elastic_param.sbatch "$NODES" "$TASKS_PER_NODE" "$CPUS_PER_TASK" "$GEN" "$EXCLUSIVE")
    JOB_ID=$(echo "$SUBMIT_OUTPUT" | awk '{print $4}')

    echo "submitted ${LABEL}, job_id=${JOB_ID}"

    if [ "$EXCLUSIVE" = false ]; then
        DEPENDENCY="$JOB_ID"
    fi
done