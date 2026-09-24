#!/bin/bash
set -euo pipefail
GEN=1x0 #test #1x0
SCALE_MODE=strong #weak
BASE_CORES=1
MAX_CORES=640
CONFIGS=(
  "1 1 16"
  "1 1 16"
  "1 1 16"
  "1 1 16"
  "1 1 32"
  "1 1 32"
  "1 1 32"
  "1 1 32"
  "1 2 32"
  "1 2 32"
  "1 2 32"
  "1 2 32"
  "1 3 16"
  "1 3 16"
  "1 3 16"
  "1 3 16"
  "1 5 16"
  "1 5 16"
  "1 5 16"
  "1 5 16"
  "1 3 32"
  "1 3 32"
  "1 3 32"
  "1 3 32"
)
CONFIGSS=(
  "1 1 1"
  "1 1 16"
  "1 1 32"
  "1 2 32"
  "1 3 16"
  "1 5 16"
  "1 3 32"

  "2 3 32"
  "4 3 32"
  "8 3 32"
  "10 3 32"

  "1 2 32"
  "2 2 32"
  "4 2 32"
  "8 2 32"
  "10 2 32"

  "1 1 1"
  "1 1 16"
  "1 1 32"
  "1 2 32"
  "1 3 32"

  "2 2 32"
  "2 2 32"
  "2 2 32"
  "2 2 32"
  "4 2 32"
  "4 2 32"
  "4 2 32"
  "4 2 32"
  "8 2 32"
  "8 2 32"
  "8 2 32"
  "8 2 32"
  "10 2 32"
  "10 2 32"
  "10 2 32"
  "10 2 32"
)
BASE_OUT="/user/maxim.barnstorf/u27934/fwmp/opt_fast_drive"

base_ds_for_max_cores() {
    local max_cores=$1
    local root=0
    while (( root + 1 <= max_cores / (root + 1) )); do
        root=$((root + 1))
    done
    printf '%d\n' "$((root + 1))"
}

BASE_DS=$(base_ds_for_max_cores "$MAX_CORES")

mkdir -p "${BASE_OUT}"
mkdir -p "${BASE_OUT}/gen${GEN}/logs"
PREV_JOB=""
for CFG in "${CONFIGS[@]}"; do
    read -r NODES TASKS_PER_NODE CPUS_PER_TASK <<< "$CFG"
    LABEL="${NODES}x${TASKS_PER_NODE}x${CPUS_PER_TASK}"
    TOTAL_CORES=$((NODES * TASKS_PER_NODE * CPUS_PER_TASK))
    FWMP_BASE_OUTPUT_DIR="${BASE_OUT}/gen${GEN}/output/${LABEL}"
    mkdir -p "${FWMP_BASE_OUTPUT_DIR}"
    echo "Submitting ${LABEL}"
    echo "  cores  = ${TOTAL_CORES}"
    echo "  mode   = ${SCALE_MODE}"
    SBATCH_ARGS=(
        -N "$NODES"
        --ntasks-per-node="$TASKS_PER_NODE"
        --cpus-per-task="$CPUS_PER_TASK"
        --output="${BASE_OUT}/gen${GEN}/logs/${LABEL}_%j.out"
        --error="${BASE_OUT}/gen${GEN}/logs/${LABEL}_%j.err"
    )
    if [ -n "$PREV_JOB" ]; then
        DEP_FLAG="--dependency=afterok:$PREV_JOB"
    else
        DEP_FLAG=""
    fi
    JOB_ID=$(sbatch --parsable \
        $DEP_FLAG \
        --export=ALL,FWMP_SCALE_MODE=$SCALE_MODE,FWMP_BASE_CORES=$BASE_CORES,FWMP_BASE_DS=$BASE_DS,FWMP_TOTAL_CORES=$TOTAL_CORES,FWMP_BASE_OUTPUT_DIR=$FWMP_BASE_OUTPUT_DIR,FWMP_SCOREP=0 \
        "${SBATCH_ARGS[@]}" \
        run_elastic_param_blosc_new.sbatch \
        "$NODES" \
        "$TASKS_PER_NODE" \
        "$CPUS_PER_TASK" \
        "$GEN"
    )
    echo "submitted ${LABEL}, job_id=${JOB_ID}"
    PREV_JOB="$JOB_ID"
done
