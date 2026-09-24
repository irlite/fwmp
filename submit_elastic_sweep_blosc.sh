#!/bin/bash
set -euo pipefail
GEN=1x_old_blosc
WEAK_SCALING=false
EXCLUSIVE=false
BASE_CORES=1
#BASE_DS=39
BASE_DS=47
STRONG_DS=1
CONFIGS=(
  "1 2 32"
  "1 2 32"
  "1 2 32"
  "1 2 32"
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
CONFIGSS=(
  "1 2 32"
  "2 2 32"
  "4 2 32"
  "8 2 32"

  "1 1 1"
  "1 1 16"
  "1 1 32"
  "1 2 32"
  "1 3 32"
)
BASE_OUT="/user/maxim.barnstorf/u27934/fwmp/opt_fast_drive"
mkdir -p "${BASE_OUT}"
mkdir -p "${BASE_OUT}/gen${GEN}/logs"
PREV_JOB=""
for CFG in "${CONFIGS[@]}"; do
    read -r NODES TASKS_PER_NODE CPUS_PER_TASK <<< "$CFG"
    LABEL="${NODES}x${TASKS_PER_NODE}x${CPUS_PER_TASK}"
    TOTAL_CORES=$((NODES * TASKS_PER_NODE * CPUS_PER_TASK))
    if [ "$WEAK_SCALING" = true ]; then
        DS=$(python3 - <<EOF
import math
base_ds = $BASE_DS
base_cores = $BASE_CORES
cores = $TOTAL_CORES
print(max(1, round(base_ds * (base_cores / cores) ** 0.5)))
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
    if [ -n "$PREV_JOB" ]; then
        DEP_FLAG="--dependency=afterok:$PREV_JOB"
    else
        DEP_FLAG=""
    fi
    JOB_ID=$(sbatch --parsable \
        $DEP_FLAG \
        --export=ALL,FWMP_DS=$DS,FWMP_BASE_OUTPUT_DIR=$FWMP_BASE_OUTPUT_DIR,FWMP_SCOREP=0 \
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
