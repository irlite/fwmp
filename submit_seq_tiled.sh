#!/bin/bash
set -euo pipefail

if (( $# != 1 )); then
    echo "Usage: $0 NUMBER_OF_RUNS" >&2
    exit 1
fi

NUM_RUNS="$1"
JOB_SCRIPT="run_seq_tiled.sbatch"

if ! [[ "$NUM_RUNS" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUMBER_OF_RUNS must be a positive integer." >&2
    exit 1
fi

if [[ ! -f "$JOB_SCRIPT" ]]; then
    echo "Job script not found: $JOB_SCRIPT" >&2
    exit 1
fi

mkdir -p "$HOME/fwmp/opt_fast_drive/seq_tiled/logs"

# Submit the first job without a dependency.
result=$(sbatch \
    --parsable \
    --export="ALL,CHAIN_RUN=1,CHAIN_SIZE=${NUM_RUNS}" \
    "$JOB_SCRIPT")

previous_job=${result%%;*}

echo "Run 1/${NUM_RUNS}: job ${previous_job}"

# Each subsequent job waits until the preceding job has completed.
for ((run = 2; run <= NUM_RUNS; run++)); do
    result=$(sbatch \
        --parsable \
        --dependency="afterany:${previous_job}" \
        --export="ALL,CHAIN_RUN=${run},CHAIN_SIZE=${NUM_RUNS}" \
        "$JOB_SCRIPT")

    current_job=${result%%;*}

    echo "Run ${run}/${NUM_RUNS}: job ${current_job} waits for ${previous_job}"

    previous_job="$current_job"
done

echo "Last job in chain: ${previous_job}"
