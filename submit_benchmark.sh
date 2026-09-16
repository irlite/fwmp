#!/bin/bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# Usage: ./submit_benchmark.sh <gen_number>
# Example: ./submit_benchmark.sh 1
# -----------------------------------------------------------------------------

GEN_NUMBER="${1:?Usage: $0 <gen_number>}"
GEN_TAG="gen${GEN_NUMBER}"

SBATCH_SCRIPT="$(dirname "$0")/run_elastic2.sbatch"

# Format: "nodes ranks_per_node cpus_per_task"
CONFIGS=(
  "1 4 16"
)

CONFIGSS=(
  "2 1 8"
  "2 2 8"
  "4 2 8"
  "8 2 8"
  "1 1 8"
  "1 1 64"
  "1 2 32"
  "1 4 16"
  "1 8 8"
  "1 16 4"
  "1 32 2"
  "1 64 1"
)

echo "============================================================"
echo "Submitting benchmark generation: ${GEN_TAG}"
echo "Total configs: ${#CONFIGS[@]}"
echo "============================================================"

for config in "${CONFIGS[@]}"; do
    read -r nodes ranks_per_node cpus_per_task <<< "${config}"

    total_ranks=$(( nodes * ranks_per_node ))
    total_cpus=$(( total_ranks * cpus_per_task ))

    job_name="elastic_${GEN_TAG}_n${nodes}_r${ranks_per_node}_c${cpus_per_task}"

    echo "Submitting: nodes=${nodes} ranks/node=${ranks_per_node} cpus/task=${cpus_per_task} | total_ranks=${total_ranks} total_cpus=${total_cpus}"

    sbatch \
        --nodes="${nodes}" \
        --ntasks-per-node="${ranks_per_node}" \
        --cpus-per-task="${cpus_per_task}" \
        --job-name="${job_name}" \
        --export="ALL,GEN_TAG=${GEN_TAG}" \
        "${SBATCH_SCRIPT}"

done

echo "============================================================"
echo "All jobs submitted for ${GEN_TAG}."
echo "============================================================"
