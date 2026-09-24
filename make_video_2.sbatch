#!/bin/bash
#SBATCH -p scc-cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --job-name=postprocess
#SBATCH --output=post_%j.out
#SBATCH --error=post_%j.err

set -euo pipefail
shopt -s nullglob

DEFAULT_GENERATION="gen999"
DEFAULT_JOB_ID="16203439"

GENERATION="${1:-${DEFAULT_GENERATION}}"
JOB_ID="${2:-${DEFAULT_JOB_ID}}"

module purge
module load python ffmpeg gcc

REPO_DIR="${HOME}/fwmp"

cd "${REPO_DIR}"
source .venv/bin/activate

RUN_BASE="${HOME}/.project/dir.project/maxim"

if [[ ! -d "${RUN_BASE}" ]]; then
    echo "ERROR: Output directory does not exist:"
    echo "  ${RUN_BASE}"
    exit 1
fi

MATCHES=( "${RUN_BASE}"/*/job_"${JOB_ID}" )

if (( ${#MATCHES[@]} == 0 )); then
    echo "ERROR: Could not find run directory for:"
    echo "  generation: ${GENERATION}"
    echo "  job ID:     ${JOB_ID}"
    echo "Looked for:"
    echo "  ${RUN_BASE}/*/job_${JOB_ID}"
    exit 1
fi

if (( ${#MATCHES[@]} > 1 )); then
    echo "ERROR: Found multiple matching run directories:"
    printf '  %s\n' "${MATCHES[@]}"
    exit 1
fi

RUN_DIR="${MATCHES[0]}"
H5_FILE="${RUN_DIR}/elastic_wavefield.h5"
MP4_FILE="${RUN_DIR}/elastic_wavefield_job_${JOB_ID}.mp4"

if [[ ! -f "${H5_FILE}" ]]; then
    echo "ERROR: HDF5 file not found:"
    echo "  ${H5_FILE}"
    exit 1
fi

echo "Generation: ${GENERATION}"
echo "Job ID:     ${JOB_ID}"
echo "Run dir:    ${RUN_DIR}"
echo "Input:      ${H5_FILE}"
echo "Output:     ${MP4_FILE}"

cd "${REPO_DIR}/src"

python make_video.py \
    "${H5_FILE}" \
    "${MP4_FILE}"

echo "Video created successfully:"
echo "  ${MP4_FILE}"
