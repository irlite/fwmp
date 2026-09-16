#!/bin/bash
#SBATCH -p scc-cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --job-name=postprocess
#SBATCH --output=post_%j.out
#SBATCH --error=post_%j.err

set -e

DEFAULT_GENERATION=gen1
DEFAULT_JOB_ID=14422228

GENERATION=${1:-$DEFAULT_GENERATION}
JOB_ID=${2:-$DEFAULT_JOB_ID}

module purge
module load python ffmpeg gcc

cd ~/fwmp
source .venv/bin/activate

RUN_BASE="previous_runs/${GENERATION}"

MATCHES=( "${RUN_BASE}"/*_"${JOB_ID}" )

if [ ! -d "${MATCHES[0]}" ]; then
    echo "ERROR: Could not find run directory for generation=${GENERATION}, job_id=${JOB_ID}"
    echo "Looked for: ${RUN_BASE}/*_${JOB_ID}"
    exit 1
fi

if [ "${#MATCHES[@]}" -ne 1 ]; then
    echo "ERROR: Found multiple matching run directories:"
    printf '  %s\n' "${MATCHES[@]}"
    exit 1
fi

RUN_DIR="${MATCHES[0]}"
H5_FILE="${RUN_DIR}/elastic_wavefield.h5"
MP4_FILE="${RUN_DIR}/elastic_wavefield_job_${JOB_ID}.mp4"

if [ ! -f "${H5_FILE}" ]; then
    echo "ERROR: HDF5 file not found:"
    echo "  ${H5_FILE}"
    exit 1
fi

cd src

python make_video.py \
  "../${H5_FILE}" \
  "../${MP4_FILE}"
