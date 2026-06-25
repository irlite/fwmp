#!/bin/bash
#SBATCH -p scc-cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --job-name=postprocess
#SBATCH --output=post_%j.out
#SBATCH --error=post_%j.err

set -e

DEFAULT_JOB_ID=14422228
JOB_ID=${1:-$DEFAULT_JOB_ID}

module load python ffmpeg gcc

source .venv/bin/activate
cd src

python combine_hdf5.py "$JOB_ID"

python make_video.py \
  "../output/elastic_wavefield_job_${JOB_ID}.h5" \
  "../output/elastic_wavefield_job_${JOB_ID}.mp4"
