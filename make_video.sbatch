#!/bin/bash
#SBATCH -p scc-cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --job-name=postprocess
#SBATCH --output=post_%j.out
#SBATCH --error=post_%j.err

module load python ffmpeg gcc
source .venv/bin/activate
cd src

python combine_hdf5.py

python make_video.py \
  ../output/elastic_wavefield.h5 \
  ../output/elastic_wavefield.mp4
