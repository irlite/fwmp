#!/bin/bash
#SBATCH -p scc-cpu
#SBATCH -N 1
#SBATCH --mem=8G
#SBATCH --contiguous
#SBATCH --time=00:20:00
#SBATCH --job-name=elastic-single-threaded
#SBATCH --output=logs/elastic-single-threaded.out
#SBATCH --error=logs/elastic-single-threaded.err

set -e

module purge
module load gcc
module load openmpi
module load python

source .venv/bin/activate
cd src
python -u elastic.py
