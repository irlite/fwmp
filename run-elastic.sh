#!/bin/bash
#SBATCH --job-name=seismic-wave-equation-2d
##SBATCH --nodes=2
##SBATCH --ntasks=2
##SBATCH --ntasks-per-node=1
#SBATCH --time=01:30:00
#SBATCH --output=seismic-wave-equation_%j.out
#SBATCH --partition=scc-cpu

module load python/3.11.9

source .venv/bin/activate

cd src/
python elastic.py