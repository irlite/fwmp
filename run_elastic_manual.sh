#!/bin/bash

module purge
module load gcc
module load openmpi
module load python

cd ~/fwmp
source .venv/bin/activate

cd src/
mpirun -np 4 python -u elastic.py
