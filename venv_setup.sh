#!/bin/bash
set -e

module purge
module load gcc
module load openmpi
module load python

source .venv/bin/activate

python -m pip install --upgrade pip

python -m pip install numpy numba mpi4py h5py segyio matplotlib
