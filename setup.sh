#!/bin/bash
set -e

module purge
module load gcc
module load openmpi
module load python
module load scorep

scorep gcc -O3 -fopenmp -shared -fPIC src/elastic_kernels.c -o src/libelastic_kernels.so

#gcc -O3 -shared -fPIC src/elastic_kernels_single_threaded.c -o src/libelastic_kernels_single_threaded.so
scorep gcc -O3 -shared -fPIC src/elastic_kernels_single_threaded.c -o src/libelastic_kernels_single_threaded.so

source .venv/bin/activate

python -m pip install --upgrade pip

python -m pip install scorep numpy numba mpi4py h5py segyio matplotlib
