#!/bin/bash
set -e

module purge
module load gcc
module load openmpi
module load python
module load scorep

gcc -O3 -fopenmp -shared -fPIC src/elastic_kernels.c -o src/libelastic_kernels.so
gcc -O3 -fopenmp -shared -fPIC src/elastic_kernels_1d.c -o src/libelastic_kernels_1d.so
gcc -O3 -fPIC -shared src/elastic_kernels_seq.c -o src/libelastic_kernels_seq.so

#gcc -O3 -shared -fPIC src/elastic_kernels_single_threaded.c -o src/libelastic_kernels_single_threaded.so
gcc -O3 -shared -fPIC src/elastic_kernels_single_threaded.c -o src/libelastic_kernels_single_threaded.so
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    python -m venv .venv
    source .venv/bin/activate
fi

# making sure that output ,previous_runs, scorep_results directory exists
mkdir -p output/ previous_runs/ scorep_results/

python -m pip install --upgrade pip

python -m pip install scorep numpy numba mpi4py h5py segyio matplotlib
