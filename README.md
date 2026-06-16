# fwmp

## Setup

First create and install the Python environment:

./venv_setup.sh

This installs all required Python dependencies into the local .venv.

---

## Run the full pipeline

Submit both the simulation and post-processing jobs:

./run_pipeline.sh

This will:
1. Submit the wave simulation job
2. Automatically submit the post-processing job after completion using SLURM dependencies (afterok)

---

## Output

All results are written to:

./output/

Including:
- Per-rank HDF5 simulation outputs (elastic_wavefield_rank*.h5)
- Combined virtual dataset (elastic_wavefield.h5)
- Final animation video (elastic_wavefield.mp4)
