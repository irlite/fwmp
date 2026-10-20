# fwmp

This is the code for "Parallel Seismic Wave Forward Modeling", see the [techincal report](https://hps.vi4io.org/_media/teaching/summer_term_2026/stud/pchpc/group_3.pdf) and [slideset](https://hps.vi4io.org/_media/teaching/summer_term_2026/stud/pchpc/group_3_presentation.pdf) for more information.

## Setup

First create and install the Python environment:

./setup.sh

This installs all required Python dependencies into the local .venv.

It also compiles the C kernels.

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
