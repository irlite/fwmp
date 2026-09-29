# fwmp

A 2D elastic wave simulation. Python sets up the problem and splits it across MPI ranks. C kernels (OpenMP) do the math. Output is `vz` snapshots in HDF5.

There are two versions of the code:

- **Main version.** `fwmp.py`, run with `fwmp.sbatch`. It has four kernel levels (v1 to v4) you can compare.
- **Tiled version.** `src/elastic_tiled.py`, run with `fwmp_tiled.sbatch`. It uses one fused kernel and overlaps halo exchange with compute.

## What's in the repo

```
fwmp/
├── fwmp.py                        main simulation
├── fwmp.sbatch                    SLURM script for fwmp.py
├── elastic_kernels_v1_naive.c     kernel level 1
├── elastic_kernels_v2_simd.c      kernel level 2 (default)
├── elastic_kernels_v3_blocked.c   kernel level 3
├── elastic_kernels_v4_fused.c     kernel level 4
├── submit_sweep.sh                scaling sweep for fwmp.py
├── fwmp_tiled.sbatch              SLURM script for the tiled version
├── submit_sweep_tiled.sh          scaling sweep for the tiled version
├── src/
│   ├── elastic_tiled.py           tiled simulation
│   └── elastic_kernels_tiled.c    tiled kernel
└── data/
    ├── MODEL_P-WAVE_VELOCITY_1.25m.segy
    ├── MODEL_S-WAVE_VELOCITY_1.25m.segy
    └── MODEL_DENSITY_1.25m.segy
```

## What you need

- A SLURM cluster with these modules: `gcc`, `openmpi`, `python`
- `scorep` module, only if you want profiling
- The three SEG-Y model files above

Python packages (the setup step installs them): `numpy`, `h5py`, `segyio`, `hdf5plugin`, `mpi4py`.

## Setup (do this once)

### 1. Get the code and data

```bash
git clone <your-repo> fwmp
cd fwmp
mkdir -p data
# copy the three .segy files into data/
```

### 2. Set your output folder

Both `fwmp.sbatch` and `fwmp_tiled.sbatch` have this line near the top:

```bash
DEFAULT_WORK_DIR="/user/utkarsh.pathak/..."
```

It points at another user's folder. Change it to your own scratch space, for example:

```bash
DEFAULT_WORK_DIR="/scratch/$USER/fwmp"
```

Set it to `""` to write output next to the code instead. Don't do that for long runs. A full run writes tens of GB and will fill a home directory.

You can also leave the file alone and set `FWMP_WORK_DIR` each time you submit.

### 3. Check the partition

Both sbatch scripts use `#SBATCH --partition=scc-cpu`. If your cluster uses a different name, change it. The sweep scripts read `FWMP_PARTITION` (default `scc-cpu`).

### 4. Build everything with a short test run

```bash
FWMP_SETUP=1 sbatch fwmp.sbatch --niter 200 --frame-stride 50
```

This one job:

1. Creates `.venv` and installs the Python packages
2. Builds `mpi4py` against the cluster's MPI, if it doesn't already work
3. Compiles all four kernels into `libelastic_kernels_v1.so` to `v4.so`
4. Runs 200 steps as a test

Check `fwmp_<jobid>.out`. It should end with `done: .../elastic_wavefield.h5`.

## Running the main version

Everything after `fwmp.sbatch` goes to `fwmp.py`. SLURM options go before it.

```bash
# default run: 50000 steps, v2 kernel
sbatch fwmp.sbatch

# fewer steps, compressed output
sbatch fwmp.sbatch --niter 2000 --compression blosc

# more nodes
sbatch -N 4 --ntasks-per-node=2 --cpus-per-task=8 fwmp.sbatch

# a different kernel level
FWMP_KERNEL=v3 sbatch fwmp.sbatch --niter 2000
```

Default job size: 1 node, 4 ranks, 8 threads per rank, 4 hours.

### Command-line options

| Option | Default | What it does |
|---|---|---|
| `--niter` | 50000 | Number of time steps |
| `--frame-stride` | 100 | Save a snapshot every N steps |
| `--downsample` | 1 | Keep every Nth model sample. Use 4 or 8 for quick tests |
| `--compression` | none | `none`, `gzip`, `blosc` or `lz4` |
| `--output-dir` | set by sbatch | Where results go |
| `--kernel-lib` | set by sbatch | Which compiled kernel to load |

Other settings (sponge width, source, write mode, local-disk staging) are constants at the top of `fwmp.py`. Edit them there.

### Environment variables for fwmp.sbatch

| Variable | What it does |
|---|---|
| `FWMP_SETUP=1` | Build `.venv` and kernels, then run |
| `FWMP_BUILD=1` | Only compile the kernels, then exit |
| `FWMP_KERNEL` | `v1`, `v2`, `v3` or `v4`. Default `v2` |
| `FWMP_WORK_DIR` | Output folder, overrides `DEFAULT_WORK_DIR` |
| `FWMP_BLOCK_ROWS`, `FWMP_BLOCK_COLUMNS` | Tile size for v3 and v4, applied at build time. Default 32 x 512 |
| `FWMP_SCOREP=1` | Profile with Score-P |

### After editing a kernel

Jobs don't recompile on their own. Rebuild after any change to a `.c` file or the block size:

```bash
FWMP_BUILD=1 sbatch fwmp.sbatch
```

## Running the tiled version

The tiled version uses the same `.venv`, so do the setup step first.

```bash
# 1. compile the tiled kernel
sbatch --export=ALL,FWMP_BUILD=1 fwmp_tiled.sbatch

# 2. run it
sbatch fwmp_tiled.sbatch
```

It takes settings from environment variables, not flags:

| Variable | Default | What it does |
|---|---|---|
| `FWMP_NITER` | 50000 | Number of time steps |
| `FWMP_FRAME_STRIDE` | 100 | Save a snapshot every N steps |
| `FWMP_TILE_ROWS` | 64 | Tile height in rows |
| `FWMP_COMPRESSION` | blosc | `none`, `gzip`, `blosc` or `lz4` |
| `FWMP_DS` | 1 | Downsample factor |

Example:

```bash
sbatch --export=ALL,FWMP_NITER=2000,FWMP_TILE_ROWS=32 fwmp_tiled.sbatch
```

The job time limit is 15 minutes. Raise `#SBATCH --time` for long runs.

The tiled version reports whole-job wall time only. Don't compare it with the loop time from `fwmp.py`.

## Scaling sweeps

A sweep submits one job per layout (nodes x ranks x threads) and collects the timings into one CSV. Build the kernels first.

```bash
chmod +x submit_sweep.sh submit_sweep_tiled.sh

# main version, pick one kernel level per sweep
./submit_sweep.sh v2 --dry-run     # show what would be submitted
./submit_sweep.sh v2               # submit

# tiled version
./submit_sweep_tiled.sh --dry-run
./submit_sweep_tiled.sh
```

The script prints the sweep folder. When the jobs finish:

```bash
./submit_sweep.sh --collect <sweep folder>
```

This writes `results.csv` in that folder. You can run it again while some jobs are still queued.

Edit the `CONFIGS` list at the top of each script to change which layouts run. Each line is `NODES TASKS_PER_NODE CPUS_PER_TASK`. Layouts that need more cores than a node has are skipped. If core detection fails, set `FWMP_CORES_PER_NODE`.

## Output

```
<work dir>/output/job_<id>/
├── elastic_wavefield.h5      open this one
├── timing.csv                timings for this run
└── rank_0000/
    └── elastic_wavefield.h5  one rank's piece
```

The top-level file links to the rank files. It holds no data itself. Keep the whole `job_<id>` folder together when you move it.

Datasets:

- `vz`, shape `(frames, rows, columns)`
- `vp`, the P-wave model
- `time`, the time of each frame

Read it in Python:

```python
import hdf5plugin   # needed for compressed runs
import h5py

with h5py.File("output/job_12345/elastic_wavefield.h5") as f:
    print(f["vz"].shape)
    print(dict(f.attrs))
```

At the end of a run, `fwmp.py` prints something like this:

```
time loop       0.58 s   (slowest rank)
  frames        0.00 s
  close         0.00 s
throughput     519.4 steps/s   156.5 MLUP/s
```

## Profiling with Score-P

Build instrumented kernels, then run:

```bash
FWMP_SCOREP=1 FWMP_BUILD=1 sbatch fwmp.sbatch
FWMP_SCOREP=1 sbatch fwmp.sbatch --niter 2000
```

The launch uses `python -m scorep`, so the `scorep` Python package must be in `.venv`. Install it once with the module loaded:

```bash
module load gcc openmpi python scorep
source .venv/bin/activate
pip install scorep
```

Traces go to `output/job_<id>/scorep/`. Profiled runs are slower, so keep them short.

For the tiled version:

```bash
sbatch --export=ALL,FWMP_BUILD=1,FWMP_SCOREP=1 fwmp_tiled.sbatch
sbatch --export=ALL,FWMP_SCOREP=1 fwmp_tiled.sbatch
```

## Running without SLURM

For a quick test on a laptop or login node:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy h5py segyio hdf5plugin mpi4py

gcc -O3 -fopenmp -march=native -fPIC -shared \
    -o libelastic_kernels_v2.so elastic_kernels_v2_simd.c

OMP_NUM_THREADS=2 mpirun -n 2 python fwmp.py \
    --niter 200 --frame-stride 50 --downsample 8 --output-dir test_out
```

## Common problems

**`cannot load MPI library` or `libmpi.so.40: cannot open shared object file`**

`mpi4py` was installed from a prebuilt wheel. Rebuild it against the cluster's MPI:

```bash
module purge && module load gcc openmpi python
source .venv/bin/activate
pip uninstall -y mpi4py
MPICC=$(which mpicc) pip install --no-binary=mpi4py --no-cache-dir mpi4py
```

If `which mpicc` prints nothing, your MPI module has no compiler wrapper. Ask your cluster admins which module to load.

**`missing SEG-Y model file(s)`**

Put the three `.segy` files in `data/`.

**`missing libelastic_kernels_vX.so`**

Build the kernels: `FWMP_BUILD=1 sbatch fwmp.sbatch`.

**`no .venv here`**

Run the setup step once: `FWMP_SETUP=1 sbatch fwmp.sbatch --niter 200`.

**`cannot write to output directory`**

`DEFAULT_WORK_DIR` or `FWMP_WORK_DIR` points somewhere you can't write. See setup step 2.

**`rank N ended up with an empty tile`**

Too many ranks for the grid size. Use fewer ranks or a smaller `--downsample`.