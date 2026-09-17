# fwmp

2D elastic wave simulation, MPI-parallel, with the finite-difference kernels in C.

Two files do the work. `fwmp.py` is the simulation; `fwmp.sbatch` builds the
kernels and submits it. The kernels come in four optimisation levels, described
below.

## Layout on the cluster

```
fwmp/
├── fwmp.py                 the simulation
├── fwmp.sbatch             SLURM job script
├── elastic_kernels_v1_naive.c     OpenMP kernels, four levels,
├── elastic_kernels_v2_simd.c      all compiled by the job script
├── elastic_kernels_v3_blocked.c
├── elastic_kernels_v4_fused.c
└── data/                   the three SEG-Y models
```

`fwmp.py` also works from a `src/` subdirectory with `data/` as a sibling; it
checks both places. Anywhere else, pass `--data-dir`.

Output does not go here by default on a real run. See below.

## First time

```bash
git clone <your-repo> fwmp
cd fwmp
FWMP_SETUP=1 sbatch fwmp.sbatch --niter 200 --frame-stride 50
```

`FWMP_SETUP=1` creates `.venv` and installs the dependencies, then imports
mpi4py to prove it works before any job depends on it.

mpi4py needs care, and is the usual reason a first run fails. Recent pip wheels
are prebuilt against a generic ABI and `dlopen` `libmpi.so` when imported, which
dies if your cluster's MPI is not on the loader path:

```
RuntimeError: cannot load MPI library
libmpi.so.40: cannot open shared object file: No such file or directory
```

The setup step avoids this by keeping a working system mpi4py if there is one,
and otherwise compiling from source against the module's `mpicc`. To repair a
venv that already has a bad wheel in it:

```bash
module purge && module load gcc openmpi python
source .venv/bin/activate
pip uninstall -y mpi4py
MPICC=$(which mpicc) pip install --no-binary=mpi4py --no-cache-dir mpi4py
python -c "from mpi4py import MPI; print(MPI.Get_library_version().splitlines()[0])"
```

If `which mpicc` is empty, the MPI module is not providing a compiler wrapper
and that is the actual problem. `module list`, `mpicc -show` and
`echo $LD_LIBRARY_PATH | tr : '\n' | grep -i mpi` will show what you have.

That first submission also compiles all four kernel levels. Later jobs only
recompile the ones whose C source you have edited.

The short `--niter 200` is deliberate: check the output looks sane before
queueing a real run.

## Where output goes

A long run writes tens of gigabytes of HDF5, which will blow through a home
directory quota. Point `FWMP_WORK_DIR` at scratch:

```bash
FWMP_WORK_DIR=/scratch/$USER/fwmp sbatch fwmp.sbatch --niter 50000
```

Results then land in `/scratch/$USER/fwmp/output/job_<id>/`. The code, the venv
and the compiled kernels stay where you cloned the repo; only output moves.
Easiest is to export it in your shell profile so you cannot forget it.

Running `fwmp.py` directly, the knob is `--output-dir`.

Rank 0 creates the directory and writes a probe file before anything else
starts, so a typo in the scratch path fails in the first second rather than
after the models have loaded.

The SEG-Y inputs are read from `data/` next to the script, or one level up if
the script lives in `src/`. The file names are in the Settings block.

## Normal runs

```bash
sbatch fwmp.sbatch
sbatch fwmp.sbatch --niter 50000 --frame-stride 100 --compression blosc
sbatch -N 4 --ntasks-per-node=2 --cpus-per-task=8 fwmp.sbatch --stage-local
```

Everything after the script name is passed to `fwmp.py`. SLURM options go
before it. Run `python fwmp.py --help` for the full list.

The defaults in the `#SBATCH` header are 1 node, 4 ranks, 8 threads per rank,
4 hours. Override any of them on the command line.

## Kernel optimisation levels

All four compute the same expressions in the same order, so they produce
bit-for-bit identical output and do exactly the same number of flops. Only the
loop structure and the compiler hints differ, which makes a timing comparison
between them a fair one.

| Level | File | What changes |
|-------|------|--------------|
| v1 | `elastic_kernels_v1_naive.c` | `omp parallel for` on the outer loop, nothing else |
| v2 | `elastic_kernels_v2_simd.c` | `restrict`, `omp simd`, hoisted row offsets, `schedule(static)` |
| v3 | `elastic_kernels_v3_blocked.c` | v2 plus 2D cache tiling with `collapse(2)` |
| v4 | `elastic_kernels_v4_fused.c` | v3 plus row-pair fusion and software prefetch |

v2 is the default and the one the project has used all along. Pick another with
`FWMP_KERNEL`:

```bash
FWMP_KERNEL=v1 sbatch fwmp.sbatch --niter 2000
FWMP_KERNEL=v3 FWMP_BLOCK_COLUMNS=1024 sbatch fwmp.sbatch --niter 2000
```

The jump from v1 to v2 is the big one, and `restrict` does most of that work on
its own. Without it the compiler has to assume `sxx` and `vx` might be the same
memory, so it cannot keep values in registers across a store and will not
vectorise at all.

v3 and v4 might not beat v2 on your machine, and that is a real result rather
than a failed experiment. v3 only pays off once a grid row stops fitting in
cache, which depends on `nx`, which is the model width plus twice the sponge.
v4 is micro-optimisation, and on a kernel already running at memory bandwidth
it can do nothing at all. Measure before you believe either of them.

`FWMP_BLOCK_ROWS` and `FWMP_BLOCK_COLUMNS` set the tile shape for v3 and v4
(default 32 x 512). Too small and you drown in loop overhead; too large and the
tile stops fitting in cache. With few threads and a small tile per rank, watch
for the other failure: not enough tiles to keep every thread busy.

## Measuring

Every run prints this when it finishes:

```
time loop       0.58 s   (slowest rank)
  frames        0.00 s
  close         0.00 s
throughput     519.4 steps/s   156.5 MLUP/s
```

Timings are reduced with max across ranks, not mean, because a step is not done
until the slowest rank has finished it. One lattice update is one cell carried
through one whole time step, meaning both kernels.

For a clean comparison, use enough steps that the run is not dominated by
startup, and keep `--frame-stride` high so I/O stays out of the way:

```bash
for level in v1 v2 v3 v4; do
    FWMP_KERNEL=$level sbatch fwmp.sbatch --niter 5000 --frame-stride 5000
done
```

## Scaling sweeps

`submit_sweep.sh` submits one job per rank/thread configuration and collects
the timings into a single CSV. You choose the kernel level; one sweep uses one
level, so run it again with a different level to compare kernels.

```bash
./submit_sweep.sh v2 --dry-run      # see what it would submit
./submit_sweep.sh v2                # submit
./submit_sweep.sh --collect /scratch/$USER/fwmp/sweeps/sweep_<id>
```

Configurations live in the `CONFIGS` array at the top of the script, as
`NODES TASKS_PER_NODE CPUS_PER_TASK`. They are grouped by the question each
group answers: the best rank/thread split at a fixed core count, pure OpenMP
scaling, pure MPI scaling, and scaling across nodes. Comment out the groups you
do not need.

Before submitting, the script asks `sinfo` how many cores a node has and skips
anything that cannot be scheduled, rather than leaving it stuck in the queue.
It also skips duplicate configurations, which would otherwise submit two jobs
writing to the same directory. Set `FWMP_CORES_PER_NODE` if the detection is
wrong.

Every run writes `timing.csv` into its own directory, and `--collect` joins
them with a `label` column. Runs that have not finished are reported on stderr
and left out of the CSV, so you can collect a sweep while some jobs are still
queued and run it again later.

Since every configuration solves the same problem for the same number of steps,
the `cells` column is identical across all of them. If it is not, something has
changed between runs and the comparison is not measuring what you think.

## Settings

Six things are command-line options:

```
--niter --frame-stride --downsample --compression --output-dir --kernel-lib
```

Everything else is a constant in the Settings block at the top of `fwmp.py`.
That includes the sponge width, the source wavelet, and the write strategy.
They all still work, they just are not worth a flag.

The write strategy is three independent settings:

`WRITE_MODE` is `batched`, `direct` or `async`. Batched buffers
`FRAMES_PER_FLUSH` frames and writes them together. Direct writes one frame at a
time and uses the least memory. Async hands frames to a writer thread pinned to
the rank's last core so compute never blocks on I/O; it needs at least 2 cores
per rank and refuses to start otherwise.

`STAGE_ON_LOCAL_DISK` writes each rank's file to node-local disk and copies it
to the output directory at the end, keeping the shared filesystem out of the
time loop. `LOCAL_DISK_DIRECTORY` overrides `$TMPDIR`.

`--compression` picks the HDF5 filter; `GZIP_LEVEL`, `BLOSC_LEVEL` and
`TARGET_CHUNK_MEGABYTES` tune it. `none` is fastest to write and largest on
disk, `blosc` is usually the best trade, `gzip` shrinks more but will bottleneck
the time loop on a wide tile.

## Output

```
output/job_<id>/
├── elastic_wavefield.h5        virtual dataset, read this one
└── rank_NNNN/
    └── elastic_wavefield.h5    one rank's tile
```

The top-level file is an HDF5 virtual dataset. It copies nothing and points at
the per-rank files with relative paths, so the whole `job_<id>` directory can be
moved as a unit, but the rank files have to travel with it.

Datasets are `vz` with shape `(frames, rows, columns)`, `vp`, and `time`.

## Profiling

```bash
FWMP_SCOREP=1 sbatch fwmp.sbatch --niter 2000
```

This recompiles the kernels through `scorep-gcc`, runs Python under Score-P, and
writes traces to `output/job_<id>/scorep/`. Expect it to be slower; use a short
run. If the `scorep` Python module is missing the job still runs, it just
produces no trace.

## Checking a run

```python
import h5py
with h5py.File("output/job_12345/elastic_wavefield.h5") as f:
    print(f["vz"].shape, dict(f.attrs))
```

For compressed runs, `import hdf5plugin` before opening the file or HDF5 will
not find the filter.

## Notes

The simulation is bit-for-bit reproducible. Changing the rank count or the write
strategy does not change a single value in `vz`, which makes any difference
between two runs a real difference rather than noise.

`--sponge-width` sets the absorbing boundary thickness in cells, default 240.
The padded grid is the model plus twice that in each dimension, so it dominates
memory use and runtime. Shrink it for quick tests, but know that a thin sponge
reflects energy back into the model.
