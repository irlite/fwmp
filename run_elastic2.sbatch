#!/bin/bash
#SBATCH -p scc-cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:20:00
#SBATCH --job-name=elastic

set -euo pipefail

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

export REPO_DIR="/user/maxim.barnstorf/u27934/fwmp"
export RUN_DIR="${REPO_DIR}/src/elastic"

export PROJECT_STORE="/user/maxim.barnstorf/u27934/.project/dir.project/maxim/new_structure"

# GEN_TAG is passed via --export from the calling script
# e.g. GEN_TAG=gen1
export GEN_TAG="${GEN_TAG:-gen0}"

export OUTPUT_ROOT="${PROJECT_STORE}/${GEN_TAG}/output"
export SCOREP_ROOT="${PROJECT_STORE}/${GEN_TAG}/scorep"

export FWMP_BASE_OUTPUT_DIR="${OUTPUT_ROOT}/job_${SLURM_JOB_ID}"
export SCOREP_EXPERIMENT_DIRECTORY="${SCOREP_ROOT}/job_${SLURM_JOB_ID}"

export DATA_DIR="${REPO_DIR}/data"

export FWMP_VP_PATH="${DATA_DIR}/MODEL_P-WAVE_VELOCITY_1.25m.segy"
export FWMP_VS_PATH="${DATA_DIR}/MODEL_S-WAVE_VELOCITY_1.25m.segy"
export FWMP_RHO_PATH="${DATA_DIR}/MODEL_DENSITY_1.25m.segy"

export FWMP_KERNEL_LIB="${RUN_DIR}/libelastic_kernels.so"

mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${SCOREP_ROOT}"
mkdir -p "${FWMP_BASE_OUTPUT_DIR}"

# -----------------------------------------------------------------------------
# Modules / environment
# -----------------------------------------------------------------------------

module purge
module load gcc
module load openmpi
module load python
module load scorep

cd "${REPO_DIR}"

source .venv/bin/activate

export PYTHON_BIN="$(command -v python)"

# -----------------------------------------------------------------------------
# Simulation settings
# -----------------------------------------------------------------------------

export FWMP_SCOREP="${FWMP_SCOREP:-1}"
export FWMP_NITER="${FWMP_NITER:-6000}"
export FWMP_FRAME_STRIDE="${FWMP_FRAME_STRIDE:-100}"
export FWMP_OUTPUT_BATCH="${FWMP_OUTPUT_BATCH:-4}"
export FWMP_DS="${FWMP_DS:-10}"

# -----------------------------------------------------------------------------
# Threading settings
# -----------------------------------------------------------------------------

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export PYTHONUNBUFFERED=1
export PMIX_MCA_psec=native

# -----------------------------------------------------------------------------
# Score-P settings
# -----------------------------------------------------------------------------

export SCOREP_OVERWRITE_EXPERIMENT_DIRECTORY=true
export SCOREP_ENABLE_TRACING=1
export SCOREP_ENABLE_PROFILING=1
export SCOREP_PROFILING_MAX_CALLPATH_DEPTH=150

# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------

echo "============================================================"
echo "Job ID:              ${SLURM_JOB_ID}"
echo "Gen tag:             ${GEN_TAG}"
echo "Nodes:               ${SLURM_JOB_NUM_NODES}"
echo "Tasks per node:      ${SLURM_NTASKS_PER_NODE}"
echo "CPUs per task:       ${SLURM_CPUS_PER_TASK}"
echo "Submit directory:    ${SLURM_SUBMIT_DIR}"
echo "Repository:          ${REPO_DIR}"
echo "Run directory:       ${RUN_DIR}"
echo "Project store:       ${PROJECT_STORE}"
echo "HDF5 output dir:     ${FWMP_BASE_OUTPUT_DIR}"
echo "Score-P dir:         ${SCOREP_EXPERIMENT_DIRECTORY}"
echo "Python:              ${PYTHON_BIN}"
echo "FWMP_SCOREP:         ${FWMP_SCOREP}"
echo "FWMP_NITER:          ${FWMP_NITER}"
echo "FWMP_FRAME_STRIDE:   ${FWMP_FRAME_STRIDE}"
echo "FWMP_OUTPUT_BATCH:   ${FWMP_OUTPUT_BATCH}"
echo "FWMP_DS:             ${FWMP_DS}"
echo "OMP_NUM_THREADS:     ${OMP_NUM_THREADS}"
echo "============================================================"

echo "Checking run files:"
ls -lh "${RUN_DIR}/run_elastic.py"
ls -lh "${RUN_DIR}/libelastic_kernels.so"

echo "Checking input files:"
ls -lh "${FWMP_VP_PATH}"
ls -lh "${FWMP_VS_PATH}"
ls -lh "${FWMP_RHO_PATH}"

cd "${RUN_DIR}"

echo "Current working directory:"
pwd

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------

if [ "${FWMP_SCOREP}" = "1" ]; then
    echo "Running with Score-P"
    echo "SCOREP_EXPERIMENT_DIRECTORY=${SCOREP_EXPERIMENT_DIRECTORY}"

    srun --distribution=block:block \
         --cpu-bind=cores \
         --export=ALL \
         "${PYTHON_BIN}" -m scorep --mpp=mpi --thread=omp "${RUN_DIR}/run_elastic.py"
else
    echo "Running without Score-P"

    srun --distribution=block:block \
         --cpu-bind=cores \
         --export=ALL \
         "${PYTHON_BIN}" "${RUN_DIR}/run_elastic.py"
fi

echo "============================================================"
echo "Run finished."
echo "HDF5 output:"
find "${FWMP_BASE_OUTPUT_DIR}" -maxdepth 3 -type f | sort | head -50

if [ "${FWMP_SCOREP}" = "1" ]; then
    echo "Score-P output:"
    find "${SCOREP_EXPERIMENT_DIRECTORY}" -maxdepth 3 -type f | sort | head -50
fi

echo "============================================================"
