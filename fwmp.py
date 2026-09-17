"""
2D elastic wave simulation, MPI-parallel, writing vz snapshots to HDF5.

This is the merge of the fourteen elastic*.py files that used to sit in src/.
They all ran identical physics; what differed was how each rank got its frames
onto disk.

Six things are worth changing per run and are command-line options:

    --niter --frame-stride --downsample --compression --output-dir --kernel-lib

Everything else lives in the Settings block below. That includes the sponge
width, the source, and the write strategy (batched, direct or async writes, and
node-local staging), all of which still work; they are just not worth a flag.

The finite-difference kernels are C with OpenMP, reached through ctypes, and
come in four optimisation levels. Pick one with --kernel-lib. Build them with:

    gcc -O3 -fopenmp -march=native -fPIC -shared \
        -o libelastic_kernels_v2.so elastic_kernels_v2_simd.c

Every rank writes its own tile to output/<job>/rank_NNNN/elastic_wavefield.h5.
Rank 0 then builds a virtual dataset at output/<job>/elastic_wavefield.h5 that
makes all those tiles look like one array.

Score-P stays off unless FWMP_SCOREP=1 is set. That one is an environment
variable rather than a flag because scorep.user has to be imported before
anything touches MPI, long before argparse runs.
"""

import argparse
import ctypes
import os
import queue
import shutil
import sys
import threading
from contextlib import nullcontext
from dataclasses import dataclass, field

# Blosc spins up its own thread pool. Inside an MPI rank that already owns a
# fixed set of cores that just means oversubscription, so hold it to one thread.
# This has to happen before hdf5plugin is imported.
os.environ.setdefault("BLOSC_NTHREADS", "1")

import numpy as np
import h5py
import segyio

try:
    from mpi4py import MPI
except Exception as mpi_import_error:
    # A pip-installed mpi4py wheel dlopens libmpi.so at import time and fails if
    # the cluster's MPI is not on the loader path. Every rank hits this at once,
    # so a full traceback from each is unreadable. Say the useful part instead.
    sys.exit(
        f"mpi4py could not start MPI: {mpi_import_error}\n"
        "Usually this means mpi4py was installed from a wheel instead of being\n"
        "built against the MPI that srun uses. With the MPI module loaded:\n"
        "  pip uninstall -y mpi4py\n"
        "  MPICC=$(which mpicc) pip install --no-binary=mpi4py --no-cache-dir mpi4py\n"
    )


SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Settings
#
# These were command-line options once. In practice nothing ever changed them
# per run, so they live here instead. Edit and rerun.
# ---------------------------------------------------------------------------

# Absorbing boundary thickness in cells. The padded grid is the model plus
# twice this in each direction, so it dominates memory and runtime. A thin
# sponge reflects energy back into the model.
SPONGE_WIDTH = 240

# Ricker wavelet.
SOURCE_FREQUENCY = 8.0        # Hz
SOURCE_AMPLITUDE = 1e9
INJECT_SOURCE_INTO_VZ = False  # otherwise the source only enters the stresses

# SEG-Y file names, looked for in data/ beside this script or one level up.
VP_MODEL_FILE = "MODEL_P-WAVE_VELOCITY_1.25m.segy"
VS_MODEL_FILE = "MODEL_S-WAVE_VELOCITY_1.25m.segy"
DENSITY_MODEL_FILE = "MODEL_DENSITY_1.25m.segy"

# How frames reach the file.
#   batched  buffer FRAMES_PER_FLUSH frames and write them in one go
#   direct   one dataset write per frame, least memory
#   async    writer thread on its own core, needs 2+ cores per rank
WRITE_MODE = "batched"
FRAMES_PER_FLUSH = 8
IO_BUFFER_COUNT = 2

# Write to node-local disk and copy to the output directory at the end, which
# keeps the shared filesystem out of the time loop. None means $TMPDIR.
STAGE_ON_LOCAL_DISK = False
LOCAL_DISK_DIRECTORY = None

# Compression detail. The filter itself is --compression.
GZIP_LEVEL = 4
BLOSC_LEVEL = 3
TARGET_CHUNK_MEGABYTES = 8.0

# Cell size of the SEG-Y models in metres, before downsampling.
MODEL_CELL_SIZE_METRES = 1.25

# Courant number. Time step is COURANT_NUMBER * cell_size / fastest_velocity.
COURANT_NUMBER = 0.4

# Peak absorption at the outer edge of the sponge, 1/s. The bottom gets twice
# the sides and top because that is where most of the energy ends up.
ABSORPTION_SIDES = 60.0
ABSORPTION_BOTTOM = 120.0


# ---------------------------------------------------------------------------
# Score-P
# ---------------------------------------------------------------------------

def _load_scorep():
    """Import scorep.user if FWMP_SCOREP is set, otherwise return None."""
    requested = os.environ.get("FWMP_SCOREP", "")
    if requested.strip().lower() not in ("1", "true", "yes", "on"):
        return None

    experiment_directory = os.environ.get("SCOREP_EXPERIMENT_DIRECTORY", ".")
    os.makedirs(experiment_directory, exist_ok=True)
    os.environ["SCOREP_EXPERIMENT_DIRECTORY"] = experiment_directory

    try:
        import scorep.user
        return scorep.user
    except ImportError:
        # Asking for Score-P without the module installed shouldn't kill a run
        # that would otherwise be fine. You just get no trace.
        return None


SCOREP_USER = _load_scorep()


def scorep_region(name):
    """Named Score-P region, or a do-nothing context manager when it's off."""
    if SCOREP_USER is not None and hasattr(SCOREP_USER, "region"):
        return SCOREP_USER.region(name)
    return nullcontext()


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def find_data_directory():
    """
    Locate data/, whether the script sits beside it or one level down.

    Both layouts are in use: fwmp.py at the top of the repo next to data/, and
    fwmp.py in src/ with data/ as a sibling of src/.
    """
    candidates = [
        os.path.join(SCRIPT_DIRECTORY, "data"),
        os.path.join(os.path.dirname(SCRIPT_DIRECTORY), "data"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--niter", type=int, default=50000, dest="time_steps",
        help="number of time steps (default: %(default)s)",
    )
    parser.add_argument(
        "--frame-stride", type=int, default=100,
        help="save a vz snapshot every N steps (default: %(default)s)",
    )
    parser.add_argument(
        "--downsample", type=int, default=1, metavar="N",
        help="keep every Nth sample of the input models (default: %(default)s)",
    )
    parser.add_argument(
        "--compression", choices=("none", "gzip", "blosc", "lz4"), default="none",
        help="HDF5 filter for the vz dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir", default=None, dest="output_directory",
        help=(
            "where results go, normally on scratch "
            "(default: output/job_<id> beside this script)"
        ),
    )
    parser.add_argument(
        "--kernel-lib", default=None, dest="kernel_library_path",
        help="compiled C kernel library (default: libelastic_kernels_v2.so)",
    )

    arguments = parser.parse_args(argv)

    if arguments.output_directory is None:
        job_id = os.environ.get("SLURM_JOB_ID", "noslurm")
        arguments.output_directory = os.path.join(
            SCRIPT_DIRECTORY, "output", f"job_{job_id}",
        )
    arguments.output_directory = os.path.abspath(arguments.output_directory)

    if arguments.kernel_library_path is None:
        arguments.kernel_library_path = os.path.join(
            SCRIPT_DIRECTORY, "libelastic_kernels_v2.so",
        )

    arguments.compression_level = (
        GZIP_LEVEL if arguments.compression == "gzip" else BLOSC_LEVEL
    )

    data_directory = find_data_directory()
    arguments.data_directory = data_directory
    arguments.vp_path = os.path.join(data_directory, VP_MODEL_FILE)
    arguments.vs_path = os.path.join(data_directory, VS_MODEL_FILE)
    arguments.density_path = os.path.join(data_directory, DENSITY_MODEL_FILE)

    return arguments


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

def load_segy_model(path, downsample):
    """Read a SEG-Y file into a (rows, columns) float32 array."""
    with segyio.open(path, "r", ignore_geometry=True) as segy_file:
        # Traces are columns of the model, so stacking them gives (columns,
        # rows) and needs a transpose.
        model = np.stack([np.array(trace) for trace in segy_file.trace]).T
    return model[::downsample, ::downsample].astype(np.float32)


def extend_edges(model, sponge_width):
    """
    Wrap a model in `sponge_width` cells of padding on all four sides.

    The padding repeats the outermost row or column rather than tapering to
    zero, because a jump in material properties at the edge would reflect waves
    straight back into the region we care about.
    """
    model_rows, model_columns = model.shape
    padded = np.empty(
        (model_rows + 2 * sponge_width, model_columns + 2 * sponge_width),
        dtype=np.float32,
    )

    interior_rows = slice(sponge_width, sponge_width + model_rows)
    interior_columns = slice(sponge_width, sponge_width + model_columns)
    padded[interior_rows, interior_columns] = model

    left_edge = sponge_width
    right_edge = sponge_width + model_columns
    top_edge = sponge_width
    bottom_edge = sponge_width + model_rows

    # Columns first, then rows. The row pass then copies whole rows that already
    # include the left and right padding, which fills the corners for free.
    padded[:, :left_edge] = padded[:, left_edge:left_edge + 1]
    padded[:, right_edge:] = padded[:, right_edge - 1:right_edge]
    padded[:top_edge, :] = padded[top_edge:top_edge + 1, :]
    padded[bottom_edge:, :] = padded[bottom_edge - 1:bottom_edge, :]

    return padded


@dataclass
class MaterialModel:
    """Elastic parameters on the padded grid."""
    p_velocity: np.ndarray        # vp
    shear_modulus: np.ndarray     # mu
    lame_first: np.ndarray        # lambda
    lame_first_plus_two_mu: np.ndarray
    inverse_density: np.ndarray
    grid_rows: int
    grid_columns: int
    sponge_width: int


def build_material_model(p_velocity, s_velocity, density, sponge_width):
    """Pad the input models and turn velocities and density into elastic moduli."""
    padded_p_velocity = extend_edges(p_velocity, sponge_width)
    padded_s_velocity = extend_edges(s_velocity, sponge_width)
    padded_density = extend_edges(density, sponge_width)

    shear_modulus = (padded_density * padded_s_velocity ** 2).astype(np.float32)
    lame_first = (
        padded_density * padded_p_velocity ** 2 - 2.0 * shear_modulus
    ).astype(np.float32)

    return MaterialModel(
        p_velocity=padded_p_velocity,
        shear_modulus=shear_modulus,
        lame_first=lame_first,
        lame_first_plus_two_mu=(lame_first + 2.0 * shear_modulus).astype(np.float32),
        inverse_density=(1.0 / padded_density).astype(np.float32),
        grid_rows=padded_p_velocity.shape[0],
        grid_columns=padded_p_velocity.shape[1],
        sponge_width=sponge_width,
    )


def build_absorption_factors(grid_rows, grid_columns, sponge_width, time_step):
    """
    Per-cell damping multiplier for the sponge layer.

    Absorption ramps up quadratically to its peak at the outer boundary; a
    sudden jump would reflect almost as badly as no sponge. The kernels apply
    this after every update, and the interior is exactly 1.0.
    """
    absorption_rate = np.zeros((grid_rows, grid_columns), dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, sponge_width, dtype=np.float32) ** 2

    for depth in range(sponge_width):
        # depth 0 is the outermost cell, so read the ramp backwards.
        strength = ramp[sponge_width - 1 - depth]
        absorption_rate[:, depth] = np.maximum(
            absorption_rate[:, depth], ABSORPTION_SIDES * strength)
        absorption_rate[:, -1 - depth] = np.maximum(
            absorption_rate[:, -1 - depth], ABSORPTION_SIDES * strength)
        absorption_rate[depth, :] = np.maximum(
            absorption_rate[depth, :], ABSORPTION_SIDES * strength)
        absorption_rate[-1 - depth, :] = np.maximum(
            absorption_rate[-1 - depth, :], ABSORPTION_BOTTOM * strength)

    return np.clip(
        1.0 - absorption_rate * float(time_step), 0.0, 1.0,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Domain decomposition
# ---------------------------------------------------------------------------

@dataclass
class Domain:
    """One rank's tile of the padded grid, and which ranks hold the neighbours."""
    cartesian_comm: object
    rank: int
    rank_count: int
    rank_grid_rows: int           # ranks stacked vertically
    rank_grid_columns: int
    grid_row: int                 # this rank's position in the rank grid
    grid_column: int
    neighbour_above: int          # MPI.PROC_NULL at the edge of the grid
    neighbour_below: int
    neighbour_left: int
    neighbour_right: int
    row_start: int                # tile extent in global padded coordinates
    row_stop: int
    column_start: int
    column_stop: int

    @property
    def tile_rows(self):
        return self.row_stop - self.row_start

    @property
    def tile_columns(self):
        return self.column_stop - self.column_start


def split_evenly(total, parts, index):
    """Divide `total` cells over `parts` ranks, remainder to the lowest indices."""
    sizes = [total // parts + (1 if i < total % parts else 0) for i in range(parts)]
    start = sum(sizes[:index])
    return start, start + sizes[index]


def choose_rank_grid(rank_count, grid_rows, grid_columns):
    """
    Pick a 2D arrangement of ranks that keeps tiles as square as possible.

    Halo traffic scales with a tile's perimeter, so the factorisation with the
    smallest tile_rows + tile_columns wins. Arrangements that would starve a
    rank are skipped.
    """
    best_arrangement = None
    best_perimeter = None

    for rows in range(1, rank_count + 1):
        if rank_count % rows != 0:
            continue
        columns = rank_count // rows
        if rows > grid_rows or columns > grid_columns:
            continue

        perimeter = grid_rows / rows + grid_columns / columns
        if best_perimeter is None or perimeter < best_perimeter:
            best_perimeter = perimeter
            best_arrangement = (rows, columns)

    if best_arrangement is None:
        # Every factorisation starves someone. Let MPI decide and let the
        # emptiness check below produce a clear error.
        best_arrangement = MPI.Compute_dims(rank_count, 2)

    return int(best_arrangement[0]), int(best_arrangement[1])


def create_domain(comm, grid_rows, grid_columns):
    rank = comm.Get_rank()
    rank_count = comm.Get_size()

    rank_grid_rows, rank_grid_columns = choose_rank_grid(
        rank_count, grid_rows, grid_columns,
    )
    cartesian_comm = comm.Create_cart(
        dims=[rank_grid_rows, rank_grid_columns],
        periods=[False, False],
        reorder=False,
    )

    grid_row, grid_column = cartesian_comm.Get_coords(rank)
    neighbour_above, neighbour_below = cartesian_comm.Shift(0, 1)
    neighbour_left, neighbour_right = cartesian_comm.Shift(1, 1)

    row_start, row_stop = split_evenly(grid_rows, rank_grid_rows, grid_row)
    column_start, column_stop = split_evenly(
        grid_columns, rank_grid_columns, grid_column,
    )

    if row_stop <= row_start or column_stop <= column_start:
        raise RuntimeError(
            f"rank {rank} ended up with an empty tile. "
            f"Too many ranks for a {grid_rows}x{grid_columns} grid."
        )

    return Domain(
        cartesian_comm=cartesian_comm,
        rank=rank,
        rank_count=rank_count,
        rank_grid_rows=rank_grid_rows,
        rank_grid_columns=rank_grid_columns,
        grid_row=grid_row,
        grid_column=grid_column,
        neighbour_above=neighbour_above,
        neighbour_below=neighbour_below,
        neighbour_left=neighbour_left,
        neighbour_right=neighbour_right,
        row_start=row_start,
        row_stop=row_stop,
        column_start=column_start,
        column_stop=column_stop,
    )


def add_halo_ring(tile):
    """
    Put a one-cell border around a tile, seeded from the tile's own edges.

    Material borders are never exchanged, so the seeding is all there is.
    Wavefield borders are overwritten by the halo exchange each step, and the
    seeding only matters at the outermost ranks with no neighbour to hear from.
    """
    bordered = np.empty((tile.shape[0] + 2, tile.shape[1] + 2), dtype=tile.dtype)

    bordered[1:-1, 1:-1] = tile
    bordered[0, 1:-1] = bordered[1, 1:-1]
    bordered[-1, 1:-1] = bordered[-2, 1:-1]
    bordered[:, 0] = bordered[:, 1]
    bordered[:, -1] = bordered[:, -2]

    return np.ascontiguousarray(bordered)


@dataclass
class LocalMaterial:
    """This rank's slice of the material model, each array with a halo ring."""
    shear_modulus: np.ndarray
    lame_first: np.ndarray
    lame_first_plus_two_mu: np.ndarray
    inverse_density: np.ndarray
    absorption: np.ndarray


def extract_local_material(material, absorption, domain):
    rows = slice(domain.row_start, domain.row_stop)
    columns = slice(domain.column_start, domain.column_stop)

    return LocalMaterial(
        shear_modulus=add_halo_ring(material.shear_modulus[rows, columns]),
        lame_first=add_halo_ring(material.lame_first[rows, columns]),
        lame_first_plus_two_mu=add_halo_ring(
            material.lame_first_plus_two_mu[rows, columns]),
        inverse_density=add_halo_ring(material.inverse_density[rows, columns]),
        absorption=add_halo_ring(absorption[rows, columns]),
    )


class HaloExchanger:
    """
    Nearest-neighbour halo exchange on the staggered grid.

    Two exchanges per step, moving data in opposite directions: the stress
    update reads velocities one cell towards the origin, the velocity update
    reads stresses one cell away from it. Get the direction wrong and the
    result still looks plausible, just wrong at the tile seams.
    """

    # Separate tags per direction and axis. Without them a rank that falls
    # behind can have its column exchange matched against a neighbour's row
    # exchange, which corrupts the halo in a way that is horrible to debug.
    TAG_INWARD_COLUMNS = 10
    TAG_INWARD_ROWS = 20
    TAG_OUTWARD_COLUMNS = 30
    TAG_OUTWARD_ROWS = 40

    def __init__(self, domain, max_fields=3):
        self.domain = domain
        self.column_send = np.empty((max_fields, domain.tile_rows), dtype=np.float32)
        self.column_receive = np.empty((max_fields, domain.tile_rows), dtype=np.float32)
        self.row_send = np.empty((max_fields, domain.tile_columns), dtype=np.float32)
        self.row_receive = np.empty((max_fields, domain.tile_columns), dtype=np.float32)

    def exchange_inward(self, fields):
        """Send the first interior row and column to the left/upper neighbours."""
        self._exchange(
            fields,
            send_offset=1,
            receive_offset=-1,
            column_destination=self.domain.neighbour_left,
            column_origin=self.domain.neighbour_right,
            row_destination=self.domain.neighbour_above,
            row_origin=self.domain.neighbour_below,
            column_tag=self.TAG_INWARD_COLUMNS,
            row_tag=self.TAG_INWARD_ROWS,
        )

    def exchange_outward(self, fields):
        """Send the last interior row and column to the right/lower neighbours."""
        self._exchange(
            fields,
            send_offset=-2,
            receive_offset=0,
            column_destination=self.domain.neighbour_right,
            column_origin=self.domain.neighbour_left,
            row_destination=self.domain.neighbour_below,
            row_origin=self.domain.neighbour_above,
            column_tag=self.TAG_OUTWARD_COLUMNS,
            row_tag=self.TAG_OUTWARD_ROWS,
        )

    def _exchange(
        self, fields, send_offset, receive_offset,
        column_destination, column_origin,
        row_destination, row_origin,
        column_tag, row_tag,
    ):
        field_count = len(fields)

        send_buffer = self.column_send[:field_count]
        receive_buffer = self.column_receive[:field_count]
        for slot, field_array in enumerate(fields):
            send_buffer[slot, :] = field_array[1:-1, send_offset]
        self.domain.cartesian_comm.Sendrecv(
            send_buffer, dest=column_destination, sendtag=column_tag,
            recvbuf=receive_buffer, source=column_origin, recvtag=column_tag,
        )
        if column_origin != MPI.PROC_NULL:
            for slot, field_array in enumerate(fields):
                field_array[1:-1, receive_offset] = receive_buffer[slot, :]

        send_buffer = self.row_send[:field_count]
        receive_buffer = self.row_receive[:field_count]
        for slot, field_array in enumerate(fields):
            send_buffer[slot, :] = field_array[send_offset, 1:-1]
        self.domain.cartesian_comm.Sendrecv(
            send_buffer, dest=row_destination, sendtag=row_tag,
            recvbuf=receive_buffer, source=row_origin, recvtag=row_tag,
        )
        if row_origin != MPI.PROC_NULL:
            for slot, field_array in enumerate(fields):
                field_array[receive_offset, 1:-1] = receive_buffer[slot, :]


@dataclass
class KernelBounds:
    """
    Half-open index window the C kernels update, in halo-inclusive coordinates.

    Interior cells run from 1 to tile_rows inclusive. A rank holding the top or
    bottom of the whole grid steps one cell further in, because the stencil
    would otherwise read off the end of the array.
    """
    row_begin: int
    row_end: int
    column_begin: int
    column_end: int


def compute_kernel_bounds(domain, grid_rows):
    return KernelBounds(
        row_begin=2 if domain.row_start == 0 else 1,
        row_end=domain.tile_rows if domain.row_stop == grid_rows
        else domain.tile_rows + 1,
        column_begin=1,
        column_end=domain.tile_columns + 1,
    )


@dataclass
class OutputWindow:
    """
    Where this rank's tile overlaps the real model, ignoring the sponge.

    Ranks whose tile sits entirely inside the sponge layer have nothing worth
    saving. They get has_data=False and write no vz dataset at all.
    """
    has_data: bool
    tile_rows: slice          # source, into the halo-inclusive wavefield
    tile_columns: slice
    model_row_start: int      # destination, in unpadded model coordinates
    model_row_stop: int
    model_column_start: int
    model_column_stop: int

    @property
    def shape(self):
        return (
            self.model_row_stop - self.model_row_start,
            self.model_column_stop - self.model_column_start,
        )


EMPTY_OUTPUT_WINDOW = OutputWindow(
    has_data=False,
    tile_rows=slice(0, 0),
    tile_columns=slice(0, 0),
    model_row_start=-1,
    model_row_stop=-1,
    model_column_start=-1,
    model_column_stop=-1,
)


def compute_output_window(domain, model_rows, model_columns, sponge_width):
    saved_row_start = max(domain.row_start, sponge_width)
    saved_row_stop = min(domain.row_stop, sponge_width + model_rows)
    saved_column_start = max(domain.column_start, sponge_width)
    saved_column_stop = min(domain.column_stop, sponge_width + model_columns)

    if saved_row_start >= saved_row_stop or saved_column_start >= saved_column_stop:
        return EMPTY_OUTPUT_WINDOW

    # The +1 converts a tile-local index into a halo-inclusive one.
    return OutputWindow(
        has_data=True,
        tile_rows=slice(
            saved_row_start - domain.row_start + 1,
            saved_row_stop - domain.row_start + 1,
        ),
        tile_columns=slice(
            saved_column_start - domain.column_start + 1,
            saved_column_stop - domain.column_start + 1,
        ),
        model_row_start=saved_row_start - sponge_width,
        model_row_stop=saved_row_stop - sponge_width,
        model_column_start=saved_column_start - sponge_width,
        model_column_stop=saved_column_stop - sponge_width,
    )


# ---------------------------------------------------------------------------
# C kernels
# ---------------------------------------------------------------------------

FLOAT32_ARRAY_2D = np.ctypeslib.ndpointer(
    dtype=np.float32, ndim=2, flags="C_CONTIGUOUS",
)


class ElasticKernels:
    """ctypes binding for the OpenMP kernels in elastic_kernels.c."""

    def __init__(self, library_path):
        if not os.path.exists(library_path):
            raise FileNotFoundError(
                f"{library_path} not found. Build it with:\n"
                f"  gcc -O3 -fopenmp -march=native -fPIC -shared "
                f"-o {library_path} elastic_kernels.c"
            )

        self.library = ctypes.CDLL(library_path)

        spacing_arguments = (ctypes.c_float,) * 3    # time_step, cell_x, cell_z
        index_arguments = (ctypes.c_int,) * 6        # shape and loop bounds

        self.library.update_stress.argtypes = [
            FLOAT32_ARRAY_2D,    # velocity_x
            FLOAT32_ARRAY_2D,    # velocity_z
            FLOAT32_ARRAY_2D,    # stress_xx
            FLOAT32_ARRAY_2D,    # stress_zz
            FLOAT32_ARRAY_2D,    # stress_xz
            FLOAT32_ARRAY_2D,    # lame_first
            FLOAT32_ARRAY_2D,    # lame_first_plus_two_mu
            FLOAT32_ARRAY_2D,    # shear_modulus
            FLOAT32_ARRAY_2D,    # absorption
            *spacing_arguments,
            *index_arguments,
        ]
        self.library.update_stress.restype = None

        self.library.update_velocity.argtypes = [
            FLOAT32_ARRAY_2D,    # velocity_x
            FLOAT32_ARRAY_2D,    # velocity_z
            FLOAT32_ARRAY_2D,    # stress_xx
            FLOAT32_ARRAY_2D,    # stress_zz
            FLOAT32_ARRAY_2D,    # stress_xz
            FLOAT32_ARRAY_2D,    # inverse_density
            FLOAT32_ARRAY_2D,    # absorption
            *spacing_arguments,
            *index_arguments,
        ]
        self.library.update_velocity.restype = None


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class Wavefields:
    """Velocity and stress components, each carrying a one-cell halo ring."""
    velocity_x: np.ndarray
    velocity_z: np.ndarray
    stress_xx: np.ndarray
    stress_zz: np.ndarray
    stress_xz: np.ndarray

    @classmethod
    def zeros(cls, tile_rows, tile_columns):
        shape = (tile_rows + 2, tile_columns + 2)
        return cls(*(np.zeros(shape, dtype=np.float32) for _ in range(5)))

    def reset(self):
        for field_array in (
            self.velocity_x, self.velocity_z,
            self.stress_xx, self.stress_zz, self.stress_xz,
        ):
            field_array.fill(0.0)


@dataclass
class RickerSource:
    """
    A Ricker wavelet injected into a single cell of the padded grid.

    The float32 rounding is deliberate. On the wavelet's leading tail the
    amplitude is exp(-a) for large a, so it is exponentially sensitive to t,
    and the scheme amplifies any perturbation. float64 gives an equally valid
    answer, but not the same one.
    """
    row: int
    column: int
    peak_frequency: float
    amplitude: float
    delay: np.float32 = field(init=False)

    def __post_init__(self):
        self.peak_frequency = np.float32(self.peak_frequency)
        self.amplitude = np.float32(self.amplitude)
        # Push the wavelet peak far enough past t=0 that the run starts from
        # something close to silence.
        self.delay = np.float32(1.2 / self.peak_frequency)

    def value_at(self, time):
        exponent = (
            np.pi * float(self.peak_frequency) * (float(time) - float(self.delay))
        ) ** 2
        wavelet = (1.0 - 2.0 * exponent) * np.exp(-exponent)
        # Scale as a separate multiply. Folding the amplitude into the line
        # above changes the rounding by one ulp, which is enough to break the
        # bit-for-bit comparison described in the class docstring.
        return np.float32(self.amplitude * wavelet)

    def lives_on(self, domain):
        return (
            domain.row_start <= self.row < domain.row_stop
            and domain.column_start <= self.column < domain.column_stop
        )

    def local_position(self, domain):
        return (
            self.row - domain.row_start + 1,
            self.column - domain.column_start + 1,
        )


class ElasticSimulation:
    """Advances the wavefields one time step at a time."""

    def __init__(
        self, kernels, halo, domain, wavefields, local_material,
        time_step, cell_size, bounds, source, inject_source_into_vz,
    ):
        self.kernels = kernels
        self.halo = halo
        self.domain = domain
        self.fields = wavefields
        self.material = local_material

        self.time_step = np.float32(time_step)
        self.cell_size = np.float32(cell_size)
        self.bounds = bounds
        self.source = source
        self.inject_source_into_vz = inject_source_into_vz

    def _update_stress(self):
        fields = self.fields
        bounds = self.bounds
        self.kernels.library.update_stress(
            fields.velocity_x, fields.velocity_z,
            fields.stress_xx, fields.stress_zz, fields.stress_xz,
            self.material.lame_first,
            self.material.lame_first_plus_two_mu,
            self.material.shear_modulus,
            self.material.absorption,
            self.time_step, self.cell_size, self.cell_size,
            fields.velocity_x.shape[0], fields.velocity_x.shape[1],
            bounds.row_begin, bounds.row_end,
            bounds.column_begin, bounds.column_end,
        )

    def _update_velocity(self):
        fields = self.fields
        bounds = self.bounds
        self.kernels.library.update_velocity(
            fields.velocity_x, fields.velocity_z,
            fields.stress_xx, fields.stress_zz, fields.stress_xz,
            self.material.inverse_density,
            self.material.absorption,
            self.time_step, self.cell_size, self.cell_size,
            fields.velocity_x.shape[0], fields.velocity_x.shape[1],
            bounds.row_begin, bounds.row_end,
            bounds.column_begin, bounds.column_end,
        )

    def warm_up(self):
        """
        Run one throwaway step and then wipe the fields.

        The first call pays for lazy symbol resolution, OpenMP thread pool
        setup and page faults. Doing that before the timed loop keeps step zero
        from dominating every profile.
        """
        with scorep_region("warmup_kernels"):
            self._update_stress()
            self._update_velocity()
        self.fields.reset()

    def step(self, step_index):
        fields = self.fields

        self.halo.exchange_inward([fields.velocity_x, fields.velocity_z])

        source_value = self.source.value_at(
            np.float32(step_index) * self.time_step,
        )
        if self.source.lives_on(self.domain):
            row, column = self.source.local_position(self.domain)
            fields.stress_xx[row, column] += source_value
            fields.stress_zz[row, column] += source_value
            if self.inject_source_into_vz:
                fields.velocity_z[row, column] += np.float32(
                    float(self.time_step)
                    * self.material.inverse_density[row, column]
                    * source_value
                )

        self._update_stress()

        self.halo.exchange_outward(
            [fields.stress_xx, fields.stress_zz, fields.stress_xz],
        )

        self._update_velocity()


# ---------------------------------------------------------------------------
# HDF5 output
# ---------------------------------------------------------------------------

# Aligning objects and reserving a decent metadata block keeps HDF5 from
# scattering small writes across a parallel filesystem.
HDF5_FILE_OPTIONS = dict(
    libver="latest",
    alignment_threshold=4 * 1024 ** 2,
    alignment_interval=4 * 1024 ** 2,
    meta_block_size=4 * 1024 ** 2,
)


def dataset_filter_options(
    compression, compression_level, frame_shape, target_chunk_megabytes,
):
    """
    Build the create_dataset keywords for the chosen compression filter.

    Compressed datasets must be chunked. One frame per chunk matches how frames
    are written, but for a big tile that chunk gets unwieldy, so both
    dimensions shrink by the same factor until it fits the target.
    """
    if compression == "none":
        return {}

    frame_rows, frame_columns = frame_shape
    target_bytes = target_chunk_megabytes * 1024 ** 2
    frame_bytes = frame_rows * frame_columns * 4

    if frame_bytes <= target_bytes:
        chunk_shape = (1, frame_rows, frame_columns)
    else:
        shrink = (target_bytes / frame_bytes) ** 0.5
        chunk_shape = (
            1,
            max(1, int(frame_rows * shrink)),
            max(1, int(frame_columns * shrink)),
        )

    if compression == "gzip":
        return dict(
            chunks=chunk_shape,
            compression="gzip",
            compression_opts=compression_level,
            shuffle=True,
        )

    # Imported here so a run without compression never needs hdf5plugin
    # installed at all.
    import hdf5plugin

    if compression == "blosc":
        return dict(chunks=chunk_shape, **hdf5plugin.Blosc(
            cname="lz4",
            clevel=compression_level,
            shuffle=hdf5plugin.Blosc.SHUFFLE,
        ))

    return dict(chunks=chunk_shape, shuffle=True, **hdf5plugin.LZ4())


class BatchedFrameWriter:
    """
    Collects frames in memory and writes them in contiguous groups.

    Fewer large writes beat many small ones on a parallel filesystem. The price
    is holding frames_per_flush frames in RAM, which for a big tile is not
    nothing.
    """

    def __init__(self, dataset, frame_shape, frames_per_flush):
        self.dataset = dataset
        self.frames_per_flush = max(1, frames_per_flush)
        self.buffer = np.empty(
            (self.frames_per_flush, *frame_shape), dtype=np.float32,
        )
        self.buffered_count = 0
        self.first_buffered_frame = 0

    def write(self, frame_index, frame):
        if self.buffered_count == 0:
            self.first_buffered_frame = frame_index

        with scorep_region("copy_output_frame_to_batch"):
            self.buffer[self.buffered_count] = frame

        self.buffered_count += 1
        if self.buffered_count == self.frames_per_flush:
            self.flush()

    def flush(self):
        if self.buffered_count == 0:
            return

        with scorep_region("hdf5_write_vz_batch"):
            stop = self.first_buffered_frame + self.buffered_count
            self.dataset[self.first_buffered_frame:stop] = (
                self.buffer[:self.buffered_count]
            )
        self.buffered_count = 0

    def close(self):
        self.flush()


class DirectFrameWriter:
    """One dataset write per frame. Simplest, and uses the least memory."""

    def __init__(self, dataset):
        self.dataset = dataset

    def write(self, frame_index, frame):
        with scorep_region("hdf5_write_vz_frame"):
            self.dataset[frame_index] = np.ascontiguousarray(frame)

    def close(self):
        pass


class AsyncFrameWriter:
    """
    Hands frames to a background thread so the time loop never waits on I/O.

    The thread pins itself to the rank's last core and owns the HDF5 file while
    it runs, so the main thread must not touch that file until close() returns.
    A fixed buffer pool provides back pressure: if I/O falls behind, the main
    thread blocks instead of growing a queue without bound.
    """

    def __init__(self, dataset, frame_shape, buffer_count):
        allowed_cores = sorted(os.sched_getaffinity(0))
        if len(allowed_cores) < 2:
            raise RuntimeError(
                "WRITE_MODE = \"async\" needs at least 2 cores per rank, "
                f"but this rank may only use {allowed_cores}. "
                "Raise --cpus-per-task or set WRITE_MODE to \"batched\"."
            )

        self.dataset = dataset
        self.io_core = allowed_cores[-1]
        self.failure = None

        buffer_count = max(2, buffer_count)
        self.free_buffers = queue.Queue(maxsize=buffer_count)
        self.queued_frames = queue.Queue(maxsize=buffer_count)
        for _ in range(buffer_count):
            self.free_buffers.put(np.empty(frame_shape, dtype=np.float32))

        self.thread = threading.Thread(target=self._write_loop, daemon=False)
        self.thread.start()

    def _write_loop(self):
        try:
            os.sched_setaffinity(0, {self.io_core})
            while True:
                item = self.queued_frames.get()
                if item is None:
                    return
                frame_index, buffer = item
                try:
                    self.dataset[frame_index] = buffer
                finally:
                    self.free_buffers.put(buffer)
        except BaseException as error:
            self.failure = error
            # Unblock a main thread that is already waiting on a buffer. The
            # array is a placeholder; the raise below happens before anything
            # reads it.
            try:
                self.free_buffers.put_nowait(np.empty(1, dtype=np.float32))
            except queue.Full:
                pass

    def _raise_if_failed(self):
        if self.failure is not None:
            raise RuntimeError("async HDF5 writer thread died") from self.failure

    def write(self, frame_index, frame):
        # Waiting with a timeout rather than blocking forever means a writer
        # thread that dies mid-wait still gets noticed instead of hanging the
        # whole job until the wall clock runs out.
        while True:
            self._raise_if_failed()
            try:
                buffer = self.free_buffers.get(timeout=0.1)
                break
            except queue.Empty:
                continue

        self._raise_if_failed()
        np.copyto(buffer, frame)
        self.queued_frames.put((frame_index, buffer))

    def close(self):
        self.queued_frames.put(None)
        self.thread.join()
        self._raise_if_failed()


class DiscardingFrameWriter:
    """For ranks whose tile lies entirely inside the sponge layer."""

    def write(self, frame_index, frame):
        pass

    def close(self):
        pass


class RankOutputFile:
    """
    This rank's HDF5 file: dataset layout, write strategy, and staging.

    With STAGE_ON_LOCAL_DISK the file is built on node-local disk and copied to its
    final home on close, which keeps the shared filesystem out of the time loop
    completely.
    """

    def __init__(
        self, arguments, domain, output_window,
        p_velocity, frame_times, frame_count, attributes,
    ):
        self.domain = domain
        self.window = output_window
        self.rank_directory_name = f"rank_{domain.rank:04d}"

        self.final_path = os.path.join(
            arguments.output_directory,
            self.rank_directory_name,
            "elastic_wavefield.h5",
        )

        if STAGE_ON_LOCAL_DISK:
            staging_root = (
                LOCAL_DISK_DIRECTORY
                or os.environ.get("TMPDIR", arguments.output_directory)
            )
            self.write_path = os.path.join(
                staging_root, "fwmp", self.rank_directory_name,
                "elastic_wavefield.h5",
            )
        else:
            self.write_path = self.final_path

        os.makedirs(os.path.dirname(self.write_path), exist_ok=True)

        with scorep_region("hdf5_open_rank_file"):
            self.file = h5py.File(self.write_path, "w", **HDF5_FILE_OPTIONS)

        with scorep_region("hdf5_create_rank_datasets"):
            self.frame_writer = self._create_datasets(
                arguments, p_velocity, frame_times, frame_count,
            )

        with scorep_region("hdf5_write_rank_attrs"):
            self._write_attributes(attributes)

    def _create_datasets(
        self, arguments, p_velocity, frame_times, frame_count,
    ):
        self.file.create_dataset("time", data=frame_times, track_times=False)

        if not self.window.has_data:
            return DiscardingFrameWriter()

        frame_shape = self.window.shape

        vz_dataset = self.file.create_dataset(
            "vz",
            shape=(frame_count, *frame_shape),
            dtype=np.float32,
            track_times=False,
            **dataset_filter_options(
                arguments.compression,
                arguments.compression_level,
                frame_shape,
                TARGET_CHUNK_MEGABYTES,
            ),
        )

        self.file.create_dataset(
            "vp",
            data=p_velocity[
                self.window.model_row_start:self.window.model_row_stop,
                self.window.model_column_start:self.window.model_column_stop,
            ].astype(np.float32),
            track_times=False,
        )

        if WRITE_MODE == "direct":
            return DirectFrameWriter(vz_dataset)

        if WRITE_MODE == "async":
            return AsyncFrameWriter(
                vz_dataset, frame_shape, IO_BUFFER_COUNT,
            )

        # Never buffer more frames than the run will produce.
        return BatchedFrameWriter(
            vz_dataset,
            frame_shape,
            min(FRAMES_PER_FLUSH, max(1, frame_count)),
        )

    def _write_attributes(self, attributes):
        window = self.window
        # These key names are terse because combine_hdf5.py and make_video.py
        # read them. Renaming them here breaks post-processing.
        placement = {
            "rank": self.domain.rank,
            "coord_z": self.domain.grid_row,
            "coord_x": self.domain.grid_column,
            "z0": window.model_row_start,
            "z1": window.model_row_stop,
            "x0": window.model_column_start,
            "x1": window.model_column_stop,
        }
        for key, value in {**placement, **attributes}.items():
            self.file.attrs[key] = value

    def write_frame(self, frame_index, velocity_z):
        self.frame_writer.write(
            frame_index,
            velocity_z[self.window.tile_rows, self.window.tile_columns],
        )

    def close(self):
        self.frame_writer.close()

        with scorep_region("hdf5_close_rank_file"):
            self.file.close()

        if self.write_path != self.final_path:
            self._copy_to_final_location()

    def _copy_to_final_location(self):
        """
        Move the staged file to where the virtual dataset expects it.

        The copy lands on a temporary name and is renamed, so a reader never
        finds a half-written file. Each rank removes only its own staging
        directory; racing to delete the shared parent invites mystery failures.
        """
        os.makedirs(os.path.dirname(self.final_path), exist_ok=True)
        temporary_path = f"{self.final_path}.partial"

        with scorep_region("copy_staged_output"):
            shutil.copy2(self.write_path, temporary_path)
            os.replace(temporary_path, self.final_path)

        shutil.rmtree(os.path.dirname(self.write_path), ignore_errors=True)

    def placement(self):
        """What rank 0 needs to slot this rank's tile into the virtual dataset."""
        return {
            "has_data": self.window.has_data,
            "row_start": int(self.window.model_row_start),
            "row_stop": int(self.window.model_row_stop),
            "column_start": int(self.window.model_column_start),
            "column_stop": int(self.window.model_column_stop),
            "relative_path": os.path.join(
                self.rank_directory_name, "elastic_wavefield.h5",
            ),
        }


def write_virtual_dataset(
    path, placements, p_velocity, frame_times, frame_count, attributes,
):
    """
    Stitch the per-rank files into something that reads like one array.

    Nothing is copied. The virtual dataset is a set of pointers into the rank
    files, and the paths are relative, so the whole output directory can be
    moved or renamed without breaking it.
    """
    model_rows, model_columns = p_velocity.shape

    with scorep_region("create_vds_file"):
        with h5py.File(path, "w", **HDF5_FILE_OPTIONS) as virtual_file:
            layout = h5py.VirtualLayout(
                shape=(frame_count, model_rows, model_columns),
                dtype=np.float32,
            )

            for placement in placements:
                if not placement["has_data"]:
                    continue

                source = h5py.VirtualSource(
                    placement["relative_path"],
                    "vz",
                    shape=(
                        frame_count,
                        placement["row_stop"] - placement["row_start"],
                        placement["column_stop"] - placement["column_start"],
                    ),
                )
                layout[
                    :,
                    placement["row_start"]:placement["row_stop"],
                    placement["column_start"]:placement["column_stop"],
                ] = source

            # fillvalue covers any gap, which should not exist, but a run with a
            # bug in the decomposition then shows up as zeros rather than junk.
            virtual_file.create_virtual_dataset("vz", layout, fillvalue=0.0)
            virtual_file.create_dataset(
                "vp", data=p_velocity.astype(np.float32), track_times=False,
            )
            virtual_file.create_dataset(
                "time", data=frame_times, track_times=False,
            )

            for key, value in attributes.items():
                virtual_file.attrs[key] = value


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def check_inputs(comm, arguments):
    """
    Confirm every file the run needs exists, before anyone opens one.

    segyio's FileNotFoundError does not name the file it wanted, so all ranks
    print a traceback saying nothing useful. Check on rank 0 and name the path.
    """
    problem = None

    if comm.Get_rank() == 0:
        missing_models = [
            path for path in
            (arguments.vp_path, arguments.vs_path, arguments.density_path)
            if not os.path.isfile(path)
        ]

        if missing_models:
            problem = (
                "missing SEG-Y model file(s):\n  "
                + "\n  ".join(missing_models)
                + f"\nLooked in {arguments.data_directory}. "
                "Put them there, or edit the file names in the Settings block."
            )
        elif not os.path.isfile(arguments.kernel_library_path):
            problem = (
                f"missing kernel library {arguments.kernel_library_path}\n"
                "Build it with:\n"
                "  gcc -O3 -fopenmp -march=native -fPIC -shared \\\n"
                "      -o libelastic_kernels_v2.so elastic_kernels_v2_simd.c\n"
                "or pass --kernel-lib."
            )

    problem = comm.bcast(problem, root=0)
    if problem is not None:
        raise SystemExit(problem)


def prepare_output_directory(comm, path):
    """
    Create the output directory and prove it is writable, on rank 0 only.

    A typo in a scratch path is easy to make, and checking now turns a crash
    deep inside HDF5 into one line at the start. The verdict is broadcast so
    every rank fails together; rank 0 raising alone would leave the others at
    the barrier until the wall clock ran out.
    """
    problem = None

    if comm.Get_rank() == 0:
        probe_path = os.path.join(path, ".fwmp_write_probe")
        try:
            os.makedirs(path, exist_ok=True)
            with open(probe_path, "w") as probe:
                probe.write("ok")
            os.remove(probe_path)
        except OSError as error:
            problem = f"cannot write to output directory {path}: {error}"

    problem = comm.bcast(problem, root=0)
    if problem is not None:
        raise SystemExit(problem)


def report_performance(
    comm, arguments, domain, kernel_bounds,
    seconds_in_loop, seconds_writing_frames, seconds_closing,
):
    """
    Print how long the time loop took, and drop a one-row CSV next to the data.

    Timings reduce with max, not mean: a step is not done until the slowest
    rank has finished it, and averaging would flatter a badly balanced run. The
    CSV lets a sweep be collected without scraping stdout.
    """
    cells_this_rank = (
        (kernel_bounds.row_end - kernel_bounds.row_begin)
        * (kernel_bounds.column_end - kernel_bounds.column_begin)
    )
    total_cells = comm.allreduce(cells_this_rank, op=MPI.SUM)

    slowest_loop = comm.reduce(seconds_in_loop, op=MPI.MAX, root=0)
    slowest_writes = comm.reduce(seconds_writing_frames, op=MPI.MAX, root=0)
    slowest_close = comm.reduce(seconds_closing, op=MPI.MAX, root=0)

    if comm.Get_rank() != 0 or slowest_loop <= 0.0:
        return

    steps_per_second = arguments.time_steps / slowest_loop
    # One lattice update is one cell carried through one whole time step, which
    # here means both the stress and the velocity kernel.
    lattice_updates = total_cells * arguments.time_steps / slowest_loop

    print(
        f"time loop   {slowest_loop:8.2f} s   (slowest rank)\n"
        f"  frames    {slowest_writes:8.2f} s\n"
        f"  close     {slowest_close:8.2f} s\n"
        f"throughput  {steps_per_second:8.1f} steps/s   "
        f"{lattice_updates / 1e6:.1f} MLUP/s",
        flush=True,
    )

    row = {
        "job_id": os.environ.get("SLURM_JOB_ID", "noslurm"),
        "nodes": os.environ.get("SLURM_JOB_NUM_NODES", "1"),
        "ranks": domain.rank_count,
        "threads_per_rank": os.environ.get("OMP_NUM_THREADS", "1"),
        "rank_grid": f"{domain.rank_grid_rows}x{domain.rank_grid_columns}",
        "kernel": os.path.basename(arguments.kernel_library_path),
        "niter": arguments.time_steps,
        "frame_stride": arguments.frame_stride,
        "downsample": arguments.downsample,
        "sponge_width": SPONGE_WIDTH,
        "compression": arguments.compression,
        "write_mode": WRITE_MODE,
        "stage_local": int(STAGE_ON_LOCAL_DISK),
        "cells": total_cells,
        "loop_seconds": round(slowest_loop, 4),
        "frame_seconds": round(slowest_writes, 4),
        "close_seconds": round(slowest_close, 4),
        "steps_per_second": round(steps_per_second, 3),
        "mlups": round(lattice_updates / 1e6, 3),
    }

    csv_path = os.path.join(arguments.output_directory, "timing.csv")
    with open(csv_path, "w") as csv_file:
        csv_file.write(",".join(row.keys()) + "\n")
        csv_file.write(",".join(str(value) for value in row.values()) + "\n")


def main(argv=None):
    arguments = parse_arguments(argv)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    rank_count = comm.Get_size()

    check_inputs(comm, arguments)

    with scorep_region("load_kernel_library"):
        kernels = ElasticKernels(arguments.kernel_library_path)

    prepare_output_directory(comm, arguments.output_directory)

    # Every rank reads the models itself. They are tiny next to the simulation
    # grid, and this beats scattering the full model over MPI.
    with scorep_region("load_segy_inputs"):
        p_velocity = load_segy_model(arguments.vp_path, arguments.downsample)
        s_velocity = load_segy_model(arguments.vs_path, arguments.downsample)
        density = load_segy_model(arguments.density_path, arguments.downsample)

    model_rows, model_columns = p_velocity.shape
    cell_size = np.float32(MODEL_CELL_SIZE_METRES * arguments.downsample)

    with scorep_region("build_material_model"):
        material = build_material_model(
            p_velocity, s_velocity, density, SPONGE_WIDTH,
        )

    # The fastest material anywhere on the grid sets the stable time step.
    fastest_velocity = float(material.p_velocity.max())
    time_step = np.float32(
        COURANT_NUMBER * float(cell_size) / fastest_velocity,
    )

    with scorep_region("build_damping"):
        absorption = build_absorption_factors(
            material.grid_rows,
            material.grid_columns,
            material.sponge_width,
            time_step,
        )

    with scorep_region("create_cartesian_topology"):
        domain = create_domain(comm, material.grid_rows, material.grid_columns)

    with scorep_region("extract_local_material"):
        local_material = extract_local_material(material, absorption, domain)

    with scorep_region("allocate_wavefields"):
        wavefields = Wavefields.zeros(domain.tile_rows, domain.tile_columns)

    output_window = compute_output_window(
        domain, model_rows, model_columns, material.sponge_width,
    )

    frame_count = len(range(0, arguments.time_steps, arguments.frame_stride))
    frame_times = (
        np.arange(frame_count, dtype=np.float32)
        * arguments.frame_stride
        * time_step
    )

    source_row = material.sponge_width + 1
    source_column = material.sponge_width + model_columns // 2

    # Short keys again, for the benefit of the post-processing scripts.
    shared_attributes = {
        "size": rank_count,
        "dims_z": domain.rank_grid_rows,
        "dims_x": domain.rank_grid_columns,
        "nz0": model_rows,
        "nx0": model_columns,
        "dx": float(cell_size),
        "dz": float(cell_size),
        "dt": float(time_step),
        "frame_stride": arguments.frame_stride,
        "n_frames": frame_count,
        "src_z0": source_row - material.sponge_width,
        "src_x0": source_column - material.sponge_width,
        "compression": arguments.compression,
        "write_mode": WRITE_MODE,
    }

    if rank == 0:
        print(
            f"grid {material.grid_rows}x{material.grid_columns} "
            f"({model_rows}x{model_columns} model, "
            f"{material.sponge_width}-cell sponge), "
            f"dt={float(time_step):.3e}s, "
            f"{frame_count} frames over {arguments.time_steps} steps",
            flush=True,
        )
        print(
            f"{domain.rank_grid_rows}x{domain.rank_grid_columns} rank grid, "
            f"output -> {arguments.output_directory}",
            flush=True,
        )

    comm.Barrier()

    output_file = RankOutputFile(
        arguments=arguments,
        domain=domain,
        output_window=output_window,
        p_velocity=p_velocity,
        frame_times=frame_times,
        frame_count=frame_count,
        attributes={
            "local_nz_phys": output_window.shape[0],
            "local_nx_phys": output_window.shape[1],
            **shared_attributes,
        },
    )

    kernel_bounds = compute_kernel_bounds(domain, material.grid_rows)

    simulation = ElasticSimulation(
        kernels=kernels,
        halo=HaloExchanger(domain),
        domain=domain,
        wavefields=wavefields,
        local_material=local_material,
        time_step=time_step,
        cell_size=cell_size,
        bounds=kernel_bounds,
        source=RickerSource(
            row=source_row,
            column=source_column,
            peak_frequency=SOURCE_FREQUENCY,
            amplitude=SOURCE_AMPLITUDE,
        ),
        inject_source_into_vz=INJECT_SOURCE_INTO_VZ,
    )

    simulation.warm_up()

    frame_index = 0
    seconds_writing_frames = 0.0

    loop_started = MPI.Wtime()
    with scorep_region("time_loop"):
        for step_index in range(arguments.time_steps):
            simulation.step(step_index)

            if step_index % arguments.frame_stride == 0:
                write_started = MPI.Wtime()
                output_file.write_frame(
                    frame_index, simulation.fields.velocity_z,
                )
                seconds_writing_frames += MPI.Wtime() - write_started
                frame_index += 1
    seconds_in_loop = MPI.Wtime() - loop_started

    close_started = MPI.Wtime()
    output_file.close()
    seconds_closing = MPI.Wtime() - close_started

    report_performance(
        comm, arguments, domain, kernel_bounds,
        seconds_in_loop, seconds_writing_frames, seconds_closing,
    )

    # Staged files have to be in place before rank 0 writes a virtual dataset
    # pointing at them, hence the barrier.
    with scorep_region("gather_vds_metadata"):
        placements = comm.gather(output_file.placement(), root=0)
    comm.Barrier()

    if rank == 0:
        virtual_dataset_path = os.path.join(
            arguments.output_directory, "elastic_wavefield.h5",
        )
        write_virtual_dataset(
            path=virtual_dataset_path,
            placements=placements,
            p_velocity=p_velocity,
            frame_times=frame_times,
            frame_count=frame_count,
            attributes=shared_attributes,
        )
        print(f"done: {virtual_dataset_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
