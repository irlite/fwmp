"""
2D elastic wave simulation with MPI. Writes vz snapshots to HDF5.

Options:
    --niter --frame-stride --downsample --compression --output-dir --kernel-lib

Build the kernels:
    gcc -O3 -fopenmp -march=native -fPIC -shared \
        -o libelastic_kernels_v2.so elastic_kernels_v2_simd.c

Set FWMP_SCOREP=1 to turn on Score-P regions.
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


os.environ.setdefault("BLOSC_NTHREADS", "1")

import numpy as np
import h5py
import segyio

try:
    from mpi4py import MPI
except Exception as mpi_import_error:
    sys.exit(
        f"mpi4py could not start MPI: {mpi_import_error}\n"
        "Usually this means mpi4py was installed from a wheel instead of being\n"
        "built against the MPI that srun uses. With the MPI module loaded:\n"
        "  pip uninstall -y mpi4py\n"
        "  MPICC=$(which mpicc) pip install --no-binary=mpi4py --no-cache-dir mpi4py\n"
    )


SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


SPONGE_WIDTH = 240

# Ricker wavelet.
SOURCE_FREQUENCY = 8.0        # Hz
SOURCE_AMPLITUDE = 1e9
INJECT_SOURCE_INTO_VZ = False  # if False, the source only goes into the stresses

# SEG-Y files, looked up in data/ next to this script or one level up.
VP_MODEL_FILE = "MODEL_P-WAVE_VELOCITY_1.25m.segy"
VS_MODEL_FILE = "MODEL_S-WAVE_VELOCITY_1.25m.segy"
DENSITY_MODEL_FILE = "MODEL_DENSITY_1.25m.segy"

# How frames get written.
#   batched  hold FRAMES_PER_FLUSH frames, then write them together
#   direct   one write per frame, least memory
#   async    a writer thread on its own core, needs 2+ cores per rank
WRITE_MODE = "batched"
FRAMES_PER_FLUSH = 8
IO_BUFFER_COUNT = 2

# Write to node-local disk and copy to the output directory at the end.
# None means $TMPDIR.
STAGE_ON_LOCAL_DISK = False
LOCAL_DISK_DIRECTORY = None

# Compression settings. --compression picks the filter.
GZIP_LEVEL = 4
BLOSC_LEVEL = 3
TARGET_CHUNK_MEGABYTES = 8.0

# Cell size of the SEG-Y models in metres, before downsampling.
MODEL_CELL_SIZE_METRES = 1.25

# dt = COURANT_NUMBER * cell_size / max velocity.
COURANT_NUMBER = 0.4

# Peak absorption at the outer edge of the sponge, in 1/s. The bottom is
# twice the sides and top because most energy goes down.
ABSORPTION_SIDES = 60.0
ABSORPTION_BOTTOM = 120.0


# ---------------------------------------------------------------------------
# Score-P
# ---------------------------------------------------------------------------

def _load_scorep():
    """Return scorep.user if FWMP_SCOREP is set, else None."""
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
        # Run without a trace instead of failing.
        return None


SCOREP_USER = _load_scorep()


def scorep_region(name):
    """Named Score-P region, or a no-op when Score-P is off."""
    if SCOREP_USER is not None and hasattr(SCOREP_USER, "region"):
        return SCOREP_USER.region(name)
    return nullcontext()


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def find_data_directory():
    """Find data/ next to this script, or next to its parent (src/ layout)."""
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
        "--compression", choices=("none", "gzip", "blosc", "lz4"), default="lz4",
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
        # Each trace is a column, so transpose after stacking.
        model = np.stack([np.array(trace) for trace in segy_file.trace]).T
    return model[::downsample, ::downsample].astype(np.float32)


def extend_edges(model, sponge_width):
    """
    Pad a model by sponge_width cells on every side, copying the edge values.

    Padding with zeros would create a sharp jump that reflects waves back in.
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

    # Pad columns first so the row pass also fills the corners.
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
    """Pad the models and compute the elastic moduli."""
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
    Damping factor per cell for the sponge layer.

    Absorption grows quadratically toward the outer edge. The interior is 1.0.
    The kernels multiply by this after every update.
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
    """This rank's tile of the padded grid and its neighbour ranks."""
    cartesian_comm: object
    rank: int
    rank_count: int
    rank_grid_rows: int           # ranks stacked vertically
    rank_grid_columns: int
    grid_row: int                 # this rank's position in the rank grid
    grid_column: int
    neighbour_above: int          # MPI.PROC_NULL at the grid edge
    neighbour_below: int
    neighbour_left: int
    neighbour_right: int
    row_start: int                # tile extent in padded grid coordinates
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
    """Split total cells over parts ranks. Lower indices get the remainder."""
    sizes = [total // parts + (1 if i < total % parts else 0) for i in range(parts)]
    start = sum(sizes[:index])
    return start, start + sizes[index]


def choose_rank_grid(rank_count, grid_rows, grid_columns):
    """
    Pick the rank layout with the most square tiles.

    Halo traffic grows with tile perimeter, so pick the smallest
    tile_rows + tile_columns. Skip layouts that leave a rank with no cells.
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
        # No layout works. Let MPI pick one and let create_domain report the error.
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
    Add a one-cell border to a tile, filled from the tile's edges.

    Material borders are never exchanged, so this fill is final. Wavefield
    borders get overwritten by the halo exchange, except at the outer ranks.
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
    """This rank's slice of the material model, with halo rings."""
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
    Halo exchange with the four neighbour ranks on the staggered grid.

    There are two exchanges per step, in opposite directions. The stress
    update reads velocities one cell toward the origin. The velocity update
    reads stresses one cell away from it. A wrong direction gives errors only
    at tile seams, which are easy to miss.
    """

    # One tag per direction and axis, so a slow rank can't match a column
    # message with a row message.
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
        """Send the first interior row and column to the upper and left ranks."""
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
        """Send the last interior row and column to the lower and right ranks."""
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
    Index range the C kernels update, counting the halo. End is exclusive.

    Interior cells are 1 to tile_rows. Ranks at the top or bottom of the grid
    stop one cell short so the stencil stays inside the array.
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
    The part of this rank's tile that lies inside the model, not the sponge.

    Ranks fully inside the sponge get has_data=False and write no vz.
    """
    has_data: bool
    tile_rows: slice          # source, in the wavefield with halo
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

    # +1 skips the halo cell.
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
    """ctypes wrapper for the OpenMP kernels in elastic_kernels_v2_simd.c."""

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
    """Velocity and stress arrays, each with a one-cell halo."""
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
    Ricker wavelet added to one cell of the padded grid.

    Values are rounded to float32 on purpose. The wavelet's early tail is very
    sensitive to t, and the scheme amplifies small differences. float64 is
    just as correct but gives different output.
    """
    row: int
    column: int
    peak_frequency: float
    amplitude: float
    delay: np.float32 = field(init=False)

    def __post_init__(self):
        self.peak_frequency = np.float32(self.peak_frequency)
        self.amplitude = np.float32(self.amplitude)
        # Delay the peak so the run starts near zero.
        self.delay = np.float32(1.2 / self.peak_frequency)

    def value_at(self, time):
        exponent = (
            np.pi * float(self.peak_frequency) * (float(time) - float(self.delay))
        ) ** 2
        wavelet = (1.0 - 2.0 * exponent) * np.exp(-exponent)
        # Keep this multiply separate. Merging it changes rounding by one ulp
        # and breaks bit-for-bit matching with earlier runs.
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
    """Advances the wavefields one step at a time."""

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
        Run one step and then zero the fields.

        The first kernel call pays for symbol lookup, OpenMP thread startup
        and page faults. Doing it here keeps that cost out of the timed loop.
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

# Aligned objects and a large metadata block avoid many small writes on a
# parallel filesystem.
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
    Return create_dataset arguments for the chosen compression.

    Compressed datasets need chunks. A chunk is one frame, shrunk evenly in
    both directions if it is larger than the target size.
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

    # Imported here so uncompressed runs don't need hdf5plugin.
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
    Buffers frames in memory and writes them in groups.

    Fewer, larger writes are faster on a parallel filesystem. The cost is
    frames_per_flush frames of RAM.
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
    """Writes each frame as it arrives. Uses the least memory."""

    def __init__(self, dataset):
        self.dataset = dataset

    def write(self, frame_index, frame):
        with scorep_region("hdf5_write_vz_frame"):
            self.dataset[frame_index] = np.ascontiguousarray(frame)

    def close(self):
        pass


class AsyncFrameWriter:
    """
    Writes frames from a background thread so the time loop doesn't wait.

    The thread runs on the rank's last core and owns the HDF5 file until
    close() returns. A fixed pool of buffers makes the main thread wait if
    writing falls behind, so memory use stays bounded.
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
            # Wake the main thread if it is waiting for a buffer. It raises
            # before using this placeholder.
            try:
                self.free_buffers.put_nowait(np.empty(1, dtype=np.float32))
            except queue.Full:
                pass

    def _raise_if_failed(self):
        if self.failure is not None:
            raise RuntimeError("async HDF5 writer thread died") from self.failure

    def write(self, frame_index, frame):
        # Poll with a timeout so a dead writer thread raises an error
        # instead of hanging the job.
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
    """For ranks whose tile is entirely in the sponge."""

    def write(self, frame_index, frame):
        pass

    def close(self):
        pass


class RankOutputFile:
    """
    This rank's HDF5 file, its datasets and how frames get written.

    With STAGE_ON_LOCAL_DISK, the file is written on local disk and copied to
    the output directory on close.
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

        # Don't buffer more frames than the run produces.
        return BatchedFrameWriter(
            vz_dataset,
            frame_shape,
            min(FRAMES_PER_FLUSH, max(1, frame_count)),
        )

    def _write_attributes(self, attributes):
        window = self.window
        # combine_hdf5.py and make_video.py read these key names. Don't rename.
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
        Copy the staged file to its final path.

        Copy to a temporary name, then rename, so readers never see a partial
        file. Each rank deletes only its own staging directory.
        """
        os.makedirs(os.path.dirname(self.final_path), exist_ok=True)
        temporary_path = f"{self.final_path}.partial"

        with scorep_region("copy_staged_output"):
            shutil.copy2(self.write_path, temporary_path)
            os.replace(temporary_path, self.final_path)

        shutil.rmtree(os.path.dirname(self.write_path), ignore_errors=True)

    def placement(self):
        """Info rank 0 needs to place this tile in the virtual dataset."""
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
    Join the per-rank files into one virtual dataset.

    No data is copied. Paths are relative, so the output directory can be
    moved or renamed.
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

            # Gaps shouldn't exist. If a decomposition bug leaves one, it
            # reads as zeros.
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
    Check on rank 0 that all input files exist, and name any missing one.

    segyio's error doesn't say which file is missing.
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
    Create the output directory on rank 0 and check it is writable.

    The result goes to all ranks so they exit together. If only rank 0
    exited, the rest would wait at the barrier until the job timed out.
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
    Print loop timings and write a one-row timing.csv to the output directory.

    Timings use the slowest rank, since a step ends when the slowest rank
    finishes it.
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
    # One lattice update is one cell through one full step (stress and velocity).
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

    # Each rank reads the models itself. They are small, so this is cheaper
    # than scattering them over MPI.
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

    # The highest velocity sets the stable time step.
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

    # Short key names, read by the post-processing scripts.
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

    # Staged files must be in place before rank 0 builds the virtual dataset.
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