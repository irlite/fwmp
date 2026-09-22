import os
import ctypes
import hdf5plugin
import h5py
import numpy as np
import segyio
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n_iterations = int(os.environ.get("FWMP_NITER", "50000"))
frame_stride = int(os.environ.get("FWMP_FRAME_STRIDE", "100"))
direct_vz_source = int(os.environ.get("FWMP_DIRECT_VZ_SOURCE", "0"))
scale_mode = os.environ.get("FWMP_SCALE_MODE", "none")
base_ds = float(os.environ.get("FWMP_BASE_DS", "39"))
base_cores = float(os.environ.get("FWMP_BASE_CORES", "1"))
total_cores = int(os.environ.get("FWMP_TOTAL_CORES", str(size)))
ds_legacy = int(os.environ.get("FWMP_DS", "1"))
cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("OMP_NUM_THREADS", "1")))

base_output_dir = os.environ["FWMP_BASE_OUTPUT_DIR"]
rank_output_dir = os.path.join(base_output_dir, f"rank_{rank:04d}")

vp_path = "../data/MODEL_P-WAVE_VELOCITY_1.25m.segy"
vs_path = "../data/MODEL_S-WAVE_VELOCITY_1.25m.segy"
rho_path = "../data/MODEL_DENSITY_1.25m.segy"
lib = ctypes.CDLL(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'src',
    os.environ.get("FWMP_KERNEL_LIB", "libelastic_kernels_tiled.so"),
))
_float2 = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")

lib.update_stress_edges.argtypes = [
    _float2, _float2,
    _float2, _float2, _float2,
    _float2, _float2, _float2, _float2,
    ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int
]
lib.update_stress_edges.restype = None

lib.update_stress_velocity_interior.argtypes = [
    _float2, _float2,
    _float2, _float2, _float2,
    _float2, _float2, _float2, _float2, _float2,
    ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int
]
lib.update_stress_velocity_interior.restype = None

lib.update_velocity_boundary.argtypes = [
    _float2, _float2,
    _float2, _float2, _float2,
    _float2, _float2,
    ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int
]
lib.update_velocity_boundary.restype = None

if rank == 0:
    os.makedirs(base_output_dir, exist_ok=True)

comm.Barrier()
os.makedirs(rank_output_dir, exist_ok=True)
comm.Barrier()

rank_h5_path = os.path.join(rank_output_dir, "elastic_wavefield.h5")
vds_path = os.path.join(base_output_dir, "elastic_wavefield.h5")

def load_segy(path):
    with segyio.open(path, "r", ignore_geometry=True) as f:
        return np.stack([np.asarray(trace) for trace in f.trace]).T

def resample_field(a, nz_target, nx_target):
    nz_source, nx_source = a.shape
    iz = np.linspace(0, nz_source - 1, nz_target).astype(np.int64)
    ix = np.linspace(0, nx_source - 1, nx_target).astype(np.int64)
    return a[np.ix_(iz, ix)]

def split_1d(total, parts, coordinate):
    counts = [total // parts + (1 if p < total % parts else 0) for p in range(parts)]
    starts = np.cumsum([0] + counts[:-1])
    start = int(starts[coordinate])
    end = start + counts[coordinate]
    return start, end, counts

def choose_dims(nprocs, nz, nx, threads):
    best = None
    best_score = None
    target_rows = max(1, 4 * threads)

    for pz in range(1, nprocs + 1):
        if nprocs % pz:
            continue

        px = nprocs // pz

        if pz > nz or px > nx:
            continue

        local_z = nz / pz
        local_x = nx / px
        penalty = 0.0

        if local_z < target_rows:
            penalty = 1.0e6 * (target_rows - local_z)

        score = local_z + local_x + penalty

        if best_score is None or score < best_score:
            best = (pz, px)
            best_score = score

    if best is None:
        best = MPI.Compute_dims(nprocs, 2)

    return [int(best[0]), int(best[1])]

def add_halo_2d(a):
    result = np.empty((a.shape[0] + 2, a.shape[1] + 2), dtype=a.dtype)
    result[1:-1, 1:-1] = a
    result[0, 1:-1] = result[1, 1:-1]
    result[-1, 1:-1] = result[-2, 1:-1]
    result[:, 0] = result[:, 1]
    result[:, -1] = result[:, -2]
    return result

def make_halo_buffers(nz_local, nx_local, max_fields=3):
    return {
        "x_send_minus": np.empty((max_fields, nz_local), dtype=np.float32),
        "x_recv_minus": np.empty((max_fields, nz_local), dtype=np.float32),
        "x_send_plus": np.empty((max_fields, nz_local), dtype=np.float32),
        "x_recv_plus": np.empty((max_fields, nz_local), dtype=np.float32),
        "z_send_minus": np.empty((max_fields, nx_local), dtype=np.float32),
        "z_recv_minus": np.empty((max_fields, nx_local), dtype=np.float32),
        "z_send_plus": np.empty((max_fields, nx_local), dtype=np.float32),
        "z_recv_plus": np.empty((max_fields, nx_local), dtype=np.float32)
    }

def exchange_forward_halos(fields):
    nf = len(fields)
    x_send = halo_bufs["x_send_minus"][:nf]
    x_recv = halo_bufs["x_recv_plus"][:nf]

    for k, field in enumerate(fields):
        x_send[k] = field[1:-1, 1]

    cart.Sendrecv(x_send, x_minus, 10, x_recv, x_plus, 10)

    if x_plus != MPI.PROC_NULL:
        for k, field in enumerate(fields):
            field[1:-1, -1] = x_recv[k]

    z_send = halo_bufs["z_send_minus"][:nf]
    z_recv = halo_bufs["z_recv_plus"][:nf]

    for k, field in enumerate(fields):
        z_send[k] = field[1, 1:-1]

    cart.Sendrecv(z_send, z_minus, 20, z_recv, z_plus, 20)

    if z_plus != MPI.PROC_NULL:
        for k, field in enumerate(fields):
            field[-1, 1:-1] = z_recv[k]

def begin_backward_halos(fields):
    nf = len(fields)
    x_send = halo_bufs["x_send_plus"][:nf]
    x_recv = halo_bufs["x_recv_minus"][:nf]
    z_send = halo_bufs["z_send_plus"][:nf]
    z_recv = halo_bufs["z_recv_minus"][:nf]

    for k, field in enumerate(fields):
        x_send[k] = field[1:-1, -2]
        z_send[k] = field[-2, 1:-1]

    requests = [
        cart.Irecv(x_recv, source=x_minus, tag=30),
        cart.Irecv(z_recv, source=z_minus, tag=40),
        cart.Isend(x_send, dest=x_plus, tag=30),
        cart.Isend(z_send, dest=z_plus, tag=40)
    ]

    return requests, nf

def finish_backward_halos(fields, state):
    requests, nf = state
    MPI.Request.Waitall(requests)

    if x_minus != MPI.PROC_NULL:
        x_recv = halo_bufs["x_recv_minus"][:nf]
        for k, field in enumerate(fields):
            field[1:-1, 0] = x_recv[k]

    if z_minus != MPI.PROC_NULL:
        z_recv = halo_bufs["z_recv_minus"][:nf]
        for k, field in enumerate(fields):
            field[0, 1:-1] = z_recv[k]

def update_stress_edges_c(vx, vz, sxx, szz, sxz, lam, lam2mu, mu, damp, dt, dx, dz, iz0, iz1, jx0, jx1):
    lib.update_stress_edges(
        vx, vz, sxx, szz, sxz,
        lam, lam2mu, mu, damp,
        np.float32(dt), np.float32(dx), np.float32(dz),
        vx.shape[0], vx.shape[1],
        iz0, iz1, jx0, jx1
    )

def update_stress_velocity_interior_c(vx, vz, sxx, szz, sxz, lam, lam2mu, mu, inv_rho, damp, dt, dx, dz, iz0, iz1, jx0, jx1):
    lib.update_stress_velocity_interior(
        vx, vz, sxx, szz, sxz,
        lam, lam2mu, mu, inv_rho, damp,
        np.float32(dt), np.float32(dx), np.float32(dz),
        vx.shape[0], vx.shape[1],
        iz0, iz1, jx0, jx1
    )

def update_velocity_boundary_c(vx, vz, sxx, szz, sxz, inv_rho, damp, dt, dx, dz, iz0, iz1, jx0, jx1):
    lib.update_velocity_boundary(
        vx, vz, sxx, szz, sxz,
        inv_rho, damp,
        np.float32(dt), np.float32(dx), np.float32(dz),
        vx.shape[0], vx.shape[1],
        iz0, iz1, jx0, jx1
    )

vp_full = load_segy(vp_path).astype(np.float32)
vs_full = load_segy(vs_path).astype(np.float32)
rho_full = load_segy(rho_path).astype(np.float32)

nz_full, nx_full = vp_full.shape

if scale_mode == "weak":
    ds_exact = base_ds * (base_cores / total_cores) ** 0.5
    nz0 = max(1, round(nz_full / ds_exact))
    nx0 = max(1, round(nx_full / ds_exact))
    dz = np.float32(1.25 * nz_full / nz0)
    dx = np.float32(1.25 * nx_full / nx0)
elif scale_mode == "strong":
    ds_exact = 1.0
    nz0 = nz_full
    nx0 = nx_full
    dz = np.float32(1.25)
    dx = np.float32(1.25)
else:
    ds_exact = float(ds_legacy)
    nz0 = max(1, round(nz_full / ds_exact))
    nx0 = max(1, round(nx_full / ds_exact))
    dz = np.float32(1.25 * nz_full / nz0)
    dx = np.float32(1.25 * nx_full / nx0)

vp0 = resample_field(vp_full, nz0, nx0)
vs0 = resample_field(vs_full, nz0, nx0)
rho0 = resample_field(rho_full, nz0, nx0)

nb = max(8, round(240 / ds_exact))
pad_top = nb
pad_bottom = nb
pad_left = nb
pad_right = nb
nz = nz0 + pad_top + pad_bottom
nx = nx0 + pad_left + pad_right

dims = choose_dims(size, nz, nx, cpus_per_task)
cart = comm.Create_cart(dims=dims, periods=[False, False], reorder=False)
coord_z, coord_x = cart.Get_coords(rank)
z_minus, z_plus = cart.Shift(0, 1)
x_minus, x_plus = cart.Shift(1, 1)

z_start, z_end, z_counts = split_1d(nz, dims[0], coord_z)
x_start, x_end, x_counts = split_1d(nx, dims[1], coord_x)
nz_loc = z_end - z_start
nx_loc = x_end - x_start

if nz_loc <= 0 or nx_loc <= 0:
    raise RuntimeError("empty local domain")

def pad_field(field):
    result = np.empty((nz, nx), dtype=np.float32)
    result[pad_top:pad_top + nz0, pad_left:pad_left + nx0] = field
    result[:, :pad_left] = result[:, pad_left:pad_left + 1]
    result[:, pad_left + nx0:] = result[:, pad_left + nx0 - 1:pad_left + nx0]
    result[:pad_top] = result[pad_top:pad_top + 1]
    result[pad_top + nz0:] = result[pad_top + nz0 - 1:pad_top + nz0]
    return result

vp = pad_field(vp0)
vs = pad_field(vs0)
rho = pad_field(rho0)

mu = (rho * vs ** 2).astype(np.float32)
lam = (rho * vp ** 2 - 2.0 * mu).astype(np.float32)
lam2mu = (lam + 2.0 * mu).astype(np.float32)
inv_rho = (1.0 / rho).astype(np.float32)

dt = np.float32(0.4 * float(min(dx, dz)) / float(vp.max()))
f0 = np.float32(8.0)
src_t0 = np.float32(1.2 / f0)
src_amp = np.float32(1e9)

def ricker(t):
    value = (np.pi * float(f0) * (float(t) - float(src_t0))) ** 2
    return (1.0 - 2.0 * value) * np.exp(-value)

src_x = pad_left + nx0 // 2
src_z = pad_top + 1

sigma = np.zeros((nz, nx), dtype=np.float32)

def ramp(n):
    return np.linspace(0.0, 1.0, n, dtype=np.float32) ** 2

r = ramp(pad_left)
for i in range(pad_left):
    sigma[:, i] = np.maximum(sigma[:, i], 60.0 * r[pad_left - 1 - i])

r = ramp(pad_right)
for i in range(pad_right):
    sigma[:, -1 - i] = np.maximum(sigma[:, -1 - i], 60.0 * r[pad_right - 1 - i])

r = ramp(pad_top)
for i in range(pad_top):
    sigma[i] = np.maximum(sigma[i], 60.0 * r[pad_top - 1 - i])

r = ramp(pad_bottom)
for i in range(pad_bottom):
    sigma[-1 - i] = np.maximum(sigma[-1 - i], 120.0 * r[pad_bottom - 1 - i])

damp = np.clip(1.0 - sigma * float(dt), 0.0, 1.0).astype(np.float32)

mu_loc = np.ascontiguousarray(add_halo_2d(mu[z_start:z_end, x_start:x_end]))
lam_loc = np.ascontiguousarray(add_halo_2d(lam[z_start:z_end, x_start:x_end]))
lam2mu_loc = np.ascontiguousarray(add_halo_2d(lam2mu[z_start:z_end, x_start:x_end]))
inv_rho_loc = np.ascontiguousarray(add_halo_2d(inv_rho[z_start:z_end, x_start:x_end]))
damp_loc = np.ascontiguousarray(add_halo_2d(damp[z_start:z_end, x_start:x_end]))

shape = (nz_loc + 2, nx_loc + 2)
vx = np.zeros(shape, dtype=np.float32)
vz = np.zeros(shape, dtype=np.float32)
sxx = np.zeros(shape, dtype=np.float32)
szz = np.zeros(shape, dtype=np.float32)
sxz = np.zeros(shape, dtype=np.float32)
halo_bufs = make_halo_buffers(nz_loc, nx_loc)

iz0 = 2 if z_start == 0 else 1
iz1 = nz_loc if z_end == nz else nz_loc + 1
jx0 = 1
jx1 = nx_loc + 1

physical_z_start = pad_top
physical_z_end = pad_top + nz0
physical_x_start = pad_left
physical_x_end = pad_left + nx0

write_z0 = max(z_start, physical_z_start)
write_z1 = min(z_end, physical_z_end)
write_x0 = max(x_start, physical_x_start)
write_x1 = min(x_end, physical_x_end)
has_physical_output = write_z0 < write_z1 and write_x0 < write_x1

if has_physical_output:
    local_i0 = write_z0 - z_start + 1
    local_i1 = write_z1 - z_start + 1
    local_j0 = write_x0 - x_start + 1
    local_j1 = write_x1 - x_start + 1
    out_z0 = write_z0 - pad_top
    out_z1 = write_z1 - pad_top
    out_x0 = write_x0 - pad_left
    out_x1 = write_x1 - pad_left
    local_nz_phys = out_z1 - out_z0
    local_nx_phys = out_x1 - out_x0
else:
    local_i0 = local_i1 = local_j0 = local_j1 = 0
    out_z0 = out_z1 = out_x0 = out_x1 = -1
    local_nz_phys = local_nx_phys = 0

n_frames = len(range(0, n_iterations, frame_stride))

comm.Barrier()
h5 = h5py.File(rank_h5_path, "w")

if has_physical_output:
    frame_bytes = local_nz_phys * local_nx_phys * 4
    max_chunk_bytes = cpus_per_task * 8 * 1024 ** 2

    if frame_bytes <= max_chunk_bytes:
        chunk_z = local_nz_phys
        chunk_x = local_nx_phys
    else:
        chunk_scale = (max_chunk_bytes / (4 * local_nz_phys * local_nx_phys)) ** 0.5
        chunk_z = max(1, min(local_nz_phys, int(local_nz_phys * chunk_scale)))
        chunk_x = max(1, min(local_nx_phys, int(local_nx_phys * chunk_scale)))

    dset_vz = h5.create_dataset(
        "vz",
        shape=(n_frames, local_nz_phys, local_nx_phys),
        dtype=np.float32,
        chunks=(1, chunk_z, chunk_x),
        **hdf5plugin.Blosc(cname="lz4", clevel=3, shuffle=hdf5plugin.Blosc.SHUFFLE)
    )
    h5.create_dataset("vp", data=vp0[out_z0:out_z1, out_x0:out_x1].astype(np.float32))
else:
    dset_vz = None

times = np.arange(n_frames, dtype=np.float32) * frame_stride * dt
h5.create_dataset("time", data=times)

attributes = {
    "rank": rank,
    "size": size,
    "dims_z": dims[0],
    "dims_x": dims[1],
    "coord_z": coord_z,
    "coord_x": coord_x,
    "z0": out_z0,
    "z1": out_z1,
    "x0": out_x0,
    "x1": out_x1,
    "local_nz_phys": local_nz_phys,
    "local_nx_phys": local_nx_phys,
    "nz0": nz0,
    "nx0": nx0,
    "dx": float(dx),
    "dz": float(dz),
    "dt": float(dt),
    "frame_stride": frame_stride,
    "n_frames": n_frames,
    "scale_mode": scale_mode,
    "total_cores": total_cores
}

for key, value in attributes.items():
    h5.attrs[key] = value

update_stress_edges_c(
    vx, vz, sxx, szz, sxz,
    lam_loc, lam2mu_loc, mu_loc, damp_loc,
    dt, dx, dz, iz0, iz1, jx0, jx1
)

update_stress_velocity_interior_c(
    vx, vz, sxx, szz, sxz,
    lam_loc, lam2mu_loc, mu_loc, inv_rho_loc, damp_loc,
    dt, dx, dz, iz0, iz1, jx0, jx1
)

update_velocity_boundary_c(
    vx, vz, sxx, szz, sxz,
    inv_rho_loc, damp_loc,
    dt, dx, dz, iz0, iz1, jx0, jx1
)

vx.fill(0.0)
vz.fill(0.0)
sxx.fill(0.0)
szz.fill(0.0)
sxz.fill(0.0)

def step(it):
    exchange_forward_halos([vx, vz])

    src = np.float32(src_amp * ricker(np.float32(it) * dt))

    if z_start <= src_z < z_end and x_start <= src_x < x_end:
        li = src_z - z_start + 1
        lj = src_x - x_start + 1
        sxx[li, lj] += src
        szz[li, lj] += src

        if direct_vz_source:
            vz[li, lj] += np.float32(dt * inv_rho_loc[li, lj] * src)

    update_stress_edges_c(
        vx, vz, sxx, szz, sxz,
        lam_loc, lam2mu_loc, mu_loc, damp_loc,
        dt, dx, dz, iz0, iz1, jx0, jx1
    )

    exchange_state = begin_backward_halos([sxx, szz, sxz])

    update_stress_velocity_interior_c(
        vx, vz, sxx, szz, sxz,
        lam_loc, lam2mu_loc, mu_loc, inv_rho_loc, damp_loc,
        dt, dx, dz, iz0, iz1, jx0, jx1
    )

    finish_backward_halos([sxx, szz, sxz], exchange_state)

    update_velocity_boundary_c(
        vx, vz, sxx, szz, sxz,
        inv_rho_loc, damp_loc,
        dt, dx, dz, iz0, iz1, jx0, jx1
    )

frame_id = 0

for it in range(n_iterations):
    step(it)

    if it % frame_stride == 0:
        if has_physical_output:
            dset_vz[frame_id] = np.ascontiguousarray(
                vz[local_i0:local_i1, local_j0:local_j1]
            )
        frame_id += 1

h5.close()

meta = {
    "rank": rank,
    "has": bool(has_physical_output),
    "z0": int(out_z0),
    "z1": int(out_z1),
    "x0": int(out_x0),
    "x1": int(out_x1),
    "path": os.path.join(f"rank_{rank:04d}", "elastic_wavefield.h5")
}

all_meta = comm.gather(meta, root=0)
comm.Barrier()

if rank == 0:
    with h5py.File(vds_path, "w", libver="latest") as vf:
        layout = h5py.VirtualLayout(
            shape=(n_frames, nz0, nx0),
            dtype=np.float32
        )

        for item in all_meta:
            if not item["has"]:
                continue

            source_shape = (
                n_frames,
                item["z1"] - item["z0"],
                item["x1"] - item["x0"]
            )

            source = h5py.VirtualSource(
                item["path"],
                "vz",
                shape=source_shape
            )

            layout[
                :,
                item["z0"]:item["z1"],
                item["x0"]:item["x1"]
            ] = source

        vf.create_virtual_dataset("vz", layout, fillvalue=0.0)
        vf.create_dataset("vp", data=vp0.astype(np.float32))
        vf.create_dataset("time", data=times)
        vf.attrs["size"] = size
        vf.attrs["dims_z"] = dims[0]
        vf.attrs["dims_x"] = dims[1]
        vf.attrs["nz0"] = nz0
        vf.attrs["nx0"] = nx0
        vf.attrs["dx"] = float(dx)
        vf.attrs["dz"] = float(dz)
        vf.attrs["dt"] = float(dt)
        vf.attrs["frame_stride"] = frame_stride
        vf.attrs["n_frames"] = n_frames

    print("done", flush=True)