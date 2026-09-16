import os
import hdf5plugin
import ctypes
import segyio
import h5py
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n_iterations = int(os.environ.get("FWMP_NITER", "50000"))
frame_stride = int(os.environ.get("FWMP_FRAME_STRIDE", "100"))
direct_vz_source = int(os.environ.get("FWMP_DIRECT_VZ_SOURCE", "0"))
#ds = 1
ds = int(os.environ.get("FWMP_DS", "1"))

base_output_dir = os.environ["FWMP_BASE_OUTPUT_DIR"]
rank_output_dir = os.path.join(base_output_dir, f"rank_{rank:04d}")

vp_path = "../data/MODEL_P-WAVE_VELOCITY_1.25m.segy"
vs_path = "../data/MODEL_S-WAVE_VELOCITY_1.25m.segy"
rho_path = "../data/MODEL_DENSITY_1.25m.segy"

lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "libelastic_kernels.so"))
_float2 = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")

lib.update_stress.argtypes = [
    _float2, _float2,
    _float2, _float2, _float2,
    _float2, _float2, _float2, _float2,
    ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int
]
lib.update_stress.restype = None

lib.update_velocity.argtypes = [
    _float2, _float2,
    _float2, _float2, _float2,
    _float2, _float2,
    ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int
]
lib.update_velocity.restype = None

comm.Barrier()

if rank == 0:
    os.makedirs(base_output_dir, exist_ok=True)

comm.Barrier()
os.makedirs(rank_output_dir, exist_ok=True)
comm.Barrier()

rank_h5_path = os.path.join(rank_output_dir, "elastic_wavefield.h5")
vds_path = os.path.join(base_output_dir, "elastic_wavefield.h5")

def load_segy(path):
    with segyio.open(path, "r", ignore_geometry=True) as f:
        return np.stack([np.array(tr) for tr in f.trace]).T

def split_1d(total, parts, coord):
    counts = [total // parts + (1 if r < total % parts else 0) for r in range(parts)]
    starts = np.cumsum([0] + counts[:-1])
    start = int(starts[coord])
    end = int(start + counts[coord])
    return start, end, counts

def choose_dims(nprocs, nz, nx):
    best = None
    best_score = None
    for pz in range(1, nprocs + 1):
        if nprocs % pz != 0:
            continue
        px = nprocs // pz
        if pz > nz or px > nx:
            continue
        nz_loc = nz / pz
        nx_loc = nx / px
        score = nz_loc + nx_loc
        if best_score is None or score < best_score:
            best_score = score
            best = (pz, px)
    if best is None:
        best = MPI.Compute_dims(nprocs, 2)
    return [int(best[0]), int(best[1])]

def add_halo_2d(a):
    out = np.empty((a.shape[0] + 2, a.shape[1] + 2), dtype=a.dtype)
    out[1:-1, 1:-1] = a
    out[0, 1:-1] = out[1, 1:-1]
    out[-1, 1:-1] = out[-2, 1:-1]
    out[:, 0] = out[:, 1]
    out[:, -1] = out[:, -2]
    return out

def make_halo_buffers(nz_loc, nx_loc, max_fields=3, dtype=np.float32):
    return {
        "x_send_minus": np.empty((max_fields, nz_loc), dtype=dtype),
        "x_recv_minus": np.empty((max_fields, nz_loc), dtype=dtype),
        "x_send_plus": np.empty((max_fields, nz_loc), dtype=dtype),
        "x_recv_plus": np.empty((max_fields, nz_loc), dtype=dtype),
        "z_send_minus": np.empty((max_fields, nx_loc), dtype=dtype),
        "z_recv_minus": np.empty((max_fields, nx_loc), dtype=dtype),
        "z_send_plus": np.empty((max_fields, nx_loc), dtype=dtype),
        "z_recv_plus": np.empty((max_fields, nx_loc), dtype=dtype),
    }

def exchange_forward_halos(fields):
    nf = len(fields)

    send_minus = halo_bufs["x_send_minus"][:nf]
    recv_plus = halo_bufs["x_recv_plus"][:nf]

    for k, a in enumerate(fields):
        send_minus[k, :] = a[1:-1, 1]

    cart.Sendrecv(send_minus, dest=x_minus, sendtag=10, recvbuf=recv_plus, source=x_plus, recvtag=10)

    if x_plus != MPI.PROC_NULL:
        for k, a in enumerate(fields):
            a[1:-1, -1] = recv_plus[k, :]

    send_minus = halo_bufs["z_send_minus"][:nf]
    recv_plus = halo_bufs["z_recv_plus"][:nf]

    for k, a in enumerate(fields):
        send_minus[k, :] = a[1, 1:-1]

    cart.Sendrecv(send_minus, dest=z_minus, sendtag=20, recvbuf=recv_plus, source=z_plus, recvtag=20)

    if z_plus != MPI.PROC_NULL:
        for k, a in enumerate(fields):
            a[-1, 1:-1] = recv_plus[k, :]

def exchange_backward_halos(fields):
    nf = len(fields)

    send_plus = halo_bufs["x_send_plus"][:nf]
    recv_minus = halo_bufs["x_recv_minus"][:nf]

    for k, a in enumerate(fields):
        send_plus[k, :] = a[1:-1, -2]

    cart.Sendrecv(send_plus, dest=x_plus, sendtag=30, recvbuf=recv_minus, source=x_minus, recvtag=30)

    if x_minus != MPI.PROC_NULL:
        for k, a in enumerate(fields):
            a[1:-1, 0] = recv_minus[k, :]

    send_plus = halo_bufs["z_send_plus"][:nf]
    recv_minus = halo_bufs["z_recv_minus"][:nf]

    for k, a in enumerate(fields):
        send_plus[k, :] = a[-2, 1:-1]

    cart.Sendrecv(send_plus, dest=z_plus, sendtag=40, recvbuf=recv_minus, source=z_minus, recvtag=40)

    if z_minus != MPI.PROC_NULL:
        for k, a in enumerate(fields):
            a[0, 1:-1] = recv_minus[k, :]

def update_stress_c(vx, vz, sxx, szz, sxz, lam, lam2mu, mu, damp, dt, dx, dz, iz0, iz1, jx0, jx1):
    lib.update_stress(
        vx, vz,
        sxx, szz, sxz,
        lam, lam2mu, mu, damp,
        np.float32(dt), np.float32(dx), np.float32(dz),
        vx.shape[0], vx.shape[1],
        iz0, iz1,
        jx0, jx1
    )

def update_velocity_c(vx, vz, sxx, szz, sxz, inv_rho, damp, dt, dx, dz, iz0, iz1, jx0, jx1):
    lib.update_velocity(
        vx, vz, sxx, szz, sxz,
        inv_rho, damp,
        np.float32(dt), np.float32(dx), np.float32(dz),
        vx.shape[0], vx.shape[1],
        iz0, iz1,
        jx0, jx1
    )

vp0 = load_segy(vp_path)[::ds, ::ds].astype(np.float32)
vs0 = load_segy(vs_path)[::ds, ::ds].astype(np.float32)
rho0 = load_segy(rho_path)[::ds, ::ds].astype(np.float32)

nz0, nx0 = vp0.shape
dx = np.float32(1.25 * ds)
dz = np.float32(1.25 * ds)

nb = 240
pad_top = nb
pad_bottom = nb
pad_left = nb
pad_right = nb

nz = nz0 + pad_top + pad_bottom
nx = nx0 + pad_left + pad_right

dims = choose_dims(size, nz, nx)
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

def pad_field(f0):
    f = np.empty((nz, nx), dtype=np.float32)
    f[pad_top:pad_top + nz0, pad_left:pad_left + nx0] = f0
    f[:, :pad_left] = f[:, pad_left:pad_left + 1]
    f[:, pad_left + nx0:] = f[:, pad_left + nx0 - 1:pad_left + nx0]
    f[:pad_top, :] = f[pad_top:pad_top + 1, :]
    f[pad_top + nz0:, :] = f[pad_top + nz0 - 1:pad_top + nz0, :]
    return f

vp = pad_field(vp0)
vs = pad_field(vs0)
rho = pad_field(rho0)

mu = (rho * vs ** 2).astype(np.float32)
lam = (rho * vp ** 2 - 2.0 * mu).astype(np.float32)
lam2mu = (lam + 2.0 * mu).astype(np.float32)
inv_rho = (1.0 / rho).astype(np.float32)

vp_max = float(vp.max())
dt = np.float32(0.4 * float(dx) / vp_max)

f0 = np.float32(8.0)
src_t0 = np.float32(1.2 / f0)
src_amp = np.float32(1e9)

def ricker(t):
    a = (np.pi * float(f0) * (float(t) - float(src_t0))) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)

src_x0 = nx0 // 2
src_z0 = 1
src_x = pad_left + src_x0
src_z = pad_top + src_z0

sigma = np.zeros((nz, nx), dtype=np.float32)

def ramp(n, power=2.0):
    return np.linspace(0.0, 1.0, n, dtype=np.float32) ** power

r = ramp(pad_left)
for i in range(pad_left):
    sigma[:, i] = np.maximum(sigma[:, i], 60.0 * r[pad_left - 1 - i])

r = ramp(pad_right)
for i in range(pad_right):
    sigma[:, -1 - i] = np.maximum(sigma[:, -1 - i], 60.0 * r[pad_right - 1 - i])

r = ramp(pad_top)
for i in range(pad_top):
    sigma[i, :] = np.maximum(sigma[i, :], 60.0 * r[pad_top - 1 - i])

r = ramp(pad_bottom)
for i in range(pad_bottom):
    sigma[-1 - i, :] = np.maximum(sigma[-1 - i, :], 120.0 * r[pad_bottom - 1 - i])

damp = np.clip(1.0 - sigma * float(dt), 0.0, 1.0).astype(np.float32)

mu_loc = np.ascontiguousarray(add_halo_2d(mu[z_start:z_end, x_start:x_end]))
lam_loc = np.ascontiguousarray(add_halo_2d(lam[z_start:z_end, x_start:x_end]))
lam2mu_loc = np.ascontiguousarray(add_halo_2d(lam2mu[z_start:z_end, x_start:x_end]))
inv_rho_loc = np.ascontiguousarray(add_halo_2d(inv_rho[z_start:z_end, x_start:x_end]))
damp_loc = np.ascontiguousarray(add_halo_2d(damp[z_start:z_end, x_start:x_end]))

vx = np.zeros((nz_loc + 2, nx_loc + 2), dtype=np.float32)
vz = np.zeros((nz_loc + 2, nx_loc + 2), dtype=np.float32)
sxx = np.zeros((nz_loc + 2, nx_loc + 2), dtype=np.float32)
szz = np.zeros((nz_loc + 2, nx_loc + 2), dtype=np.float32)
sxz = np.zeros((nz_loc + 2, nx_loc + 2), dtype=np.float32)

halo_bufs = make_halo_buffers(nz_loc, nx_loc)

iz0 = 1
iz1 = nz_loc + 1

if z_start == 0:
    iz0 = 2

if z_end == nz:
    iz1 = nz_loc

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
    local_i0 = 0
    local_i1 = 0
    local_j0 = 0
    local_j1 = 0
    out_z0 = -1
    out_z1 = -1
    out_x0 = -1
    out_x1 = -1
    local_nz_phys = 0
    local_nx_phys = 0

n_frames = len(range(0, n_iterations, frame_stride))

comm.Barrier()

h5 = h5py.File(rank_h5_path, "w")

cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
if has_physical_output:
    frame_bytes = local_nz_phys * local_nx_phys * 4
    max_chunk_bytes = cpus_per_task * 8 * 1024**2

    if frame_bytes <= max_chunk_bytes:
        chunk_z = local_nz_phys
        chunk_x = local_nx_phys
    else:
        scale = (max_chunk_bytes / 4 / (local_nz_phys * local_nx_phys)) ** 0.5
        chunk_z = max(1, min(local_nz_phys, int(local_nz_phys * scale)))
        chunk_x = max(1, min(local_nx_phys, int(local_nx_phys * scale)))

    dset_vz = h5.create_dataset(
        "vz",
        shape=(n_frames, local_nz_phys, local_nx_phys),
        dtype=np.float32,
        chunks=(1, chunk_z, chunk_x),
        **hdf5plugin.Blosc(
            cname="lz4",
            clevel=3,
            shuffle=hdf5plugin.Blosc.SHUFFLE,
        )
    )
    h5.create_dataset("vp", data=vp0[out_z0:out_z1, out_x0:out_x1].astype(np.float32))
else:
    dset_vz = None

times = np.arange(n_frames, dtype=np.float32) * frame_stride * dt
h5.create_dataset("time", data=times)

h5.attrs["rank"] = rank
h5.attrs["size"] = size
h5.attrs["dims_z"] = dims[0]
h5.attrs["dims_x"] = dims[1]
h5.attrs["coord_z"] = coord_z
h5.attrs["coord_x"] = coord_x
h5.attrs["z0"] = out_z0
h5.attrs["z1"] = out_z1
h5.attrs["x0"] = out_x0
h5.attrs["x1"] = out_x1
h5.attrs["local_nz_phys"] = local_nz_phys
h5.attrs["local_nx_phys"] = local_nx_phys
h5.attrs["nz0"] = nz0
h5.attrs["nx0"] = nx0
h5.attrs["dx"] = float(dx)
h5.attrs["dz"] = float(dz)
h5.attrs["dt"] = float(dt)
h5.attrs["frame_stride"] = frame_stride
h5.attrs["n_frames"] = n_frames

update_stress_c(vx, vz, sxx, szz, sxz, lam_loc, lam2mu_loc, mu_loc, damp_loc, dt, dx, dz, iz0, iz1, jx0, jx1)
update_velocity_c(vx, vz, sxx, szz, sxz, inv_rho_loc, damp_loc, dt, dx, dz, iz0, iz1, jx0, jx1)

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

    update_stress_c(vx, vz, sxx, szz, sxz, lam_loc, lam2mu_loc, mu_loc, damp_loc, dt, dx, dz, iz0, iz1, jx0, jx1)

    exchange_backward_halos([sxx, szz, sxz])

    update_velocity_c(vx, vz, sxx, szz, sxz, inv_rho_loc, damp_loc, dt, dx, dz, iz0, iz1, jx0, jx1)

frame_id = 0

for it in range(n_iterations):
    step(it)

    if it % frame_stride == 0:
        if has_physical_output:
            local_view = np.ascontiguousarray(vz[local_i0:local_i1, local_j0:local_j1])
            dset_vz[frame_id] = local_view
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
        layout = h5py.VirtualLayout(shape=(n_frames, nz0, nx0), dtype=np.float32)

        for m in all_meta:
            if not m["has"]:
                continue

            src_shape = (n_frames, m["z1"] - m["z0"], m["x1"] - m["x0"])
            source = h5py.VirtualSource(m["path"], "vz", shape=src_shape)
            layout[:, m["z0"]:m["z1"], m["x0"]:m["x1"]] = source

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
