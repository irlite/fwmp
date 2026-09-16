import os
import ctypes
import segyio
import h5py
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n_iterations = int(os.environ.get("FWMP_NITER", "6000"))
frame_stride = int(os.environ.get("FWMP_FRAME_STRIDE", "10"))
direct_vz_source = int(os.environ.get("FWMP_DIRECT_VZ_SOURCE", "0"))
ds = 4

slurm_job_id = os.environ.get("SLURM_JOB_ID", "noslurm")
base_output_dir = os.path.join("..", "output", f"job_{slurm_job_id}")
rank_output_dir = os.path.join(base_output_dir, f"rank_{rank:04d}")

vp_path = "../data/MODEL_P-WAVE_VELOCITY_1.25m.segy"
vs_path = "../data/MODEL_S-WAVE_VELOCITY_1.25m.segy"
rho_path = "../data/MODEL_DENSITY_1.25m.segy"

lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "libelastic_kernels_1d.so"))
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

def split_1d(total_columns, size, rank):
    counts = [total_columns // size + (1 if r < total_columns % size else 0) for r in range(size)]
    starts = np.cumsum([0] + counts[:-1])
    start = int(starts[rank])
    end = int(start + counts[rank])
    return start, end, counts

def add_halo_columns(a):
    out = np.empty((a.shape[0], a.shape[1] + 2), dtype=a.dtype)
    out[:, 1:-1] = a
    out[:, 0] = out[:, 1]
    out[:, -1] = out[:, -2]
    return out

def make_halo_buffers(nz, dtype=np.float32):
    return {
        "x2_send_left": np.empty((nz, 2), dtype=dtype),
        "x2_recv_left": np.empty((nz, 2), dtype=dtype),
        "x2_send_right": np.empty((nz, 2), dtype=dtype),
        "x2_recv_right": np.empty((nz, 2), dtype=dtype),
        "x3_send_left": np.empty((nz, 3), dtype=dtype),
        "x3_recv_left": np.empty((nz, 3), dtype=dtype),
        "x3_send_right": np.empty((nz, 3), dtype=dtype),
        "x3_recv_right": np.empty((nz, 3), dtype=dtype),
    }

def exchange_halo_x2(a, b):
    left = rank - 1 if rank > 0 else MPI.PROC_NULL
    right = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    send_left = halo_bufs["x2_send_left"]
    recv_right = halo_bufs["x2_recv_right"]
    send_right = halo_bufs["x2_send_right"]
    recv_left = halo_bufs["x2_recv_left"]

    send_left[:, 0] = a[:, 1]
    send_left[:, 1] = b[:, 1]

    comm.Sendrecv(send_left, dest=left, sendtag=10, recvbuf=recv_right, source=right, recvtag=10)

    if right != MPI.PROC_NULL:
        a[:, -1] = recv_right[:, 0]
        b[:, -1] = recv_right[:, 1]

    send_right[:, 0] = a[:, -2]
    send_right[:, 1] = b[:, -2]

    comm.Sendrecv(send_right, dest=right, sendtag=20, recvbuf=recv_left, source=left, recvtag=20)

    if left != MPI.PROC_NULL:
        a[:, 0] = recv_left[:, 0]
        b[:, 0] = recv_left[:, 1]

def exchange_halo_x3(a, b, c):
    left = rank - 1 if rank > 0 else MPI.PROC_NULL
    right = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    send_left = halo_bufs["x3_send_left"]
    recv_right = halo_bufs["x3_recv_right"]
    send_right = halo_bufs["x3_send_right"]
    recv_left = halo_bufs["x3_recv_left"]

    send_left[:, 0] = a[:, 1]
    send_left[:, 1] = b[:, 1]
    send_left[:, 2] = c[:, 1]

    comm.Sendrecv(send_left, dest=left, sendtag=30, recvbuf=recv_right, source=right, recvtag=30)

    if right != MPI.PROC_NULL:
        a[:, -1] = recv_right[:, 0]
        b[:, -1] = recv_right[:, 1]
        c[:, -1] = recv_right[:, 2]

    send_right[:, 0] = a[:, -2]
    send_right[:, 1] = b[:, -2]
    send_right[:, 2] = c[:, -2]

    comm.Sendrecv(send_right, dest=right, sendtag=40, recvbuf=recv_left, source=left, recvtag=40)

    if left != MPI.PROC_NULL:
        a[:, 0] = recv_left[:, 0]
        b[:, 0] = recv_left[:, 1]
        c[:, 0] = recv_left[:, 2]

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
        vx, vz,
        sxx, szz, sxz,
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

x_start, x_end, x_counts = split_1d(nx, size, rank)
nx_loc = x_end - x_start

if nx_loc <= 0:
    raise RuntimeError("empty local domain")

mu_loc = np.ascontiguousarray(add_halo_columns(mu[:, x_start:x_end]))
lam_loc = np.ascontiguousarray(add_halo_columns(lam[:, x_start:x_end]))
lam2mu_loc = np.ascontiguousarray(add_halo_columns(lam2mu[:, x_start:x_end]))
inv_rho_loc = np.ascontiguousarray(add_halo_columns(inv_rho[:, x_start:x_end]))
damp_loc = np.ascontiguousarray(add_halo_columns(damp[:, x_start:x_end]))

vx = np.zeros((nz, nx_loc + 2), dtype=np.float32)
vz = np.zeros((nz, nx_loc + 2), dtype=np.float32)
sxx = np.zeros((nz, nx_loc + 2), dtype=np.float32)
szz = np.zeros((nz, nx_loc + 2), dtype=np.float32)
sxz = np.zeros((nz, nx_loc + 2), dtype=np.float32)

halo_bufs = make_halo_buffers(nz)

iz0 = 1
iz1 = nz - 1
jx0 = 1
jx1 = nx_loc + 1

physical_x_start = pad_left
physical_x_end = pad_left + nx0

write_g0 = max(x_start, physical_x_start)
write_g1 = min(x_end, physical_x_end)

has_physical_output = write_g0 < write_g1

if has_physical_output:
    local_j0 = write_g0 - x_start + 1
    local_j1 = write_g1 - x_start + 1
    out_x0 = write_g0 - pad_left
    out_x1 = write_g1 - pad_left
    local_nx_phys = out_x1 - out_x0
else:
    local_j0 = 0
    local_j1 = 0
    out_x0 = -1
    out_x1 = -1
    local_nx_phys = 0

n_frames = len(range(0, n_iterations, frame_stride))

comm.Barrier()

h5 = h5py.File(rank_h5_path, "w")

if has_physical_output:
    chunk_x = max(1, int(4 * 1024 ** 2 / (nz0 * 4)))
    chunk_x = min(chunk_x, max(1, local_nx_phys))

    dset_vz = h5.create_dataset(
        "vz",
        shape=(n_frames, nz0, local_nx_phys),
        dtype=np.float32,
        chunks=(1, nz0, chunk_x),
        compression="gzip",
        compression_opts=4,
        shuffle=True
    )

    h5.create_dataset("vp", data=vp0[:, out_x0:out_x1].astype(np.float32))
else:
    dset_vz = None

times = np.arange(n_frames, dtype=np.float32) * frame_stride * dt
h5.create_dataset("time", data=times)

h5.attrs["rank"] = rank
h5.attrs["size"] = size
h5.attrs["x0"] = out_x0
h5.attrs["x1"] = out_x1
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
    exchange_halo_x2(vx, vz)

    src = np.float32(src_amp * ricker(np.float32(it) * dt))

    if x_start <= src_x < x_end:
        lx = src_x - x_start + 1
        sxx[src_z, lx] += src
        szz[src_z, lx] += src
        if direct_vz_source:
            vz[src_z, lx] += np.float32(dt * inv_rho_loc[src_z, lx] * src)

    update_stress_c(vx, vz, sxx, szz, sxz, lam_loc, lam2mu_loc, mu_loc, damp_loc, dt, dx, dz, iz0, iz1, jx0, jx1)

    exchange_halo_x3(sxx, szz, sxz)

    update_velocity_c(vx, vz, sxx, szz, sxz, inv_rho_loc, damp_loc, dt, dx, dz, iz0, iz1, jx0, jx1)

frame_id = 0

for it in range(n_iterations):
    step(it)

    if it % frame_stride == 0:
        if has_physical_output:
            local_view = np.ascontiguousarray(vz[pad_top:pad_top + nz0, local_j0:local_j1])
            dset_vz[frame_id] = local_view
        frame_id += 1

h5.close()

meta = {
    "rank": rank,
    "has": bool(has_physical_output),
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

            src_shape = (n_frames, nz0, m["x1"] - m["x0"])
            source = h5py.VirtualSource(m["path"], "vz", shape=src_shape)
            layout[:, :, m["x0"]:m["x1"]] = source

        vf.create_virtual_dataset("vz", layout, fillvalue=0.0)
        vf.create_dataset("vp", data=vp0.astype(np.float32))
        vf.create_dataset("time", data=times)

        vf.attrs["size"] = size
        vf.attrs["nz0"] = nz0
        vf.attrs["nx0"] = nx0
        vf.attrs["dx"] = float(dx)
        vf.attrs["dz"] = float(dz)
        vf.attrs["dt"] = float(dt)
        vf.attrs["frame_stride"] = frame_stride
        vf.attrs["n_frames"] = n_frames

    print("done", flush=True)
