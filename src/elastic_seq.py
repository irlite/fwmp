import os
import ctypes
import segyio
import h5py
import numpy as np

n_iterations = int(os.environ.get("FWMP_NITER", "50000"))
frame_stride = int(os.environ.get("FWMP_FRAME_STRIDE", "100"))
direct_vz_source = int(os.environ.get("FWMP_DIRECT_VZ_SOURCE", "0"))
ds = 1

slurm_job_id = os.environ.get("SLURM_JOB_ID", "noslurm")
base_output_dir = os.path.join("..", "output", f"job_{slurm_job_id}")
os.makedirs(base_output_dir, exist_ok=True)

h5_path = os.path.join(base_output_dir, "elastic_wavefield.h5")

vp_path = "../data/MODEL_P-WAVE_VELOCITY_1.25m.segy"
vs_path = "../data/MODEL_S-WAVE_VELOCITY_1.25m.segy"
rho_path = "../data/MODEL_DENSITY_1.25m.segy"

lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "libelastic_kernels_seq.so"))

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


def load_segy(path):
    with segyio.open(path, "r", ignore_geometry=True) as f:
        return np.stack([np.array(tr) for tr in f.trace]).T


def add_halo_2d(a):
    out = np.empty((a.shape[0] + 2, a.shape[1] + 2), dtype=a.dtype)

    out[1:-1, 1:-1] = a

    out[0, 1:-1] = out[1, 1:-1]
    out[-1, 1:-1] = out[-2, 1:-1]

    out[:, 0] = out[:, 1]
    out[:, -1] = out[:, -2]

    return out


def update_stress_c(
    vx, vz,
    sxx, szz, sxz,
    lam, lam2mu, mu, damp,
    dt, dx, dz,
    iz0, iz1,
    jx0, jx1
):
    lib.update_stress(
        vx, vz,
        sxx, szz, sxz,
        lam, lam2mu, mu, damp,
        np.float32(dt), np.float32(dx), np.float32(dz),
        vx.shape[0], vx.shape[1],
        iz0, iz1,
        jx0, jx1
    )


def update_velocity_c(
    vx, vz,
    sxx, szz, sxz,
    inv_rho, damp,
    dt, dx, dz,
    iz0, iz1,
    jx0, jx1
):
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

mu_h = np.ascontiguousarray(add_halo_2d(mu))
lam_h = np.ascontiguousarray(add_halo_2d(lam))
lam2mu_h = np.ascontiguousarray(add_halo_2d(lam2mu))
inv_rho_h = np.ascontiguousarray(add_halo_2d(inv_rho))
damp_h = np.ascontiguousarray(add_halo_2d(damp))

vx = np.zeros((nz + 2, nx + 2), dtype=np.float32)
vz = np.zeros((nz + 2, nx + 2), dtype=np.float32)
sxx = np.zeros((nz + 2, nx + 2), dtype=np.float32)
szz = np.zeros((nz + 2, nx + 2), dtype=np.float32)
sxz = np.zeros((nz + 2, nx + 2), dtype=np.float32)

# Same effective global-boundary behavior as the MPI version on one rank.
iz0 = 2
iz1 = nz

jx0 = 1
jx1 = nx + 1

physical_z_start = pad_top
physical_z_end = pad_top + nz0
physical_x_start = pad_left
physical_x_end = pad_left + nx0

local_i0 = physical_z_start + 1
local_i1 = physical_z_end + 1
local_j0 = physical_x_start + 1
local_j1 = physical_x_end + 1

n_frames = len(range(0, n_iterations, frame_stride))
times = np.arange(n_frames, dtype=np.float32) * frame_stride * dt

h5 = h5py.File(h5_path, "w")

target_elems = max(1, int(4 * 1024 ** 2 / 4))
chunk_z = min(nz0, max(1, int(np.sqrt(target_elems))))
chunk_x = min(nx0, max(1, target_elems // chunk_z))

dset_vz = h5.create_dataset(
    "vz",
    shape=(n_frames, nz0, nx0),
    dtype=np.float32,
    chunks=(1, chunk_z, chunk_x),
    compression="gzip",
    compression_opts=4,
    shuffle=True
)

h5.create_dataset("vp", data=vp0.astype(np.float32))
h5.create_dataset("time", data=times)

h5.attrs["size"] = 1
h5.attrs["dims_z"] = 1
h5.attrs["dims_x"] = 1
h5.attrs["nz0"] = nz0
h5.attrs["nx0"] = nx0
h5.attrs["dx"] = float(dx)
h5.attrs["dz"] = float(dz)
h5.attrs["dt"] = float(dt)
h5.attrs["frame_stride"] = frame_stride
h5.attrs["n_frames"] = n_frames

# Warm-up call, matching the structure of the original code.
update_stress_c(
    vx, vz,
    sxx, szz, sxz,
    lam_h, lam2mu_h, mu_h, damp_h,
    dt, dx, dz,
    iz0, iz1,
    jx0, jx1
)

update_velocity_c(
    vx, vz,
    sxx, szz, sxz,
    inv_rho_h, damp_h,
    dt, dx, dz,
    iz0, iz1,
    jx0, jx1
)

vx.fill(0.0)
vz.fill(0.0)
sxx.fill(0.0)
szz.fill(0.0)
sxz.fill(0.0)


def step(it):
    src = np.float32(src_amp * ricker(np.float32(it) * dt))

    li = src_z + 1
    lj = src_x + 1

    sxx[li, lj] += src
    szz[li, lj] += src

    if direct_vz_source:
        vz[li, lj] += np.float32(dt * inv_rho_h[li, lj] * src)

    update_stress_c(
        vx, vz,
        sxx, szz, sxz,
        lam_h, lam2mu_h, mu_h, damp_h,
        dt, dx, dz,
        iz0, iz1,
        jx0, jx1
    )

    update_velocity_c(
        vx, vz,
        sxx, szz, sxz,
        inv_rho_h, damp_h,
        dt, dx, dz,
        iz0, iz1,
        jx0, jx1
    )


frame_id = 0

for it in range(n_iterations):
    step(it)

    if it % frame_stride == 0:
        dset_vz[frame_id] = np.ascontiguousarray(
            vz[local_i0:local_i1, local_j0:local_j1]
        )
        frame_id += 1

h5.close()

print("done", flush=True)
print(f"wrote {h5_path}", flush=True)
