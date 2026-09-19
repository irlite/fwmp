#!/usr/bin/env python3

import os
import hdf5plugin
import ctypes
import segyio
import h5py
import numpy as np

n_iterations = int(os.environ.get("FWMP_NITER", "50000"))
frame_stride = int(os.environ.get("FWMP_FRAME_STRIDE", "100"))
direct_vz_source = int(os.environ.get("FWMP_DIRECT_VZ_SOURCE", "0"))
#ds = 1
ds = int(os.environ.get("FWMP_DS", "1"))

base_output_dir = os.environ["FWMP_BASE_OUTPUT_DIR"]

vp_path = "../data/MODEL_P-WAVE_VELOCITY_1.25m.segy"
vs_path = "../data/MODEL_S-WAVE_VELOCITY_1.25m.segy"
rho_path = "../data/MODEL_DENSITY_1.25m.segy"

lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "libelastic_kernels_seq.so"))
_float2 = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")

lib.update_stress_seq.argtypes = [
    _float2, _float2,
    _float2, _float2, _float2,
    _float2, _float2, _float2, _float2,
    ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int
]
lib.update_stress_seq.restype = None

lib.update_velocity_seq.argtypes = [
    _float2, _float2,
    _float2, _float2, _float2,
    _float2, _float2,
    ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int
]
lib.update_velocity_seq.restype = None

os.makedirs(base_output_dir, exist_ok=True)

h5_path = os.path.join(base_output_dir, "elastic_wavefield.h5")

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

def update_stress_c(vx, vz, sxx, szz, sxz, lam, lam2mu, mu, damp, dt, dx, dz, iz0, iz1, jx0, jx1):
    lib.update_stress_seq(
        vx, vz,
        sxx, szz, sxz,
        lam, lam2mu, mu, damp,
        np.float32(dt), np.float32(dx), np.float32(dz),
        vx.shape[0], vx.shape[1],
        iz0, iz1,
        jx0, jx1
    )

def update_velocity_c(vx, vz, sxx, szz, sxz, inv_rho, damp, dt, dx, dz, iz0, iz1, jx0, jx1):
    lib.update_velocity_seq(
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

mu_loc = np.ascontiguousarray(add_halo_2d(mu))
lam_loc = np.ascontiguousarray(add_halo_2d(lam))
lam2mu_loc = np.ascontiguousarray(add_halo_2d(lam2mu))
inv_rho_loc = np.ascontiguousarray(add_halo_2d(inv_rho))
damp_loc = np.ascontiguousarray(add_halo_2d(damp))

vx = np.zeros((nz + 2, nx + 2), dtype=np.float32)
vz = np.zeros((nz + 2, nx + 2), dtype=np.float32)
sxx = np.zeros((nz + 2, nx + 2), dtype=np.float32)
szz = np.zeros((nz + 2, nx + 2), dtype=np.float32)
sxz = np.zeros((nz + 2, nx + 2), dtype=np.float32)

# whole domain, no halo exchange needed (single process)
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

out_z0 = 0
out_z1 = nz0
out_x0 = 0
out_x1 = nx0

local_nz_phys = nz0
local_nx_phys = nx0

n_frames = len(range(0, n_iterations, frame_stride))

h5 = h5py.File(h5_path, "w")

target_elems = max(1, int(4 * 1024 ** 2 / 4))
chunk_z = min(local_nz_phys, max(1, int(np.sqrt(target_elems))))
chunk_x = min(local_nx_phys, max(1, target_elems // chunk_z))

dset_vz = h5.create_dataset(
    "vz",
    shape=(n_frames, local_nz_phys, local_nx_phys),
    dtype=np.float32,
    chunks=(1, chunk_z, chunk_x),
    **hdf5plugin.LZ4(),
    shuffle=True
)
h5.create_dataset("vp", data=vp0[out_z0:out_z1, out_x0:out_x1].astype(np.float32))

times = np.arange(n_frames, dtype=np.float32) * frame_stride * dt
h5.create_dataset("time", data=times)

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
    src = np.float32(src_amp * ricker(np.float32(it) * dt))

    li = src_z + 1
    lj = src_x + 1
    sxx[li, lj] += src
    szz[li, lj] += src
    if direct_vz_source:
        vz[li, lj] += np.float32(dt * inv_rho_loc[li, lj] * src)

    update_stress_c(vx, vz, sxx, szz, sxz, lam_loc, lam2mu_loc, mu_loc, damp_loc, dt, dx, dz, iz0, iz1, jx0, jx1)

    update_velocity_c(vx, vz, sxx, szz, sxz, inv_rho_loc, damp_loc, dt, dx, dz, iz0, iz1, jx0, jx1)

frame_id = 0

for it in range(n_iterations):
    step(it)

    if it % frame_stride == 0:
        local_view = np.ascontiguousarray(vz[local_i0:local_i1, local_j0:local_j1])
        dset_vz[frame_id] = local_view
        frame_id += 1

h5.close()

process_total_ns = time.perf_counter_ns() - process_start_ns

input_ns = sum(
    timings_ns[name]
    for name in [
        "Read SEG-Y vp",
        "Downsample and convert vp",
        "Read SEG-Y vs",
        "Downsample and convert vs",
        "Read SEG-Y rho",
        "Downsample and convert rho",
    ]
)

setup_ns = sum(
    timings_ns[name]
    for name in [
        "Python imports",
        "Load C kernel library",
        "Create output directory",
        "Grid and parameter setup",
        "Pad material models",
        "Compute elastic parameters",
        "Build damping model",
        "Prepare material halo arrays",
        "Allocate wavefields",
        "HDF5 file and dataset creation",
        "HDF5 static data and metadata",
        "Kernel warm-up stress",
        "Kernel warm-up velocity",
        "Clear wavefields after warm-up",
    ]
)

kernel_ns = (
    timings_ns["Stress kernel"]
    + timings_ns["Velocity kernel"]
)

output_during_simulation_ns = (
    timings_ns["Output frame copy"]
    + timings_ns["HDF5 frame assignment"]
    + timings_ns["HDF5 per-frame flush"]
)

finalization_ns = (
    timings_ns["HDF5 final flush"]
    + timings_ns["HDF5 close"]
)

output_total_ns = (
    timings_ns["HDF5 file and dataset creation"]
    + timings_ns["HDF5 static data and metadata"]
    + output_during_simulation_ns
    + finalization_ns
)

source_ns = timings_ns["Source calculation and injection"]

simulation_other_ns = max(
    0,
    simulation_elapsed_ns
    - kernel_ns
    - output_during_simulation_ns
    - source_ns
)

input_seconds = input_ns / 1e9
setup_seconds = setup_ns / 1e9
kernel_seconds = kernel_ns / 1e9
source_seconds = source_ns / 1e9
output_simulation_seconds = output_during_simulation_ns / 1e9
finalization_seconds = finalization_ns / 1e9
output_seconds = output_total_ns / 1e9
simulation_other_seconds = simulation_other_ns / 1e9
simulation_seconds = simulation_elapsed_ns / 1e9
process_seconds = process_total_ns / 1e9

stress_seconds = timings_ns["Stress kernel"] / 1e9
velocity_seconds = timings_ns["Velocity kernel"] / 1e9

iterations_per_second = (
    n_iterations / simulation_seconds
    if simulation_seconds > 0.0
    else 0.0
)

kernel_ms_per_iteration = (
    kernel_ns / n_iterations / 1e6
    if n_iterations > 0
    else 0.0
)

output_ms_per_frame = (
    output_total_ns / n_frames / 1e6
    if n_frames > 0
    else 0.0
)

logical_vz_bytes = (
    n_frames
    * local_nz_phys
    * local_nx_phys
    * np.dtype(np.float32).itemsize
)

file_bytes = os.path.getsize(h5_path)

compression_ratio = (
    logical_vz_bytes / file_bytes
    if file_bytes > 0
    else 0.0
)

print()
print("Benchmark summary")
print("=" * 58)
print(f"{'Input reading and conversion':40s}{input_seconds:14.6f}")
print(f"{'Setup and initialization':40s}{setup_seconds:14.6f}")
print(f"{'Stress kernel':40s}{stress_seconds:14.6f}")
print(f"{'Velocity kernel':40s}{velocity_seconds:14.6f}")
print(f"{'Kernel total':40s}{kernel_seconds:14.6f}")
print(f"{'Source calculation/injection':40s}{source_seconds:14.6f}")
print(f"{'Output during simulation':40s}{output_simulation_seconds:14.6f}")
print(f"{'Simulation other/overhead':40s}{simulation_other_seconds:14.6f}")
print(f"{'Simulation loop total':40s}{simulation_seconds:14.6f}")
print(f"{'HDF5 final flush and close':40s}{finalization_seconds:14.6f}")
print(f"{'Output/compression total':40s}{output_seconds:14.6f}")
print(f"{'Whole process total':40s}{process_seconds:14.6f}")
print("-" * 58)
print(f"{'Iterations per second':40s}{iterations_per_second:14.3f}")
print(f"{'Kernel ms per iteration':40s}{kernel_ms_per_iteration:14.6f}")
print(f"{'Output ms per frame':40s}{output_ms_per_frame:14.6f}")
print(f"{'Logical output size (GiB)':40s}{logical_vz_bytes / 1024**3:14.6f}")
print(f"{'HDF5 file size (GiB)':40s}{file_bytes / 1024**3:14.6f}")
print(f"{'Logical/file size ratio':40s}{compression_ratio:14.6f}")
print(f"{'Compression mode':40s}{compression_mode:>14s}")
print("=" * 58)
print("done", flush=True)
