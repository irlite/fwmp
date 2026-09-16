import os
import glob
import ctypes
import segyio
import h5py
import numpy as np
import time

t_total_start = time.perf_counter()

n_iterations = int(os.environ.get("FWMP_NITER", "6000"))
frame_stride = int(os.environ.get("FWMP_FRAME_STRIDE", "10"))
debug = int(os.environ.get("FWMP_DEBUG", "0"))
direct_vz_source = int(os.environ.get("FWMP_DIRECT_VZ_SOURCE", "0"))
debug_source_every = int(os.environ.get("FWMP_DEBUG_SOURCE_EVERY", "100"))
ds = 4
output_dir = "../output"
vds_path = os.path.join(output_dir, "elastic_wavefield.h5")
rank_h5_path = os.path.join(output_dir, "elastic_wavefield_single.h5")
vp_path = "../data/MODEL_P-WAVE_VELOCITY_1.25m.segy"
vs_path = "../data/MODEL_S-WAVE_VELOCITY_1.25m.segy"
rho_path = "../data/MODEL_DENSITY_1.25m.segy"

lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "libelastic_kernels_single_threaded.so"))
_float2 = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")

lib.update_stress.argtypes = [
    _float2, _float2,
    _float2, _float2, _float2,
    _float2, _float2, _float2, _float2,
    ctypes.c_float, ctypes.c_float, ctypes.c_float,
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
    ctypes.c_int, ctypes.c_int
]
lib.update_velocity.restype = None

os.makedirs(output_dir, exist_ok=True)
for path in glob.glob(os.path.join(output_dir, "elastic_wavefield_rank*.h5")):
    os.remove(path)
for path in glob.glob(os.path.join(output_dir, "elastic_wavefield_single.h5")):
    os.remove(path)
if os.path.exists(vds_path):
    os.remove(vds_path)

def load_segy(path):
    with segyio.open(path, "r", ignore_geometry=True) as f:
        return np.stack([np.array(tr) for tr in f.trace]).T

def add_halo_columns(a):
    out = np.empty((a.shape[0], a.shape[1] + 2), dtype=a.dtype)
    out[:, 1:-1] = a
    out[:, 0] = out[:, 1]
    out[:, -1] = out[:, -2]
    return out

def make_halo_buffers(n0, dtype=np.float32):
    return {
        "x2_send_left":  np.empty((n0, 2), dtype=dtype),
        "x2_recv_left":  np.empty((n0, 2), dtype=dtype),
        "x2_send_right": np.empty((n0, 2), dtype=dtype),
        "x2_recv_right": np.empty((n0, 2), dtype=dtype),
        "x3_send_left":  np.empty((n0, 3), dtype=dtype),
        "x3_recv_left":  np.empty((n0, 3), dtype=dtype),
        "x3_send_right": np.empty((n0, 3), dtype=dtype),
        "x3_recv_right": np.empty((n0, 3), dtype=dtype),
    }

def report_timing(name, value):
    print(f"TIMING {name:18s} time={float(value):10.3f}s", flush=True)

def update_stress_c(vx, vz, sxx, szz, sxz, lam, lam2mu, mu, damp, dt, dx, dz, jx0, jx1):
    lib.update_stress(vx, vz, sxx, szz, sxz, lam, lam2mu, mu, damp, np.float32(dt), np.float32(dx), np.float32(dz), vx.shape[0], vx.shape[1], jx0, jx1)

def update_velocity_c(vx, vz, sxx, szz, sxz, inv_rho, damp, dt, dx, dz, jx0, jx1):
    lib.update_velocity(vx, vz, sxx, szz, sxz, inv_rho, damp, np.float32(dt), np.float32(dx), np.float32(dz), vx.shape[0], vx.shape[1], jx0, jx1)

t_load_start = time.perf_counter()
vp0 = load_segy(vp_path)[::ds, ::ds].astype(np.float32)
vs0 = load_segy(vs_path)[::ds, ::ds].astype(np.float32)
rho0 = load_segy(rho_path)[::ds, ::ds].astype(np.float32)
t_load = time.perf_counter() - t_load_start

t_setup_start = time.perf_counter()
nz0, nx0 = vp0.shape
dx = np.float32(1.25 * ds)
dz = np.float32(1.25 * ds)

print(f"Vp  min={vp0.min():.1f}  max={vp0.max():.1f}", flush=True)
print(f"Vs  min={vs0.min():.1f}  max={vs0.max():.1f}", flush=True)
print(f"Rho min={rho0.min():.1f}  max={rho0.max():.1f}", flush=True)

zmax_plot_m = 3500.0
nz_plot = min(nz0, int(round(zmax_plot_m / float(dz))) + 1)
zmax_plot_m = (nz_plot - 1) * float(dz)
nb = 240
pad_top = nb
pad_bottom = nb
pad_left = nb
pad_right = nb
nz = nz0 + pad_top + pad_bottom
nx = nx0 + pad_left + pad_right

def pad_field(f0):
    f = np.empty((nz, nx), dtype=np.float32)
    f[pad_top:pad_top+nz0, pad_left:pad_left+nx0] = f0
    f[:, :pad_left] = f[:, pad_left:pad_left+1]
    f[:, pad_left+nx0:] = f[:, pad_left+nx0-1:pad_left+nx0]
    f[:pad_top, :] = f[pad_top:pad_top+1, :]
    f[pad_top+nz0:, :] = f[pad_top+nz0-1:pad_top+nz0, :]
    return f

vp = pad_field(vp0)
vs = pad_field(vs0)
rho = pad_field(rho0)
mu = (rho * vs**2).astype(np.float32)
lam = (rho * vp**2 - 2.0 * mu).astype(np.float32)
lam2mu = (lam + 2.0 * mu).astype(np.float32)
inv_rho = (1.0 / rho).astype(np.float32)
vp_max = float(vp.max())
dt = np.float32(0.4 * float(dx) / vp_max)

print(f"dt={float(dt)*1000:.4f} ms  dx={float(dx):.2f} m  CFL={vp_max*float(dt)/float(dx):.3f}", flush=True)
print(f"nz0={nz0} nx0={nx0}  nz={nz} nx={nx}", flush=True)
print(f"n_iterations={n_iterations}  frame_stride={frame_stride}", flush=True)
print(f"Ranks=1", flush=True)
print(f"debug={debug} direct_vz_source={direct_vz_source}", flush=True)

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
sigma_lr_max = 60.0
sigma_top_max = 60.0
sigma_bottom_max = 120.0

def ramp(n, power=2.0):
    return np.linspace(0.0, 1.0, n, dtype=np.float32) ** power

r = ramp(pad_left)
for i in range(pad_left):
    sigma[:, i] = np.maximum(sigma[:, i], sigma_lr_max * r[pad_left - 1 - i])
r = ramp(pad_right)
for i in range(pad_right):
    sigma[:, -1-i] = np.maximum(sigma[:, -1-i], sigma_lr_max * r[pad_right - 1 - i])
r = ramp(pad_top)
for i in range(pad_top):
    sigma[i, :] = np.maximum(sigma[i, :], sigma_top_max * r[pad_top - 1 - i])
r = ramp(pad_bottom)
for i in range(pad_bottom):
    sigma[-1-i, :] = np.maximum(sigma[-1-i, :], sigma_bottom_max * r[pad_bottom - 1 - i])

damp = np.clip(1.0 - sigma * float(dt), 0.0, 1.0).astype(np.float32)

# Single rank owns the entire domain
x_start = 0
x_end = nx
nx_loc = nx
owns_src = True
src_owner = 0

print("x_counts per rank:", [nx], flush=True)
print(f"src_x0={src_x0} src_z0={src_z0} src_x={src_x} src_z={src_z} src_t0={float(src_t0):.6e} src_owner={src_owner}", flush=True)
print(f"x_start={x_start}, x_end={x_end}, nx_loc={nx_loc}", flush=True)

mu_loc = np.ascontiguousarray(add_halo_columns(mu), dtype=np.float32)
lam_loc = np.ascontiguousarray(add_halo_columns(lam), dtype=np.float32)
lam2mu_loc = np.ascontiguousarray(add_halo_columns(lam2mu), dtype=np.float32)
inv_rho_loc = np.ascontiguousarray(add_halo_columns(inv_rho), dtype=np.float32)
damp_loc = np.ascontiguousarray(add_halo_columns(damp), dtype=np.float32)
vx = np.zeros((nz, nx_loc + 2), dtype=np.float32)
vz = np.zeros((nz, nx_loc + 2), dtype=np.float32)
sxx = np.zeros((nz, nx_loc + 2), dtype=np.float32)
szz = np.zeros((nz, nx_loc + 2), dtype=np.float32)
sxz = np.zeros((nz, nx_loc + 2), dtype=np.float32)

halo_bufs = make_halo_buffers(nz, dtype=np.float32)

jx0_update = 1
jx1_update = nx_loc + 1
if x_start == 0:
    jx0_update = 2
if x_end == nx:
    jx1_update = nx_loc

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
target_chunk_mb = 4

if has_physical_output:
    chunk_x = max(1, int(target_chunk_mb * 1024**2 / (nz_plot * np.dtype(np.float32).itemsize)))
    chunk_x = min(chunk_x, local_nx_phys)
else:
    chunk_x = 1

t_setup = time.perf_counter() - t_setup_start
t_h5setup_start = time.perf_counter()
h5 = h5py.File(rank_h5_path, "w")

if has_physical_output:
    dset_vz = h5.create_dataset("vz", shape=(n_frames, nz_plot, local_nx_phys), dtype=np.float32, chunks=(1, nz_plot, chunk_x), compression="gzip", compression_opts=4, shuffle=True)
    h5.create_dataset("vp", data=vp0[:nz_plot, out_x0:out_x1].astype(np.float32), chunks=(nz_plot, chunk_x), compression="gzip", compression_opts=4, shuffle=True)
else:
    dset_vz = None

times = np.arange(n_frames, dtype=np.float32) * frame_stride * dt
h5.create_dataset("time", data=times)
h5.attrs["rank"] = 0
h5.attrs["size"] = 1
h5.attrs["x0"] = out_x0
h5.attrs["x1"] = out_x1
h5.attrs["local_nx_phys"] = local_nx_phys
h5.attrs["dx"] = float(dx)
h5.attrs["dz"] = float(dz)
h5.attrs["dt"] = float(dt)
h5.attrs["frame_stride"] = frame_stride
h5.attrs["n_frames"] = n_frames
h5.attrs["nz_plot"] = nz_plot
h5.attrs["nx0"] = nx0
h5.attrs["zmax_plot_m"] = zmax_plot_m
h5.attrs["src_x0"] = src_x0
h5.attrs["src_z0"] = src_z0
t_h5setup = time.perf_counter() - t_h5setup_start

t_jit_start = time.perf_counter()
update_stress_c(vx, vz, sxx, szz, sxz, lam_loc, lam2mu_loc, mu_loc, damp_loc, dt, dx, dz, jx0_update, jx1_update)
update_velocity_c(vx, vz, sxx, szz, sxz, inv_rho_loc, damp_loc, dt, dx, dz, jx0_update, jx1_update)
vx.fill(0.0)
vz.fill(0.0)
sxx.fill(0.0)
szz.fill(0.0)
sxz.fill(0.0)
t_jit = time.perf_counter() - t_jit_start

print(f"Writing HDF5 file to {output_dir}", flush=True)
print(f"C/OpenMP OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'runtime default')}", flush=True)

def step(it):
    global vx, vz, sxx, szz, sxz
    src = np.float32(src_amp * ricker(np.float32(it) * dt))
    local_src_x = src_x - x_start + 1
    sxx[src_z, local_src_x] += src
    szz[src_z, local_src_x] += src
    if direct_vz_source:
        vz[src_z, local_src_x] += np.float32(dt * inv_rho_loc[src_z, local_src_x] * src)
    if debug and it % debug_source_every == 0:
        print(f"it={it} src={float(src):.6e} local_src_x={local_src_x} sxx_src={float(sxx[src_z, local_src_x]):.6e} vz_src={float(vz[src_z, local_src_x]):.6e}", flush=True)
    update_stress_c(vx, vz, sxx, szz, sxz, lam_loc, lam2mu_loc, mu_loc, damp_loc, dt, dx, dz, jx0_update, jx1_update)
    update_velocity_c(vx, vz, sxx, szz, sxz, inv_rho_loc, damp_loc, dt, dx, dz, jx0_update, jx1_update)

frame_id = 0
t_step_sum = 0.0
t_io_sum = 0.0
t_loop_start = time.perf_counter()

for it in range(n_iterations):
    tic = time.perf_counter()
    step(it)
    t_step_sum += time.perf_counter() - tic
    if it % frame_stride == 0:
        tic = time.perf_counter()
        if has_physical_output:
            local_view = np.ascontiguousarray(vz[pad_top:pad_top+nz_plot, local_j0:local_j1], dtype=np.float32)
            dset_vz[frame_id, :, :] = local_view
            local_saved_vz = np.array(np.max(np.abs(local_view)), dtype=np.float32)
        else:
            local_saved_vz = np.array(0.0, dtype=np.float32)
        if debug:
            local_full_vx = np.array(np.max(np.abs(vx[:, 1:-1])), dtype=np.float32)
            local_full_vz = np.array(np.max(np.abs(vz[:, 1:-1])), dtype=np.float32)
            local_full_sxx = np.array(np.max(np.abs(sxx[:, 1:-1])), dtype=np.float32)
            local_full_szz = np.array(np.max(np.abs(szz[:, 1:-1])), dtype=np.float32)
            local_full_sxz = np.array(np.max(np.abs(sxz[:, 1:-1])), dtype=np.float32)
            global_full_vx = local_full_vx
            global_full_vz = local_full_vz
            global_full_sxx = local_full_sxx
            global_full_szz = local_full_szz
            global_full_sxz = local_full_sxz
        global_saved_vz = local_saved_vz
        if debug:
            print(f"it={it} frame={frame_id}/{n_frames} saved_vz={float(global_saved_vz):.6e} full_vx={float(global_full_vx):.6e} full_vz={float(global_full_vz):.6e} full_sxx={float(global_full_sxx):.6e} full_szz={float(global_full_szz):.6e} full_sxz={float(global_full_sxz):.6e}", flush=True)
        else:
            print(f"it={it}  frame={frame_id}/{n_frames}  max(vz)={float(global_saved_vz):.6e}", flush=True)
        if frame_id % 10 == 0:
            h5.flush()
        frame_id += 1
        t_io_sum += time.perf_counter() - tic

t_loop = time.perf_counter() - t_loop_start
t_close_start = time.perf_counter()
h5.flush()
h5.close()
t_close = time.perf_counter() - t_close_start

print(f"Saved {frame_id} frames", flush=True)

report_timing("segy_load", t_load)
report_timing("setup", t_setup)
report_timing("hdf5_setup", t_h5setup)
report_timing("kernel_warmup", t_jit)
report_timing("step_sum", t_step_sum)
report_timing("io_sum", t_io_sum)
report_timing("loop_total", t_loop)
report_timing("hdf5_close", t_close)
report_timing("total", time.perf_counter() - t_total_start)

max_step = t_step_sum
max_io = t_io_sum

logical_output_bytes = n_frames * nz_plot * nx0 * np.dtype(np.float32).itemsize
rank_files = [rank_h5_path]
actual_output_bytes = sum(os.path.getsize(path) for path in rank_files)
grid_updates = n_iterations * nz * nx

print(f"BENCH ranks=1", flush=True)
print(f"BENCH c_openmp_threads={os.environ.get('OMP_NUM_THREADS', 'runtime default')}", flush=True)
print(f"BENCH grid_padded nz={nz} nx={nx} points={nz*nx}", flush=True)
print(f"BENCH frames={n_frames} nz_plot={nz_plot} nx0={nx0}", flush=True)
print(f"BENCH logical_vz_output={logical_output_bytes/1024**2:.2f} MiB", flush=True)
print(f"BENCH actual_rank_h5_size={actual_output_bytes/1024**2:.2f} MiB", flush=True)
if max_step > 0.0:
    print(f"BENCH padded_grid_updates_per_second={grid_updates/max_step:.6e}", flush=True)
if max_io > 0.0:
    print(f"BENCH logical_vz_io_rate={logical_output_bytes/max_io/1024**2:.2f} MiB/s", flush=True)
print("Run combine_hdf5.py to create the virtual combined HDF5 file", flush=True)