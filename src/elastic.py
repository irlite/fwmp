import os
import segyio
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
n_iterations = 6000
frame_stride = 10
ds = 8
frames_dir = "../frames"
vp_path = "../data/MODEL_P-WAVE_VELOCITY_1.25m.segy"
vs_path = "../data/MODEL_S-WAVE_VELOCITY_1.25m.segy"
rho_path = "../data/MODEL_DENSITY_1.25m.segy"

if rank == 0:
    os.makedirs(frames_dir, exist_ok=True)
    for fname in os.listdir(frames_dir):
        if fname.endswith(".png"):
            os.remove(os.path.join(frames_dir, fname))
comm.Barrier()

def load_segy(path):
    with segyio.open(path, "r", ignore_geometry=True) as f:
        return np.stack([np.array(tr) for tr in f.trace]).T

def split_1d(total_columns, size, rank):
    counts = [total_columns // size + (1 if r < total_columns % size else 0)
        for r in range(size)]
    starts = np.cumsum([0] + counts[:-1])
    start = int(starts[rank])
    end = int(start + counts[rank])
    return start, end, counts

def add_halo_columns(target_array):
    out = np.empty((target_array.shape[0], target_array.shape[1] + 2),
                   dtype=target_array.dtype)
    out[:, 1:-1] = target_array
    out[:, 0] = out[:, 1]
    out[:, -1] = out[:, -2]
    return out

def exchange_halo_x(a):
    left = rank - 1 if rank > 0 else MPI.PROC_NULL
    right = rank + 1 if rank < size - 1 else MPI.PROC_NULL
    send_left = np.ascontiguousarray(a[:, 1])
    recv_right = np.empty(a.shape[0], dtype=a.dtype)
    comm.Sendrecv(sendbuf=send_left, dest=left, sendtag=10, recvbuf=recv_right, source=right, recvtag=10)
    if right != MPI.PROC_NULL:
        a[:, -1] = recv_right
    send_right = np.ascontiguousarray(a[:, -2])
    recv_left = np.empty(a.shape[0], dtype=a.dtype)
    comm.Sendrecv(sendbuf=send_right, dest=right, sendtag=20, recvbuf=recv_left, source=left, recvtag=20)
    if left != MPI.PROC_NULL:
        a[:, 0] = recv_left

def gather_global_field(local_field):
    owned = local_field[:, 1:-1].copy()
    pieces = comm.gather(owned, root=0)
    if rank == 0:
        return np.concatenate(pieces, axis=1)
    return None

vp0 = load_segy(vp_path)[::ds, ::ds].astype(np.float32)
vs0 = load_segy(vs_path)[::ds, ::ds].astype(np.float32)
rho0 = load_segy(rho_path)[::ds, ::ds].astype(np.float32)
nz0, nx0 = vp0.shape
dx = dz = 1.25 * ds

if rank == 0:
    print(f"Vp  min={vp0.min():.1f}  max={vp0.max():.1f}")
    print(f"Vs  min={vs0.min():.1f}  max={vs0.max():.1f}")
    print(f"Rho min={rho0.min():.1f}  max={rho0.max():.1f}")

zmax_plot_m = 3500.0
nz_plot = min(nz0, int(round(zmax_plot_m / dz)) + 1)
zmax_plot_m = (nz_plot - 1) * dz
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
dt = 0.4 * dx / vp_max

if rank == 0:
    print(f"dt={dt*1000:.4f} ms  dx={dx:.2f} m  CFL={vp_max*dt/dx:.3f}")
    print(f"nz0={nz0} nx0={nx0}  nz={nz} nx={nx}")
    print(f"n_iterations={n_iterations}  frame_stride={frame_stride}")
    print(f"MPI ranks={size}")

f0 = 8.0
t0 = 1.2 / f0
src_amp = 1e9

def ricker(t):
    a = (np.pi * f0 * (t - t0)) ** 2
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

damp = np.clip(1.0 - sigma * dt, 0.0, 1.0).astype(np.float32)
x_start, x_end, x_counts = split_1d(nx, size, rank)
nx_loc = x_end - x_start

if nx_loc <= 0:
    raise RuntimeError(f"Rank {rank} has nx_loc={nx_loc}. Use fewer MPI ranks than global x-columns.")

if rank == 0:
    print("x_counts per rank:", x_counts)

print(f"rank {rank}: x_start={x_start}, x_end={x_end}, nx_loc={nx_loc}")
comm.Barrier()

mu_loc = add_halo_columns(mu[:, x_start:x_end])
lam_loc = add_halo_columns(lam[:, x_start:x_end])
lam2mu_loc = add_halo_columns(lam2mu[:, x_start:x_end])
inv_rho_loc = add_halo_columns(inv_rho[:, x_start:x_end])
damp_loc = add_halo_columns(damp[:, x_start:x_end])
vx = np.zeros((nz, nx_loc + 2), dtype=np.float32)
vz = np.zeros((nz, nx_loc + 2), dtype=np.float32)
sxx = np.zeros((nz, nx_loc + 2), dtype=np.float32)
szz = np.zeros((nz, nx_loc + 2), dtype=np.float32)
sxz = np.zeros((nz, nx_loc + 2), dtype=np.float32)

def step(it):
    global vx, vz, sxx, szz, sxz
    exchange_halo_x(vx)
    exchange_halo_x(vz)
    src = src_amp * ricker(it * dt)
    if x_start <= src_x < x_end:
        local_src_x = src_x - x_start + 1
        sxx[src_z, local_src_x] += src
        szz[src_z, local_src_x] += src
    jx0 = 1
    jx1 = nx_loc + 1
    if x_start == 0:
        jx0 = 2
    if x_end == nx:
        jx1 = nx_loc
    jx = slice(jx0, jx1)
    jxp = slice(jx0 + 1, jx1 + 1)
    jxm = slice(jx0 - 1, jx1 - 1)
    iz = slice(1, nz - 1)
    izp = slice(2, nz)
    izm = slice(0, nz - 2)
    dvx_dx = (vx[iz, jxp] - vx[iz, jx]) / dx
    dvx_dz = (vx[izp, jx] - vx[iz, jx]) / dz
    dvz_dx = (vz[iz, jxp] - vz[iz, jx]) / dx
    dvz_dz = (vz[izp, jx] - vz[iz, jx]) / dz
    sxx[iz, jx] += dt * (lam2mu_loc[iz, jx] * dvx_dx + lam_loc[iz, jx] * dvz_dz)
    szz[iz, jx] += dt * (lam_loc[iz, jx] * dvx_dx + lam2mu_loc[iz, jx] * dvz_dz)
    sxz[iz, jx] += dt * mu_loc[iz, jx] * (dvx_dz + dvz_dx)
    sxx[:, 1:-1] *= damp_loc[:, 1:-1]
    szz[:, 1:-1] *= damp_loc[:, 1:-1]
    sxz[:, 1:-1] *= damp_loc[:, 1:-1]
    exchange_halo_x(sxx)
    exchange_halo_x(szz)
    exchange_halo_x(sxz)
    dsxx_dx = (sxx[iz, jx] - sxx[iz, jxm]) / dx
    dsxz_dz = (sxz[iz, jx] - sxz[izm, jx]) / dz
    dsxz_dx = (sxz[iz, jx] - sxz[iz, jxm]) / dx
    dszz_dz = (szz[iz, jx] - szz[izm, jx]) / dz
    vx[iz, jx] += dt * inv_rho_loc[iz, jx] * (dsxx_dx + dsxz_dz)
    vz[iz, jx] += dt * inv_rho_loc[iz, jx] * (dsxz_dx + dszz_dz)
    vx[:, 1:-1] *= damp_loc[:, 1:-1]
    vz[:, 1:-1] *= damp_loc[:, 1:-1]

def physical_view_global(field):
    f = field[pad_top:pad_top+nz0, pad_left:pad_left+nx0]
    return f[:nz_plot, :]

if rank == 0:
    vel_view = vp0[:nz_plot, :]
    x_extent = (nx0 - 1) * dx
    z_extent = zmax_plot_m
    main_width = 12
    main_height = main_width * (z_extent / x_extent)
    fig = plt.figure(figsize=(main_width + 1.4, 2.0 * main_height), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.06], height_ratios=[1.0, 1.0])
    axw = fig.add_subplot(gs[0, 0])
    caxw = fig.add_subplot(gs[0, 1])
    axv = fig.add_subplot(gs[1, 0])
    caxv = fig.add_subplot(gs[1, 1])
    initial_view = np.zeros((nz_plot, nx0), dtype=np.float32)
    im_w = axw.imshow(initial_view, cmap="seismic", extent=[0, x_extent, z_extent, 0], aspect="equal", vmin=-1e-3, vmax=1e-3)
    axw.set_title("Elastic Wavefield — Vz")
    axw.set_xlabel("Distance (m)")
    axw.set_ylabel("Depth (m)")
    axw.plot(src_x0 * dx, src_z0 * dz, "k*", markersize=8)
    fig.colorbar(im_w, cax=caxw, label="Vz (m/s)")
    im_v = axv.imshow(vel_view, cmap="jet", extent=[0, x_extent, z_extent, 0], aspect="equal")
    axv.set_title("Vp Model")
    axv.set_xlabel("Distance (m)")
    axv.set_ylabel("Depth (m)")
    axv.plot(src_x0 * dx, src_z0 * dz, "w*", markersize=8)
    fig.colorbar(im_v, cax=caxv, label="Velocity (m/s)")
    frame_id = 0
else:
    fig = None
    im_w = None
    axw = None
    frame_id = None

for it in range(n_iterations):
    step(it)
    if it % frame_stride == 0:
        vz_global = gather_global_field(vz)
        if rank == 0:
            view = physical_view_global(vz_global)
            m = float(np.max(np.abs(view)))
            print(f"it={it}  max(vz)={m:.6e}")
            if m > 1e-20:
                im_w.set_clim(-0.6 * m, 0.6 * m)
            im_w.set_data(view)
            axw.set_title(f"Elastic Wavefield — Vz | Time = {it * dt * 1000:.1f} ms")
            outpath = os.path.join(frames_dir, f"frame_{frame_id:05d}.png")
            fig.savefig(outpath, dpi=150)
            frame_id += 1

if rank == 0:
    plt.close(fig)
    print(f"Saved {frame_id} frames to {frames_dir}")
