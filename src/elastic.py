import segyio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

vp_path  = "../data/MODEL_P-WAVE_VELOCITY_1.25m.segy"
vs_path  = "../data/MODEL_S-WAVE_VELOCITY_1.25m.segy"
rho_path = "../data/MODEL_DENSITY_1.25m.segy"

def load_segy(path):
    with segyio.open(path, "r", ignore_geometry=True) as f:
        return np.stack([np.array(tr) for tr in f.trace]).T

ds = 8
vp0  = load_segy(vp_path)[::ds, ::ds].astype(np.float32)
vs0  = load_segy(vs_path)[::ds, ::ds].astype(np.float32)
rho0 = load_segy(rho_path)[::ds, ::ds].astype(np.float32)

nz0, nx0 = vp0.shape
dx = dz = 1.25 * ds

print(f"Vp  min={vp0.min():.1f}  max={vp0.max():.1f}")
print(f"Vs  min={vs0.min():.1f}  max={vs0.max():.1f}")
print(f"Rho min={rho0.min():.1f}  max={rho0.max():.1f}")

zmax_plot_m = 3500.0
nz_plot = min(nz0, int(round(zmax_plot_m / dz)) + 1)
zmax_plot_m = (nz_plot - 1) * dz

nb = 240
pad_top    = nb
pad_bottom = nb
pad_left   = nb
pad_right  = nb

nz = nz0 + pad_top  + pad_bottom
nx = nx0 + pad_left + pad_right

def pad_field(f0):
    f = np.empty((nz, nx), dtype=np.float32)
    f[pad_top:pad_top+nz0, pad_left:pad_left+nx0] = f0
    f[:, :pad_left]      = f[:, pad_left:pad_left+1]
    f[:, pad_left+nx0:]  = f[:, pad_left+nx0-1:pad_left+nx0]
    f[:pad_top, :]       = f[pad_top:pad_top+1, :]
    f[pad_top+nz0:, :]   = f[pad_top+nz0-1:pad_top+nz0, :]
    return f

vp  = pad_field(vp0)
vs  = pad_field(vs0)
rho = pad_field(rho0)

mu      = (rho * vs**2).astype(np.float32)
lam     = (rho * vp**2 - 2.0 * mu).astype(np.float32)
lam2mu  = (lam + 2.0 * mu).astype(np.float32)
inv_rho = (1.0 / rho).astype(np.float32)

vp_max = float(vp.max())
dt = 0.4 * dx / vp_max
nt = 6000

print(f"dt={dt*1000:.4f} ms  dx={dx:.2f} m  CFL={vp_max*dt/dx:.3f}")
print(f"nz0={nz0} nx0={nx0}  nz={nz} nx={nx}")

f0 = 8.0
t0 = 1.2 / f0
src_amp = 1e9

def ricker(t):
    a = (np.pi * f0 * (t - t0)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)

sigma = np.zeros((nz, nx), dtype=np.float32)
sigma_lr_max     = 60.0
sigma_top_max    = 60.0
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

vx  = np.zeros((nz, nx), dtype=np.float32)
vz  = np.zeros((nz, nx), dtype=np.float32)
sxx = np.zeros((nz, nx), dtype=np.float32)
szz = np.zeros((nz, nx), dtype=np.float32)
sxz = np.zeros((nz, nx), dtype=np.float32)

src_x0 = nx0 // 2
src_z0 = 1
src_x  = pad_left + src_x0
src_z  = pad_top  + src_z0

def step(it):
    global vx, vz, sxx, szz, sxz
    src = src_amp * ricker(it * dt)
    sxx[src_z, src_x] += src
    szz[src_z, src_x] += src

    dvx_dx = (vx[1:-1, 2:]   - vx[1:-1, 1:-1]) / dx
    dvx_dz = (vx[2:,   1:-1] - vx[1:-1, 1:-1]) / dz
    dvz_dx = (vz[1:-1, 2:]   - vz[1:-1, 1:-1]) / dx
    dvz_dz = (vz[2:,   1:-1] - vz[1:-1, 1:-1]) / dz

    sxx[1:-1, 1:-1] += dt * (lam2mu[1:-1, 1:-1] * dvx_dx + lam[1:-1, 1:-1]    * dvz_dz)
    szz[1:-1, 1:-1] += dt * (lam[1:-1, 1:-1]    * dvx_dx + lam2mu[1:-1, 1:-1] * dvz_dz)
    sxz[1:-1, 1:-1] += dt * mu[1:-1, 1:-1] * (dvx_dz + dvz_dx)

    sxx *= damp
    szz *= damp
    sxz *= damp

    dsxx_dx = (sxx[1:-1, 1:-1] - sxx[1:-1, :-2]) / dx
    dsxz_dz = (sxz[1:-1, 1:-1] - sxz[:-2,  1:-1]) / dz
    dsxz_dx = (sxz[1:-1, 1:-1] - sxz[1:-1, :-2]) / dx
    dszz_dz = (szz[1:-1, 1:-1] - szz[:-2,  1:-1]) / dz

    vx[1:-1, 1:-1] += dt * inv_rho[1:-1, 1:-1] * (dsxx_dx + dsxz_dz)
    vz[1:-1, 1:-1] += dt * inv_rho[1:-1, 1:-1] * (dsxz_dx + dszz_dz)

    vx *= damp
    vz *= damp

def physical_view(field):
    f = field[pad_top:pad_top+nz0, pad_left:pad_left+nx0]
    return f[:nz_plot, :]

vel_view = vp0[:nz_plot, :]

x_extent = (nx0 - 1) * dx
z_extent = zmax_plot_m

main_width  = 12
main_height = main_width * (z_extent / x_extent)

fig = plt.figure(figsize=(main_width + 1.4, 2.0 * main_height), constrained_layout=True)
gs  = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.06], height_ratios=[1.0, 1.0])

axw  = fig.add_subplot(gs[0, 0])
caxw = fig.add_subplot(gs[0, 1])
axv  = fig.add_subplot(gs[1, 0])
caxv = fig.add_subplot(gs[1, 1])

im_w = axw.imshow(
    physical_view(vz),
    cmap="seismic",
    extent=[0, x_extent, z_extent, 0],
    aspect="equal",
    vmin=-1e-3,
    vmax=1e-3,
)
axw.set_title("Elastic Wavefield — Vz")
axw.set_xlabel("Distance (m)")
axw.set_ylabel("Depth (m)")
axw.plot(src_x0 * dx, src_z0 * dz, "k*", markersize=8)
fig.colorbar(im_w, cax=caxw, label="Vz (m/s)")

im_v = axv.imshow(
    vel_view,
    cmap="jet",
    extent=[0, x_extent, z_extent, 0],
    aspect="equal",
)
axv.set_title("Vp Model")
axv.set_xlabel("Distance (m)")
axv.set_ylabel("Depth (m)")
axv.plot(src_x0 * dx, src_z0 * dz, "w*", markersize=8)
fig.colorbar(im_v, cax=caxv, label="Velocity (m/s)")

steps_per_frame = 10

def update(frame):
    for _ in range(steps_per_frame):
        step(update.it)
        update.it += 1
    view = physical_view(vz)
    m = float(np.max(np.abs(view)))
    print(f"it={update.it}  max(vz)={m:.6e}")
    if m > 1e-20:
        im_w.set_clim(-0.6 * m, 0.6 * m)
    im_w.set_data(view)
    axw.set_title(f"Elastic Wavefield — Vz | Time = {update.it * dt * 1000:.1f} ms")
    return (im_w,)

update.it = 0
ani = FuncAnimation(fig, update, frames=nt // steps_per_frame, interval=30, blit=False)
plt.show()
