
import numpy as np
from fwmp.domain import add_halo_2d

def pad_field(f0, nz, nx, nz0, nx0, pad_top, pad_left):
    f = np.empty((nz, nx), dtype=np.float32)
    f[pad_top:pad_top + nz0, pad_left:pad_left + nx0] = f0
    f[:, :pad_left] = f[:, pad_left:pad_left + 1]
    f[:, pad_left + nx0:] = f[:, pad_left + nx0 - 1:pad_left + nx0]
    f[:pad_top, :] = f[pad_top:pad_top + 1, :]
    f[pad_top + nz0:, :] = f[pad_top + nz0 - 1:pad_top + nz0, :]
    return f

def build_material_model(vp0, vs0, rho0, nb):
    nz0, nx0 = vp0.shape
    pad_top = nb
    pad_bottom = nb
    pad_left = nb
    pad_right = nb
    nz = nz0 + pad_top + pad_bottom
    nx = nx0 + pad_left + pad_right
    vp = pad_field(vp0, nz, nx, nz0, nx0, pad_top, pad_left)
    vs = pad_field(vs0, nz, nx, nz0, nx0, pad_top, pad_left)
    rho = pad_field(rho0, nz, nx, nz0, nx0, pad_top, pad_left)
    mu = (rho * vs ** 2).astype(np.float32)
    lam = (rho * vp ** 2 - 2.0 * mu).astype(np.float32)
    lam2mu = (lam + 2.0 * mu).astype(np.float32)
    inv_rho = (1.0 / rho).astype(np.float32)
    return {
        "vp": vp,
        "vs": vs,
        "rho": rho,
        "mu": mu,
        "lam": lam,
        "lam2mu": lam2mu,
        "inv_rho": inv_rho,
        "nz": nz,
        "nx": nx,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_left": pad_left,
        "pad_right": pad_right,
    }

def ramp(n, power=2.0):
    return np.linspace(0.0, 1.0, n, dtype=np.float32) ** power

def build_damping(nz, nx, pad_top, pad_bottom, pad_left, pad_right, dt):
    sigma = np.zeros((nz, nx), dtype=np.float32)
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
    return damp

def extract_local_material(material, damp, domain):
    z0 = domain.z_start
    z1 = domain.z_end
    x0 = domain.x_start
    x1 = domain.x_end
    mu_loc = np.ascontiguousarray(add_halo_2d(material["mu"][z0:z1, x0:x1]))
    lam_loc = np.ascontiguousarray(add_halo_2d(material["lam"][z0:z1, x0:x1]))
    lam2mu_loc = np.ascontiguousarray(add_halo_2d(material["lam2mu"][z0:z1, x0:x1]))
    inv_rho_loc = np.ascontiguousarray(add_halo_2d(material["inv_rho"][z0:z1, x0:x1]))
    damp_loc = np.ascontiguousarray(add_halo_2d(damp[z0:z1, x0:x1]))
    return mu_loc, lam_loc, lam2mu_loc, inv_rho_loc, damp_loc
