from dataclasses import dataclass
import numpy as np
from fwmp.profiling import scorep_region

@dataclass
class Wavefields:
    vx: np.ndarray
    vz: np.ndarray
    sxx: np.ndarray
    szz: np.ndarray
    sxz: np.ndarray
    def zero(self):
        self.vx.fill(0.0)
        self.vz.fill(0.0)
        self.sxx.fill(0.0)
        self.szz.fill(0.0)
        self.sxz.fill(0.0)

def allocate_wavefields(nz_loc, nx_loc):
    with scorep_region("allocate_wavefields"):
        shape = (nz_loc + 2, nx_loc + 2)
        return Wavefields(
            vx=np.zeros(shape, dtype=np.float32),
            vz=np.zeros(shape, dtype=np.float32),
            sxx=np.zeros(shape, dtype=np.float32),
            szz=np.zeros(shape, dtype=np.float32),
            sxz=np.zeros(shape, dtype=np.float32),
        )

class ElasticSimulation:
    def __init__(
        self,
        kernels,
        halo,
        domain,
        wavefields,
        mu_loc,
        lam_loc,
        lam2mu_loc,
        inv_rho_loc,
        damp_loc,
        dt,
        dx,
        dz,
        iz0,
        iz1,
        jx0,
        jx1,
        src_z,
        src_x,
        src_amp,
        f0,
        src_t0,
        direct_vz_source,
    ):
        self.kernels = kernels
        self.halo = halo
        self.domain = domain
        self.w = wavefields
        self.mu_loc = mu_loc
        self.lam_loc = lam_loc
        self.lam2mu_loc = lam2mu_loc
        self.inv_rho_loc = inv_rho_loc
        self.damp_loc = damp_loc
        self.dt = np.float32(dt)
        self.dx = np.float32(dx)
        self.dz = np.float32(dz)
        self.iz0 = iz0
        self.iz1 = iz1
        self.jx0 = jx0
        self.jx1 = jx1
        self.src_z = src_z
        self.src_x = src_x
        self.src_amp = np.float32(src_amp)
        self.f0 = np.float32(f0)
        self.src_t0 = np.float32(src_t0)

        self.direct_vz_source = int(direct_vz_source)

    def ricker(self, t):
        a = (np.pi * float(self.f0) * (float(t) - float(self.src_t0))) ** 2
        return (1.0 - 2.0 * a) * np.exp(-a)

    def warmup_and_reset(self):
        with scorep_region("warmup_kernels"):
            self.kernels.update_stress(
                self.w.vx,
                self.w.vz,
                self.w.sxx,
                self.w.szz,
                self.w.sxz,
                self.lam_loc,
                self.lam2mu_loc,
                self.mu_loc,
                self.damp_loc,
                self.dt,
                self.dx,
                self.dz,
                self.iz0,
                self.iz1,
                self.jx0,
                self.jx1,
            )
            self.kernels.update_velocity(
                self.w.vx,
                self.w.vz,
                self.w.sxx,
                self.w.szz,
                self.w.sxz,
                self.inv_rho_loc,
                self.damp_loc,
                self.dt,
                self.dx,
                self.dz,
                self.iz0,
                self.iz1,
                self.jx0,
                self.jx1,
            )
        self.w.zero()

    def step(self, it):
        d = self.domain
        w = self.w
        self.halo.exchange_forward([w.vx, w.vz])
        src = np.float32(self.src_amp * self.ricker(np.float32(it) * self.dt))
        if d.z_start <= self.src_z < d.z_end and d.x_start <= self.src_x < d.x_end:
            li = self.src_z - d.z_start + 1
            lj = self.src_x - d.x_start + 1
            w.sxx[li, lj] += src
            w.szz[li, lj] += src
            if self.direct_vz_source:
                w.vz[li, lj] += np.float32(self.dt * self.inv_rho_loc[li, lj] * src)
        self.kernels.update_stress(
            w.vx,
            w.vz,
            w.sxx,
            w.szz,
            w.sxz,
            self.lam_loc,
            self.lam2mu_loc,
            self.mu_loc,
            self.damp_loc,
            self.dt,
            self.dx,
            self.dz,
            self.iz0,
            self.iz1,
            self.jx0,
            self.jx1,
        )
        self.halo.exchange_backward([w.sxx, w.szz, w.sxz])
        self.kernels.update_velocity(
            w.vx,
            w.vz,
            w.sxx,
            w.szz,
            w.sxz,
            self.inv_rho_loc,
            self.damp_loc,
            self.dt,
            self.dx,
            self.dz,
            self.iz0,
            self.iz1,
            self.jx0,
            self.jx1,
        )

def run_time_loop(simulation, writer, n_iterations, frame_stride):
    frame_id = 0
    with scorep_region("time_loop"):
        for it in range(n_iterations):
            simulation.step(it)
            if it % frame_stride == 0:
                writer.write_vz_frame(frame_id, simulation.w.vz)
                frame_id += 1
    writer.flush()
