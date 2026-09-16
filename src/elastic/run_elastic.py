# run_elastic.py

import os

import numpy as np
from mpi4py import MPI

from fwmp.config import load_config
from fwmp.profiling import scorep_region
from fwmp.segy_io import load_segy
from fwmp.kernels import ElasticKernels
from fwmp.domain import (
    create_domain,
    make_halo_buffers,
    HaloExchanger,
    compute_kernel_bounds,
    compute_output_window,
)
from fwmp.material import (
    build_material_model,
    build_damping,
    extract_local_material,
)
from fwmp.simulation import (
    allocate_wavefields,
    ElasticSimulation,
    run_time_loop,
)
from fwmp.hdf5_output import (
    RankHDF5Writer,
    create_vds_file,
)


def main():
    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    cfg = load_config()

    rank_output_dir = os.path.join(cfg.base_output_dir, f"rank_{rank:04d}")
    rank_h5_path = os.path.join(rank_output_dir, "elastic_wavefield.h5")
    vds_path = os.path.join(cfg.base_output_dir, "elastic_wavefield.h5")

    relative_vds_source_path = os.path.join(
        f"rank_{rank:04d}",
        "elastic_wavefield.h5",
    )

    with scorep_region("load_kernel_library"):
        kernels = ElasticKernels()

    comm.Barrier()

    if rank == 0:
        with scorep_region("create_base_output_dir"):
            os.makedirs(cfg.base_output_dir, exist_ok=True)

    comm.Barrier()

    with scorep_region("create_rank_output_dir"):
        os.makedirs(rank_output_dir, exist_ok=True)

    comm.Barrier()

    with scorep_region("load_segy_inputs"):
        vp0 = load_segy(cfg.vp_path)[::cfg.ds, ::cfg.ds].astype(np.float32)
        vs0 = load_segy(cfg.vs_path)[::cfg.ds, ::cfg.ds].astype(np.float32)
        rho0 = load_segy(cfg.rho_path)[::cfg.ds, ::cfg.ds].astype(np.float32)

    nz0, nx0 = vp0.shape

    dx = np.float32(1.25 * cfg.ds)
    dz = np.float32(1.25 * cfg.ds)

    with scorep_region("build_material_model"):
        material = build_material_model(
            vp0=vp0,
            vs0=vs0,
            rho0=rho0,
            nb=cfg.nb,
        )

    nz = material["nz"]
    nx = material["nx"]

    pad_top = material["pad_top"]
    pad_bottom = material["pad_bottom"]
    pad_left = material["pad_left"]
    pad_right = material["pad_right"]

    vp_max = float(material["vp"].max())
    dt = np.float32(0.4 * float(dx) / vp_max)

    with scorep_region("build_damping"):
        damp = build_damping(
            nz=nz,
            nx=nx,
            pad_top=pad_top,
            pad_bottom=pad_bottom,
            pad_left=pad_left,
            pad_right=pad_right,
            dt=dt,
        )

    with scorep_region("create_cartesian_topology"):
        domain = create_domain(comm, nz, nx)

    with scorep_region("extract_local_material"):
        mu_loc, lam_loc, lam2mu_loc, inv_rho_loc, damp_loc = extract_local_material(
            material=material,
            damp=damp,
            domain=domain,
        )

    wavefields = allocate_wavefields(domain.nz_loc, domain.nx_loc)

    halo_bufs = make_halo_buffers(domain.nz_loc, domain.nx_loc)
    halo = HaloExchanger(domain, halo_bufs)

    iz0, iz1, jx0, jx1 = compute_kernel_bounds(domain, nz)

    output_window = compute_output_window(
        domain=domain,
        nz0=nz0,
        nx0=nx0,
        pad_top=pad_top,
        pad_left=pad_left,
    )

    n_frames = len(range(0, cfg.n_iterations, cfg.frame_stride))
    output_batch = max(1, min(cfg.output_batch, max(1, n_frames)))

    times = np.arange(n_frames, dtype=np.float32) * cfg.frame_stride * dt

    h5_attrs = {
        "nz0": nz0,
        "nx0": nx0,
        "dx": float(dx),
        "dz": float(dz),
        "dt": float(dt),
        "frame_stride": cfg.frame_stride,
        "n_frames": n_frames,
    }

    comm.Barrier()

    writer = RankHDF5Writer(
        rank_h5_path=rank_h5_path,
        relative_vds_source_path=relative_vds_source_path,
        rank=rank,
        size=size,
        domain=domain,
        output_window=output_window,
        vp0=vp0,
        times=times,
        n_frames=n_frames,
        output_batch=output_batch,
        attrs=h5_attrs,
    )

    f0 = np.float32(8.0)
    src_t0 = np.float32(1.2 / f0)
    src_amp = np.float32(1e9)

    src_x0 = nx0 // 2
    src_z0 = 1

    src_x = pad_left + src_x0
    src_z = pad_top + src_z0

    simulation = ElasticSimulation(
        kernels=kernels,
        halo=halo,
        domain=domain,
        wavefields=wavefields,
        mu_loc=mu_loc,
        lam_loc=lam_loc,
        lam2mu_loc=lam2mu_loc,
        inv_rho_loc=inv_rho_loc,
        damp_loc=damp_loc,
        dt=dt,
        dx=dx,
        dz=dz,
        iz0=iz0,
        iz1=iz1,
        jx0=jx0,
        jx1=jx1,
        src_z=src_z,
        src_x=src_x,
        src_amp=src_amp,
        f0=f0,
        src_t0=src_t0,
        direct_vz_source=cfg.direct_vz_source,
    )

    simulation.warmup_and_reset()

    run_time_loop(
        simulation=simulation,
        writer=writer,
        n_iterations=cfg.n_iterations,
        frame_stride=cfg.frame_stride,
    )

    writer.close()

    meta = writer.metadata()

    with scorep_region("gather_vds_metadata"):
        all_meta = comm.gather(meta, root=0)

    comm.Barrier()

    if rank == 0:
        vds_attrs = {
            "size": size,
            "dims_z": domain.dims[0],
            "dims_x": domain.dims[1],
            "nz0": nz0,
            "nx0": nx0,
            "dx": float(dx),
            "dz": float(dz),
            "dt": float(dt),
            "frame_stride": cfg.frame_stride,
            "n_frames": n_frames,
        }

        create_vds_file(
            vds_path=vds_path,
            all_meta=all_meta,
            vp0=vp0,
            times=times,
            n_frames=n_frames,
            nz0=nz0,
            nx0=nx0,
            attrs=vds_attrs,
        )

        print("done", flush=True)


if __name__ == "__main__":
    main()
