from dataclasses import dataclass
import numpy as np
from mpi4py import MPI


@dataclass
class Domain:
    comm: object
    cart: object
    rank: int
    size: int
    dims: list
    coord_z: int
    coord_x: int
    z_minus: int
    z_plus: int
    x_minus: int
    x_plus: int
    z_start: int
    z_end: int
    x_start: int
    x_end: int
    nz_loc: int
    nx_loc: int

@dataclass
class OutputWindow:
    has: bool
    local_i0: int
    local_i1: int
    local_j0: int
    local_j1: int
    out_z0: int
    out_z1: int
    out_x0: int
    out_x1: int
    local_nz_phys: int
    local_nx_phys: int

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

def create_domain(comm, nz, nx):
    rank = comm.Get_rank()
    size = comm.Get_size()
    dims = choose_dims(size, nz, nx)
    cart = comm.Create_cart(
        dims=dims,
        periods=[False, False],
        reorder=False,
    )
    coord_z, coord_x = cart.Get_coords(rank)
    z_minus, z_plus = cart.Shift(0, 1)
    x_minus, x_plus = cart.Shift(1, 1)
    z_start, z_end, _ = split_1d(nz, dims[0], coord_z)
    x_start, x_end, _ = split_1d(nx, dims[1], coord_x)
    nz_loc = z_end - z_start
    nx_loc = x_end - x_start
    if nz_loc <= 0 or nx_loc <= 0:
        raise RuntimeError("empty local domain")
    return Domain(
        comm=comm,
        cart=cart,
        rank=rank,
        size=size,
        dims=dims,
        coord_z=coord_z,
        coord_x=coord_x,
        z_minus=z_minus,
        z_plus=z_plus,
        x_minus=x_minus,
        x_plus=x_plus,
        z_start=z_start,
        z_end=z_end,
        x_start=x_start,
        x_end=x_end,
        nz_loc=nz_loc,
        nx_loc=nx_loc,
    )

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


class HaloExchanger:
    def __init__(self, domain, halo_bufs):
        self.domain = domain
        self.halo_bufs = halo_bufs

    def exchange_forward(self, fields):
        d = self.domain
        nf = len(fields)
        send_minus = self.halo_bufs["x_send_minus"][:nf]
        recv_plus = self.halo_bufs["x_recv_plus"][:nf]
        for k, a in enumerate(fields):
            send_minus[k, :] = a[1:-1, 1]
        d.cart.Sendrecv(
            send_minus,
            dest=d.x_minus,
            sendtag=10,
            recvbuf=recv_plus,
            source=d.x_plus,
            recvtag=10,
        )
        if d.x_plus != MPI.PROC_NULL:
            for k, a in enumerate(fields):
                a[1:-1, -1] = recv_plus[k, :]
        send_minus = self.halo_bufs["z_send_minus"][:nf]
        recv_plus = self.halo_bufs["z_recv_plus"][:nf]
        for k, a in enumerate(fields):
            send_minus[k, :] = a[1, 1:-1]
        d.cart.Sendrecv(
            send_minus,
            dest=d.z_minus,
            sendtag=20,
            recvbuf=recv_plus,
            source=d.z_plus,
            recvtag=20,
        )
        if d.z_plus != MPI.PROC_NULL:
            for k, a in enumerate(fields):
                a[-1, 1:-1] = recv_plus[k, :]

    def exchange_backward(self, fields):
        d = self.domain
        nf = len(fields)
        send_plus = self.halo_bufs["x_send_plus"][:nf]
        recv_minus = self.halo_bufs["x_recv_minus"][:nf]
        for k, a in enumerate(fields):
            send_plus[k, :] = a[1:-1, -2]
        d.cart.Sendrecv(
            send_plus,
            dest=d.x_plus,
            sendtag=30,
            recvbuf=recv_minus,
            source=d.x_minus,
            recvtag=30,
        )
        if d.x_minus != MPI.PROC_NULL:
            for k, a in enumerate(fields):
                a[1:-1, 0] = recv_minus[k, :]
        send_plus = self.halo_bufs["z_send_plus"][:nf]
        recv_minus = self.halo_bufs["z_recv_minus"][:nf]
        for k, a in enumerate(fields):
            send_plus[k, :] = a[-2, 1:-1]
        d.cart.Sendrecv(
            send_plus,
            dest=d.z_plus,
            sendtag=40,
            recvbuf=recv_minus,
            source=d.z_minus,
            recvtag=40,
        )
        if d.z_minus != MPI.PROC_NULL:
            for k, a in enumerate(fields):
                a[0, 1:-1] = recv_minus[k, :]

def compute_kernel_bounds(domain, nz):
    iz0 = 1
    iz1 = domain.nz_loc + 1
    if domain.z_start == 0:
        iz0 = 2
    if domain.z_end == nz:
        iz1 = domain.nz_loc
    jx0 = 1
    jx1 = domain.nx_loc + 1
    return iz0, iz1, jx0, jx1

def compute_output_window(domain, nz0, nx0, pad_top, pad_left):
    physical_z_start = pad_top
    physical_z_end = pad_top + nz0
    physical_x_start = pad_left
    physical_x_end = pad_left + nx0
    write_z0 = max(domain.z_start, physical_z_start)
    write_z1 = min(domain.z_end, physical_z_end)
    write_x0 = max(domain.x_start, physical_x_start)
    write_x1 = min(domain.x_end, physical_x_end)
    has = write_z0 < write_z1 and write_x0 < write_x1
    if has:
        local_i0 = write_z0 - domain.z_start + 1
        local_i1 = write_z1 - domain.z_start + 1
        local_j0 = write_x0 - domain.x_start + 1
        local_j1 = write_x1 - domain.x_start + 1
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
    return OutputWindow(
        has=has,
        local_i0=local_i0,
        local_i1=local_i1,
        local_j0=local_j0,
        local_j1=local_j1,
        out_z0=out_z0,
        out_z1=out_z1,
        out_x0=out_x0,
        out_x1=out_x1,
        local_nz_phys=local_nz_phys,
        local_nx_phys=local_nx_phys,
    )
