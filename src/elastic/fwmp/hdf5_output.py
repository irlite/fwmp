import os
import h5py
import numpy as np
from fwmp.profiling import scorep_region

class RankHDF5Writer:
    def __init__(
        self,
        rank_h5_path,
        relative_vds_source_path,
        rank,
        size,
        domain,
        output_window,
        vp0,
        times,
        n_frames,
        output_batch,
        attrs,
    ):
        self.rank_h5_path = rank_h5_path
        self.relative_vds_source_path = relative_vds_source_path
        self.rank = rank
        self.size = size
        self.domain = domain
        self.window = output_window
        self.vp0 = vp0
        self.times = times
        self.n_frames = n_frames
        self.output_batch = max(1, int(output_batch))
        self.has = output_window.has
        self.h5 = None
        self.dset_vz = None
        self.batch_buf = None
        self.batch_count = 0
        self.batch_base = 0
        with scorep_region("hdf5_open_rank_file"):
            self.h5 = h5py.File(
                rank_h5_path,
                "w",
                libver="latest",
                alignment_threshold=4 * 1024 ** 2,
                alignment_interval=4 * 1024 ** 2,
                meta_block_size=4 * 1024 ** 2,
            )

        with scorep_region("hdf5_create_rank_datasets"):
            self._create_datasets()
        with scorep_region("hdf5_write_rank_attrs"):
            self._write_attrs(attrs)

    def _create_datasets(self):
        w = self.window
        if self.has:
            self.dset_vz = self.h5.create_dataset(
                "vz",
                shape=(self.n_frames, w.local_nz_phys, w.local_nx_phys),
                dtype=np.float32,
                track_times=False,
            )
            self.h5.create_dataset(
                "vp",
                data=self.vp0[w.out_z0:w.out_z1, w.out_x0:w.out_x1].astype(np.float32),
                track_times=False,
            )
            self.batch_buf = np.empty(
                (
                    self.output_batch,
                    w.local_nz_phys,
                    w.local_nx_phys,
                ),
                dtype=np.float32,
            )
        self.h5.create_dataset(
            "time",
            data=self.times,
            track_times=False,
        )

    def _write_attrs(self, attrs):
        w = self.window
        d = self.domain
        base_attrs = {
            "rank": self.rank,
            "size": self.size,
            "dims_z": d.dims[0],
            "dims_x": d.dims[1],
            "coord_z": d.coord_z,
            "coord_x": d.coord_x,
            "z0": w.out_z0,
            "z1": w.out_z1,
            "x0": w.out_x0,
            "x1": w.out_x1,
            "local_nz_phys": w.local_nz_phys,
            "local_nx_phys": w.local_nx_phys,
        }
        for key, value in base_attrs.items():
            self.h5.attrs[key] = value
        for key, value in attrs.items():
            self.h5.attrs[key] = value

    def write_vz_frame(self, frame_id, vz):
        if not self.has:
            return
        w = self.window
        if self.batch_count == 0:
            self.batch_base = frame_id
        with scorep_region("copy_output_frame_to_batch"):
            self.batch_buf[self.batch_count, :, :] = vz[
                w.local_i0:w.local_i1,
                w.local_j0:w.local_j1,
            ]
        self.batch_count += 1
        if self.batch_count == self.output_batch:
            self.flush()

    def flush(self):
        if not self.has:
            return
        if self.batch_count <= 0:
            return
        with scorep_region("hdf5_write_vz_batch"):
            self.dset_vz[
                self.batch_base:self.batch_base + self.batch_count,
                :, :,
            ] = self.batch_buf[:self.batch_count, :, :]
        self.batch_count = 0

    def close(self):
        self.flush()
        with scorep_region("hdf5_close_rank_file"):
            self.h5.close()

    def metadata(self):
        w = self.window
        return {
            "rank": self.rank,
            "has": bool(self.has),
            "z0": int(w.out_z0),
            "z1": int(w.out_z1),
            "x0": int(w.out_x0),
            "x1": int(w.out_x1),
            "path": self.relative_vds_source_path,
        }


def create_vds_file(
    vds_path,
    all_meta,
    vp0,
    times,
    n_frames,
    nz0,
    nx0,
    attrs,
):
    with scorep_region("create_vds_file"):
        with h5py.File(
            vds_path,
            "w",
            libver="latest",
            alignment_threshold=4 * 1024 ** 2,
            alignment_interval=4 * 1024 ** 2,
            meta_block_size=4 * 1024 ** 2,
        ) as vf:
            layout = h5py.VirtualLayout(
                shape=(n_frames, nz0, nx0),
                dtype=np.float32,
            )
            for m in all_meta:
                if not m["has"]:
                    continue
                src_shape = (
                    n_frames,
                    m["z1"] - m["z0"],
                    m["x1"] - m["x0"],
                )
                source = h5py.VirtualSource(
                    m["path"],
                    "vz",
                    shape=src_shape,
                )
                layout[
                    :,
                    m["z0"]:m["z1"],
                    m["x0"]:m["x1"],
                ] = source
            vf.create_virtual_dataset(
                "vz",
                layout,
                fillvalue=0.0,
            )
            vf.create_dataset(
                "vp",
                data=vp0.astype(np.float32),
                track_times=False,
            )
            vf.create_dataset(
                "time",
                data=times,
                track_times=False,
            )
            for key, value in attrs.items():
                vf.attrs[key] = value
