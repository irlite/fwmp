import os
import glob
import h5py
import numpy as np

output_dir = "../output"
vds_path = os.path.join(output_dir, "elastic_wavefield.h5")
rank_files = sorted(glob.glob(os.path.join(output_dir, "elastic_wavefield_rank*.h5")))
if not rank_files:
    raise RuntimeError("No per-rank HDF5 files found")
if os.path.exists(vds_path):
    os.remove(vds_path)
with h5py.File(rank_files[0], "r") as f0:
    n_frames = int(f0.attrs["n_frames"])
    nz_plot = int(f0.attrs["nz_plot"])
    nx0 = int(f0.attrs["nx0"])
    attrs = dict(f0.attrs.items())
    time = f0["time"][:]
layout_vz = h5py.VirtualLayout(shape=(n_frames, nz_plot, nx0), dtype=np.float32)
layout_vp = h5py.VirtualLayout(shape=(nz_plot, nx0), dtype=np.float32)
for path in rank_files:
    with h5py.File(path, "r") as f:
        x0 = int(f.attrs["x0"])
        x1 = int(f.attrs["x1"])
        if x0 < 0 or x1 <= x0:
            continue
        local_nx = x1 - x0
    source_name = os.path.basename(path)
    source_vz = h5py.VirtualSource(source_name, "vz", shape=(n_frames, nz_plot, local_nx))
    source_vp = h5py.VirtualSource(source_name, "vp", shape=(nz_plot, local_nx))
    layout_vz[:, :, x0:x1] = source_vz[:, :, :]
    layout_vp[:, x0:x1] = source_vp[:, :]
with h5py.File(vds_path, "w", libver="latest") as h5:
    h5.create_virtual_dataset("vz", layout_vz, fillvalue=0.0)
    h5.create_virtual_dataset("vp", layout_vp, fillvalue=0.0)
    h5.create_dataset("time", data=time)
    for key, value in attrs.items():
        if key not in ["rank", "x0", "x1", "local_nx_phys"]:
            h5.attrs[key] = value
    h5.attrs["is_virtual_dataset"] = True
    h5.attrs["source_file_count"] = len(rank_files)
print(f"Created virtual HDF5 file: {vds_path}")
print("Do not delete or move the elastic_wavefield_rank*.h5 files unless you also recreate the VDS")
