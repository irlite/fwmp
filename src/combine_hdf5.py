import os
import sys
import glob
import h5py
import numpy as np

if len(sys.argv) != 2:
    raise SystemExit("usage: python combine_hdf5.py <job_id>")

job_id = sys.argv[1]
job_name = job_id if job_id.startswith("job_") else f"job_{job_id}"

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, ".."))
output_dir = os.path.join(root_dir, "output")
job_dir = os.path.join(output_dir, job_name)

if not os.path.isdir(job_dir):
    raise RuntimeError(f"job directory not found: {job_dir}")

rank_files = sorted(glob.glob(os.path.join(job_dir, "rank_*", "elastic_wavefield.h5")))

if not rank_files:
    raise RuntimeError(f"no rank files found in {job_dir}")

combined_path = os.path.join(output_dir, f"elastic_wavefield_{job_name}.h5")

if os.path.exists(combined_path):
    os.remove(combined_path)

with h5py.File(rank_files[0], "r") as f0:
    n_frames = int(f0.attrs["n_frames"])
    nz0 = int(f0.attrs["nz0"])
    nx0 = int(f0.attrs["nx0"])
    attrs = dict(f0.attrs.items())
    time = f0["time"][:]

layout_vz = h5py.VirtualLayout(shape=(n_frames, nz0, nx0), dtype=np.float32)
layout_vp = h5py.VirtualLayout(shape=(nz0, nx0), dtype=np.float32)

usable = []

for path in rank_files:
    with h5py.File(path, "r") as f:
        if "vz" not in f:
            continue

        x0 = int(f.attrs["x0"])
        x1 = int(f.attrs["x1"])

        if "z0" in f.attrs and "z1" in f.attrs:
            z0 = int(f.attrs["z0"])
            z1 = int(f.attrs["z1"])
        else:
            z0 = 0
            z1 = nz0

        if x0 < 0 or x1 <= x0 or z0 < 0 or z1 <= z0:
            continue

        rel_path = os.path.relpath(path, output_dir)
        vz_shape = tuple(f["vz"].shape)

        usable.append((rel_path, z0, z1, x0, x1, vz_shape, "vp" in f))

for rel_path, z0, z1, x0, x1, vz_shape, has_vp in usable:
    local_nz = z1 - z0
    local_nx = x1 - x0

    source_vz = h5py.VirtualSource(rel_path, "vz", shape=vz_shape)
    layout_vz[:, z0:z1, x0:x1] = source_vz[:, :, :]

    if has_vp:
        source_vp = h5py.VirtualSource(rel_path, "vp", shape=(local_nz, local_nx))
        layout_vp[z0:z1, x0:x1] = source_vp[:, :]

with h5py.File(combined_path, "w", libver="latest") as h5:
    h5.create_virtual_dataset("vz", layout_vz, fillvalue=0.0)
    h5.create_virtual_dataset("vp", layout_vp, fillvalue=0.0)
    h5.create_dataset("time", data=time)

    for key, value in attrs.items():
        if key not in [
            "rank",
            "coord_z",
            "coord_x",
            "z0",
            "z1",
            "x0",
            "x1",
            "local_nz_phys",
            "local_nx_phys",
            "local_nx",
            "local_nx_phys",
        ]:
            h5.attrs[key] = value

    h5.attrs["job_id"] = job_id
    h5.attrs["job_name"] = job_name
    h5.attrs["is_virtual_dataset"] = True
    h5.attrs["source_file_count"] = len(usable)

print(f"Created virtual HDF5 file: {combined_path}")
print(f"Sources are in: {job_dir}")
