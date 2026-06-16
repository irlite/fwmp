import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

def main():
    h5_path = sys.argv[1] if len(sys.argv) > 1 else "../output/elastic_wavefield.h5"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "../output/elastic_wavefield.webm"
    h5 = h5py.File(h5_path, "r")
    vz = h5["vz"]
    vp = h5["vp"][:]
    dx = float(h5.attrs["dx"])
    dz = float(h5.attrs["dz"])
    dt = float(h5.attrs["dt"])
    frame_stride = int(h5.attrs["frame_stride"])
    n_frames, nz, nx = vz.shape
    x_extent = (nx - 1) * dx
    z_extent = (nz - 1) * dz
    times = h5["time"][:] if "time" in h5 else np.arange(n_frames, dtype=np.float32) * frame_stride * dt
    src_x0 = int(h5.attrs["src_x0"]) if "src_x0" in h5.attrs else None
    src_z0 = int(h5.attrs["src_z0"]) if "src_z0" in h5.attrs else None
    print(f"Loaded: {h5_path}")
    print(f"vz shape: {vz.shape}")
    print(f"vp shape: {vp.shape}")
    print(f"n_frames={n_frames}, nx={nx}, nz={nz}")
    sample_ids = np.linspace(0, n_frames - 1, min(n_frames, 50), dtype=int)
    vmax = 0.0
    for i in sample_ids:
        vmax = max(vmax, float(np.percentile(np.abs(vz[i]), 99.5)))
    if vmax <= 0.0:
        vmax = 1e-12
    print(f"Wavefield color scale: +/- {vmax:.3e}")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    axw = axes[0]
    axv = axes[1]
    im_w = axw.imshow(vz[0], cmap="seismic", extent=[0.0, x_extent, z_extent, 0.0], aspect="equal", vmin=-vmax, vmax=vmax, interpolation="nearest")
    axw.set_title("Elastic Wavefield — Vz")
    axw.set_xlabel("Distance x (m)")
    axw.set_ylabel("Depth z (m)")
    fig.colorbar(im_w, ax=axw, label="Vz (m/s)")
    im_v = axv.imshow(vp, cmap="jet", extent=[0.0, x_extent, z_extent, 0.0], aspect="equal", interpolation="nearest")
    axv.set_title("P-wave Velocity Model")
    axv.set_xlabel("Distance x (m)")
    axv.set_ylabel("Depth z (m)")
    fig.colorbar(im_v, ax=axv, label="Vp (m/s)")
    if src_x0 is not None and src_z0 is not None:
        src_x_m = src_x0 * dx
        src_z_m = src_z0 * dz
        axw.plot(src_x_m, src_z_m, marker="*", color="yellow", markersize=12, markeredgecolor="black", label="Source")
        axv.plot(src_x_m, src_z_m, marker="*", color="yellow", markersize=12, markeredgecolor="black", label="Source")
        axw.legend(loc="upper right")
        axv.legend(loc="upper right")

    def update(i):
        frame = vz[i]
        im_w.set_data(frame)
        t_ms = float(times[i]) * 1000.0
        m = float(np.max(np.abs(frame)))
        axw.set_title(f"Elastic Wavefield — Vz | frame {i + 1}/{n_frames} | t = {t_ms:.1f} ms | max = {m:.2e}")
        return (im_w,)

    anim = FuncAnimation(fig, update, frames=n_frames, interval=40, blit=False, repeat=False)

    writer = FFMpegWriter(fps=25, bitrate=2000,
                          codec="libvpx-vp9",
                          extra_args=["-pix_fmt", "yuv420p"])

    print(f"Saving animation to: {output_path}")
    anim.save(output_path, writer=writer, dpi=150,
              progress_callback=lambda i, n: print(f"\r  frame {i + 1}/{n}", end="", flush=True))
    print("\nDone.")

    plt.close(fig)
    h5.close()

if __name__ == "__main__":
    main()
