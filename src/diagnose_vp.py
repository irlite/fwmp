#!/usr/bin/env python3

import argparse
import os
import sys

import hdf5plugin  # Must be imported before reading plugin-compressed HDF5 data.
import h5py
import numpy as np
import segyio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def result(status, message):
    print(f"[{status:5s}] {message}")


def safe_load_segy(path):
    """
    Load one SEG-Y trace per x-column, explicitly copying each trace.

    Returned shape:
        (number_of_samples, number_of_traces) == (nz, nx)
    """
    with segyio.open(path, "r", ignore_geometry=True) as f:
        return np.stack(
            [
                np.array(f.trace[i], dtype=np.float32, copy=True)
                for i in range(f.tracecount)
            ],
            axis=1,
        )


def test_segy_iterator_buffer(path):
    """
    Determine whether iteration over f.trace reuses memory.
    """
    print("\n=== SEG-Y iterator-buffer test ===")

    with segyio.open(path, "r", ignore_geometry=True) as f:
        print(f"Trace count:       {f.tracecount}")
        print(f"Samples per trace: {len(f.samples)}")

        if f.tracecount < 3:
            result("WARN", "Not enough traces for the iterator-buffer test.")
            return

        iterator = iter(f.trace)

        trace0 = np.asarray(next(iterator))
        trace0_snapshot = trace0.copy()
        pointer0 = trace0.__array_interface__["data"][0]

        trace1 = np.asarray(next(iterator))
        pointer1 = trace1.__array_interface__["data"][0]

        trace2 = np.asarray(next(iterator))
        pointer2 = trace2.__array_interface__["data"][0]

        trace0_changed = not np.array_equal(trace0, trace0_snapshot)
        shared_01 = np.shares_memory(trace0, trace1)
        shared_12 = np.shares_memory(trace1, trace2)

        print(f"Trace 0 pointer: {pointer0}")
        print(f"Trace 1 pointer: {pointer1}")
        print(f"Trace 2 pointer: {pointer2}")
        print(f"shares_memory(trace0, trace1): {shared_01}")
        print(f"shares_memory(trace1, trace2): {shared_12}")
        print(f"trace0 changed after iteration: {trace0_changed}")

        if pointer0 == pointer1 or pointer1 == pointer2:
            result(
                "FAIL",
                "The SEG-Y iterator reuses its data buffer. "
                "np.asarray(trace) is unsafe here.",
            )
        elif shared_01 or shared_12:
            result(
                "FAIL",
                "Consecutive trace arrays share memory. "
                "np.asarray(trace) is unsafe here.",
            )
        elif trace0_changed:
            result(
                "FAIL",
                "An earlier trace changed after advancing the iterator.",
            )
        else:
            result(
                "PASS",
                "No buffer reuse was detected in this short iterator test.",
            )


def array_statistics(name, a):
    print(f"\n=== {name} statistics ===")
    print(f"Shape:             {a.shape}")
    print(f"Data type:         {a.dtype}")
    print(f"Minimum:           {np.nanmin(a):.8g}")
    print(f"Maximum:           {np.nanmax(a):.8g}")
    print(f"Mean:              {np.nanmean(a):.8g}")
    print(f"Standard deviation:{np.nanstd(a):.8g}")
    print(f"NaN count:         {np.isnan(a).sum()}")
    print(f"Inf count:         {np.isinf(a).sum()}")

    if a.ndim != 2:
        result("FAIL", f"{name} is not two-dimensional.")
        return

    # Variation within each depth row as x changes.
    lateral_ranges = np.ptp(a, axis=1)

    # Variation within each x-column as depth changes.
    vertical_ranges = np.ptp(a, axis=0)

    lateral_difference = np.mean(
        np.abs(np.diff(a.astype(np.float64), axis=1))
    )
    vertical_difference = np.mean(
        np.abs(np.diff(a.astype(np.float64), axis=0))
    )

    print(f"Maximum lateral row range: {lateral_ranges.max():.8g}")
    print(f"Mean lateral row range:    {lateral_ranges.mean():.8g}")
    print(f"Mean adjacent-x difference:{lateral_difference:.8g}")
    print(f"Mean adjacent-z difference:{vertical_difference:.8g}")

    varying_rows = np.count_nonzero(lateral_ranges > 0)
    print(
        "Rows with lateral variation: "
        f"{varying_rows}/{a.shape[0]} "
        f"({100.0 * varying_rows / a.shape[0]:.2f}%)"
    )

    # Sample columns instead of running np.unique on the entire large model.
    sample_count = min(100, a.shape[1])
    sample_ids = np.linspace(
        0, a.shape[1] - 1, sample_count, dtype=np.int64
    )
    sampled_columns = a[:, sample_ids].T
    unique_sampled_columns = np.unique(sampled_columns, axis=0).shape[0]

    print(
        f"Unique columns among {sample_count} sampled columns: "
        f"{unique_sampled_columns}"
    )

    if varying_rows == 0:
        result(
            "FAIL",
            f"{name} has no lateral variation; every row is constant in x.",
        )
    elif unique_sampled_columns == 1:
        result(
            "FAIL",
            f"All sampled columns in {name} are identical.",
        )
    else:
        result("PASS", f"{name} contains lateral variation.")


def max_abs_difference_chunked(a, b, columns_per_chunk=256):
    if a.shape != b.shape:
        return np.inf

    maximum = 0.0

    for x0 in range(0, a.shape[1], columns_per_chunk):
        x1 = min(x0 + columns_per_chunk, a.shape[1])
        difference = np.max(
            np.abs(
                a[:, x0:x1].astype(np.float64)
                - b[:, x0:x1].astype(np.float64)
            )
        )
        maximum = max(maximum, float(difference))

    return maximum


def repeated_trace_error(stored, trace):
    if stored.shape[0] != trace.shape[0]:
        return np.inf

    maximum = 0.0

    for x0 in range(0, stored.shape[1], 256):
        x1 = min(x0 + 256, stored.shape[1])
        difference = np.max(
            np.abs(
                stored[:, x0:x1].astype(np.float64)
                - trace[:, None].astype(np.float64)
            )
        )
        maximum = max(maximum, float(difference))

    return maximum


def compare_source_and_hdf5(source, stored):
    print("\n=== Source/HDF5 comparison ===")
    print(f"Safe SEG-Y shape: {source.shape}")
    print(f"HDF5 vp shape:    {stored.shape}")

    if source.shape != stored.shape:
        result(
            "FAIL",
            "The SEG-Y and HDF5 dimensions differ. With ds=1 they should match.",
        )

        if source.T.shape == stored.shape:
            result(
                "FAIL",
                "The HDF5 shape matches the transpose of the SEG-Y model. "
                "There may be an orientation error.",
            )

        return

    difference = max_abs_difference_chunked(source, stored)
    print(f"Maximum |safe SEG-Y - HDF5|: {difference:.8g}")

    if difference == 0:
        result(
            "PASS",
            "HDF5 /vp exactly matches the safely loaded SEG-Y model.",
        )
    elif np.allclose(source, stored, rtol=1e-6, atol=1e-5):
        result(
            "PASS",
            "HDF5 /vp matches the safely loaded SEG-Y model within tolerance.",
        )
    else:
        result(
            "FAIL",
            "HDF5 /vp does not match the safely loaded SEG-Y model.",
        )

    first_error = repeated_trace_error(stored, source[:, 0])
    middle_error = repeated_trace_error(
        stored, source[:, source.shape[1] // 2]
    )
    last_error = repeated_trace_error(stored, source[:, -1])

    print("\nRepeated-source-trace tests:")
    print(f"  Every HDF5 column = first SEG-Y trace:  error {first_error:.8g}")
    print(f"  Every HDF5 column = middle SEG-Y trace: error {middle_error:.8g}")
    print(f"  Every HDF5 column = last SEG-Y trace:   error {last_error:.8g}")

    if last_error == 0:
        result(
            "FAIL",
            "Every HDF5 column equals the last SEG-Y trace. "
            "This is strong evidence of segyio iterator-buffer reuse.",
        )
    elif first_error == 0:
        result(
            "FAIL",
            "Every HDF5 column equals the first SEG-Y trace.",
        )
    elif middle_error == 0:
        result(
            "FAIL",
            "Every HDF5 column equals one middle SEG-Y trace.",
        )


def inspect_hdf5(h5_path):
    print("\n=== HDF5 metadata ===")
    print(f"HDF5 path:     {os.path.realpath(h5_path)}")
    print(f"HDF5 size:     {os.path.getsize(h5_path)} bytes")
    print(f"HDF5 modified: {os.path.getmtime(h5_path)}")

    with h5py.File(h5_path, "r") as h5:
        print("Datasets:")

        def show(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(
                    f"  {name}: shape={obj.shape}, "
                    f"dtype={obj.dtype}, virtual={obj.is_virtual}"
                )

        h5.visititems(show)

        print("Attributes:")
        for key, value in h5.attrs.items():
            print(f"  {key} = {value}")

        if "vp" not in h5:
            raise RuntimeError("The HDF5 file does not contain /vp")

        vp = h5["vp"][:]

        if "vz" in h5:
            expected_shape = h5["vz"].shape[1:]
            print(f"Expected model shape from /vz: {expected_shape}")

            if vp.shape == expected_shape:
                result("PASS", "/vp and individual /vz frames have the same shape.")
            else:
                result(
                    "FAIL",
                    f"/vp shape {vp.shape} differs from /vz frame "
                    f"shape {expected_shape}.",
                )

        if "nz0" in h5.attrs and "nx0" in h5.attrs:
            attribute_shape = (
                int(h5.attrs["nz0"]),
                int(h5.attrs["nx0"]),
            )

            if vp.shape == attribute_shape:
                result("PASS", "/vp matches the nz0/nx0 attributes.")
            else:
                result(
                    "FAIL",
                    f"/vp shape {vp.shape} differs from attributes "
                    f"{attribute_shape}.",
                )

    return vp


def test_rank_tiles(h5_path, source):
    base_dir = os.path.dirname(os.path.realpath(h5_path))
    rank_dirs = sorted(
        name for name in os.listdir(base_dir)
        if name.startswith("rank_")
        and os.path.isdir(os.path.join(base_dir, name))
    )

    print("\n=== Per-rank model-tile tests ===")

    if not rank_dirs:
        result("WARN", "No rank_* directories were found.")
        return

    passed = 0
    failed = 0
    skipped = 0

    for rank_dir in rank_dirs:
        path = os.path.join(
            base_dir, rank_dir, "elastic_wavefield.h5"
        )

        if not os.path.isfile(path):
            result("WARN", f"Missing {path}")
            skipped += 1
            continue

        with h5py.File(path, "r") as h5:
            if "vp" not in h5:
                # Ranks containing only padding have no vp dataset.
                skipped += 1
                continue

            z0 = int(h5.attrs["z0"])
            z1 = int(h5.attrs["z1"])
            x0 = int(h5.attrs["x0"])
            x1 = int(h5.attrs["x1"])
            tile = h5["vp"][:]
            expected = source[z0:z1, x0:x1]

            if tile.shape != expected.shape:
                result(
                    "FAIL",
                    f"{rank_dir}: tile shape {tile.shape}, "
                    f"expected {expected.shape}",
                )
                failed += 1
                continue

            difference = max_abs_difference_chunked(tile, expected)

            if difference == 0:
                passed += 1
            else:
                result(
                    "FAIL",
                    f"{rank_dir}: maximum source difference {difference:.8g}",
                )
                failed += 1

    print(
        f"Rank tiles: passed={passed}, failed={failed}, skipped={skipped}"
    )

    if failed == 0 and passed > 0:
        result("PASS", "All tested rank /vp tiles match the safe SEG-Y load.")
    elif failed > 0:
        result("FAIL", "Some rank /vp tiles do not match the SEG-Y source.")


def make_plots(source, stored, output_path):
    print("\n=== Creating comparison plot ===")

    same_shape = source.shape == stored.shape

    if same_shape:
        difference = stored.astype(np.float64) - source.astype(np.float64)

        fig, axes = plt.subplots(
            3, 1, figsize=(16, 11), constrained_layout=True
        )

        panels = [
            (source, "Safely loaded SEG-Y Vp", "viridis"),
            (stored, "HDF5 /vp", "viridis"),
            (difference, "HDF5 /vp minus safe SEG-Y", "seismic"),
        ]
    else:
        fig, axes = plt.subplots(
            2, 1, figsize=(16, 8), constrained_layout=True
        )

        panels = [
            (source, "Safely loaded SEG-Y Vp", "viridis"),
            (stored, "HDF5 /vp", "viridis"),
        ]

    for ax, (data, title, cmap) in zip(axes, panels):
        image = ax.imshow(
            data,
            cmap=cmap,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel("x index")
        ax.set_ylabel("z index")
        fig.colorbar(image, ax=ax)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--segy",
        required=True,
        help="Path to MODEL_P-WAVE_VELOCITY_1.25m.segy",
    )
    parser.add_argument(
        "--h5",
        required=True,
        help="Path to the assembled elastic_wavefield.h5",
    )
    parser.add_argument(
        "--plot",
        default="vp_diagnostic.png",
        help="Output comparison image",
    )
    parser.add_argument(
        "--skip-ranks",
        action="store_true",
        help="Do not inspect rank_*/elastic_wavefield.h5",
    )
    args = parser.parse_args()

    segy_path = os.path.realpath(args.segy)
    h5_path = os.path.realpath(args.h5)
    plot_path = os.path.realpath(args.plot)

    print("=== Resolved paths ===")
    print(f"SEG-Y: {segy_path}")
    print(f"HDF5:  {h5_path}")
    print(f"Plot:  {plot_path}")

    if not os.path.isfile(segy_path):
        raise FileNotFoundError(segy_path)

    if not os.path.isfile(h5_path):
        raise FileNotFoundError(h5_path)

    test_segy_iterator_buffer(segy_path)

    print("\nLoading SEG-Y safely...")
    source = safe_load_segy(segy_path)
    array_statistics("Safely loaded SEG-Y", source)

    stored = inspect_hdf5(h5_path)
    array_statistics("HDF5 /vp", stored)

    compare_source_and_hdf5(source, stored)

    if not args.skip_ranks:
        test_rank_tiles(h5_path, source)

    make_plots(source, stored, plot_path)


if __name__ == "__main__":
    main()
