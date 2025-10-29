import ctypes
import os
import sys
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree

from .coordinates import System

lib_name = "lib.dll" if sys.platform.startswith("win") or os.name == "nt" else "lib.so"
lib_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "lib", lib_name)
)
try:
    lib = ctypes.CDLL(lib_path)
except OSError as e:
    raise OSError(f"Failed to load native library at {lib_path}: {e}")


def time_average(
    function: Callable, system_1: System, system_2: System = None, skip: int = 0
):
    """Find the time average of a function.

    If system_2 is not provided, system_1 will be analyzed against itself for functions requiring multiple coordinate sets.

    args:
        function: Callable function with np.arraylike return
        system_1: First system to analyze
        system_2: Second system to analyze
        skip: Skip every n frames
    """
    skip += 1

    if system_2 is None:
        system_2 = system_1

    # This part is slow-ish (when loading frames that may or may not be included in the analysis)
    results = None
    counts = 0
    for i, (frame_1, frame_2) in enumerate(zip(system_1.frames, system_2.frames)):
        if i % skip != 0:
            continue

        if i >= 25:
            break

        counts += 1

        pbc = np.diagonal(frame_1.box)

        assert np.array_equal(pbc, np.diagonal(frame_2.box)), (
            f"Systems 1 and 2 do not have matching boxes at time {frame_1.time}."
        )
        frame_results = function(pbc, frame_1.coordinates, frame_2.coordinates)

        if results is None:
            results = frame_results
        else:
            results += frame_results

    return results / counts


def radial_distribution_py(
    pbc: ArrayLike,
    coords_1: ArrayLike,
    coords_2: ArrayLike = None,
    bins: ArrayLike = None,
    dist_cutoff: float = 2.50,
):
    """Compute radical distribution between two sets of coordinates.

    If frame_2 is not provided, radial distances amongst coordinates in frame_1 will be considered."""
    block_size = 10000

    if coords_2 is None:
        coords_2 = coords_1
        distinct = False
    else:
        distinct = True

    if bins is None:
        bins = np.linspace(0.1, dist_cutoff, num=int(dist_cutoff / 0.01))

    kdt_2 = cKDTree(coords_2 % pbc, boxsize=pbc)

    histogram = np.zeros(len(bins) - 1, dtype=np.float32)
    for start in range(0, len(coords_1), block_size):
        end = min(start + block_size, len(coords_1))
        current_block = coords_1[start:end]

        neighbors = kdt_2.query_ball_point(current_block, r=dist_cutoff)

        for i, neighbor in enumerate(neighbors):
            if not neighbor:
                continue

            ref = current_block[i]

            displacements = abs(coords_2[neighbor] - ref)
            distances = np.linalg.norm(displacements, axis=1)

            if not distinct:
                mask = neighbors > i + start  # Don't double count atoms
                if not np.any(mask):
                    continue
                distances = distances[mask]

            block_histogram, _ = np.histogram(distances, bins=bins)
            histogram += block_histogram

    histogram = histogram / len(coords_2)
    histogram = histogram / (
        4 / 3 * np.pi * bins[1:] ** 3 - 4 / 3 * np.pi * bins[:-1] ** 3
    )
    histogram = histogram / (len(coords_1) / np.prod(pbc))

    return histogram


def radial_distribution_c(
    box: ArrayLike,
    coords_1: ArrayLike,
    coords_2: ArrayLike = None,
    num_bins: int = 100,
    dist_cutoff: float = 2.50,
):
    if coords_2 is None:
        coords_2 = coords_1

    lib.radial_distribution.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # coords_1
        ctypes.c_int,  # n1
        ctypes.POINTER(ctypes.c_float),  # coords_2
        ctypes.c_int,  # n2
        ctypes.POINTER(ctypes.c_float),  # g_r
        ctypes.c_int,  # num_bins
        ctypes.POINTER(ctypes.c_float),  # box
        ctypes.c_float,  # r_max
    ]
    lib.radial_distribution.restype = ctypes.c_int

    g_r = np.zeros(num_bins, dtype=np.float32)

    res = lib.radial_distribution(
        coords_1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        len(coords_1),
        coords_2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        len(coords_2),
        g_r.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        num_bins,
        box.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_float(dist_cutoff),
    )

    print("Histogram:", g_r)


if __name__ == "__main__":
    radial_distribution_c()
