import ctypes
import os
import sys
from time import perf_counter
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

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

        if i >= 1:
            break

        counts += 1

        pbc = np.diagonal(frame_1.box)

        assert np.array_equal(pbc, np.diagonal(frame_2.box)), (
            f"Systems 1 and 2 do not have matching boxes at time {frame_1.time}."
        )
        frame_results, elapsed_time = function(
            pbc, frame_1.coordinates, frame_2.coordinates
        )

        print(f"Iteration {i}: {elapsed_time} s")

        if results is None:
            results = frame_results
        else:
            results += frame_results

    return results / counts


def radial_distribution_py(
    box: np.ndarray,
    coords_1: np.ndarray,
    coords_2: np.ndarray,
    num_bins: int = 100,
    r_max: float = 2.50,
):
    """
    Direct Python equivalent of the C function `radial_distribution`.
    Uses 32-bit floats and identical arithmetic/branching for bit-for-bit comparison.
    """

    # --- Argument validation ---
    if coords_1 is None or coords_2 is None or box is None:
        raise ValueError("Null pointer argument")

    # Match C data layout (flat, float32)
    coords_1 = np.ascontiguousarray(coords_1, dtype=np.float32).ravel()
    coords_2 = np.ascontiguousarray(coords_2, dtype=np.float32).ravel()
    box = np.ascontiguousarray(box, dtype=np.float32).ravel()

    n1 = len(coords_1) // 3
    n2 = len(coords_2) // 3

    # Allocate output
    g_r = np.zeros(num_bins, dtype=np.float32)

    # Match C: float division -> float
    r_max = np.float32(r_max)
    bin_width = np.float32(r_max / np.float32(num_bins))

    half = np.float32(0.5)
    one = np.float32(1.0)

    start = perf_counter()

    # --- identical logic to C code ---
    for i in range(n1):
        for j in range(n2):
            dx = np.float32(abs(coords_1[3 * i + 0] - coords_2[3 * j + 0]))
            dy = np.float32(abs(coords_1[3 * i + 1] - coords_2[3 * j + 1]))
            dz = np.float32(abs(coords_1[3 * i + 2] - coords_2[3 * j + 2]))

            if dx > half * box[0]:
                dx = box[0] - dx
            if dy > half * box[1]:
                dy = box[1] - dy
            if dz > half * box[2]:
                dz = box[2] - dz

            r = np.float32(np.sqrt(dx * dx + dy * dy + dz * dz))
            bin_idx = int(r / bin_width)

            if 0 <= bin_idx < num_bins:
                g_r[bin_idx] += one

    elapsed = perf_counter() - start
    return g_r, elapsed


def radial_distribution_c(
    box: ArrayLike,
    coords_1: ArrayLike,
    coords_2: ArrayLike = None,
    num_bins: int = 100,
    dist_cutoff: float = 2.50,
):
    if coords_2 is None:
        coords_2 = coords_1

    coords_1 = np.asarray(coords_1, dtype=np.float32).ravel()
    coords_2 = np.asarray(coords_2, dtype=np.float32).ravel()
    box = np.asarray(box, dtype=np.float32).ravel()

    n1 = len(coords_1) // 3
    n2 = len(coords_2) // 3

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

    start_time = perf_counter()

    res = lib.radial_distribution(
        coords_1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n1,
        coords_2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n2,
        g_r.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        num_bins,
        box.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_float(dist_cutoff),
    )

    elapsed_time = perf_counter() - start_time
    return g_r, elapsed_time
