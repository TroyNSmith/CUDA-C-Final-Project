import ctypes
import os
import numpy as np

lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib', 'lib.so'))
lib = ctypes.CDLL(lib_path)

lib.radial_distribution.argtypes = [
    ctypes.c_char_p,                 # .xtc file location
    ctypes.POINTER(ctypes.c_float),  # pointer to pairs
    ctypes.c_int,                    # number of pairs
    ctypes.POINTER(ctypes.c_float),  # bins
    ctypes.POINTER(ctypes.c_float),  # g_r
    ctypes.c_int                     # num_bins
]
lib.radial_distribution.restype = ctypes.c_int

def radial_distribution(path: str, pairs: list, num_bins: int = 1000):
    """Calculate a radial distribution function based on residue centers of mass.
    """
    pairs_ptr = pairs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) # Pairs pointer

    bins = np.zeros(num_bins, dtype=np.float32)
    g_r = np.zeros(num_bins, dtype=np.float32)

    bins_ptr = bins.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    g_r_ptr = g_r.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    lib.radial_distribution(path.encode(), pairs_ptr, pairs.size, bins_ptr, g_r_ptr, num_bins)

    return bins, g_r