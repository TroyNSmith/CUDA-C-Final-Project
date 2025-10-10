import ctypes
import os

lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib', 'lib.so'))
lib = ctypes.CDLL(lib_path)

lib.read_xtc_file.argtypes = [ctypes.c_char_p]
lib.read_xtc_file.restype = ctypes.c_int

def read_xtc(path: str):
    """Read a GROMACS .xtc file"""
    return lib.read_xtc_file(path.encode())