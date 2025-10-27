from pydantic import BaseModel, Field, computed_field
import numpy as np
from . import py_types as pt
from . import constants
from typing import Optional
import ctypes
import os
import sys


class System(BaseModel):
    topology: str = Field(pattern=r".gro|.tpr")
    trajectory: str = Field(pattern=r".xtc")

    center_of_masses: Optional[bool] = False
    unwrap_pbc: Optional[bool] = False

    model_config = {"arbitrary_types_allowed": True}

    @computed_field
    @property
    def atom_count(self) -> np.ndarray:
        return len(self.identifiers)

    @computed_field
    @property
    def frames(self) -> pt.Coordinates.Frame:
        """
        Returns a generator object that yields one frame at a time to improve speed and memory consumption.

        # To-do: Filter the residue_idxs vector (can be done to filter out both atoms and residues)
        """
        residue_idxs = self.identifiers["residue_idx"] if (self.unwrap_pbc or self.center_of_masses) else None
        masses = self.masses if self.center_of_masses else None
        if ".xtc" in self.trajectory:
            wrapper = Wrappers()
            return wrapper.xtc_reader(
                self.trajectory, self.atom_count, residue_idxs, masses
            )

    @computed_field
    @property
    def elements(self) -> np.ndarray:
        get_element_vec = np.vectorize(lambda b: Properties.get_element(b.decode()))
        return get_element_vec(self.identifiers["atom_name"])

    @computed_field
    @property
    def identifiers(self) -> np.ndarray:
        if ".gro" in self.topology:
            return Readers.gromacs(self.topology)

    @computed_field
    @property
    def masses(self) -> np.ndarray:
        get_mass_vec = np.vectorize(lambda el: constants.MASSES.get(el, np.nan))
        return get_mass_vec(self.elements)


class Readers:
    def gromacs(topology: str) -> list[dict]:
        """Parses a GROMACS .gro file into a NumPy array with valid ctypes."""
        with open(topology, "r") as f:
            lines = f.readlines()[2:-1]

        data = np.empty(len(lines), dtype=pt.identifiers_dtype)

        for i, line in enumerate(lines):
            data[i] = (
                int(line[0:5]),
                line[5:10].strip().encode(),
                line[10:15].strip().encode(),
                int(line[15:20]),
            )

        return data


class Properties:
    def get_element(atom_name: str) -> str:
        """Extracts the chemical element symbol from atom name."""
        if len(atom_name) == 0:
            return ""

        # Check if second character is alphabetic; if so, assume that it is a part of element symbol.
        if len(atom_name) > 1 and atom_name[1].isalpha():
            return atom_name[:2].capitalize()
        else:
            return atom_name[0].capitalize()


def pbc_unwrap(coordinates, box, residue_idxs, masses=None):
    """
    Unwrap residues under periodic boundary conditions.
    
    args:
        coordinates : (N,3) array
        box         : (3,3) box matrix or array of lengths
        residue_idxs: (N,) residue IDs
        masses      : (N,) atom masses, optional
    """
    coords = np.asarray(coordinates, dtype=np.float64)
    box = np.asarray(box)

    if box.shape == (3,3):
        pbc = np.diagonal(box)
    else:
        pbc = box

    coords %= pbc

    unique_residues = np.unique(residue_idxs)


    if masses is not None:
        com_coords = np.zeros((len(unique_residues), 3))

        for i, resid in enumerate(unique_residues):
            idx = np.where(residue_idxs == resid)[0]
            residue_coords = coords[idx]
            m = masses[idx, np.newaxis]
            com_coords[i] = (residue_coords * m).sum(axis=0) / m.sum()
        
        coords = com_coords

    else:
        for resid in unique_residues:
            idx = np.where(residue_idxs == resid)[0]
            residue_coords = coords[idx]

            center = residue_coords.mean(axis=0)

            disp = residue_coords - center
            disp -= np.round(disp / pbc) * pbc

            coords[idx] = center + disp

    return coords


class Wrappers:
    def __init__(self):
        lib_name = "lib.dll" if sys.platform.startswith("win") or os.name == "nt" else "lib.so"
        lib_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "lib", lib_name)
        )
        try:
            self.lib = ctypes.CDLL(lib_path)
        except OSError as e:
            raise OSError(f"Failed to load native library at {lib_path}: {e}")
    
    def xtc_reader(
        self,
        xtc_path: str,
        atom_count: int,
        residue_idxs: Optional[np.ndarray] = None,
        masses: Optional[np.ndarray] = None,
        precision: float = 1000.0,
    ) -> np.ndarray:
        """
        Reads all frames from a GROMACS .xtc file and returns atom coordinates as a NumPy array.

        If a vector of residue_idxs is provided, the coordinates will be unwrapped from PBC according to residue groups.
        If masses and residue_idx vectors are provided, the coordinates will be condensed into centers of mass.
        """

        lib = self.lib

        # ctypes arrays matching C types
        rvec3 = ctypes.c_float * 3
        matrix3x3 = rvec3 * 3

        # Memory allocations
        coords = (rvec3 * atom_count)()  # rvec * natoms
        box = matrix3x3()  # orthorhombic box dimensions
        step = ctypes.c_int()
        time = ctypes.c_float()
        prec = ctypes.c_float(precision)

        # Open XTC file
        lib.xdrfile_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        lib.xdrfile_open.restype = ctypes.POINTER(pt.XDRFile)

        xd_ptr = lib.xdrfile_open(xtc_path.encode(), b"r")
        if not xd_ptr:
            raise RuntimeError(f"Failed to open XTC file: {xtc_path}")

        # Set read_xtc argument types
        lib.read_xtc.argtypes = [
            ctypes.POINTER(pt.XDRFile),  # XDRFILE*
            ctypes.c_int,  # natoms
            ctypes.POINTER(ctypes.c_int),  # step
            ctypes.POINTER(ctypes.c_float),  # time
            matrix3x3,  # box
            ctypes.POINTER(rvec3),  # coordinates
            ctypes.POINTER(ctypes.c_float),  # precision
        ]
        lib.read_xtc.restype = ctypes.c_int
        while True:
            result = lib.read_xtc(
                xd_ptr,
                atom_count,
                ctypes.byref(step),
                ctypes.byref(time),
                box,
                coords,
                ctypes.byref(prec),
            )

            if result != 0:
                break

            # Convert coordinates to NumPy array for manipulating
            coords_np = np.ctypeslib.as_array(coords)
            coords_np.shape = (atom_count, 3)

            if residue_idxs is not None:
                coords_np = pbc_unwrap(coords_np, box, residue_idxs, masses)

            yield pt.Coordinates.Frame(coords_np, box, time)
