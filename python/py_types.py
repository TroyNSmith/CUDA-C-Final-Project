import numpy as np
import numpy.typing as npt
import ctypes
from typing import Annotated, Literal
from dataclasses import dataclass

# Generic atoms
Atom_Count = int
Frame_Count = int


# Coordinates types
identifiers_dtype = np.dtype(
    [
        ("residue_idx", np.int32),
        ("residue_name", "S5"),
        ("atom_name", "S5"),
        ("atom_idx", np.int32),
    ]
)
Identifiers = Annotated[
    npt.NDArray[identifiers_dtype],
    Literal[Atom_Count, Atom_Count, Atom_Count, Atom_Count],
]


# Trajectory types
class XDRFile(ctypes.Structure):
    pass


Trajectory = Annotated[npt.NDArray[np.float32], Literal[Frame_Count, Atom_Count, 3]]


class Coordinates:
    @dataclass
    class Frame:
        coordinates: npt.ArrayLike
        box: npt.ArrayLike
        time: ctypes.c_float
