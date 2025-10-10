import numpy as np
import pandas as pd

from python.constants import MASSES

def atom_res_idx_pairs(topology: pd.DataFrame, masses: bool = True) -> np.array:
    """Returns atom-residue index pairs as a 1D Numpy array with dtype = np.float32

    Use read_gro_file to initialize the topology dataframe.

    Return format: [atom_id1, res_id1, atom_mass1, atom_id2, res_id2, atom_mass2, ...] (with masses)
    
    Return format: [atom_id1, res_id1, atom_id2, res_id2, ...] (without masses)

    To-do: Apply a filter for only specific atoms/residues
    """
    columns = ["atom_id", "res_id"]
    if masses:
        columns += ["mass"]
    pairs = topology[columns].to_numpy(dtype=np.float32) # Returns N x 3 matrix for N atoms
    return pairs.flatten() # Convert to 1D array
    
def get_element(atom_name: str) -> str:
    """Extracts the chemical element symbol from atom name."""
    if len(atom_name) == 0:
        return ""

    # Check if second character is alphabetic; if so, assume that is is a part of element symbol.
    if len(atom_name) > 1 and atom_name[1].isalpha():
        return atom_name[:2].capitalize()
    else:
        return atom_name[0].capitalize()

def read_gro_file(path: str, coords: bool = False, masses: bool = True) -> pd.DataFrame:
    """Parses information from gro file into pandas dataframe based on standardized column widths."""
    widths = [5, 5, 5, 5]  # res_id, res_name, atom_name, atom_id
    names = ["res_id", "res_name", "atom_name", "atom_id"]

    if coords:
        widths += [8, 8, 8]
        names += ["x", "y", "z"]

    df = pd.read_fwf(path, widths=widths, names=names, skiprows=2, skipfooter=1, engine="python")
    df[["res_id", "atom_id"]] = df[["res_id", "atom_id"]].astype(int)

    if coords:
        df[["x", "y", "z"]] = df[["x", "y", "z"]].astype(float)

    if masses:
        # Extract elements and map to masses
        df["element"] = df["atom_name"].apply(get_element)
        df["mass"] = df["element"].map(MASSES)

    return df