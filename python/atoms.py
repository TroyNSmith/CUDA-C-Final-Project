import pandas as pd

def read_gro_file(path: str) -> pd.DataFrame:
    """Parse information from gro file into pandas dataframe based on standardized column widths
    """
    widths = [5, 5, 5, 5] # We don't need x, y, z coordinates
    names = ["res_id", "res_name", "atom_name", "atom_id"]

    df = pd.read_fwf(path, widths=widths, names=names, skiprows=2, skipfooter=1, engine="python")

    df[['resid', 'atomid']] = df[['resid', 'atomid']].astype(int)

    return df