import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))
from python import wrapper, atoms
import time
import matplotlib.pyplot as plt

gro_file = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "data/test.gro"))
xtc_file = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "data/test.xtc"))

df = atoms.read_gro_file(gro_file, coords=True)
pairs = atoms.atom_res_idx_pairs(df)

start_time = time.perf_counter()
bins, g_r = wrapper.radial_distribution(xtc_file, pairs)
print(type(bins))
end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(elapsed_time)

plt.plot(bins, g_r)
plt.savefig(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "python/test.png")))