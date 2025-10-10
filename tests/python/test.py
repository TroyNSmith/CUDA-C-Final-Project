from python import wrapper, atoms
import time
import matplotlib.pyplot as plt

gro_file = "/home/tns97255/CUDA-C-Final-Project/tests/data/test.gro"
xtc_file = "/home/tns97255/CUDA-C-Final-Project/tests/data/test.xtc"

df = atoms.read_gro_file(gro_file, coords=True)
pairs = atoms.atom_res_idx_pairs(df)

start_time = time.perf_counter()
bins, g_r = wrapper.radial_distribution(xtc_file, pairs)
print(type(bins))
end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(elapsed_time)

plt.plot(bins, g_r)
plt.savefig("/home/tns97255/CUDA-C-Final-Project/tests/python/test.png")