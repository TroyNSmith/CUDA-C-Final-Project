import os
import sys

import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
)
from python import functions

rows = 100

# Generate random coordinates matrices
rng = np.random.default_rng()
pbc = rng.uniform(low=0.0, high=10.0, size=(1, 3))
test_coords_1 = rng.uniform(low=0.0, high=10.0, size=(rows, 3))
test_coords_2 = rng.uniform(low=0.0, high=10.0, size=(rows, 3))

frame_results_py, elapsed_time_py = functions.radial_distribution_naive_py(
    pbc, test_coords_1, test_coords_2
)

frame_results_C, elapsed_time_C = functions.radial_distribution_naive_c(
    pbc, test_coords_1, test_coords_2
)

frame_results_Cu, elapsed_time_Cu, kernel_elapsed_time = functions.radial_distribution_naive_cuda(
    pbc, test_coords_1, test_coords_2
)

bins = np.linspace(0, 2.50, num=100)

print(f"Python elapsed time: {elapsed_time_py}")
print(f"C elapsed time: {elapsed_time_C}")
print(f"Cuda total elapsed time: {elapsed_time_Cu}, Cuda kernel time: {kernel_elapsed_time}")
print(f"Arrays match: {np.allclose(frame_results_py, frame_results_C) and np.allclose(frame_results_py, frame_results_Cu)}")
