import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
)
from python import coordinates, functions
from functools import partial

system = coordinates.System(
    topology="/home/tns97255/CUDA-C-Final-Project/tests/data/test.gro",
    trajectory="/home/tns97255/CUDA-C-Final-Project/tests/data/test.xtc",
    center_of_masses=True,
)
bins = np.linspace(0.1, 2.50, num=int(2.50 / 0.01))
rdf = functions.time_average(partial(functions.radial_distribution), system)

plt.plot(bins[:-1], rdf)
plt.savefig("/home/tns97255/CUDA-C-Final-Project/tests/python/test.png")
