import os
import sys

import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
)
from python import coordinates, functions

system = coordinates.System(
    topology="/home/tns97255/CUDA-C-Final-Project/tests/data/test.gro",
    trajectory="/home/tns97255/CUDA-C-Final-Project/tests/data/test.xtc",
)
for i, frame in enumerate(system.frames):
    if i > 1:
        break
    print(len(frame.coordinates))
    functions.radial_distribution_c(np.diagonal(frame.box), frame.coordinates)

# bins = np.linspace(0.1, 2.50, num=int(2.50 / 0.01))
# rdf = functions.time_average(partial(functions.radial_distribution), system)

# plt.plot(bins[:-1], rdf)
# plt.savefig("/home/tns97255/CUDA-C-Final-Project/tests/python/test.png")
