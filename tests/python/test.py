from python import wrapper, atoms

gro_file = "/home/myid/tns97255/CUDA-C-Final-Project/tests/data/test.gro"
xtc_file = "/home/myid/tns97255/CUDA-C-Final-Project/tests/data/test.xtc"

natoms = wrapper.read_xtc(xtc_file)

lines = atoms.read_gro_file(gro_file)