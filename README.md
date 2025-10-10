# CUDA-C-Final-Project

## Prerequisites
Install [Pixi](https://pixi.sh/latest/) to manage necessary dependencies.

Linux & macOS: ```curl -fsSL https://pixi.sh/install.sh | sh```

Windows: [Download Installer](https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-pc-windows-msvc.msi)

Once installed, run ```pixi shell``` in the terminal to initialize the environment.

## What is an RDF?
First of all, let's define what a system is. A system is a closed region of space containing atoms. These atoms can be organized into higher-order structures, molecules, by little packets of energy connecting them (bonds), as shown below.

![Image of two molecules with labelled atoms](https://enthu.com/blog/media/image-62.png)

In reality, these atoms are little packets of energy governed by the laws of quantum mechanics. In an MD simulation, we instead assume that Newtonian mechanics can describe the atoms within reasonable error, and we assign empirical constants to them and their bonds to simulate energies & forces.

When we simulate a system of atoms or molecules over time using Molecular Dynamics (MD), we're generating a time series of positions for every atom. We typically do a bunch of math using Newtonian force equations, like Coulomb's Law, and use this to simulate how the atoms move. Every so often, we take a "picture" of the atoms (i.e., record their positions), which we call a frame. At the end of the simulation, we have a "movie" containing pictures of the simulation at every time step T, which is what we call the trajectory. These trajectories allow us to calculate various physical properties of the system, like density, viscosity, ... One important property is the Radial Distribution Function (RDF), often written as g(r). The RDF is a way to measure how atoms or molecules (often referred to as residues) are distributed relative to each other — how likely you are to find a pair of atoms at a certain distance apart.

![Example of how an RDF is defined in real space](https://upload.wikimedia.org/wikipedia/commons/e/ea/Molecular_Schematic_for_Interpreting_a_Radial_Distribution_Function.png)

![Example of an RDF graph](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b8/Simulated_Radial_Distribution_Functions_for_Solid%2C_Liquid%2C_and_Gaseous_Argon.svg/2560px-Simulated_Radial_Distribution_Functions_for_Solid%2C_Liquid%2C_and_Gaseous_Argon.svg.png)

To compute an RDF, we need to:

1. Loop over all pairs of atoms / residues (or a subset, e.g., oxygen–oxygen only).
2. For each pair, calculate the Euclidean distance r between atoms.
3. Bin that distance into a histogram.
4. Normalize the histogram to produce g(r).

Step 2 results in a distance matrix, D, where each row & column index corresponds to an atom index, and the entry at D[i][j] describes the distance between atoms i and j. Consequently, a symmetric matrix is produced, where D[i][j] = 0 if i = j and D[i][j] = D[j][i] if i != j. In step 3, we count all of the entries with lower bound < distance =< upper bound and put those counts into a histogram. Each lower and upper bound is defined as a bin edge, and typically the radial distance is divided into an equal number of bins. There are various ways of going about step 4, normalizing the histogram.

With pairwise distance calculations scaling up to O(N<sup>2</sup>) for N number of atoms when calculations are performed sequentially, resource usage quickly becomes a critical concern for large systems (lots of atoms) or long trajectories (lots of time). However, there are two key opportunities for optimization:

1. Frame-Level Parallelism | Since each frame of the trajectory is independent with respect to RDF computation, they can be processed in parallel across CPU threads or GPU blocks.
2. Atom-Pair Parallelism | Within a single frame, each atom pair contributes independently to the RDF histogram. This makes it possible to compute distances and binning in parallel. For example, assigning each thread (on a GPU) a unique atom pair or tiling the distance matrix to minimize global memory usage.

Additionally, redundant work can be avoided by:
1. Skipping calculations along the main diagonal (D[i][j] = 0 if i = j).
2. Processing only one triangle of the matrix (D[i][j] = D[j][i] if i != j).

## How do we extract data from an MD simulation?
One popular open-source software for performing an MD simulation is GROMACS (GROningen MAchine for Chemical Simulations), which typically produces:

1. A .gro file describing the topology of the system (atom indexes, residues, elements (and molecular environments), coordinates). Each row corresponds to one atom and the columns are padded for consistent parsing.
2. A .trr or .xtc file describing the trajectory of the system (coordinates at each frame of simulation).

A .gro file only contains the initial frame of coordinates, and the trajectory file only contains coordinates for each frame, so that combined, a user can parse any information about the system and its evolution in time.

If we know how each row in a .gro is formatted, we can parse the file manually using Python. Trajectory files are not human-readable, however, so we rely on libraries like [libxdrfile](https://github.com/wesbarnett/libxdrfile) to parse the data for us in C, Python, Rust, ...