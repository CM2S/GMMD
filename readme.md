# GMMD

Geometrical microstructure generation by molecular dynamics.

GMMD is a numerical tool developed in the context of computational mechanics to aid the design and development of advanced materials. Employing a time-driven molecular dynamics simulation, it generates microstructures of matrix-composite materials in a computationally **efficient** and **robust** way, meshes them, and analyses them.

- [Overview](#overview)
- [Main features](#main-features)
- [Installation](#installation)
- [Usage](#usage)
- [Output](#output)

## Overview

### Description

GMMD has been designed with the main purpose of generating microstructures of matrix-composite materials in a computationally efficient and robust way, an important task in the development of new materials with innovative and enhanced properties. This is achieved using a time-driven **molecular dynamics simulation**, where the forces are repulsive and proportional to the overlap length of the particles.

Although nothing prevents the use of GMMD as a standalone program to produce microstructures for the analysis of a given material's behavior using multi-scale analysis, it is in applications such as the more recent data-driven material design frameworks, requiring large material response databases to train the underlying machine learning models, that its reasonable efficiency stands out.

### Computational framework

GMMD is designed and implemented in Python (Python 3 release), making it easily portable between all major computer platforms, easily integrated with other software implemented in different programming languages, and benefiting from an extensive collection of prebuilt (standard library) and third-party libraries. Given the extensive numerical nature of the program, its implementation relies heavily on the well-known [NumPy](https://numpy.org/devdocs/index.html) and [SciPy](https://www.scipy.org/) scientific computing packages, being most numerical tasks dispatched to compiled C code inside the Python interpreter.

### Authors

This program initial version was documented and fully coded by José Luís P. Vila-Chã<sup>[1](#f1)</sup> ([jvc@fe.up.pt](mailto:jvc@fe.up.pt)) and developed in colaboration with Bernardo P. Ferreira<sup>[1](#f1)</sup> ([bpferreira@fe.up.pt](mailto:bpferreira@fe.up.pt)) and Francisco M. Andrade Pires<sup>[2](#f2)</sup> ([fpires@fe.up.pt](mailto:fpires@fe.up.pt)).

<sup id="f1">1</sup> Member of CM2S research group, Department of Mechanical Engineering, Faculty of Engineering, University of Porto  
<sup id="f2">2</sup> Leader of CM2S research group, Department of Mechanical Engineering, Faculty of Engineering, University of Porto

## Main features

### Phase descriptors

- Diverse particle shapes available:
  - in two dimensions: disks and ellipses;
  - in three dimensions: spheres, ellipsoids, cylinders, and long cylindrical fibers.
- Flexible modeling of geometrical descriptors for particles within a given phase:
  - fixed value for all particles in a phase;
  - distributed according to a statistical distribution (uniform, normal, discrete, ...).

### Generation method

- Time-driven molecular dynamics simulation with repulsive forces proportional to the intersection length of the particles.
- Intersection length computed for general particles with convex shape using the GJK algorithm.
- Force computation sped up by a cell list, a Verlet list or a Verlet list computed from a cell list, as chosen in the input data file.
- Integration of the equations of motion using the Verlet integration scheme.
- Isokinetic thermostats: at a fixed temperature, or at temperature stages lowered until a legal configuration is found.
- Physically-based temperature lowering criterion capable of detecting equilibrium.
- Adaptive time step preventing instability of the integration method.
- Starting configuration for the simulation found through a Poisson point process or placed on a regular grid.
- Reproducible runs: a fixed seed gives the same microstructure on every run, and a different one for each sample of the set.
- Lightweight output option for data-driven frameworks.

### Meshing

- Regular mesh with the desired number of voxels in each spatial direction.
- Finite element mesh conforming to the particle boundaries, using Gmsh: first and second order triangles, quadrilaterals and tetrahedra.
- Meshes written as VTK files, and translated into the input files of LINKS, CRATE and Abaqus and into XDMF for FEniCS, each checked by running the solver on what is written; the Gmsh format and legacy VTK are written as well, for other meshers and viewers.

### Analysis

- Visualization of the microstructure and its meshes in ParaView (material phases, ...).
- Statistical descriptors of the microstructure: 2-point correlation function, Ripley's K function, nearest neighbour distances.
- Voronoi metrics based on the Minkowski structure metrics and the Minkowski irreducible tensors.
- Analysis of the generation run: kinetic energy, overlap and time step histories, and the paths of the particles.

## Installation

### Requirements

- **Python 3.10 or newer** (see [here](https://www.python.org/downloads/)). GMMD is tested on Python 3.10 to 3.13, on Linux and Windows.

  > In Linux/UNIX operative systems, Python can be installed from the apt library:
  > `sudo apt install python3`

- **pip** (see [here](https://pypi.org/project/pip/)), to install Python packages (learn [here](https://docs.python.org/3/installing/)).

  > In Linux/UNIX operative systems, pip can be installed from the apt library:
  > `sudo apt install python3-pip`

### Installing GMMD

Clone the repository:

```bash
git clone https://github.com/CM2S/GMMD.git
cd GMMD
```

Then install the package from the cloned directory, where `pyproject.toml` is located. That file defines the package metadata, dependencies, and the command-line entry points; `pip install` installs all required dependencies and registers the commands on your system.

For a **regular install**:

```bash
pip install .
```

For an **editable (development) install**, where changes to the source code are reflected immediately without reinstalling:

```bash
pip install -e .
```

After installation, the `geommicgen`, `geommicgen-mesh`, `geommicgen-translate`, `geommicgen-analyze` and `geommicgen-convert-mic` commands are available from any directory.

### Optional dependencies

- **Gmsh** is required to produce finite element meshes of the microstructures and the three dimensional visualizations. A run that only asks for a regular grid needs none of it. Install it with the extra:

  ```bash
  pip install 'geommicgen[gmsh]'
  ```

  GMMD requires **Gmsh 4.15 or newer**, the version it is tested against. If you instead install the SDK tarball by hand from [gmsh.info](https://gmsh.info/bin/Linux/), add its Python API to your `PYTHONPATH`:

  ```bash
  export PYTHONPATH=$PYTHONPATH:/path/to/gmsh/lib
  ```

- **h5py** is required only by the `xdmf` format, read by FEniCS, which stores its arrays in HDF5. The extra is named after the format:

  ```bash
  pip install 'geommicgen[xdmf]'
  ```

  Asking for one of these formats without its library fails that format alone, with the command that installs it; the mesh itself and the other formats are still written.

- **ParaView** (see [here](https://www.paraview.org/download/)) is required only to visualize the microstructures and meshes GMMD writes as VTK files (`.vti`, `.vtu`, `.vtk`), which ParaView opens directly (learn [here](https://www.paraview.org/resources/)).

  > In Linux/UNIX operative systems, ParaView can be installed by placing the tarball in the installation directory and extracting it:
  > `sudo tar -xvf ParaView-< version >.tar.gz`

### Tests and documentation

The test suite runs with pytest, installed by the `test` extra. The tests that need Gmsh are skipped when it is not installed.

```bash
pip install -e '.[test]'
pytest geommicgen/tests
```

`geommicgen/tests/test_solvers.py` runs the files GMMD writes through the solvers themselves, and each of its tests is skipped unless its solver is found: LINKS through `GEOMMICGEN_LINKS` or `LINKS` on the `PATH`, Abaqus through `GEOMMICGEN_ABAQUS` or `abaqus` on the `PATH`, FEniCS through `GEOMMICGEN_DOLFINX_PYTHON`, the path of a Python that has dolfinx, and CRATE through `GEOMMICGEN_CRATE_PYTHON`, the path of a Python that has `cratepy` -- an environment of its own, since `cratepy` pins versions of SciPy older than GMMD runs with. Their meshes stay under the thousand nodes the learning edition of Abaqus runs.

```bash
GEOMMICGEN_DOLFINX_PYTHON=/path/to/envs/dolfinx/bin/python pytest geommicgen/tests/test_solvers.py
```

The documentation is built with Sphinx, installed by the `docs` extra:

```bash
pip install -e '.[docs]'
python -m sphinx -b html docs docs/_build/html
```

## Usage

GMMD generates a set of microstructures from an **input data file**, and post-processes each one as the file asks. The post-processing -- **meshing** and **analysis** -- can also be run on its own, on any microstructure file, by a command of its own: this is how a microstructure generated earlier is meshed again or analysed further.

| What | Command |
|---|---|
| Generate a set of microstructures | `geommicgen input_data_file.mdsim` |
| Mesh a microstructure | `geommicgen-mesh mic.yaml ...` |
| Translate a mesh for a solver | `geommicgen-translate mesh.vtu --to ...` |
| Analyse a microstructure | `geommicgen-analyze mic.yaml ...` |
| Convert a `.mic` file of an earlier version | `geommicgen-convert-mic mic.mic` |

Each command lists its options with `--help`.

### The input data file

The input data file (`.mdsim`) contains all the required information to generate the samples of a microstructure -- its descriptors and the parameters of the generation process -- and the meshes and analyses to produce of each sample. A mesh is asked for under the name of the mesher that produces it, `gmsh` or `voxel`, the same names `geommicgen-mesh` takes with `--mesher`. A complete input data file where each parameter specification (either mandatory or optional) is fully documented (meaning, syntax, available options) is [examples/MIC_input_data_file.dat](examples/MIC_input_data_file.dat). This file can be copied to a given directory and be readily used by replacing the `[insert here]` boxes with the suitable specification. Two worked examples, a two dimensional microstructure of ellipses and a three dimensional one of ellipsoids, are beside it in [examples/](examples/); the test suite runs both.

### Generating a set of microstructures

To generate a new set of microstructures, provide the input data file as the only argument to the `geommicgen` command:

```bash
geommicgen input_data_file.mdsim
```

Each sample is generated, written to a folder of its own, and then meshed and analysed as the input data file asks. The program execution can be followed in the terminal, where data associated with the program launch, progress of the main execution phases, and the program end is output.

### Meshing a microstructure

The meshing is two stages -- a mesh of the microstructure, and that mesh in the formats solvers read -- and each has a command of its own. `geommicgen-mesh` discretises a microstructure file and writes the mesh; `geommicgen-translate` writes a mesh in the formats solvers read, taking the file the previous stage wrote, or one another tool produced:

```bash
geommicgen-mesh mic.yaml --mesher gmsh --mesh-size 0.05 --element-type tri6 --to links
geommicgen-mesh mic.yaml --mesher voxel --n-voxels-dims 100 100 --to crate
geommicgen-translate mic.vtu --to links,xdmf
geommicgen-translate --list-formats
```

A mesh is written as a VTK file whichever mesher produced it, `.vtu` for an unstructured mesh and `.vti` for a grid, with a small `.mesh.json` beside it holding what those formats cannot carry. The files are named after the microstructure file and what discretised it -- `mic_tri6.vtu`, `mic_100_100.vti` -- so that meshing one microstructure several ways into one directory keeps them apart; `--name` names them outright. Giving no `--to` stops after that file, which is a complete stage: `geommicgen-translate` picks up from it later, or somewhere else.

The LINKS and Abaqus decks, and the example CRATE input file written beside a grid, carry placeholder materials and a placeholder loading, a stretch along x, marked as such: a microstructure says nothing about either, but with them the decks run as written, which is how they are checked. The CRATE file also declares the dimensions of the RVE, which the grid does not carry. FEniCS reads the `.xdmf` file, with the phase of every cell as a cell tag; [`examples/fenics_elasticity.py`](examples/fenics_elasticity.py) reads one, gives each phase a material and solves.

The gmsh mesher offers `tri3`, `tri6`, `quad4` and `quad8` in two dimensions and `tetra4` and `tetra10` in three. A quadrilateral mesh is made by recombining triangles, and Gmsh keeps a triangle where it finds no pair for it, so a `quad4` or `quad8` mesh may hold a few triangles; when it does, the mesher says so with the counts, and the mesh is written with both, each type in a group of its own.

### Analysing a microstructure

`geommicgen-analyze` runs the analyses from a microstructure file and, for the motion of the particles, from the `md_state.npz` a generation run writes beside it:

```bash
geommicgen-analyze mic_0/mic.yaml --motion-analysis --stat-two-pt-corr -o mic_0/analysis
geommicgen-analyze mic.yaml --voronoi-analysis --voronoi-type set --plot-voronoi
```

Every analysis the input data file can ask for is a flag of the same name, and an earlier analysis in the output directory is written over.

### Microstructures of an earlier version

A microstructure used to be stored as a `.mic` file, a pickle of the objects that held it. It is a YAML file now, which any tool can read and a person can edit. `geommicgen-convert-mic` turns the one into the other, writing the state of the generation run beside it:

```bash
geommicgen-convert-mic mic_0/mic.mic
```

## Output

Running GMMD on an input data file, say `input_data_file.mdsim`, creates a folder of the same name beside it holding a copy of the file and a folder `mic_*` for each microstructure generated. A run never writes over an earlier one: a second run of the same file goes into `input_data_file_1/`, a third into `input_data_file_2/`.

```
input_data_file/
├── input_data_file.mdsim          copy of the input data file
├── mic_0/
│   ├── mic.yaml                   microstructure file
│   ├── status                     status of the generation: flag, time, final overlap
│   ├── md_state.npz               histories of the generation run, read by the motion analysis  (not with save_min)
│   ├── mic.screen                 log of the run, everything printed to the terminal             (not with save_min)
│   ├── final_config.pdf           visualization of the microstructure, when final_config is asked for
│   │                              (final_config.msh and final_config.vtk in three dimensions)
│   ├── meshes/                    the meshes asked for
│   ├── motion_results/            motion analysis: kinetic energy, total overlap, paths, ...
│   ├── stat_analysis_results/     statistical analysis: 2-point correlation, Ripley's K, ...
│   │                              plots, and stat_results.npz of one array per descriptor
│   └── voronoi_analysis_results/  Voronoi analysis: diagrams, diagrams with IMTs, histograms,
│                                  and voronoi_results.npz of the diagram and its metrics
├── mic_1/
│   └── ...
└── ...
```

`save_min`, the lightweight mode geared towards data-driven frameworks, keeps only the microstructure and status files.

`geommicgen-mesh`, `geommicgen-translate` and `geommicgen-analyze` write where their `-o` option says, the current directory by default.
