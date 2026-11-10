# GMMD

[![Tests](https://github.com/CM2S/GMMD/actions/workflows/test.yaml/badge.svg?branch=master)](https://github.com/CM2S/GMMD/actions/workflows/test.yaml)
[![Coverage](https://codecov.io/gh/CM2S/GMMD/graph/badge.svg)](https://codecov.io/gh/CM2S/GMMD)
[![Documentation](https://img.shields.io/readthedocs/gmmd)](https://gmmd.readthedocs.io)
[![License](https://img.shields.io/github/license/CM2S/GMMD)](https://github.com/CM2S/GMMD/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24077934.svg)](https://doi.org/10.5281/zenodo.24077934)

Geometric Methods for Microstructure Design.

GMMD is a numerical tool developed in the context of computational mechanics to aid the design and development of advanced materials. Employing a time-driven molecular dynamics simulation or a random sequential adsorption, it generates microstructures of matrix-composite materials in a computationally **efficient** and **robust** way, meshes them, and analyses them.

- [Overview](#overview)
- [Main features](#main-features)
- [Installation](#installation)
- [Usage](#usage)
- [Output](#output)

## Overview

### Description

GMMD has been designed with the main purpose of generating microstructures of matrix-composite materials in a computationally efficient and robust way, an important task in the development of new materials with innovative and enhanced properties. This is achieved using a time-driven **molecular dynamics simulation**, where the forces are repulsive and proportional to the overlap length of the particles. The particles can also be placed by a **random sequential adsorption**, one at a time at random positions, each kept only when it intersects no particle placed before it.

Although nothing prevents the use of GMMD as a standalone program to produce microstructures for the analysis of a given material's behavior using multi-scale analysis, it is in applications such as the more recent data-driven material design frameworks, requiring large material response databases to train the underlying machine learning models, that its reasonable efficiency stands out.

### Computational framework

GMMD is designed and implemented in Python (Python 3 release), making it easily portable between all major computer platforms, easily integrated with other software implemented in different programming languages, and benefiting from an extensive collection of prebuilt (standard library) and third-party libraries. Given the extensive numerical nature of the program, its implementation relies heavily on the well-known [NumPy](https://numpy.org/devdocs/index.html) and [SciPy](https://www.scipy.org/) scientific computing packages, being most numerical tasks dispatched to compiled C code inside the Python interpreter.

### Authors

José Luís P. Vila-Chã ([jvc@fe.up.pt](mailto:jvc@fe.up.pt)), CM2S research group, Department of Mechanical Engineering, Faculty of Engineering, University of Porto.

### Contributors

Bernardo P. Ferreira, Francisco M. Andrade Pires, Rui P. Cardoso Coelho, João A. G. Cruz de Morais.

### Acknowledgements

António M. Couto Carneiro, whose gmsh2links package gives the order in which LINKS numbers the nodes of each element, which the LINKS format of GMMD follows.

### Use of AI

The first version of GMMD, written in 2020 and 2021, was made without AI tools. Since 2026, AI coding assistants are used in developing GMMD: Claude Code, with Claude Opus and Sonnet models, and GitHub Copilot, for code completion in the editor and for Copilot Autofix. They are used to write and refactor code, write tests, review code and draft documentation, and to copy-edit the JOSS paper. The person committing reviews every change and is responsible for it. A commit made with an assistant says so in an `Assisted-by:` trailer naming the tool and model (`Co-authored-by:` in earlier commits). Code completion in the editor is not marked.

### Contributing and support

Bug reports, questions and pull requests are welcome on GitHub. See [CONTRIBUTING.md](https://github.com/CM2S/GMMD/blob/master/CONTRIBUTING.md).

### How to cite

If you use GMMD in your work, please cite the software and the papers on its method. [CITATION.cff](https://github.com/CM2S/GMMD/blob/master/CITATION.cff) gives the citation of the software, and GitHub formats it under "Cite this repository". The software's DOI, [10.5281/zenodo.24077934](https://doi.org/10.5281/zenodo.24077934), always resolves to the latest version. The molecular dynamics method is described in:

- J. L. P. Vila-Chã, B. P. Ferreira, F. M. Andrade Pires, "An adaptive multi-temperature isokinetic method for the RVE generation of particle reinforced heterogeneous materials, Part I: Theoretical formulation and computational framework", *Mechanics of Materials* 163 (2021) 104069. https://doi.org/10.1016/j.mechmat.2021.104069
- B. P. Ferreira, J. L. P. Vila-Chã, F. M. Andrade Pires, "An adaptive multi-temperature isokinetic method for the RVE generation of particle reinforced heterogeneous materials, Part II: Numerical assessment and statistical analysis", *Mechanics of Materials* 165 (2022) 104068. https://doi.org/10.1016/j.mechmat.2021.104068

## Main features

### Phase descriptors

- Diverse particle shapes available:
  - in two dimensions: disks, ellipses and squares;
  - in three dimensions: spheres, ellipsoids, cylinders, and long cylindrical fibers.
- Flexible modeling of geometrical descriptors for particles within a given phase:
  - fixed value for all particles in a phase;
  - distributed according to a statistical distribution (uniform, normal, discrete, ...).

### Generation method

- Time-driven molecular dynamics simulation with repulsive forces proportional to the intersection length of the particles.
- Intersection length computed for general particles with convex shape using the GJK algorithm.
- Force computation sped up by a cell list, a Verlet list or a Verlet list computed from a cell list, as chosen in the input file.
- Integration of the equations of motion using the Verlet integration scheme.
- Isokinetic thermostats: at a fixed temperature, or at temperature stages lowered until a legal configuration is found.
- Physically-based temperature lowering criterion capable of detecting equilibrium.
- Adaptive time step preventing instability of the integration method.
- Starting configuration for the simulation found through a Poisson point process or placed on a regular grid.
- Random sequential adsorption as a second method, with a cell list or a test of every pair to find the particles a new one could intersect.
- Reproducible runs: a fixed seed gives the same microstructure on every run, and a different one for each sample of the set.
- Lightweight output option for data-driven frameworks.
- Each microstructure written as a human-readable, tool-agnostic JSON file listing the particles, one to a line, from which meshing and analysis start.

### Meshing

- Regular mesh with the desired number of voxels in each spatial direction.
- Finite element mesh conforming to the particle boundaries, using Gmsh: first and second order triangles, quadrilaterals and tetrahedra.
- Meshes written as VTK files, and translated into the input files of LINKS, CRATE and Abaqus and into XDMF for FEniCS, each checked by running the solver on what is written; the Gmsh format and legacy VTK are written as well, for other meshers and viewers.

### Analysis

- Visualization of the microstructure and its meshes in ParaView (material phases, ...).
- Statistical descriptors of the microstructure: 2-point correlation function, Ripley's K function, nearest neighbour distances.
- Voronoi metrics based on the Minkowski structure metrics and the Minkowski irreducible tensors.
- Analysis of the generation run: kinetic energy, overlap and time step histories, and the paths of the particles; for a random sequential adsorption, the volume fraction reached at every step.
- A GIF of a two-dimensional generation, one frame per step.

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

### Updating GMMD

An editable install picks up changes to the source code, but not changes to `pyproject.toml`: a new command or a new dependency comes only with installing again. After pulling, install again from the cloned directory, with the extras you use:

```bash
git pull
pip install -e '.[gmsh]'
```

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

The documentation is at https://gmmd.readthedocs.io. To build it locally, install Sphinx with the `docs` extra:

```bash
pip install -e '.[docs]'
python -m sphinx -b html docs docs/_build/html
```

## Usage

GMMD generates a set of microstructures from an **input file**, and post-processes each one as the file asks. The post-processing -- **meshing** and **analysis** -- can also be run on its own, on any microstructure file, by a command of its own: this is how a microstructure generated earlier is meshed again or analysed further.

| What | Command |
|---|---|
| Generate a set of microstructures | `geommicgen input_file.yaml` |
| Mesh a microstructure | `geommicgen-mesh mic.json ...` |
| Translate a mesh for a solver | `geommicgen-translate mesh.vtu --to ...` |
| Analyse a microstructure | `geommicgen-analyze mic.json ...` |
| Convert a `.mic` file of an earlier version | `geommicgen-convert-mic mic.mic` |
| Convert an old input data file | `geommicgen-convert-mdsim input_data_file.mdsim` |

Each command lists its options with `--help`.

### The input file

The input file is a YAML file holding everything a run needs: the phases of the microstructure and their descriptors, the parameters of the generation, and the meshes and analyses to make of each sample. A three dimensional microstructure of ellipsoids, meshed on a grid:

```yaml
samples: 5
rve_dimensions: [1, 1, 1]

phases:
  0: {type: matrix}
  1:
    type: ellipsoid
    vf: 0.1
    axis_1: {distribution: normal, mean: 0.125, sigma: 0.025}
    ratio_12: {distribution: normal, mean: 1.7, sigma: 0.1}
    ratio_13: {distribution: normal, mean: 1.7, sigma: 0.1}
    rot_axis_comp_x: 0
    rot_axis_comp_y: {distribution: uniform, low: -1, high: 1}
    rot_axis_comp_z: {distribution: uniform, low: -1, high: 1}
    angle: {distribution: normal, mean: 0.1, sigma: 0.2}

generation:
  min_distance: 0.002
  molecular_dynamics:
    max_residue_per_particle: 0
    max_step: 20000
    speed_up: {name: verlet, factor: 1.1}

meshes:
  voxel: {n_voxels_dims: [70, 70, 70]}

analyses:
  final_config: true
```

Each phase is keyed by a whole number, which the meshes label it with, and gives the type of its particles and a descriptor for each parameter of one of the descriptions the type accepts: a number for a value every particle shares, or a distribution each particle's value is drawn from, written as the microstructure file writes one. The generation gives the options every method takes, and the options of the method the microstructure is generated by under its name, `molecular_dynamics` or `random_sequential_adsorption`. A mesh is asked for under the name of the mesher that produces it, `gmsh` or `voxel`, the same names `geommicgen-mesh` takes with `--mesher`, and an analysis under its name, with its options, `true` standing for it at its defaults: `final_config`, `motion`, `voronoi`, `statistics`, `rsa_history` and `simulation_gif`, the flags `geommicgen-analyze` asks for them with.

The file is validated whole before anything is generated or written. A key the file does not take is refused, as are values a run would stop on, and every problem is named at once by its path in the file:

```
geommicgen: input_file.yaml has 2 problems:
  - generation.molecular_dynamics.max_step: is missing
  - phases.1.axis_1.sigma: Input should be greater than or equal to 0 (given -0.025)
```

[examples/reference.yaml](https://github.com/CM2S/GMMD/blob/master/examples/reference.yaml) documents every key -- what it means, the values it takes and its default -- and runs as it is written. Two worked examples, a two dimensional microstructure of ellipses and a three dimensional one of ellipsoids, are beside it in [examples/](https://github.com/CM2S/GMMD/tree/master/examples); the test suite runs both.

An old input data file (`.mdsim`) is still run, with a line saying that the format is on its way out. `geommicgen-convert-mdsim` writes the equivalent input file beside it, and says what the old file gave that nothing read:

```bash
geommicgen-convert-mdsim input_data_file.mdsim
```

### Generating a set of microstructures

To generate a new set of microstructures, provide the input file as the only argument to the `geommicgen` command:

```bash
geommicgen input_file.yaml
```

Each sample is generated, written to a folder of its own, and then meshed and analysed as the input file asks. The program execution can be followed in the terminal, where data associated with the program launch, progress of the main execution phases, and the program end is output.

### Meshing a microstructure

The meshing is two stages -- a mesh of the microstructure, and that mesh in the formats solvers read -- and each has a command of its own. `geommicgen-mesh` discretises a microstructure file and writes the mesh; `geommicgen-translate` writes a mesh in the formats solvers read, taking the file the previous stage wrote, or one another tool produced:

```bash
geommicgen-mesh mic.json --mesher gmsh --mesh-size 0.05 --element-type tri6 --to links
geommicgen-mesh mic.json --elements-per-particle 4,8 --element-type tri3,tri6
geommicgen-mesh input_file/mic_*/mic.json --elements-per-particle 4 --to links
geommicgen-mesh mic.json --mesher voxel --n-voxels-dims 100 100 --to crate
geommicgen-translate mic.vtu --to links,xdmf
geommicgen-translate --list-formats
```

A mesh is written as a VTK file whichever mesher produced it, `.vtu` for an unstructured mesh and `.vti` for a grid, with a small `.mesh.json` beside it holding what those formats cannot carry. The files are named after the microstructure file and what discretised it -- `mic_tri6_h0.05.vtu`, `mic_tri6_epp4.vtu`, `mic_100_100.vti` -- so that meshing one microstructure several ways into one directory keeps them apart; `--name` names them outright. The size, the elements per particle and the element each take several values, separated by commas, and there is a mesh for every combination: the second line above writes four. Several microstructures can be given at once, as the third line meshes every sample of a run; each is meshed into `meshes/` beside it unless `-o` says otherwise. Giving no `--to` stops after that file, which is a complete stage: `geommicgen-translate` picks up from it later, or somewhere else.

The LINKS and Abaqus input files, and the example CRATE input file written beside a grid, carry placeholder materials and a placeholder loading, a stretch along x, marked as such: a microstructure says nothing about either, but with them the files run as written, which is how they are checked. The CRATE file also declares the dimensions of the RVE, which the grid does not carry. FEniCS reads the `.xdmf` file, with the phase of every cell as a cell tag; [`examples/fenics_elasticity.py`](https://github.com/CM2S/GMMD/blob/master/examples/fenics_elasticity.py) reads one, gives each phase a material and solves.

The gmsh mesher offers `tri3`, `tri6`, `quad4` and `quad8` in two dimensions and `tetra4` and `tetra10` in three. A quadrilateral mesh is made by recombining triangles, and Gmsh keeps a triangle where it finds no pair for it, so a `quad4` or `quad8` mesh may hold a few triangles; when it does, the mesher says so with the counts, and the mesh is written with both, each type in a group of its own.

### Analysing a microstructure

`geommicgen-analyze` runs the analyses from a microstructure file and, for the motion of the particles, from the `md_state.npz` a generation run writes beside it:

```bash
geommicgen-analyze mic_0/mic.json --motion --statistics --statistics-descriptors two_pt_corr -o mic_0/analysis
geommicgen-analyze mic.json --voronoi --voronoi-type set --voronoi-plot
```

Every analysis the input file can ask for is a flag of its name, and each of its options a flag named after the analysis and the option, `--voronoi-type` for the `type` of `voronoi`. An earlier analysis in the output directory is written over.

### Microstructures of an earlier version

A microstructure used to be stored as a `.mic` file, a pickle of the objects that held it. It is a JSON file now, one particle to a line, which any tool can read and a person can edit. A `mic.yaml`, which development versions wrote before, is still read. `geommicgen-convert-mic` turns the one into the other, writing the state of the generation run beside it:

```bash
geommicgen-convert-mic mic_0/mic.mic
```

## Output

Running GMMD on an input file, say `input_file.yaml`, creates a folder of the same name beside it holding a copy of the file and a folder `mic_*` for each microstructure generated. Each microstructure file records the git commit of GMMD the run used, whether the package had uncommitted changes, and the versions of numpy, scipy and gmsh: with the input file, that is the whole of the run, its defaults included. A run never writes over an earlier one: a second run of the same file goes into `input_file_1/`, a third into `input_file_2/`.

```
input_file/
├── input_file.yaml                copy of the input file
├── mic_0/
│   ├── mic.json                   microstructure file
│   ├── status                     status of the generation: flag, time, final overlap
│   ├── md_state.npz               histories of an MD run, read by the motion analysis            (not with save_min)
│   ├── mic.screen                 log of the run, everything printed to the terminal             (not with save_min)
│   ├── final_config.pdf           visualization of the microstructure, when final_config is asked for
│   │                              (final_config.msh and final_config.vtk in three dimensions)
│   ├── RSA_vf_history.pdf         volume fraction at every step, when rsa_history is asked for
│   ├── Simulation.gif             the generation, one frame per step, when simulation_gif is asked for
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

The microstructure file, `mic.json`, is a human-readable, tool-agnostic description of the microstructure: the dimensions of the RVE, the phases, and the list of the particles, one to a line, each with only what defines its geometry. Any program that reads JSON can use it, and `geommicgen-mesh` and `geommicgen-analyze` start from it.

`save_min`, the lightweight mode geared towards data-driven frameworks, keeps only the microstructure and status files.

`geommicgen-mesh` writes the meshes of a microstructure into `meshes/` beside it, as a run does, and `geommicgen-translate` and `geommicgen-analyze` write into the current directory; each writes where its `-o` option says instead.
