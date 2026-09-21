# GMMD

### Summary
GMMD is a numerical tool developed in the context of computational mechanics to aid the design and development of advanced materials.
Employing a time-driven molecular dynamics simulation, GMMD offers a solution to generate microstructures of matrix-composite materials in a computationally **efficient** and **robust** way.


### Authors
This program initial version was documented and fully coded by José Luís P. Vila-Chã<sup>[1](#f1) </sup> ([jvc@fe.up.pt](mailto:jvc@fe.up.pt)) and developed in colaboration with Bernardo P. Ferreira<sup>[1](#f1) </sup> ([bpferreira@fe.up.pt](mailto:bpferreira@fe.up.pt)) and Francisco M. Andrade Pires<sup>[2](#2) </sup> ([fpires@fe.up.pt](mailto:fpires@fe.up.pt)).

<sup id="f1"> 1 </sup> Member of CM2S research group, Department of Mechanical Engineering, Faculty of Engineering, University of Porto  
<sup id="f2"> 2 </sup> Leader of CM2S research group, Department of Mechanical Engineering, Faculty of Engineering, University of Porto

### Description
GMMD has been designed with the main purpose of generating microstructures of matrix-composite materials in a computationally efficient and robust way, an important task in the development of new materials with innovative and enhanced properties.
This is achieved using a time-driven **molecular dynamics simulation**, where the forces are repulsive and proportional to the overlap length of the particles.

Although nothing prevents the use of GMMD as a standalone program to produce microstructures for the analysis of a given material's behavior using multi-scale analysis, it is in applications such as the more recent data-driven material design frameworks, requiring large material response databases to train the underlying machine learning models that its reasonable efficiency stands out.

### Computational framework
GMMD is designed and implemented in Python (Python 3 release), making it easily portable between all major computer platforms, easily integrated with
other software implemented in different programming languages, and benefiting from an extensive collection of prebuilt (standard library) and third-party libraries. Given the extensive numerical nature of the program, its implementation relies heavily on the well-known [NumPy](https://numpy.org/devdocs/index.html) and [SciPy](https://www.scipy.org/) scientific computing packages, being most numerical tasks dispatched to compiled C code inside the Python interpreter.


# Main features

### Phase Descriptors:
* Diverse particle shapes available:
  - In two dimensions: disks and ellipses;
  - In three dimensions: spheres, ellipsoids, cylinders, and long cylindrical fibers.
* Flexible modeling of geometrical descriptors for particles within a given phase.
  - Fixed value for all particles in a phase.
  - Distributed according to a statistical distribution (Uniform, Normal, Discrete, ...)

### Methods:
* Time-driven molecular dynamics simulation with repulsive forces proportional to the intersection length of the particles.
* Intersection length computed for general particles with convex shape using the GJK algorithm.
* Force computation sped up by a cell list, a Verlet list or a Verlet list computed from a cell list, as chosen in the input data file.
* Integration of the equations of motion using the Verlet integration scheme.
* Isokinetic thermostats: at a fixed temperature, or at temperature stages lowered until a legal configuration is found.
* Physically-based temperature lowering criterion capable of detecting equilibrium.
* Adaptive time step preventing instability of the integration method.
* Starting configuration for the simulation found through a Poisson Point Process or placed on a regular grid.
* Reproducible runs: a fixed seed gives the same microstructure on every run, and a different one for each sample of the set.

### Data-driven framework
* Option with lightweight output for data-driven-based frameworks.

### Post-processing:
* Mesh output files.
  - Regular mesh with the desired number of voxels in each spatial direction.
  - Non conform finite element mesh using Gmsh.
  - Written as VTK files, and translated into the formats solvers read (LINKS, CRATE, XDMF, MED, Exodus, ...).
* VTK output files allowing the visualization of the microstructure and its meshes in ParaView (material phases, ...);
* Statistical analysis of the microstructure
  - Statistical descriptors (2-point correlation function, Ripley's K function, nearest neighbour distances)
  - Voronoi metrics based on the Minkowski Structure Metrics and the Minkowski Irreducible Tensors.
* Analysis of the generation run: kinetic energy, overlap and time step histories, and the paths of the particles.

# Quick guide

### Requirements
Some software must be installed to successfully run GMMD:
* Python **3.10 or newer** (see [here](https://www.python.org/downloads/)) - Required to compile (byte code) and run (Python Virtual Machine) GMMD. GMMD is tested on Python 3.10 to 3.13, on Linux and Windows;

  > In Linux/UNIX operative systems, python can be simply installed from apt library by executing the following command:  
  `sudo apt install python3`  

* PyPi pip (see [here](https://pypi.org/project/pip/)) - Required to install Python 3 packages (learn [here](https://docs.python.org/3/installing/));

  > In Linux/UNIX operative systems, pip can be simply installed from apt library by executing the following command:  
  `sudo apt install python3-pip`

* ParaView (see [here](https://www.paraview.org/download/)) - Required only to visualize the microstructures and meshes GMMD writes as VTK files (`.vti`, `.vtu`, `.vtk`), which ParaView opens directly (learn [here](https://www.paraview.org/resources/));  

  > In Linux/UNIX operative systems, ParaView can be installed by placing the tarball in the installation directory and extracting it by executing the following command:  
  `sudo tar -xvf ParaView-< version >.tar.gz`

* Gmsh - Required to produce finite element meshes of the microstructures and the three dimensional visualizations. It is an optional dependency, so a run that only asks for a regular grid needs none of it. Install it with the extra:
  ```bash
  pip install 'geommicgen[gmsh]'
  ```
  `GMMD` requires **Gmsh 4.15 or newer**, the version it is tested against. If you instead
  install the SDK tarball by hand from [gmsh.info](https://gmsh.info/bin/Linux/), add its
  Python API to your `PYTHONPATH`:
  ```bash
  export PYTHONPATH=$PYTHONPATH:/path/to/gmsh/lib
  ```
* h5py and netCDF4 - Required only by the mesh formats that store their arrays in HDF5 or
  NetCDF: `xdmf` (read by FEniCS), `med` (Code_Aster) and `exodus` (MOOSE). Each is an
  extra named after the format, and `formats` installs all of them:
  ```bash
  pip install 'geommicgen[xdmf]'      # or [med], [exodus], [formats]
  ```
  Asking for one of these formats without its library fails that format alone, with the
  command that installs it; the mesh itself and the other formats are still written.

### Installation
`GMMD` can be installed by first cloning this repository:
```bash
git clone https://github.com/CM2S/GMMD.git
```
Then, change directory into the cloned repository (where the `pyproject.toml` is located) and install the package. The `pyproject.toml` file defines the package metadata, dependencies, and the command-line entry points. Running `pip install` will automatically install all required dependencies and register the `geommicgen` commands on your system.

For a **regular install**:
```bash
pip install .
```

For an **editable (development) install**, where changes to the source code are reflected immediately without reinstalling:
```bash
pip install -e .
```

After installation, the `geommicgen`, `geommicgen-mesh`, `geommicgen-translate`, `geommicgen-analyze` and `geommicgen-convert-mic` commands become available and can be called from any directory.

### Testing
The test suite runs with pytest, installed by the `test` extra:
```bash
pip install -e '.[test]'
pytest geommicgen/tests
```
The tests that need Gmsh are skipped when it is not installed. The documentation is built with Sphinx, installed by the `docs` extra:
```bash
pip install -e '.[docs]'
python -m sphinx -b html docs docs/_build/html
```

### GMMD workflow
GMMD is used in two main ways: from an **input data file**, which generates a new set of microstructures and post-processes each one as the file asks; or from an **existing microstructure file**, which post-processes a microstructure generated earlier. The post-processing is the same in both -- meshing and analysis -- and each of the two can also be run on its own by a command of its own.

1. **Write input data file.** This file contains all the required information to generate the samples of a microstructure, including its descriptors and parameters of the generation process, and the meshes and analyses to produce of each sample.
A complete GMMD input data file where each parameter specification (either mandatory or optional) is fully documented (meaning, syntax, available options) can be found in the `geommicgen/resources` directory (or [here](https://github.com/CM2S/GMMD/blob/master/geommicgen/resources/MIC_input_data_file.dat)). This file can be copied to a given directory and be readily used by replacing the `[insert here]` boxes with the suitable specification.

2. **Run GMMD.**

  2.1. *From an input data file -- new set of microstructures:* To generate a new set of microstructures, provide the input data file as the only argument to the `geommicgen` command:
    ```bash
    geommicgen input_data_file.mdsim
    ```
    Each sample is generated, written to a folder of its own, and then meshed and analysed as the input data file asks. The program execution can be followed in the terminal, where data associated with the program launch, progress of the main execution phases, and the program end is output.

  2.2. *From an existing microstructure -- meshing/analysis:* To generate new meshes or perform an analysis of a previously generated microstructure, provide both the input data file (`.mdsim`) and the microstructure file (`.yaml`), in this order:
    ```bash
    geommicgen input_data_file.mdsim previous_mic.yaml
    ```
    The input data file needs only the meshing and post-processing options; any generation parameters in it are ignored. The microstructure file may also be a `.csv` of particles exported from ImageJ.

  2.3. *Meshing on its own:* The meshing is two stages -- a mesh of the microstructure, and that mesh in the formats solvers read -- and each has a command of its own. `geommicgen-mesh` discretises a microstructure file and writes the mesh; `geommicgen-translate` writes a mesh in the formats solvers read, taking the file the previous stage wrote, or one another tool produced:
    ```bash
    geommicgen-mesh mic.yaml --mesher gmsh --mesh-size 0.05 --element-type tri6 --to links
    geommicgen-mesh mic.yaml --mesher voxel --n-voxels-dims 100 100 --to crate
    geommicgen-translate mic.vtu --to links,xdmf
    geommicgen-translate --list-formats
    ```
    A mesh is written as a VTK file whichever mesher produced it, `.vtu` for an unstructured mesh and `.vti` for a grid, with a small `.mesh.json` beside it holding what those formats cannot carry. Giving no `--to` stops after that file, which is a complete stage: `geommicgen-translate` picks up from it later, or somewhere else. Each command lists its options with `--help`.

  2.4. *Analysis on its own:* `geommicgen-analyze` runs the analyses from a microstructure file and, for the motion of the particles, from the `md_state.npz` a generation run writes beside it:
    ```bash
    geommicgen-analyze mic_0/mic.yaml --motion-analysis --stat-two-pt-corr -o mic_0/analysis
    geommicgen-analyze mic.yaml --voronoi-analysis --voronoi-type set --plot-voronoi
    ```
    Every analysis the input data file can ask for is a flag of the same name, and an earlier analysis in the output directory is written over.

  2.5. *Microstructures generated before the change of format:* a microstructure used to be stored as a `.mic` file, a pickle of the objects that held it. It is a YAML file now, which any tool can read and a person can edit. `geommicgen-convert-mic` turns the one into the other, writing the state of the generation run beside it:
    ```bash
    geommicgen-convert-mic mic_0/mic.mic
    ```

3. **Get results.** As soon as GMMD is executed according to an input data file (let us say, `input_data_file.mdsim`), a folder with the same name is created in the same directory (`input_data_file/`). This folder contains all the output data related to the microstructure generation, namely:
  * a folder `mic_*` for each microstructure generated.
    - microstructure file (`mic.yaml`)<sup>[+](#f5)</sup>;
    - status file (`status`), containing a flag for the status of the generation, time and final overlap<sup>[+](#f5)</sup>;
    - state of the generation run (`md_state.npz`), holding the histories the motion analysis plots<sup>[*](#f6)</sup>;
    - log file (`mic.screen`), where all data printed to the default standard output is stored<sup>[*](#f6)</sup>;
    - visualization of the microstructure, when `final_config` is asked for: `final_config.pdf` in two dimensions, `final_config.msh` and `final_config.vtk` in three;
    - folder containing the specified meshes (`meshes`);
    - folder containing the motion analysis (`motion_results`), such as the plot of the kinetic energy, total overlap, ...;
    - folder containing the statistical analysis (`stat_analysis_results`), such as the 2-point correlation function, Ripley's K function, as specified in the input file;
    - folder containing the Voronoi analysis (`voronoi_analysis_results`), such as the Voronoi diagrams, Voronoi diagrams with IMTs, and corresponding histograms, as specified in the input file
  * a copy of the input file `.mdsim`

  When run from an existing microstructure (2.2), the meshes and analyses are written into the `input_data_file/` folder itself, there being no sample to write them beside.


> <sup id="f5"> + </sup> Files always generated as output.
 <sup id="f6"> * </sup> Files not generated in the lightweight mode geared towards data-driven frameworks (`save_min`).
