

<p align="center">
  <a href=""><img alt="logo" src="doc/media/CRATE_logo_horizontal_long.png" width="80%"></a>
</p>

# Overview

### Summary
GMMD is a numerical tool developed in the context of computational mechanics to aid the design and development of advanced materials.
Employing a time-driven molecular dynamics simulation, GMMD offers a solution to generate microstructures of matrix-composite materials in a computationally **efficient** and **robust** way.


### Authors
This program initial version was documented and fully coded by José Luís P. Vila-Chã<sup>[1](#f1) </sup> ([up201506192@fe.up.pt](mailto:jvc@fe.up.pt)) and developed in colaboration with Bernardo P. Ferreira<sup>[1](#f1) </sup> ([bpferreira@fe.up.pt](mailto:bpferreira@fe.up.pt)) and Francisco M. Andrade Pires<sup>[2](#2) </sup> ([fpires@fe.up.pt](mailto:fpires@fe.up.pt)).

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
* Force computation using a Verlet lists computed from Cell lists
* Integration of the equations of motion using the Verlet integration scheme.
* Thermostat used is a multi-temperature isokinetic scheme. The temperature is lowered until a legal configuration is found.
* Physically-based temperature lowering criterion capable of detecting equilibrium.
* Adaptive time step preventing instability of the integration method.
* Starting configuration for the simulation found through a Poisson Point Process;

### Data-driven framework
* Option with lightweight output for data-driven-based frameworks.

### Post-processing:
* Mesh output files.
  - Regular mesh with the desired number of voxels in each spatial direction.
  - Non conform finite element mesh using Gmsh.
* VTK  output files allowing the visualization of data associated with the material microstructure (material phases, ...);
* Statistical analysis of the microstructure
  - Statistical descriptors (2-point correlation function, Ripley's K function, ...)
  - Voronoi metrics based on the Minkowski Structure Metrics and the Minkowski Irreducible Tensors.

# Quick guide

### Requirements
Some software must be installed to successfully run GMMD:
* Python 3.X (see [here](https://www.python.org/downloads/)) - Required to compile (byte code) and run (Python Virtual Machine) GMMD;

  > In Linux/UNIX operative systems, python can be simply installed from apt library by executing the following command:  
  `sudo apt install python3.X`  

* PyPi pip (see [here](https://pypi.org/project/pip/)) - Required to install Python 3 packages (learn [here](https://docs.python.org/3/installing/));

  > In Linux/UNIX operative systems, pip can be simply installed from apt library by executing the following command:  
  `sudo apt install python3-pip`

* ParaView (see [here](https://www.paraview.org/download/)) - Required to visualize the data contained in the VTK output files (learn [here](https://www.paraview.org/resources/));  

  > In Linux/UNIX operative systems, ParaView can be installed by placing the tarball in the installation directory and extracting it by executing the following command:  
  `sudo tar -xvf ParaView-< version >.tar.gz`

* Gmsh e gmsh2links (see [here](https://github.com/CM2S/Utilities/tree/master/gmsh)) - Required to produce meshes of the microstructures to be used in FEM analysis through LINKS.

> **Note:** When trying to run GMMD for the first couple of times, it is expected that Python's ImportError and ModuleNotFoundError are raised depending on the required packages that are not installed. Install them in turn and rerun GMMD until these exceptions are no longer raised, meaning that all required packages are properly installed and accessed.

### GMMD workflow
In what follows, the general workflow of GMMD in the generation of a set of samples with a given set of microstructural descriptors:

1. **Write input data file.** This file contains all the required information to generate the samples of a microstructure, including its descriptors and parameters of the generation process.
A complete GMMD input data file where each parameter specification (either mandatory or optional) is fully documented (meaning, syntax, available options) can be found in the `geommicgen/resources` directory (or [here](https://github.com/josevilacha/GMMD/blob/master/geommicgen/resources/MIC_input_data_file.dat)). This file can be copied to a given directory and be readily used by replacing the `[insert here]` boxes with the suitable specification.

2. **Run GMMD.**

  2.1. *New set of microstructures:* To generate a new microstructure using GMMD, one must simply execute the module (`geommicgen`) with Python 3.X and provide the input data file (argument parsing).
    > In Linux/UNIX operative systems, open a terminal console window and execute the following command:  
    `python3.X geommicgen input_data_file.mdsim`
     <br/><br/>
    The program execution can be followed in the terminal console window, where the data associated with the program launch, to the progress of the main execution phases, and the program end is output.

  2.2. *Meshing/analysis of microstructures:* To generate a new mesh or perform statistical analysis on a previously generated microstructure using GMMD, one must simply execute the module (`geommicgen`) with Python 3.X, provide the input data file (`.mdsim`) and the microstructure file (`.mic`), in this order.
    > In Linux/UNIX operative systems, open a terminal console window and execute the following command:  
    `python3.X geommicgen input_data_file.mdsim previous_mic.mic`
     <br/><br/>
    The program execution can be followed in the terminal console window, where the data associated with the program launch, to the progress of the main execution phases, and the program end is output.

3. **Get results.** As soon as GMMD is executed according to an input data file (let us say, `input_data_file.mdsim`), a folder with the same name is created in the same directory (`input_data_file/`). This folder contains all the output data related to the microstructure generation, namely:
  * a folder `mic_*` for each microstructure generated.
    - microstructure file (`mic.mic`)<sup>[*](#f6)[+](#f5)</sup>;
    - status file (`status`), containing a flag for the status of the generation, time and final overlap<sup>[*](#f6)[+](#f5)</sup>;
    - log file (`input_data_file.screen`), where all data printed to the default standard output is stored<sup>[+](#f5)</sup>;
    - visualization file for the microstructure (`final_config.vtk`);
    - folder containing the specified meshes (`meshes`);
    - folder containing the motion analysis (`motion_results`), such as the plot of the kinetic energy, total overlap, ...;
    - folder containing the statistical analysis (`stat_analysis_results`), such as the 2-point correlation function, Ripley's K function, as specified in the input file;
    - folder containing the statistical analysis (`voronoi_analysis_results`), such as the Voronoi diagrams, Voronoi diagrams with IMTs, and corresponding histograms, as specified in the input file
  * a copy of the input file `.mdsim`


> <sup id="f6"> * </sup> Files generated in the mode geared towards data-driven frameworks.
 <sup id="f5"> + </sup> Files always generated as output.
