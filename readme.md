

<p align="center">
  <a href=""><img alt="logo" src="doc/media/CRATE_logo_horizontal_long.png" width="80%"></a>
</p>

# Overview

### Summary
(???) is a numerical tool developed in the context of computational mechanics to aid the design and development of advanced materials.
Employing a time-driven molecular dynamics simulation, (???) offers a solution to generate microstructures of matrix-composite materials in a computationally **efficient** and **robust** way. 


### Authors
This program initial version was documented and fully coded by José Luís P. Vila-Chã<sup>[1](#f1) </sup> ([up201506192@fe.up.pt](mailto:up201506192@fe.up.pt)) and developed in colaboration with Bernardo P. Ferreira<sup>[1](#f1) </sup> ([bpferreira@fe.up.pt](mailto:bpferreira@fe.up.pt)) and Francisco M. Andrade Pires<sup>[2](#2) </sup> ([fpires@fe.up.pt](mailto:fpires@fe.up.pt)).

<sup id="f1"> 1 </sup> Member of CM2S research group, Department of Mechanical Engineering, Faculty of Engineering, University of Porto  
<sup id="f2"> 2 </sup> Leader of CM2S research group, Department of Mechanical Engineering, Faculty of Engineering, University of Porto

### Description
(???) has been designed with the main purpose of generating microstructures of matrix-composite materials in a computational efficient and robust way, an important task in the development of new materials with innovative and enhanced properties.
This is achieved using a time-driven **molecular dynamics simulation**, where the forces are repulsive and proportional to the overlap area/volume of the particles.

Although nothing prevents the use of (???) as a standalone program to produce microstructures for the analysis a given material's behavior using multi-scale analysis, it is in applications such as the more recent data-driven material design frameworks, requiring large material response databases to train the underlying machine learning models that its reasonable efficiency stands out.

### Computational framework
(???) is designed and implemented in Python (Python 3 release), making it easily portable between all major computer platforms, easily integrated with
other softwares implemented in different programming languages and benefiting from an extensive collection of prebuilt (standard library) and third-party libraries. Given the extensive numerical nature of the program, its implementation relies heavily on the well-known [NumPy](https://numpy.org/devdocs/index.html) and [SciPy](https://www.scipy.org/) scientific computing packages, being most numerical tasks dispatched to compiled C code inside the Python interpreter.


# Main features

### Phase Descriptors:
* Quasi-static loading conditions;
* Monotonic loading paths;
* Infinitesimal strains;
* Nonlinear material constitutive behavior (elasticity and plasticity).

### Methods:
* Time-driven molecular dynamics simulation with repulsive forces proportional to the overlap area/volume of the particles. Force computation using a Verlet lists computed from Cell lists and integration of the equations of motion using the Verlet integration scheme. The thermostat used is a multi-temperature isokinetic scheme that lowers the temperature until a legal configuration is found.

### Post-processing:
* Mesh output files.
  - Regular mesh with the desired number of voxels in each spatial direction.
  - Non conform finite element mesh. 
* VTK (XML format)(?) output files allowing the visualization of data associated to the material microstructure (material phases, material clusters, ...) and response local fields (strain, stress, internal variables, ...);
* Voronoi metrics based on the Minkowski Structure Metrics and the Minkowski Irreducible Tensors.

***
<sup id="f5"> 5 </sup> LINKS (Large Strain Implicit Non-linear Analysis of Solids Linking Scales) is a multi-scale finite element code developed by CM2S research group at Faculty of Engineering of University of Porto.  
<sup id="f6"> 6 </sup> Liu, Z., Bessa, M., and Liu, W. K. (2016a). Self-consistent clustering analysis: An efficient multi-
scale scheme for inelastic heterogeneous materials. Computer Methods in Applied Mechanics
and Engineering, 306:319–341.  
<sup id="f7"> 7 </sup> Moulinec, H. and Suquet, P. (1994). A fast numerical method for computing the linear and
nonlinear mechanical properties of composites. A fast numerical method for computing the
linear and nonlinear mechanical properties of composites, 318(11):1417–1423.

# Quick guide

### Requirements
Some software must be installed in order to successfully run (???):
* Python 3.X (see [here](https://www.python.org/downloads/)) - Required to compile (byte code) and run (Python Virtual Machine) (???);

  > In Linux/UNIX operative systems, python can be simply installed from apt library by executing the following command:  
  `sudo apt install python3.X`  

* PyPi pip (see [here](https://pypi.org/project/pip/)) - Required to install Python 3 packages (learn [here](https://docs.python.org/3/installing/));

  > In Linux/UNIX operative systems, pip can be simply installed from apt library by executing the following command:  
  `sudo apt install python3-pip`
  
* ParaView (see [here](https://www.paraview.org/download/)) - Required to visualize the data contained in the VTK output files (learn [here](https://www.paraview.org/resources/));  

  > In Linux/UNIX operative systems, ParaView can be installed by placing the tarball in the installation directory and extracting it by executing the following command:  
  `sudo tar -xvf ParaView-< version >.tar.gz`

* Gmsh e programa do António

> **Note:** When trying to run (???) for the first couple times, it is expected that Python's ImportError and ModuleNotFoundError are raised depending on the required packages that are not installed. Install them in turn and rerun (???) until these exceptions are no longer raised, meaning that all required packages are properly installed and accessed.

### ((???)) workflow
In what follows, the general workflow of (???) in the generation of a set of samples with a given set of microstructural descriptors:

1. **Write input data file.** This file contains all the required information to generate the samples of a microstructure, including its descriptors and parameters of the generation process.
A complete (???) input data file where each parameter specification (either mandatory or optional) is fully documented (meaning, syntax, available options) can be found in the `doc/` directory (or [here](https://github.com/BernardoFerreira/(???)/blob/master/doc/CRATE_input_data_file.dat)). This file can be copied to a given directory and be readily used by replacing the `[insert here]` boxes with the suitable specification.

2. **Run (???).** In order to run (???), one must simply execute the main file (`(???).py`) with Python 3.X and provide the input data file (argument parsing).
    > In Linux/UNIX operative systems, open a terminal console window and execute the following command:  
    `python3.X (???).py input_data_file.dat`
     <br/><br/>
    The program execution can be followed in the terminal console window, where the data associated to the program launch, to the progress of the main execution phases and to the program end is output. 
  
3. **Get results.** As soon as (???) is executed according to an input data file (lets say, `input_data_file.dat`), a folder with the same name is created in the same directory (`input_data_file/`). This folder contains all the output data related to the microstructure generation, namely: (? - to be decided) a log file (`input_data_file.screen`), where all data printed to the default standard ouput is stored; a homogenized results file (`input_data_file.hres`), where the homogenized results are stored; and one or more VTK output files (`.vti`) that can be read with a suitable software (e.g. [ParaView](https://www.paraview.org/)) to visualize and analyse the problem data.
