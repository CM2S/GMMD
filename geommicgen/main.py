"""
Microstructure Generation Interface (DATAGEM Program)(????)
==========================================================================================
Summary:
...
------------------------------------------------------------------------------------------
Development history:
Zé Luís P. Vila-Chã | March 2020 | Initial coding.
"""
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
import numpy as np

# Working with arrays
import pickle

# Dumping files in a binary format
import time

# To compute the time taken
from integration_methods import Newmark, VerletSync

# Importing an integration method for the equation of motion
from particle_classes import (
    Disk,
    Particle,
    Ellipse,
    Sphere,
    Ellipsoid,
    CylindricalFiber,
    RVE,
    Phase,
    GeometricalParameter,
    PhaseDescriptor,
)

# Importing the particle class
from meshing_interface import generateMesh, checkMeshSpecs

# Importing meshing interfaces
import error_classes as errors

# Importing the error clases
import printing as print_funcs

from voronoi_analysis import doVoronoiAnalysis

from motion_analysis import doMotionAnalysis

import os
import shutil

import sys


def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def createResultsDirectory(particles, dp_dir, remesh=False):
    """
    Create the results directory.

    Parameters
    ----------
    particles: `.particle`
        Particles in the system.

    dp_dir: string
        Directory where the results are going to be stored.

    remesh: boolean, optional
        Signals if the program is currently being used for a remesh action.
    """
    Particle.file_name = "mic"
    # Defining the file name associated with this sampling. The filenames of the particles
    # are always prefixed by mic
    results_folder = os.path.join(dp_dir, Particle.file_name)
    # Creating a tentative path for the results folder
    results_folder_old = results_folder
    # Saving the original name of the results folder
    i = 0
    # Initializing the filename suffix
    while True:
        results_folder = results_folder_old + "_" + str(i)
        # Creating a new folder name appending an integer to the name of the original
        # folder
        i += 1
        # Increasing the filenam suffix
        if not os.path.exists(results_folder):
            # Repeat while the folder names already exists
            break
    os.makedirs(results_folder)
    # Creating the directory
    if os.path.exists("input_data\\info_micro.p") and not remesh:
        shutil.copy(
            "input_data\\info_micro.p", os.path.join(results_folder, "info_micro.p")
        )
        # copying input file
    Particle.file_path = os.path.join(results_folder, Particle.file_name)
    # Saving the file path in the Particle class


def particleGeneration(
    descriptors,
    phase_types,
    rve_dims,
    problem_type,
    dp_dir,
    type_init_conf,
    save_history=True,
):
    """
    Generate all the particles from the geometrical descriptors.

    Parameters
    ----------
    descriptors: dictionary
        Dictionary containing the particle descriptors

    phase_types: dictionary(str:int)
        Dictionary containing the phase type of each phase.
            1: Matrix
            2: Disk (2D)
            3: Ellipse (2D)
            4: Sphere (3D)
            5: Ellipsoid (3D)

    rve_dims: list(float)
        Length of the RVE sides in each direction.

    problem_type: integer
        Type of problem.
            1: 2D problem (plain strain)
            2: 2D problem (plain stress)
            3: 2D problem (axisymmetric)
            4: 3D problem

    dp_dir: string
        Directory where the microstructure spatial discretization file(s) associated
        with the given design point are to be stored

    type_init_conf: {'random', 'grid'}
        Type of initial configuration for the particle centers.

    save_history: bool, optional
        Save the motion of the particles for later analysis.
    """
    Particle.box = rve_dims
    # Setting the size of the simulation box. It may be changed later if the phases are
    # made from cylindrical fibers, as their simulated in a plane despite being 3D
    Particle.volume_RVE = np.prod(rve_dims)
    # Volume of the RVE
    Particle.phases = {
        i_phase: Phase(i_phase, phase_types[i_phase]) for i_phase in descriptors
    }
    Particle.list_phases = [i_phase for i_phase in descriptors]
    # Dictionary containing the phases
    try:
        if list(phase_types.values()).count(1) == 0:
            # No matrix phase was specified
            raise errors.NoMatrixPhase()
        elif list(phase_types.values()).count(1) > 1:
            # Too many phases were specified as the matrix phase
            raise errors.TooManyMatrixPhases()
    except (errors.NoMatrixPhase, errors.TooManyMatrixPhases) as error:
        error.message()
        quit()
    particles = []
    # Initializing the list containing the particles
    # if problem_type == 1:
    #     # 2D problem (plain strain)
    #     dim = 2
    #     # (FIX)
    #     # Setting the dimension
    for i_phase in Particle.phases.values():
        # Running through all the phases listed in the dictionary
        try:
            if i_phase.type == 1:
                # This phase is the matrix
                Particle.matrix_phase = i_phase.name
                # No particles are generated
            elif i_phase.type == 2:
                # This phase is made up by disks
                if len(rve_dims) != 2:
                    # The RVE must be 2D
                    raise errors.IncompatibleDimensionsRVEphase(
                        "Disks", 2, 3, i_phase.name
                    )
                particles = particles + generateDisks(
                    i_phase, rve_dims, descriptors[i_phase.name]
                )
                # Generating the number of disks requested and appending them to the list of
                # particles
            elif i_phase.type == 3:
                # This phase is made up by ellipses
                if len(rve_dims) != 2:
                    # The RVE must be 2D
                    raise errors.IncompatibleDimensionsRVEphase(
                        "Ellipses", 2, 3, i_phase.name
                    )
                particles = particles + generateEllipses(
                    i_phase, rve_dims, descriptors[i_phase.name]
                )
                # Generating the number of ellipses requested and appending them to the list
                # of particles
            elif i_phase.type == 4:
                # This phase is made up by spheres
                if len(rve_dims) != 3:
                    # The RVE must be 3D
                    raise errors.IncompatibleDimensionsRVEphase(
                        "Spheres", 3, 2, i_phase.name
                    )
                particles = particles + generateSpheres(
                    i_phase, rve_dims, descriptors[i_phase.name]
                )
                # Generating the number of spheres requested and appending them to the list
                # of  particles
            elif i_phase.type == 5:
                # This phase is made up by ellipsoids
                if len(rve_dims) != 3:
                    # The RVE must be 3D
                    raise errors.IncompatibleDimensionsRVEphase(
                        "Ellipsoids", 3, 2, i_phase.name
                    )
                particles = particles + generateEllipsoids(
                    i_phase, rve_dims, descriptors[i_phase.name]
                )
                # Generating the number of ellipsoids requested and appending them to the
                # list of particles
            elif i_phase.type == 6:
                # This phase is made up by cylindrical fibers
                if len(rve_dims) != 3:
                    # The RVE must be 3D
                    raise errors.IncompatibleDimensionsRVEphase(
                        "Cylindrical Fibers", 3, 2, i_phase
                    )
                if any(
                    [
                        i_phase.type != 1 and i_phase.type != 6
                        for i_phase.type in list(phase_types.values())
                    ]
                ):
                    raise errors.OnlyCylindricalFibers()
                particles = particles + generateCylindricalFibers(
                    i_phase, rve_dims, descriptors[i_phase.name]
                )
                # Generating the number of cylindrical fibers requested and appending them
                # to the list of particles
            else:
                raise errors.UnsupportedPhaseType(i_phase.type, i_phase.name)
        except (
            errors.IncompatibleDimensionsRVEphase,
            errors.OnlyCylindricalFibers,
        ) as error:
            error.message()
            quit()

    print_funcs.printToFile("**PHASE DESCRIPTORS**\n")
    for i_phase in Particle.phases.values():
        # Running through all the phases to print their info
        i_phase.printSpecDescriptors()
        i_phase.printRealDescriptors()
    print_funcs.printToFile("=" * 80)

    generateInitialConfiguration(particles, type_init_conf, save_history=True)
    # FIXME: save history as option

    createResultsDirectory(particles, dp_dir)

    return particles


# ==========================================================================================


def readDescriptors():
    """
    Load the descriptors and options to generate the microstructure.

    This function loads the descriptors and returns the microstructure descriptors, the
    phase types and options.

    Returns
    ----------
    dp_dir: string
        Directory where the microstructure spatial discretization file(s) associated
        with the given design point are to be stored

    mic_gen_program: integer
        Integer variable (read from the user input data file) which specifies an
        available program to generate the microstructure(s) and associated
        discretization file(s) of a given design point

    mic_gen_parameters: array
        An array which contains all the required parameters (or options)
        for the selected program to generate the microstructure(s) and
        and associated discretization file(s) of a given design point.

        ================================ ======================================
        Option                           Description
        ================================ ======================================
        "max_residue_per_particle"       Maximum overlap residue per particle.
        "max_step"                       Maximum number of iterations.
        "integration_scheme"             Optional. {'Newmark'}. Integration scheme
                                         for the equations of motion.
        "speed_up_scheme"                Optional. {'Naive', 'Cell', 'Verlet'}.
                                         Speed up scheme used for force computation
        "remesh"                         Optional. Boolean signaling a remesh action.
        "dir_previous_mic"               Optional. Directory where the input and
                                         output files of a previous microstructure
                                         are saved. They must have their original names.
        ================================ ======================================

    problem_type: integer
        Problem type    | 1. 2D problem (plain strain)
                        | 2. 2D problem (plain stress)
                        | 3. 2D problem (axisymmetric)
                        | 4. 3D problem

    n_dp_samples: integer
        Number of microstructures (samples) to be generated, associated to
        the given design point

    mic_gen_descriptors_array: array
        A dictionary which contains all the microstructure
        descriptor-related information required to generate the
        given design point microstructure(s) automatically stored as:

                                        Microstructure Descriptors
                                  _                                    _
        dictionary['phase_id'] = |  'desc_name'   'desc_name'     ...   |
                                 |_  < value >     < value >      ...  _|.

        See notes_.

    phase_types: dictionary
        Dictionary which contains each material phase type, stored as
                       dictionary['phase_id'] = phase_type
    discret_file_ext: list
        List which contains the required spatial discretization file(s), stored as:

                        array = [ < discret_type > < discret_type >  ... ]

    discret_spec_array: dictionary
        Dictionary which contains the required parameters to generate
        each type of specified discretization file, stored as:

                               dictionary['disc_ext']['parameter'] = [ ... ]
    Notes
    -----
    The parameters for microstructure generation depend on the shape of the particle. They
    are detailed in the following tables. Particular choices of their values may lead to
    incompatibilities.

        ================================ ======================================
        Disk: Choose 2 of the parameters
        -----------------------------------------------------------------------
        'r'                              Radius of the disk
        'n'                              Number of particles
        'vf'                             Volume fraction
        ================================ ======================================

        ================================ ======================================
        Ellipse: Choose 4 of the parameters, including 'angle'
        -----------------------------------------------------------------------
        'major_axis'                     Radius of the disk
        'minor_axis'                     Number of particles
        'angle'                          Volume fraction
        'n'                              Number of particles
        'vf'                             Volume fraction
        ================================ ======================================

    Any parameter may have a chosen distribution, specified as detailed below:
    - Fixed distribution: The parameters are fixed. Simply specify the parameter.
    - Discrete distribution: There parameters follow a discrete distribution, where the
    parameters take only the given values with the given probability.
        1. Specify the distribution of parameter *param* as::

                    np.array([['param_distribution']['fixed'], dtype=obj)

        2. Specify the value of the parameter and the probability of that value occuring::

                (np.array([['param_1', 'prob_param_1', 'param_2', 'prob_param_2'],
                        [1, 0.4, 2, 0.6]], dtype=obj))

    - Uniform distribution: "*_distribution"

            np.array([['distribution_param']['uniform'], dtype=obj)

    - Gaussian distribution:
    """
    info_dict = pickle.load(open("input_data\\info_micro.p", "rb"))
    # Loading the dictionary containing the information about the microstructure and its
    # generation
    dp_dir = info_dict.get("dp_dir")
    # Directory where the microstructure spatial discretization file(s) associated
    # with the given design point are to be stored
    options = info_dict.get("mic_gen_parameters")
    # An array which contains all the required parameters (or options)
    # for the selected program to generate the microstructure(s) and
    # and associated discretization file(s) of a given design point
    problem_type = info_dict.get("problem_type")
    # Getting the problem type
    n_dp_samples = info_dict.get("n_dp_samples", 1)
    # Number of samples to be generated using the descriptors supplied
    try:
        if not isinstance(n_dp_samples, int) or n_dp_samples < 1:
            # The number of samples must be an integer larger or equal to 1
            raise errors.NumberSamples(n_dp_samples)
    except errors.NumberSamples() as error:
        error.message()
        quit()

    descriptors = info_dict.get("mic_gen_descriptors", {})
    # mic_gen_descriptors_array: dictionary

    phase_types = info_dict.get("phase_types", {})
    # phase_types: dictionary
    try:
        if set(phase_types.keys()) != set(descriptors.keys()):
            # There are phases which not have descriptors or a phase type
            for phase in descriptors:
                if phase not in phase_types:
                    # If there is a phase that has descriptors but no phase type
                    raise errors.PhaseDescriptorsMatch(phase)
    except errors.PhaseDescriptorsMatch as error:
        error.message()
        quit()
    try:
        for phase in phase_types:
            if not RepresentsInt(phase) or not isinstance(phase, str):
                raise errors.UnexpectedValue(
                    phase, "key of phase_types", "string containing an integer"
                )
    except errors.UnexpectedValue as error:
        error.message()
        quit()

    discret_file_ext = info_dict.get("discret_file_ext", {})
    # Saving the list containing the meshes required
    discret_spec_array = info_dict.get("discret_spec_array", {})
    # Dictionary containing arrays with the specifications for each meash
    for ext in discret_spec_array:
        # Completing the list of extensions from the specification
        if ext not in discret_file_ext:
            discret_file_ext.append(ext)
    try:
        if len(discret_file_ext) == 0:
            # No mesh was specified
            raise errors.NoMesh()
    except errors.NoMesh as error:
        error.message()
        quit()
    try:
        for ext in discret_file_ext:
            # Check if all the required outputs have a description
            if ext not in discret_spec_array:
                raise errors.MissingInfoExtension(ext)
            for spec in discret_spec_array[ext]:
                # Check if the required extensions specify the bare minimum
                checkMeshSpecs(ext, discret_spec_array[ext])
    except errors.MissingInfoExtension as error:
        error.message()
        quit()

    rve_dims_spec = []
    for ext in discret_file_ext:
        # Running through the specified meshes
        rve_dims_spec.append(tuple(discret_spec_array[ext]["rve_dims"]))
        # Collecting the RVE dimensions specified
    rve_dims_spec = set(rve_dims_spec)
    # Obtaining the unique RVE size specifications
    if len(rve_dims_spec) > 1:
        # There are multiple RVE size specifications
        print_funcs.printToFile(
            "Warning: Different RVE sizes in the mesh specifications."
        )
        rve_dims = np.array(list(rve_dims_spec)[0])
        # Keeping the first
    else:
        rve_dims = np.array(list(rve_dims_spec)[0])

    return [
        dp_dir,
        descriptors,
        phase_types,
        options,
        n_dp_samples,
        rve_dims,
        problem_type,
        discret_spec_array,
        discret_file_ext,
    ]
