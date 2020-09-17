"""
This module contains the Phase class and the Descriptor class and its subclasses.

Each instance of Phase its a phase of the microstructure and each instance of Descriptor is
a phase descriptors. Different subclasses are able to generate samples from different
statistical distributions.
"""

import numpy as np

from .particle_classes import Disk, Ellipse, Sphere, Ellipsoid


class GeometricalParameter:
    """
    This is the class for GeometricalParameters, characterized by a name, statistical
    distribution and the corresponding statistical parameters.

    Attributes
    ----------
    name: string
        Name of the geometrical parameter

    distribution: string
        Name of the distribution

    distribution_parameters: list(tuples)
        List of tuples containing the parameter name and the corresponding value.
        (name, value)
    """

    def __init__(self, geom_parameter, distribution, distribution_parameters):
        self.name = geom_parameter
        self.distribution = distribution
        self.distribution_parameters = distribution_parameters


class Phase:
    """
    This is the classe for a phase.

    Attributes
    ----------
    number_particles: integer
        Number of particles in the phase.

    volume_fraction: float
        Volume fraction of the phase.

    type: int
        Type of phase.

    type_name: string
        Name of the phase type.

    geometrical_descriptors: set(strings)
        Geometrical descriptors of the phase.

    Class Attributes
    ----------------
    phase_type_name: dict
        Dictinary containing the correspondence between a phase type and its name.
    """

    phase_types = {1: "Matrix", 2: Disk, 3: Ellipse, 4: Sphere, 5: Ellipsoid}
    # Correspondence between phase type and phase type class

    def __init__(self, name, phase_descriptors):
        """
        Constructor for the Phase class.

        Parameter
        ---------
        name: string
            Name of the phase.

        phase_descriptors: dict
            Phase descriptors such as the phase type, the volume fraction, number of
            particles and so on.
        """

        self.name = name
        # Name of the phase
        self.type = Phase.phase_types[phase_descriptors.pop("Phase_Type")]
        # Type of the phase
        self.descriptors = {
            possible_descriptor: None
            for possible_descriptor in self.type.possible_parameters
            if any(
                [
                    descriptor.startswith(possible_descriptor)
                    for descriptor in phase_descriptors.keys()
                ]
            )
        }

        self.type.checkAcceptableDescription(self.descriptors)
        for descriptor in self.descriptors:
            descriptor_distribution = phase_descriptors.get(
                descriptor + "_distribution", "fixed"
            )
            try:
                if descriptor_distribution == "uniform":
                    # the radius follows a uniform distribution
                    self.descriptors[descriptor] = UniformDistribution(
                        descriptor,
                        phase_descriptors[descriptor + "_low"],
                        phase_descriptors[descriptor + "_high"],
                    )
                elif descriptor_distribution == "normal":
                    self.descriptors[descriptor] = NormalDistribution(
                        descriptor,
                        phase_descriptors[descriptor + "_mean"],
                        phase_descriptors[descriptor + "_sigma"],
                    )
                elif descriptor_distribution == "discrete":
                    # Recovering the pairs value/probability, specified as
                    # descriptor_value_* and descriptor_prob_*
                    values = []
                    probabilities = []
                    for i_descriptor in phase_descriptors:
                        if i_descriptor.startswith(descriptor + "_value"):
                            values.append(phase_descriptors[i_descriptor])
                            probabilities.append(
                                phase_descriptors[i_descriptor.replace("value", "prob")]
                            )
                            # Save the value

                    self.phase[descriptor] = DiscreteDistribution(
                        descriptor, values, probabilities
                    )
                elif descriptor_distribution == "fixed":
                    self.phase[descriptor] = FixedValue(
                        descriptor, phase_descriptors[descriptor]
                    )
                else:
                    raise ValueError
            except KeyError as e:
                print("Error")
                quit()
            except ValueError as e:
                print("Error")
                quit()
        self.real_volume_fraction = 0
        self.spec_volume_fraction = 0
        self.virtual_volume_fraction = 0
        self.real_number = 0
        self.spec_number = 0
        self.geometrical_parameters = []
        # Initializing the list containing the geometrical parameters specified

    # def specNumber():
    #     pass
    #
    # def realNumber():
    #     pass
    #
    # def specVolumeFraction(self, vf):
    #     self.volume_fraction = vf
    #
    # def realVolumeFraction():
    #     pass
    #
    # def virtualVolumeFraction():
    #     pass
    @staticmethod
    def generate_particles(rve_dims, particle_class, phase, **kwargs):
        """
        Generate particles for a microstructure.

        Parameters
        ----------
        rve_dims: list
            List containing the size of the microstructure in each dimension.

        particle_class: class `.Particle`
            Reference to the class of the particles to be generated.

        phase: str
            Phase name.

        Keyword Parameters
        ------------------
        descriptor_name*: `.PhaseDescriptor`
            Phase descriptor
        """
        particles = []
        descriptors = kwargs
        if "vf" in descriptors and "n" not in descriptors:
            # The desired volume fraction was specfied
            current_sample = {}
            # Initializing the dictionary containing the samples for each parameter used
            vf_real = 0
            # Initializing the real volume fraction
            while vf_real < descriptors["vf"].value:
                for i_descriptor_name, i_descriptor in descriptors.items():
                    current_sample[i_descriptor_name] = i_descriptor.generateSample()
                    particles.append(particle_class(phase, **current_sample))
                    vf_real += particles[-1].volume / np.prod(rve_dims)
        else:
            # The desired number of disks was specified
            samples = {}
            # Initializing the dictionary containing the samples for each parameter used
            for i_descriptor_name, i_descriptor in descriptors.items():
                samples[i_descriptor_name] = i_descriptor.generateSample(
                    n_samples=descriptors["n"].value
                )
            for i_particle in range(descriptors["n"]):
                i_particle_descriptors = {
                    descriptor_name: descriptor_values[i_particle]
                    for descriptor_name, descriptor_values in samples.items()
                }
                particles.append(particle_class(phase, **i_particle_descriptors))

        return particles

    def addGeomParameter(self, name, distribution_name, distribution_parameters):
        """
        Add a new geometrical parameter with a name, a distribution and the corresponding
        parameters.

        Parameter
        ---------
        name: str
            Name of the geometrical parameter

        distribution_name: str
            Name of the distribution

        distribution_parameters: list(tuple(str, float))
            List containing the pairs parameter name and value in tuples.
        """
        if all(
            [
                name != geom_parameter.name
                for geom_parameter in self.geometrical_parameters
            ]
        ):
            self.geometrical_parameters.append(
                GeometricalParameter(name, distribution_name, distribution_parameters)
            )

    def printGeomParameteres(self):
        for geom_parameter in self.geometrical_parameters:
            print_funcs.printToFile("\t\t- {0}:".format(geom_parameter.name))
            print_funcs.printToFile(
                "\t\t\t- Distribution: {0}".format(geom_parameter.distribution)
            )
            for (
                dist_param_name,
                dist_param_value,
            ) in geom_parameter.distribution_parameters:
                print_funcs.printToFile(
                    "\t\t\t- {0}: {1}".format(dist_param_name, dist_param_value)
                )

    def printSpecDescriptors(self):
        print_funcs.printToFile("\tPhase {0}: ({1})".format(self.name, self.type_name))
        if self.spec_volume_fraction == 0:
            print_funcs.printToFile(
                "\t\t- Volume fraction: {:.6f}%".format(self.real_volume_fraction * 100)
            )
        else:
            print_funcs.printToFile("\t\t- Volume fraction:")
            print_funcs.printToFile(
                "\t\t\t- Specified: {:.6f}%".format(self.spec_volume_fraction * 100)
            )
            print_funcs.printToFile(
                "\t\t\t- Real: {:.6f}%".format(self.real_volume_fraction * 100)
            )

        if self.type != 1:
            if self.spec_number == 0:
                print_funcs.printToFile(
                    "\t\t- Number of particles: {0}".format(self.real_number)
                )
            else:
                print_funcs.printToFile("\t\t- Number of particles:")
                print_funcs.printToFile(
                    "\t\t\t- Specified: {0}".format(int(self.spec_number))
                )
                print_funcs.printToFile(
                    "\t\t\t- Real: {0}".format(int(self.real_number))
                )
            self.printGeomParameteres()
            # print_funcs.printToFile()

    def printRealDescriptors(self):
        pass


class PhaseDescriptor:
    """
    This is the class for phase descriptors.

    Attributes
    ----------
    name: str
        Name of the descriptor

    """

    def __init__(self, name):
        """
        Initializer for the PhaseDescriptor class.

        Parameters
        ----------
        name: str
            Name of the descriptor.
        """
        self.name = name


class FixedValue(PhaseDescriptor):
    """
    This is the class for phase descriptors with a fixed value.

    Attributes
    ----------
    value: object
        Specified value of the descriptor.

    real_value: object
        Real value of the descriptor. Used when a given descriptor cannot be exactly
        satisfied.

    Class Attributes
    ----------------
    parameters: set
        Parameters of the statistical distribution.
    """

    parameters = {}

    def __init__(self, value):
        """
        Initializer for the FixedValue class.

        Parameters
        ----------
        value: object
            Specified value of the descriptor.
        """
        self.value = value

    def generateSample(self):
        return self.value


class NormalDistribution(PhaseDescriptor):
    """
    This is the class for phase descriptors with a normal distribution.

    Attributes
    ----------
    mean: float
        Mean of the normal distribution.

    sigma: float
        Standard deviation of the normal distribution.

    Class Attributes
    ----------------
    parameters: set
        Parameters of the statistical distribution.
    """

    parameters = {"mean", "sigma"}

    def __init__(self, mean, sigma):
        """
        Initializer for the NormalDistribution class.

        Parameters
        ----------
        mean: float
            Mean of the normal distribution.

        sigma: float
            Standard deviation of the normal distribution.
        """
        self.mean = mean
        self.sigma = sigma

    def generateSample(self, n_samples=1):
        """
        Generate sample from a normal distribution.
        """
        sample = np.random.normal(loc=self.mean, scale=self.sigma, size=n_samples)
        return sample


class UniformDistribution(PhaseDescriptor):
    """
    This is the class for phase descriptors with a uniform distribution.

    Attributes
    ----------
    value: object
        Specified value of the descriptor.

    real_value: object
        Real value of the descriptor. Used when a given descriptor cannot be exactly
        satisfied.

    Class Attributes
    ----------------
    parameters: set
        Parameters of the statistical distribution.
    """

    parameters = {"low", "high"}

    def __init__(self, low, high):
        """
        Initializer for the NormalDistribution class.

        Parameters
        ----------
        mean: float
            Mean of the normal distribution.

        sigma: float
            Standard deviation of the normal distribution.
        """
        if low > high:
            raise ValueError

        self.low = low
        self.high = high

    def generateSample(self, n_samples=1):
        """
        Generate sample from a normal distribution.
        """
        sample = np.random.uniform(low=self.low, scale=self.high, size=n_samples)
        return sample


class DiscreteDistribution(PhaseDescriptor):
    """
    This is the class for phase descriptors with a discrete distribution.

    Attributes
    ----------
    value: object
        Specified value of the descriptor.

    real_value: object
        Real value of the descriptor. Used when a given descriptor cannot be exactly
        satisfied.

    Class Attributes
    ----------------
    parameters: set
        Parameters of the statistical distribution.
    """

    parameters = {"value_" + str(i) for i in range(1, 11)}.union(
        {"prob_" + str(i) for i in range(1, 11)}
    )

    def __init__(self, low, high):
        """
        Initializer for the NormalDistribution class.

        Parameters
        ----------
        mean: float
            Mean of the normal distribution.

        sigma: float
            Standard deviation of the normal distribution.
        """
        if low > high:
            raise ValueError

        self.low = low
        self.high = high

    def generateSample(self, n_samples=1):
        """
        Generate sample from a normal distribution.
        """
        sample = np.random.uniform(low=self.low, scale=self.high, size=n_samples)
        return sample
