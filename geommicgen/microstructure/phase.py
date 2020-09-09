"""
This module contains the Phase class and the Descriptor class and its subclasses.

Each instance of Phase its a phase of the microstructure and each instance of Descriptor is
a phase descriptors. Different subclasses are able to generate samples from different
statistical distributions.
"""

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
                        phase_descriptors[descriptor + "_low"],
                        phase_descriptors[descriptor + "_high"],
                    )
                elif descriptor_distribution == "normal":
                    self.descriptors[descriptor] = NormalDistribution(
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
                    self.phase[descriptor] = DiscreteDistribution(values, probabilities)
                elif descriptor_distribution == "fixed":
                    self.phase[descriptor] = FixedValue(phase_descriptors[descriptor])
                else:
                    self.phase[phase_name][parameter] = UniformDistribution(
                        descriptors[parameter + "_low"],
                        descriptors[parameter + "_high"],
                    )
            except KeyError as e:
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
    pass


class FixedValue(PhaseDescriptor):
    parameters = {}


class NormalDistribution(PhaseDescriptor):
    parameters = {"mean", "sigma"}


class UniformDistribution(PhaseDescriptor):
    parameters = {"low", "high"}


class DiscreteDistribution(PhaseDescriptor):
    parameters = {"value_" + str(i) for i in range(1, 11)}.union(
        {"prob_" + str(i) for i in range(1, 11)}
    )
