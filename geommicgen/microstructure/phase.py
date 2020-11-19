"""
This module contains the Phase class and the Descriptor class and its subclasses.

Each instance of Phase its a phase of the microstructure and each instance of Descriptor is
a phase descriptors. Different subclasses are able to generate samples from different
statistical distributions.
"""

import abc

import numpy as np

from .particle_classes import (
    Matrix,
    Disk,
    Ellipse,
    Sphere,
    Ellipsoid,
    CylindricalFiber,
    Cylinder,
)


class Phase:
    """
    This is the classe for a phase.

    Attributes
    ----------
    name: str
        Name of the phase.

    type: `.Particle`
        Type of phase.

    descriptors: dict
        Dictionary containing as keys the names of the descriptors and as values the
        descriptor class objects.

    microstucutre: `.Microstructure`
        Microstructure containing the phase.

    particles: list(`.Particles`)
        List of particles belonging to the phase.

    Class Attributes
    ----------------
    phase_types: dict
        Dictinary containing the correspondence between a phase type and its name.
    """

    phase_types = {
        1: Matrix,
        2: Disk,
        3: Ellipse,
        4: Sphere,
        5: Ellipsoid,
        6: CylindricalFiber,
        7: Cylinder,
    }
    # Correspondence between phase type and phase type class

    def __init__(self, name, phase_descriptors):
        """
        Intialize a Phase class instance.

        Parameters
        ---------
        name: string
            Name of the phase.

        phase_descriptors: dict
            Phase descriptors such as the phase type, the volume fraction, number of
            particles and so on.

        Raises
        -------
        KeyError:
            If there are descriptor parameters missing.

        ValueError:
            If the distribution specified for one of the descriptors is not supported.
        """
        self.microstucutre = None
        self.name = name
        # Name of the phase
        self.type = Phase.phase_types[phase_descriptors["phase_type"]]
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
        self.type.check_acceptable_description(set(self.descriptors.keys()))

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
                            try:
                                probabilities.append(
                                    phase_descriptors[
                                        i_descriptor.replace("value", "prob")
                                    ]
                                )
                                # Save the value
                            except KeyError:
                                print(
                                    "The probability for {0} was not supplied in Phase {1}.".format(
                                        descriptor + "_value", self.name
                                    )
                                )
                                raise

                    self.descriptors[descriptor] = DiscreteDistribution(
                        descriptor, values, probabilities
                    )
                elif descriptor_distribution == "fixed":
                    self.descriptors[descriptor] = FixedValue(
                        descriptor, phase_descriptors[descriptor]
                    )
                else:
                    raise ValueError(
                        "The distribution supplied for {0} in Phase {1} is not supported".format(
                            descriptor, self.name
                        )
                    )
            except KeyError:
                print(
                    "The descriptor {0} in Phase {1} is not completly defined.".format(
                        descriptor, self.name
                    )
                )
                raise

        self.particles = []

    @property
    def volume_fraction(self):
        """Volume fraction in decimal."""
        volume_fraction = 0
        for particle in self.particles:
            volume_fraction += particle.volume / self.microstructure.volume

        return volume_fraction

    @property
    def volume_fraction_circ(self):
        """Volume fraction in decimal of the circumscribed spheres/disks."""
        volume_fraction = 0
        for particle in self.particles:
            volume_fraction += particle.volume_circ / self.microstructure.volume

        return volume_fraction

    @property
    def number_particles(self):
        """Get number of particles."""
        return len(self.particles)

    def generate_particles(self, rve_dims):
        """
        Generate particles for a microstructure.

        Parameters
        ----------
        rve_dims: list
            List containing the size of the microstructure in each dimension.
        """
        particles = []
        if "vf" in self.descriptors and "n" not in self.descriptors:
            # The desired volume fraction was specfied
            current_sample = {}
            # Initializing the dictionary containing the samples for each parameter used
            vf_real = 0
            # Initializing the real volume fraction
            while vf_real < self.descriptors["vf"].value:

                for i_descriptor_name, i_descriptor in self.descriptors.items():
                    current_sample[i_descriptor_name] = i_descriptor.generate_sample()
                particles.append(self.type(self.name, current_sample, rve_dims))
                vf_real += particles[-1].volume / np.prod(rve_dims)
        else:
            # The desired number of disks was specified
            samples = {}
            # Initializing the dictionary containing the samples for each parameter used
            for i_descriptor_name, i_descriptor in self.descriptors.items():
                samples[i_descriptor_name] = i_descriptor.generate_sample(
                    n_samples=self.descriptors["n"].value
                )
            for i_particle in range(int(self.descriptors["n"].value)):
                i_particle_descriptors = (
                    {
                        descriptor_name: descriptor_values[i_particle]
                        for descriptor_name, descriptor_values in samples.items()
                    }
                    if self.descriptors["n"].value != 1
                    else samples
                )
                particles.append(self.type(self.name, i_particle_descriptors, rve_dims))

        self.particles = particles


class PhaseDescriptor(abc.ABC):
    """
    This is the class for phase descriptors.

    Attributes
    ----------
    name: str
        Name of the descriptor

    """

    def __init__(self, name):
        """
        Initialize a PhaseDescriptor class object.

        Parameters
        ----------
        name: str
            Name of the descriptor.
        """
        self.name = name

    def generate_sample(self, n_samples):
        """Generate a sample."""


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

    parameters = {"value"}

    def __init__(self, name, value):
        """
        Initialize for the FixedValue class object.

        Parameters
        ----------
        name: str
            Name of the descriptor

        value: object
            Specified value of the descriptor.
        """
        self.value = value
        super().__init__(name)

    def generate_sample(self, n_samples=1):
        """Return the fixed value of the descriptor."""
        sample = np.full((int(n_samples)), self.value)
        return sample if len(sample) > 1 else sample[0]


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

    def __init__(self, name, mean, sigma):
        """
        Initialize a NormalDistribution class object.

        Parameters
        ----------
        name: str
            Name of the descriptor

        mean: float
            Mean of the normal distribution.

        sigma: float
            Standard deviation of the normal distribution.
        """
        self.mean = mean
        self.sigma = sigma
        super().__init__(name)

    def generate_sample(self, n_samples=1):
        """Generate sample from a normal distribution."""
        sample = np.random.normal(loc=self.mean, scale=self.sigma, size=n_samples)
        return sample if len(sample) > 1 else sample[0]


class UniformDistribution(PhaseDescriptor):
    """
    This is the class for phase descriptors with a uniform distribution.

    Attributes
    ----------
    low: float
        Lower bound of the uniform distribution.

    high: float
        Upper bound of the uniform distribution.

    Class Attributes
    ----------------
    parameters: set
        Parameters of the statistical distribution.
    """

    parameters = {"low", "high"}

    def __init__(self, name, low, high):
        """
        Initialize a NormalDistribution class object.

        Parameters
        ----------
        name: str
            Name of the descriptor

        low: float
            Lower bound of the uniform distribution.

        high: float
            Upper bound of the uniform distribution.
        """
        if low > high:
            raise ValueError(
                "Descriptor {0}: For a uniform distribution the lower bound must be smaller than the upper bound.".format(
                    name
                )
            )

        self.low = low
        self.high = high
        super().__init__(name)

    def generate_sample(self, n_samples=1):
        """Generate sample from a normal distribution."""
        sample = np.random.uniform(low=self.low, high=self.high, size=n_samples)
        return sample if len(sample) > 1 else sample[0]


class DiscreteDistribution(PhaseDescriptor):
    """
    This is the class for phase descriptors with a discrete distribution.

    Attributes
    ----------
    values: list(float)
        Values that the descriptor can take.

    probabilities: list(float)
        Probability for each value.

    Class Attributes
    ----------------
    parameters: set
        Parameters of the statistical distribution.
    """

    parameters = {"value_" + str(i) for i in range(1, 11)}.union(
        {"prob_" + str(i) for i in range(1, 11)}
    )

    def __init__(self, name, values, probabilities):
        """
        Initialize the DiscreteDistribution class.

        Parameters
        ----------
        name: str
            Name of the descriptor

        values: list(float)
            Values that the descriptor can take.

        probabilities: list(float)
            Probability for each value.
        """
        if np.abs(np.sum(probabilities) - 1) > 1e-2:
            # probabilities don't add up to one
            raise ValueError(
                "Descriptor {0}: The probabilities must sum to one.".format(name)
            )

        self.values = values
        self.probabilities = probabilities

        super().__init__(name)

    def generate_sample(self, n_samples=1):
        """Generate sample from the discrete distribution."""
        sample = np.random.choice(self.values, n_samples, p=self.probabilities)

        return sample if len(sample) > 1 else sample[0]
