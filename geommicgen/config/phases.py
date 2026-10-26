"""
Module holding the models of the phases of the input file, and of their descriptors.

A phase is given by the type of its particles and a descriptor for each parameter of
one of the descriptions the type accepts. The model of each type is built from what
the particle class declares, its parameters and its descriptions, so that the class is
the one place they are written. A descriptor is a fixed value, given as a bare number,
or a distribution the value of each particle is drawn from, written as the
microstructure file writes one.
"""

import math
import re
from typing import Annotated, ClassVar, Literal, Optional, Union

import numpy as np
import pydantic

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.config.base import ConfigModel, Problems
from geommicgen.microstructure.phase import Phase

PROBABILITY_TOLERANCE = 1e-8
# How far from one the probabilities of a discrete distribution may add up to: the
# drawing refuses them beyond it, so a file that is let through with less would be
# refused once the particles are drawn


class Fixed(ConfigModel):
    """Model of a descriptor that is the same for every particle."""

    distribution: Literal["fixed"]
    value: Union[int, float]

    def options(self, name, kind):
        """Give the descriptor as the input data file keys it."""
        return {name: int(self.value) if kind == "int" else float(self.value)}


class Specified(ConfigModel):
    """Model of a descriptor given particle by particle."""

    distribution: Literal["specified"]
    values: list[float] = pydantic.Field(min_length=1)

    def options(self, name, kind):
        """Give the descriptor as the input data file keys it."""
        return {
            name + "_distribution": "specified",
            name: np.array(self.values, dtype=float),
        }


class Normal(ConfigModel):
    """Model of a descriptor drawn from a normal distribution."""

    distribution: Literal["normal"]
    mean: float
    sigma: float = pydantic.Field(ge=0)

    def options(self, name, kind):
        """Give the descriptor as the input data file keys it."""
        return {
            name + "_distribution": "normal",
            name + "_mean": float(self.mean),
            name + "_sigma": float(self.sigma),
        }


class LogNormal(ConfigModel):
    """Model of a descriptor drawn from a lognormal distribution."""

    distribution: Literal["lognormal"]
    mean: float = pydantic.Field(gt=0)
    sigma: float = pydantic.Field(gt=0)
    # The scale and the shape of scipy's lognormal, which gives no number for either
    # at zero or below

    def options(self, name, kind):
        """Give the descriptor as the input data file keys it."""
        return {
            name + "_distribution": "lognormal",
            name + "_mean": float(self.mean),
            name + "_sigma": float(self.sigma),
        }


class Uniform(ConfigModel):
    """Model of a descriptor drawn from a uniform distribution."""

    distribution: Literal["uniform"]
    low: float
    high: float

    @pydantic.model_validator(mode="after")
    def low_is_not_above_high(self):
        """Refuse a lower bound above the upper one."""
        if self.low > self.high:
            raise ValueError(
                "the lower bound, {0}, is above the upper one, {1}".format(
                    self.low, self.high
                )
            )

        return self

    def options(self, name, kind):
        """Give the descriptor as the input data file keys it."""
        return {
            name + "_distribution": "uniform",
            name + "_low": float(self.low),
            name + "_high": float(self.high),
        }


class VonMises(ConfigModel):
    """Model of a descriptor drawn from a von Mises distribution."""

    distribution: Literal["vonmises"]
    kappa: float = pydantic.Field(gt=0)
    loc: float
    scale: float = pydantic.Field(gt=0)

    def options(self, name, kind):
        """Give the descriptor as the input data file keys it."""
        return {
            name + "_distribution": "vonmises",
            name + "_kappa": float(self.kappa),
            name + "_loc": float(self.loc),
            name + "_scale": float(self.scale),
        }


class Discrete(ConfigModel):
    """Model of a descriptor that takes one of a few values."""

    distribution: Literal["discrete"]
    values: list[float] = pydantic.Field(min_length=1)
    probabilities: list[pydantic.NonNegativeFloat] = pydantic.Field(min_length=1)

    @pydantic.model_validator(mode="after")
    def a_probability_for_each_value_adding_up_to_one(self):
        """Refuse probabilities that are not one per value, or do not add up to one."""
        if len(self.values) != len(self.probabilities):
            raise ValueError(
                "there are {0} values and {1} probabilities".format(
                    len(self.values), len(self.probabilities)
                )
            )
        if abs(math.fsum(self.probabilities) - 1) > PROBABILITY_TOLERANCE:
            raise ValueError(
                "the probabilities add up to {0!r}, not to one".format(
                    math.fsum(self.probabilities)
                )
            )

        return self

    def options(self, name, kind):
        """Give the descriptor as the input data file keys it."""
        options = {name + "_distribution": "discrete"}
        for i_index, (i_value, i_probability) in enumerate(
            zip(self.values, self.probabilities)
        ):
            options["{0}_value_{1}".format(name, i_index + 1)] = float(i_value)
            options["{0}_prob_{1}".format(name, i_index + 1)] = float(i_probability)

        return options


def bare_value(value):
    """Take a bare number for a fixed value."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {"distribution": "fixed", "value": value}

    return value


Descriptor = Annotated[
    Union[Fixed, Specified, Normal, LogNormal, Uniform, VonMises, Discrete],
    pydantic.Field(discriminator="distribution"),
    pydantic.BeforeValidator(bare_value),
]
# A descriptor, named by its distribution as the microstructure file names it


def phase_name(value):
    """Take a phase named by a number, as a YAML key without quotes is read."""
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)

    return value


def a_number(name):
    """Refuse a phase name that is not a whole number."""
    if not re.fullmatch(r"\d+", name):
        raise ValueError(
            "a phase is named by a whole number, which the meshes label it with"
        )

    return name


PhaseName = Annotated[
    str, pydantic.BeforeValidator(phase_name), pydantic.AfterValidator(a_number)
]
# The name of a phase. The meshes label each phase with it as a number


def type_name(particle_class):
    """
    Give the name a type of particle is asked for by in the input file.

    Parameters
    ----------
    particle_class: class
        Class of the particles.

    Returns
    -------
    str
        The name of the class in lower case, with an underscore between its words.
    """
    return re.sub(r"(?<!^)(?=[A-Z])", "_", particle_class.__name__).lower()


class PhaseConfig(ConfigModel):
    """
    Model of a phase, from which the model of each type of particle derives.

    Class Attributes
    ----------------
    particle_class: class
        Class of the particles of the phase.
    """

    particle_class: ClassVar = None
    inside: Optional[PhaseName] = None

    @property
    def descriptors(self):
        """The descriptors the phase gives, keyed by their parameter."""
        return {
            i_name: getattr(self, i_name)
            for i_name in self.particle_class.possible_parameters
            if getattr(self, i_name) is not None
        }

    @pydantic.model_validator(mode="after")
    def is_a_description(self):
        """Refuse descriptors that are not a description of the type."""
        problems = []
        given = set(self.descriptors)
        descriptions = self.particle_class.acceptable_descriptions
        if given not in descriptions:
            problems.append(
                (
                    (),
                    "{0} do not describe a {1}, which is described by one of: "
                    "{2}".format(
                        ", ".join(sorted(given)) or "no descriptors",
                        type_name(self.particle_class),
                        "; ".join(
                            ", ".join(sorted(i_description))
                            for i_description in descriptions
                        ),
                    ),
                )
            )
        for i_name, i_descriptor in self.descriptors.items():
            kind = self.particle_class.possible_parameters[i_name][2]
            if i_name in ("n", "vf") and not isinstance(i_descriptor, Fixed):
                problems.append(((i_name,), "has to be a fixed value, a number"))
            elif (
                kind == "int"
                and isinstance(i_descriptor, Fixed)
                and i_descriptor.value != int(i_descriptor.value)
            ):
                problems.append(((i_name,), "has to be a whole number"))
        # The number of particles and the volume fraction are read as the one value
        # they are, which a distribution does not have
        number = self.descriptors.get("n")
        for i_name, i_descriptor in self.descriptors.items():
            if (
                isinstance(i_descriptor, Specified)
                and isinstance(number, Fixed)
                and len(i_descriptor.values) != number.value
            ):
                problems.append(
                    (
                        (i_name, "values"),
                        "gives {0} values, one for each particle, and the phase has "
                        "{1} particles".format(
                            len(i_descriptor.values), int(number.value)
                        ),
                    )
                )
        # A phase given by its volume fraction has as many particles as it takes, which
        # is found out as they are drawn
        if problems:
            raise Problems(problems)

        return self

    def problems(self, rve_dims):
        """
        Give the fixed descriptors that are outside their bounds in an RVE.

        Parameters
        ----------
        rve_dims: list
            Dimensions of the RVE.

        Returns
        -------
        list
            Tuples *(location, message)*, the location below the phase.
        """
        problems = []
        for i_name, i_descriptor in self.descriptors.items():
            label, bound, kind = self.particle_class.possible_parameters[i_name]
            if not isinstance(i_descriptor, Fixed):
                continue
            value = i_descriptor.options(i_name, kind)[i_name]
            if not bound(value, rve_dims):
                problems.append(
                    (
                        (i_name,),
                        "the {0} cannot be {1} in an RVE of {2}".format(
                            label.lower(), value, list(rve_dims)
                        ),
                    )
                )
        # A distribution is drawn from particle by particle, and each value drawn is
        # held to the bounds then

        return problems

    def to_options(self):
        """
        Give the phase as the input data file keys it.

        Returns
        -------
        dict
            The descriptors of the phase, each by its keys in the input data file,
            with the type of the phase by its number and the phase it is inside.
        """
        codes = {i_class: i_code for i_code, i_class in Phase.phase_types.items()}
        options = {"phase_type": codes[self.particle_class]}
        if self.inside is not None:
            options.update({"inner_phase": True, "outer_phase": int(self.inside)})
        for i_name, i_descriptor in self.descriptors.items():
            options.update(
                i_descriptor.options(
                    i_name, self.particle_class.possible_parameters[i_name][2]
                )
            )

        return options


def phase_model(particle_class):
    """
    Build the model of a phase of a type of particle, from what the class declares.

    Parameters
    ----------
    particle_class: class
        Class of the particles.

    Returns
    -------
    class
        The model, a `.PhaseConfig` with a descriptor for each parameter the class
        declares, each described by the name the class gives it.
    """
    fields = {
        i_name: (Optional[Descriptor], pydantic.Field(None, description=i_label))
        for i_name, (i_label, _, _) in particle_class.possible_parameters.items()
    }
    model = pydantic.create_model(
        particle_class.__name__ + "Phase",
        __base__=PhaseConfig,
        type=(Literal[type_name(particle_class)], ...),
        **fields
    )
    model.particle_class = particle_class

    return model


PHASE_MODELS = {
    type_name(i_class): phase_model(i_class) for i_class in Phase.phase_types.values()
}
# The model of a phase of each type of particle, under the name it is asked for by

PhaseModel = Annotated[
    Union[tuple(PHASE_MODELS.values())], pydantic.Field(discriminator="type")
]
# A phase, of the type its key names
