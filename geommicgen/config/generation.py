"""
Module holding the model of the generation section of the input file.

The generation is done by a method, named in the section, whose parameters the rest of
the section gives. The thermostat and the speed up scheme of the molecular dynamics
simulation are sections of their own, each chosen by its name and taking only its own
parameters, where the input data file gave every parameter of every one of them side by
side.
"""

from typing import Annotated, Literal, Optional, Union

import pydantic

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.config.base import ConfigModel, named
from geommicgen.micgenmethod.molecular_dynamics_sim import INITIAL_CONFIGURATIONS


class CellList(ConfigModel):
    """Model of the cell list, which bins the particles in cells of the largest."""

    name: Literal["cell"]

    def to_options(self):
        """Give the scheme as the input data file keys it."""
        return {"speed_up_scheme": "Cell"}


class NaiveList(ConfigModel):
    """Model of the naive list, which tests every pair of particles."""

    name: Literal["naive"]

    def to_options(self):
        """Give the scheme as the input data file keys it."""
        return {"speed_up_scheme": "Naive"}


class VerletList(ConfigModel):
    """Model of the Verlet list, which keeps the neighbours within a skin."""

    name: Literal["verlet"]
    factor: float = pydantic.Field(
        gt=1, description="radius of the neighbourhood, as a multiple of the particle's"
    )

    def to_options(self):
        """Give the scheme as the input data file keys it."""
        return {"speed_up_scheme": "Verlet", "verlet_factor": float(self.factor)}


SPEED_UP_SCHEMES = (CellList, NaiveList, VerletList)
# The models of the schemes, one for each name `SPEED_UP_SCHEMES` of the simulation
# offers, which a test holds them to

SpeedUp = Annotated[
    Union[SPEED_UP_SCHEMES],
    pydantic.Field(discriminator="name"),
    pydantic.BeforeValidator(named),
]


class RatioInOut(ConfigModel):
    """Model of lowering the temperature when the overlap oscillates."""

    name: Literal["ratio_in_out"]
    max_ratio_osc: int = pydantic.Field(2, ge=1)


class RollingAverage(ConfigModel):
    """Model of lowering the temperature when the overlap's rolling average stalls."""

    name: Literal["rolling_ave"]
    average_window: int = pydantic.Field(25, ge=1)


class Original(ConfigModel):
    """Model of lowering the temperature after a number of steps at it."""

    name: Literal["original"]
    min_eq_steps_at_temp: int = pydantic.Field(25, ge=1)


CRITERIA = (RatioInOut, RollingAverage, Original)
# The models of the criteria, one for each name `LOWERING_TEMP_CRITERIA` of the
# thermostats offers, which a test holds them to

Criterion = Annotated[
    Union[CRITERIA],
    pydantic.Field(discriminator="name"),
    pydantic.BeforeValidator(named),
]


class MultiTemperature(ConfigModel):
    """Model of the thermostat that lowers the temperature by a criterion."""

    name: Literal["multi_temperature"]
    initial_temp: Optional[float] = pydantic.Field(None, ge=0)
    criterion: Criterion = RatioInOut(name="ratio_in_out")
    temp_low_ratio: float = pydantic.Field(0.25, gt=0)

    def to_options(self):
        """Give the thermostat as the input data file keys it."""
        return {
            "initial_temp": self.initial_temp,
            "lowering_temp_criterion": self.criterion.name,
            "temp_low_ratio": self.temp_low_ratio,
            **self.criterion.model_dump(exclude={"name"}),
        }


class Isokinetic(ConfigModel):
    """Model of the thermostat that keeps the temperature constant."""

    name: Literal["isokinetic"]
    initial_temp: Optional[float] = pydantic.Field(None, ge=0)

    def to_options(self):
        """Give the thermostat as the input data file keys it."""
        return {"initial_temp": self.initial_temp}


class MicroCanonical(ConfigModel):
    """Model of no thermostat, which keeps the energy."""

    name: Literal["micro_canonical"]

    def to_options(self):
        """Give the thermostat as the input data file keys it."""
        return {}


class Berendsen(ConfigModel):
    """Model of the thermostat that pushes the temperature towards one."""

    name: Literal["berendsen"]
    initial_temp: float = pydantic.Field(ge=0)
    coeff: float = 1e-2

    def to_options(self):
        """Give the thermostat as the input data file keys it."""
        return {"initial_temp": self.initial_temp, "berendsen_coeff": self.coeff}


THERMOSTATS = (MultiTemperature, Isokinetic, MicroCanonical, Berendsen)
# The models of the thermostats, one for each name `THERMOSTATS` of the thermostats
# offers, which a test holds them to

Thermostat = Annotated[
    Union[THERMOSTATS],
    pydantic.Field(discriminator="name"),
    pydantic.BeforeValidator(named),
]


def thermostat_defaults():
    """
    Give every parameter of every thermostat, at its default, as the deck keys it.

    Returns
    -------
    dict
        What an input data file gives for the thermostats when it names none of their
        parameters, which it gives whatever the thermostat.
    """
    defaults = MultiTemperature(name="multi_temperature").to_options()
    for i_criterion in CRITERIA:
        defaults.update(
            {
                j_name: j_field.default
                for j_name, j_field in i_criterion.model_fields.items()
                if j_name != "name"
            }
        )
    defaults["berendsen_coeff"] = Berendsen.model_fields["coeff"].default

    return defaults


class MolecularDynamics(ConfigModel):
    """Model of the generation by a molecular dynamics simulation."""

    method: Literal["molecular_dynamics"] = "molecular_dynamics"
    max_residue_per_particle: float = pydantic.Field(
        ge=0,
        description="overlap per particle a configuration may keep and be accepted, a "
        "length in the units of the RVE",
    )
    max_step: int = pydantic.Field(ge=0, description="most steps the simulation takes")
    max_steps_to_relax: int = pydantic.Field(0, ge=0)
    dt: float = pydantic.Field(0.05, gt=0, description="time step")
    dt_adapt: bool = True
    save_history: bool = False
    type_initial_configuration: Literal[INITIAL_CONFIGURATIONS] = "random"
    min_distance: float = pydantic.Field(0, ge=0)
    initial_vel_coeff: float = 0.1
    final_overlap_check: bool = False
    damping_coeff: float = 0
    particle_mass_opt: Literal["volume", "radius", "unit"] = "radius"
    offset: bool = True
    speed_up: SpeedUp = CellList(name="cell")
    thermostat: Thermostat = MultiTemperature(name="multi_temperature")

    def to_options(self):
        """
        Give the generation parameters as the input data file keys them.

        Returns
        -------
        dict
            Every parameter the input data file gives, whether the file names it or
            not, as it gives them.
        """
        options = self.model_dump(exclude={"method", "speed_up", "thermostat"})
        options.update(
            {
                "force_option": "intersection_length",
                "force_rescale": False,
                "thermostat": self.thermostat.name,
                **thermostat_defaults(),
                **self.thermostat.to_options(),
                **self.speed_up.to_options(),
            }
        )
        # The two force options the input data file offered have one value each now,
        # and are not asked for

        return options
