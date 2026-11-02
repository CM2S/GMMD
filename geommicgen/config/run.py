"""
Module holding the model of the whole input file, and reading it.

What one section can only be checked against another is checked here: the phases
against the RVE, and the meshes against the phases. The model gives the options the
package is built from, keyed as the input data file keyed them, so that everything
after the reading is the same whichever file the options came from.
"""

import os
from typing import Optional

import numpy as np
import pydantic

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.config.analyses import AnalysesConfig
from geommicgen.config.base import (
    ConfigError,
    ConfigModel,
    Problems,
    load_yaml,
    read_model,
)
from geommicgen.config.generation import Generation
from geommicgen.config.meshes import MeshesConfig, to_mesh_options
from geommicgen.config.phases import Fixed, PhaseModel, PhaseName
from geommicgen.microstructure.microstructure import DIRECTION_VARIES, phase_problems
from geommicgen.microstructure.particleclasses import Matrix


def fibre_direction(phase):
    """Give the direction of a phase as `.phase_problems` takes it."""
    direction = getattr(phase, "direction", None)
    if direction is None:
        return None

    return direction.value if isinstance(direction, Fixed) else DIRECTION_VARIES


class RunConfig(ConfigModel):
    """Model of the input file."""

    samples: int = pydantic.Field(ge=1, description="microstructures to generate")
    rve_dimensions: list[pydantic.PositiveFloat] = pydantic.Field(
        min_length=2, max_length=3, description="sides of the RVE"
    )
    seed: Optional[int] = pydantic.Field(
        None,
        description="seed of the first sample, each after it taking the next, so that "
        "a run gives the same microstructures every time",
    )
    save_min: bool = pydantic.Field(
        False, description="keep the least of what a sample leaves behind"
    )
    phases: dict[PhaseName, PhaseModel] = pydantic.Field(min_length=1)
    generation: Generation
    meshes: MeshesConfig = MeshesConfig()
    analyses: AnalysesConfig = AnalysesConfig()

    @pydantic.model_validator(mode="after")
    def sections_agree(self):
        """Refuse phases that do not fit the RVE or each other, or meshes the phases."""
        problems = self.phases_problems() + self.mesh_problems()
        if problems:
            raise Problems(problems)

        return self

    def phases_problems(self):
        """Give what is wrong with the phases, taken together and with the RVE."""
        dim = len(self.rve_dimensions)
        problems = [
            (("phases",) + i_loc, i_message)
            for i_loc, i_message in phase_problems(
                dim,
                {
                    i_name: (
                        i_phase.particle_class,
                        i_phase.inside,
                        fibre_direction(i_phase),
                    )
                    for i_name, i_phase in self.phases.items()
                },
            )
        ]
        # The rules a microstructure holds its phases to, as it is built
        for i_name, i_phase in self.phases.items():
            here = ("phases", i_name)
            particle_class = i_phase.particle_class
            if particle_class is not Matrix and particle_class.dim == dim:
                problems += [
                    (here + j_loc, j_message)
                    for j_loc, j_message in i_phase.problems(self.rve_dimensions)
                ]

        return problems

    def mesh_problems(self):
        """Give what is wrong with the meshes, taken with the phases and the RVE."""
        problems = []
        voxel = getattr(self.meshes, "voxel", None)
        if voxel is None:
            return problems
        resolutions = to_mesh_options(voxel).get("n_voxels_dims", [])
        for i_resolution in resolutions:
            if len(i_resolution) != len(self.rve_dimensions):
                problems.append(
                    (
                        ("meshes", "voxel", "n_voxels_dims"),
                        "{0} has {1} directions, and the RVE {2}".format(
                            i_resolution, len(i_resolution), len(self.rve_dimensions)
                        ),
                    )
                )

        return problems

    def to_options(self):
        """
        Give the run as the input data file keys it.

        Returns
        -------
        dict
            The options the rest of the package is built from, as the reader of the
            input data file gave them.
        """
        options = {
            "n_dp_samples": self.samples,
            "save_min": self.save_min,
            "mic_gen_parameters": {
                **self.generation.to_options(),
                "rve_dimensions": np.array(self.rve_dimensions, dtype=float),
                "fixed_seed": self.seed,
            },
            "mic_gen_descriptors": {
                i_name: i_phase.to_options() for i_name, i_phase in self.phases.items()
            },
            "post_proc": self.analyses.to_options(),
        }
        mesh_options = self.meshes.to_options()
        if mesh_options:
            options["mesh_options"] = mesh_options
        # Left out when none is asked for, as the input data file left it

        return options


def read_config(path):
    """
    Read an input file.

    Parameters
    ----------
    path: str
        Path of the file.

    Returns
    -------
    `.RunConfig`
        The run the file asks for, with every default filled in.

    Raises
    ------
    ConfigError:
        If the file cannot be read, or does not validate, naming every problem in it.
    """
    data = load_yaml(path)
    if not isinstance(data, dict):
        raise ConfigError(
            "{0} holds no sections, which an input file is made of.".format(
                os.path.basename(path)
            )
        )

    return read_model(RunConfig, data, os.path.basename(path))
