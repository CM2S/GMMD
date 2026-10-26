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
from geommicgen.config.analyses import AnalysesConfig, analyses_to_options
from geommicgen.config.base import (
    ConfigError,
    ConfigModel,
    Problems,
    load_yaml,
    read_model,
)
from geommicgen.config.generation import Generation
from geommicgen.config.meshes import MeshesConfig
from geommicgen.config.phases import Fixed, PhaseModel, PhaseName
from geommicgen.microstructure.particleclasses import CylindricalFiber, Matrix


def fibres_in(phases):
    """Give the phases of cylindrical fibres, keyed by their names."""
    return {
        i_name: i_phase
        for i_name, i_phase in phases.items()
        if i_phase.particle_class is CylindricalFiber
    }


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
        problems = self.phase_problems() + self.mesh_problems()
        if problems:
            raise Problems(problems)

        return self

    def phase_problems(self):
        """Give what is wrong with the phases, taken together and with the RVE."""
        problems = []
        dim = len(self.rve_dimensions)
        matrices = [
            i_name
            for i_name, i_phase in self.phases.items()
            if i_phase.particle_class is Matrix
        ]
        if len(matrices) != 1:
            problems.append(
                (
                    ("phases",),
                    "one phase has to be the matrix, and {0}".format(
                        "phases {0} are".format(", ".join(matrices))
                        if matrices
                        else "none is"
                    ),
                )
            )
        fibres = fibres_in(self.phases)
        for i_name, i_phase in self.phases.items():
            here = ("phases", i_name)
            particle_class = i_phase.particle_class
            if particle_class is not Matrix and particle_class.dim != dim:
                problems.append(
                    (
                        here + ("type",),
                        "a {0} is in {1} dimensions, and the RVE in {2}".format(
                            i_phase.type, particle_class.dim, dim
                        ),
                    )
                )
            if fibres and particle_class not in (CylindricalFiber, Matrix):
                problems.append(
                    (
                        here + ("type",),
                        "cylindrical fibres go with no other particles, and phase {0} "
                        "is of them".format(next(iter(fibres))),
                    )
                )
            if i_phase.inside is not None:
                outer = self.phases.get(i_phase.inside)
                if (
                    outer is None
                    or outer.particle_class is Matrix
                    or i_phase.inside == i_name
                ):
                    problems.append(
                        (
                            here + ("inside",),
                            "has to name another phase of particles",
                        )
                    )
            if particle_class is not Matrix and particle_class.dim == dim:
                problems += [
                    (here + j_loc, j_message)
                    for j_loc, j_message in i_phase.problems(self.rve_dimensions)
                ]
        directions = {
            i_phase.direction.value
            for i_phase in fibres.values()
            if isinstance(i_phase.direction, Fixed)
        }
        if len(directions) > 1 or any(
            not isinstance(i_phase.direction, Fixed) for i_phase in fibres.values()
        ):
            problems.append(
                (
                    ("phases",),
                    "the cylindrical fibres have to run in one direction, given as a "
                    "fixed value",
                )
            )
        # The simulation moves the fibres in the plane across them

        return problems

    def mesh_problems(self):
        """Give what is wrong with the meshes, taken with the phases and the RVE."""
        problems = []
        voxel = getattr(self.meshes, "voxel", None)
        if voxel is None:
            return problems
        if fibres_in(self.phases):
            problems.append(
                (
                    ("meshes", "voxel"),
                    "cylindrical fibres are not stamped on a grid of voxels; gmsh "
                    "meshes them",
                )
            )
        resolutions = voxel.n_voxels_dims or []
        if resolutions and isinstance(resolutions[0], int):
            resolutions = [resolutions]
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
            "post_proc": analyses_to_options(self.analyses),
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
