"""
Module containing the reading and writing of microstructures in the YAML format.

This module provides a human readable, tool agnostic representation of a microstructure.
Only the geometry defining attributes of each particle are stored, since every derived
quantity, such as the rotation matrices, the radii and the volumes, is rebuilt by the
constructors of the particle classes when the file is read back.
"""

import datetime

import numpy as np
import yaml

# pylint: disable=import-error
from geommicgen import __version__

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.microstructure.microstructure import Microstructure
from geommicgen.microstructure.phase import (
    Phase,
    DiscreteDistribution,
    FixedValue,
    LogNormalDistribution,
    NormalDistribution,
    SpecifiedValue,
    UniformDistribution,
    VonMisesDistribution,
)

FORMAT_NAME = "geommicgen-microstructure"
# Identifier written in the header of every file produced by this module

FORMAT_VERSION = 1
# Version of the schema, so that readers can detect incompatible files


class _FlowMapping(dict):
    """Mapping that is dumped in the flow style, so that a record stays on one line."""


class _MicrostructureDumper(yaml.SafeDumper):
    """Dumper that keeps every particle record on a single line."""


def _represent_flow_mapping(dumper, data):
    """Represent a `._FlowMapping` in the flow style."""
    return dumper.represent_mapping("tag:yaml.org,2002:map", data, flow_style=True)


_MicrostructureDumper.add_representer(_FlowMapping, _represent_flow_mapping)
_MicrostructureDumper.add_representer(
    np.ndarray, lambda dumper, data: dumper.represent_data(data.tolist())
)
_MicrostructureDumper.add_multi_representer(
    np.generic, lambda dumper, data: dumper.represent_data(data.item())
)
# Registered on a private dumper so that the global safe dumper is left untouched. The
# numpy representers use the conversions numpy itself provides, so that no value has to
# be converted by hand before it is written

SHAPE_CLASSES = {
    i_type.__name__: i_type for i_type in Phase.phase_types.values()
}
# Correspondence between the name written in a record and the particle class


def _shape_parameters(particle):
    """
    Get the geometry defining parameters of a particle.

    Parameters
    ----------
    particle: `.Particle`
        Particle whose parameters are to be collected.

    Returns
    -------
    dict
        Dictionary of the form *{parameter_name: value}*.

    Raises
    ------
    ValueError:
        If the particle type is not supported by this format.
    """
    shape = type(particle).__name__
    if shape == "Disk":
        parameters = {"r": particle.major_axis / 2}
    elif shape == "CylindricalFiber":
        parameters = {
            "r": particle.major_axis / 2,
            "direction": int(particle.direction_fibers),
        }
    elif shape == "Ellipse":
        parameters = {
            "major_axis": particle.major_axis,
            "minor_axis": particle.minor_axis,
            "angle": particle.angle,
        }
    elif shape == "Sphere":
        parameters = {"r": particle.axis_1 / 2}
    elif shape == "Ellipsoid":
        parameters = {
            "axis_1": particle.axis_1,
            "axis_2": particle.axis_2,
            "axis_3": particle.axis_3,
            "rotation_axis": particle.rotation_axis,
            "angle": particle.angle,
        }
    elif shape == "Cylinder":
        parameters = {
            "r_cyl": particle.r_cyl,
            "length": particle.length,
            "azimuth_angle": particle.azimuth_angle,
            "polar_angle": particle.polar_angle,
        }
    else:
        raise ValueError(
            "The particle type {0} is not supported by the YAML format.".format(shape)
        )
    # Collecting only the attributes that define the geometry

    return parameters


DESCRIPTOR_FIELDS = {
    FixedValue: ("fixed", (("value", "value"),)),
    SpecifiedValue: ("specified", (("values", "array_vals"),)),
    NormalDistribution: ("normal", (("mean", "mean"), ("sigma", "sigma"))),
    LogNormalDistribution: ("lognormal", (("mean", "mean"), ("sigma", "sigma"))),
    UniformDistribution: ("uniform", (("low", "low"), ("high", "high"))),
    VonMisesDistribution: (
        "vonmises",
        (("kappa", "kappa"), ("loc", "loc"), ("scale", "scale")),
    ),
    DiscreteDistribution: (
        "discrete",
        (("values", "values"), ("probabilities", "probabilities")),
    ),
}
# Name written for every kind of descriptor and the attributes that describe it. The
# pairs differ only where the stored attribute is not named as the written key


def _descriptors_to_records(phase):
    """
    Convert the descriptors of a phase into plain dictionaries.

    Parameters
    ----------
    phase: `.Phase`
        Phase whose descriptors are to be converted.

    Returns
    -------
    dict
        Dictionary of the form *{descriptor_name: {parameter_name: value}}*.

    Raises
    ------
    ValueError:
        If a descriptor is of a kind this format cannot write.
    """
    records = {}
    for i_name, i_descriptor in phase.descriptors.items():
        if i_descriptor is None:
            continue
        fields = None
        for i_class in type(i_descriptor).__mro__:
            if i_class in DESCRIPTOR_FIELDS:
                fields = DESCRIPTOR_FIELDS[i_class]
                break
        if fields is None:
            raise ValueError(
                "The descriptor type {0} is not supported by the YAML "
                "format.".format(type(i_descriptor).__name__)
            )
        kind, attributes = fields
        record = {"distribution": kind}
        for i_key, i_attribute in attributes:
            record[i_key] = getattr(i_descriptor, i_attribute)
        records[i_name] = record
    # The kind is looked up along the inheritance chain, so a descriptor deriving from
    # a known one is written as the closest kind that is registered

    return records


def write_microstructure_yaml(microstructure, file_path, provenance=None):
    """
    Write a microstructure to a YAML file.

    Parameters
    ----------
    microstructure: `.Microstructure`
        Microstructure to be written.

    file_path: str
        Path of the file to be written.

    provenance: dict
        Additional information about the generation of the microstructure, such as the
        input data file and the random seed. Optional.
    """
    type_codes = {i_type: i_code for i_code, i_type in Phase.phase_types.items()}
    # Inverting the correspondence between phase type and phase type class

    document = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "rve_dims": microstructure.rve_dims.tolist()
        if hasattr(microstructure.rve_dims, "tolist")
        else list(microstructure.rve_dims),
        "dim": int(microstructure.dim),
        "periodic": True,
        "matrix_phase": microstructure.matrix_phase,
        "provenance": _provenance_record(microstructure, provenance),
        "phases": {},
        "particles": [],
    }

    for i_name, i_phase in microstructure.phases.items():
        record = {
            "type": i_phase.type.__name__,
            "type_code": type_codes[i_phase.type],
            "inner_phase": bool(i_phase.inner_phase),
            "outer_phase": (
                str(i_phase.outer_phase) if i_phase.inner_phase else None
            ),
        }
        descriptors = _descriptors_to_records(i_phase)
        if descriptors:
            record["descriptors"] = descriptors
        document["phases"][i_name] = record
    # Writing one record per phase, in the order in which they were added

    particles = microstructure.particles
    particle_ids = {id(i_particle): i_ind for i_ind, i_particle in enumerate(particles)}
    for i_ind, i_particle in enumerate(particles):
        record = {
            "id": i_ind,
            "phase": i_particle.phase,
            "shape": type(i_particle).__name__,
            "center": i_particle.position_center,
        }
        record.update(_shape_parameters(i_particle))
        parent = getattr(i_particle, "parent", None)
        if parent is not None and id(parent) in particle_ids:
            record["parent_id"] = particle_ids[id(parent)]
        document["particles"].append(_FlowMapping(record))
    # The identifier is the position in the particle list, which is the order the
    # meshers and the analyses iterate

    with open(file_path, "w") as yaml_file:
        yaml.dump(
            document,
            yaml_file,
            Dumper=_MicrostructureDumper,
            sort_keys=False,
            default_flow_style=None,
            width=200,
            allow_unicode=True,
        )
    # Each particle record is a mapping of scalars and flat lists, so the dumper keeps
    # it on a single line


def _provenance_record(microstructure, provenance):
    """
    Build the provenance record of a microstructure.

    Parameters
    ----------
    microstructure: `.Microstructure`
        Microstructure being written.

    provenance: dict
        Additional information supplied by the caller. Optional.

    Returns
    -------
    dict
        Dictionary with the provenance of the microstructure.
    """
    record = {
        "geommicgen_version": __version__,
        "generated": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
    }
    try:
        record["achieved_volume_fraction"] = float(microstructure.volume_fraction)
    except (AttributeError, TypeError, ZeroDivisionError):
        pass
    if microstructure.total_overlap is not None:
        record["total_overlap"] = float(microstructure.total_overlap)
    if provenance:
        record.update(provenance)
    # The volume fraction is a derived property and is informational only

    return record


def particle_from_record(record, rve_dims):
    """
    Build a particle from its record in a YAML file.

    Parameters
    ----------
    record: dict
        Record of the particle, as read from the file.

    rve_dims: list
        List containing the dimensions of the microstructure in each direction.

    Returns
    -------
    `.Particle`
        The particle described by the record.

    Raises
    ------
    ValueError:
        If the shape of the particle is not supported, or if the number of coordinates
        of the centre does not match the dimension of the particle.
    """
    shape = record["shape"]
    if shape in ("Disk", "Sphere"):
        descriptors = {"r": record["r"]}
    elif shape == "CylindricalFiber":
        descriptors = {"r": record["r"], "direction": int(record["direction"])}
    elif shape == "Ellipse":
        descriptors = {
            "major_axis": record["major_axis"],
            "minor_axis": record["minor_axis"],
            "angle": record["angle"],
        }
    elif shape == "Ellipsoid":
        rotation_axis = record["rotation_axis"]
        descriptors = {
            "axis_1": record["axis_1"],
            "axis_2": record["axis_2"],
            "axis_3": record["axis_3"],
            "angle": record["angle"],
            "rot_axis_comp_x": rotation_axis[0],
            "rot_axis_comp_y": rotation_axis[1],
            "rot_axis_comp_z": rotation_axis[2],
        }
    elif shape == "Cylinder":
        descriptors = {
            "r_cyl": record["r_cyl"],
            "length": record["length"],
            "azimuth_angle": record["azimuth_angle"],
            "polar_angle": record["polar_angle"],
        }
    else:
        raise ValueError(
            "The particle shape {0} is not supported by the YAML format.".format(shape)
        )
    # A fresh dictionary is built for every particle because some of the constructors
    # remove entries from the one they are given

    particle = SHAPE_CLASSES[shape](record["phase"], descriptors, list(rve_dims))
    center = np.asarray(record["center"], dtype=float)
    if len(center) != particle.dim:
        raise ValueError(
            "The centre of particle {0} has {1} coordinates but the particle is "
            "{2}-dimensional.".format(record.get("id"), len(center), particle.dim)
        )
    particle.position_center = center
    particle.delta = 0
    # The dilation offset is a quantity of the generation and is never stored

    return particle


def read_microstructure_yaml(file_path):
    """
    Read a microstructure from a YAML file.

    Parameters
    ----------
    file_path: str
        Path of the file to be read.

    Returns
    -------
    `.Microstructure`
        The microstructure described by the file.

    Raises
    ------
    ValueError:
        If the file is not in the expected format or version.
    """
    with open(file_path, "r") as yaml_file:
        document = yaml.safe_load(yaml_file)

    if document.get("format") != FORMAT_NAME:
        raise ValueError(
            "The file {0} is not a {1} file.".format(file_path, FORMAT_NAME)
        )
    if document.get("version") != FORMAT_VERSION:
        raise ValueError(
            "Unsupported {0} version {1}, expected {2}.".format(
                FORMAT_NAME, document.get("version"), FORMAT_VERSION
            )
        )

    rve_dims = [float(i_dim) for i_dim in document["rve_dims"]]
    microstructure = Microstructure(rve_dims)

    for i_name, i_record in document["phases"].items():
        phase = Phase.from_type(
            i_name,
            Phase.phase_types[i_record["type_code"]],
            inner_phase=i_record.get("inner_phase", False),
            outer_phase=i_record.get("outer_phase"),
            descriptors=_descriptors_from_records(i_record.get("descriptors", {})),
        )
        microstructure.add_phase(phase)
    # Adding the phases in file order keeps the ordering of the particle list and lets
    # add_phase apply its own consistency checks

    particles = []
    for i_record in document["particles"]:
        particle = particle_from_record(i_record, rve_dims)
        microstructure.phases[i_record["phase"]].particles.append(particle)
        particles.append(particle)

    for i_record, i_particle in zip(document["particles"], particles):
        if i_record.get("parent_id") is not None:
            i_particle.parent = particles[i_record["parent_id"]]
    # The parents are resolved only after every particle exists

    return microstructure


def _descriptors_from_records(records):
    """
    Rebuild the descriptors of a phase from their records.

    Parameters
    ----------
    records: dict
        Dictionary of the form *{descriptor_name: {parameter_name: value}}*.

    Returns
    -------
    dict
        Dictionary of the form *{descriptor_name: `.PhaseDescriptor`}*.
    """
    builders = {
        "fixed": lambda name, rec: FixedValue(name, rec["value"]),
        "specified": lambda name, rec: SpecifiedValue(name, rec["values"]),
        "normal": lambda name, rec: NormalDistribution(
            name, rec["mean"], rec["sigma"]
        ),
        "lognormal": lambda name, rec: LogNormalDistribution(
            name, rec["mean"], rec["sigma"]
        ),
        "uniform": lambda name, rec: UniformDistribution(
            name, rec["low"], rec["high"]
        ),
        "vonmises": lambda name, rec: VonMisesDistribution(
            name, rec["kappa"], rec["loc"], rec["scale"]
        ),
        "discrete": lambda name, rec: DiscreteDistribution(
            name, rec["values"], rec["probabilities"]
        ),
    }

    descriptors = {}
    for i_name, i_record in records.items():
        distribution = i_record.get("distribution", "fixed")
        if distribution not in builders:
            raise ValueError(
                "The distribution {0} of descriptor {1} is not supported.".format(
                    distribution, i_name
                )
            )
        descriptors[i_name] = builders[distribution](i_name, i_record)

    return descriptors
