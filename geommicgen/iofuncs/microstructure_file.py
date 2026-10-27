"""
Module containing the reading and writing of microstructure files.

This module provides a human readable, tool agnostic representation of a microstructure.
Only the geometry defining attributes of each particle are stored, since every derived
quantity, such as the rotation matrices, the radii and the volumes, is rebuilt by the
constructors of the particle classes when the file is read back.

The file is JSON, one particle to a line. It was YAML, and a file written then is still
read: the document is the same, only its syntax differs. JSON is what a machine writes
for machines, which this file is: every language reads it with its own library, and
Python reads a microstructure of twenty thousand ellipsoids in a twentieth of a second
where PyYAML took fifteen. A number is also a number in it however it is spelt, where
YAML 1.1 reads 1e-3 as a string.
"""

import datetime
import json
import os

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
# Version of the schema, so that readers can detect incompatible files. The schema did
# not change when the syntax went from YAML to JSON

YAML_EXTENSIONS = (".yaml", ".yml")
# Extensions of the files written before the syntax was JSON, which are read as YAML


def _plain(value):
    """
    Give a numpy value as the Python value JSON writes.

    Parameters
    ----------
    value: object
        Value the JSON encoder does not know.

    Returns
    -------
    object
        The value as a list or as a Python scalar.

    Raises
    ------
    TypeError:
        If the value is not a numpy one.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(
        "A value of type {0} cannot be written to a microstructure file.".format(
            type(value).__name__
        )
    )
    # The conversions numpy itself provides, so that no value has to be converted by
    # hand before it is written


def _json(value):
    """Write a value as JSON on one line."""
    return json.dumps(value, default=_plain, ensure_ascii=False)


def _json_document(document):
    """
    Write a document as JSON, with every phase and every particle on a line of its own.

    Parameters
    ----------
    document: dict
        Document to be written.

    Returns
    -------
    str
        The JSON text.
    """
    members = []
    for i_key, i_value in document.items():
        if i_key == "phases" and i_value:
            value = (
                "{\n"
                + ",\n".join(
                    "    {0}: {1}".format(_json(j_name), _json(j_record))
                    for j_name, j_record in i_value.items()
                )
                + "\n  }"
            )
        elif i_key == "particles" and i_value:
            value = (
                "[\n"
                + ",\n".join("    " + _json(j_record) for j_record in i_value)
                + "\n  ]"
            )
        else:
            value = _json(i_value)
        members.append("  {0}: {1}".format(_json(i_key), value))

    return "{\n" + ",\n".join(members) + "\n}\n"
    # One record to a line keeps the file as readable, and as easy to compare between
    # two runs, as it was in YAML; json.dumps with an indent would put every coordinate
    # of every particle on a line of its own


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
            "The particle type {0} cannot be written to a microstructure file.".format(
                shape
            )
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
                "The descriptor type {0} cannot be written to a microstructure "
                "file.".format(type(i_descriptor).__name__)
            )
        kind, attributes = fields
        record = {"distribution": kind}
        for i_key, i_attribute in attributes:
            record[i_key] = getattr(i_descriptor, i_attribute)
        records[i_name] = record
    # The kind is looked up along the inheritance chain, so a descriptor deriving from
    # a known one is written as the closest kind that is registered

    return records


def write_microstructure_file(microstructure, file_path, provenance=None):
    """
    Write a microstructure to a microstructure file, in JSON.

    Parameters
    ----------
    microstructure: `.Microstructure`
        Microstructure to be written.

    file_path: str
        Path of the file to be written.

    provenance: dict
        Additional information about the generation of the microstructure, such as the
        input data file and the random seed. Optional.

    Raises
    ------
    ValueError:
        If the file is named as a YAML one.
    """
    if os.path.splitext(file_path)[1].lower() in YAML_EXTENSIONS:
        raise ValueError(
            "{0} is named as a YAML file, and microstructure files are written in "
            "JSON. Name it .json instead.".format(file_path)
        )
    # The reader takes the extension at its word, and JSON read as YAML 1.1 turns a
    # number spelt 1e-05, which JSON writes for a small one, into a string

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
        document["particles"].append(record)
    # The identifier is the position in the particle list, which is the order the
    # meshers and the analyses iterate

    text = _json_document(document)
    with open(file_path, "w", encoding="utf-8") as mic_file:
        mic_file.write(text)
    # The whole text is made before the file is opened, so a value that cannot be
    # written leaves no half written file behind


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
    Build a particle from its record in a microstructure file.

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
            "The particle shape {0} is not one a microstructure file holds.".format(
                shape
            )
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


def read_microstructure_file(file_path):
    """
    Read a microstructure from a microstructure file.

    The file is read as JSON, unless its extension says it was written as YAML, before
    the syntax changed.

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
    with open(file_path, "r", encoding="utf-8") as mic_file:
        if os.path.splitext(file_path)[1].lower() in YAML_EXTENSIONS:
            document = yaml.safe_load(mic_file)
        else:
            document = json.load(mic_file)

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
