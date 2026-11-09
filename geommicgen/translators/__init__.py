"""Package containing the translation of meshes into the formats solvers read."""

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.translators.base import (
    SolverWriter,
    available_writers,
    get_writer,
    register_writer,
    writer_options,
)
from geommicgen.translators.abaqus import AbaqusWriter
from geommicgen.translators.crate import CrateWriter
from geommicgen.translators.links import LinksWriter
from geommicgen.translators.meshio_writer import MeshioWriter, register_meshio_writers
# Importing the module registers a loader; the meshio formats are added, and meshio
# imported, only when a writer is first looked up

__all__ = [
    "AbaqusWriter",
    "CrateWriter",
    "LinksWriter",
    "MeshioWriter",
    "SolverWriter",
    "available_writers",
    "get_writer",
    "register_meshio_writers",
    "register_writer",
    "writer_options",
]
