"""Package containing the translation of meshes into the formats solvers read."""

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.translators.base import (
    SolverWriter,
    available_writers,
    get_writer,
    register_writer,
)
from geommicgen.translators.crate import CrateWriter, grid_file_name
from geommicgen.translators.links import LinksWriter
from geommicgen.translators.meshio_writer import MeshioWriter, register_meshio_writers
# Importing the module registers a loader; the meshio formats are added, and meshio
# imported, only when a writer is first looked up

__all__ = [
    "CrateWriter",
    "LinksWriter",
    "MeshioWriter",
    "SolverWriter",
    "available_writers",
    "get_writer",
    "grid_file_name",
    "register_meshio_writers",
    "register_writer",
]
