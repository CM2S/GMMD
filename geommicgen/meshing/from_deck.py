"""
Module containing the meshing jobs an input data file asks for.

An input data file asks for a discretisation by naming it, femsh for a finite element
mesh and rgmsh for a regular grid, and this module turns that request into the mesher
that produces the mesh and the writers the mesh is then handed to. It is the only place
that knows the keywords of the input file, so the meshers and the writers stay usable
without one, and the two branches of the program that used to build mesh generators
side by side now build them here.
"""

import os
import time

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.meshing.gmsh_mesher import GmshMesher
from geommicgen.meshing.voxel_mesher import VoxelMesher
from geommicgen.meshing.writers import write_vtk_image, write_vtu
from geommicgen.translators.base import get_writer
from geommicgen.translators.crate import grid_file_name

DEFAULT_SOLVER_FORMATS = ("links",)
# Formats a finite element mesh is written in when the input file does not say

DEFAULT_VOXEL_FORMATS = ("crate",)
# Formats a regular grid is written in when the input file does not say

MESH_DIRECTORY = "meshes"
# Directory of a sample the meshes are written into


class MeshJob:
    """
    Class for one discretisation an input data file asks for.

    A job owns the mesher that produces the mesh and the names of the formats the mesh
    is written in. It runs them together and keeps the outcome, so that one
    discretisation failing does not stop the others: the sample records what happened
    and the program reports it at the end.

    Attributes
    ----------
    mesher: `.Mesher`
        Mesher that produces the mesh.

    formats: list
        Names of the solver formats the mesh is written in, beyond the standard one.

    base_name: str
        Name of the files that are written, without any extension.

    description: str
        Name the job is reported under in the summary of execution times.

    time: float
        Seconds the job took, or None if it has not run.

    files: list
        Paths of the files that were written.

    error: Exception
        Error that stopped the job, or None if it succeeded or has not run.
    """

    def __init__(self, mesher, formats, base_name, description):
        """
        Initizalizer for the MeshJob Class.

        Parameters
        ----------
        mesher: `.Mesher`
            Mesher that produces the mesh.

        formats: list
            Names of the solver formats the mesh is written in.

        base_name: str
            Name of the files that are written, without any extension.

        description: str
            Name the job is reported under in the summary of execution times.
        """
        self.mesher = mesher
        self.formats = list(formats)
        self.base_name = base_name
        self.description = description
        self.time = None
        self.files = []
        self.error = None

    def run(self, microstructure, sample_dir, report=None):
        """
        Mesh a microstructure and write it in every format the job asks for.

        Parameters
        ----------
        microstructure: `.Microstructure`
            Microstructure to be meshed.

        sample_dir: str
            Directory of the sample the meshes are written into.

        report: callable
            Called with the index of the particle that has been dealt with and the
            total number of particles.

        Returns
        -------
        bool
            Whether the job succeeded.
        """
        start = time.time()
        result_dir = os.path.join(sample_dir, MESH_DIRECTORY)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        base_path = os.path.join(result_dir, self.base_name)
        try:
            mesh = self.mesher.mesh(microstructure, report=report)
            self.files = self.write(mesh, base_path)
        except Exception as error:
            self.error = error
            self.time = time.time() - start

            return False
        # The error is kept rather than raised, so that the discretisations that were
        # also asked for still get their chance and the program can report every one of
        # them at the end

        self.time = time.time() - start

        return True

    def write(self, mesh, base_path):
        """
        Write a mesh in the standard format and in every format the job asks for.

        Parameters
        ----------
        mesh: `.Mesh`
            Mesh to be written.

        base_path: str
            Path of the files to be written, without any extension.

        Returns
        -------
        list
            Paths of the files that were written.
        """
        written = []
        if mesh.structured is None:
            write_vtu(mesh, base_path + ".vtu")
            written += [base_path + ".vtu", base_path + ".mesh.json"]
        else:
            write_vtk_image(mesh, base_path + ".vtk")
            written.append(base_path + ".vtk")
        # Every mesh is written in a standard format that a viewer reads, whichever
        # solver formats were asked for

        for i_format in self.formats:
            writer_class = get_writer(i_format)
            written += writer_class().write(
                mesh, base_path + writer_class.extension
            )

        return written


def formats_from_options(options, key, defaults, write_msh):
    """
    Read the formats a discretisation is to be written in.

    Parameters
    ----------
    options: dict
        Options given for the discretisation in the input data file.

    key: str
        Keyword holding the formats.

    defaults: tuple
        Formats used when the keyword is absent.

    write_msh: bool
        Whether the mesh is also written as a gmsh file.

    Returns
    -------
    list
        Names of the formats.
    """
    requested = options.get(key, None)
    formats = list(defaults) if requested is None else [
        i_format.strip() for i_format in requested.split(",") if i_format.strip()
    ]
    if write_msh and "gmsh" not in formats:
        formats.append("gmsh")
        # The gmsh file is no longer written on the way to anything else, so it is asked
        # for like any other format

    return formats


def build_mesh_jobs(mesh_options, deck_name=None):
    """
    Build the meshing jobs an input data file asks for.

    Parameters
    ----------
    mesh_options: dict
        Mesh options read from the input data file, keyed by the name of the
        discretisation.

    deck_name: str
        Name of the input data file, used to name the grids of a regular mesh.

    Returns
    -------
    list
        The `.MeshJob` objects to run.

    Raises
    ------
    ValueError:
        If a discretisation is asked for that there is no mesher for, or if an option
        is given that no longer does anything.
    """
    jobs = []
    for i_name in mesh_options:
        options = mesh_options[i_name]
        write_msh = options.get("write_msh", False)
        if i_name == "femsh":
            jobs.append(
                MeshJob(
                    GmshMesher(
                        mesh_size=options.get("mesh_size", None),
                        element_type=options["element_type"],
                        elements_per_particle=options.get(
                            "elements_per_particle", None
                        ),
                    ),
                    formats_from_options(
                        options, "solver_formats", DEFAULT_SOLVER_FORMATS, write_msh
                    ),
                    "femsh",
                    "Finite element mesh generation",
                )
            )
        elif i_name == "rgmsh":
            if options.get("slice_dir", None) is not None:
                raise ValueError(
                    "Slice_Dir no longer does anything: it used to decide whether the "
                    "grid of a three dimensional microstructure was written at all, "
                    "and the grid is now always written. Remove it."
                )
            for j_n_voxels_dims in options["n_voxels_dims"]:
                jobs.append(
                    MeshJob(
                        VoxelMesher(j_n_voxels_dims),
                        formats_from_options(
                            options, "voxel_formats", DEFAULT_VOXEL_FORMATS, write_msh
                        ),
                        os.path.splitext(
                            grid_file_name(deck_name or "", j_n_voxels_dims)
                        )[0],
                        "Regular mesh generation",
                    )
                )
                # The extension belongs to the writer, so the name of the grid is taken
                # without the one grid_file_name puts on it
        else:
            raise ValueError("Specified mesh {0} is not supported.".format(i_name))

    return jobs
