"""
Module containing the meshing jobs an input data file asks for.

An input data file asks for a discretisation by naming it, femsh for a finite element
mesh and rgmsh for a regular grid, and this module turns that request into the mesher
that produces the mesh and the writers the mesh is then handed to. It is the only place
that knows the keywords of the input file, so the meshers and the writers stay usable
without one, and the two branches of the program that used to build mesh generators
side by side now build them here.

Everything the input file can get wrong is settled while it is being read: the mesher
and every writer are resolved before the first sample is generated, so a name that does
not exist is refused straight away rather than after a mesh has been built.
"""

import os
import time
import traceback

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.meshing.mesher import get_mesher
from geommicgen.meshing.writers import sidecar_path, write_vtk_image, write_vtu
from geommicgen.translators.base import get_writer

DECK_MESHERS = {"femsh": "gmsh", "rgmsh": "voxel"}
# Correspondence between the discretisations an input data file names and the meshers
# registered to produce them

DEFAULT_MESH_FORMATS = ("links",)
# Formats a finite element mesh is written in when the input file does not say

DEFAULT_GRID_FORMATS = ("crate",)
# Formats a regular grid is written in when the input file does not say. Only the
# default differs between the two: what a mesh can be turned into is settled by the
# writer, which declares whether it needs the cells or the grid, not by which
# discretisation produced it -- a grid writes a LINKS deck perfectly well, from the
# cells it is built into

MESH_DIRECTORY = "meshes"
# Directory of a sample the meshes are written into


class MeshJob:
    """
    Class for one discretisation an input data file asks for.

    A job owns the mesher that produces the mesh and the writers the mesh is handed to.
    It runs them together and keeps the outcome, so that one discretisation failing does
    not stop the others: the program reports at the end which of them could not be
    produced, and ends unsuccessfully.

    Attributes
    ----------
    mesher: `.Mesher`
        Mesher that produces the mesh.

    writers: list
        Classes of the solver writers the mesh is handed to, beyond the standard output.

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

    trace: str
        Traceback of that error, as text.
    """

    def __init__(self, mesher, writers, base_name, description=None):
        """
        Initizalizer for the MeshJob Class.

        Parameters
        ----------
        mesher: `.Mesher`
            Mesher that produces the mesh.

        writers: list
            Classes of the solver writers the mesh is handed to.

        base_name: str
            Name of the files that are written, without any extension.

        description: str
            Name the job is reported under. Defaults to the one the mesher gives.
        """
        self.mesher = mesher
        self.writers = list(writers)
        self.base_name = base_name
        self.description = description or mesher.description
        self.time = None
        self.files = []
        self.error = None
        self.trace = None

    def run(self, microstructure, sample_dir, report=None):
        """
        Mesh a microstructure and write it in every format the job asks for.

        Whether the job succeeded is told by its *error* afterwards.

        Parameters
        ----------
        microstructure: `.Microstructure`
            Microstructure to be meshed.

        sample_dir: str
            Directory of the sample the meshes are written into.

        report: callable
            Called with the index of the particle that has been dealt with and the
            total number of particles.
        """
        start = time.time()
        result_dir = os.path.join(sample_dir, MESH_DIRECTORY)
        os.makedirs(result_dir, exist_ok=True)
        base_path = os.path.join(result_dir, self.base_name)
        try:
            mesh = self.mesher.mesh(microstructure, report=report)
            self.files = self.write(mesh, base_path)
        except Exception as error:  # pylint: disable=broad-except
            self.trace = traceback.format_exc()
            self.error = error.with_traceback(None)
            # The error is kept rather than raised, so that the discretisations asked
            # for beside this one still get their chance. The traceback is kept as text
            # and taken off the exception, because a live one holds every frame between
            # here and the raise, and through them the whole mesh, for as long as the
            # job is kept -- and the failed jobs are kept until the batch ends
        finally:
            self.time = time.time() - start

    def write(self, mesh, base_path):
        """
        Write a mesh in the standard format and with every writer the job asks for.

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

        Raises
        ------
        ValueError:
            If a writer would write over the standard output.
        """
        standard_path = base_path + (".vtu" if mesh.structured is None else ".vtk")
        targets = [
            (i_writer, base_path + i_writer.extension) for i_writer in self.writers
        ]
        over_standard = [
            i_writer.name for i_writer, i_path in targets if i_path == standard_path
        ]
        if over_standard:
            raise ValueError(
                "The format {0} writes {1}, which is where the standard output of this "
                "mesh goes as well. Ask for one or the other.".format(
                    ", ".join(over_standard), os.path.basename(standard_path)
                )
            )
        # Checked before anything is written, since the two would otherwise be written
        # one over the other and only the second would survive

        written = []
        if mesh.structured is None:
            write_vtu(mesh, standard_path)
            written += [standard_path, sidecar_path(standard_path)]
        else:
            write_vtk_image(mesh, standard_path)
            written.append(standard_path)
        # Every mesh is written in a standard format that a viewer reads, whichever
        # solver formats were asked for

        for i_writer, i_path in targets:
            written += i_writer().write(mesh, i_path)

        return written


def job_base_name(deck_name, label):
    """
    Name the files of a job after the input data file and what discretises it.

    Parameters
    ----------
    deck_name: str
        Name of the input data file the microstructure was generated from.

    label: str
        What tells this discretisation apart from another of the same microstructure:
        the element for a mesh, the number of voxels for a grid.

    Returns
    -------
    str
        Name of the files, for a writer to put its own extension on.
    """
    if deck_name:
        return "{0}_{1}".format(os.path.splitext(deck_name)[0], label)

    return label
    # Naming the files after the deck and the discretisation keeps two runs of the same
    # microstructure apart, which neither does on its own


def writers_from_options(options, defaults):
    """
    Resolve the writers a discretisation is to be written with.

    Parameters
    ----------
    options: dict
        Options given for the discretisation in the input data file.

    defaults: tuple
        Formats used when the keyword is absent.

    Returns
    -------
    list
        Classes of the writers.

    Raises
    ------
    ValueError:
        If a format is named that there is no writer for.
    """
    names = list(options.get("formats", None) or defaults)
    if options.get("write_msh", False) and "gmsh" not in names:
        names.append("gmsh")
        # The gmsh file is no longer written on the way to anything else, so it is
        # produced by asking for the format meshio writes it in

    return [get_writer(i_name) for i_name in names]


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
        If a discretisation is asked for that there is no mesher for, if a format is
        named that there is no writer for, or if an option is given that no longer does
        anything.
    """
    jobs = []
    for i_name in mesh_options:
        options = mesh_options[i_name]
        if i_name not in DECK_MESHERS:
            raise ValueError("Specified mesh {0} is not supported.".format(i_name))
        mesher_class = get_mesher(DECK_MESHERS[i_name])
        if i_name == "femsh":
            element_type = options["element_type"]
            jobs.append(
                MeshJob(
                    mesher_class(
                        mesh_size=options.get("mesh_size", None),
                        element_type=element_type,
                        elements_per_particle=options.get(
                            "elements_per_particle", None
                        ),
                    ),
                    writers_from_options(options, DEFAULT_MESH_FORMATS),
                    job_base_name(deck_name, element_type),
                )
            )
        else:
            if options.get("slice_dir", None) is not None:
                raise ValueError(
                    "Slice_Dir no longer does anything: it used to decide whether the "
                    "grid of a three dimensional microstructure was written at all, "
                    "and the grid is now always written. Remove it."
                )
            writers = writers_from_options(options, DEFAULT_GRID_FORMATS)
            for j_n_voxels_dims in options["n_voxels_dims"]:
                jobs.append(
                    MeshJob(
                        mesher_class(j_n_voxels_dims),
                        writers,
                        job_base_name(
                            deck_name,
                            "_".join(
                                str(int(i_size)) for i_size in j_n_voxels_dims
                            ),
                        ),
                    )
                )
        # The constructors genuinely differ, and a grid fans out into one job per
        # resolution, so the two are built apart; only the names are shared

    return jobs
