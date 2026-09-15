"""
Module containing the pipeline that joins the meshers to the writers.

A `.MeshJob` holds the mesher that produces a mesh and the writers the mesh is then
handed to, and runs them together, so that this is the one place a mesh passes from
the second stage to the third. The meshers and the writers know nothing of each other,
and nothing of the input file: an input data file asks for a discretisation by naming
it, femsh for a finite element mesh and rgmsh for a regular grid, and `build_mesh_jobs`
turns that request into the jobs; the command line builds the same jobs from its
arguments.

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
from geommicgen.meshing.writers import standard_mesh_path, write_standard_mesh
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
        Solver writers the mesh is handed to, beyond the standard output.

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

    def __init__(self, mesher, writers, base_name):
        """
        Initizalizer for the MeshJob Class.

        Parameters
        ----------
        mesher: `.Mesher`
            Mesher that produces the mesh.

        writers: list
            Solver writers the mesh is handed to.

        base_name: str
            Name of the files that are written, without any extension.
        """
        self.mesher = mesher
        self.writers = list(writers)
        self.base_name = base_name
        self.description = mesher.description
        self.time = None
        self.files = []
        self.error = None
        self.trace = None

    def run(self, microstructure, result_dir, report=None):
        """
        Mesh a microstructure and write it in every format the job asks for.

        Whether the job succeeded is told by its *error* afterwards.

        Parameters
        ----------
        microstructure: `.Microstructure`
            Microstructure to be meshed.

        result_dir: str
            Directory the files are written into. The caller chooses it, so that a run
            of the program can put them under the sample and the command line can put
            them where it was told to.

        report: callable
            Called with the index of the particle that has been dealt with and the
            total number of particles.
        """
        start = time.time()
        self.files = []
        os.makedirs(result_dir, exist_ok=True)
        base_path = os.path.join(result_dir, self.base_name)
        try:
            mesh = self.mesher.mesh(microstructure, report=report)
            self.write(mesh, base_path)
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

    def summary(self):
        """
        Say in one line what became of the discretisation.

        Returns
        -------
        str
            The name of the files, what discretised them, and the outcome.
        """
        if self.time is None:
            outcome = "not run"
        elif self.error is None:
            outcome = "ok"
        else:
            outcome = "failed: {0}: {1}".format(
                type(self.error).__name__, self.error
            )
        # A job that has not run has no error either, so the absence of one is not on
        # its own the same as having succeeded

        return "{0} ({1}): {2}".format(self.base_name, self.description, outcome)

    def write(self, mesh, base_path):
        """
        Write a mesh in the standard format and with every writer the job asks for.

        Parameters
        ----------
        mesh: `.Mesh`
            Mesh to be written.

        base_path: str
            Path of the files to be written, without any extension.

        Every file is recorded on the job as soon as it is written, so that a format
        failing part of the way through still reports what did reach the disk.

        Raises
        ------
        ValueError:
            If a writer would write over the standard output.
        """
        refuse_overwrites(
            base_path, self.writers, [standard_mesh_path(mesh, base_path)]
        )
        self.files += write_standard_mesh(mesh, base_path)
        # Every mesh is written in a standard format that a viewer reads, whichever
        # other formats were asked for. With its sidecar that file is the whole of the
        # second stage: it reads back as the same kind of mesh it was written from, so
        # the third stage runs off it alone, later or somewhere else

        write_formats(mesh, base_path, self.writers, written=self.files)


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


def refuse_overwrites(base_path, writers, protected):
    """
    Refuse a writer aimed at a file this run needs as it is.

    Parameters
    ----------
    base_path: str
        Path of the files to be written, without an extension.

    writers: list
        The writers.

    protected: list
        Paths no writer may be aimed at: the standard output of the mesh, or the file
        the mesh was read from.

    Raises
    ------
    ValueError:
        If a writer would write over one of them.
    """
    guarded = {os.path.abspath(i_path) for i_path in protected}
    over = [
        (i_writer.name, base_path + i_writer.extension)
        for i_writer in writers
        if os.path.abspath(base_path + i_writer.extension) in guarded
    ]
    if over:
        raise ValueError(
            "The format {0} writes {1}, which this run needs as it is. Ask for it "
            "under another name, or somewhere else.".format(
                ", ".join(i_name for i_name, _ in over),
                ", ".join(os.path.basename(i_path) for _, i_path in over),
            )
        )
    # Asked before anything is written, so that a request that cannot be honoured
    # leaves nothing behind rather than half of it


def write_formats(mesh, base_path, writers, protected=(), written=None):
    """
    Write a mesh with every writer.

    Parameters
    ----------
    mesh: `.Mesh`
        Mesh to be written.

    base_path: str
        Path of the files, without an extension.

    writers: list
        The writers.

    protected: list
        Paths no writer may be aimed at.

    written: list
        List the paths are appended to as they are written, so that a format failing
        part of the way through still leaves the caller with what did reach the disk.

    Returns
    -------
    list
        Paths of the files that were written.
    """
    refuse_overwrites(base_path, writers, protected)
    written = [] if written is None else written
    for i_writer in writers:
        written += i_writer.write(mesh, base_path + i_writer.extension)

    return written


def writers_from_options(options, defaults):
    """
    Resolve the writers a discretisation is to be written with.

    Parameters
    ----------
    options: dict
        Options given for the discretisation in the input data file.

    defaults: tuple
        Formats used when the keyword is absent. An empty list is not absent: it asks
        for the standard output and nothing else, which is the second stage on its own.

    Returns
    -------
    list
        The writers, each built from the options the discretisation was given.

    Raises
    ------
    ValueError:
        If a format is named that there is no writer for, or if an option a writer does
        understand is not a value it accepts.
    """
    names = options.get("formats", None)
    names = list(defaults) if names is None else list(names)
    if options.get("write_msh", False) and "gmsh" not in names:
        names.append("gmsh")
        # The gmsh file is no longer written on the way to anything else, so it is
        # produced by asking for the format meshio writes it in

    return [get_writer(i_name).from_options(options) for i_name in names]
    # Each writer takes from the options what it understands, so a keyword meant for
    # one format costs the others nothing


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
    for i_name, i_options in mesh_options.items():
        if i_name not in DECK_MESHERS:
            raise ValueError("Specified mesh {0} is not supported.".format(i_name))
        mesher_class = get_mesher(DECK_MESHERS[i_name])

        if i_name == "femsh":
            element_type = i_options["element_type"]
            mesher = mesher_class(
                mesh_size=i_options.get("mesh_size"),
                element_type=element_type,
                elements_per_particle=i_options.get("elements_per_particle"),
            )
            jobs.append(
                MeshJob(
                    mesher,
                    writers_from_options(i_options, DEFAULT_MESH_FORMATS),
                    job_base_name(deck_name, element_type),
                )
            )
            continue
        # The constructors genuinely differ, and a grid fans out into one job per
        # resolution, so the two are built apart; only the names are shared

        if i_options.get("slice_dir") is not None:
            raise ValueError(
                "Slice_Dir no longer does anything: it used to decide whether the "
                "grid of a three dimensional microstructure was written at all, "
                "and the grid is now always written. Remove it."
            )
        writers = writers_from_options(i_options, DEFAULT_GRID_FORMATS)
        voxel_filename = i_options.get("voxel_filename")
        if voxel_filename and len(i_options["n_voxels_dims"]) > 1:
            raise ValueError(
                "Voxel_Filename names one grid, and {0} resolutions were asked for. "
                "Remove it, and the grids are named after the input data file and the "
                "number of voxels, which tells them apart.".format(
                    len(i_options["n_voxels_dims"])
                )
            )
        # Refused rather than resolved, because every resolution would otherwise be
        # written over the one before it and the run would end with the last alone

        for j_n_voxels_dims in i_options["n_voxels_dims"]:
            label = "_".join(str(int(i_size)) for i_size in j_n_voxels_dims)
            jobs.append(
                MeshJob(
                    mesher_class(j_n_voxels_dims),
                    writers,
                    voxel_filename or job_base_name(deck_name, label),
                )
            )

    return jobs
