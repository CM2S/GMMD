"""
Module containing the unstructured mesher built on gmsh.

The microstructure is built as a CAD model, one primitive per particle and per periodic
image of a particle, cut against the box of the RVE and fragmented so that the matrix
and the particles share their interfaces. Opposite faces of the box are declared
periodic before meshing, which is what makes the discretisations of the two sides match
node for node.

The mesh is read out of the gmsh session in memory. Nothing is written and nothing is
parsed back, so the mesher produces the same `.Mesh` as any other and the formats a
solver reads are the concern of `geommicgen.translators`.

Notes
-----
The descriptors of the elements in `ELEMENT_DESCRIPTORS` and their possible values are

dim: int
    Dimension of the element.

mesh_alg: int
    2D meshing algorithm. 1 Mesh Adapt, 2 Automatic, 5 Delaunay (default), 6
    Frontal-Delaunay, 7 BAMG, 8 Frontal-Delaunay for Quads, 9 Packing of
    Parallelograms.

mesh_alg_3d: int
    3D meshing algorithm. 1 Delaunay (default), 2 Frontal, 7 MMG3D, 9 R-tree, 10 HXT.

force_recomb_all_surf: {0, 1}
    Force the recombination of all surfaces.

force_recomb_all_vol: {0, 1}
    Force the recombination of all volumes.

element_order: int
    Order of the element.

recomb_alg: int
    Quad/Hex recombination algorithm. 0 simple, 1 blossom (default), 2 simple
    full-quad, 3 blossom full-quad.

recomb_alg_3d: int
    Recombination level in 3D. 0 hex (default), 1 hex + prisms, 2 hex + prisms +
    pyramids.

recombine_3d_conformity: int
    Recombination conformity in 3D. 0 nonconforming (default), 1 trihedra, 2 pyramids
    + trihedra, 3 pyramids + hexSplit + trihedra, 4 hexSplit + trihedra.

element_order_incomp: {0, 1}
    Second order incomplete elements.
"""

import contextlib
import itertools

import numpy as np

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen._optional import require_gmsh
from geommicgen._process import in_own_process
from geommicgen.errors.error_classes import UnsupportedParticleShape
from geommicgen.meshing.images import periodic_images
from geommicgen.meshing.mesh import Mesh
from geommicgen.meshing.mesher import Mesher, register_mesher
from geommicgen.microstructure.microstructure import unit_scale
from geommicgen.microstructure.particleclasses import (
    Cylinder,
    CylindricalFiber,
    Disk,
    Ellipse,
    Ellipsoid,
    Sphere,
)

ELEMENT_DESCRIPTORS = {
    "tri3": {
        "dim": 2,
        "mesh_alg": 5,
        "force_recomb_all_surf": 0,
        "element_order": 1,
        "recomb_alg": 1,
        "element_order_incomp": 0,
    },
    "tri6": {
        "dim": 2,
        "mesh_alg": 5,
        "force_recomb_all_surf": 0,
        "element_order": 2,
        "recomb_alg": 1,
        "element_order_incomp": 0,
    },
    "quad4": {
        "dim": 2,
        "mesh_alg": 5,
        "force_recomb_all_surf": 1,
        "element_order": 1,
        "recomb_alg": 1,
        "element_order_incomp": 0,
    },
    "quad8": {
        "dim": 2,
        "mesh_alg": 5,
        "force_recomb_all_surf": 1,
        "element_order": 2,
        "recomb_alg": 1,
        "element_order_incomp": 1,
    },
    "tetra4": {
        "dim": 3,
        "mesh_alg": 5,
        "mesh_alg_3d": 1,
        "force_recomb_all_surf": 0,
        "element_order": 1,
        "recomb_alg": 1,
        "element_order_incomp": 0,
    },
    "tetra10": {
        "dim": 3,
        "mesh_alg": 5,
        "mesh_alg_3d": 1,
        "force_recomb_all_surf": 0,
        "element_order": 2,
        "recomb_alg": 1,
        "element_order_incomp": 0,
    },
}
# Options of gmsh that produce each element this mesher offers

GMSH_CELL_TYPES = {
    2: "triangle",
    3: "quad",
    4: "tetra",
    9: "triangle6",
    11: "tetra10",
    16: "quad8",
}
# Correspondence between the element types of gmsh and the names meshio uses. Only the
# types the elements of `ELEMENT_DESCRIPTORS` produce are listed, so an element type
# that is added there without its entry here fails loudly on the first mesh

GMSH_TO_VTK_ORDER = {"tetra10": [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]}
# Permutations taking the nodes of an element from the order gmsh lists them in to the
# order VTK expects. The elements that are not named here are listed alike by both

FILL_TOLERANCE = 1.0e-6
# Relative tolerance on the area or volume the cells of a mesh cover, and on how far a
# node may lie outside the RVE. A mesh that fills the RVE covers it to the last few
# digits, unless OpenCASCADE has merged a particle within its own tolerance, 1e-7 of
# the model, onto a face; a piece of a particle lost is far larger than either

PHASE_VOLUME_TOLERANCE = 1.0e-2
# Relative tolerance on the volume the fragments of a phase take up against the volume
# of its particles. Both are measured by OpenCASCADE, which measures a disk, an ellipse,
# a fibre, a sphere and a cylinder, cut by the faces of the RVE or not, to a part in a
# million or better, but an ellipsoid, a sphere it has stretched, to half a per cent
# once a face has cut it

PBC_TOLERANCE = 1.0e-3
# Tolerance of the bounding boxes used to pair opposite faces of the RVE, along the
# faces, in the units of the model, which is built with the shortest side of the RVE one

PBC_PLANE_TOLERANCE = 1.0e-6
# Tolerance of the same boxes across the faces. An entity on a face has no extent across
# it, but for what OpenCASCADE merges within its own tolerance, 1e-7 of the model. The
# tolerance along the faces let in the cap a particle crossing a face by less than 1e-3
# leaves beside the opposite one, whose curved side was paired with the flat face of
# the cut and collapsed onto it, into cells of no area
# TODO: a cap thinner than this and thicker than what OpenCASCADE merges, from 1e-7 to
# 1e-6 of the model, is still paired so; a particle crosses a face by that little
# about once in a million times


@contextlib.contextmanager
def gmsh_session():
    """
    Open a gmsh session, and close it however the block that uses it ends.

    Yields
    ------
    module
        The gmsh module, initialized.
    """
    gmsh = require_gmsh()
    gmsh.initialize()
    try:
        yield gmsh
    finally:
        gmsh.finalize()


def failing_surfaces(error):
    """
    Read the surfaces gmsh could not mesh out of the error it raised.

    Gmsh names them in its message, for instance "Invalid boundary mesh (overlapping
    facets) on surface 75 surface 76". Those are the thin ligaments between particles
    that nearly touch.

    Parameters
    ----------
    error: Exception
        Error raised by gmsh.

    Returns
    -------
    list
        Tags of the surfaces named in the message.
    """
    words = str(error).split()

    return [
        int(i_tag)
        for i_previous, i_tag in zip(words, words[1:])
        if i_previous == "surface" and i_tag.isdigit()
    ]


def corner_measures(points, cell_type, connectivity):
    """
    Give the area or the volume of cells, with straight sides through their corners.

    The cells of a mesh tile its domain this way whatever their order, because two cells
    that share a side share the corners of it, and the faces of the RVE are flat.

    Parameters
    ----------
    points: array
        Coordinates of the nodes, one to a row.

    cell_type: str
        Name of the cells, as `GMSH_CELL_TYPES` gives it.

    connectivity: array
        Nodes of each cell, one cell to a row, the corners first.

    Returns
    -------
    array
        Area or volume of each cell.
    """
    corners = points[connectivity]
    if cell_type.startswith("tetra"):
        measures = np.abs(np.linalg.det(corners[:, 1:4] - corners[:, :1])) / 6
    else:
        n_corners = 4 if cell_type.startswith("quad") else 3
        x_corners = corners[:, :n_corners, 0]
        y_corners = corners[:, :n_corners, 1]
        measures = 0.5 * np.abs(
            np.sum(
                x_corners * np.roll(y_corners, -1, axis=1)
                - np.roll(x_corners, -1, axis=1) * y_corners,
                axis=1,
            )
        )
        # The shoelace formula, over the corners in the order they go around the cell

    return measures


def check_fills_rve(points, cells, rve_dims):
    """
    Refuse a mesh that does not fill its RVE, or that reaches outside of it.

    Parameters
    ----------
    points: array
        Coordinates of the nodes, one to a row.

    cells: list(tuple)
        Blocks of cells, each a name and a connectivity.

    rve_dims: list(float)
        Dimensions of the microstructure in each spatial direction.

    Raises
    ------
    ValueError:
        If the cells cover more or less than the RVE, or if a node lies outside it.
    """
    rve_dims = np.asarray(rve_dims, dtype=float)
    volume = np.prod(rve_dims)
    covered = sum(
        corner_measures(points, i_type, i_connectivity).sum()
        for i_type, i_connectivity in cells
    )
    coordinates = points[:, : len(rve_dims)]
    outside = np.any(coordinates < -FILL_TOLERANCE * rve_dims) or np.any(
        coordinates > (1 + FILL_TOLERANCE) * rve_dims
    )
    if abs(covered - volume) > FILL_TOLERANCE * volume or outside:
        raise ValueError(
            "The cells gmsh produced cover {0:.9g} of an RVE of {1:.9g}, off by a part "
            "in {2:.3g}{3}: part of the geometry was lost, and the mesh is not "
            "taken.".format(
                covered,
                volume,
                volume / max(abs(covered - volume), np.finfo(float).tiny),
                ", and reach outside it" if outside else "",
            )
        )


@in_own_process("meshing with gmsh")
def mesh_keeping_state(mesher, microstructure, report=None):
    """
    Mesh a microstructure with gmsh, giving with the mesh what was set on the mesher.

    Parameters
    ----------
    mesher: `.GmshMesher`
        Mesher that makes the mesh.

    microstructure: `.Microstructure`
        Microstructure to be meshed.

    report: callable
        Called with the index of the particle that was added to the model and the total
        number of particles.

    Returns
    -------
    tuple
        The mesh, and the attributes of the mesher.
    """
    try:
        mesh = mesher.mesh_in_this_process(microstructure, report)
    except Exception as error:
        error.mesher_state = vars(mesher)
        raise
    # The warnings are wanted most when meshing failed, so the error carries them

    return mesh, vars(mesher)


def phase_volume_warnings(factory, dim, materials, particle_volumes, rve_volume):
    """
    Report every phase whose fragments take up another volume than its particles.

    Parameters
    ----------
    factory: module
        The OpenCASCADE geometry kernel of gmsh, holding the fragments.

    dim: int
        Dimension of the fragments.

    materials: dict
        Tags of the fragments of each phase, by the name of the phase.

    particle_volumes: dict
        Volume the particles of each phase take up, by the name of the phase, less that
        of the particles placed inside them.

    rve_volume: float
        Volume of the RVE, in the units of the model.

    Returns
    -------
    list(str)
        One warning for each phase off by more than `PHASE_VOLUME_TOLERANCE`.
    """
    warnings = []
    for i_name, i_expected in particle_volumes.items():
        kept = sum(factory.getMass(dim, j_tag) for j_tag in materials[i_name])
        if abs(kept - i_expected) > PHASE_VOLUME_TOLERANCE * i_expected:
            warnings.append(
                "WARNING: phase {0} takes up {1:.4g} of the RVE where its particles "
                "take up {2:.4g}: part of a particle was lost to another phase, or the "
                "particles overlap.".format(
                    i_name, kept / rve_volume, i_expected / rve_volume
                )
            )

    return warnings


def resolution_label(element_type, mesh_size=None, elements_per_particle=None):
    """
    Name a discretisation after its element and the resolution asked for.

    Parameters
    ----------
    element_type: str
        Name of the element.

    mesh_size: float
        Largest element size, when one was asked for.

    elements_per_particle: float
        Number of elements across the smallest particle, when that was asked for.

    Returns
    -------
    str
        The element, then h and the size, then epp and the elements per particle, of
        those that were asked for: tri6_h0.05, tri6_epp4, tri6_h0.05_epp4.
    """
    parts = [element_type]
    if mesh_size is not None:
        parts.append("h{0:g}".format(mesh_size))
    if elements_per_particle is not None:
        parts.append("epp{0:g}".format(elements_per_particle))

    return "_".join(parts)
    # The files are named after the label, so a label of the element alone gave a
    # microstructure meshed at two sizes one name, and the second mesh was written over
    # the first without a word. What was asked for names it, rather than the size it
    # came to, which depends on the microstructure: one request names the files alike
    # across the samples of a run


@register_mesher
class GmshMesher(Mesher):
    """
    Class for the mesher that builds an unstructured mesh with gmsh.

    Attributes
    ----------
    mesh_size: float
        Largest element size, in the units of the RVE.

    elements_per_particle: float
        Number of elements across the smallest particle. When given, it sets the
        element size, since a size given as a length carries no relation to the
        microstructure.

    element_type: str
        Name of the element, one of the keys of `ELEMENT_DESCRIPTORS`.

    descriptors: dict
        Options of gmsh that produce the chosen element.

    max_attempts: int
        Number of times the model is built. A second attempt refines around the
        surfaces the first one could not mesh.

    warnings: list
        Messages about the resolution that was asked for, for the caller to report.
    """

    name = "gmsh"
    description = "Finite element mesh generation"
    default_formats = ("links",)
    options = {
        "Mesh_Size": {"type": "float_list", "help": "largest element size"},
        "Elements_Per_Particle": {
            "type": "float_list",
            "help": "elements across the smallest particle, instead of a size",
        },
        "Element_Type": {
            "type": "str_list",
            "help": "element to mesh with (default: tri3)",
        },
    }
    # Each takes several values, separated by commas on the command line and written
    # as a list in an input data file, and there is a mesh for every combination

    def __init__(
        self, mesh_size=None, element_type="tri3", elements_per_particle=None,
        max_attempts=2,
    ):
        """
        Initizalizer for the GmshMesher Class.

        Parameters
        ----------
        mesh_size: float
            Largest element size, in the units of the RVE.

        element_type: str
            Name of the element, one of the keys of `ELEMENT_DESCRIPTORS`.

        elements_per_particle: float
            Number of elements across the smallest particle.

        max_attempts: int
            Number of times the model is built.

        Raises
        ------
        ValueError:
            If neither size is given, if a size is not positive, or if the element is
            not one this mesher knows.
        """
        if mesh_size is None and elements_per_particle is None:
            raise ValueError(
                "Either mesh_size or elements_per_particle must be specified."
            )
        if mesh_size is not None and mesh_size < 0:
            raise ValueError("The mesh size must be a positive number.")
        if elements_per_particle is not None and elements_per_particle <= 0:
            raise ValueError("elements_per_particle must be a positive number.")
        if max_attempts < 1:
            raise ValueError("The mesher has to be allowed at least one attempt.")
        if element_type not in ELEMENT_DESCRIPTORS:
            raise ValueError("Unknown element: {0}".format(element_type))
        self.requested_mesh_size = mesh_size
        self.mesh_size = mesh_size
        self.elements_per_particle = elements_per_particle
        self.element_type = element_type
        self.descriptors = ELEMENT_DESCRIPTORS[element_type]
        self.max_attempts = max_attempts
        self.warnings = []
        self.label = resolution_label(element_type, mesh_size, elements_per_particle)

    @classmethod
    def from_options(cls, options):
        """
        Build one mesher for every element and every resolution the options ask for.

        Parameters
        ----------
        options: dict
            Options given for the discretisation. Each of *mesh_size*,
            *elements_per_particle* and *element_type* holds one value or a list of
            them.

        Returns
        -------
        list
            One mesher for every combination of the values given, in the order they
            were given, the same combination once.
        """
        given = {}
        names = [i_name.lower() for i_name in cls.options]
        for i_name in sorted(names, key=lambda i_name: i_name != "element_type"):
            value = options.get(i_name)
            values = [] if value is None else np.atleast_1d(value).tolist()
            if values:
                given[i_name] = list(dict.fromkeys(values))
        # One value is a list of one; an option that was not given, or given no value,
        # is left to the default of the initializer. A value given twice is one mesh,
        # since the second would be written over the first under the same name. The
        # element varies slowest, so the meshes of one element come together

        return [
            cls(**dict(zip(given, i_values)))
            for i_values in itertools.product(*given.values())
        ]
        # A size sweep, or two elements at one size, is one request: each mesh is named
        # after its element and resolution, so none is written over another

    def mesh(self, microstructure, report=None):
        """
        Build an unstructured mesh of a microstructure with gmsh, in its own process.

        Gmsh can end the process it runs in rather than raise, so it is run in one of
        its own, and the one that asked for the mesh is told with an error instead.

        Parameters
        ----------
        microstructure: `.Microstructure`
            Microstructure to be meshed.

        report: callable
            Called with the index of the particle that was added to the model and the
            total number of particles.

        Returns
        -------
        `.Mesh`
            The mesh, with the phase of every cell.

        Raises
        ------
        ValueError:
            If the element does not have the dimension of the microstructure.

        MissingOptionalDependency:
            If gmsh is not installed.

        ProcessDied:
            If the process the mesh was being made in ended without giving it.
        """
        self.refuse_the_microstructure(microstructure)
        require_gmsh()
        # Refused here rather than in the process that would mesh, before one is
        # started: a microstructure this mesher cannot mesh, whether gmsh is there or
        # not, and then a gmsh that is not there
        try:
            mesh, state = mesh_keeping_state(self, microstructure, report=report)
        except Exception as error:
            vars(self).update(vars(error).pop("mesher_state", {}))
            raise
        vars(self).update(state)
        # What meshing sets on the mesher, the warnings and the element size it settled
        # on, was set on the one in the other process

        return mesh

    def refuse_the_microstructure(self, microstructure):
        """
        Refuse a microstructure this mesher cannot mesh.

        Parameters
        ----------
        microstructure: `.Microstructure`
            Microstructure to be meshed.

        Raises
        ------
        ValueError:
            If the element does not have the dimension of the microstructure, or the
            microstructure has no matrix phase.
        """
        if self.descriptors["dim"] != len(microstructure.rve_dims):
            raise ValueError(
                "The element {0} has dimension {1} and the microstructure has "
                "dimension {2}.".format(
                    self.element_type,
                    self.descriptors["dim"],
                    len(microstructure.rve_dims),
                )
            )
        if microstructure.matrix_phase is None:
            raise ValueError(
                "The microstructure has no matrix phase to fill the RVE with."
            )
        # Checked before the model is built rather than being met as a missing key once
        # the whole geometry has been fragmented

    def mesh_in_this_process(self, microstructure, report=None):
        """
        Build an unstructured mesh of a microstructure with gmsh, in this process.

        See `mesh`, which calls this in a process of its own.
        """
        self.refuse_the_microstructure(microstructure)
        self.resolve_mesh_size(microstructure)
        scale = unit_scale(microstructure.rve_dims)
        unit_microstructure = microstructure.scaled(scale)
        # The model is built from the microstructure brought to a shortest side of one,
        # and the mesh read out of it is brought back. OpenCASCADE and gmsh work with
        # tolerances that are lengths, which no option reaches: built in the user's
        # units, a micrometre RVE lost its particles in the booleans or had its faces
        # paired with themselves, and a large one lost them too, or crashed the
        # process. The size of the elements and what the mesher reports of it stay in
        # the user's units
        refine_surfaces = []
        for i_attempt in range(self.max_attempts):
            with gmsh_session() as gmsh:
                try:
                    phase_groups = self.build_model(
                        gmsh, unit_microstructure, refine_surfaces, report, scale
                    )
                except Exception as error:
                    refine_surfaces = failing_surfaces(error)
                    if not refine_surfaces or i_attempt == self.max_attempts - 1:
                        raise
                    continue
                    # The model has to be rebuilt in a new session: once a meshing pass
                    # has failed, gmsh will not produce a mesh for that model again

                return self.extract_mesh(gmsh, microstructure, phase_groups, scale)
        # Reading the mesh back is deliberately outside the retry: a failure there is
        # not something refining a surface can fix, and its message can name a surface
        # too, which would have the real error retried away instead of raised

    @staticmethod
    def smallest_particle_radius(microstructure):
        """
        Give the smallest inscribed radius over the particles of a microstructure.

        This is the shortest half dimension present, so twice it is the thinnest
        particle a mesh has to resolve.

        Parameters
        ----------
        microstructure: `.Microstructure`
            Microstructure whose particles are measured.

        Returns
        -------
        float
            The smallest radius, or None when no particle reports one.
        """
        radii = []
        for i_particle in microstructure.particles:
            radius = getattr(i_particle, "radius_insc", None)
            if radius is None:
                radius = getattr(i_particle, "radius", None)
            if radius is not None:
                radii.append(float(radius))

        return min(radii) if radii else None

    def resolve_mesh_size(self, microstructure):
        """
        Turn a resolution given per particle into an element size, and flag inert sizes.

        A mesh size given as an absolute length carries no relation to the
        microstructure: change the RVE or the particle size and the same number means a
        different resolution. Worse, once it exceeds the particle size it stops doing
        anything at all, because the faceted geometry already forces a finer mesh, so a
        request of 0.4 and one of 0.15 can produce the identical mesh.

        Parameters
        ----------
        microstructure: `.Microstructure`
            Microstructure that is about to be meshed.
        """
        self.warnings = []
        self.mesh_size = self.requested_mesh_size
        # Restored from what was asked for, so that meshing a second microstructure
        # with the same mesher does not inherit the size derived for the first
        smallest_radius = self.smallest_particle_radius(microstructure)
        if smallest_radius is None or smallest_radius <= 0:
            return
        if self.elements_per_particle is not None:
            derived = 2 * smallest_radius / self.elements_per_particle
            self.mesh_size = (
                derived if self.mesh_size is None else min(self.mesh_size, derived)
            )
            self.warnings.append(
                "Element size {0:.4g} for {1:g} elements across the smallest particle "
                "({2:.4g} across)".format(
                    self.mesh_size, self.elements_per_particle, 2 * smallest_radius
                )
            )
        elif self.mesh_size > smallest_radius:
            self.warnings.append(
                "WARNING: mesh size {0:.4g} exceeds the smallest particle radius "
                "{1:.4g}, so it no longer controls the mesh; the geometry does. Use "
                "elements_per_particle to set the resolution.".format(
                    self.mesh_size, smallest_radius
                )
            )

    def set_options(self, gmsh):
        """
        Set the options of gmsh that produce the chosen element.

        Parameters
        ----------
        gmsh: module
            The gmsh module, in an initialized session.
        """
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.Algorithm", self.descriptors["mesh_alg"])
        gmsh.option.setNumber(
            "Mesh.Algorithm3D", self.descriptors.get("mesh_alg_3d", 1)
        )
        gmsh.option.setNumber("Mesh.MeshSizeFactor", 1)
        gmsh.option.setNumber("Mesh.MaxNumThreads1D", 1)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", 1)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", 1)
        # One thread, so that the same microstructure gives the same mesh on every run.
        # On four, gmsh gave meshes different in their last digits and in the order of
        # their cells from one run to the next, and the optimizer of second order
        # tetrahedra aborted the whole process on ellipsoids
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber(
            "Mesh.RecombinationAlgorithm", self.descriptors["recomb_alg"]
        )
        gmsh.option.setNumber(
            "Mesh.RecombineAll", self.descriptors["force_recomb_all_surf"]
        )
        gmsh.option.setNumber("Mesh.RecombineOptimizeTopology", 5)
        gmsh.option.setNumber(
            "Mesh.Recombine3DAll", self.descriptors.get("force_recomb_all_vol", 0)
        )
        gmsh.option.setNumber(
            "Mesh.Recombine3DLevel", self.descriptors.get("recomb_alg_3d", 0)
        )
        gmsh.option.setNumber(
            "Mesh.Recombine3DConformity",
            self.descriptors.get("recombine_3d_conformity", 0),
        )
        gmsh.option.setNumber("Mesh.Renumber", 1)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.option.setNumber("Mesh.Smoothing", 1)
        gmsh.option.setNumber("Mesh.ElementOrder", self.descriptors["element_order"])
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
        gmsh.option.setNumber(
            "Mesh.SecondOrderIncomplete", self.descriptors["element_order_incomp"]
        )

    def build_model(
        self, gmsh, microstructure, refine_surfaces=(), report=None, scale=1.0
    ):
        """
        Build the CAD model of a microstructure and mesh it.

        Parameters
        ----------
        gmsh: module
            The gmsh module, in an initialized session.

        microstructure: `.Microstructure`
            Microstructure to be built.

        refine_surfaces: list
            Tags of the surfaces a previous attempt could not mesh, around which the
            element size is driven down.

        report: callable
            Called with the index of the particle that was added and the total number
            of particles.

        scale: float
            Factor the microstructure was multiplied by, which the element size, given
            in the user's units, is multiplied by as well.

        Returns
        -------
        dict
            Correspondence between the name of a phase and the *(dimension, tag)* of
            the physical group holding it.
        """
        self.set_options(gmsh)
        model = gmsh.model
        factory = model.occ
        model.add("microstructure")
        rve_dims = microstructure.rve_dims
        dim = len(rve_dims)
        if dim == 2:
            box_tag = factory.addRectangle(0, 0, 0, rve_dims[0], rve_dims[1])
        else:
            box_tag = factory.addBox(
                0, 0, 0, rve_dims[0], rve_dims[1], rve_dims[2]
            )

        primitives = []
        primitive_phases = []
        particle_volumes = {}
        particles = microstructure.particles
        for i_particle_ind, i_particle in enumerate(particles):
            for j_image, j_center in enumerate(periodic_images(i_particle, rve_dims)):
                dim_tags = self.add_primitive(factory, model, i_particle, j_center)
                if j_image == 0:
                    volume = sum(factory.getMass(*k_dim_tag) for k_dim_tag in dim_tags)
                    particle_volumes[i_particle.phase] = (
                        particle_volumes.get(i_particle.phase, 0.0) + volume
                    )
                    if i_particle.parent is not None:
                        particle_volumes[i_particle.parent.phase] = (
                            particle_volumes.get(i_particle.parent.phase, 0.0) - volume
                        )
                for k_dim_tag in dim_tags:
                    primitives.append(k_dim_tag)
                    primitive_phases.append(i_particle.phase)
            if report is not None:
                report(i_particle_ind, len(particles))
        # The images of a particle are translations of one another, so the first one
        # measures what the images cut by the RVE add up to. A particle placed inside
        # another takes its volume from the phase of that one

        out_dim_tag, cut_map = factory.intersect(
            [(dim, box_tag)], primitives, removeObject=False, removeTool=True
        )
        # Cutting the particles against the box leaves only what is inside the RVE.
        # The map has one entry per input, the box first and then the primitives, so it
        # says which pieces came from which particle

        phase_of_piece = {}
        for i_pieces, i_phase in zip(cut_map[1:], primitive_phases):
            for j_piece in i_pieces:
                phase_of_piece[j_piece] = i_phase
        factory.synchronize()

        _, fragment_map = factory.fragment(
            [(dim, box_tag)], out_dim_tag, removeObject=True, removeTool=True
        )
        # Fragmenting against the box makes the matrix and the particles share their
        # interfaces

        phase_of_fragment = {}
        for i_fragments, i_piece in zip(fragment_map[1:], out_dim_tag):
            for j_fragment in i_fragments:
                phase_of_fragment[j_fragment] = phase_of_piece[i_piece]
        for i_fragment in fragment_map[0]:
            phase_of_fragment.setdefault(i_fragment, microstructure.matrix_phase)
        # Whatever is left of the box once every particle has claimed its fragments is
        # the matrix. Reading it off the map rather than off the position of the
        # entities in the result is what makes this independent of the order gmsh
        # happens to return them in, which is not the order the tools were given in
        # TODO: with particles that overlap, the map has been seen to name among the
        # fragments of a particle one of a neighbour it does not touch: in a random set
        # of ellipsoids, one of which overlapped three others, and never in a set with
        # no overlaps. Across two phases such a fragment takes the wrong one, which the
        # check of the volumes sees only past a per cent of the phase. Telling the
        # phase of each fragment by a point inside it would not rely on the map

        materials = {i_name: [] for i_name in microstructure.phases}
        for i_fragment, i_name in phase_of_fragment.items():
            materials[i_name].append(i_fragment[1])
        self.warnings += phase_volume_warnings(
            factory, dim, materials, particle_volumes, np.prod(rve_dims)
        )
        # A piece of a particle the booleans lose becomes matrix, and the cells still
        # fill the RVE, so the check of the mesh does not see it; the volume of each
        # phase, against that of its particles, does
        factory.synchronize()

        phase_groups = {}
        for i_name in microstructure.phases:
            group = model.addPhysicalGroup(dim, materials[i_name])
            model.setPhysicalName(dim, group, "Phase {0}".format(i_name))
            phase_groups[i_name] = (dim, group)
        # Every phase becomes a physical group, which is what carries the phase of a
        # cell out of gmsh

        self.enforce_pbc(gmsh, rve_dims)

        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 6)
        # Target number of elements per 2*pi radians of curvature. Before Gmsh 4.7 this
        # option was a boolean and the count lived in Mesh.MinimumElementsPerTwoPi,
        # whose default was 6; the two were merged, so 6 preserves the original intent.
        gmsh.option.setNumber("Mesh.MeshSizeMax", self.mesh_size * scale)
        if refine_surfaces:
            self.refine_around(gmsh, refine_surfaces, self.mesh_size * scale)

        model.mesh.generate(dim)
        if model.mesh.getLastEntityError():
            self.warnings.append("Gmsh detected an error while generating the mesh")
        if self.descriptors["element_order"] > 1:
            model.mesh.optimize("HighOrder", force=False, niter=10)
            if model.mesh.getLastEntityError():
                self.warnings.append(
                    "Gmsh detected an error while optimizing the mesh"
                )
        # The optimizer places the mid-side nodes of second order elements; on straight
        # sided ones it has nothing to place, and on first order quads it raised over
        # the element quality instead, so a quad4 mesh of ellipses could not be made

        self.enforce_pbc(gmsh, rve_dims)
        # Repeated because gmsh sometimes behaves unpredictably

        return phase_groups

    @staticmethod
    def refine_around(gmsh, surfaces, mesh_size):
        """
        Drive the element size down near the surfaces a previous attempt could not mesh.

        The other sources of element size stay enabled and gmsh takes the smallest, so
        the refinement by curvature is preserved away from these surfaces.

        Parameters
        ----------
        gmsh: module
            The gmsh module, in an initialized session.

        surfaces: list
            Tags of the surfaces to refine around.

        mesh_size: float
            Largest element size, in the units of the model.
        """
        field_distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(field_distance, "SurfacesList", surfaces)
        gmsh.model.mesh.field.setNumber(field_distance, "Sampling", 100)
        field_threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(field_threshold, "InField", field_distance)
        gmsh.model.mesh.field.setNumber(field_threshold, "SizeMin", mesh_size / 8)
        gmsh.model.mesh.field.setNumber(field_threshold, "SizeMax", mesh_size)
        gmsh.model.mesh.field.setNumber(field_threshold, "DistMin", 0)
        gmsh.model.mesh.field.setNumber(field_threshold, "DistMax", mesh_size)
        gmsh.model.mesh.field.setAsBackgroundMesh(field_threshold)

    @staticmethod
    def add_primitive(factory, model, particle, center):
        """
        Add the geometry of one particle, or of one of its periodic images, to a model.

        This is the only place that builds a primitive, so a new particle shape becomes
        meshable by being given a branch here.

        Parameters
        ----------
        factory: module
            The OpenCASCADE geometry kernel of gmsh.

        model: module
            The model of the gmsh session.

        particle: `.Particle`
            Particle whose geometry is wanted.

        center: tuple
            Coordinates of the centre of the image being added.

        Returns
        -------
        list
            The *(dimension, tag)* pairs of the entities that were added.

        Raises
        ------
        UnsupportedParticleShape:
            If there is no geometry for the shape of the particle.
        """
        center_x, center_y, center_z = center
        if isinstance(particle, CylindricalFiber):
            face_tag = factory.addDisk(
                center_x,
                center_y,
                center_z,
                particle.semi_major_axis,
                particle.semi_minor_axis,
            )
            if particle.direction_fibers == 0:
                factory.rotate(
                    [(2, face_tag)], center_x, center_y, center_z, 0, 1, 0, np.pi / 2
                )
            elif particle.direction_fibers == 1:
                factory.rotate(
                    [(2, face_tag)], center_x, center_y, center_z, 1, 0, 0, np.pi / 2
                )
            extrude_direction = [0, 0, 0]
            extrude_direction[particle.direction_fibers] = particle.length_dir_fibers
            # The face is drawn in the xy plane about the centre of the end of the fibre
            # and turned about that centre into the plane across the fibre. It was
            # turned about the origin, which took a fibre along x to the place with its
            # two coordinates swapped

            entities = [
                i_dim_tag
                for i_dim_tag in factory.extrude([(2, face_tag)], *extrude_direction)
                if i_dim_tag[0] == 3
            ]

            return entities

        if isinstance(particle, Disk):
            return [
                (
                    2,
                    factory.addDisk(
                        center_x,
                        center_y,
                        center_z,
                        particle.semi_major_axis,
                        particle.semi_minor_axis,
                    ),
                )
            ]

        if isinstance(particle, Ellipse):
            tag = factory.addDisk(
                center_x,
                center_y,
                center_z,
                particle.semi_major_axis,
                particle.semi_minor_axis,
            )
            factory.synchronize()
            # The only synchronize this method needs: getBoundary reads the model, not
            # the geometry kernel, so the disk has to have reached it first
            rotate_tags = [(2, tag)] + model.getBoundary([(2, tag)])
            factory.rotate(
                rotate_tags, center_x, center_y, center_z, 0, 0, 1, particle.angle
            )

            return [(2, tag)]

        if isinstance(particle, Sphere):
            return [
                (3, factory.addSphere(center_x, center_y, center_z, particle.radius))
            ]

        if isinstance(particle, Ellipsoid):
            tag = factory.addSphere(center_x, center_y, center_z, 1)
            factory.dilate(
                [(3, tag)],
                center_x,
                center_y,
                center_z,
                particle.semi_axis_1,
                particle.semi_axis_2,
                particle.semi_axis_3,
            )
            factory.rotate(
                [(3, tag)],
                center_x,
                center_y,
                center_z,
                particle.rotation_axis[0],
                particle.rotation_axis[1],
                particle.rotation_axis[2],
                particle.angle,
            )
            # A sphere of unit radius stretched onto the semi axes and then turned,
            # since the kernel has no ellipsoid of its own

            return [(3, tag)]

        if isinstance(particle, Cylinder):
            face_tag = factory.addDisk(
                center_x,
                center_y,
                center_z - particle.length / 2,
                particle.r_cyl,
                particle.r_cyl,
            )
            entities = [
                i_dim_tag
                for i_dim_tag in factory.extrude(
                    [(2, face_tag)], 0, 0, particle.length
                )
                if i_dim_tag[0] == 3
            ]
            if particle.polar_angle != 0:
                factory.rotate(
                    [entities[-1]],
                    center_x,
                    center_y,
                    center_z,
                    -particle.sym_axis_unit_vec[1],
                    particle.sym_axis_unit_vec[0],
                    0,
                    particle.polar_angle,
                )

            return entities

        raise UnsupportedParticleShape(type(particle).__name__)

    def enforce_pbc(self, gmsh, rve_dims):
        """
        Declare every pair of opposite faces of the RVE periodic.

        Parameters
        ----------
        gmsh: module
            The gmsh module, in an initialized session.

        rve_dims: list(float)
            Dimensions of the microstructure in each spatial direction.
        """
        gmsh.model.occ.synchronize()
        bounding_dims = (1,) if len(rve_dims) == 2 else (1, 2)
        # The boundary of a two dimensional RVE is made of edges, and that of a three
        # dimensional one of edges and faces

        for i_direction in range(len(rve_dims)):
            for j_dim in bounding_dims:
                self.enforce_pbc_one_way(gmsh, rve_dims, i_direction, j_dim)

    @staticmethod
    def enforce_pbc_one_way(
        gmsh, rve_dims, direction, dim, eps=PBC_TOLERANCE, plane_eps=PBC_PLANE_TOLERANCE
    ):
        """
        Declare the two faces of the RVE normal to one direction periodic.

        Parameters
        ----------
        gmsh: module
            The gmsh module, in an initialized session.

        rve_dims: list(float)
            Dimensions of the microstructure in each spatial direction.

        direction: {0, 1, 2}
            Axis the paired faces are normal to.

        dim: {1, 2}
            Dimension of the bounding entity, 1 for edges and 2 for faces.

        eps: float
            Tolerance of the bounding boxes used to find the pairs, along the faces.

        plane_eps: float
            Tolerance of the same boxes across the faces.
        """
        gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
        translation = [0, 0, 0]
        translation[direction] = rve_dims[direction]
        normal_plane = list(rve_dims) if len(rve_dims) == 3 else list(rve_dims) + [0]
        normal_plane[direction] = 0
        affine = [
            1, 0, 0, translation[0],
            0, 1, 0, translation[1],
            0, 0, 1, translation[2],
            0, 0, 0, 1,
        ]
        tolerance = [eps, eps, eps]
        tolerance[direction] = plane_eps
        main_face = gmsh.model.getEntitiesInBoundingBox(
            -tolerance[0],
            -tolerance[1],
            -tolerance[2],
            normal_plane[0] + tolerance[0],
            normal_plane[1] + tolerance[1],
            normal_plane[2] + tolerance[2],
            dim,
        )
        for i_entity in main_face:
            limits = gmsh.model.getBoundingBox(i_entity[0], i_entity[1])
            opposite = gmsh.model.getEntitiesInBoundingBox(
                limits[0] - tolerance[0] + translation[0],
                limits[1] - tolerance[1] + translation[1],
                limits[2] - tolerance[2] + translation[2],
                limits[3] + tolerance[0] + translation[0],
                limits[4] + tolerance[1] + translation[1],
                limits[5] + tolerance[2] + translation[2],
                dim,
            )
            # The bounding box of the entity is translated to the opposite face and
            # whatever sits inside it is a candidate partner

            for j_entity in opposite:
                other = gmsh.model.getBoundingBox(j_entity[0], j_entity[1])
                shifted = [
                    other[i_corner] - translation[i_corner % 3] for i_corner in range(6)
                ]
                if all(
                    [
                        abs(shifted[i_corner] - limits[i_corner])
                        < tolerance[i_corner % 3]
                        for i_corner in range(6)
                    ]
                ):
                    gmsh.model.mesh.setPeriodic(
                        dim, [j_entity[1]], [i_entity[1]], affine
                    )
                    # The bounding boxes match once translated, so the two entities are
                    # the same geometry on opposite faces

        gmsh.model.occ.synchronize()

    def extract_mesh(self, gmsh, microstructure, phase_groups, scale=1.0):
        """
        Read the mesh out of a gmsh session.

        Parameters
        ----------
        gmsh: module
            The gmsh module, in an initialized session holding a mesh.

        microstructure: `.Microstructure`
            Microstructure that was meshed.

        phase_groups: dict
            Correspondence between the name of a phase and the *(dimension, tag)* of
            its physical group.

        scale: float
            Factor the model was built at, which the coordinates are divided by to give
            them in the units of the microstructure.

        Returns
        -------
        `.Mesh`
            The mesh, with the phase of every cell.

        Raises
        ------
        ValueError:
            If gmsh produced no cell of the dimension of the microstructure.
        """
        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        node_tags = np.asarray(node_tags, dtype=np.int64)
        points = np.asarray(coordinates, dtype=float).reshape(-1, 3) / scale
        index_of_tag = np.zeros(int(node_tags.max()) + 1, dtype=np.int64)
        index_of_tag[node_tags] = np.arange(len(node_tags), dtype=np.int64)
        # Gmsh identifies a node by a tag that is neither dense nor ordered, so the
        # connectivity is translated into positions in the array of coordinates

        blocks = {}
        phases = {}
        nodes_per_element = {}
        for i_name, (i_dim, i_group) in phase_groups.items():
            for j_entity in gmsh.model.getEntitiesForPhysicalGroup(i_dim, i_group):
                types, _, nodes_per_type = gmsh.model.mesh.getElements(
                    i_dim, int(j_entity)
                )
                for k_type, k_nodes in zip(types, nodes_per_type):
                    cell_type = GMSH_CELL_TYPES[int(k_type)]
                    n_nodes = nodes_per_element.get(int(k_type))
                    if n_nodes is None:
                        n_nodes = gmsh.model.mesh.getElementProperties(int(k_type))[3]
                        nodes_per_element[int(k_type)] = n_nodes
                    # The node count depends on the element type alone, and there is one
                    # entity per particle image and per matrix fragment to loop over

                    connectivity = index_of_tag[
                        np.asarray(k_nodes).reshape(-1, n_nodes)
                    ]
                    order = GMSH_TO_VTK_ORDER.get(cell_type)
                    if order is not None:
                        connectivity = connectivity[:, order]
                    # The nodes of a second order element are listed by gmsh in an
                    # order of its own, which is turned here into the one VTK uses

                    blocks.setdefault(cell_type, []).append(connectivity)
                    phases.setdefault(cell_type, []).append(
                        np.full(len(connectivity), int(i_name), dtype=int)
                    )
        if not blocks:
            raise ValueError(
                "Gmsh produced no element of dimension {0}.".format(
                    len(microstructure.rve_dims)
                )
            )

        cells = [
            (i_type, np.vstack(i_blocks)) for i_type, i_blocks in blocks.items()
        ]
        phase = [np.concatenate(phases[i_type]) for i_type, _ in cells]
        check_fills_rve(points, cells, microstructure.rve_dims)
        # Gmsh only warns when it leaves a surface or a volume without elements, and
        # the booleans that cut the particles against the box can lose a piece without
        # a word; the pairing of the faces sees neither. Such a mesh was taken as it
        # was, and at a large scale one was written with no matrix at all
        if len(cells) > 1:
            self.warnings.append(
                "WARNING: {0} was asked for and gmsh produced {1}; the mesh is written "
                "with every element it produced.".format(
                    self.element_type,
                    " and ".join(
                        "{0} {1}".format(len(i_connectivity), i_type)
                        for i_type, i_connectivity in cells
                    ),
                )
            )
        # One element type is asked for, so a second one is gmsh's doing: the quad
        # recombination leaves a triangle behind where it finds no pair for it, and
        # says nothing. The writers carry every type through, so the solver gets the
        # mix, and the user is told about it here

        return Mesh(
            microstructure.rve_dims,
            points=points,
            cells=cells,
            phase=phase,
            phase_names={int(i_name): i_name for i_name in microstructure.phases},
            matrix_phase=microstructure.matrix_phase,
            periodic=True,
            source={
                "mesher": self.name,
                "element_type": self.element_type,
                "mesh_size": self.mesh_size,
            },
        )
