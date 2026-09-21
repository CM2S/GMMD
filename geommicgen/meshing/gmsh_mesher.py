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

import numpy as np

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen._optional import require_gmsh
from geommicgen.errors.error_classes import UnsupportedParticleShape
from geommicgen.meshing.images import periodic_images
from geommicgen.meshing.mesh import Mesh
from geommicgen.meshing.mesher import Mesher, register_mesher
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

PBC_TOLERANCE = 1.0e-3
# Tolerance of the bounding boxes used to pair opposite faces of the RVE


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
        "Mesh_Size": {"type": "float", "help": "largest element size"},
        "Elements_Per_Particle": {
            "type": "float",
            "help": "elements across the smallest particle, instead of a size",
        },
        "Element_Type": {
            "type": "str",
            "help": "element to mesh with (default: tri3)",
        },
    }

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
        self.label = element_type

    def mesh(self, microstructure, report=None):
        """
        Build an unstructured mesh of a microstructure with gmsh.

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

        self.resolve_mesh_size(microstructure)
        refine_surfaces = []
        for i_attempt in range(self.max_attempts):
            with gmsh_session() as gmsh:
                try:
                    phase_groups = self.build_model(
                        gmsh, microstructure, refine_surfaces, report
                    )
                except Exception as error:
                    refine_surfaces = failing_surfaces(error)
                    if not refine_surfaces or i_attempt == self.max_attempts - 1:
                        raise
                    continue
                    # The model has to be rebuilt in a new session: once a meshing pass
                    # has failed, gmsh will not produce a mesh for that model again

                return self.extract_mesh(gmsh, microstructure, phase_groups)
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
        gmsh.option.setNumber("Mesh.MaxNumThreads1D", 4)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", 4)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", 4)
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

    def build_model(self, gmsh, microstructure, refine_surfaces=(), report=None):
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
        particles = microstructure.particles
        for i_particle_ind, i_particle in enumerate(particles):
            for j_center in periodic_images(i_particle, rve_dims):
                for k_dim_tag in self.add_primitive(
                    factory, model, i_particle, j_center
                ):
                    primitives.append(k_dim_tag)
                    primitive_phases.append(i_particle.phase)
            if report is not None:
                report(i_particle_ind, len(particles))

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

        materials = {i_name: [] for i_name in microstructure.phases}
        for i_fragment, i_name in phase_of_fragment.items():
            materials[i_name].append(i_fragment[1])
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
        gmsh.option.setNumber("Mesh.MeshSizeMax", self.mesh_size)
        if refine_surfaces:
            self.refine_around(gmsh, refine_surfaces)

        model.mesh.generate(dim)
        if model.mesh.getLastEntityError():
            self.warnings.append("Gmsh detected an error while generating the mesh")
        model.mesh.optimize("HighOrder", force=False, niter=10)
        if model.mesh.getLastEntityError():
            self.warnings.append("Gmsh detected an error while optimizing the mesh")

        self.enforce_pbc(gmsh, rve_dims)
        # Repeated because gmsh sometimes behaves unpredictably

        return phase_groups

    def refine_around(self, gmsh, surfaces):
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
        """
        field_distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(field_distance, "SurfacesList", surfaces)
        gmsh.model.mesh.field.setNumber(field_distance, "Sampling", 100)
        field_threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(field_threshold, "InField", field_distance)
        gmsh.model.mesh.field.setNumber(field_threshold, "SizeMin", self.mesh_size / 8)
        gmsh.model.mesh.field.setNumber(field_threshold, "SizeMax", self.mesh_size)
        gmsh.model.mesh.field.setNumber(field_threshold, "DistMin", 0)
        gmsh.model.mesh.field.setNumber(field_threshold, "DistMax", self.mesh_size)
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
                factory.rotate([(2, face_tag)], 0, 0, 0, 0, 1, 0, 3 * np.pi / 2)
                extrude_direction = [particle.length_dir_fibers, 0, 0]
            elif particle.direction_fibers == 1:
                factory.rotate([(2, face_tag)], 0, 0, 0, 1, 0, 0, np.pi / 2)
                extrude_direction = [0, particle.length_dir_fibers, 0]
            else:
                extrude_direction = [0, 0, particle.length_dir_fibers]
            # The face is drawn in the xy plane and turned to the plane normal to the
            # direction the fibres run in

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
    def enforce_pbc_one_way(gmsh, rve_dims, direction, dim, eps=PBC_TOLERANCE):
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
            Tolerance of the bounding boxes used to find the pairs.
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
        main_face = gmsh.model.getEntitiesInBoundingBox(
            -eps,
            -eps,
            -eps,
            normal_plane[0] + eps,
            normal_plane[1] + eps,
            normal_plane[2] + eps,
            dim,
        )
        for i_entity in main_face:
            limits = gmsh.model.getBoundingBox(i_entity[0], i_entity[1])
            opposite = gmsh.model.getEntitiesInBoundingBox(
                limits[0] - eps + translation[0],
                limits[1] - eps + translation[1],
                limits[2] - eps + translation[2],
                limits[3] + eps + translation[0],
                limits[4] + eps + translation[1],
                limits[5] + eps + translation[2],
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
                        abs(shifted[i_corner] - limits[i_corner]) < eps
                        for i_corner in range(6)
                    ]
                ):
                    gmsh.model.mesh.setPeriodic(
                        dim, [j_entity[1]], [i_entity[1]], affine
                    )
                    # The bounding boxes match once translated, so the two entities are
                    # the same geometry on opposite faces

        gmsh.model.occ.synchronize()

    def extract_mesh(self, gmsh, microstructure, phase_groups):
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
        points = np.asarray(coordinates, dtype=float).reshape(-1, 3)
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
