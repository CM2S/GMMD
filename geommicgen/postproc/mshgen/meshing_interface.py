"""Module containing the meshing inteface classes.

This module contains the FEMMeshGenerator and the RGMeshGenerator classes, whose object
instances generate finite element method mesh and regular grid meshes, respectively, using
the information form Microstructure classs object.
"""
import os
import shutil
import abc

# GMSH module
import gmsh

# Simple math tools

# Finite element mesh conversor to LINKS
from gmsh2links.main import readMesh
from microstructure.particle_classes import (
    Disk,
    Ellipse,
    Ellipsoid,
    Sphere,
    CylindricalFiber,
)

# Importing the particle class
import numpy as np
import errors.error_classes as errors

# from plotting_functions import plot_pixels, plot_voxels


class MeshGenerator(abc.ABC):
    """Class for the mesh generators."""

    @abc.abstractmethod
    def generate_mesh(self, microstructure_sample, file_path):
        """Generate a mesh for *microstructure_sample*."""


class FEMMeshGenerator(MeshGenerator):
    """Class for the mesh generator of finite element meshes.

    Attributes
    ----------
    mesh_size: float
        Maximum mesh size.

    element_type: str
        Element type.

    descriptors_element_type: dict
        Descriptors of the elements used to mesh the microstucutre.

    output_term: {0, 1}
        Flag for the gmsh output.

    particle_tags: list(int)
        List containing the particle tags of the gmsh model.

    box_tag: int
        Box tag in the gmsh model.

    phase_dim_tag: dict()
        Tags of the particles in each phase.

    enforce_pbc_flag: bool
        Flag for the enforcement of periodic boundary conditions. By defalut True. Only set
        to  False if there are Ellipsoids in the microstructure. Gmsh has not been able to
        produce  microstucutres containing Ellipsoids and with pbcs.

    Class Attributes
    ----------------
    known_element_descriptors: dict
        Dictionary whose keys are the name of the elements and the values are dictionaries
        containing their descriptors.

    Notes
    ------
    The descriptors of the elements and their possible values are

    dim: int
        Dimension of the element.

    mesh_alg: int
        2D Meshing algorithm
        1 - Mesh Adapt
        2 - Automatic
        5 - Delaunay (default)
        6 - Frontal-Delaunay
        7 - BAMG
        8 - Frontal-Delaunay for Quads
        9 - Packing of Parallelograms

    mesh_alg_3d: int
        3D Meshing algorithm
        1 - Delaunay (default)
        2 - Frontal
        7 - MMG3D
        9 - R-tree
        10 - HXT

    force_recomb_all_surf: {0, 1}

    force_recomb_all_vol: {0, 1}

    element_order: int
        Order of the element

    recomb_alg: int
        Quad/Hex recombination algorithms
        0 - simple
        1 - blossom (default)
        2 - simple full-quad
        3 - blosson full-quad

    recomb_alg_3d: int
        Recombination level in 3D
        0 - hex (default)
        1 - hex + prisms
        2 - hex + prisms + pyramids

    recombine_3d_conformity: int
        0 - nonconforming (default)
        1 - trihedra
        2 - pyramids + trihedra
        2 - pyramids + hexSplit + trihedra
        4 - hexSplit + trihedra

    element_order_incomp: {0, 1}
        Second order incomplete elements.
    """

    known_element_descriptors = {
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

    def __init__(self, mesh_size, element_type, rve_dims, **kwargs):
        """
        Intialize a FEMMeshGenerator class object.

        Parameters
        ----------
        mesh_size: float
            Maximum mesh size.

        element_type: str
            Element type.

        rve_dims: list(float)
            Dimensions of the microstucutre in each spatial direciton.

        Keyword Arguments
        -----------------
        Element descriptors.
        """
        if mesh_size < 0:
            raise ValueError("The mesh size must be a positive number.")
        self.mesh_size = mesh_size
        if element_type not in FEMMeshGenerator.known_element_descriptors:
            raise ValueError("Unknown element: {0}".format(element_type))
        if FEMMeshGenerator.known_element_descriptors[element_type]["dim"] != len(
            rve_dims
        ):
            raise ValueError("Element chosen has the wrong dimension.")
        self.element_type = element_type
        self.descriptors_element_type = FEMMeshGenerator.known_element_descriptors[
            element_type
        ]
        kwargs.update(self.descriptors_element_type)
        self.output_term = kwargs.get("output_term", False)
        self.particle_tags = []
        self.box_tag = None
        self.phase_dim_tag = None
        self.enforce_pbc_flag = True

    def generate_mesh(self, microstructure_sample, sample_dir):
        """
        Generate the mesh for the Finite Element Method using gmsh.

        Parameters
        ----------
        microstructure_sample: `.Microstructure`
            Microstructure to be meshed.

        sample_dir: str
            Path to store the meshes.
        """
        self.init_gmsh_model()
        self.generate_mesh_gmsh(
            microstructure_sample,
            sample_dir,
        )
        mesh_results_dir = os.path.join(sample_dir, "meshes")
        if not os.path.exists(mesh_results_dir):
            os.makedirs(mesh_results_dir)
        meshfile = self.write_mesh_gmsh(mesh_results_dir, "femsh")
        self.gmsh_to_links(
            meshfile,
            mesh_results_dir,
            2,
            list(microstructure_sample.phases.keys()),
            microstructure_sample.matrix_phase,
        )
        #

    def init_gmsh_model(self):
        """Initialize and set the options for the gmsh model."""
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", self.output_term)
        # Outupt to terminal

        gmsh.option.setNumber(
            "Mesh.Algorithm", self.descriptors_element_type["mesh_alg"]
        )
        # 2D Meshing algorithm

        gmsh.option.setNumber(
            "Mesh.Algorithm3D", self.descriptors_element_type.get("mesh_alg_3d", 1)
        )
        # 3D Meshing algorithm

        gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1)
        # Characteristic mesh length factor (applied acroos all mesh)

        gmsh.option.setNumber("Mesh.MaxNumThreads1D", 4)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", 4)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", 4)
        # Multi-threading

        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        # MSH file version

        gmsh.option.setNumber(
            "Mesh.RecombinationAlgorithm", self.descriptors_element_type["recomb_alg"]
        )
        # Quad/Hex recombination algorithms

        gmsh.option.setNumber(
            "Mesh.RecombineAll", self.descriptors_element_type["force_recomb_all_surf"]
        )
        # Force recombination in all surfaces

        gmsh.option.setNumber("Mesh.RecombineOptimizeTopology", 5)
        # Number of topological optimization passes of recombined surface meshes (5 by
        # default)

        gmsh.option.setNumber(
            "Mesh.Recombine3DAll",
            self.descriptors_element_type.get("force_recomb_all_vol", 0),
        )
        # Force recombination in all volumes

        gmsh.option.setNumber(
            "Mesh.Recombine3DLevel",
            self.descriptors_element_type.get("recomb_alg_3d", 0),
        )
        # Recombination level in 3D

        gmsh.option.setNumber(
            "Mesh.Recombine3DConformity",
            self.descriptors_element_type.get("recombine_3d_conformity", 0),
        )
        # Recombination conformity type in 3D meshes

        gmsh.option.setNumber("Mesh.Renumber", 1)
        # Renumber nodes and elements after mesh generation

        gmsh.option.setNumber("Mesh.SaveAll", 0)
        # Save all elements even if they do not belong to physical groups

        gmsh.option.setNumber("Mesh.Smoothing", 1)
        # Number of smoothing step applied to the final mesh

        gmsh.option.setNumber(
            "Mesh.ElementOrder", self.descriptors_element_type["element_order"]
        )
        # Element order

        gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
        # Crete second-order nodes by linear interpolation

        gmsh.option.setNumber(
            "Mesh.SecondOrderIncomplete",
            self.descriptors_element_type["element_order_incomp"],
            # Second-order incomplete elements
        )

    def write_mesh_gmsh(self, mesh_results_dir, name):
        """Write the mesh to the .msh and .vtk file."""

        meshfile_temp = os.path.join(mesh_results_dir, name + "_temp.msh")
        meshfile = os.path.join(mesh_results_dir, name + ".msh")
        vtk_temp = os.path.join(mesh_results_dir, name + "_temp.vtk")
        vtk = os.path.join(mesh_results_dir, name + ".vtk")
        gmsh.write(meshfile_temp)
        gmsh.write(vtk_temp)

        # Close GMSH
        gmsh.finalize()

        fin = open(meshfile_temp, "rt")
        fout = open(meshfile, "wt")

        for line in fin:
            fout.write(line.replace(",", "."))

        fin.close()
        fout.close()
        os.remove(meshfile_temp)
        # Sometimes gmsh swaps periods for commas

        fin = open(vtk_temp, "rt")
        fout = open(vtk, "wt")

        for line in fin:
            fout.write(line.replace(",", "."))

        fin.close()
        fout.close()
        os.remove(vtk_temp)
        # Sometimes gmsh swaps periods for commas

        return meshfile

    def generate_mesh_gmsh(
        self,
        microstructure_sample,
        file_path,
    ):
        """
        Generate the mesh for the Finite Element Method using gmsh.

        Parameters
        ----------
        microstructure_sample: `.Microstructure`
            Sample microstructure to be meshed.

        file_path: str
            Path to store the meshes.
        """
        model = gmsh.model
        factory = model.occ
        # occ - OpenCASCADE CAD (more advanced)

        title = file_path
        model.add(title)

        rve_dims = microstructure_sample.rve_dims
        particles = microstructure_sample.particles
        dim = len(rve_dims)

        if dim == 2:
            self.box_tag = factory.addRectangle(0, 0, 0, rve_dims[0], rve_dims[1])
        elif dim == 3:
            self.box_tag = factory.addBox(
                0, 0, 0, rve_dims[0], rve_dims[1], rve_dims[2]
            )

        # RVE

        self.phase_dim_tag = {
            phase_name: [] for phase_name in microstructure_sample.phases
        }
        for i_particle in particles:
            # Running through all the particles
            self.add_particle_pbc_to_model(i_particle, rve_dims)

        out_dim_tag, _ = factory.intersect(
            [(dim, self.box_tag)],
            [(dim, particle_tag) for particle_tag in self.particle_tags],
            removeObject=False,
            removeTool=True,
        )
        # Computing the intersection of the particles with the volume element

        temp = set(out_dim_tag)
        for i_phase in microstructure_sample.phases:
            self.phase_dim_tag[i_phase] = [
                value for value in self.phase_dim_tag[i_phase] if value in temp
            ]
        # Saving what tags/geometry belongs to which phase

        factory.synchronize()

        out_dim_tag_2, _ = factory.fragment(
            [(dim, self.box_tag)], out_dim_tag, removeObject=True, removeTool=True
        )
        # Computing the fragment of the particles with the matrix

        self.phase_dim_tag[microstructure_sample.matrix_phase] = out_dim_tag_2[
            len(out_dim_tag) :
        ]
        materials = dict()
        for i_phase in microstructure_sample.phases:
            temp = set(self.phase_dim_tag[i_phase])
            materials[i_phase] = [value[1] for value in out_dim_tag_2 if value in temp]
        # Collection the tags in the correct phase

        factory.synchronize()
        # Synchronize the CAD engine (always needed before generating the mesh) It may also
        # be useful for some intermidate operations, like checking the tags of entities

        for i_phase in microstructure_sample.phases:
            material_tag = model.addPhysicalGroup(dim, materials[i_phase])
            model.setPhysicalName(dim, material_tag, "Phase {0}".format(i_phase))
        # Setting each phase as a physical group

        self.enforce_pbc(rve_dims)

        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.mesh_size)

        # Generate a 3D mesh
        model.mesh.generate(dim)

        model.mesh.optimize("HighOrder", force=False, niter=10)

        self.enforce_pbc(rve_dims)
        # Repeated becaused gmsh sometines behaves unpredictably

    def add_particle_pbc_to_model(self, i_particle, rve_dims):
        """
        Add to the gmsh model the particle "i_particle" and its periodic images.

        Parameters
        ----------
        i_particle: `.Particle`
            Particle to be added to the gmsh model.

        rve_dims: list(float)
            Dimensions of the microstructure in each spatial direction.
        """
        model = gmsh.model
        factory = gmsh.model.occ
        eps = 0
        for (j_pbc, p_pbc) in [
            (j_pbc, p_pbc) for j_pbc in range(-1, 2) for p_pbc in range(-1, 2)
        ]:
            if i_particle.dim == 2:
                x_c = i_particle.position_center[0] + rve_dims[0] * j_pbc
                y_c = i_particle.position_center[1] + rve_dims[1] * p_pbc
                z_c = 0
                r_x = i_particle.semi_major_axis
                r_y = i_particle.semi_minor_axis
                if (
                    x_c > rve_dims[0] + r_x - eps
                    or x_c < -r_x + eps
                    or y_c > rve_dims[1] + r_y - eps
                    or y_c < -r_y + eps
                ):
                    continue
                # Saving the properties of the particles

                factory.synchronize()
                if isinstance(i_particle, CylindricalFiber):
                    face_tag = factory.addDisk(x_c, y_c, z_c, r_x, r_y)
                    # Saving the properties of the particles
                    if i_particle.direction_fibers == 0:
                        # The fibers run in the x direction
                        factory.rotate([(2, face_tag)], 0, 0, 0, 0, 1, 0, 3 * np.pi / 2)
                        # Rotating the fiber face to the yz plane as it was ploted
                        # in the xy plane
                        extrude_direction = [i_particle.length_dir_fibers, 0, 0]
                    elif i_particle.direction_fibers == 1:
                        # The fibers run in the y direction
                        factory.rotate([(2, face_tag)], 0, 0, 0, 1, 0, 0, np.pi / 2)
                        # Rotating the fiber faces to the xz plane as it was ploted
                        # in the xy plane
                        extrude_direction = [0, i_particle.length_dir_fibers, 0]
                    elif i_particle.direction_fibers == 2:
                        # The fibers run in the z direction
                        extrude_direction = [0, 0, i_particle.length_dir_fibers]
                    extrusion_tags = factory.extrude(
                        [(2, face_tag)], *extrude_direction
                    )
                    # Extruding the fiber
                    for i_dim_tag in extrusion_tags:
                        if i_dim_tag[0] == 3:
                            self.particle_tags.append(i_dim_tag[1])

                            self.phase_dim_tag[str(i_particle.phase)].append(
                                (3, self.particle_tags[-1])
                            )
                            # break

                            factory.synchronize()
                elif isinstance(i_particle, Disk):
                    self.particle_tags.append(factory.addDisk(x_c, y_c, z_c, r_x, r_y))
                    # Particle is a Disk
                    self.phase_dim_tag[i_particle.phase].append(
                        (2, self.particle_tags[-1])
                    )
                elif isinstance(i_particle, Ellipse):
                    self.particle_tags.append(factory.addDisk(x_c, y_c, z_c, r_x, r_y))
                    # Particle is a Disk
                    self.phase_dim_tag[i_particle.phase].append(
                        (2, self.particle_tags[-1])
                    )
                    alpha = i_particle.angle
                    factory.synchronize()
                    rotate_tag = [(2, self.particle_tags[-1])]
                    rotate_tag.extend(model.getBoundary([2, self.particle_tags[-1]]))
                    factory.rotate(rotate_tag, x_c, y_c, z_c, 0, 0, 1, alpha)

                    self.phase_dim_tag[i_particle.phase].append(
                        (2, self.particle_tags[-1])
                    )
            elif i_particle.dim == 3:
                for l_pbc in range(-1, 2):
                    # Particle is a Sphere
                    x_c = i_particle.position_center[0] + rve_dims[0] * j_pbc
                    y_c = i_particle.position_center[1] + rve_dims[1] * p_pbc
                    z_c = i_particle.position_center[2] + rve_dims[2] * l_pbc
                    r = i_particle.radius
                    # Saving the properties of the particles
                    if (
                        x_c > rve_dims[0] + r
                        or x_c < -r
                        or y_c > rve_dims[1] + r
                        or y_c < -r
                        or z_c > rve_dims[2] + r
                        or z_c < -r
                    ):
                        continue
                    if isinstance(i_particle, Sphere):
                        self.particle_tags.append(factory.addSphere(x_c, y_c, z_c, r))

                        self.phase_dim_tag[str(i_particle.phase)].append(
                            (3, self.particle_tags[-1])
                        )

                        factory.synchronize()
                    elif isinstance(i_particle, Ellipsoid):
                        # Particle is an Ellipsoid
                        self.enforce_pbc_flag = False
                        # Do not enforce periodic boundary conditions
                        fake_radius = 1
                        self.particle_tags.append(
                            factory.addSphere(x_c, y_c, z_c, fake_radius)
                        )
                        # Creating a sphere without rotation
                        # Rotate the disk
                        factory.synchronize()
                        affine_tags = [(3, self.particle_tags[-1])]
                        # affine_tags.extend(
                        #     model.getBoundary([(3, particle_tags[k_particle_image])]))
                        factory.dilate(
                            affine_tags,
                            x_c,
                            y_c,
                            z_c,
                            i_particle.semi_axis_1,
                            i_particle.semi_axis_2,
                            i_particle.semi_axis_3,
                        )
                        factory.rotate(
                            affine_tags,
                            x_c,
                            y_c,
                            z_c,
                            i_particle.rotation_axis[0],
                            i_particle.rotation_axis[1],
                            i_particle.rotation_axis[2],
                            i_particle.angle,
                        )

                        self.phase_dim_tag[str(i_particle.phase)].append(
                            (3, self.particle_tags[-1])
                        )

                        factory.synchronize()

    def enforce_pbc(self, rve_dims):
        """Enforce the pbcs for all the boundaries of the microstructure."""
        factory = gmsh.model.occ
        factory.synchronize()

        if len(rve_dims) == 2:
            for i_direction in range(2):
                self.enforce_pbc_one_way(rve_dims, i_direction, 1)
        elif len(rve_dims) == 3:
            if self.enforce_pbc_flag:
                for i_direction in range(3):
                    self.enforce_pbc_one_way(rve_dims, i_direction, 1)
                    self.enforce_pbc_one_way(rve_dims, i_direction, 2)

    def enforce_pbc_one_way(self, rve_dims, direction, dim, eps=1e-3):
        """Enforce the pbcs in one particular direciton.

        Parameters
        ----------
        rve_dims: list(float)
            Dimensions of the microstructure in each spatial direction.

        direction: {0, 1, 2}
            The pbc will enforce between the faces/edges normal to the x axis(0), y axis(1)
            or z axis(2).

        dim: {1, 2}:
            Dimension of the bounding element. 1 for edges and 2 for faces.

        eps: float, optional
            Tolerance for the bounding boxes used to enforce the pbc.
        """
        factory = gmsh.model.occ
        gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
        trans_vec = [0, 0, 0]
        trans_vec[direction] = rve_dims[direction]
        normal_plane = list(rve_dims) if len(rve_dims) == 3 else list(rve_dims) + [0]
        normal_plane[direction] = 0
        translation_4_mat = [
            1,
            0,
            0,
            trans_vec[0],
            0,
            1,
            0,
            trans_vec[1],
            0,
            0,
            1,
            trans_vec[2],
            0,
            0,
            0,
            1,
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
        for i_surf in main_face:
            # Then we get the bounding box of each left surface
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(
                i_surf[0], i_surf[1]
            )
            # We translate the bounding box to the right and look for surfaces inside
            # it:
            opposite_surf = gmsh.model.getEntitiesInBoundingBox(
                xmin - eps + trans_vec[0],
                ymin - eps + trans_vec[1],
                zmin - eps + trans_vec[2],
                xmax + eps + trans_vec[0],
                ymax + eps + trans_vec[1],
                zmax + eps + trans_vec[2],
                dim,
            )
            # For all the matches, we compare the corresponding bounding boxes...
            for j_surf in opposite_surf:
                xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = gmsh.model.getBoundingBox(
                    j_surf[0], j_surf[1]
                )
                # ...and if they match, we apply the periodicity constraint
                if (
                    abs(xmin2 - trans_vec[0] - xmin) < eps
                    and abs(xmax2 - trans_vec[0] - xmax) < eps
                    and abs(ymin2 - trans_vec[1] - ymin) < eps
                    and abs(ymax2 - trans_vec[1] - ymax) < eps
                    and abs(zmin2 - trans_vec[2] - zmin) < eps
                    and abs(zmax2 - trans_vec[2] - zmax) < eps
                ):
                    gmsh.model.mesh.setPeriodic(
                        dim, [j_surf[1]], [i_surf[1]], translation_4_mat
                    )
                    # Ensuring periodicity

        factory.synchronize()

    def gmsh_to_links(self, meshfile, title, dim, list_phases, matrix_phase):
        """
        Write an input file for LINKS using a gmsh mesh saved at meshfile.

        The header for the LINKS file can be found in /resources.

        Parameters
        ----------
        meshfile: string
            Path to the gmsh file containing the FEM mesh.

        title: string
            Name of the file where the input file to LINKS will be saved.

        dim: int
            Number of dimnesions of the problem.

        list_phases: list(str)
            List containing the names of the phases

        matrix_phase: str
            Name of the matrix phase
        """
        mesh = readMesh(meshfile)

        node_id, coord = mesh.getAllNodes()
        element_id, _, _ = mesh.getElementsByDim(dim)
        elem_mat = [0 for _ in list_phases]
        # Initializing the list containing the elements
        con_mat = [0 for _ in list_phases]
        # Initializing the list containing the connectivity lists
        type_mat = [0 for _ in list_phases]
        # Initializing the list containing the type of elements
        elem_mat[0], con_mat[0], type_mat[0] = mesh.getElementsByName(
            "Phase " + str(matrix_phase)
        )
        list_phases.remove(matrix_phase)
        # Collecting the information about the matrix phase
        for i_phase, _ in enumerate(list_phases):
            # Running through all the phases
            (
                elem_mat[i_phase + 1],
                con_mat[i_phase + 1],
                type_mat[i_phase + 1],
            ) = mesh.getElementsByName("Phase " + str(list_phases[i_phase]))
            # Collecting the information about phase i_phase
        if dim == 2:
            shutil.copy(
                "geommicgen/resources/LINKS_header.dat",
                os.path.join(title, "femsh.rve"),
            )
            # Header for 2D
        elif dim == 3:
            shutil.copy(
                "geommicgen/resources/LINKS_header_3d.dat",
                os.path.join(title, "femsh.rve"),
            )
            # Header for 3D

        with open(os.path.join(title, "femsh.rve"), "a") as dat:
            dat.write("\n\nNODE_COORDINATES {0} CARTESIAN".format(len(node_id)))
            for i_coord, i_node_id in zip(coord, node_id):
                dat.write(
                    "\n{0} {1} {2} {3}".format(
                        i_node_id, i_coord[0], i_coord[1], i_coord[2]
                    )
                )
            # Writing the coordinates of each node
            dat.write("\n\nELEMENTS {0}".format(len(element_id)))
            for i_phase in range(len(list_phases) + 1):
                # For each phase
                for j_node in range(len(elem_mat[i_phase])):
                    # For each node
                    dat.write(
                        "\n{0} {1} ".format(elem_mat[i_phase][j_node], i_phase + 1)
                    )
                    for k_con in con_mat[i_phase][j_node]:
                        dat.write("{0} ".format(k_con))


class RegularGridMeshGenerator(MeshGenerator):
    def __init__(self, n_voxels_dims, rve_dims):
        """
        Initialize a RegularGridMeshGenerator class object.

        Attributes
        ----------
        n_voxels_dims: array(int)
            Number of voxels in each spatial direction.

        rve_dims: list(float)
            Dimensions of the microstucutre in each spatial direction.
        """
        if len(n_voxels_dims) != len(rve_dims):
            raise ValueError(
                "Number of voxels specified incompatible with RVE dimensions."
            )
        if any([n_voxel < 1 for n_voxel in n_voxels_dims]):
            raise ValueError(
                "The number of voxels in each direction has to be larger than 1."
            )
        self.n_voxels_dims = np.array(n_voxels_dims)

    def generate_mesh(self, microstructure_sample, sample_dir):
        """
        Generate the mesh for the Fast Fourier Transform method.

        Parameters
        ----------
        microstructure_sample: `.Microstructure`
            Microstructure to be meshed.

        sample_dir: str
            Path to store the meshes.
        """
        rve_dims = np.array(microstructure_sample.rve_dims)
        pixel_dims = rve_dims / self.n_voxels_dims
        # Dimension of the pixels
        if len(microstructure_sample.rve_dims) == 2:
            # This is a 2D dimnensional problem
            regular_grid = np.full(
                (self.n_voxels_dims[0], self.n_voxels_dims[1]),
                int(microstructure_sample.matrix_phase),
                dtype=int,
            )
            # Initializing the regular grid
            for (i_row, j_column) in [
                (i, j)
                for i in range(self.n_voxels_dims[0])
                for j in range(self.n_voxels_dims[1])
            ]:
                # Running through the pixels from left to right, bottom to top, front to
                # back
                center_pixel_i_j = np.array(
                    [
                        (i_row + 0.5) * pixel_dims[0],
                        (j_column + 0.5) * pixel_dims[1],
                    ]
                )
                # Center of the pixel corresponding to row i_row, column j_column and
                # layer k_layer
                for l_particle in microstructure_sample.particles:
                    # Running through all the particles
                    diff_in_box = l_particle.position_center - center_pixel_i_j
                    # Difference vector between the center of the two ellipses
                    diff_nearest_other = rve_dims * np.round(diff_in_box / rve_dims)
                    # Vector from the particle whose center is in the RVE to the nearest
                    # image
                    if l_particle.point_inside(center_pixel_i_j + diff_nearest_other):
                        # The center of the pixel is inside particle k_particle
                        regular_grid[i_row, j_column] = l_particle.phase
                        # Setting pixel [i_row, j_column, k_layer] as belong to the
                        # phase of particle k_particle

            filename = "{0[0]}_{0[1]}.{1}".format(self.n_voxels_dims, "rgmsh")
            np.save(
                os.path.join(sample_dir, "meshes", filename),
                regular_grid,
            )

        elif len(microstructure_sample.rve_dims) == 3:
            # This is a 2D dimnensional problem
            regular_grid = np.full(
                (self.n_voxels_dims[0], self.n_voxels_dims[1], self.n_voxels_dims[2]),
                int(microstructure_sample.matrix_phase),
                dtype=int,
            )
            # Initializing the regular grid
            for (i_row, j_column, k_layer) in [
                (i, j, k)
                for i in range(self.n_voxels_dims[0])
                for j in range(self.n_voxels_dims[1])
                for k in range(self.n_voxels_dims[2])
            ]:
                # Running through the pixels from left to right, bottom to top, front to
                # back
                center_pixel_i_j_k = np.array(
                    [
                        (i_row + 0.5) * pixel_dims[0],
                        (j_column + 0.5) * pixel_dims[1],
                        (k_layer + 0.5) * pixel_dims[2],
                    ]
                )
                # Center of the pixel corresponding to row i_row, column j_column and
                # layer k_layer
                for l_particle in microstructure_sample.particles:
                    # Running through all the particles
                    diff_in_box = l_particle.position_center - center_pixel_i_j_k
                    # Difference vector between the center of the two ellipses
                    diff_nearest_other = rve_dims * np.round(diff_in_box / rve_dims)
                    # Vector from the particle whose center is in the RVE to the nearest
                    # image
                    if l_particle.point_inside(center_pixel_i_j_k + diff_nearest_other):
                        # The center of the pixel is inside particle k_particle
                        regular_grid[i_row, j_column, k_layer] = l_particle.phase
                        # Setting pixel [i_row, j_column, k_layer] as belong to the
                        # phase of particle k_particle

            filename = "{0[0]}_{0[1]}_{0[2]}.{1}".format(self.n_voxels_dims, "rgmsh")
            np.save(
                os.path.join(
                    sample_dir,
                    "meshes",
                    filename,
                ),
                regular_grid,
            )

        if True:
            plot_voxels(
                regular_grid,
                Particle.matrix_phase,
                Particle.list_phases,
                Particle.file_path
                + "_"
                + str(n_voxels_dims[0])
                + "_"
                + str(n_voxels_dims[1])
                + "_"
                + str(n_voxels_dims[0])
                + "."
                + disc_ext,
            )
        # Ploting the regular grid
