"""Module containing the meshing inteface classes.

This module contains the FEMMeshGenerator and the RGMeshGenerator classes, whose object
instances generate finite element method mesh and regular grid meshes, respectively, using
the information form Microstructure classs object.
"""
import os
import sys
import shutil
import abc
import time

# GMSH module
import gmsh

# pylint: disable=import-error
import geommicgen.iofuncs.printing as print_funcs
from gmsh2links.main import readMesh

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
# Importing the particle class
from geommicgen.microstructure.particleclasses import (
    Disk,
    Ellipse,
    Ellipsoid,
    Sphere,
    CylindricalFiber,
    Cylinder,
)


import numpy as np


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
        Descriptors of the elements used to mesh the microstructure.

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
        to  False if there are Ellipsoids or CylindricalFibers in the microstructure. Gmsh
        has not been able to produce  microstructures containing Ellipsoids or
        CylindricalFibers and with pbcs.

    time: float
        Time in seconds to generate the mesh.

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
            Dimensions of the microstructure in each spatial direciton.

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
        self.output_term = kwargs.get("output_term", True)
        self.particle_tags = []
        self.box_tag = None
        self.phase_dim_tag = None
        self.enforce_pbc_flag = True
        self.time = None

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
        start = time.time()
        print_funcs.print_to_file(
            "Finite Element Mesh using Gmsh",
            to_terminal=self.output_term,
            to_screen=self.output_term,
        )
        print_funcs.print_to_file(
            "." * 80 + "\n", to_terminal=self.output_term, to_screen=self.output_term
        )
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
        self.time = time.time() - start
        print_funcs.print_to_file(
            "Time ellapsed: {0:.3f}s\n".format(self.time),
            to_terminal=self.output_term,
            to_screen=self.output_term,
        )
        #

    def init_gmsh_model(self):
        """Initialize and set the options for the gmsh model."""
        print_funcs.print_to_file(
            "\t> Initialising Gmsh model and setting options",
            to_terminal=self.output_term,
            to_screen=self.output_term,
        )
        print_funcs.print_to_file(
            "\t\t- Element type: {0}\n".format(self.element_type),
            to_terminal=self.output_term,
            to_screen=self.output_term,
        )
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
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
        print_funcs.print_to_file(
            "\t> Writing .vtk and .msh files.",
            to_terminal=self.output_term,
            to_screen=self.output_term,
        )
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

        print_funcs.print_to_file(
            "\t\t- {0}\n".format(vtk),
            to_terminal=self.output_term,
            to_screen=self.output_term,
        )

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
            self.box_tag = factory.addRectangle(
                0,
                0,
                0,
                rve_dims[0],
                rve_dims[1],
            )
        elif dim == 3:
            self.box_tag = factory.addBox(
                0, 0, 0, rve_dims[0], rve_dims[1], rve_dims[2]
            )

        # RVE

        self.phase_dim_tag = {
            phase_name: [] for phase_name in microstructure_sample.phases
        }
        print_funcs.print_to_file(
            "\t> Adding particles to the model",
            to_terminal=self.output_term,
            to_screen=self.output_term,
        )
        for i_particle_ind, i_particle in enumerate(particles):
            # Running through all the particles
            self.add_particle_pbc_to_model(i_particle, rve_dims)
            print(
                "\t\t- Particle {0} of {1}".format(i_particle_ind + 1, len(particles))
            )
            if i_particle_ind + 1 != len(particles):
                print_funcs.print_to_file(
                    "\033[F\033[K",
                    end="",
                    to_terminal=self.output_term,
                    to_screen=False,
                )
        print_funcs.print_to_file(
            "", to_terminal=self.output_term, to_screen=self.output_term
        )

        print_funcs.print_to_file(
            "\t> Processing model\n",
            to_terminal=self.output_term,
            to_screen=self.output_term,
        )
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
        # gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self.mesh_size_min)

        # Generate a 3D mesh
        print_funcs.print_to_file(
            "\t> Generating mesh\n",
            to_terminal=self.output_term,
            to_screen=self.output_term,
        )
        model.mesh.generate(dim)
        if model.mesh.getLastEntityError():
            print_funcs.print_to_file(
                "\t\t- WARNING: Gmsh detected an Error",
                to_terminal=self.output_term,
                to_screen=self.output_term,
            )

        print_funcs.print_to_file(
            "\t> Optimizing mesh\n",
            to_terminal=self.output_term,
            to_screen=self.output_term,
        )
        model.mesh.optimize("HighOrder", force=False, niter=10)
        if model.mesh.getLastEntityError():
            print_funcs.print_to_file(
                "\t\t- WARNING: Gmsh detected an Error",
                to_terminal=self.output_term,
                to_screen=self.output_term,
            )

        self.enforce_pbc(rve_dims)
        # Repeated becaused gmsh sometines behaves unpredictably

    def add_particle_pbc_to_model(self, i_particle, rve_dims, add_pbc_images=True):
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

        # Periodic images considered
        if add_pbc_images:
            pbc_images = [-1, 0, 1]
        else:
            pbc_images = [0]

        for (j_pbc, p_pbc) in [
            (j_pbc, p_pbc) for j_pbc in pbc_images for p_pbc in pbc_images
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
                    self.enforce_pbc_flag = False
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
                for l_pbc in pbc_images:
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
                    elif isinstance(i_particle, Cylinder):
                        i_particle: Cylinder
                        self.enforce_pbc_flag = False
                        r_x = i_particle.r_cyl
                        r_y = i_particle.r_cyl
                        face_tag = factory.addDisk(
                            x_c, y_c, z_c - i_particle.length / 2, r_x, r_y
                        )
                        # Saving the properties of the particles
                        extrude_direction = [0, 0, i_particle.length]
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
                        if i_particle.polar_angle != 0:
                            factory.rotate(
                                [(3, self.particle_tags[-1])],
                                x_c,
                                y_c,
                                z_c,
                                -i_particle.sym_axis_unit_vec[1],
                                i_particle.sym_axis_unit_vec[0],
                                0,
                                i_particle.polar_angle,
                            )
                            # Rotating the fiber face to the correct plane as it was ploted
                            # in the xy plane

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
        print_funcs.print_to_file(
            "\t> Writing LINKS input file",
            to_terminal=self.output_term,
            to_screen=self.output_term,
        )
        stdout = sys.stdout
        sys.stdout = None
        mesh = readMesh(meshfile)
        sys.stdout = stdout

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

        print_funcs.print_to_file(
            "\t\t- {0}\n".format(os.path.join(title, "femsh.rve")),
            to_terminal=self.output_term,
            to_screen=self.output_term,
        )


class RegularGridMeshGenerator(MeshGenerator):
    """
    Class for the mesh generator of finite element meshes.

    Attributes
    ----------
    n_voxels_dims: array(int)
        Number of voxels in each spatial direction.

    slice_dir: int
        Direction along which a 3D microstructure is to be sliced to obtain the 2D sections.

    time: float
        CPU time taken to generate the regular grid.
    """

    def __init__(self, n_voxels_dims, rve_dims, **kwargs):
        """
        Initialize a RegularGridMeshGenerator class object.

        Attributes
        ----------
        n_voxels_dims: array(int)
            Number of voxels in each spatial direction.

        rve_dims: list(float)
            Dimensions of the microstructure in each spatial direction.
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
        self.slice_dir = kwargs.get("slice_dir", None)
        self.time = None

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
        start = time.time()
        print_funcs.print_to_file("Regular mesh grid")
        print_funcs.print_to_file("." * 80 + "\n")
        rve_dims = np.array(microstructure_sample.rve_dims)
        pixel_dims = rve_dims / self.n_voxels_dims
        dim = len(rve_dims)

        if dim == 2:
            regular_grid = np.full(
                (self.n_voxels_dims[0], self.n_voxels_dims[1]),
                int(microstructure_sample.matrix_phase),
                dtype=int,
            )
            # Initializing the regular grid
            print_funcs.print_to_file("\t> Processing particles")
            for l_particle_ind, l_particle in enumerate(
                microstructure_sample.particles
            ):
                lim_x = [
                    l_particle.support_function(np.array([-1, 0, 0]))[0],
                    l_particle.support_function(np.array([1, 0, 0]))[0],
                ]
                lim_y = [
                    l_particle.support_function(np.array([0, -1, 0]))[1],
                    l_particle.support_function(np.array([0, 1, 0]))[1],
                ]
                # Furthest point in each coordingate, both positve and negative
                lim_rows = [lim_x[0] // pixel_dims[0], lim_x[1] // pixel_dims[0] + 1]
                lim_columns = [lim_y[0] // pixel_dims[1], lim_y[1] // pixel_dims[1] + 1]
                # Corresponding pixel to the furthest points
                for (i_row, j_column) in (
                    (i_row, j_column)
                    for i_row in range(int(lim_rows[0]), int(lim_rows[1] + 1))
                    for j_column in range(int(lim_columns[0]), int(lim_columns[1] + 1))
                ):
                    # Running through all the pixels in the bounding rectange for the
                    # particle
                    center_pixel_i_j = np.array(
                        [
                            (i_row + 0.5) * pixel_dims[0],
                            (j_column + 0.5) * pixel_dims[1],
                        ]
                    )
                    if l_particle.point_inside(center_pixel_i_j, rve_dims):
                        # The center of the pixel is inside particle k_particle

                        regular_grid[
                            np.mod(i_row, self.n_voxels_dims[0]),
                            np.mod(j_column, self.n_voxels_dims[1]),
                        ] = l_particle.phase
                        # Setting pixel [i_row, j_column, k_layer] as belong to the
                        # phase of particle k_particle
                print(
                    "\t\t- Particle {0} of {1}".format(
                        l_particle_ind + 1, len(microstructure_sample.particles)
                    )
                )
                if l_particle_ind + 1 != len(microstructure_sample.particles):
                    print("\033[F\033[K", end="")
            print_funcs.print_to_file("")
            filename = "{0[0]}_{0[1]}".format(self.n_voxels_dims)
        elif dim == 3:
            regular_grid = np.full(
                (self.n_voxels_dims[0], self.n_voxels_dims[1], self.n_voxels_dims[2]),
                int(microstructure_sample.matrix_phase),
                dtype=int,
            )
            print_funcs.print_to_file("\t> Processing particles")
            # Initializing the regular grid
            for l_particle_ind, l_particle in enumerate(
                microstructure_sample.particles
            ):
                lim_x = [
                    l_particle.support_function(np.array([-1, 0, 0]))[0],
                    l_particle.support_function(np.array([1, 0, 0]))[0],
                ]
                lim_y = [
                    l_particle.support_function(np.array([0, -1, 0]))[1],
                    l_particle.support_function(np.array([0, 1, 0]))[1],
                ]
                lim_z = [
                    l_particle.support_function(np.array([0, 0, -1]))[2],
                    l_particle.support_function(np.array([0, 0, 1]))[2],
                ]
                # Furthest point in each coordinate, both positve and negative
                lim_rows = [lim_x[0] // pixel_dims[0], lim_x[1] // pixel_dims[0] + 1]
                lim_columns = [lim_y[0] // pixel_dims[1], lim_y[1] // pixel_dims[1] + 1]
                lim_layers = [lim_z[0] // pixel_dims[2], lim_z[1] // pixel_dims[2] + 1]
                # Corresponding voxel to the furthest points
                for (i_row, j_column, k_layer) in (
                    (i_row, j_column, k_layer)
                    for i_row in range(int(lim_rows[0]), int(lim_rows[1] + 1))
                    for j_column in range(int(lim_columns[0]), int(lim_columns[1] + 1))
                    for k_layer in range(int(lim_layers[0]), int(lim_layers[1] + 1))
                ):
                    # Running through all the voxels in the bounding box for the
                    # particle
                    center_pixel_i_j = np.array(
                        [
                            (i_row + 0.5) * pixel_dims[0],
                            (j_column + 0.5) * pixel_dims[1],
                            (k_layer + 0.5) * pixel_dims[2],
                        ]
                    )
                    if l_particle.point_inside(center_pixel_i_j, rve_dims):
                        # The center of the pixel is inside particle k_particle

                        regular_grid[
                            np.mod(i_row, self.n_voxels_dims[0]),
                            np.mod(j_column, self.n_voxels_dims[1]),
                            np.mod(k_layer, self.n_voxels_dims[2]),
                        ] = l_particle.phase
                        # Setting pixel [i_row, j_column, k_layer] as belong to the
                        # phase of particle k_particle

                print(
                    "\t\t- Particle {0} of {1}".format(
                        l_particle_ind + 1, len(microstructure_sample.particles)
                    )
                )
                if l_particle_ind + 1 != len(microstructure_sample.particles):
                    print("\033[F\033[K", end="")
            print_funcs.print_to_file("")
            filename = "{0[0]}_{0[1]}_{0[2]}".format(self.n_voxels_dims)

        result_dir = os.path.join(sample_dir, "meshes")
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        print_funcs.print_to_file("\t> Writing mesh file.")
        if dim == 2 or (dim == 3 and self.slice_dir is None):
            file_path = os.path.join(result_dir, filename + ".rgmsh")
            np.save(file_path, regular_grid)
            print_funcs.print_to_file("\t\t- {0}\n".format(file_path))

        save_plot = False
        if save_plot:
            from postproc.plotfuncs.plotting_functions import plot_pixels, plot_voxels

            if len(microstructure_sample.rve_dims) == 2:
                plot_pixels(
                    regular_grid,
                    os.path.join(result_dir, filename + ".pdf"),
                    show=False,
                )
            elif len(microstructure_sample.rve_dims) == 3:
                if self.slice_dir is None:

                    plot_voxels(
                        regular_grid,
                        microstructure_sample.matrix_phase,
                        list(microstructure_sample.phases.keys()),
                        os.path.join(result_dir, filename + ".pdf"),
                        show=False,
                    )

                else:
                    for j_ind_slice in range(self.n_voxels_dims[self.slice_dir]):
                        lims = [[None, None], [None, None], [None, None]]
                        lims[self.slice_dir][0] = j_ind_slice
                        lims[self.slice_dir][1] = j_ind_slice + 1
                        plot_pixels(
                            regular_grid[
                                lims[0][0] : lims[0][1],
                                lims[1][0] : lims[1][1],
                                lims[2][0] : lims[2][1],
                            ]
                            .reshape(
                                (
                                    self.n_voxels_dims[np.mod(self.slice_dir + 1, 3)],
                                    self.n_voxels_dims[np.mod(self.slice_dir + 1, 3)],
                                )
                            )
                            .T,
                            os.path.join(
                                result_dir,
                                "{0}_{1}_{2}.pdf".format(
                                    filename, self.slice_dir, j_ind_slice
                                ),
                            ),
                            show=False,
                        )

        self.time = time.time() - start
        print_funcs.print_to_file("Time ellapsed: {0:.3f}s\n".format(self.time))
