"""Module containing the meshing inteface classes.

This module contains the FEMMeshGenerator and the RGMeshGenerator classes, whose object
instances generate finite element method mesh and regular grid meshes, respectively, using
the information form Microstructure classs object.
"""
import os
import sys
import shutil
import abc


# GMSH module
import gmsh

# Simple math tools

# Finite element mesh conversor to LINKS
from gmsh2links.main import readMesh
from microstructure.particle_classes import (
    Disk,
    Particle,
    Ellipse,
    Ellipsoid,
    Sphere,
    CylindricalFiber,
)

# Importing the particle class
import numpy as np
import errors.error_classes as errors

# from plotting_functions import plotPixels, plotVoxels


class MeshGenerator(abc.ABC):
    """Class for the mesh generators."""

    @abc.abstractmethod
    def generate_mesh(self, microstructure_sample, file_path):
        """Generate a mesh for *microstructure_sample*."""


class FEMMeshGenerator(MeshGenerator):
    """Class for the mesh generator of finite element meshes."""

    known_element = {"tri3", "tri6", "quad4", "quad8", "tetra4", "tetra10"}
    known_element_descriptors = {
        "tri3": {
            "mesh_alg": 5,
            "force_recomb_all": 0,
            "element_order": 1,
            "recomb_alg": 1,
            "element_order_incomp": 0,
        },
        "tri6": {
            "mesh_alg": 5,
            "force_recomb_all": 0,
            "element_order": 2,
            "recomb_alg": 1,
            "element_order_incomp": 0,
        },
        "quad4": {
            "mesh_alg": 5,
            "force_recomb_all": 1,
            "element_order": 1,
            "recomb_alg": 1,
            "element_order_incomp": 0,
        },
        "quad8": {
            "mesh_alg": 5,
            "force_recomb_all": 1,
            "element_order": 2,
            "recomb_alg": 1,
            "element_order_incomp": 1,
        },
        "tetra4": {
            "mesh_alg": 5,
            "force_recomb_all": 0,
            "element_order": 1,
            "recomb_alg": 1,
            "element_order_incomp": 0,
        },
        "tetra10": {
            "mesh_alg": 5,
            "force_recomb_all": 0,
            "element_order": 1,
            "recomb_alg": 1,
            "element_order_incomp": 0,
        },
    }

    def __init__(self, mesh_size, element_type, **kwargs):

        if mesh_size < 0:
            raise ValueError
        self.mesh_size = mesh_size
        if element_type not in FEMMeshGenerator.known_element:
            raise ValueError
        self.element_type = element_type
        self.descriptors_element_type = FEMMeshGenerator.known_element_descriptors[
            element_type
        ]
        kwargs.update(self.descriptors_element_type)
        self.output_term = kwargs.get("output_term", False)
        self.particle_tags = []

    def generate_mesh(self, microstructure_sample, file_path):
        """
        Generate the mesh for the Finite Element Method.

        Parameters
        ----------
        microstructure_sample: `.Microstructure`
            Microstructure to be meshed.
        """

        gmsh.initialize()
        self.set_gmsh_model(
            **self.descriptors_element_type,
            output_term=self.output_term,
        )
        self.generate_mesh_gmsh(
            microstructure_sample,
            self.mesh_size,
            file_path,
        )
        self.write_mesh_gmsh(microstructure_sample, file_path)

    def set_gmsh_model(
        self,
        mesh_alg=5,
        mesh_alg_3d=1,
        force_recomb_all=0,
        element_order=1,
        recomb_alg=1,
        element_order_incomp=0,
        output_term=0,
    ):
        model = gmsh.model
        factory = model.occ
        gmsh.option.setNumber("General.Terminal", 1)  # output_term)

        # 2D Meshing algorithm
        # --------------------
        # 1 - Mesh Adapt
        # 2 - Automatic
        # 5 - Delaunay (default)
        # 6 - Frontal-Delaunay
        # 7 - BAMG
        # 8 - Frontal-Delaunay for Quads
        # 9 - Packing of Parallelograms
        gmsh.option.setNumber("Mesh.Algorithm", mesh_alg)

        # 3D Meshing algorithm
        # --------------------
        # 1 - Delaunay (default)
        # 2 - Frontal
        # 7 - MMG3D
        # 9 - R-tree
        # 10 - HXT
        gmsh.option.setNumber("Mesh.Algorithm3D", mesh_alg_3d)

        # Characteristic mesh length factor (applied acroos all mesh)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1)

        # Multi-threading
        gmsh.option.setNumber("Mesh.MaxNumThreads1D", 4)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", 4)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", 4)

        # MSH file version
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)

        # Quad/Hex recombination algorithms
        # ---------------------------------
        # 0 - simple
        # 1 - blossom (default)
        # 2 - simple full-quad
        # 3 - blosson full-quad
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", recomb_alg)

        # Force recombination in all surfaces
        gmsh.option.setNumber("Mesh.RecombineAll", force_recomb_all)

        # Number of topological optimization passes of recombined surface meshes (5 by default)
        gmsh.option.setNumber("Mesh.RecombineOptimizeTopology", 5)

        # Force recombination in all volumes
        gmsh.option.setNumber("Mesh.Recombine3DAll", force_recomb_all)

        # Recombination level in 3D
        # -------------------------
        # 0 - hex (default)
        # 1 - hex + prisms
        # 2 - hex + prisms + pyramids
        gmsh.option.setNumber("Mesh.Recombine3DLevel", recomb_alg)

        # Recombination conformity type in 3D meshes
        # ------------------------------------------
        # 0 - nonconforming (default)
        # 1 - trihedra
        # 2 - pyramids + trihedra
        # 2 - pyramids + hexSplit + trihedra
        # 4 - hexSplit + trihedra
        gmsh.option.setNumber("Mesh.Recombine3DConformity", 0)

        # Renumber nodes and elements after mesh generation
        gmsh.option.setNumber("Mesh.Renumber", 1)

        # Save all elements even if they do not belong to physical groups
        gmsh.option.setNumber("Mesh.SaveAll", 0)

        # Number of smoothing step applied to the final mesh
        gmsh.option.setNumber("Mesh.Smoothing", 1)

        # Element order
        gmsh.option.setNumber("Mesh.ElementOrder", element_order)

        # Crete second-order nodes by linear interpolation
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)

        # Second-order incomplete elements
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", element_order_incomp)
        # ==========================================================================================
        # Generate the finite element mesh
        # ==========================================================================================
        # Define model name

    def write_mesh_gmsh(self, microstructure_sample, file_path):
        # Write the mesh to the .msh file
        meshfile_temp = file_path + "_temp.msh"
        meshfile = file_path + ".msh"
        vtk_temp = file_path + "_temp.vtk"
        vtk = file_path + ".vtk"
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

        self.gmsh_to_links(
            meshfile,
            file_path,
            2,
            list(microstructure_sample.phases.keys()),
            microstructure_sample.matrix_phase,
        )
        #

        fin = open(vtk_temp, "rt")
        fout = open(vtk, "wt")

        for line in fin:
            fout.write(line.replace(",", "."))

        fin.close()
        fout.close()
        os.remove(vtk_temp)
        # Sometimes gmsh swaps periods for commas

    def generate_mesh_gmsh(
        self,
        microstructure_sample,
        mesh_size,
        file_path,
    ):
        """
        Generate the mesh for the Finite Element Method in 2D.

        Parameters
        ----------
        particles: list(`.Particle`)
            List containing the particles of the microstructure.

        mesh_size: float
            Size of the mesh.

        output_term: {0, 1}, optional
            Output to the terminal

        mesh_alg: integer, optional
            2D Meshing algorithm
                1: Mesh Adapt
                2: Automatic
                5: Delaunay (default)
                6: Frontal-Delaunay
                7: BAMG
                8: Frontal-Delaunay for Quads
                9: Packing of Parallelograms

        force_recomb_all: {0, 1}, optional
            Recombination into quads.

        element_order: integer, optional
            Order of the element

        recomb_alg: integer, optional
            Quad/Hex recombination algorithms
                0: simple
                1: blossom (default)
                2: simple full-quad
                3: blosson full-quad

        element_order_incomp: {0, 1}, optional
            Remove interior nodes for second order elements
        """
        # ======================================================================================
        # Set up GMSH in Python
        # ======================================================================================
        # Select the geometry engine
        # occ - OpenCASCADE CAD (more advanced)
        # geo - built-in CAD kernel (less sophisticated)
        model = gmsh.model
        factory = model.occ

        title = file_path
        model.add(title)

        mesh_size = self.mesh_size
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
        print(self.particle_tags)

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
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

        # Generate a 3D mesh
        model.mesh.generate(dim)

        model.mesh.optimize("HighOrder", force=False, niter=10)

        self.enforce_pbc(rve_dims)
        # Repeated becaused gmsh sometines behaves unpredictably

    def add_particle_pbc_to_model(self, i_particle, rve_dims):
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
                r_x = i_particle.radius
                r_y = i_particle.radius
                if (
                    x_c > rve_dims[0] + r_x - eps
                    or x_c < -r_x + eps
                    or y_c > rve_dims[1] + r_y - eps
                    or y_c < -r_y + eps
                ):
                    continue
                # Saving the properties of the particles
                self.particle_tags.append(factory.addDisk(x_c, y_c, z_c, r_x, r_y))

                factory.synchronize()
                if isinstance(i_particle, Disk):
                    # Particle is a Disk
                    self.phase_dim_tag[i_particle.phase].append(
                        (2, self.particle_tags[-1])
                    )
                elif isinstance(i_particle, Ellipse):
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
                elif isinstance(i_particle, CylindricalFiber):
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
                            break

                    factory.synchronize()
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
        factory = gmsh.model.occ
        factory.synchronize()

        if len(rve_dims) == 2:
            self.enforce_pbc_2d(rve_dims)
        elif len(rve_dims) == 3:
            self.enforce_pbc_3d(rve_dims)

    def enforce_pbc_one_way(self, rve_dims, direction, dim, eps=1e3):

        factory = gmsh.model.occ
        gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
        trans_vec = [0, 0, 0]
        trans_vec[direction] = rve_dims[direction]
        normal_plane = rve_dims if len(rve_dims) == 3 else rve_dims.append(0)
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
                        1, [j_surf[1]], [i_surf[1]], translation_4_mat
                    )
                    # Ensuring periodicity

        factory.synchronize()

    def enforce_pbc_2d(self, rve_dims):
        gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
        dim = len(rve_dims)
        eps = 1e-3
        # Ask OpenCASCADE to compute more accurate bounding boxes of entities using
        # the STL mesh
        # --------------------------------------------------------------------------------------
        translation_l_r = [1, 0, 0, rve_dims[0], 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        # Translation of the left face of the cube to the rigth face of the cube given as an
        # affine transformation, (4x4), written by row
        l_edge = gmsh.model.getEntitiesInBoundingBox(
            -eps, -eps, -eps, +eps, rve_dims[1] + eps, +eps, 1
        )
        # First we get all surfaces on the left:
        for i_edge in l_edge:
            # Then we get the bounding box of each left surface
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(
                i_edge[0], i_edge[1]
            )
            # We translate the bounding box to the right and look for surfaces inside
            # it:
            r_edge = gmsh.model.getEntitiesInBoundingBox(
                xmin - eps + rve_dims[0],
                ymin - eps,
                zmin - eps,
                xmax + eps + rve_dims[0],
                ymax + eps,
                zmax + eps,
                1,
            )
            # For all the matches, we compare the corresponding bounding boxes...
            for j_edge in r_edge:
                xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = gmsh.model.getBoundingBox(
                    j_edge[0], j_edge[1]
                )
                xmin2 -= rve_dims[0]
                xmax2 -= rve_dims[0]
                # ...and if they match, we apply the periodicity constraint
                if (
                    abs(xmin2 - xmin) < eps
                    and abs(xmax2 - xmax) < eps
                    and abs(ymin2 - ymin) < eps
                    and abs(ymax2 - ymax) < eps
                    and abs(zmin2 - zmin) < eps
                    and abs(zmax2 - zmax) < eps
                ):
                    gmsh.model.mesh.setPeriodic(
                        1, [j_edge[1]], [i_edge[1]], translation_l_r
                    )
                    # Ensuring periodicity

        factory.synchronize()

        gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
        eps = 1e-3

        translation_b_t = [1, 0, 0, 0, 0, 1, 0, rve_dims[1], 0, 0, 1, 0, 0, 0, 0, 1]
        # Translation of the left face of the cube to the rigth face of the cube given as an
        # affine transformation, (4x4), written by row
        b_edge = gmsh.model.getEntitiesInBoundingBox(
            -eps, -eps, -eps, rve_dims[0] + eps, +eps, +eps, 1
        )
        # First we get all surfaces on the left:
        for i_edge in b_edge:
            # Then we get the bounding box of each left surface
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(
                i_edge[0], i_edge[1]
            )
            # We translate the bounding box to the right and look for surfaces inside
            # it:
            t_edge = gmsh.model.getEntitiesInBoundingBox(
                xmin - eps,
                ymin - eps + rve_dims[1],
                zmin - eps,
                xmax + eps,
                ymax + eps + rve_dims[1],
                zmax + eps,
                1,
            )
            # For all the matches, we compare the corresponding bounding boxes...
            for j_edge in t_edge:
                xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = gmsh.model.getBoundingBox(
                    j_edge[0], j_edge[1]
                )
                ymin2 -= rve_dims[1]
                ymax2 -= rve_dims[1]
                # ...and if they match, we apply the periodicity constraint
                if (
                    abs(xmin2 - xmin) < eps
                    and abs(xmax2 - xmax) < eps
                    and abs(ymin2 - ymin) < eps
                    and abs(ymax2 - ymax) < eps
                    and abs(zmin2 - zmin) < eps
                    and abs(zmax2 - zmax) < eps
                ):
                    gmsh.model.mesh.setPeriodic(
                        1, [j_edge[1]], [i_edge[1]], translation_b_t
                    )

    def enforce_pbc_3d(self, rve_dims):
        if self.enforce_pbc:
            gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
            eps = 1e-4
            # Ask OpenCASCADE to compute more accurate bounding boxes of entities using
            # the STL mesh
            # --------------------------------------------------------------------------------------
            translation_l_r = [1, 0, 0, rve_dims[0], 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
            # Translation of the left face of the cube to the rigth face of the cube given as an
            # affine transformation, (4x4), written by row
            l_face = gmsh.model.getEntitiesInBoundingBox(
                -eps, -eps, -eps, +eps, rve_dims[1] + eps, rve_dims[2] + eps, 2
            )
            # First we get all surfaces on the left:
            for i_surf in l_face:
                # Then we get the bounding box of each left surface
                xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(
                    i_surf[0], i_surf[1]
                )
                # We translate the bounding box to the right and look for surfaces inside
                # it:
                r_face = gmsh.model.getEntitiesInBoundingBox(
                    xmin - eps + rve_dims[0],
                    ymin - eps,
                    zmin - eps,
                    xmax + eps + rve_dims[0],
                    ymax + eps,
                    zmax + eps,
                    2,
                )
                # For all the matches, we compare the corresponding bounding boxes...
                for j_surf in r_face:
                    (
                        xmin2,
                        ymin2,
                        zmin2,
                        xmax2,
                        ymax2,
                        zmax2,
                    ) = gmsh.model.getBoundingBox(j_surf[0], j_surf[1])
                    xmin2 -= rve_dims[0]
                    xmax2 -= rve_dims[0]
                    # ...and if they match, we apply the periodicity constraint
                    if (
                        abs(xmin2 - xmin) < eps
                        and abs(xmax2 - xmax) < eps
                        and abs(ymin2 - ymin) < eps
                        and abs(ymax2 - ymax) < eps
                        and abs(zmin2 - zmin) < eps
                        and abs(zmax2 - zmax) < eps
                    ):
                        gmsh.model.mesh.setPeriodic(
                            2, [j_surf[1]], [i_surf[1]], translation_l_r
                        )
                        print(j_surf[1], i_surf[1])
                        print(gmsh.model.getBoundary(j_surf))
                        print(gmsh.model.getBoundary(i_surf))
            # --------------------------------------------------------------------------------------
            # Ensuring periodicity
            # --------------------------------------------------------------------------------------
            gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
            translation_b_t = [1, 0, 0, 0, 0, 1, 0, rve_dims[1], 0, 0, 1, 0, 0, 0, 0, 1]
            # Translation of the top face of the cube to the bottom face of the cube given as an
            # affine transformation, (4x4), written by row
            b_face = gmsh.model.getEntitiesInBoundingBox(
                -eps, -eps, -eps, rve_dims[0] + eps, +eps, rve_dims[2] + eps, 2
            )
            # First we get all surfaces on the bottom:
            for i_surf in b_face:
                # Then we get the bounding box of each bottom surface
                xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(
                    i_surf[0], i_surf[1]
                )
                # We translate the bounding box upwards and look for surfaces inside
                # it:
                t_face = gmsh.model.getEntitiesInBoundingBox(
                    xmin - eps,
                    ymin - eps + rve_dims[1],
                    zmin - eps,
                    xmax + eps,
                    ymax + eps + rve_dims[1],
                    zmax + eps,
                    2,
                )
                # For all the matches, we compare the corresponding bounding boxes...
                for j_surf in t_face:
                    (
                        xmin2,
                        ymin2,
                        zmin2,
                        xmax2,
                        ymax2,
                        zmax2,
                    ) = gmsh.model.getBoundingBox(j_surf[0], j_surf[1])

                    ymin2 -= rve_dims[1]
                    ymax2 -= rve_dims[1]
                    # ...and if they match, we apply the periodicity constraint
                    if (
                        abs(xmin2 - xmin) < eps
                        and abs(xmax2 - xmax) < eps
                        and abs(ymin2 - ymin) < eps
                        and abs(ymax2 - ymax) < eps
                        and abs(zmin2 - zmin) < eps
                        and abs(zmax2 - zmax) < eps
                    ):
                        gmsh.model.mesh.setPeriodic(
                            2, [j_surf[1]], [i_surf[1]], translation_b_t
                        )
                        print(j_surf[1], i_surf[1])
                        print(gmsh.model.getBoundary(j_surf))
                        print(gmsh.model.getBoundary(i_surf))
            # --------------------------------------------------------------------------------------
            gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
            translation_f_b = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, rve_dims[2], 0, 0, 0, 1]
            # Translation of the front face of the cube to the back face of the cube given as an
            # affine transformation, (4x4), written by row
            f_face = gmsh.model.getEntitiesInBoundingBox(
                -eps, -eps, -eps, rve_dims[0] + eps, rve_dims[1] + eps, +eps, 2
            )
            # First we get all surfaces on the front:
            for i_surf in f_face:
                # Then we get the bounding box of each front surface
                xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(
                    i_surf[0], i_surf[1]
                )
                # We translate the bounding box to the back and look for surfaces inside
                # it:
                b_face = gmsh.model.getEntitiesInBoundingBox(
                    xmin - eps,
                    ymin - eps,
                    zmin - eps + rve_dims[2],
                    xmax + eps,
                    ymax + eps,
                    zmax + eps + rve_dims[2],
                    2,
                )
                # For all the matches, we compare the corresponding bounding boxes...
                for j_surf in b_face:
                    (
                        xmin2,
                        ymin2,
                        zmin2,
                        xmax2,
                        ymax2,
                        zmax2,
                    ) = gmsh.model.getBoundingBox(j_surf[0], j_surf[1])
                    zmin2 -= rve_dims[2]
                    zmax2 -= rve_dims[2]
                    # ...and if they match, we apply the periodicity constraint
                    if (
                        abs(xmin2 - xmin) < eps
                        and abs(xmax2 - xmax) < eps
                        and abs(ymin2 - ymin) < eps
                        and abs(ymax2 - ymax) < eps
                        and abs(zmin2 - zmin) < eps
                        and abs(zmax2 - zmax) < eps
                    ):
                        gmsh.model.mesh.setPeriodic(
                            2, [j_surf[1]], [i_surf[1]], translation_f_b
                        )
                        print(j_surf[1], i_surf[1])
                        print(gmsh.model.getBoundary(j_surf))
                        print(gmsh.model.getBoundary(i_surf))

            gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
            eps = 1e-4
            # Ask OpenCASCADE to compute more accurate bounding boxes of entities using
            # the STL mesh
            # --------------------------------------------------------------------------------------
            translation_l_r = [1, 0, 0, rve_dims[0], 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
            # Translation of the left face of the cube to the rigth face of the cube given as an
            # affine transformation, (4x4), written by row
            l_face = gmsh.model.getEntitiesInBoundingBox(
                -eps, -eps, -eps, +eps, rve_dims[1] + eps, rve_dims[2] + eps, 1
            )
            # First we get all surfaces on the left:
            for i_surf in l_face:
                # Then we get the bounding box of each left surface
                xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(
                    i_surf[0], i_surf[1]
                )
                # We translate the bounding box to the right and look for surfaces inside
                # it:
                r_face = gmsh.model.getEntitiesInBoundingBox(
                    xmin - eps + rve_dims[0],
                    ymin - eps,
                    zmin - eps,
                    xmax + eps + rve_dims[0],
                    ymax + eps,
                    zmax + eps,
                    1,
                )
                # For all the matches, we compare the corresponding bounding boxes...
                for j_surf in r_face:
                    (
                        xmin2,
                        ymin2,
                        zmin2,
                        xmax2,
                        ymax2,
                        zmax2,
                    ) = gmsh.model.getBoundingBox(j_surf[0], j_surf[1])
                    xmin2 -= rve_dims[0]
                    xmax2 -= rve_dims[0]
                    # ...and if they match, we apply the periodicity constraint
                    if (
                        abs(xmin2 - xmin) < eps
                        and abs(xmax2 - xmax) < eps
                        and abs(ymin2 - ymin) < eps
                        and abs(ymax2 - ymax) < eps
                        and abs(zmin2 - zmin) < eps
                        and abs(zmax2 - zmax) < eps
                    ):
                        gmsh.model.mesh.setPeriodic(
                            1, [j_surf[1]], [i_surf[1]], translation_l_r
                        )
                        print(j_surf[1], i_surf[1])
                        print(gmsh.model.getBoundary(j_surf))
                        print(gmsh.model.getBoundary(i_surf))
            # --------------------------------------------------------------------------------------
            # Ensuring periodicity
            # --------------------------------------------------------------------------------------
            gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
            translation_b_t = [1, 0, 0, 0, 0, 1, 0, rve_dims[1], 0, 0, 1, 0, 0, 0, 0, 1]
            # Translation of the top face of the cube to the bottom face of the cube given as an
            # affine transformation, (4x4), written by row
            b_face = gmsh.model.getEntitiesInBoundingBox(
                -eps, -eps, -eps, rve_dims[0] + eps, +eps, rve_dims[2] + eps, 1
            )
            # First we get all surfaces on the bottom:
            for i_surf in b_face:
                # Then we get the bounding box of each bottom surface
                xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(
                    i_surf[0], i_surf[1]
                )
                # We translate the bounding box upwards and look for surfaces inside
                # it:
                t_face = gmsh.model.getEntitiesInBoundingBox(
                    xmin - eps,
                    ymin - eps + rve_dims[1],
                    zmin - eps,
                    xmax + eps,
                    ymax + eps + rve_dims[1],
                    zmax + eps,
                    1,
                )
                # For all the matches, we compare the corresponding bounding boxes...
                for j_surf in t_face:
                    (
                        xmin2,
                        ymin2,
                        zmin2,
                        xmax2,
                        ymax2,
                        zmax2,
                    ) = gmsh.model.getBoundingBox(j_surf[0], j_surf[1])

                    ymin2 -= rve_dims[1]
                    ymax2 -= rve_dims[1]
                    # ...and if they match, we apply the periodicity constraint
                    if (
                        abs(xmin2 - xmin) < eps
                        and abs(xmax2 - xmax) < eps
                        and abs(ymin2 - ymin) < eps
                        and abs(ymax2 - ymax) < eps
                        and abs(zmin2 - zmin) < eps
                        and abs(zmax2 - zmax) < eps
                    ):
                        gmsh.model.mesh.setPeriodic(
                            1, [j_surf[1]], [i_surf[1]], translation_b_t
                        )
                        print(j_surf[1], i_surf[1])
                        print(gmsh.model.getBoundary(j_surf))
                        print(gmsh.model.getBoundary(i_surf))
            # --------------------------------------------------------------------------------------
            gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
            translation_f_b = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, rve_dims[2], 0, 0, 0, 1]
            # Translation of the front face of the cube to the back face of the cube given as an
            # affine transformation, (4x4), written by row
            f_face = gmsh.model.getEntitiesInBoundingBox(
                -eps, -eps, -eps, rve_dims[0] + eps, rve_dims[1] + eps, +eps, 1
            )
            # First we get all surfaces on the front:
            for i_surf in f_face:
                # Then we get the bounding box of each front surface
                xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(
                    i_surf[0], i_surf[1]
                )
                # We translate the bounding box to the back and look for surfaces inside
                # it:
                b_face = gmsh.model.getEntitiesInBoundingBox(
                    xmin - eps,
                    ymin - eps,
                    zmin - eps + rve_dims[2],
                    xmax + eps,
                    ymax + eps,
                    zmax + eps + rve_dims[2],
                    1,
                )
                # For all the matches, we compare the corresponding bounding boxes...
                for j_surf in b_face:
                    (
                        xmin2,
                        ymin2,
                        zmin2,
                        xmax2,
                        ymax2,
                        zmax2,
                    ) = gmsh.model.getBoundingBox(j_surf[0], j_surf[1])
                    zmin2 -= rve_dims[2]
                    zmax2 -= rve_dims[2]
                    # ...and if they match, we apply the periodicity constraint
                    if (
                        abs(xmin2 - xmin) < eps
                        and abs(xmax2 - xmax) < eps
                        and abs(ymin2 - ymin) < eps
                        and abs(ymax2 - ymax) < eps
                        and abs(zmin2 - zmin) < eps
                        and abs(zmax2 - zmax) < eps
                    ):
                        gmsh.model.mesh.setPeriodic(
                            1, [j_surf[1]], [i_surf[1]], translation_f_b
                        )
                        print(j_surf[1], i_surf[1])
                        print(gmsh.model.getBoundary(j_surf))
                        print(gmsh.model.getBoundary(i_surf))
        # --------------------------------------------------------------------------------------

    def gmsh_to_links(self, meshfile, title, dim, list_phases, matrix_phase):
        """
        Write an input file for LINKS using a gmsh mesh saved at meshfile.

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

        nodeID, coord = mesh.getAllNodes()
        elementID, connectivities, elementType = mesh.getElementsByDim(dim)
        elemMat = len(list_phases) * [0]
        # Initializing the list containing the elements
        conMat = len(list_phases) * [0]
        # Initializing the list containing the connectivity lists
        typeMat = len(list_phases) * [0]
        # Initializing the list containing the type of elements
        elemMat[0], conMat[0], typeMat[0] = mesh.getElementsByName(
            "Phase " + str(matrix_phase)
        )
        list_phases.remove(matrix_phase)
        # Collecting the information about the matrix phase
        for i_phase in range(len(list_phases)):
            # Running through all the phases
            (
                elemMat[i_phase + 1],
                conMat[i_phase + 1],
                typeMat[i_phase + 1],
            ) = mesh.getElementsByName("Phase " + str(list_phases[i_phase]))
            # Collecting the information about phase i_phase
        if dim == 2:
            shutil.copy("geommicgen/resources/LINKS_header.dat", title + ".rve")
            # Header for 2D
        elif dim == 3:
            shutil.copy("geommicgen/resources/LINKS_header_3d.dat", title + ".rve")
            # Header for 3D

        with open(title + ".rve", "a") as dat:
            dat.write("\n\nNODE_COORDINATES {0} CARTESIAN".format(len(nodeID)))
            for i in range(len(nodeID)):
                dat.write(
                    "\n{0} {1} {2} {3}".format(
                        nodeID[i], coord[i][0], coord[i][1], coord[i][2]
                    )
                )
            # Writing the coordinates of each node
            dat.write("\n\nELEMENTS {0}".format(len(elementID)))
            for i_phase in range(len(list_phases) + 1):
                # For each phase
                for j_node in range(len(elemMat[i_phase])):
                    # For each node
                    dat.write(
                        "\n{0} {1} ".format(elemMat[i_phase][j_node], i_phase + 1)
                    )
                    for k_con in conMat[i_phase][j_node]:
                        dat.write("{0} ".format(k_con))


def generate_mesh(particles, options, disc_ext):
    """
    This functions generates a regular grid as an array to be used in an FFT analysis.
    """
    for i_mesh_size in range(len(options["n_voxels_dims"])):
        # Running through the different mesh sizes
        n_voxels_dims = options["n_voxels_dims"][i_mesh_size]
        # current mesh size
        pixel_dims = options["rve_dims"] / n_voxels_dims
        # Dimension of the pixels
        if len(rve_dims) == 2:
            # This is a 2D dimnensional problem
            if len(options["rve_dims"]) == 3:
                # Tridimensional problem with particles obtained by extruding the 2D simulation
                # box
                n_voxels_dims_og = n_voxels_dims
                # Saving the original voxel descretization
                n_voxels_length = n_voxels_dims[particles[0].direction_fibers]
                # Number of voxels in the orthogonal direction to the simulation box
                n_voxels_dims = np.delete(n_voxels_dims, particles[0].direction_fibers)
                # Voxel descritizing the 2D box
            regular_grid = np.full(
                (n_voxels_dims[0], n_voxels_dims[1]),
                int(Particle.matrix_phase),
                dtype=int,
            )
            # Initializing the regular
            for i_row in range(n_voxels_dims[0]):
                # Running through the pixels from left to right
                for j_column in range(n_voxels_dims[1]):
                    # Running thorugh the pixels from bottom to top
                    center_pixel_i_j = np.array(
                        [
                            (i_row + 0.5) * pixel_dims[0],
                            (j_column + 0.5) * pixel_dims[1],
                        ]
                    )
                    # Center of the pixel corresponding to row i_row and column j_column
                    for k_particle in particles:
                        # Running through all the particles
                        diff_in_box = k_particle.position_center - center_pixel_i_j
                        # Difference vector between the center of the two ellipses
                        diff_nearest_other = rve_dims * np.round(diff_in_box / rve_dims)
                        # Vector from the particle whose center is in the RVE to the neares
                        # image
                        if k_particle.point_inside(
                            center_pixel_i_j + diff_nearest_other
                        ):
                            # The center of the pixel is inside particle k_particle
                            regular_grid[i_row, j_column] = k_particle.phase
                            # Setting pixel [i_row, j_column] as belong to the phase of
                            # particle k_particle
            if len(options["rve_dims"]) == 3:
                # Tridimensional problem with particles obtained by extruding the 2D simulation
                # box
                regular_grid = np.stack(
                    [regular_grid for _ in range(n_voxels_length)],
                    axis=particles[0].direction_fibers,
                )
                # Obtaining the extrusion of the 2D box by stacking it in the direction of
                # the fibers
                if True:
                    plotVoxels(
                        regular_grid,
                        Particle.matrix_phase,
                        Particle.list_phases,
                        Particle.file_path
                        + "_"
                        + str(n_voxels_dims_og[0])
                        + "_"
                        + str(n_voxels_dims_og[1])
                        + "_"
                        + str(n_voxels_dims_og[0])
                        + "."
                        + disc_ext,
                    )
                # Ploting the regular grid

                np.save(
                    Particle.file_path
                    + "_"
                    + str(n_voxels_dims_og[0])
                    + "_"
                    + str(n_voxels_dims_og[1])
                    + "_"
                    + str(n_voxels_dims_og[2])
                    + ".rgmsh",
                    regular_grid,
                )
            else:
                if True:
                    plotPixels(
                        regular_grid,
                        Particle.file_path
                        + "_"
                        + str(n_voxels_dims[0])
                        + "_"
                        + str(n_voxels_dims[1])
                        + "."
                        + disc_ext,
                    )
                # Ploting the regular grid
                np.save(
                    Particle.file_path
                    + "_"
                    + str(n_voxels_dims[0])
                    + "_"
                    + str(n_voxels_dims[1])
                    + ".rgmsh",
                    regular_grid,
                )
        elif len(rve_dims) == 3:
            # This is a 2D dimnensional problem
            regular_grid = np.full(
                (n_voxels_dims[0], n_voxels_dims[1], n_voxels_dims[2]),
                int(Particle.matrix_phase),
                dtype=int,
            )
            # Initializing the regular grid
            for i_row in range(n_voxels_dims[0]):
                # Running through the pixels from left to right
                for j_column in range(n_voxels_dims[1]):
                    # Running thorugh the pixels from bottom to top
                    for k_layer in range(n_voxels_dims[2]):
                        # Running thorugh the pixels from bottom to top
                        center_pixel_i_j_k = np.array(
                            [
                                (i_row + 0.5) * pixel_dims[0],
                                (j_column + 0.5) * pixel_dims[1],
                                (k_layer + 0.5) * pixel_dims[2],
                            ]
                        )
                        # Center of the pixel corresponding to row i_row, column j_column and
                        # layer k_layer
                        for l_particle in particles:
                            # Running through all the particles
                            diff_in_box = (
                                l_particle.position_center - center_pixel_i_j_k
                            )
                            # Difference vector between the center of the two ellipses
                            diff_nearest_other = rve_dims * np.round(
                                diff_in_box / Particle.box
                            )
                            # Vector from the particle whose center is in the RVE to the nearest
                            # image
                            if l_particle.point_inside(
                                center_pixel_i_j_k + diff_nearest_other
                            ):
                                # The center of the pixel is inside particle k_particle
                                regular_grid[
                                    i_row, j_column, k_layer
                                ] = l_particle.phase
                                # Setting pixel [i_row, j_column, k_layer] as belong to the
                                # phase of particle k_particle
            if True:
                plotVoxels(
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

            np.save(
                Particle.file_path
                + "_"
                + str(n_voxels_dims[0])
                + "_"
                + str(n_voxels_dims[1])
                + "_"
                + str(n_voxels_dims[2])
                + ".rgmsh",
                regular_grid,
            )

    def checkMeshSpecs(disc_ext, discret_spec_array):
        """Check if the extension has been correctly specified."""
        if disc_ext == "rgmsh":
            # A regular mesh was specified
            necessary_parameters = {"rve_dims", "n_voxels_dims"}
            try:
                if any(
                    [
                        necessary_parameter not in discret_spec_array
                        for necessary_parameter in necessary_parameters
                    ]
                ):
                    # Checking if all the required parameters were supplied
                    raise errors.InsufficientInfoMesh(
                        list(discret_spec_array.keys()), necessary_parameters, disc_ext
                    )
                rve_dims = discret_spec_array["rve_dims"]
                n_voxels_dims = discret_spec_array["n_voxels_dims"]
                # Saving the RVE dims and the number of voxels in each direction
                if rve_dims.shape != (3,) and rve_dims.shape != (2,):
                    # The RVE dims must be given as 1-arrays with 2 or 3 elements
                    raise (
                        errors.UnexpectedValue(
                            rve_dims, "rve_dims", "1-array with shape (2,) or (3,)"
                        )
                    )
                if any(rve_dims < 0):
                    # The RVE dimensions must be positive real numbers
                    raise errors.UnexpectedValue(
                        rve_dims, "rve_dims", "array of positive reals"
                    )
                if len(rve_dims) != len(n_voxels_dims.T):
                    # The dimension of RVE is not compatible with number of voxels specified
                    raise errors.IncompatibleDimension("rve_dims", "n_voxels_dims")
                if any(
                    [
                        not np.issubdtype(n_voxels_dims.flat[i_voxel_dim], np.integer)
                        or n_voxels_dims.flat[i_voxel_dim] < 1
                        for i_voxel_dim in range(n_voxels_dims.size)
                    ]
                ):
                    # The specified number of voxels in any direction must be a positve integer
                    raise errors.UnexpectedValue(
                        n_voxels_dims, "n_voxels_dims", "array of positive integers"
                    )
            except (
                errors.InsufficientInfoMesh,
                errors.IncompatibleDimension,
                errors.UnexpectedValue,
            ) as error:
                error.message()

        elif disc_ext == "femsh":
            # A finite elment mesh was specified
            necessary_parameters = {"rve_dims", "mesh_size", "element_type"}
            try:
                if any(
                    [
                        necessary_parameter not in discret_spec_array
                        for necessary_parameter in necessary_parameters
                    ]
                ):
                    raise errors.InsufficientInfoMesh(
                        list(discret_spec_array.keys()), necessary_parameters, disc_ext
                    )
                rve_dims = discret_spec_array["rve_dims"]
                mesh_size = discret_spec_array["mesh_size"]
                if rve_dims.shape != (3,) and rve_dims.shape != (2,):
                    # The RVE dims must be given as 1-arrays with 2 or 3 elements
                    raise errors.UnexpectedValue(
                        rve_dims, "rve_dims", "1-array with shape (2,) or (3,)"
                    )
                if any(rve_dims < 0):
                    # The RVE dimensions must be positive real numbers
                    raise errors.UnexpectedValue(
                        rve_dims, "rve_dims", "array of positive reals"
                    )
                # Saving the RVE dims and the mesh size
                if mesh_size <= 0:
                    # The meshsize is smaller than one
                    raise errors.UnexpectedValue(
                        mesh_size, "mesh_size", "positive real"
                    )
            except (errors.InsufficientInfoMesh, errors.UnexpectedValue) as error:
                error.message()

        elif disc_ext == "nomsh":
            # No mesh was specified
            necessary_parameters = {"rve_dims"}
            try:
                if any(
                    [
                        necessary_parameter not in discret_spec_array
                        for necessary_parameter in necessary_parameters
                    ]
                ):
                    raise errors.InsufficientInfoMesh(
                        list(discret_spec_array.keys()), necessary_parameters, disc_ext
                    )
                rve_dims = discret_spec_array["rve_dims"]
                if rve_dims.shape != (3,) and rve_dims.shape != (2,):
                    # The RVE dims must be given as 1-arrays with 2 or 3 elements
                    raise errors.UnexpectedValue(
                        rve_dims, "rve_dims", "1-array with saphe (2,) or (3,)"
                    )
                if any(rve_dims < 0):
                    # The RVE dimensions must be positive real numbers
                    raise errors.UnexpectedValue(
                        rve_dims, "rve_dims", "array of positive reals"
                    )
            except (errors.InsufficientInfoMesh, errors.UnexpectedValue) as error:
                error.message()

        else:
            # Unsupported mesh
            try:
                raise errors.UnsupportedMesh(disc_ext)
            except errors.UnsupportedMesh as error:
                error.message()
