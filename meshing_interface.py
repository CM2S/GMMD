# ==========================================================================================
# Example of mesh generation with GMSH and conversion to LINKS format:
# >> Rectangle with elliptical inclusion and TRI3
#
# António Manuel Couto Carneiro
# @ CM2S                                                Department of Mechanical Engineering
#                                          Faculty of Engineering of the University of Porto
#                               ------------------------------------------------------------
#                               Computational Multi-Scale Modelling of Solids and Structures
# ==========================================================================================
# Import modules
# --------------
# Utility to get the name of the current module
import genericpath
# Operating system directives
import os
# GMSH module
import gmsh
# Simple math tools
import math
# Finite element mesh conversor to LINKS
from gmsh2links.main import readMesh
from particle_classes import Disk, Particle, Ellipse
# Importing the particle class
import numpy as np
import shutil
import error_classes as errors
from plotting_functions import plotPixels, plotVoxels

def generateMeshFEM(particles, mesh_size, rve_dims, element_type="tri6", **kwargs):
    '''
    This function generates the mesh for the Finite Element Method.

    Parameters
    ----------
    particles: list(`.Particle`)
        List containing the particles of the microstructure.

    mesh_size: float
        Size of the mesh.

    element_type: {'tri3','tri6','quad4','quad8','other'}, optional
        String related to the type of element.

    Other Parameters
    ----------------
    **kwargs:
        Optional parameters for the `generateMeshFEM2D` and `generateMeshFEM3D` functions.
    '''

    output_term = kwargs.pop('output_term',1)
    # Option for the output in the terminal
    if len(rve_dims) == 2:
    # It is a 2D problem
        if element_type == "tri3":
        # (Defaul option) Linear Triangular element
            generateMeshFEM2D(particles, mesh_size, output_term=output_term)
            # Generating a mesh of linear triangular elements
        elif element_type == "tri6":
        # Quadratic Triangular element
            generateMeshFEM2D(particles, mesh_size, element_order=2,
                              output_term=output_term)
            # Generating a mesh of linear triangular elements
        elif element_type=="quad4":
        # Linear Rectangular element
            generateMeshFEM2D(particles, mesh_size, force_recomb_all=1,
                              output_term=output_term)
            # Generating a mesh of linear triangular elements
        elif element_type == "quad8":
        # 2nd order rectangular elment of the serendipity family
            generateMeshFEM2D(particles, mesh_size, force_recomb_all=1,
                              element_order=2, elemnet_order_incomp=1,
                              output_term=output_term)
            # Generating a mesh of linear triangular elements
        else:
            mesh_alg = kwargs.pop('mesh_alg', 6)
            force_recomb_all = kwargs.pop('force_recomb_all', 0)
            element_order = kwargs.pop('element_order', 1)
            recomb_alg = kwargs.pop('recomb_alg', 1)
            element_order_incomp = kwargs.pop('element_order_incomp', 0)
            output_term = kwargs.pop('output_term', 0)
            # Saving the extra options
            generateMeshFEM2D(particles, mesh_size, mesh_alg=mesh_alg,
                force_recomb_all=force_recomb_all, element_order=element_order,
                recomb_alg=recomb_alg, element_order_incomp=element_order_incomp,
                output_term=output_term)
            # Generating a mesh of linear triangular elements
    elif len(rve_dims) == 3:
    # It is a 3D problem
        generateMeshFEM3D(particles, mesh_size, rve_dims)

def generateMeshFEM2D(particles, mesh_size, mesh_alg=5, force_recomb_all=0, element_order=1,
                      recomb_alg=1, element_order_incomp=0, output_term=0):
    '''
    This function generates the mesh for the Finite Element Method in 2D. It generates by
    default linear triangular elements.

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
    '''
    # ======================================================================================
    # Set up GMSH in Python
    # ======================================================================================
    # Select the geometry engine
    # occ - OpenCASCADE CAD (more advanced)
    # geo - built-in CAD kernel (less sophisticated)
    model = gmsh.model
    factory = model.occ

    # Initialise GMSH
    gmsh.initialize()

    # Output to terminal
    gmsh.option.setNumber("General.Terminal", output_term)

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

    # Characteristic mesh length factor (applied acroos all mesh)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1)

    # Multi-threading
    gmsh.option.setNumber("Mesh.MaxNumThreads1D", 4)
    gmsh.option.setNumber("Mesh.MaxNumThreads2D", 4)
    gmsh.option.setNumber("Mesh.MaxNumThreads3D", 4 )

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
    title = Particle.file_path
    model.add(title)

    x = 0
    y = 0
    z = 0
    lx = Particle.box[0]
    ly = Particle.box[1]
    eps =  0 #1.5*mesh_size

    rectTag = factory.addRectangle(0, 0, 0, Particle.box[0], Particle.box[1])
    # RVE

    particleTags = []
    rotateTags = []
    k_particle_image = 0
    phaseDimTag = {phase: [] for phase in Particle.list_phases}
    for i_particle in particles:
    # Running through all the particles
        class_name_i_particle = i_particle.__class__.__name__
        # Saving the class name of the particle as a string
        for j in range(-1,2):
        # Periodic images in the horizontal direction
            for p in range(-1,2):
            # Periodic images in the vertical direction
                if 'Disk'==class_name_i_particle:
                # Particle is a Disk
                    xc = i_particle.position_center[0] + Particle.box[0]*j
                    yc = i_particle.position_center[1] + Particle.box[1]*p
                    zc = 0
                    rx = i_particle.radius
                    ry = i_particle.radius
                    if xc > Particle.box[0] + rx - eps or xc < -rx + eps or yc > Particle.box[1] + ry - eps or yc < -ry + eps:
                        continue
                    # Saving the properties of the particles
                    particleTags.append(factory.addDisk(xc, yc, zc, rx, ry))

                    phaseDimTag[i_particle.phase].append(
                        (2, particleTags[k_particle_image]))

                    factory.synchronize()
                    k_particle_image += 1
                elif 'Ellipse'==class_name_i_particle:
                # Particle is an Ellipse
                    xc = i_particle.position_center[0] + Particle.box[0]*j
                    yc = i_particle.position_center[1] + Particle.box[1]*p
                    zc = 0
                    rx = i_particle.semi_major_axis
                    ry = i_particle.semi_minor_axis
                    alpha = i_particle.angle
                    if xc > Particle.box[0] + rx - eps or xc < -rx + eps or yc > Particle.box[1] + rx - eps or yc < -rx + eps:
                        continue
                    # Saving the properties of the particles
                    particleTags.append(factory.addDisk(xc, yc, zc, rx, ry))
                    # Creating the ellipse without rotation
                    # Rotate the disk
                    factory.synchronize()
                    rotateTags.append([(2, particleTags[k_particle_image])])
                    rotateTags[-1].extend(
                        model.getBoundary([2, particleTags[k_particle_image]]))
                    factory.rotate(rotateTags[-1], xc, yc, zc, 0, 0, 1, alpha)

                    phaseDimTag[i_particle.phase].append(
                        (2, particleTags[k_particle_image]))

                    factory.synchronize()
                    k_particle_image += 1

    outDimTag, outDimTagMap = factory.intersect(
        [(2, rectTag)], [(2, particleTags[k]) for k in range(k_particle_image)], removeObject=False, removeTool=True)

    temp = set(outDimTag)
    for i_phase in Particle.list_phases:
        phaseDimTag[i_phase] = [ value for value in phaseDimTag[i_phase] if value in temp ]

    factory.synchronize()

    outDimTag2, outDimTagMap2 = factory.fragment(
        [(2, rectTag)], outDimTag, removeObject=True, removeTool=True)
    
    phaseDimTag[Particle.matrix_phase] = outDimTag2[len(outDimTag):]
    materials = []
    for i_phase in Particle.list_phases:
        temp = set(phaseDimTag[i_phase])
        materials.append([value[1] for value in outDimTag2 if value in temp])


    # Set the mesh size on the geometry points
    # Synchronize the CAD engine (always needed before generating the mesh)
    # It may also be useful for some intermidate operations, like checking the tags of entities
    factory.synchronize()

    for i_phase in range(len(Particle.list_phases)):
        materialTag = model.addPhysicalGroup(2, materials[i_phase])
        model.setPhysicalName(2, materialTag, "Phase " + Particle.list_phases[i_phase])

    # material1Tag = model.addPhysicalGroup(material1[0][0], [material1[0][1]])
    # model.setPhysicalName(material1[0][0], material1Tag, "Material 1")
    # 
    # material2Tag = model.addPhysicalGroup(material2[0][0], [material2[i][1] for i in range(len(material2))])
    # model.setPhysicalName(material2[0][0], material2Tag, "Material 2")

    factory.synchronize()

    gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
    eps = 1e-3
    # Ask OpenCASCADE to compute more accurate bounding boxes of entities using
    # the STL mesh
    # --------------------------------------------------------------------------------------
    translation_l_r = [1, 0, 0, Particle.box[0],
                       0, 1, 0, 0,
                       0, 0, 1, 0,
                       0, 0, 0, 1]
    # Translation of the left face of the cube to the rigth face of the cube given as an
    # affine transformation, (4x4), written by row
    l_edge = gmsh.model.getEntitiesInBoundingBox(
                - eps,                 - eps, - eps,
                + eps, Particle.box[1] + eps, + eps,
                1)
    # First we get all surfaces on the left:
    for i_edge in l_edge:
        # Then we get the bounding box of each left surface
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(i_edge[0], i_edge[1])
        # We translate the bounding box to the right and look for surfaces inside
        # it:
        r_edge = gmsh.model.getEntitiesInBoundingBox(
                    xmin - eps + Particle.box[0], ymin - eps, zmin - eps,
                    xmax + eps + Particle.box[0], ymax + eps, zmax + eps,
                    1)
        # For all the matches, we compare the corresponding bounding boxes...
        for j_edge in r_edge:
            xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = gmsh.model.getBoundingBox(
                j_edge[0], j_edge[1])
            xmin2 -= Particle.box[0]
            xmax2 -= Particle.box[0]
            # ...and if they match, we apply the periodicity constraint
            if (abs(xmin2 - xmin) < eps and abs(xmax2 - xmax) < eps
                    and abs(ymin2 - ymin) < eps and abs(ymax2 - ymax) < eps
                    and abs(zmin2 - zmin) < eps and abs(zmax2 - zmax) < eps):
                gmsh.model.mesh.setPeriodic(1, [j_edge[1]], [i_edge[1]], translation_l_r)
                # Ensuring periodicity


    factory.synchronize()

    gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
    eps = 1e-3

    translation_b_t = [1, 0, 0, 0,
                       0, 1, 0, Particle.box[1],
                       0, 0, 1, 0,
                       0, 0, 0, 1]
    # Translation of the left face of the cube to the rigth face of the cube given as an
    # affine transformation, (4x4), written by row
    b_edge = gmsh.model.getEntitiesInBoundingBox(
                - eps, - eps, - eps,
                Particle.box[0] + eps, + eps, + eps,
                1)
    # First we get all surfaces on the left:
    for i_edge in b_edge:
        # Then we get the bounding box of each left surface
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(i_edge[0], i_edge[1])
        # We translate the bounding box to the right and look for surfaces inside
        # it:
        t_edge = gmsh.model.getEntitiesInBoundingBox(
                    xmin - eps, ymin - eps + Particle.box[1], zmin - eps,
                    xmax + eps, ymax + eps + Particle.box[1], zmax + eps,
                    1)
        # For all the matches, we compare the corresponding bounding boxes...
        for j_edge in t_edge:
            xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = gmsh.model.getBoundingBox(
                j_edge[0], j_edge[1])
            ymin2 -= Particle.box[1]
            ymax2 -= Particle.box[1]
            # ...and if they match, we apply the periodicity constraint
            if (abs(xmin2 - xmin) < eps and abs(xmax2 - xmax) < eps
                    and abs(ymin2 - ymin) < eps and abs(ymax2 - ymax) < eps
                    and abs(zmin2 - zmin) < eps and abs(zmax2 - zmax) < eps):
                gmsh.model.mesh.setPeriodic(1, [j_edge[1]], [i_edge[1]], translation_b_t)
                # Ensuring periodicity


    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

    print('here')
    # Generate a 3D mesh
    model.mesh.generate(2)

    tri_type = model.mesh.getElementTypes(dim=2, tag=-1)[0]
    local_coord, _ = model.mesh.getIntegrationPoints(tri_type, "Gauss3")
    _, all_jacobians, _ = model.mesh.getJacobians(tri_type, local_coord, tag=-1)
    _, tagsAllElements, _ = model.mesh.getElements(dim=2)
    dimTagsErrors = []
    for i_ele, tagEle in enumerate(tagsAllElements[0]):
        for k_point in range(4): #int(len(local_coord)/3)):
            if all_jacobians[i_ele*4 + k_point] <= 0.05:
                dimTagsErrors.append((2, tagEle))
                break
    print("Errors", dimTagsErrors)

    model.mesh.optimize("HighOrder", force=False, niter=10, dimTags=dimTagsErrors)

    tri_type = model.mesh.getElementTypes(dim=2, tag=-1)[0]
    local_coord, _ = model.mesh.getIntegrationPoints(tri_type, "Gauss3")
    _, all_jacobians, _ = model.mesh.getJacobians(tri_type, local_coord, tag=-1)
    _, tagsAllElements, _ = model.mesh.getElements(dim=2)
    dimTagsErrors = []
    for i_ele, tagEle in enumerate(tagsAllElements[0]):
        for k_point in range(4): #int(len(local_coord)/3)):
            if all_jacobians[i_ele*4 + k_point] <= 0.05:
                dimTagsErrors.append((2, tagEle))
                break
    print("Final Errors", dimTagsErrors)

    factory.synchronize()

    gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
    eps = 1e-3
    # Ask OpenCASCADE to compute more accurate bounding boxes of entities using
    # the STL mesh
    # --------------------------------------------------------------------------------------
    translation_l_r = [1, 0, 0, Particle.box[0],
                       0, 1, 0, 0,
                       0, 0, 1, 0,
                       0, 0, 0, 1]
    # Translation of the left face of the cube to the rigth face of the cube given as an
    # affine transformation, (4x4), written by row
    l_edge = gmsh.model.getEntitiesInBoundingBox(
                - eps,                 - eps, - eps,
                + eps, Particle.box[1] + eps, + eps,
                1)
    # First we get all surfaces on the left:
    for i_edge in l_edge:
        # Then we get the bounding box of each left surface
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(i_edge[0], i_edge[1])
        # We translate the bounding box to the right and look for surfaces inside
        # it:
        r_edge = gmsh.model.getEntitiesInBoundingBox(
                    xmin - eps + Particle.box[0], ymin - eps, zmin - eps,
                    xmax + eps + Particle.box[0], ymax + eps, zmax + eps,
                    1)
        # For all the matches, we compare the corresponding bounding boxes...
        for j_edge in r_edge:
            xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = gmsh.model.getBoundingBox(
                j_edge[0], j_edge[1])
            xmin2 -= Particle.box[0]
            xmax2 -= Particle.box[0]
            # ...and if they match, we apply the periodicity constraint
            if (abs(xmin2 - xmin) < eps and abs(xmax2 - xmax) < eps
                    and abs(ymin2 - ymin) < eps and abs(ymax2 - ymax) < eps
                    and abs(zmin2 - zmin) < eps and abs(zmax2 - zmax) < eps):
                gmsh.model.mesh.setPeriodic(1, [j_edge[1]], [i_edge[1]], translation_l_r)
                # Ensuring periodicity


    factory.synchronize()

    gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
    eps = 1e-3

    translation_b_t = [1, 0, 0, 0,
                       0, 1, 0, Particle.box[1],
                       0, 0, 1, 0,
                       0, 0, 0, 1]
    # Translation of the left face of the cube to the rigth face of the cube given as an
    # affine transformation, (4x4), written by row
    b_edge = gmsh.model.getEntitiesInBoundingBox(
                - eps, - eps, - eps,
                Particle.box[0] + eps, + eps, + eps,
                1)
    # First we get all surfaces on the left:
    for i_edge in b_edge:
        # Then we get the bounding box of each left surface
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(i_edge[0], i_edge[1])
        # We translate the bounding box to the right and look for surfaces inside
        # it:
        t_edge = gmsh.model.getEntitiesInBoundingBox(
                    xmin - eps, ymin - eps + Particle.box[1], zmin - eps,
                    xmax + eps, ymax + eps + Particle.box[1], zmax + eps,
                    1)
        # For all the matches, we compare the corresponding bounding boxes...
        for j_edge in t_edge:
            xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = gmsh.model.getBoundingBox(
                j_edge[0], j_edge[1])
            ymin2 -= Particle.box[1]
            ymax2 -= Particle.box[1]
            # ...and if they match, we apply the periodicity constraint
            if (abs(xmin2 - xmin) < eps and abs(xmax2 - xmax) < eps
                    and abs(ymin2 - ymin) < eps and abs(ymax2 - ymax) < eps
                    and abs(zmin2 - zmin) < eps and abs(zmax2 - zmax) < eps):
                gmsh.model.mesh.setPeriodic(1, [j_edge[1]], [i_edge[1]], translation_b_t)

    # Write the mesh to the .msh file
    meshfile_temp = title + "_temp.msh"
    meshfile = title + '.msh'
    vtk_temp = title + "_temp.vtk"
    vtk = title + ".vtk"
    gmsh.write(meshfile_temp)
    gmsh.write(vtk_temp)

    
    # Close GMSH
    gmsh.finalize()
    # ==========================================================================================
    # Convert it to LINKS format and write the respective input file
    # ==========================================================================================

    fin = open(meshfile_temp, "rt")
    fout = open(meshfile, "wt")

    for line in fin:
    	fout.write(line.replace(',', '.'))

    fin.close()
    fout.close()
    os.remove(meshfile_temp)

    gmshToLinks(meshfile, title, 2, Particle.list_phases, Particle.matrix_phase)

    fin = open(vtk_temp, "rt")
    fout = open(vtk, "wt")

    for line in fin:
    	fout.write(line.replace(',', '.'))

    fin.close()
    fout.close()
    os.remove(vtk_temp)

def generateMeshFEM3D(particles, mesh_size, rve_dims, mesh_alg=1, force_recomb_all=0, element_order=2,
    recomb_alg=2, element_order_incomp=0, output_term=1):
    '''
    This function generates the mesh for the Finite Element Method in 2D. It generates by
    default linear triangular elements.

    Parameters
    ----------
    particles: list(`.Particle`)
        List containing the particles of the microstructure.

    mesh_size: float
        Size of the mesh.

    rve_dims: list
        List containing the size of the RVE.

    output_term: {0, 1}, optional
        Output to the terminal

    mesh_alg: integer, optional
        3D Meshing algorithm
        1: Delaunay (default)
        2: Frontal
        7: MMG3D
        9: R-tree
        10: HXT

    force_recomb_all: {0, 1}, optional
        Recombination into quads.

    element_order: integer, optional
        Order of the element

    recomb_alg: integer, optional
        0: hex (default)
        1: hex + prisms
        2: hex + prisms + pyramids

    element_order_incomp: {0, 1}, optional
        Remove interior nodes for second order elements
    '''
    # ======================================================================================
    # Set up GMSH in Python
    # ======================================================================================
    # Select the geometry engine
    # occ - OpenCASCADE CAD (more advanced)
    # geo - built-in CAD kernel (less sophisticated)
    model = gmsh.model
    factory = model.occ

    # Initialise GMSH
    gmsh.initialize()

    # Output to terminal
    gmsh.option.setNumber("General.Terminal", output_term)

    # 2D Meshing algorithm
    # --------------------
    # 1 - Mesh Adapt
    # 2 - Automatic
    # 5 - Delaunay (default)
    # 6 - Frontal-Delaunay
    # 7 - BAMG
    # 8 - Frontal-Delaunay for Quads
    # 9 - Packing of Parallelograms
    gmsh.option.setNumber("Mesh.Algorithm", 6)

    # 3D Meshing algorithm
    # --------------------
    # 1 - Delaunay (default)
    # 2 - Frontal
    # 7 - MMG3D
    # 9 - R-tree
    # 10 - HXT
    gmsh.option.setNumber("Mesh.Algorithm3D", mesh_alg)

    # Characteristic mesh length factor (applied acroos all mesh)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1)

    # Multi-threading
    gmsh.option.setNumber("Mesh.MaxNumThreads1D", 0)
    gmsh.option.setNumber("Mesh.MaxNumThreads2D", 0)
    gmsh.option.setNumber("Mesh.MaxNumThreads3D", 0)

    # MSH file version
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)

    # Quad/Hex recombination algorithms
    # ---------------------------------
    # 0 - simple
    # 1 - blossom (default)
    # 2 - simple full-quad
    # 3 - blosson full-quad
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)

    # Force recombination in all surfaces
    gmsh.option.setNumber("Mesh.RecombineAll", 0)

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

    title = Particle.file_path
    model.add(title)

    boxTag = factory.addBox(0, 0, 0, rve_dims[0], rve_dims[1], rve_dims[2])
    # RVE

    particleTags = []
    k_particle_image = 0
    phaseDimTag = {phase: [] for phase in Particle.list_phases}
    for i_particle in particles:
    # Running through all the particles
        class_name_i_particle = i_particle.__class__.__name__
        # Saving the class name of the particle as a string
        for j in range(-1, 2):
        # Periodic images in the x direction
            for p in range(-1, 2):
            # Periodic images in the y direction
                for l in range(-1, 2):
                # Periodic images in the z direction
                    if 'CylindricalFiber' == class_name_i_particle:
                        if l != 0:
                            continue
                        xc = i_particle.position_center[0] + Particle.box[0]*j
                        yc = i_particle.position_center[1] + Particle.box[1]*p
                        zc = 0
                        rx = i_particle.radius
                        ry = i_particle.radius
                        # Speciying the position and the radius of the fibers face
                        faceTag = factory.addDisk(xc, yc, zc, rx, ry)
                        # Saving the properties of the particles
                        if i_particle.direction_fibers == 0:
                        # The fibers run in the x direction
                            factory.rotate([(2, faceTag)],
                                           0, 0, 0, 0, 1, 0, 3*np.pi/2)
                            # Rotating the fiber face to the yz plane as it was ploted
                            # in the xy plane
                            extrusionTags = factory.extrude(
                                [(2, faceTag)], i_particle.length_dir_fibers, 0, 0)
                            # Extruding the fiber from the fiber face in the yz plane in the
                            # x direction
                        elif i_particle.direction_fibers == 1:
                        # The fibers run in the y direction
                            factory.rotate([(2, faceTag)],
                                           0, 0, 0, 1, 0, 0, np.pi/2)
                            # Rotating the fiber faces to the xz plane as it was ploted
                            # in the xy plane
                            extrusionTags = factory.extrude(
                                [(2, faceTag)], 0, i_particle.length_dir_fibers, 0)
                            # Extruding the fiber from the fiber face in the xz plane in the
                            # y direction
                        elif i_particle.direction_fibers == 2:
                        # The fibers run in the z direction
                            extrusionTags = factory.extrude(
                                [(2, faceTag)], 0, 0, i_particle.length_dir_fibers)
                            # Extruding the fiber from the fiber face in the xy plane in the
                            # z direction

                        for i_dimTag in extrusionTags:
                            if i_dimTag[0] == 3:
                                particleTags.append(i_dimTag[1])
                                break

                        phaseDimTag[str(i_particle.phase)].append(
                            (3, particleTags[-1]))

                        factory.synchronize()
                        k_particle_image += 1
                    if 'Sphere' == class_name_i_particle:
                    # Particle is a Sphere
                        xc = i_particle.position_center[0] + Particle.box[0]*j
                        yc = i_particle.position_center[1] + Particle.box[1]*p
                        zc = i_particle.position_center[2] + Particle.box[2]*l
                        r = i_particle.radius
                        # Saving the properties of the particles
                        if xc > Particle.box[0] + r or xc < -r or yc > Particle.box[1] + r or yc < -r or zc > Particle.box[2] + r or zc < -r:
                            continue
                        particleTags.append(factory.addSphere(xc, yc, zc, r))

                        phaseDimTag[str(i_particle.phase)].append((3, particleTags[k_particle_image]))

                        factory.synchronize()
                        k_particle_image += 1
                    elif 'Ellipsoid' == class_name_i_particle:
                    # Particle is an Ellipsoid
                        xc = i_particle.position_center[0] + Particle.box[0]*j
                        yc = i_particle.position_center[1] + Particle.box[1]*p
                        zc = i_particle.position_center[2] + Particle.box[2]*l
                        r = i_particle.radius
                        # Saving the properties of the particles
                        if xc > Particle.box[0] + r or xc < -r or yc > Particle.box[1] + r or yc < -r or zc > Particle.box[2] + r or zc < -r:
                            continue
                        print('here')
                        r = 1
                        particleTags.append(factory.addSphere(xc, yc, zc, r))
                        # Creating a sphere without rotation
                        # Rotate the disk
                        factory.synchronize()
                        affineTags = [(3, particleTags[k_particle_image])]
                        # affineTags.extend(
                        #     model.getBoundary([(3, particleTags[k_particle_image])]))
                        factory.dilate(affineTags, xc, yc, zc,
                                       i_particle.semi_axis_1,
                                       i_particle.semi_axis_2,
                                       i_particle.semi_axis_3)
                        factory.rotate(affineTags, xc, yc, zc,
                                       i_particle.rotation_axis[0],
                                       i_particle.rotation_axis[1],
                                       i_particle.rotation_axis[2],
                                       i_particle.angle)

                        phaseDimTag[str(i_particle.phase)].append((3, particleTags[k_particle_image]))

                        factory.synchronize()
                        k_particle_image += 1

    print(phaseDimTag)
    print([(3, particleTag) for particleTag in particleTags])
    outDimTag, outDimTagMap = factory.intersect(
        [(3, boxTag)], [(3, particleTag) for particleTag in particleTags],
        removeObject=False, removeTool=True)

    print(outDimTag)
    temp = set(outDimTag)
    for i_phase in Particle.list_phases:
        phaseDimTag[i_phase] = [value for value in phaseDimTag[i_phase] if value in temp]

    factory.synchronize()

    outDimTag2, outDimTagMap2 = factory.fragment(
        [(3, boxTag)], outDimTag, removeObject=True, removeTool=True)

    phaseDimTag[Particle.matrix_phase] = outDimTag2[len(outDimTag):]
    materials = []
    for i_phase in Particle.list_phases:
        temp = set(phaseDimTag[i_phase])
        materials.append([value[1] for value in outDimTag2 if value in temp])

    # Set the mesh size on the geometry points
    # Synchronize the CAD engine (always needed before generating the mesh)
    # It may also be useful for some intermidate operations, like checking the tags of
    # entities
    factory.synchronize()

    for i_phase in range(len(Particle.list_phases)):
        materialTag = model.addPhysicalGroup(3, materials[i_phase])
        model.setPhysicalName(3, materialTag, "Phase " + Particle.list_phases[i_phase])

#     gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
#     eps = 1e-4
#     # Ask OpenCASCADE to compute more accurate bounding boxes of entities using
#     # the STL mesh
#     # --------------------------------------------------------------------------------------
#     translation_l_r = [1, 0, 0, rve_dims[0],
#                        0, 1, 0, 0,
#                        0, 0, 1, 0,
#                        0, 0, 0, 1]
#     # Translation of the left face of the cube to the rigth face of the cube given as an
#     # affine transformation, (4x4), written by row
#     l_face = gmsh.model.getEntitiesInBoundingBox(
#                 - eps,                 - eps,                 - eps,
#                 + eps, Particle.box[1] + eps, rve_dims[2] + eps,
#                 2)
#     # First we get all surfaces on the left:
#     for i_surf in l_face:
#         # Then we get the bounding box of each left surface
#         xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(i_surf[0], i_surf[1])
#         # We translate the bounding box to the right and look for surfaces inside
#         # it:
#         r_face = gmsh.model.getEntitiesInBoundingBox(
#                     xmin - eps + rve_dims[0], ymin - eps, zmin - eps,
#                     xmax + eps + rve_dims[0], ymax + eps, zmax + eps,
#                     2)
#         # For all the matches, we compare the corresponding bounding boxes...
#         for j_surf in r_face:
#             xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = gmsh.model.getBoundingBox(
#                 j_surf[0], j_surf[1])
#             xmin2 -= rve_dims[0]
#             xmax2 -= rve_dims[0]
#             # ...and if they match, we apply the periodicity constraint
#             if (abs(xmin2 - xmin) < eps and abs(xmax2 - xmax) < eps
#                     and abs(ymin2 - ymin) < eps and abs(ymax2 - ymax) < eps
#                     and abs(zmin2 - zmin) < eps and abs(zmax2 - zmax) < eps):
#                 gmsh.model.mesh.setPeriodic(2, [j_surf[1]], [i_surf[1]], translation_l_r)
#                 print(j_surf[1], i_surf[1])
#                 print(gmsh.model.getBoundary(j_surf))
#                 print(gmsh.model.getBoundary(i_surf))
# # --------------------------------------------------------------------------------------
#                 # Ensuring periodicity
#     # --------------------------------------------------------------------------------------
#     gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
#     translation_b_t = [1, 0, 0, 0,
#                        0, 1, 0, Particle.box[1],
#                        0, 0, 1, 0,
#                        0, 0, 0, 1]
#     # Translation of the top face of the cube to the bottom face of the cube given as an
#     # affine transformation, (4x4), written by row
#     b_face = gmsh.model.getEntitiesInBoundingBox(
#                                 - eps, - eps,                 - eps,
#                 rve_dims[0] + eps, + eps, rve_dims[2] + eps,
#                 2)
#     # First we get all surfaces on the bottom:
#     for i_surf in b_face:
#         # Then we get the bounding box of each bottom surface
#         xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(i_surf[0], i_surf[1])
#         # We translate the bounding box upwards and look for surfaces inside
#         # it:
#         t_face = gmsh.model.getEntitiesInBoundingBox(
#                     xmin - eps, ymin - eps + Particle.box[1], zmin - eps,
#                     xmax + eps, ymax + eps + Particle.box[1], zmax + eps,
#                     2)
#         # For all the matches, we compare the corresponding bounding boxes...
#         for j_surf in t_face:
#             xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = gmsh.model.getBoundingBox(
#                 j_surf[0], j_surf[1])
# 
#             ymin2 -= Particle.box[1]
#             ymax2 -= Particle.box[1]
#             # ...and if they match, we apply the periodicity constraint
#             if (abs(xmin2 - xmin) < eps and abs(xmax2 - xmax) < eps
#                     and abs(ymin2 - ymin) < eps and abs(ymax2 - ymax) < eps
#                     and abs(zmin2 - zmin) < eps and abs(zmax2 - zmax) < eps):
#                 gmsh.model.mesh.setPeriodic(2, [j_surf[1]], [i_surf[1]], translation_b_t)                
#                 print(j_surf[1], i_surf[1])
#                 print(gmsh.model.getBoundary(j_surf))
#                 print(gmsh.model.getBoundary(i_surf))
# # --------------------------------------------------------------------------------------
#     gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
#     translation_f_b = [1, 0, 0, 0,
#                        0, 1, 0, 0,
#                        0, 0, 1, rve_dims[2],
#                        0, 0, 0, 1]
#     # Translation of the front face of the cube to the back face of the cube given as an
#     # affine transformation, (4x4), written by row
#     f_face = gmsh.model.getEntitiesInBoundingBox(
#                                 - eps,                 - eps, - eps,
#                 rve_dims[0] + eps, Particle.box[1] + eps, + eps,
#                 2)
#     # First we get all surfaces on the front:
#     for i_surf in f_face:
#         # Then we get the bounding box of each front surface
#         xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(i_surf[0], i_surf[1])
#         # We translate the bounding box to the back and look for surfaces inside
#         # it:
#         b_face = gmsh.model.getEntitiesInBoundingBox(
#                     xmin - eps, ymin - eps, zmin - eps + rve_dims[2],
#                     xmax + eps, ymax + eps, zmax + eps + rve_dims[2],
#                     2)
#         # For all the matches, we compare the corresponding bounding boxes...
#         for j_surf in b_face:
#             xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = gmsh.model.getBoundingBox(
#                 j_surf[0], j_surf[1])
#             zmin2 -= rve_dims[2]
#             zmax2 -= rve_dims[2]
#             # ...and if they match, we apply the periodicity constraint
#             if (abs(xmin2 - xmin) < eps and abs(xmax2 - xmax) < eps
#                     and abs(ymin2 - ymin) < eps and abs(ymax2 - ymax) < eps
#                     and abs(zmin2 - zmin) < eps and abs(zmax2 - zmax) < eps):
#                 gmsh.model.mesh.setPeriodic(2, [j_surf[1]], [i_surf[1]], translation_f_b)
#                 print(j_surf[1], i_surf[1])
#                 print(gmsh.model.getBoundary(j_surf))
#                 print(gmsh.model.getBoundary(i_surf))
# 
# 
# 
#     gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
#     eps = 1e-4
#     # Ask OpenCASCADE to compute more accurate bounding boxes of entities using
#     # the STL mesh
#     # --------------------------------------------------------------------------------------
#     translation_l_r = [1, 0, 0, rve_dims[0],
#                        0, 1, 0, 0,
#                        0, 0, 1, 0,
#                        0, 0, 0, 1]
#     # Translation of the left face of the cube to the rigth face of the cube given as an
#     # affine transformation, (4x4), written by row
#     l_face = gmsh.model.getEntitiesInBoundingBox(
#                 - eps,                 - eps,                 - eps,
#                 + eps, rve_dims[1] + eps, rve_dims[2] + eps,
#                 1)
#     # First we get all surfaces on the left:
#     for i_surf in l_face:
#         # Then we get the bounding box of each left surface
#         xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(i_surf[0], i_surf[1])
#         # We translate the bounding box to the right and look for surfaces inside
#         # it:
#         r_face = gmsh.model.getEntitiesInBoundingBox(
#                     xmin - eps + rve_dims[0], ymin - eps, zmin - eps,
#                     xmax + eps + rve_dims[0], ymax + eps, zmax + eps,
#                     1)
#         # For all the matches, we compare the corresponding bounding boxes...
#         for j_surf in r_face:
#             xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = gmsh.model.getBoundingBox(
#                 j_surf[0], j_surf[1])
#             xmin2 -= rve_dims[0]
#             xmax2 -= rve_dims[0]
#             # ...and if they match, we apply the periodicity constraint
#             if (abs(xmin2 - xmin) < eps and abs(xmax2 - xmax) < eps
#                     and abs(ymin2 - ymin) < eps and abs(ymax2 - ymax) < eps
#                     and abs(zmin2 - zmin) < eps and abs(zmax2 - zmax) < eps):
#                 gmsh.model.mesh.setPeriodic(1, [j_surf[1]], [i_surf[1]], translation_l_r)
#                 print(j_surf[1], i_surf[1])
#                 print(gmsh.model.getBoundary(j_surf))
#                 print(gmsh.model.getBoundary(i_surf))
# # --------------------------------------------------------------------------------------
#                 # Ensuring periodicity
#     # --------------------------------------------------------------------------------------
#     gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
#     translation_b_t = [1, 0, 0, 0,
#                        0, 1, 0, rve_dims[1],
#                        0, 0, 1, 0,
#                        0, 0, 0, 1]
#     # Translation of the top face of the cube to the bottom face of the cube given as an
#     # affine transformation, (4x4), written by row
#     b_face = gmsh.model.getEntitiesInBoundingBox(
#                                 - eps, - eps,                 - eps,
#                 rve_dims[0] + eps, + eps, rve_dims[2] + eps,
#                 1)
#     # First we get all surfaces on the bottom:
#     for i_surf in b_face:
#         # Then we get the bounding box of each bottom surface
#         xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(i_surf[0], i_surf[1])
#         # We translate the bounding box upwards and look for surfaces inside
#         # it:
#         t_face = gmsh.model.getEntitiesInBoundingBox(
#                     xmin - eps, ymin - eps + rve_dims[1], zmin - eps,
#                     xmax + eps, ymax + eps + rve_dims[1], zmax + eps,
#                     1)
#         # For all the matches, we compare the corresponding bounding boxes...
#         for j_surf in t_face:
#             xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = gmsh.model.getBoundingBox(
#                 j_surf[0], j_surf[1])
# 
#             ymin2 -= rve_dims[1]
#             ymax2 -= rve_dims[1]
#             # ...and if they match, we apply the periodicity constraint
#             if (abs(xmin2 - xmin) < eps and abs(xmax2 - xmax) < eps
#                     and abs(ymin2 - ymin) < eps and abs(ymax2 - ymax) < eps
#                     and abs(zmin2 - zmin) < eps and abs(zmax2 - zmax) < eps):
#                 gmsh.model.mesh.setPeriodic(1, [j_surf[1]], [i_surf[1]], translation_b_t)                
#                 print(j_surf[1], i_surf[1])
#                 print(gmsh.model.getBoundary(j_surf))
#                 print(gmsh.model.getBoundary(i_surf))
# # --------------------------------------------------------------------------------------
#     gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
#     translation_f_b = [1, 0, 0, 0,
#                        0, 1, 0, 0,
#                        0, 0, 1, rve_dims[2],
#                        0, 0, 0, 1]
#     # Translation of the front face of the cube to the back face of the cube given as an
#     # affine transformation, (4x4), written by row
#     f_face = gmsh.model.getEntitiesInBoundingBox(
#                                 - eps,                 - eps, - eps,
#                 rve_dims[0] + eps, rve_dims[1] + eps, + eps,
#                 1)
#     # First we get all surfaces on the front:
#     for i_surf in f_face:
#         # Then we get the bounding box of each front surface
#         xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(i_surf[0], i_surf[1])
#         # We translate the bounding box to the back and look for surfaces inside
#         # it:
#         b_face = gmsh.model.getEntitiesInBoundingBox(
#                     xmin - eps, ymin - eps, zmin - eps + rve_dims[2],
#                     xmax + eps, ymax + eps, zmax + eps + rve_dims[2],
#                     1)
#         # For all the matches, we compare the corresponding bounding boxes...
#         for j_surf in b_face:
#             xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = gmsh.model.getBoundingBox(
#                 j_surf[0], j_surf[1])
#             zmin2 -= rve_dims[2]
#             zmax2 -= rve_dims[2]
#             # ...and if they match, we apply the periodicity constraint
#             if (abs(xmin2 - xmin) < eps and abs(xmax2 - xmax) < eps
#                     and abs(ymin2 - ymin) < eps and abs(ymax2 - ymax) < eps
#                     and abs(zmin2 - zmin) < eps and abs(zmax2 - zmax) < eps):
#                 gmsh.model.mesh.setPeriodic(1, [j_surf[1]], [i_surf[1]], translation_f_b)
#                 print(j_surf[1], i_surf[1])
#                 print(gmsh.model.getBoundary(j_surf))
#                 print(gmsh.model.getBoundary(i_surf))
# # --------------------------------------------------------------------------------------
# 
#     factory.synchronize()

    # eps = 2e-1
    # b_vol = gmsh.model.getEntitiesInBoundingBox(
    #                             - eps, - eps,                 - eps,
    #             rve_dims[0] + eps, + eps, rve_dims[2] + eps,
    #             3)
    # gmsh.model.removeEntities(b_vol, True)
    # t_vol = gmsh.model.getEntitiesInBoundingBox(
    #                             - eps, rve_dims[1] - eps,                 - eps,
    #             rve_dims[0] + eps, rve_dims[1] + eps, rve_dims[2] + eps,
    #             3)
    # gmsh.model.removeEntities(t_vol, True)
    # l_vol = gmsh.model.getEntitiesInBoundingBox(
    #             - eps,                 - eps,                 - eps,
    #             + eps, rve_dims[1] + eps, rve_dims[2] + eps,
    #             3)
    # gmsh.model.removeEntities(l_vol, True)
    # r_vol = gmsh.model.getEntitiesInBoundingBox(
    #              rve_dims[0] - eps,                 - eps,                 - eps,
    #              rve_dims[0] + eps, rve_dims[1] + eps, rve_dims[2] + eps,
    #             3)
    # gmsh.model.removeEntities(r_vol, True)
    # f_vol = gmsh.model.getEntitiesInBoundingBox(
    #                             - eps,                  - eps, - eps,
    #             rve_dims[0] + eps, rve_dims[1]  + eps, + eps,
    #             3)
    # gmsh.model.removeEntities(f_vol, True)
    # ba_vol = gmsh.model.getEntitiesInBoundingBox(
    #                             - eps,                  - eps, rve_dims[2] - eps,
    #             rve_dims[0] + eps, rve_dims[1]  + eps, rve_dims[2] + eps,
    #             3)
    # gmsh.model.removeEntities(ba_vol, True)


    # model.setColor((2, material2[0][1]),0,0,255,a=1)

    # Set mesh size
    points = model.getEntities(0)
    
    # model.mesh.setSize(points, mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    

    # Generate a 3D mesh
    model.mesh.generate(3)

    tetra_type = model.mesh.getElementTypes(dim=3, tag=-1)[0]
    local_coord, _ = model.mesh.getIntegrationPoints(tetra_type, "Gauss4")
    _, all_jacobians, _ = model.mesh.getJacobians(tetra_type, local_coord, tag=-1)
    _, tagsAllElements, _ = model.mesh.getElements(dim=3)
    print(len(local_coord), len(tagsAllElements[0]))
    dimTagsErrors = []
    for i_ele, tagEle in enumerate(tagsAllElements[0]):
        for k_point in range(11): #int(len(local_coord)/3)):
            if all_jacobians[i_ele*11 + k_point] <= 0.00:
                dimTagsErrors.append((3, tagEle))
                break
    print("Errors", dimTagsErrors)

    model.mesh.optimize("HighOrder", force=False, niter=10, dimTags=dimTagsErrors)

    tetra_type = model.mesh.getElementTypes(dim=3, tag=-1)[0]
    local_coord, _ = model.mesh.getIntegrationPoints(tetra_type, "Gauss4")
    _, all_jacobians, _ = model.mesh.getJacobians(tetra_type, local_coord, tag=-1)
    _, tagsAllElements, _ = model.mesh.getElements(dim=3)
    print(len(local_coord), len(tagsAllElements[0]))
    dimTagsErrors = []
    for i_ele, tagEle in enumerate(tagsAllElements[0]):
        for k_point in range(11): #int(len(local_coord)/3)):
            if all_jacobians[i_ele*11 + k_point] <= 0.00:
                dimTagsErrors.append((3, tagEle))
                break
    print("Errors Final", dimTagsErrors)

    gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
    eps = 1e-4
    # Ask OpenCASCADE to compute more accurate bounding boxes of entities using
    # the STL mesh
    # --------------------------------------------------------------------------------------
    translation_l_r = [1, 0, 0, rve_dims[0],
                       0, 1, 0, 0,
                       0, 0, 1, 0,
                       0, 0, 0, 1]
    # Translation of the left face of the cube to the rigth face of the cube given as an
    # affine transformation, (4x4), written by row
    l_face = gmsh.model.getEntitiesInBoundingBox(
                - eps,                 - eps,                 - eps,
                + eps, Particle.box[1] + eps, rve_dims[2] + eps,
                2)
    # First we get all surfaces on the left:
    for i_surf in l_face:
        # Then we get the bounding box of each left surface
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(i_surf[0], i_surf[1])
        # We translate the bounding box to the right and look for surfaces inside
        # it:
        r_face = gmsh.model.getEntitiesInBoundingBox(
                    xmin - eps + rve_dims[0], ymin - eps, zmin - eps,
                    xmax + eps + rve_dims[0], ymax + eps, zmax + eps,
                    2)
        # For all the matches, we compare the corresponding bounding boxes...
        for j_surf in r_face:
            xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = gmsh.model.getBoundingBox(
                j_surf[0], j_surf[1])
            xmin2 -= rve_dims[0]
            xmax2 -= rve_dims[0]
            # ...and if they match, we apply the periodicity constraint
            if (abs(xmin2 - xmin) < eps and abs(xmax2 - xmax) < eps
                    and abs(ymin2 - ymin) < eps and abs(ymax2 - ymax) < eps
                    and abs(zmin2 - zmin) < eps and abs(zmax2 - zmax) < eps):
                gmsh.model.mesh.setPeriodic(2, [j_surf[1]], [i_surf[1]], translation_l_r)
                print(j_surf[1], i_surf[1])
                print(gmsh.model.getBoundary(j_surf))
                print(gmsh.model.getBoundary(i_surf))
# --------------------------------------------------------------------------------------
                # Ensuring periodicity
    # --------------------------------------------------------------------------------------
    gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
    translation_b_t = [1, 0, 0, 0,
                       0, 1, 0, Particle.box[1],
                       0, 0, 1, 0,
                       0, 0, 0, 1]
    # Translation of the top face of the cube to the bottom face of the cube given as an
    # affine transformation, (4x4), written by row
    b_face = gmsh.model.getEntitiesInBoundingBox(
                                - eps, - eps,                 - eps,
                rve_dims[0] + eps, + eps, rve_dims[2] + eps,
                2)
    # First we get all surfaces on the bottom:
    for i_surf in b_face:
        # Then we get the bounding box of each bottom surface
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(i_surf[0], i_surf[1])
        # We translate the bounding box upwards and look for surfaces inside
        # it:
        t_face = gmsh.model.getEntitiesInBoundingBox(
                    xmin - eps, ymin - eps + Particle.box[1], zmin - eps,
                    xmax + eps, ymax + eps + Particle.box[1], zmax + eps,
                    2)
        # For all the matches, we compare the corresponding bounding boxes...
        for j_surf in t_face:
            xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = gmsh.model.getBoundingBox(
                j_surf[0], j_surf[1])

            ymin2 -= Particle.box[1]
            ymax2 -= Particle.box[1]
            # ...and if they match, we apply the periodicity constraint
            if (abs(xmin2 - xmin) < eps and abs(xmax2 - xmax) < eps
                    and abs(ymin2 - ymin) < eps and abs(ymax2 - ymax) < eps
                    and abs(zmin2 - zmin) < eps and abs(zmax2 - zmax) < eps):
                gmsh.model.mesh.setPeriodic(2, [j_surf[1]], [i_surf[1]], translation_b_t)                
                print(j_surf[1], i_surf[1])
                print(gmsh.model.getBoundary(j_surf))
                print(gmsh.model.getBoundary(i_surf))
# --------------------------------------------------------------------------------------
    gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
    translation_f_b = [1, 0, 0, 0,
                       0, 1, 0, 0,
                       0, 0, 1, rve_dims[2],
                       0, 0, 0, 1]
    # Translation of the front face of the cube to the back face of the cube given as an
    # affine transformation, (4x4), written by row
    f_face = gmsh.model.getEntitiesInBoundingBox(
                                - eps,                 - eps, - eps,
                rve_dims[0] + eps, Particle.box[1] + eps, + eps,
                2)
    # First we get all surfaces on the front:
    for i_surf in f_face:
        # Then we get the bounding box of each front surface
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(i_surf[0], i_surf[1])
        # We translate the bounding box to the back and look for surfaces inside
        # it:
        b_face = gmsh.model.getEntitiesInBoundingBox(
                    xmin - eps, ymin - eps, zmin - eps + rve_dims[2],
                    xmax + eps, ymax + eps, zmax + eps + rve_dims[2],
                    2)
        # For all the matches, we compare the corresponding bounding boxes...
        for j_surf in b_face:
            xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = gmsh.model.getBoundingBox(
                j_surf[0], j_surf[1])
            zmin2 -= rve_dims[2]
            zmax2 -= rve_dims[2]
            # ...and if they match, we apply the periodicity constraint
            if (abs(xmin2 - xmin) < eps and abs(xmax2 - xmax) < eps
                    and abs(ymin2 - ymin) < eps and abs(ymax2 - ymax) < eps
                    and abs(zmin2 - zmin) < eps and abs(zmax2 - zmax) < eps):
                gmsh.model.mesh.setPeriodic(2, [j_surf[1]], [i_surf[1]], translation_f_b)
                print(j_surf[1], i_surf[1])
                print(gmsh.model.getBoundary(j_surf))
                print(gmsh.model.getBoundary(i_surf))



    gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
    eps = 1e-4
    # Ask OpenCASCADE to compute more accurate bounding boxes of entities using
    # the STL mesh
    # --------------------------------------------------------------------------------------
    translation_l_r = [1, 0, 0, rve_dims[0],
                       0, 1, 0, 0,
                       0, 0, 1, 0,
                       0, 0, 0, 1]
    # Translation of the left face of the cube to the rigth face of the cube given as an
    # affine transformation, (4x4), written by row
    l_face = gmsh.model.getEntitiesInBoundingBox(
                - eps,                 - eps,                 - eps,
                + eps, rve_dims[1] + eps, rve_dims[2] + eps,
                1)
    # First we get all surfaces on the left:
    for i_surf in l_face:
        # Then we get the bounding box of each left surface
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(i_surf[0], i_surf[1])
        # We translate the bounding box to the right and look for surfaces inside
        # it:
        r_face = gmsh.model.getEntitiesInBoundingBox(
                    xmin - eps + rve_dims[0], ymin - eps, zmin - eps,
                    xmax + eps + rve_dims[0], ymax + eps, zmax + eps,
                    1)
        # For all the matches, we compare the corresponding bounding boxes...
        for j_surf in r_face:
            xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = gmsh.model.getBoundingBox(
                j_surf[0], j_surf[1])
            xmin2 -= rve_dims[0]
            xmax2 -= rve_dims[0]
            # ...and if they match, we apply the periodicity constraint
            if (abs(xmin2 - xmin) < eps and abs(xmax2 - xmax) < eps
                    and abs(ymin2 - ymin) < eps and abs(ymax2 - ymax) < eps
                    and abs(zmin2 - zmin) < eps and abs(zmax2 - zmax) < eps):
                gmsh.model.mesh.setPeriodic(1, [j_surf[1]], [i_surf[1]], translation_l_r)
                print(j_surf[1], i_surf[1])
                print(gmsh.model.getBoundary(j_surf))
                print(gmsh.model.getBoundary(i_surf))
# --------------------------------------------------------------------------------------
                # Ensuring periodicity
    # --------------------------------------------------------------------------------------
    gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
    translation_b_t = [1, 0, 0, 0,
                       0, 1, 0, rve_dims[1],
                       0, 0, 1, 0,
                       0, 0, 0, 1]
    # Translation of the top face of the cube to the bottom face of the cube given as an
    # affine transformation, (4x4), written by row
    b_face = gmsh.model.getEntitiesInBoundingBox(
                                - eps, - eps,                 - eps,
                rve_dims[0] + eps, + eps, rve_dims[2] + eps,
                1)
    # First we get all surfaces on the bottom:
    for i_surf in b_face:
        # Then we get the bounding box of each bottom surface
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(i_surf[0], i_surf[1])
        # We translate the bounding box upwards and look for surfaces inside
        # it:
        t_face = gmsh.model.getEntitiesInBoundingBox(
                    xmin - eps, ymin - eps + rve_dims[1], zmin - eps,
                    xmax + eps, ymax + eps + rve_dims[1], zmax + eps,
                    1)
        # For all the matches, we compare the corresponding bounding boxes...
        for j_surf in t_face:
            xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = gmsh.model.getBoundingBox(
                j_surf[0], j_surf[1])

            ymin2 -= rve_dims[1]
            ymax2 -= rve_dims[1]
            # ...and if they match, we apply the periodicity constraint
            if (abs(xmin2 - xmin) < eps and abs(xmax2 - xmax) < eps
                    and abs(ymin2 - ymin) < eps and abs(ymax2 - ymax) < eps
                    and abs(zmin2 - zmin) < eps and abs(zmax2 - zmax) < eps):
                gmsh.model.mesh.setPeriodic(1, [j_surf[1]], [i_surf[1]], translation_b_t)                
                print(j_surf[1], i_surf[1])
                print(gmsh.model.getBoundary(j_surf))
                print(gmsh.model.getBoundary(i_surf))
# --------------------------------------------------------------------------------------
    gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
    translation_f_b = [1, 0, 0, 0,
                       0, 1, 0, 0,
                       0, 0, 1, rve_dims[2],
                       0, 0, 0, 1]
    # Translation of the front face of the cube to the back face of the cube given as an
    # affine transformation, (4x4), written by row
    f_face = gmsh.model.getEntitiesInBoundingBox(
                                - eps,                 - eps, - eps,
                rve_dims[0] + eps, rve_dims[1] + eps, + eps,
                1)
    # First we get all surfaces on the front:
    for i_surf in f_face:
        # Then we get the bounding box of each front surface
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(i_surf[0], i_surf[1])
        # We translate the bounding box to the back and look for surfaces inside
        # it:
        b_face = gmsh.model.getEntitiesInBoundingBox(
                    xmin - eps, ymin - eps, zmin - eps + rve_dims[2],
                    xmax + eps, ymax + eps, zmax + eps + rve_dims[2],
                    1)
        # For all the matches, we compare the corresponding bounding boxes...
        for j_surf in b_face:
            xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = gmsh.model.getBoundingBox(
                j_surf[0], j_surf[1])
            zmin2 -= rve_dims[2]
            zmax2 -= rve_dims[2]
            # ...and if they match, we apply the periodicity constraint
            if (abs(xmin2 - xmin) < eps and abs(xmax2 - xmax) < eps
                    and abs(ymin2 - ymin) < eps and abs(ymax2 - ymax) < eps
                    and abs(zmin2 - zmin) < eps and abs(zmax2 - zmax) < eps):
                gmsh.model.mesh.setPeriodic(1, [j_surf[1]], [i_surf[1]], translation_f_b)
                print(j_surf[1], i_surf[1])
                print(gmsh.model.getBoundary(j_surf))
                print(gmsh.model.getBoundary(i_surf))
# --------------------------------------------------------------------------------------

    factory.synchronize()


    # Write the mesh to the .msh file
    meshfile_temp = title + "_temp.msh"
    meshfile = title + '.msh'
    vtk_temp = title + "_temp.vtk"
    vtk = title + ".vtk"
    gmsh.write(meshfile_temp)
    gmsh.write(vtk_temp)

    
    # Close GMSH
    gmsh.finalize()
    # ==========================================================================================
    # Convert it to LINKS format and write the respective input file
    # ==========================================================================================

    fin = open(meshfile_temp, "rt")
    fout = open(meshfile, "wt")

    for line in fin:
    	fout.write(line.replace(',', '.'))

    fin.close()
    fout.close()
    os.remove(meshfile_temp)

    gmshToLinks(meshfile, title, 3, Particle.list_phases, Particle.matrix_phase)

    fin = open(vtk_temp, "rt")
    fout = open(vtk, "wt")

    for line in fin:
    	fout.write(line.replace(',', '.'))

    fin.close()
    fout.close()
    os.remove(vtk_temp)

def gmshToLinks(meshfile, title, dim, list_phases, matrix_phase):
    '''
    This function writes an input file for LINKS using a gmsh mesh saved at meshfile.

    Parameters
    ----------
    meshfile: string
        Path to the gmsh file containing the FEM mesh.
    '''
    mesh = readMesh(meshfile)

    nodeID, coord = mesh.getAllNodes()
    elementID, connectivities, elementType = mesh.getElementsByDim(dim)
    elemMat = len(list_phases)*[0]
    conMat = len(list_phases)*[0]
    typeMat = len(list_phases)*[0]
    elemMat[0], conMat[0], typeMat[0] = \
        mesh.getElementsByName("Phase " + str(matrix_phase))
    list_phases.remove(matrix_phase)
    for i_phase in range(len(list_phases)):
        elemMat[i_phase+1], conMat[i_phase+1], typeMat[i_phase+1] = \
            mesh.getElementsByName("Phase " + str(list_phases[i_phase]))
    if dim == 2:
        shutil.copy("LINKS_header.dat", title + ".rve")
    elif dim == 3:
        shutil.copy("LINKS_header_3d.dat", title + ".rve")

    with open(title + ".rve", "a") as dat:
        # dat.write("TITLE\n{0}".format(title))
        # dat.write("\n\nELEMENT_GROUPS 2\n1 1 1")
        # dat.write("\n\nELEMENT_TYPES 1\n1 {0} \n3 GP".format(elementType[0]))
        # # dat.write("\n\nELEMENT_TYPES 1\n1 TETRA4 \n4 GP")
        # dat.write("\n\nMATERIALS 2\n1 VON_MISES\n0.0\n2000E3 0.3\n2\n0.0 10000\n1.0 12000\n2 VON_MISES\n0.0\n200E3 0.3\n2\n0.0 540\n1.0 940\n")
        dat.write("\n\nNODE_COORDINATES {0} CARTESIAN".format(len(nodeID)))
        for i in range(len(nodeID)):
            dat.write("\n{0} {1} {2} {3}".format(nodeID[i], coord[i][0], coord[i][1], coord[i][2]))
        dat.write("\n\nELEMENTS {0}".format(len(elementID)))
        for i_phase in range(len(list_phases)+1):
            for j_node in range(len(elemMat[i_phase])):
                dat.write("\n{0} {1} ".format(
                    elemMat[i_phase][j_node], i_phase+1))
                for k_con in conMat[i_phase][j_node]:
                    dat.write("{0} ".format(k_con))


def generateMeshFFT(particles, options, disc_ext):
    '''
    This functions generates a regular grid as an array to be used in an FFT analysis.
    '''
    for i_mesh_size in range(len(options['n_voxels_dims'])):
    # Running through the different mesh sizes
        n_voxels_dims = options['n_voxels_dims'][i_mesh_size]
        # current mesh size
        pixel_dims = options['rve_dims']/n_voxels_dims
        # Dimension of the pixels
        if len(Particle.box) == 2:
        # This is a 2D dimnensional problem
            if len(options['rve_dims']) == 3:
            # Tridimensional problem with particles obtained by extruding the 2D simulation
            # box
                n_voxels_dims_og = n_voxels_dims
                # Saving the original voxel descretization
                n_voxels_length = n_voxels_dims[particles[0].direction_fibers]
                # Number of voxels in the orthogonal direction to the simulation box
                n_voxels_dims = np.delete(n_voxels_dims, particles[0].direction_fibers)
                # Voxel descritizing the 2D box
            regular_grid = np.full((n_voxels_dims[0], n_voxels_dims[1]),
                                   int(Particle.matrix_phase), dtype=int)
            # Initializing the regular
            for i_row in range(n_voxels_dims[0]):
            # Running through the pixels from left to right
                for j_column in range(n_voxels_dims[1]):
                # Running thorugh the pixels from bottom to top
                    center_pixel_i_j = \
                        np.array([(i_row+0.5)*pixel_dims[0], (j_column+0.5)*pixel_dims[1]])
                    # Center of the pixel corresponding to row i_row and column j_column
                    for k_particle in particles:
                    # Running through all the particles
                        diff_in_box = k_particle.position_center - center_pixel_i_j
                        # Difference vector between the center of the two ellipses
                        diff_nearest_other = \
                            Particle.box*np.round(diff_in_box/Particle.box)
                        # Vector from the particle whose center is in the RVE to the neares
                        # image
                        if k_particle.pointInside(center_pixel_i_j + diff_nearest_other):
                        # The center of the pixel is inside particle k_particle
                            regular_grid[i_row, j_column] = k_particle.phase
                            # Setting pixel [i_row, j_column] as belong to the phase of
                            # particle k_particle
            if len(options['rve_dims']) == 3:
            # Tridimensional problem with particles obtained by extruding the 2D simulation
            # box
                regular_grid = np.stack([regular_grid for _ in range(n_voxels_length)],
                                        axis=particles[0].direction_fibers)
                # Obtaining the extrusion of the 2D box by stacking it in the direction of
                # the fibers
                if True:
                    plotVoxels(regular_grid, Particle.matrix_phase, Particle.list_phases, Particle.file_path + "_"
                               + str(n_voxels_dims_og[0]) + "_" + str(n_voxels_dims_og[1])
                               + "_" + str(n_voxels_dims_og[0]) + "." + disc_ext)
                # Ploting the regular grid

                np.save(Particle.file_path + "_" + str(n_voxels_dims_og[0]) + "_"
                        + str(n_voxels_dims_og[1]) + "_" + str(n_voxels_dims_og[2])
                        + ".rgmsh", regular_grid)
            else:
                if True:
                    plotPixels(regular_grid, Particle.file_path + "_" + str(n_voxels_dims[0])
                               + "_" + str(n_voxels_dims[1]) + "." + disc_ext)
                # Ploting the regular grid
                np.save(Particle.file_path + "_" + str(n_voxels_dims[0]) + "_"
                        + str(n_voxels_dims[1]) + ".rgmsh", regular_grid)
        elif len(Particle.box) == 3:
        # This is a 2D dimnensional problem
            regular_grid = np.full((n_voxels_dims[0], n_voxels_dims[1], n_voxels_dims[2]),
                                   int(Particle.matrix_phase), dtype=int)
            # Initializing the regular grid
            for i_row in range(n_voxels_dims[0]):
            # Running through the pixels from left to right
                for j_column in range(n_voxels_dims[1]):
                # Running thorugh the pixels from bottom to top
                    for k_layer in range(n_voxels_dims[2]):
                    # Running thorugh the pixels from bottom to top
                        center_pixel_i_j_k = \
                            np.array([(i_row + 0.5)*pixel_dims[0],
                                      (j_column + 0.5)*pixel_dims[1],
                                      (k_layer + 0.5)*pixel_dims[2]])
                        # Center of the pixel corresponding to row i_row, column j_column and
                        # layer k_layer
                        for l_particle in particles:
                        # Running through all the particles
                            diff_in_box = l_particle.position_center - center_pixel_i_j_k
                            # Difference vector between the center of the two ellipses
                            diff_nearest_other = \
                                Particle.box*np.round(diff_in_box/Particle.box)
                            # Vector from the particle whose center is in the RVE to the nearest
                            # image
                            if l_particle.pointInside(center_pixel_i_j_k + diff_nearest_other):
                            # The center of the pixel is inside particle k_particle
                                regular_grid[i_row, j_column, k_layer] = l_particle.phase
                                # Setting pixel [i_row, j_column, k_layer] as belong to the
                                # phase of particle k_particle
            if True:
                plotVoxels(regular_grid, Particle.matrix_phase, Particle.list_phases, Particle.file_path + "_"
                           + str(n_voxels_dims[0]) + "_" + str(n_voxels_dims[1]) + "_"
                           + str(n_voxels_dims[0]) + "." + disc_ext)
            # Ploting the regular grid

            np.save(Particle.file_path + "_" + str(n_voxels_dims[0]) + "_"
                    + str(n_voxels_dims[1]) + "_" + str(n_voxels_dims[2]) + ".rgmsh", regular_grid)


def generateMesh(particles, disc_ext, discret_spec_array):
    """Generate a mesh"""
    if disc_ext == 'rgmsh':
    # A mesh for FFT was requested
        generateMeshFFT(particles, discret_spec_array, disc_ext)
        # Generating the FFT mesh as a regular grid and saving it in a .dat file
    if disc_ext == 'femsh':
    # A mesh for FEM was requested
        try:
            mesh_size = discret_spec_array['mesh_size']
            # Saving the value of the mesh size
        except KeyError:
            print('The mesh size for the FEM method was not supplied correctly')
            quit()
        generateMeshFEM(particles, mesh_size, discret_spec_array['rve_dims'])
        # Generating the FEM mesh using gmsh and saving an input data file for LINKS


def checkMeshSpecs(disc_ext, discret_spec_array):
    """Check if the extension has been correctly specified."""
    if disc_ext == 'rgmsh':
    # A regular mesh was specified
        necessary_parameters = {'rve_dims', 'n_voxels_dims'}
        try:
            if any([necessary_parameter not in discret_spec_array
                    for necessary_parameter in necessary_parameters]):
            # Checking if all the required parameters were supplied
                raise errors.InsufficientInfoMesh(list(discret_spec_array.keys()),
                                                  necessary_parameters,
                                                  disc_ext)
            rve_dims = discret_spec_array['rve_dims']
            n_voxels_dims = discret_spec_array['n_voxels_dims']
            # Saving the RVE dims and the number of voxels in each direction
            if (rve_dims.shape != (3,) and rve_dims.shape != (2,)):
            # The RVE dims must be given as 1-arrays with 2 or 3 elements
                raise(errors.UnexpectedValue(rve_dims, 'rve_dims',
                                             '1-array with shape (2,) or (3,)'))
            if any(rve_dims < 0):
            # The RVE dimensions must be positive real numbers
                raise errors.UnexpectedValue(rve_dims, 'rve_dims',
                                             'array of positive reals')
            if len(rve_dims) != len(n_voxels_dims.T):
            # The dimension of RVE is not compatible with number of voxels specified
                raise errors.IncompatibleDimension('rve_dims', 'n_voxels_dims')
            if any([not np.issubdtype(n_voxels_dims.flat[i_voxel_dim], np.integer)
                    or n_voxels_dims.flat[i_voxel_dim] < 1
                    for i_voxel_dim in range(n_voxels_dims.size)]):
                # The specified number of voxels in any direction must be a positve integer
                raise errors.UnexpectedValue(n_voxels_dims, 'n_voxels_dims',
                                             'array of positive integers')
        except (errors.InsufficientInfoMesh, errors.IncompatibleDimension,
                errors.UnexpectedValue) as error:
            error.message()
            quit()
    elif disc_ext == 'femsh':
    # A finite elment mesh was specified
        necessary_parameters = {'rve_dims', 'mesh_size'}
        try:
            if any([necessary_parameter not in discret_spec_array
                    for necessary_parameter in necessary_parameters]):
                raise errors.InsufficientInfoMesh(list(discret_spec_array.keys()),
                                                  necessary_parameters,
                                                  disc_ext)
            rve_dims = discret_spec_array['rve_dims']
            mesh_size = discret_spec_array['mesh_size']
            if rve_dims.shape != (3,) and rve_dims.shape != (2,):
            # The RVE dims must be given as 1-arrays with 2 or 3 elements
                raise errors.UnexpectedValue(rve_dims, 'rve_dims',
                                             '1-array with shape (2,) or (3,)')
            if any(rve_dims < 0):
            # The RVE dimensions must be positive real numbers
                raise errors.UnexpectedValue(rve_dims, 'rve_dims',
                                             'array of positive reals')
            # Saving the RVE dims and the mesh size
            if mesh_size <= 0:
            # The meshsize is smaller than one
                raise errors.UnexpectedValue(mesh_size, 'mesh_size',
                                             'positive real')
        except (errors.InsufficientInfoMesh, errors.UnexpectedValue) as error:
            error.message()
            quit()
    elif disc_ext == 'nomsh':
    # No mesh was specified
        necessary_parameters = {'rve_dims'}
        try:
            if any([necessary_parameter not in discret_spec_array
                    for necessary_parameter in necessary_parameters]):
                raise errors.InsufficientInfoMesh(list(discret_spec_array.keys()),
                                                  necessary_parameters,
                                                  disc_ext)
            rve_dims = discret_spec_array['rve_dims']
            if rve_dims.shape != (3,) and rve_dims.shape != (2,):
            # The RVE dims must be given as 1-arrays with 2 or 3 elements
                raise errors.UnexpectedValue(rve_dims, 'rve_dims',
                                             '1-array with saphe (2,) or (3,)')
            if any(rve_dims < 0):
            # The RVE dimensions must be positive real numbers
                raise errors.UnexpectedValue(rve_dims, 'rve_dims',
                                             'array of positive reals')
        except (errors.InsufficientInfoMesh, errors.UnexpectedValue) as error:
            error.message()
            quit()
    else:
    # Unsupported mesh
        try:
            raise errors.UnsupportedMesh(disc_ext)
        except errors.UnsupportedMesh as error:
            error.message()
            quit()




if __name__ == '__main__':

    meshfile = "/home/zeluis/Documents/Tese/programa/results/error_debug/Sphere_20_0.2_8/Sphere_20_0.2.msh"
    title = 'somthing'
    list_phases = ['1', '2']
    matrix_phase = '1'
    dim = 3
    gmshToLinks(meshfile, title, dim, list_phases, matrix_phase)
