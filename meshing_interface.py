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

def generateMeshFEM(particles, mesh_size, element_type="tri3", **kwargs):
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

    output_term = kwargs.pop('output_term',0)
    # Option for the output in the terminal
    if particles[0].dim==2:
    # It is a 2D problem
        if element_type=="tri3":
        # (Defaul option) Linear Triangular element
            generateMeshFEM2D(particles, mesh_size, output_term=output_term)
            # Generating a mesh of linear triangular elements
        elif element_type=="tri6":
        # Quadratic Triangular element
            generateMeshFEM2D(particles, mesh_size, element_order=2,
                output_term=output_term)
            # Generating a mesh of linear triangular elements
        elif element_type=="quad4":
        # Linear Rectangular element
            generateMeshFEM2D(particles, mesh_size, force_recomb_all=1,
                output_term=output_term)
            # Generating a mesh of linear triangular elements
        elif elment_type=="quad8":
        # 2nd order rectangular elment of the serendipity family
            generateMeshFEM2D(particles, mesh_size, force_recomb_all=1,
                element_order=2, elemnet_order_incomp=1, output_term=output_term)
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
    elif particles[0].dim==3:
    # It is a 3D problem
        pass

def generateMeshFEM2D(particles, mesh_size, mesh_alg=6, force_recomb_all=0, element_order=1,
    recomb_alg=1, element_order_incomp=0, output_term=0):
    '''
    This function generates the mesh for the Finite Element. It generates by default linear
    triangular elements.

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
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", recomb_alg)

    # Force recombination in all surfaces
    gmsh.option.setNumber("Mesh.RecombineAll", force_recomb_all)

    # Number of topological optimization passes of recombined surface meshes (5 by default)
    gmsh.option.setNumber("Mesh.RecombineOptimizeTopology", 5)

    # Renumber nodes and elements after mesh generation
    gmsh.option.setNumber("Mesh.Renumber", 1)

    # Save all elements even if they do not belong to physical groups
    gmsh.option.setNumber("Mesh.SaveAll", 1)

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
    print(title)
    model.add(title)

    x = 0
    y = 0
    z = 0
    lx = Particle.box[0]
    ly = Particle.box[1]

    rectTag = factory.addRectangle(0, 0, 0, Particle.box[0], Particle.box[1])
    # RVE

    particleTags = []
    rotateTags = []
    k_particle_image = 0
    phaseDimTag = dict.fromkeys(Particle.list_phases, [])
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
                    rx = i_particle.radius*0.9
                    ry = i_particle.radius*0.9
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
                    rx = i_particle.semi_major_axis*0.9
                    ry = i_particle.semi_minor_axis*0.9
                    alpha = i_particle.angle
                    # Saving the properties of the particles
                    particleTags.append(factory.addDisk(xc, yc, zc, rx, ry))
                    # Creating the ellipse without rotation
                    # Rotate the disk
                    factory.synchronize()
                    rotateTags.append([(2, particleTags[k_particle_image])])
                    rotateTags[k_particle_image].extend(
                        model.getBoundary([2, particleTags[k_particle_image]]))
                    factory.rotate(rotateTags[k_particle_image], xc, yc, zc, 0, 0, 1, alpha)

                    phaseDimTag[i_particle.phase].append(
                        (2, particleTags[k_particle_image]))

                    factory.synchronize()
                    k_particle_image += 1

    outDimTag, outDimTagMap = factory.intersect(
        [(2, rectTag)], [(2, particleTags[k]) for k in range(9*len(particles))], removeObject=False, removeTool=True)

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
        materials.append([ value[1] for value in outDimTag2 if value in temp ])


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

    
    # model.setColor((2, material2[0][1]),0,0,255,a=1)

    # Set mesh size
    points = model.getEntities(0)
    model.mesh.setSize(points, mesh_size)

    # Generate a 2D mesh
    model.mesh.generate(2)



    # Write the mesh to the .msh file
    meshfile = title + ".msh"
    gmsh.write(meshfile)

    # Close GMSH
    gmsh.finalize()
    # ==========================================================================================
    # Convert it to LINKS format and write the respective input file
    # ==========================================================================================

    gmshToLinks(meshfile, title)

def generateMeshFEM3D():
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
    gmsh.option.setNumber("General.Terminal", 1)

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
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)

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
    gmsh.option.setNumber("Mesh.Recombine3DAll", 0)

    # Recombination level in 3D
    # -------------------------
    # 0 - hex (default)
    # 1 - hex + prisms
    # 2 - hex + prisms + pyramids
    gmsh.option.setNumber("Mesh.Recombine3DLevel", 0)

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
    gmsh.option.setNumber("Mesh.SaveAll", 1)

    # Number of smoothing step applied to the final mesh
    gmsh.option.setNumber("Mesh.Smoothing", 1)

    # Element order
    gmsh.option.setNumber("Mesh.ElementOrder", 1)

    # Crete second-order nodes by linear interpolation
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)

    # Second-order incomplete elements
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
    # ==========================================================================================
    # Generate the finite element mesh
    # ==========================================================================================
    # Define model name
    title = os.path.splitext(os.path.basename(__file__))[0]
    model.add(title)

    x = 0
    y = 0
    z = 0
    lx = Particle.box[0]
    ly = Particle.box[1]

    rectTag = factory.addRectangle(0, 0, 0, Particle.box[0], Particle.box[1])
    # RVE

    circTag = []
    outDimTag = [rectTag]
    rotateTags = []
    k = 0
    for i_particle in particles:
    # Running through all the particles
        for j in range(-1,2):
            for p in range(-1,2):
                xc = i_particle.position_center[0] + Particle.box[0]*j
                yc = i_particle.position_center[1] + Particle.box[1]*p
                zc = 0
                rx = i_particle.semi_major_axis*0.9
                ry = i_particle.semi_minor_axis*0.9
                alpha = i_particle.angle
                # Saving the properties of the particles
                circTag.append(factory.addDisk(xc, yc, zc, rx, ry))
                # Creating the ellipse without rotation
                print(circTag)
                print(k)
                # Rotate the disk
                factory.synchronize()
                rotateTags.append([(2, circTag[k])])
                rotateTags[k].extend(model.getBoundary([2, circTag[k]]))
                print(rotateTags)
                factory.rotate(rotateTags[k], xc, yc, zc, 0, 0, 1, alpha)
                # 
                # # Make a hole in the rectangle with the disk
                
                factory.synchronize()
                k = k + 1
                print(circTag)

    outDimTag, outDimTagMap = factory.intersect(
        [(2, rectTag)], [(2, circTag[k]) for k in range(9*len(particles))], removeObject=False, removeTool=True)

    print(outDimTag)

    factory.synchronize()

    outDimTag2, outDimTagMap = factory.fragment(
        [(2, rectTag)], outDimTag, removeObject=True, removeTool=True)

    
    # factory.remove(outDimTag[6:7])

    # print(outDimTag)
    # outDimTag2, outDimTagMap = factory.intersect(
    #     [(2, rectTag)], [outDimTag]  , removeObject=True, removeTool=True)
    
    material1 = [outDimTag2[-1]]
    material2 = outDimTag2[0:-1]
    print(material2)




    # Set the mesh size on the geometry points
    # Synchronize the CAD engine (always needed before generating the mesh)
    # It may also be useful for some intermidate operations, like checking the tags of entities
    factory.synchronize()

    # Set boundaries
    eps = 1e-3
    # bottom = model.getEntitiesInBoundingBox(x - lx/2, y - ly/2, -eps, x + eps, y + eps, eps)
    # 
    # factory.remove(bottom)
    # top = model.getEntitiesInBoundingBox(x - eps, y + ly - eps, -eps, x + lx + eps, y + ly + eps, eps, dim=2)
    # left = model.getEntitiesInBoundingBox(x - eps, y - eps, -eps, x + eps, y + ly + eps, eps, dim=2)
    # right = model.getEntitiesInBoundingBox(x + lx - eps, y - eps, -eps, x + lx + eps, y + ly + eps, eps, dim=2)
    # 
    # bottomTag = model.addPhysicalGroup(bottom[0][0], [bottom[0][1]])
    # model.setPhysicalName(bottom[0][0], bottomTag, "Bottom Boundary")
    # 
    # topTag = model.addPhysicalGroup(top[0][0], [top[0][1]])
    # model.setPhysicalName(top[0][0], topTag, "Top Boundary")
    # 
    # leftTag = model.addPhysicalGroup(left[0][0], [left[0][1]])
    # model.setPhysicalName(left[0][0], leftTag, "Left Boundary")
    # 
    # rightTag = model.addPhysicalGroup(right[0][0], [right[0][1]])
    # model.setPhysicalName(right[0][0], rightTag, "Right Boundary")
    # 
    material1Tag = model.addPhysicalGroup(material1[0][0], [material1[0][1]])
    model.setPhysicalName(material1[0][0], material1Tag, "Material 1")
    
    material2Tag = model.addPhysicalGroup(material2[0][0], [material2[i][1] for i in range(len(material2))])
    model.setPhysicalName(material2[0][0], material2Tag, "Material 2")

    factory.synchronize()

    
    model.setColor((2, material2[0][1]),0,0,255,a=1)

    # Set mesh size
    points = model.getEntities(0)
    model.mesh.setSize(points, 0.03)

    # Generate a 2D mesh
    model.mesh.generate(2)



    # Write the mesh to the .msh file
    meshfile = title + ".msh"
    gmsh.write(meshfile)

    # Close GMSH
    gmsh.finalize()
    # ==========================================================================================
    # Convert it to LINKS format and write the respective input file
    # ==========================================================================================

def gmshToLinks(meshfile, title):
    '''
    This function writes an input file for LINKS using a gmsh mesh saved at meshfile.

    Parameters
    ----------
    meshfile: string
        Path to the gmsh file containing the FEM mesh.
    '''
    mesh = readMesh(meshfile)

    nodeID, coord = mesh.getAllNodes()
    elementID, connectivities, elementType = mesh.getElementsByDim(2)
    elemMat1, conMat1, typeMat1 = mesh.getElementsByName("Phase 4")
    elemMat2, conMat2, typeMat2 = mesh.getElementsByName("Phase 2")

    shutil.copy("LINKS_header.dat",title + ".dat")

    with open(title + ".dat", "a") as dat:
        dat.write("TITLE\n{0}".format(title))
        dat.write("\n\nELEMENT_GROUPS 2\n1 1 1\n2 1 2")
        dat.write("\n\nELEMENT_TYPES 1\n1 {0} \n3 GP".format(elementType[0]))
        dat.write("\n\nMATERIALS 2\n1 VON_MISES\n0.0\n2000E3 0.3\n2\n0.0 10000\n1.0 12000\n2 VON_MISES\n0.0\n200E3 0.3\n2\n0.0 540\n1.0 940\n")
        dat.write("\n\nNODE_COORDINATES {0} CARTESIAN".format(len(nodeID)))
        for i in range(len(nodeID)):
            dat.write("\n{0} {1} {2} {3}".format(nodeID[i], coord[i][0], coord[i][1], coord[i][2]))
        dat.write("\n\nELEMENTS {0}".format(len(elementID)))
        for i in range(len(elemMat1)):
            dat.write("\n{0} 1 ".format(elemMat1[i]))
            for j in conMat1[i]:
                dat.write("{0} ".format(j))
        for i in range(len(elemMat2)):
            dat.write("\n{0} 2 ".format(elemMat2[i]))
            for j in conMat2[i]:
                dat.write("{0} ".format(j))
        dat.write("\n\nLOADINGS")
        dat.write("\n\nINCREMENTS 20")
        dat.write("\n20")
        dat.write("\n {0} 1e-6 10".format(1.0/20.0))

def generateMeshFFT(particles, options):
    '''
    This functions generates a regular grid as an array to be used in an FFT analysis.
    '''
    # discret_spec_array['rgmsh']['rve_dims'] = [ 1.0, 1.0 ]
    # discret_spec_array['rgmsh']['n_voxels_dims'] = [ 20, 10 ]

    pixel_dims = options['rve_dims']/options['n_voxels_dims']
    # Dimension of the pixels
    if len(options['rve_dims'])==2:
    # This is a 2D dimnensional problem
        regular_grid = np.zeros((options['n_voxels_dims'][0], options['n_voxels_dims'][1]))
        # Initializing the regular
        for i_row in range(options['n_voxels_dims'][0]):
        # Running through the pixels from left to right
            for j_column in range(options['n_voxels_dims'][1]):
            # Running thorugh the pixels from bottom to top
                center_pixel_i_j = \
                    np.array([(i_row+0.5)*pixel_dims[0], (j_column+0.5)*pixel_dims[1]])
                # Center of the pixel corresponding to row i_row and column j_column
                for k_particle in particles:
                # Running through all the particles
                    diff_in_box = k_particle.position_center - center_pixel_i_j
                    # Difference vector between the center of the two ellipses
                    diff_nearest_other = \
                        options['rve_dims']*np.round(diff_in_box/options['rve_dims'])
                    # Vector from the particle whose center is in the RVE to the neares image
                    if k_particle.pointInside(center_pixel_i_j + diff_nearest_other):
                    # The center of the pixel is inside particle k_particle
                        regular_grid[i_row,j_column] = k_particle.phase
                        # Setting pixel [i_row, j_column] as belong to the phase of
                        # particle k_particle
    print(regular_grid)
    np.save(Particle.file_path, regular_grid)

def doc(c):
    """
        Add text to the axes.

        Add the text *s* to the axes at location *x*, *y* in data coordinates.

        Parameters
        ----------
        x, y : scalars
            The position to place the text. By default, this is in data
            coordinates. The coordinate system can be changed using the
            *transform* parameter.

        s : str
            The text.

        fontdict : dictionary, optional, default: None
            A dictionary to override the default text properties. If fontdict
            is None, the defaults are determined by your rc parameters.

        withdash : boolean, optional, default: False
            Creates a `~matplotlib.text.TextWithDash` instance instead of a
            `~matplotlib.text.Text` instance.

        Returns
        -------
        text : `.Text`
            The created `.Text` instance.

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.text.Text` properties.
            Other miscellaneous text parameters.

        Examples
        --------
        Individual keyword arguments can be used to override any given
        parameter::

            >>> text(x, y, s, fontsize=12)

        The default transform specifies that text is in data coords,
        alternatively, you can specify text in axis coords ((0, 0) is
        lower-left and (1, 1) is upper-right).  The example below places
        text in the center of the axes::

            >>> text(0.5, 0.5, 'matplotlib', horizontalalignment='center',
            ...      verticalalignment='center', transform=ax.transAxes)

        You can put a rectangular box around the text instance (e.g., to
        set a background color) by using the keyword *bbox*.  *bbox* is
        a dictionary of `~matplotlib.patches.Rectangle`
        properties.  For example::

            >>> text(x, y, s, bbox=dict(facecolor='red', alpha=0.5))
        """
    pass
