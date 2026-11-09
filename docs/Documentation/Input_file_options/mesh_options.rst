Mesh Options
========================

Meshes (O)
--------------

**Meaning:** The meshes to make of each sample, each one under the name of
the mesher that makes it, in the ``meshes`` section of the input file.

**Syntax:**

.. code-block:: yaml

   meshes:
     mesh_type_1:
       mesh_option_1: y
       mesh_option_2: y
     mesh_type_2:
       mesh_option_1: y
       mesh_option_2: y

- ``mesh_type_X``: ``{voxel, gmsh}`` (default: no mesh)

  - ``voxel`` - Regular grid mesh. A given voxel is set to the phase to
    which its center point belongs.
  - ``gmsh`` - Finite element mesh conforming to the particles. Generated
    using gmsh.

- ``mesh_option_Y``: str. The mandatory and optional mesh options depend on
  the mesh type specified.

Example:

.. code-block:: yaml

   meshes:
     gmsh: {elements_per_particle: 4, element_type: tri3}
     voxel: {n_voxels_dims: [64, 64]}

``voxel`` - 1 mandatory parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Number of Voxels in Each Direction (M)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Meaning:** Number of voxels discretizing each side of the microstructure.
Several lists give a grid at each of several resolutions.

**Syntax:**

.. code-block:: yaml

   meshes:
     voxel:
       n_voxels_dims: [x1, x2]          # 2D
       n_voxels_dims: [x1, x2, x3]      # 3D
       n_voxels_dims: [[32, 32], [64, 64]]   # several resolutions

- ``x1``: int
- ``x2``: int
- ``x3``: int

``gmsh`` - 1 mandatory parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One of ``mesh_size`` and ``elements_per_particle`` must be given; if both
are given, the smaller of the two sizes is used.

Mesh Size (M)
^^^^^^^^^^^^^^

**Meaning:** Largest mesh size allowed. The mesh size is refined from the
curvature of the geometry. It stops controlling the mesh once it is above the
smallest radius, so ``elements_per_particle`` is preferred.

**Syntax:**

.. code-block:: yaml

   meshes:
     gmsh:
       mesh_size: x

- ``x``: float

Elements per Particle (M)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Meaning:** Number of elements across the smallest particle, from which the
mesh size is computed.

**Syntax:**

.. code-block:: yaml

   meshes:
     gmsh:
       elements_per_particle: x

- ``x``: int

Element Type (O)
^^^^^^^^^^^^^^^^^

**Meaning:** Type of element used to mesh the microstructure.

**Syntax:**

.. code-block:: yaml

   meshes:
     gmsh:
       element_type: x

- ``x``: ``{tri3, tri6, quad4, quad8}`` in 2D, ``{tetra4, tetra10}`` in 3D
  (default: ``tri3``)

.. note::
   ``mesh_size``, ``elements_per_particle`` and ``element_type`` also take a
   list, and a mesh is made for every combination, named after its element
   and resolution: ``element_type: [tri3, tri6]`` with
   ``elements_per_particle: [4, 8]`` makes four meshes, ``tri3_epp4``,
   ``tri3_epp8``, ``tri6_epp4`` and ``tri6_epp8``.

Options of both mesh types (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Formats
^^^^^^^^

**Meaning:** Formats the mesh is written in, besides the VTK file that is
always written. An empty list writes the VTK file alone, from which
``geommicgen-translate`` writes the other formats later;
``geommicgen-translate --list-formats`` lists them.

**Syntax:**

.. code-block:: yaml

   meshes:
     mesh_type:
       formats: [x1, x2]

- ``xi``: str (default: ``links`` for ``gmsh``, ``crate`` for ``voxel``)

File Name
^^^^^^^^^^

**Meaning:** Name of the mesh files, without an extension. It is refused for
several meshes or grids under one mesh type, and for two meshes that would
write a file in common.

**Syntax:**

.. code-block:: yaml

   meshes:
     mesh_type:
       file_name: x

- ``x``: str (default: the name of the input file and what discretises it)

Options of the formats
^^^^^^^^^^^^^^^^^^^^^^^

Each format takes the options it understands:

- ``gauss_points``: int. Gauss points written for each element of the LINKS
  mesh that takes such a line (default: what LINKS uses for the element).
- ``boundary_type``: str. Constraint of the RVE that the example LINKS input
  file asks for, one LINKS accepts, such as ``Taylor_Condition``,
  ``Linear_Condition`` or ``Mortar_Periodic_Condition`` (default:
  ``Periodic_Condition``).
- ``periodic_constraints``: bool. Whether the Abaqus input file carries the
  equations that make the cell periodic (default: ``true``).


Offset Use for Better Meshing (O)
------------------------------------

The offset is now given in the ``generation`` section. See
:ref:`offset_for_better_meshing`.
