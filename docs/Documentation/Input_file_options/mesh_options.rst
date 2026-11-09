Mesh Options
========================

Mesh Options
--------------

**Meaning:** Mesh options.

**Syntax:**

.. code-block:: text

   Mesh_Options
       Mesh_Type mesh_type_1
       mesh_option_1 y
       mesh_option_2 y
       Mesh_Type mesh_type_2
       mesh_option_1 y
       mesh_option_2 y

- ``mesh_type_X``: ``{'rgmsh', 'femsh'}``

  - ``'rgmsh'`` - Regular grid mesh. A given voxel is set to the phase to
    which its center point belongs.
  - ``'femsh'`` - Finite element mesh. Generated using gmsh.

- ``mesh_option_Y``: str. The mandatory and optional mesh options depend on
  the mesh type specified.

``'rgmsh'`` - 1 mandatory parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Number of Voxels in Each Direction (M)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Meaning:** Number of voxels discretizing each side of the microstructure.

**Syntax:**

.. code-block:: text

   n_voxels_dims [x1, x2] (2D) / [x1, x2, x3] (3D)

- ``x1``: float
- ``x2``: float
- ``x3``: float

``'femsh'`` - 2 mandatory parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mesh Size (M)
^^^^^^^^^^^^^^

**Meaning:** Largest mesh size allowed. The mesh size is computed from the
curvature of the geometry.

**Syntax:**

.. code-block:: text

   mesh_size x

- ``x``: float

Element Type (M)
^^^^^^^^^^^^^^^^^

**Meaning:** Type of element used to mesh the microstructure.

**Syntax:**

.. code-block:: text

   element_type x

- ``x``: ``{'tri3', 'tri6', 'tetra4', 'tetra10'}``

- ``y``: depends on the corresponding ``mesh_option_Y``.

Offset Use for Better Meshing (O)
------------------------------------

**Meaning:** Flag for the use of an offset for better FEM meshing.

**Syntax:**

.. code-block:: text

   offset x

- ``x``: bool
