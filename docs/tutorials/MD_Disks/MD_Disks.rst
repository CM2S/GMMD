==============
MD Disks
==============

Input file
==========

The file for this tutorial is located in the ``examples/`` directory and is named ``MD_Disks.yaml``. To run this example, open a termial inside the ``GMMD`` repository and run the following command:

.. code-block:: console

    geommicgen examples/MD_Disks.yaml

The full text of the input file is:

.. literalinclude:: ../../../examples/MD_Disks.yaml
    :language: yaml


Phases
======

The microstructure has two phases:

- a matrix phase, described by:

  .. code-block:: yaml

     0: {type: matrix}

- and a disk phase, described by:

  .. code-block:: yaml

     1:
       type: disk
       vf: 0.5
       r: {distribution: discrete, values: [0.05, 0.02, 0.1], probabilities: [0.4, 0.5, 0.1]}

The disks have a volume fraction of 0.5 and their radius follows a discrete distribution.

Generation Method
=================

This example uses a molecular dynamics simulation to generate the microstructure.
The maximum number of iterations is set to 100 and a minimum distance between particles of 0.005 is imposed.
The speed up scheme used is Verlet.


Mesh
====

The ``meshes`` block asks for one voxel mesh of the microstructure:

.. code-block:: yaml

   meshes:
     voxel: {n_voxels_dims: [128, 128]}

- ``voxel`` divides the RVE into a regular grid of square voxels, and gives each voxel
  the phase its centre lies in. Unlike a finite element mesh, the grid does not follow
  the boundaries of the disks, which come out as staircases. The grid is periodic by
  construction: the voxels on opposite sides of the RVE match.
- ``n_voxels_dims: [128, 128]`` asks for 128 voxels along x and 128 along y, so each
  voxel is 1/128 ≈ 0.0078 wide. The smallest disks, of radius 0.02, are then only about
  5 voxels across. A finer grid follows the disks more closely, at the cost of a larger
  mesh.
- ``formats`` is not given, so the mesh is written in the default format of ``voxel``,
  ``crate``, for the `CRATE <https://github.com/bessagroup/CRATE>`_ solver. How to ask
  for other formats is explained in the :ref:`example_MD_Ellipses` tutorial.

For this example, the mesh has 128 × 128 = 16 384 voxels.


Output files
============

After running the command, a folder named ``MD_Disks`` is created in the same directory
as the input file. It holds:

.. code-block:: text

   MD_Disks/
   ├── MD_Disks.yaml                     copy of the input file
   └── mic_0/                            the first (and here only) sample
       ├── mic.json                      microstructure file
       ├── status                        status of the generation
       ├── md_state.npz                  histories of the simulation
       ├── mic.screen                    log of the run
       ├── final_config.pdf              picture of the microstructure
       ├── Simulation.gif                animation of the simulation
       └── meshes/
           ├── MD_Disks_128_128.vti           the mesh, to view
           ├── MD_Disks_128_128.mesh.json     what the .vti file cannot hold
           ├── MD_Disks_128_128.rgmsh.npy     the grid of phases, for CRATE
           └── MD_Disks_128_128_example.dat   an example CRATE input file

The microstructure
------------------

A .pdf file of the final microstructure is created with ``final_config: true`` under ``analyses`` in the input file and a gif of the molecular dynamics simulation is created. It is important to note that the gif frames are created in each step of the simulation, which increases the simulation time. The default setting is to not create the gif.

As you can note, the final frame of the gif is different from the final configuration image. This is because, after the simulation, it is applied an offset that minimizes particles sitting tangent to the boundary (see :ref:`offset_for_better_meshing`). This ensures that the FEM mesh is the least distorted possible. Besides this, in order to guarentee the minimum distance between particles, GMMD first dilates them, runs the simulation and then contracts the particles. This dilation and contraction steps are also not seen in the gif, eventhough hardly noticeable in this example.

.. list-table:: 
   :widths: 45 55
   :align: center

   * - .. figure:: final_config.png
          :alt: Final microstructure configuration

          Final microstructure configuration
     - .. figure:: Simulation.gif
          :alt: Molecular dynamics simulation animation

          Molecular dynamics simulation animation

The mesh
--------

The files in ``meshes/`` are named after the input file and the number of voxels in each
direction: ``MD_Disks_128_128``.

- ``MD_Disks_128_128.vti`` and ``MD_Disks_128_128.mesh.json`` are the mesh in VTK format
  and the information that goes with it, as in the :ref:`example_MD_Ellipses` tutorial.
  For a voxel mesh, the VTK file is an image (``.vti``) instead of an unstructured grid
  (``.vtu``).
- ``MD_Disks_128_128.rgmsh.npy`` is the grid of phases that CRATE reads: a NumPy array
  of 128 × 128 integers, indexed ``[x, y]``, with 0 for the matrix and 1 for the disks.
  It can also be read in Python with ``numpy.load``.
- ``MD_Disks_128_128_example.dat`` is an example CRATE input file. It points to the
  ``.rgmsh.npy`` file and gives the dimensions of the RVE, which the grid does not hold.
  The materials, the loading and the number of clusters are placeholders, marked
  ``TODO`` in the file, so that the file runs as written. Replace them with the real
  values before using the results.

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: mesh.png
          :alt: Voxel mesh of the microstructure

          Voxel mesh of the microstructure: the disks in blue, the matrix in grey.
     - .. figure:: mesh_zoom.png
          :alt: Close-up of the voxel mesh

          Close-up of the mesh, where the voxels can be seen. The edges of the disks
          follow the grid.


.. include:: ../../_includes/previous_button.inc

.. include:: ../../_includes/next_button.inc
