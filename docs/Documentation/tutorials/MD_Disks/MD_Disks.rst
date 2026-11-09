==============
Basic tutorial
==============

Input file
==========

The file for this tutorial is located in the ``geommicgen/resources/examples/`` directory and is named ``MD_Disks.mgsim``. To run this example, open a termial inside the ``GMMD`` repository and run the following command:

.. code-block:: console

    python3 -m geommicgen 'geommicgen/resources/examples/MD_Disks.mgsim'

In order for you to run any other file of your making, you just need to run the command with the location of your input file.

The full text of the input file is:

.. literalinclude:: ../../../../geommicgen/resources/examples/MD_Disks.mgsim
    :language: xml


Phases
======

The microstructure has two phases:

- a matrix phase, described by::

    Phase 0
    Phase_Type 1

- and a disk phase, described by::

    Phase 1
    Phase_Type 2
    r_distribution discrete
        r_value_1 0.05
        r_prob_1 0.4
        r_value_2 0.02
        r_prob_2 0.5
        r_value_3 0.1
        r_prob_3 0.1
    vf 0.5

The disks have a volume fraction of 0.5 and their radius follows a discrete distribution.

Generation Method
=================

This example uses a molecular dynamics simulation to generate the microstructure.
In a nutshell, the molecular dynamics simulation starts by placing all the particles randomly in the simulation box. Then, in each step, it computes the repulsion forces between particles based on their overlap area and solves the equations of motion to obtain the particles' velocities. The process continues until a legal configuration is reached, that is, a configuration where there is no overlap.

The maximum number of iterations is set to 100 and a minimum distance between particles of 0.005 is imposed.

The speed up scheme used is Verlet. The speed up scheme is used to reduce the number particle intersection checks, reducing, in general, the elapsed time.



Output files
============
After running the command, a folder named ``MD_Disks`` will be created in the same directory as the input file. This folder contains all the output data related to the microstructure generation.
A .pdf file of the final microstructure is created with the line ``final_config True`` in the input file and a gif of the molecular dynamics simulation is created. It is important to note that the gif frames are created in each step of the simulation, which increases the simulation time. The default setting is to not create the gif.

As you can note, the final frame of the gif is different from the final configuration image. This is because, after the simulation, it is applied an offset that minimizes particles sitting tangent to the boundary. This ensures that the FEM mesh is the least distorted possible. Besides this, in order to guarentee the minimum distance between particles, GMMD first dilates them, runs the simulation and then contracts the particles. This dilation and contraction steps are also not seen in the gif, eventhough hardly noticeable.

.. list-table:: 
   :widths: 45 55
   :align: center

   * - .. figure:: final_config.svg
          :alt: Final microstructure configuration

          Final microstructure configuration
     - .. figure:: Simulation.gif
          :alt: Molecular dynamics simulation animation

          Molecular dynamics simulation animation
