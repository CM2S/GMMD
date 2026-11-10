====================
MD Inner Outer Disks
====================

Input file
==========

The file for this tutorial is located in the ``examples/`` directory and is named ``MD_Inner_Outer_Disks.yaml``. To run this example, open a terminal inside the ``GMMD`` repository and run the following command:

.. code-block:: console

    geommicgen examples/MD_Inner_Outer_Disks.yaml

The full text of the input file is:

.. literalinclude:: ../../../examples/MD_Inner_Outer_Disks.yaml
    :language: yaml


Phases
======

This example shows how to place the particles of one phase inside the particles of another phase. The phase that contains the particles is called the **outer phase** and the phase placed inside it is called the **inner phase**.

The microstructure has three phases:

- a matrix phase, described by:

  .. code-block:: yaml

     0: {type: matrix}

- an outer phase of disks, described by:

  .. code-block:: yaml

     1: {type: disk, r: 0.08, vf: 0.4}

- and an inner phase of smaller disks, described by:

  .. code-block:: yaml

     2: {type: disk, r: 0.015, vf: 0.04, inside: 1}

The outer disks have a radius of 0.08 and a volume fraction of 0.4. The inner disks have a radius of 0.015 and a volume fraction of 0.04.

The key ``inside: 1`` makes phase 2 an inner phase, placed inside the particles of phase 1.

The volume fraction of the inner phase is relative to the whole RVE, not to the outer phase. It must therefore be smaller than the volume fraction of the outer phase.


Generation Method
=================

This example uses a molecular dynamics simulation to generate the microstructure. The maximum number of iterations is set to 100, a minimum distance of 0.005 between particles is imposed and the speed up scheme used is Verlet.

The generation is done in two steps:

1. The molecular dynamics simulation is run with the outer phase only. The inner phase does not take part in it.
2. After the simulation, the inner particles are placed inside the outer particles using RSA.

The number of positions tried for each inner particle can be set with ``inner_phase_max_step`` under ``generation``. It is not specified in this example, so the default of 200 is used. See :ref:`input_file_options` for more details.


Output files
============

After running the command, a folder named ``MD_Inner_Outer_Disks`` will be created in the same directory as the input file. This folder contains all the output data related to the microstructure generation.
A .pdf file of the final microstructure is created with ``final_config: true`` under ``analyses`` in the input file.

In this example, 57 inner disks are distributed among 20 outer disks. Each outer disk contains a different number of inner disks, and some may contain none, since the inner disks are assigned to the outer disks at random.

.. figure:: final_config.png
    :width: 500px
    :alt: Final microstructure configuration

    Final configuration of the microstructure. The inner disks (orange) are placed inside the outer disks (blue).


.. include:: ../../_includes/previous_button.inc

.. include:: ../../_includes/next_button.inc
