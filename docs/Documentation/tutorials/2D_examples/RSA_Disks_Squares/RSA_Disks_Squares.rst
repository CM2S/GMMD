===============================================
RSA Disks and Squares with statistical analysis
===============================================

Input file
==========

The file for this tutorial is located in the ``geommicgen/resources/examples/2D_examples`` directory and is named ``RSA_Disks_Squares.mgsim``. The input file can be run by executing the following command in a terminal console window:

.. code-block:: console

    python3 -m geommicgen '../geommicgen/resources/examples/2D_examples/RSA_Disks_Squares.mgsim'

Replacing with the path to the input file according to the location of the GMMD repository in your computer.

The full text of the input file is:

.. literalinclude:: ../../../../../geommicgen/resources/examples/2D_examples/RSA_Disks_Squares.mgsim
    :language: xml


Phases
======

The microstructure has three phases:

- a matrix phase, described by::

    Phase 0
    Phase_Type 1

- a disk phase, described by::

    Phase 1
    Phase_Type 2
    r 0.05
    vf 0.2

- and a square phase, described by::

    Phase 2
    Phase_Type 8
    vf 0.2
    side_distribution normal
        side_mean 0.1
        side_sigma 0.01

The disks have a fixed radius of 0.05 and a volume fraction of 0.2. The squares have a volume fraction of 0.2 and their side follows a normal distribution with mean 0.1 and standard deviation 0.01.

In this example, the RVE has dimensions [1.9, 1.0].


Generation Method
=================

This example uses a Random Sequential Adsorption simulation to generate the microstructure. The speed up scheme used is Cell and a minimum distance of 0.005 between particles is imposed.
The maximum number of iterations, 10000, is significantly larger than the number used for MD simulations in the previous tutorials. This is beacuse RSA simulations take more, but faster, iterations than MD simulations.


Output files
============
After running the command, a folder named ``RSA_Disks_Squares`` will be created in the same directory as the input file. This folder contains all the output data related to the microstructure generation.
A .pdf file of the final microstructure is created with the line ``final_config True`` in the input file. Moreover, some statistical analysis tools are used: Ripley's K function, the nearest neighbor distance distribution and the two-point correlation function of the microstructure are computed and plotted.


.. list-table::
    :widths: 50 50

    * - .. figure:: final_config.svg
            :width: 450px
            :alt: Final microstructure configuration

            Final configuration of the microstructure.

      - .. figure:: k_ripleys_func.svg
            :width: 450px
            :alt: Ripley's K-function

            Ripley's K-function of the microstructure.

    * - .. figure:: nearest_neighbor_dist.svg
            :width: 450px
            :alt: Nearest neighbor distance distribution

            Nearest neighbor distance distribution of the microstructure.

      - .. figure:: two_pt_corr.svg
            :width: 450px
            :alt: Two-point correlation function

            Two-point correlation function of the microstructure.
