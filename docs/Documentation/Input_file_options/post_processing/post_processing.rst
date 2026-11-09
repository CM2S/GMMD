Post-Processing
===========================

The post-processing analyses are given in the ``analyses`` section of the
input file, each one under its name with its options. An analysis is carried
out when it is given; ``true`` stands for the analysis with its options at
their defaults:

.. code-block:: yaml

   analyses:
     final_config: true
     voronoi: {type: set, plot: true}

.. toctree::
   :maxdepth: 3

   general_post_processing_options
   md_post_processing_options
   rsa_post_processing_options
   statistical_analysis
