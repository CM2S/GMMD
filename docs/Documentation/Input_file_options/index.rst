Input file options
===================

In order to generate the microstructure using GMMD, the user must create a plain-text input file with the ``.mgsim`` extension and then run ``python3 -m geommicgen 'file_name.mgsim'`` in the terminal.

The input file must contain a set of parameters to describe the microstructure, how it will be generated and more.
This documentation describes all parameters (either mandatory or optional) and all possible parameter specifications.
Mandatory parameters are denoted **(M)** and optional ones **(O)**. When a
default value exists for an optional parameter, it is stated in the
corresponding ``Syntax`` block.


.. toctree::
   :maxdepth: 2

   problem_type
   microstructure_definition
   generation_parameters/generation_parameters
   post_processing/post_processing
   mesh_options
   data_driven
