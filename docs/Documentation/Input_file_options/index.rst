.. _input_file_options:

Input file options
===================

In order to generate the microstructure using GMMD, the user must create an input file in `YAML <https://yaml.org/>`_ format with the ``.yaml`` extension and then run ``geommicgen 'file_name.yaml'`` in the terminal.

In this input file, # is used for comments.

The input file is organised in sections: a few general options at the top level of the file, followed by the ``phases``, ``generation``, ``meshes`` and ``analyses`` sections. A minimal input file looks like this:

.. code-block:: yaml

   samples: 1
   rve_dimensions: [1, 1]

   phases:
     0: {type: matrix}
     1: {type: disk, r: 0.05, vf: 0.3}

   generation:
     molecular_dynamics:
       max_residue_per_particle: 0
       max_step: 100

   analyses:
     final_config: true

The whole file is checked before anything is generated. A key that is not one of the keys described in this documentation is refused, and so is a key given twice. A file that is refused is reported with every problem found in it, each at its path in the file (for example ``generation.molecular_dynamics.max_step``).

This documentation describes all parameters (either mandatory or optional) and all possible parameter specifications.
Mandatory parameters are denoted **(M)** and optional ones **(O)**. When a
default value exists for an optional parameter, it is stated in the
corresponding ``Syntax`` block.

.. note::
   An input file of the old keyword format (``.mdsim``) can be converted into the new format with ``geommicgen-convert-mdsim 'file_name.mdsim'``.


.. toctree::
   :maxdepth: 2

   problem_type
   microstructure_definition
   generation_parameters/generation_parameters
   post_processing/post_processing
   mesh_options
   data_driven
