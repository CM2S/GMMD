.. _md_motion_analysis:

Molecular Dynamics - Motion Analysis
------------------------------------------------------

Save the Complete Motion (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** If ``true``, all the positions of all the particles through the
complete motion are saved, for the motion analysis to draw their paths;
otherwise only the initial and final configurations are saved. It is given in
the ``molecular_dynamics`` section of ``generation``.

**Syntax:**

.. code-block:: yaml

   generation:
     molecular_dynamics:
       save_history: x

- ``x``: bool (default: ``false``)


.. note::
   This option shares its name with the ``save_history`` option described in :ref:`rsa_post_processing_options`, but has a slightly different meaning.

Motion Analysis (O)
~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Do motion analysis, plotting what the simulation recorded: the
total overlap, the kinetic energy, the time step used and, with
``save_history``, the particle paths.

**Syntax:**

.. code-block:: yaml

   analyses:
     motion: x

- ``x``: bool (default: not carried out)
