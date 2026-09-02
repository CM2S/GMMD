.. _md_motion_analysis:
Molecular Dynamics - Motion Analysis
------------------------------------------------------

Save the Complete Motion (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** If ``True``, all the positions of all the particles through the
complete motion are saved; otherwise only the initial and final
configurations are saved.

**Syntax:**

.. code-block:: text

   Save_History x

- ``x``: ``{'True', 'False'}`` (default: ``'False'``)

.. note::
   This option shares its keyword with the ``Save_History`` option described
   in "Random Sequential Addition post processing options", but has a slightly
   different meaning for RSA simulations.

.. note::
   This option shares its keyword with the ``Save_History`` option described in :ref:`rsa_post_processing_options`, but has a slightly different meaning.

Motion Analysis (O)
~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Do motion analysis, including plotting the particle paths,
kinetic energy, total overlap and time step used.

**Syntax:**

.. code-block:: text

   Motion_Analysis x

- ``x``: bool