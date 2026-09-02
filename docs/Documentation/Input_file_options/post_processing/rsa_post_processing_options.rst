.. _rsa_post_processing_options:
Random Sequential Addition post processing options
---------------------------------------------

Plot vf History (O)
~~~~~~~~~~~~~~~~~~~~~

**Meaning:** If ``True``, plots the volume fraction as a function of the step.

**Syntax:**

.. code-block:: text

   plot_rsa_vf_history x

- ``x``: ``{'True', 'False'}`` (default: ``'False'``)

Save History (O)
~~~~~~~~~~~~~~~~~~

**Meaning:** If ``True``, the number of intersection checks and number of particles accepted into the microstructure history is saved.

**Syntax:**

.. code-block:: text

   Save_History x

- ``x``: ``{'True', 'False'}`` (default: ``'False'``)

.. note::
   This option shares its keyword with the ``Save_History`` option described in :ref:`md_motion_analysis`, but has a slightly different meaning.