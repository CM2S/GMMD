.. _rsa_post_processing_options:

Random Sequential Addition post processing options
---------------------------------------------------

Plot vf History (O)
~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Plot the volume fraction reached by the random sequential
adsorption as a function of the step. It needs ``save_history``.

**Syntax:**

.. code-block:: yaml

   analyses:
     rsa_history: {}

- default: not carried out

Save History (O)
~~~~~~~~~~~~~~~~~~

**Meaning:** If ``true``, the volume fraction, the number of particles
accepted into the microstructure and the number of intersection checks made
at every step are saved. It is given in the ``random_sequential_adsorption``
section of ``generation``.

**Syntax:**

.. code-block:: yaml

   generation:
     random_sequential_adsorption:
       save_history: x

- ``x``: bool (default: ``false``)

.. note::
   This option shares its name with the ``save_history`` option described in :ref:`md_motion_analysis`, but has a slightly different meaning.
