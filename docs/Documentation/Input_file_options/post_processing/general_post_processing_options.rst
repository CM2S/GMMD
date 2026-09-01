General post-processing options
----------------------------------

Final Configuration (O)
~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Plot the final configuration of the microstructure.

**Syntax:**

.. code-block:: text

   final_config x

- ``x``: bool


Simulation GIF (O)
~~~~~~~~~~~~~~~~~~~~

**Meaning:** If ``True``, creates a GIF of the simulation.

**Syntax:**

.. code-block:: text

   sim_gif x

- ``x``: bool (default: ``False``)

.. note::
   In case a minimum distance between particles is imposed, dilating and
   contracting of the particles is not included in the GIF, and the
   animation is made for the virtual particle sizes.

Simulation GIF Frame Duration (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Duration of each frame in milliseconds.

**Syntax:**

.. code-block:: text

   sim_gif_frame_duration x

- ``x``: int (default: ``200``)

Simulation GIF Cleanup Frames (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** If ``True``, deletes frame PNG files used to generate the GIF.

**Syntax:**

.. code-block:: text

   sim_gif_cleanup_frames x

- ``x``: bool (default: ``True``)
