General post-processing options
----------------------------------

Final Configuration (O)
~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Plot the final configuration of the microstructure.

**Syntax:**

.. code-block:: yaml

   analyses:
     final_config: x

- ``x``: bool (default: not carried out)


Simulation GIF (O)
~~~~~~~~~~~~~~~~~~~~

**Meaning:** Put the frames the generation draws at every step together into
a GIF of the simulation, in 2D. Asking for it has the generation draw the
frames, which slows it down.

**Syntax:**

.. code-block:: yaml

   analyses:
     simulation_gif: {frame_duration: x, cleanup_frames: y}

or, with its options at their defaults:

.. code-block:: yaml

   analyses:
     simulation_gif: true

- ``frame_duration``: duration of each frame in milliseconds. ``x``: int
  (default: ``200``)
- ``cleanup_frames``: if ``true``, deletes the frame PNG files used to
  generate the GIF. ``y``: bool (default: ``true``)

.. note::
   In case a minimum distance between particles is imposed, dilating and
   contracting of the particles is not included in the GIF, and the
   animation is made for the virtual particle sizes.
