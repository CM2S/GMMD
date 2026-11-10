Statistical Analysis
------------------------------------

Statistical Descriptors (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Compute and plot statistical descriptors of the arrangement of
the particles.

**Syntax:**

.. code-block:: yaml

   analyses:
     statistics:
       descriptors: [x1, x2]
       seed: y

or, with its options at their defaults:

.. code-block:: yaml

   analyses:
     statistics: true

- ``descriptors``: the descriptors to compute (default:
  ``[nearest_neighbor, ripleys_k, two_pt_corr]``). ``xi``:

  - ``nearest_neighbor`` - the distribution of the distance from each
    particle to its nearest neighbor.
  - ``ripleys_k`` - the Ripley's K function of the microstructure.
  - ``two_pt_corr`` - the two point correlation function of the
    microstructure.

- ``seed``: seed of the random draws used to estimate the Ripley's K function
  and the two point correlation function, so that the analysis gives the same
  values every time. ``y``: int (default: none)

Voronoi Analysis (O)
~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Do a Voronoi analysis: compute the Voronoi diagram of the
particles and the Irreducible Minkowski Tensors of its cells.

**Syntax:**

.. code-block:: yaml

   analyses:
     voronoi:
       type: x
       n_surf_points: y
       plot: z
       plot_imts: w

or, with its options at their defaults:

.. code-block:: yaml

   analyses:
     voronoi: true

Voronoi Type (O)
^^^^^^^^^^^^^^^^^^

**Meaning:** Voronoi type to be used in the Voronoi analysis: ``standard``,
of the centers of the particles, or ``set``, of points on the surfaces of the
particles.

- ``x``: ``{standard, set}`` (default: ``standard``)

Number of Points for Set Voronoi (O)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Meaning:** Number of points on the surface of each particle used to
construct the set Voronoi diagram.

- ``y``: int (default: ``10``)

Plot the Voronoi Diagrams (O)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Meaning:** Flag used to plot the Voronoi diagrams.

- ``z``: bool (default: ``false``)

Plot the Voronoi Diagram with the IMTs (O)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Meaning:** Plot the Voronoi diagram including the values of the
Irreducible Minkowski Tensors corresponding to each cell.

- ``w``: bool (default: ``false``)
