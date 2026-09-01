Statistical Analysis
------------------------------------

Ripley's K Statistical Descriptor (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Compute and plot the Ripley's K function of the microstructure.

**Syntax:**

.. code-block:: text

   stat_ripleys_k x

- ``x``: bool

Nearest Neighbor Statistical Descriptor (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Compute and plot the nearest neighbor function of the
microstructure.

**Syntax:**

.. code-block:: text

   stat_nearest_neighbor x

- ``x``: bool

Two Point Correlation Statistical Descriptor (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Compute and plot the two point correlation function of the
microstructure.

**Syntax:**

.. code-block:: text

   stat_two_pt_corr x

- ``x``: bool

Voronoi Analysis (O)
~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Do a Voronoi analysis.

**Syntax:**

.. code-block:: text

   voronoi_analysis x

- ``x``: bool

Voronoi Type (O)
~~~~~~~~~~~~~~~~~~

**Meaning:** Voronoi type to be used in the Voronoi analysis.

**Syntax:**

.. code-block:: text

   voronoi_type x

- ``x``: ``{"standard", "set"}`` (default: ``"standard"``)

Number of Points for Set Voronoi (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Number of points on the surface used to construct the set
Voronoi diagram.

**Syntax:**

.. code-block:: text

   n_surf_points x

- ``x``: int (default: ``10``)

Plot the Voronoi Diagrams (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Flag used to plot the Voronoi diagrams.

**Syntax:**

.. code-block:: text

   plot_voronoi x

- ``x``: bool

Plot the Voronoi Diagram with the IMTs (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Plot the Voronoi diagram including the values of the
Irreducible Minkowski Tensors corresponding to each cell.

**Syntax:**

.. code-block:: text

   plot_imts x

- ``x``: booll