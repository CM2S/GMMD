General Options
=====================

Options that define the general nature of the problem to be solved. They are
given at the top level of the input file.

.. note::
   The problem type of the old input file (``Problem_Type``) no longer exists.
   Whether the problem is 2D or 3D follows from the number of RVE dimensions.

Number of Samples (M)
----------------------

**Meaning:** Number of microstructures to be generated, each one a sample of
the same description.

**Syntax:**

.. code-block:: yaml

   samples: n

- ``n``: integer, at least ``1``

RVE Dimensions (M)
--------------------

**Meaning:** Dimensions of the quadrilateral (2D) or parallelepipedic (3D)
RVE. Two dimensions make a 2D problem and three a 3D problem.

**Syntax:**

.. code-block:: yaml

   rve_dimensions: [x1, x2]        # 2D
   rve_dimensions: [x1, x2, x3]    # 3D

- ``xi``: float, above ``0``

Seed (O)
---------

**Meaning:** Seed of the first sample, each sample after it taking the next
one, so that a run gives the same microstructures every time. Without a
seed, every run gives different microstructures.

**Syntax:**

.. code-block:: yaml

   seed: x

- ``x``: integer (default: none)
