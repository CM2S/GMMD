Problem Type
=====================

Options that define the general nature of the problem to be solved.

Problem Type (M)
-----------------

**Meaning:** Problem type (2D: plain strain, plain stress or axisymmetric and
3D).

**Syntax:**

.. code-block:: text

   Problem_Type x

.. list-table::
   :header-rows: 1

   * - Value of ``x``
     - Meaning
   * - ``1``
     - 2D problem (plain strain)
   * - ``2``
     - 2D problem (plain stress)
   * - ``3``
     - 2D problem (axisymmetric)
   * - ``4``
     - 3D problem

Number of Samples for the Current Design Point (M)
----------------------------------------------------

**Meaning:** Number of samples to be generated with the specified
microstructural descriptors corresponding to the current design point.

**Syntax:**

.. code-block:: text

   N_DP_Samples n

- ``n``: integer

RVE Dimensions (M)
--------------------

**Meaning:** Dimensions of the quadrilateral (2D) or parallelepipedic (3D)
RVE.

**Syntax:**

.. code-block:: text

   RVE_Dimensions
   [x1, x2] (2D) / [x1, x2, x3] (3D)

- ``xi``: float
