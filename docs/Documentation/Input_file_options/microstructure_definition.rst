Microstructure Definition
=====================================

Phases (M)
-----------

**Meaning:** Description of the microstructure in terms of the shape of the
particles of each phase and the statistical description of their geometrical
parameters.

**Syntax:**

.. code-block:: yaml

   phases:
     phase_id:
       type: x
       phase_descriptor_1: y
       phase_descriptor_2: y
     phase_id:
       type: x
       phase_descriptor_1: y
       phase_descriptor_2: y
       phase_descriptor_3: y

A phase can also be written in a single line:

.. code-block:: yaml

   phases:
     0: {type: matrix}
     1: {type: disk, r: 0.05, vf: 0.3}

- ``phase_id``: integer. The name of the phase, which the meshes label the
  phase with. Exactly one phase must be the matrix.
- ``x``: str. Shape of the particles contained in the phase being specified:

  .. list-table::
     :header-rows: 1

     * - Value of ``x``
       - Shape
     * - ``matrix``
       - Matrix (2D/3D). It takes no descriptors.
     * - ``disk``
       - Disks (2D)
     * - ``ellipse``
       - Ellipses (2D)
     * - ``sphere``
       - Spheres (3D)
     * - ``ellipsoid``
       - Ellipsoids (3D)
     * - ``cylindrical_fiber``
       - Long cylindrical fibers (3D), as long as the RVE
     * - ``cylinder``
       - Cylinder (3D)
     * - ``square``
       - Square (2D), its sides along the axes

  .. note::
     2D and 3D shapes are incompatible, and long cylindrical fibers
     (``cylindrical_fiber``) are incompatible with all other particle shapes
     except ``matrix``. All the fibers run in one direction.

- ``y``: type depends on the corresponding ``phase_descriptor_Y``.
- ``phase_descriptor_Y``: str. Name of a descriptor for the phase being
  specified. The parameters available for microstructure generation depend on
  the shape of the particle, as detailed below. Each shape accepts only
  certain sets of descriptors; a phase must give exactly one of them.

Phase descriptors per particle shape
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every shape takes the following two descriptors, which are always fixed
values:

.. list-table::
   :header-rows: 1

   * - Descriptor
     - Type
     - Meaning
   * - ``n``
     - integer
     - Number of particles
   * - ``vf``
     - float, between ``0`` and ``1``
     - Volume fraction (decimal)

Disk (``disk``)
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Descriptor
     - Type
     - Meaning
   * - ``r``
     - float
     - Radius of the disk
   * - ``area``
     - float
     - Area of the disk

Described by one of: ``r, n``; ``r, vf``; ``n, vf``; ``area, vf``;
``area, n``.

Ellipse (``ellipse``)
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Descriptor
     - Type
     - Meaning
   * - ``major_axis``
     - float
     - Major principal axis
   * - ``minor_axis``
     - float
     - Minor principal axis
   * - ``ratio``
     - float
     - Ratio between the major and minor axis
   * - ``angle``
     - float
     - Angle between the major semi-axis and the positive x semi-axis

Described by one of:

- ``major_axis, minor_axis, angle, n``
- ``major_axis, minor_axis, angle, vf``
- ``major_axis, angle, n, vf``
- ``minor_axis, angle, n, vf``
- ``angle, ratio, n, vf``
- ``major_axis, angle, ratio, vf``
- ``major_axis, angle, ratio, n``

Sphere (``sphere``)
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Descriptor
     - Type
     - Meaning
   * - ``r``
     - float
     - Radius of the sphere
   * - ``volume``
     - float
     - Volume of the sphere

Described by one of: ``r, n``; ``r, vf``; ``n, vf``; ``volume, vf``;
``volume, n``.

Ellipsoid (``ellipsoid``)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Descriptor
     - Type
     - Meaning
   * - ``axis_1``
     - float
     - Principal axis along xx before applying the rotation
   * - ``semi_axis_1``
     - float
     - Half of ``axis_1``
   * - ``axis_2``
     - float
     - Principal axis along yy before applying the rotation
   * - ``axis_3``
     - float
     - Principal axis along zz before applying the rotation
   * - ``rot_axis_comp_x``
     - float
     - Component x of the vector parallel to the rotation axis
   * - ``rot_axis_comp_y``
     - float
     - Component y of the vector parallel to the rotation axis
   * - ``rot_axis_comp_z``
     - float
     - Component z of the vector parallel to the rotation axis
   * - ``angle``
     - float
     - Rotation angle around the rotation axis
   * - ``ratio_12``
     - float
     - Ratio between ``axis_1`` and ``axis_2``
   * - ``ratio_21``
     - float
     - Ratio between ``axis_2`` and ``axis_1``
   * - ``ratio_13``
     - float
     - Ratio between ``axis_1`` and ``axis_3``
   * - ``ratio_32``
     - float
     - Ratio between ``axis_3`` and ``axis_2``
   * - ``ratio_321``
     - float
     - Ratio of ``axis_3`` and ``axis_2`` to ``axis_1``, the same for both
   * - ``p_3``
     - float
     - Angle between ``axis_1`` and the xy plane
   * - ``phi_z``
     - float
     - Angle between the projection of ``axis_1`` on the xy plane and the y
       axis

Described by one of:

- ``axis_1, axis_2, axis_3, rot_axis_comp_x, rot_axis_comp_y, rot_axis_comp_z, angle, n``
- ``axis_1, axis_2, axis_3, rot_axis_comp_x, rot_axis_comp_y, rot_axis_comp_z, angle, vf``
- ``axis_1, rot_axis_comp_x, rot_axis_comp_y, rot_axis_comp_z, angle, ratio_12, ratio_13, vf``
- ``rot_axis_comp_x, rot_axis_comp_y, rot_axis_comp_z, angle, ratio_12, ratio_13, n, vf``
- ``semi_axis_1, ratio_321, p_3, phi_z, vf``
- ``semi_axis_1, ratio_21, ratio_32, p_3, phi_z, vf``
- ``axis_2, ratio_21, ratio_32, p_3, phi_z, vf``
- ``axis_1, ratio_321, p_3, phi_z, vf``
- ``axis_1, axis_2, axis_3, p_3, phi_z, vf``

Long Cylindrical Fiber (``cylindrical_fiber``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Descriptor
     - Type
     - Meaning
   * - ``r``
     - float
     - Radius of the fiber
   * - ``area``
     - float
     - Area of the section of the fiber
   * - ``direction``
     - ``{0, 1, 2}``
     - Direction of the fibers

Described by one of: ``r, direction, n``; ``r, direction, vf``;
``direction, n, vf``; ``area, direction, vf``; ``area, direction, n``.

Cylinder (``cylinder``)
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Descriptor
     - Type
     - Meaning
   * - ``r_cyl``
     - float
     - Radius of the cylinder
   * - ``length``
     - float
     - Length of the cylinder
   * - ``azimuth_angle``
     - float
     - Azimuth angle of the axis of the cylinder
   * - ``polar_angle``
     - float
     - Polar angle of the axis of the cylinder
   * - ``ratio``
     - float
     - Ratio between the length and the radius

Described by one of:

- ``r_cyl, length, azimuth_angle, polar_angle, n``
- ``r_cyl, length, azimuth_angle, polar_angle, vf``
- ``r_cyl, azimuth_angle, polar_angle, ratio, n``
- ``length, azimuth_angle, polar_angle, ratio, vf``
- ``length, azimuth_angle, polar_angle, n, vf``

Square (``square``)
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Descriptor
     - Type
     - Meaning
   * - ``side``
     - float
     - Side of the square
   * - ``area``
     - float
     - Area of the square

Described by one of: ``side, n``; ``side, vf``; ``area, n``; ``area, vf``;
``n, vf``.


Statistical distributions for descriptors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any descriptor, except for ``n`` and ``vf``, may be specified as a fixed
value or made to vary according to a specific statistical distribution, from
which the value of each particle is drawn. For a given parameter
``parameter``:

Fixed distribution
^^^^^^^^^^^^^^^^^^^

The parameter is fixed, the same for every particle. Simply specify its
value:

.. code-block:: yaml

   parameter: 1.0e-5

or, written out:

.. code-block:: yaml

   parameter: {distribution: fixed, value: 1.0e-5}

Specified values
^^^^^^^^^^^^^^^^^

One value per particle, in the order given, as many values as ``n``:

.. code-block:: yaml

   parameter: {distribution: specified, values: [0.04, 0.05, 0.06]}

Discrete distribution
^^^^^^^^^^^^^^^^^^^^^^

The parameter follows a discrete distribution, where it may only take the
specified ``values`` with the corresponding ``probabilities``:

.. code-block:: yaml

   parameter: {distribution: discrete, values: [0.5, 0.3, 0.8], probabilities: [0.2, 0.5, 0.3]}

.. note::
   The sum of the probabilities of all values must equal 1.

Uniform distribution
^^^^^^^^^^^^^^^^^^^^^

The parameter follows a uniform distribution, where the probability density
is constant for a given interval of values, ``[low, high]``. ``low`` must not
be above ``high``:

.. code-block:: yaml

   parameter: {distribution: uniform, low: 0.1, high: 0.2}

Normal distribution
^^^^^^^^^^^^^^^^^^^^

The parameter follows a normal distribution with mean ``mean`` and standard
deviation ``sigma``. ``sigma`` must not be below zero:

.. code-block:: yaml

   parameter: {distribution: normal, mean: 0.5, sigma: 0.02}

Log-normal distribution
^^^^^^^^^^^^^^^^^^^^^^^^

The logarithm of the parameter follows a normal distribution with standard
deviation ``sigma``; ``mean`` is the median of the parameter. Both must be
above zero:

.. code-block:: yaml

   parameter: {distribution: lognormal, mean: 0.5, sigma: 0.02}

von Mises distribution
^^^^^^^^^^^^^^^^^^^^^^^

For an angle. The parameter follows a von Mises distribution with
concentration ``kappa``, location ``loc`` and scale ``scale``; the larger
``kappa``, the narrower the distribution around ``loc``. ``kappa`` and
``scale`` must be above zero:

.. code-block:: yaml

   parameter: {distribution: vonmises, kappa: 4, loc: 0.5, scale: 0.02}

.. note::
   A fixed value is held to the bounds of its parameter when the file is read;
   a value drawn from a distribution is held to them as each particle is drawn.
   Particular choices of descriptor values for a given phase may lead to
   impossible (e.g. volume fraction above 100%) or very costly to obtain
   (e.g. ellipses with very high ratios) microstructures.


Inner and outer phases (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Place the particles of a phase (the **inner phase**) inside the
particles of another phase (the **outer phase**).

**Syntax:**

.. code-block:: yaml

   phases:
     phase_id:
       type: x
       inside: outer_phase_id
       phase_descriptor_1: y
       phase_descriptor_2: y

- ``inside``: integer (default: none, inside no phase). The ``phase_id`` of
  the outer phase, i.e. the phase whose particles will contain the particles
  of the inner phase.

The remaining descriptors are specified as for any other phase, according to
the shape of the particles (see `Phase descriptors per particle shape`_).

The microstructure is generated in two stages. First, all phases except the inner phases are generated with the specified generation method (see :doc:`generation_parameters/generation_parameters`). Then, the inner phase particles are placed inside the outer phase particles using RSA. Each inner particle is assigned to an outer particle chosen at random, with a probability proportional to the size of the outer particle, as long as it does not fill more than half of the space still available in that particle. The maximum number of steps of this placement is set by ``inner_phase_max_step`` (see :doc:`generation_parameters/generation_parameters`).

.. note::
   - The volume fraction ``vf`` of the inner phase is relative to the whole
     RVE, not to the outer phase. It must be smaller than the volume
     fraction of the outer phase.
   - The minimum distance between particles (``min_distance``) is not
     imposed on the inner particles.
   - The motion analysis, the Voronoi diagram and the statistical analysis
     describe the particles generated in the first stage only: the inner
     particles are left out of them. They are drawn in the final
     configuration.

For an example, see :doc:`/tutorials/MD_Inner_Outer_Disks/MD_Inner_Outer_Disks`.
