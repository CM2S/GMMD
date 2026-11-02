Microstructure Definition
=====================================

Microstructure Generation Descriptors (M)
--------------------------------------------

**Meaning:** Description of the microstructure in terms of the shape of the
phases and the statistical description of their geometrical parameters.

**Syntax:**

.. code-block:: text

   Mic_Gen_Descriptors
       Phase phase_id
       Phase_Type x
       phase_descriptor_1 y
       phase_descriptor_2 y
       Phase phase_id
       Phase_Type x
       phase_descriptor_1 y
       phase_descriptor_2 y
       phase_descriptor_3 y

- ``phase_id``: string. The name of the phase.
- ``x``: int. Shape of the particles contained in the phase being specified,
  coded as:

  .. list-table::
     :header-rows: 1

     * - Value of ``x``
       - Shape
     * - ``1``
       - Matrix (2D/3D)
     * - ``2``
       - Disks (2D)
     * - ``3``
       - Ellipses (2D)
     * - ``4``
       - Spheres (3D)
     * - ``5``
       - Ellipsoids (3D)
     * - ``6``
       - Long cylindrical fibers (3D)
     * - ``7``
       - Cylinder (3D)
     * - ``8``
       - Square (2D)

  .. note::
     2D and 3D shapes are incompatible, and long cylindrical fibers (``6``)
     are incompatible with all other particle shapes except matrix (``1``).

- ``y``: type depends on the corresponding ``phase_descriptor_Y``.
- ``phase_descriptor_Y``: str. Name of a descriptor for the phase being
  specified. The parameters available for microstructure generation depend on
  the shape of the particle, as detailed below.

Phase descriptors per particle shape
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Disk (2)
^^^^^^^^

Choose 2 of the following parameters:

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
   * - ``n``
     - integer
     - Number of particles
   * - ``vf``
     - float
     - Volume fraction (decimal)

Ellipse (3)
^^^^^^^^^^^

Choose 4 of the following parameters, always including ``angle``:

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
   * - ``n``
     - integer
     - Number of particles
   * - ``vf``
     - float
     - Volume fraction (decimal)

Sphere (4)
^^^^^^^^^^

Choose 2 of the following parameters:

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
   * - ``n``
     - integer
     - Number of particles
   * - ``vf``
     - float
     - Volume fraction (decimal)

Ellipsoid (5)
^^^^^^^^^^^^^

Choose 8 of the following parameters, always including ``rot_axis_comp_x``,
``rot_axis_comp_y``, ``rot_axis_comp_z`` and ``angle``:

.. list-table::
   :header-rows: 1

   * - Descriptor
     - Type
     - Meaning
   * - ``axis_1``
     - float
     - Principal axis along xx before applying the rotation
   * - ``axis_2``
     - float
     - Principal axis along yy before applying the rotation
   * - ``axis_3``
     - float
     - Principal axis along zz before applying the rotation
   * - ``ratio_12``
     - float
     - Ratio between the principal axis along xx (``axis_1``) and along yy
       (``axis_2``) before rotation and minor axis
   * - ``ratio_13``
     - float
     - Ratio between the principal axis along xx (``axis_1``) and along zz
       (``axis_3``) before rotation and minor axis
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
   * - ``n``
     - integer
     - Number of particles
   * - ``vf``
     - float
     - Volume fraction (decimal)

Long Cylindrical Fiber (6)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Choose 3 of the following parameters, always including ``direction``:

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
   * - ``n``
     - integer
     - Number of particles
   * - ``vf``
     - float
     - Volume fraction (decimal)
   * - ``direction``
     - ``{0, 1, 2}``
     - Direction of the fibers

Cylinder (7)
^^^^^^^^^^^^

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
     - Azimuth angle of the cylinder
   * - ``polar_angle``
     - float
     - Polar angle of the cylinder
   * - ``n``
     - integer
     - Number of particles
   * - ``vf``
     - float
     - Volume fraction (decimal)

Square (8)
^^^^^^^^^^

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
   * - ``n``
     - integer
     - Number of particles
   * - ``vf``
     - float
     - Volume fraction (decimal)


Statistical distributions for descriptors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any descriptor, except for ``n`` and ``vf``, may be specified as a fixed
value or made to vary according to a specific statistical distribution. For
a given parameter ``parameter``:

Fixed distribution
^^^^^^^^^^^^^^^^^^^

The parameter is fixed. Simply specify its value:

.. code-block:: text

   parameter 1e-5

Discrete distribution
^^^^^^^^^^^^^^^^^^^^^^

The parameter follows a discrete distribution, where it may only take the
specified values (``parameter_value_1``, ``parameter_value_2``,
``parameter_value_3``) with the corresponding probability
(``parameter_prob_1``, ``parameter_prob_2``, ``parameter_prob_3``):

.. code-block:: text

   parameter_distribution discrete
       parameter_value_1 0.5
       parameter_prob_1 0.2
       parameter_value_2 0.3
       parameter_prob_2 0.5
       parameter_value_3 0.8
       parameter_prob_3 0.3

.. note::
   The sum of the probabilities of all values must equal 1.

Uniform distribution
^^^^^^^^^^^^^^^^^^^^^

The parameter follows a uniform distribution, where the probability density
is constant for a given interval of values, ``[parameter_low,
parameter_high]``:

.. code-block:: text

   parameter_distribution uniform
   parameter_low 0.1
   parameter_high 0.2

Normal distribution
^^^^^^^^^^^^^^^^^^^^

The parameter follows a normal distribution with mean ``parameter_mean`` and
standard deviation ``parameter_sigma``:

.. code-block:: text

   parameter_distribution normal
   parameter_mean 0.5
   parameter_sigma 0.02

Log-normal distribution
^^^^^^^^^^^^^^^^^^^^^^^^

The parameter follows a log-normal distribution with mean ``parameter_mean``
and standard deviation ``parameter_sigma``:

.. code-block:: text

   parameter_distribution lognormal
   parameter_mean 0.5
   parameter_sigma 0.02

von Mises distribution
^^^^^^^^^^^^^^^^^^^^^^^

The parameter follows a von Mises distribution with concentration
``parameter_kappa``, location ``parameter_loc`` and scale
``parameter_scale``:

.. code-block:: text

   parameter_distribution vonmises
   parameter_kappa 4
   parameter_loc 0.5
   parameter_scale 0.02

.. note::
   Particular choices of descriptor values for a given phase may lead to
   impossible (e.g. volume fraction above 100%) or very costly to obtain
   (e.g. ellipses with very high ratios) microstructures.


Inner and outer phases (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Place the particles of a phase (the **inner phase**) inside the
particles of another phase (the **outer phase**).

**Syntax:**

.. code-block:: text

   Phase phase_id
   Phase_Type x
   inner_phase True
   outer_phase outer_phase_id
   phase_descriptor_1 y
   phase_descriptor_2 y

- ``inner_phase``: bool (default: ``False``). If ``True``, the phase is an
  inner phase.
- ``outer_phase``: integer. The ``phase_id`` of the outer phase, i.e. the
  phase whose particles will contain the particles of the inner phase.
  Mandatory if ``inner_phase True``.

The remaining descriptors are specified as for any other phase, according to
the shape of the particles (see `Phase descriptors per particle shape`_).

The microstructure is generated in two stages. First, all phases except the inner phases are generated with the specified generation method (``Mic_Gen_Method``). Then, the inner phase particles are placed inside the outer phase particles using RSA. Each inner particle is assigned to an outer particle chosen at random, with a probability proportional to the size of the outer particle, as long as it does not fill more than half of the space still available in that particle. The maximum number of steps of this placement is set by ``Inner_Phase_Max_Step`` (see :doc:`generation_parameters/generation_parameters`).

.. note::
   - The volume fraction ``vf`` of the inner phase is relative to the whole
     RVE, not to the outer phase. It must be smaller than the volume
     fraction of the outer phase.
   - The minimum distance between particles (``Min_Distance``) is not
     imposed on the inner particles.

For an example, see :doc:`/tutorials/MD_Inner_Outer_Disks/MD_Inner_Outer_Disks`.
