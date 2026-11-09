Generation Parameters
================================

General generation parameters
--------------------------------

GMMD generates microstructure geometry depending on the user input, such as particle (or void) shape, volume fraction and other descriptors. The generation of the microstructure is not based on the physical process from which it arose, it is purely geometric. GMMD can currently use one of two methods for generating the RVE: molecular dynamics (MD) or random sequential addition (RSA). In this page, it is laid out the general parameters that are common to both methods. The specific parameters for each method are described in the corresponding pages.

All the generation parameters are given in the ``generation`` section of the input file. The general parameters are given directly in it, and the parameters of the method under the name of the method:

.. code-block:: yaml

   generation:
     min_distance: 0.005
     molecular_dynamics:
       max_residue_per_particle: 0
       max_step: 100

Microstructure Generation Method (M)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Microstructure generation method to be used. The method is
chosen by giving its section; exactly one method is given.

**Syntax:**

.. code-block:: yaml

   generation:
     x:
       method_parameter_1: y
       method_parameter_2: y

- ``x``: ``{molecular_dynamics, random_sequential_adsorption}``

  - ``molecular_dynamics`` - a Molecular Dynamics simulation is used (see
    `Molecular Dynamics Generation Parameters`_).
  - ``random_sequential_adsorption`` - a Random Sequential Adsorption
    simulation is used (see `Random Sequential Addition Generation
    Parameters`_).

.. note::
   Generally, RSA is faster for smaller volume fractions and MD is faster
   for larger volume fractions.

Minimum Distance Between Particles (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Minimum distance between particles, a length in the units of the
RVE.

**Syntax:**

.. code-block:: yaml

   generation:
     min_distance: x

- ``x``: float, not below ``0`` (default: ``0``)

.. _offset_for_better_meshing:

Offset Use for Better Meshing (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Flag for the use of an offset for better FEM meshing. The origin
of the RVE is moved to where its sides pass furthest from the particles,
which gives a mesh the least distorted elements.

**Syntax:**

.. code-block:: yaml

   generation:
     offset: x

- ``x``: bool (default: ``true``)

Inner Phase Maximum Number of Steps (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** The maximum number of steps for placing each inner phase inside
its outer phase (see :doc:`../microstructure_definition`). Each random
position tried for an inner particle counts as one step. When this number is
reached, the placement stops and the inner particles not yet placed are
removed, so the inner phase may end up with a volume fraction smaller than the
one specified. A message is printed when this happens.

**Syntax:**

.. code-block:: yaml

   generation:
     inner_phase_max_step: x

- ``x``: integer (default: ``200``)

Speed-Up Scheme (O)
~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Speed-up scheme for particle interaction, avoiding computing
interactions between particles that can be shown a priori not to interact.
It is given in the section of the generation method.

**Syntax:**

.. code-block:: yaml

   generation:
     method:
       speed_up: x

or, for a scheme with parameters:

.. code-block:: yaml

   generation:
     method:
       speed_up: {name: x, parameter_scheme_1: y}

- ``x``: ``{naive, cell, verlet}`` (default: ``cell``). A name alone stands
  for the scheme with its default parameters.

  - ``naive`` - no scheme is used. It doesn't need any extra parameters.

    - For MD simulations, interaction checks are computed for every pair of
      particles: O(N\ :sup:`2`\ ), where N is the number of particles.
    - For RSA simulations, intersection checks for the new particle are done
      against every other particle: O(N), where N is the number of
      particles.

    .. code-block:: yaml

       speed_up: naive

  - ``cell`` - the cell list scheme is used. Only the particles in
    neighbouring cells are checked. It doesn't need any extra parameters.

    - For MD simulations: O(N), where N is the number of particles.
    - For RSA simulations: O(1).

    .. code-block:: yaml

       speed_up: cell

  - ``verlet`` - the Verlet list method is used, computing each Verlet
    list from a cell list. Only applicable to MD simulations. An extra
    parameter must be specified:

    .. code-block:: yaml

       speed_up: {name: verlet, factor: y}

- ``parameter_scheme_Y``: str. The mandatory and optional parameters to be
  specified depend on the speed-up scheme chosen.

  - ``naive`` and ``cell`` - no extra parameters needed.
  - ``verlet`` - requires:

    - ``factor``: the Verlet neighborhood has the shape of the
      particle increased by the specified factor. Larger factors imply
      fewer computations of the Verlet list, but larger Verlet lists for
      each particle; a smaller factor has the opposite effect. To achieve
      an improvement in the CPU time of the molecular dynamics simulation,
      a compromise must be struck between these two tendencies. (``y``:
      float, above ``1``. Mandatory.)

- ``y``: depends on the corresponding ``parameter_scheme_Y``.


Molecular Dynamics Generation Parameters
------------------------------------------------------

.. See :doc:`molecular_dynamics_generation_parameters` for the parameters
.. specific to microstructure generation using a Molecular Dynamics
.. simulation, including the thermostat and deep cuts options.

The parameters of the Molecular Dynamics simulation are given in the
``molecular_dynamics`` section of ``generation``:

.. code-block:: yaml

   generation:
     molecular_dynamics:
       max_residue_per_particle: 0
       max_step: 100
       speed_up: {name: verlet, factor: 1.1}

.. toctree::
   :maxdepth: 3

   molecular_dynamics_generation_parameters


Random Sequential Addition Generation Parameters
------------------------------------------------------

The parameters of the Random Sequential Addition simulation are given in the
``random_sequential_adsorption`` section of ``generation``. Particles are
drawn one at a time at random places, and each is kept when it intersects
none of the particles kept before it, until every phase has its number of
particles or its volume fraction. The particles never move, so a dense
packing is out of its reach.

.. code-block:: yaml

   generation:
     random_sequential_adsorption:
       max_step: 10000
       save_history: false
       speed_up: cell

Maximum Number of Iterations (M)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** The maximum number of particles drawn. The simulation fails when
the phases are not filled by then.

**Syntax:**

.. code-block:: yaml

   generation:
     random_sequential_adsorption:
       max_step: x

- ``x``: integer

Save History (O)
~~~~~~~~~~~~~~~~~~

See :ref:`rsa_post_processing_options`.

RSA Speed-Up Scheme (O)
~~~~~~~~~~~~~~~~~~~~~~~~~

As described in `Speed-Up Scheme (O)`_, with ``naive`` or ``cell`` (default:
``cell``). There is no Verlet list for RSA simulations.
