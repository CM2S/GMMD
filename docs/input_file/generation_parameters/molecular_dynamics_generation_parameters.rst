.. Molecular Dynamics Generation Parameters

All the parameters in this page are given in the ``molecular_dynamics``
section of ``generation``.

Maximum Allowable Overlap per Particle (M)
----------------------------------------------------------

**Meaning:** The maximum allowable overlap per particle, i.e. the total
overlap divided by the number of particles, such that a given configuration
is considered legal. The overlap of two particles is the depth by which they
overlap, a length in the units of the RVE.

**Syntax:**

.. code-block:: yaml

   generation:
     molecular_dynamics:
       max_residue_per_particle: x

- ``x``: float, not below ``0``

Maximum Number of Iterations (M)
----------------------------------

**Meaning:** The maximum number of iterations of the simulation.

**Syntax:**

.. code-block:: yaml

   generation:
     molecular_dynamics:
       max_step: x

- ``x``: integer

Type of Initial Configuration (O)
--------------------------------------

**Meaning:** Type of initial configuration for the center of mass of the
particles in the simulation box.

**Syntax:**

.. code-block:: yaml

   generation:
     molecular_dynamics:
       type_initial_configuration: x

- ``x``: ``{random, grid, bcc, fcc}`` (default: ``random``)

  - ``random`` - the initial configuration is obtained through a Poisson
    point process, i.e. each component of the center of mass's position
    vector is randomly uniformly distributed along the corresponding
    simulation box side.
  - ``grid`` - the centers of mass of the particles are placed on the
    smallest square (2D) or cubic (3D) grid with a cell per particle, with
    any eventual gaps distributed randomly across the grid.
  - ``bcc`` - exactly 128 particles, in 3D, on a body centred cubic lattice.
  - ``fcc`` - exactly 256 particles, in 3D, on a face centred cubic lattice.

Initial Velocity Coefficient (O)
----------------------------------

**Meaning:** Speed a particle of the mean radius starts at, as the fraction of
its radius it travels in a step. It also sets the default initial temperature
of the thermostats, through the connection with the kinetic energy expressed
in the equipartition theorem.

**Syntax:**

.. code-block:: yaml

   generation:
     molecular_dynamics:
       initial_vel_coeff: x

- ``x``: float (default: ``0.1``)

Final Naive Overlap Check (O)
-----------------------------------

**Meaning:** Check the final overlap considering every pair of particles.

**Syntax:**

.. code-block:: yaml

   generation:
     molecular_dynamics:
       final_overlap_check: x

- ``x``: bool (default: ``false``)

Adaptive Time Step (O)
----------------------------

**Meaning:** Flag for the use of the adaptive time step scheme, in which the
time step is computed anew at every step.

**Syntax:**

.. code-block:: yaml

   generation:
     molecular_dynamics:
       dt_adapt: x

- ``x``: bool (default: ``true``)

Save the Complete Motion (O)
------------------------------

See :ref:`md_motion_analysis`.

Thermostat
--------------------------

Thermostat Used During the Simulation (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Thermostat used during the simulation, which keeps the
temperature of the particles. The temperature is not in kelvin, but the one
understood from the equipartition theorem from statistical mechanics.

**Syntax:**

.. code-block:: yaml

   generation:
     molecular_dynamics:
       thermostat: x

or, for a thermostat with parameters:

.. code-block:: yaml

   generation:
     molecular_dynamics:
       thermostat:
         name: x
         thermostat_parameter_1: y
         thermostat_parameter_2: y

- ``x``: ``{multi_temperature, isokinetic, micro_canonical, berendsen}``
  (default: ``multi_temperature``). A name alone stands for the thermostat
  with its default parameters.

  - ``multi_temperature`` - the system of particles is simulated at
    decreasing temperature stages enforced through the isokinetic scheme.
    The temperature is lowered when equilibrium has been reached. Takes
    ``initial_temp``, ``criterion`` and ``temp_low_ratio``.
  - ``isokinetic`` - the system of particles is simulated at a constant
    temperature enforced through the isokinetic scheme. Takes
    ``initial_temp``.
  - ``micro_canonical`` - no thermostat is used: the energy is left to the
    damping. It takes no parameters.
  - ``berendsen`` - the temperature is pushed towards a given temperature.
    Takes ``initial_temp`` (mandatory) and ``coeff``.

Examples:

.. code-block:: yaml

   thermostat:
     name: multi_temperature
     criterion: {name: ratio_in_out, max_ratio_osc: 2}
     temp_low_ratio: 0.25

   thermostat: {name: isokinetic, initial_temp: 1.0e+5}

   thermostat: micro_canonical

   thermostat: {name: berendsen, initial_temp: 1.0e+5, coeff: 0.01}

Initial Temperature of the System of Particles (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Initial temperature of the system of particles, as understood
from the equipartition theorem from statistical mechanics. It is not in
kelvin. For the ``multi_temperature`` thermostat, it is the temperature of the
first stage.

**Syntax:**

.. code-block:: yaml

   thermostat: {name: x, initial_temp: y}

- ``x``: ``{multi_temperature, isokinetic, berendsen}``
- ``y``: float, not below ``0`` (default: the temperature at which a particle
  of the mean radius travels ``initial_vel_coeff`` of its radius in a step;
  mandatory for ``berendsen``)

Lowering Temperature Criterion (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Temperature lowering criterion of the ``multi_temperature``
thermostat.

**Syntax:**

.. code-block:: yaml

   thermostat:
     name: multi_temperature
     criterion: {name: x, criterion_parameter: y}

- ``x``: ``{ratio_in_out, rolling_ave, original}`` (default:
  ``ratio_in_out``)

  - ``ratio_in_out`` - the increase and decrease in overlap, when taken
    particle pair by particle pair, is used to decide if the temperature
    can be lowered. Takes ``max_ratio_osc``.
  - ``rolling_ave`` - a rolling average of the residual overlap is used to
    decide if it has stabilized. If so, the temperature is lowered. Takes
    ``average_window``.
  - ``original`` - stays a given specified number of iterations at the
    current temperature before lowering it. Takes ``min_eq_steps_at_temp``.

Number of Overlap Oscillations (O)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Meaning:** Number of sign changes of the overlap ratio
(increase/decrease) accepted as signaling that thermal equilibrium has been
reached.

**Syntax:**

.. code-block:: yaml

   criterion: {name: ratio_in_out, max_ratio_osc: x}

- ``x``: int (default: ``2``)

Size of the Window Used for the Rolling Average (O)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Meaning:** Size of the window used to compute the rolling average in the
corresponding lowering temperature criterion.

**Syntax:**

.. code-block:: yaml

   criterion: {name: rolling_ave, average_window: x}

- ``x``: int (default: ``25``)

Minimum Number of Iterations at a Temperature (O)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Meaning:** Minimum number of iterations spent at each temperature stage
before the temperature can be lowered.

**Syntax:**

.. code-block:: yaml

   criterion: {name: original, min_eq_steps_at_temp: x}

- ``x``: int (default: ``25``)

Temperature Lowering Ratio (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** When using the ``multi_temperature`` thermostat, the ratio used
to compute the next temperature from the current temperature.

**Syntax:**

.. code-block:: yaml

   thermostat: {name: multi_temperature, temp_low_ratio: x}

- ``x``: float (default: ``0.25``)

Berendsen Coefficient (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Value of the coefficient used in the Berendsen thermostat, which
sets how hard the temperature is pushed.

**Syntax:**

.. code-block:: yaml

   thermostat: {name: berendsen, initial_temp: y, coeff: x}

- ``x``: float (default: ``0.01``)

Deep Cuts
--------------------------

Number of Iterations for Relaxation (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** The number of iterations allowed for the system to still move
after reaching a legal configuration.

**Syntax:**

.. code-block:: yaml

   generation:
     molecular_dynamics:
       max_steps_to_relax: x

- ``x``: integer (default: ``0``)

Integration Time Step (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Integration time step for the numerical integration of the
equations of motion in the molecular dynamics simulation.

**Syntax:**

.. code-block:: yaml

   generation:
     molecular_dynamics:
       dt: x

- ``x``: float, above ``0`` (default: ``0.05``)

Seed for Initial Configuration (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The seed is now given for the whole run, at the top level of the input file
(``seed``). See :doc:`../problem_type`.

Option for the Particle Mass (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Option for the mass of the particles.

**Syntax:**

.. code-block:: yaml

   generation:
     molecular_dynamics:
       particle_mass_opt: x

- ``x``: ``{volume, radius, unit}`` (default: ``radius``)

  - ``volume`` - mass equal to the volume/area of the particle.
  - ``radius`` - mass equal to the radius of the circumscribed
    sphere/disk.
  - ``unit`` - unit mass for all particles.

Viscous Damping (O)
~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Viscous damping coefficient.

**Syntax:**

.. code-block:: yaml

   generation:
     molecular_dynamics:
       damping_coeff: x

- ``x``: float (default: ``0``)

.. note::
   The ``Force_Rescale`` and ``Force_Option`` options of the old input file no
   longer exist. The interaction force is always proportional to the
   intersection length between the interacting particles, without rescaling.
