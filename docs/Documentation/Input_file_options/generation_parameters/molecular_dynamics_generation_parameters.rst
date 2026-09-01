.. Molecular Dynamics Generation Parameters


Maximum Allowable Overlap Area/Volume per Particle (M)
----------------------------------------------------------

.. note::
   Mandatory only in case of using a Molecular Dynamics simulation.

**Meaning:** The maximum allowable overlap area/volume per particle, i.e.
the total overlap area/volume divided by the number of particles, such that
a given configuration is considered legal.

**Syntax:**

.. code-block:: text

   Max_Residue_Per_Particle x

- ``x``: float

Type of Initial Configuration (O)
--------------------------------------

**Meaning:** Type of initial configuration for the center of mass of the
particles in the simulation box.

**Syntax:**

.. code-block:: text

   Type_Initial_Configuration x

- ``x``: ``{'random', 'grid'}`` (default: ``'random'``)

  - ``'random'`` - the initial configuration is obtained through a Poisson
    point process, i.e. each component of the center of mass's position
    vector is randomly uniformly distributed along the corresponding
    simulation box side.
  - ``'grid'`` - the centers of mass of the particles are placed in a grid,
    with side ``ceil(sqrt(N))``, with any eventual gaps distributed
    randomly across the grid.

Final Naive Overlap Check (O)
-----------------------------------

**Meaning:** Check the final overlap considering every pair of particles.

**Syntax:**

.. code-block:: text

   final_overlap_check x

- ``x``: float

Adaptive Time Step (O)
----------------------------

**Meaning:** Flag for the use of the adaptive time step scheme.

**Syntax:**

.. code-block:: text

   dt_adapt x

- ``x``: bool

Thermostat
--------------------------

Thermostat Used During the Simulation (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Thermostat used during the simulation.

**Syntax:**

.. code-block:: text

   Thermostat x

- ``x``: ``{'isokinetic', 'multi_temperature'}`` (default:
  ``'multi_temperature'``)

  - ``'isokinetic'`` - the system of particles is simulated at a constant
    temperature enforced through the isokinetic scheme.
  - ``'multi_temperature'`` - the system of particles is simulated at
    decreasing temperature stages enforced through the isokinetic scheme.
    The temperature is lowered when equilibrium has been reached.

Lowering Temperature Criterion (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Temperature lowering criterion.

**Syntax:**

.. code-block:: text

   Lowering_Temp_Criterion x

- ``x``: ``{"original", "rolling_ave", "ratio_in_out"}``

  - ``"original"`` - stays a given specified number of iterations at the
    current temperature before lowering it.
  - ``"rolling_ave"`` - a rolling average of the residual overlap is used to
    decide if it has stabilized. If so, the temperature is lowered.
  - ``"ratio_in_out"`` - the increase and decrease in overlap, when taken
    particle pair by particle pair, is used to decide if the temperature
    can be lowered.

Temperature Lowering Ratio (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** When using the multi-temperature isokinetic thermostat, the
ratio used to compute the next temperature from the current temperature.

**Syntax:**

.. code-block:: text

   temp_low_ratio x

- ``x``: float

Number of Overlap Oscillations (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Number of sign changes of the overlap ratio
(increase/decrease) accepted as signaling that thermal equilibrium has been
reached.

**Syntax:**

.. code-block:: text

   Max_Ratio_Osc x

- ``x``: int

Size of the Window Used for the Rolling Average (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Size of the window used to compute the rolling average in the
corresponding lowering temperature criterion.

**Syntax:**

.. code-block:: text

   Average_Window x

- ``x``: int

Initial Temperature of the System of Particles (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Initial temperature of the system of particles, as understood
from the equipartition theorem from statistical mechanics. It is not in
kelvin.

**Syntax:**

.. code-block:: text

   Initial_Temp x

- ``x``: float

Initial Temperature Coefficient (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Coefficient used to set the initial temperature through the
connection with the kinetic energy expressed in the equipartition theorem.

**Syntax:**

.. code-block:: text

   final_overlap_check x

- ``x``: bool

.. note::
   This keyword and type are reproduced as specified in the reference input
   file; they appear inconsistent with the option's name and may be an
   error carried over from the source documentation.

Berendsen Coefficient (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Value of the coefficient used in the Berendsen thermostat.

**Syntax:**

.. code-block:: text

   Berendsen_Coeff x

- ``x``: float

Minimum Number of Iterations at a Temperature (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Minimum number of iterations spent at each temperature stage
before the temperature can be lowered.

**Syntax:**

.. code-block:: text

   Min_Eq_Steps_At_Temp x

- ``x``: int (default: ``25``)

Deep Cuts
--------------------------

Number of Iterations for Relaxation (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** The number of iterations allowed for the system to still move
after reaching a legal configuration.

**Syntax:**

.. code-block:: text

   Max_Steps_To_Relax x

- ``x``: integer (default: ``0``)

Integration Time Step (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Integration time step for the numerical integration of the
equations of motion in the molecular dynamics simulation.

**Syntax:**

.. code-block:: text

   dt x

- ``x``: float (default: ``0.05``)

Seed for Initial Configuration (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Use the given value to seed the initial configuration.

**Syntax:**

.. code-block:: text

   fixed_seed x

- ``x``: int

Force Rescale (O)
~~~~~~~~~~~~~~~~~~~~

**Meaning:** Flag for the use of a force rescaling procedure, which
approximates the scheme described in Salnikov et al. (2015).

**Syntax:**

.. code-block:: text

   Force_Rescale x

- ``x``: float

Option for the Particle Mass (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Option for the mass of the particles.

**Syntax:**

.. code-block:: text

   Particle_Mass_Opt x

- ``x``: ``{"volume", "radius", "unit"}``

  - ``"volume"`` - mass equal to the volume/area of the particle.
  - ``"radius"`` - mass equal to the radius of the circumscribed
    sphere/disk.
  - ``"unit"`` - unit mass for all particles.

Type of Interaction Forces (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Type of the interaction forces prescribed for intersecting
particles.

**Syntax:**

.. code-block:: text

   Force_Option x

- ``x``: ``{"intersection_length", "intersection_area"}``

  - ``"intersection_length"`` - intersection force proportional to the
    intersection length between the interacting particles.
  - ``"intersection_area"`` - intersection force proportional to the
    intersection area/volume between the interacting particles.

Viscous Damping (O)
~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Viscous damping coefficient.

**Syntax:**

.. code-block:: text

   Damping_Coeff x

- ``x``: float
