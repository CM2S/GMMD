Generation Parameters
================================

General generation parameters
--------------------------------

GMMD generates microstructure geometry depending on the user input, such as particle (or void) shape, volume fraction and other descriptors. The generation of the microstructure is not based on the physical process from which it arose, it is purely geometric. GMMD can currently use one of two methods for generating the RVE: molecular dynamics (MD) or random sequential addition (RSA). In this page, it is laid out the general parameters that are common to both methods. The specific parameters for each method are described in the corresponding pages.

Microstructure Generation Method (M)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Microstructure generation method to be used.


**Syntax:**

.. code-block:: text

   Mic_Gen_Method x

- ``x``: ``{'MD', 'RSA'}``

  - ``'MD'`` - a Molecular Dynamics simulation is used.
  - ``'RSA'`` - a Random Sequential Adsorption simulation is used.

.. note::
   Generally, RSA is faster for smaller volume fractions and MD is faster
   for larger volume fractions.

Maximum Number of Iterations (M)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** The maximum number of iterations for a given microstructure
generation method.

**Syntax:**

.. code-block:: text

   Max_Step x

- ``x``: integer

Minimum Distance Between Particles (O)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Minimum distance between particles.

**Syntax:**

.. code-block:: text

   Min_Distance x

- ``x``: float (default: ``0``)

Speed-Up Scheme (O)
~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Speed-up scheme for particle interaction, avoiding computing
interactions between particles that can be shown a priori not to interact.

**Syntax:**

.. code-block:: text

   Speed_Up_Scheme x
       parameter_scheme_1 y
       parameter_scheme_2 y

- ``x``: ``{'Naive', 'Cell', 'Verlet'}`` (default: ``'Cell'``)

  - ``'Naive'`` - no scheme is used. It doesn't need any extra parameters.

    - For MD simulations, interaction checks are computed for every pair of
      particles: O(N\ :sup:`2`\ ), where N is the number of particles.
    - For RSA simulations, intersection checks for the new particle are done
      against every other particle: O(N), where N is the number of
      particles.

    .. code-block:: text

       Speed_Up_Scheme Naive

  - ``'Cell'`` - the cell list scheme is used. It doesn't need any extra
    parameters.

    - For MD simulations: O(N), where N is the number of particles.
    - For RSA simulations: O(1).

    .. code-block:: text

       Speed_Up_Scheme Cell

  - ``'Verlet'`` - the Verlet list method is used, computing each Verlet
    list from a cell list. Only applicable to MD simulations. An extra
    parameter must be specified:

    .. code-block:: text

       Speed_Up_Scheme Verlet
       Verlet_Factor y

- ``parameter_scheme_Y``: str. The mandatory and optional parameters to be
  specified depend on the speed-up scheme chosen.

  - ``'Naive'`` and ``'Cell'`` - no extra parameters needed.
  - ``'Verlet'`` - requires:

    - ``Verlet_Factor``: the Verlet neighborhood has the shape of the
      particle increased by the specified factor. Larger factors imply
      fewer computations of the Verlet list, but larger Verlet lists for
      each particle; a smaller factor has the opposite effect. To achieve
      an improvement in the CPU time of the molecular dynamics simulation,
      a compromise must be struck between these two tendencies. (``y``:
      float, default: ``1.5``)

- ``y``: depends on the corresponding ``parameter_scheme_Y``.


Molecular Dynamics Generation Parameters
------------------------------------------------------

.. See :doc:`molecular_dynamics_generation_parameters` for the parameters
.. specific to microstructure generation using a Molecular Dynamics
.. simulation, including the thermostat and deep cuts options.

.. toctree::
   :maxdepth: 3

   molecular_dynamics_generation_parameters


Random Sequential Addition Generation Parameters
------------------------------------------------------
No extra parameters beyond the general ones are needed for microstructure generation using a Random Sequential Addition simulation