Build a custom Newton solver
============================

This tutorial implements a spring-slider solver with DexSim's Newton extension
interface. All of its code lives in ``scripts/tutorials/sim/simple_newton_solver/``;
the provider is an example, not an EmbodiChain solver or package entry point.
The scene contains two independent prismatic sliders, one in each environment.

Run the example
---------------

Use a DexSim build that provides ``ProviderNewtonSolverCfg``,
``FunctionalDynamics`` and ``Scene.dynamics_capabilities()``. From the repository
root, run against an installed EmbodiChain environment or expose this checkout:

.. code-block:: bash

   export PYTHONPATH="$PWD:$PWD/extensions${PYTHONPATH:+:$PYTHONPATH}"
   python scripts/tutorials/sim/simple_newton_solver/demo.py --device cpu \
       --preview-path simple_newton_solver_preview.png

Use ``--device cuda:0`` to run the same numerical example on CUDA. The script
runs headlessly. It checks native/functional parity and state isolation before
printing JSON. A successful CPU run reports approximately:

.. code-block:: json

   {
     "native_functional_parity": true,
     "published_position": [0.020873, 0.031310],
     "partial_reset_position": [0.0, 0.031310],
     "force_gradient": [0.00012048, 0.00050749],
     "force_scale_gradient": 0.00352685
   }

The optional PNG plots the two slider trajectories. Its dashed final segment
shows that resetting ``arena_0`` returns only the first slider to zero.

For an image rendered by EmbodiChain's native offscreen camera, run:

.. code-block:: bash

   python scripts/tutorials/sim/simple_newton_solver/demo.py --device cpu \
       --render-path simple_newton_offscreen.png

The resulting PNG has environment 0 and 1 in the left and right columns, with
initial and after-30-step views in the top and bottom rows. The script reserves
FastRT before constructing the World, creates a ``CameraCfg`` in each arena,
renders through ``SimulationManager.render_camera_group()``, and reads the RGB
frames from ``Camera.get_data()``. This needs a working native GPU renderer even
when the solver runs on the CPU.

To inspect the actual EmbodiChain scene in Viser, run:

.. code-block:: bash

   python scripts/tutorials/sim/simple_newton_solver/demo.py --device cpu \
       --viser --viser-seconds 60

Open ``http://127.0.0.1:8080`` while the script runs. Viser receives both URDF
slider meshes, then shows 30 native simulation steps. The scene remains open
for the requested number of seconds after replay. Use ``--viser-port PORT`` if
8080 is in use. The ``base`` link has no visual geometry, so the visible objects
are the two cubes. This browser view uses EmbodiChain's
``SimulationManager.capture_visualization()`` and does not require a native
render window.

Follow the data flow
--------------------

1. ``demo.py`` passes the local ``SimpleSolverProvider`` class to
   ``NewtonPhysicsCfg`` through DexSim's ``ProviderNewtonSolverCfg``. The direct
   factory path does not require package entry-point registration.
2. ``SimulationManager.prepare()`` finalizes the scene and asks DexSim to create
   a fresh provider for the model. The provider validates the slider model and
   declares effort-only controls, reset support and no CUDA Graph support.
3. ``_SliderDynamics.step()`` applies effort, spring force and damping, then
   updates velocity and position with semi-implicit Euler. Its Torch operations
   preserve gradients in the functional path.
4. The native ``SolverBase`` adapter calls that same transition under
   ``torch.no_grad()`` and writes the resulting state to the next Newton state.
   ``demo.py`` checks that one native step matches one functional step.
5. Functional stepping with ``publish=False`` leaves the live scene untouched.
   ``publish=True`` publishes a detached physical state and advances the Scene
   clock once. The returned functional tensors retain their gradient graph.
   Reset selects arenas independently.

For a slider coordinate ``q``, velocity ``v``, effort ``u``, trainable scale
``g`` and effective mass ``m``, the unconstrained transition is:

.. code-block:: text

   force = clamp(g * u, -effort_limit, effort_limit)
   acceleration = (force - stiffness * q - damping * v) / m
   v_next = clamp(v + dt * acceleration, -velocity_limit, velocity_limit)
   q_next = clamp(q + dt * v_next, lower_limit, upper_limit)

The implementation also includes joint damping in ``damping`` and stops the
slider when a position limit is reached. Gradients at clamp boundaries are
piecewise-defined; the example stays away from those boundaries.

.. dropdown:: Complete provider (solver.py)
   :icon: code

   .. literalinclude:: ../../../scripts/tutorials/sim/simple_newton_solver/solver.py
      :language: python
      :linenos:

.. dropdown:: Complete scene and rollout (demo.py)
   :icon: code

   .. literalinclude:: ../../../scripts/tutorials/sim/simple_newton_solver/demo.py
      :language: python
      :linenos:

Scope and next steps
--------------------

This educational solver accepts independent dynamic prismatic leaves with
fixed ancestors and zero gravity. It rejects moving-body collisions, particles,
mimic constraints, friction and position/velocity target drives. Its parameters
belong to one finalized model; save learned values explicitly before rebuilding
the scene. See the tutorial directory's ``README.md`` for SDK prerequisites,
limit behavior, and notes about differentiating Cartesian losses.
