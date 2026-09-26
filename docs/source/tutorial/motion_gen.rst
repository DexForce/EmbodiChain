.. _tutorial_motion_generator:

Motion Generator
================

This tutorial creates one CobotMagic arm, plans joint and Cartesian waypoint
paths with TOPPRA, and replays them in simulation. For backend selection,
minimal API examples, and result shapes, see
:doc:`/overview/sim/motion/motion_generator`.

Run the tutorial
----------------

Run from the repository root with the simulation runtime and assets installed:

.. code-block:: bash

   python scripts/tutorials/sim/motion_generator.py --headless --device cpu

Use ``--num-envs N`` for a batched scene, ``--physics newton`` for the Newton
backend, or ``--viser`` to publish the scene in a browser. Headless runs record
a whole-scene MP4 by default; use ``--disable-record`` to skip recording or
``--record-save-path PATH`` to choose its destination.

Follow the planning workflow
----------------------------

1. Create the simulation and robot, including an IK solver for the arm.
2. Configure ``MotionGenerator`` with ``ToppraPlannerCfg`` and place velocity,
   acceleration, and sampling settings in ``ToppraPlanOptions``.
3. Read the arm's starting joint positions and construct batched ``PlanState``
   waypoints. Cartesian pre-interpolation uses the configured IK solver.
4. Call ``generate()`` and check the result's success flags.
5. Replay the timed joint trajectory using ``play_joint_trajectory()`` on a
   fixed control grid. The helper keeps the physics period unchanged and
   recomputes velocity targets for the executed grid.

.. dropdown:: Complete tutorial script
   :icon: code

   .. literalinclude:: ../../../scripts/tutorials/sim/motion_generator.py
      :language: python
      :linenos:

Compare physical tracking
-------------------------

To compare position-only and position-plus-velocity commands with the same
initial state, gains, reference, and cadence, run:

.. code-block:: bash

   python examples/sim/motion/trajectory_velocity_tracking.py --headless \
       --device cpu --output-dir trajectory_velocity_results

The script restores the configured initial state before each trial. The
position-only trial explicitly writes zero target velocity. Outputs include
``tracking.csv`` with measurement timestamps, ``tracking.png`` with joint
errors, and printed RMSE, P95, and maximum absolute error. These measurements
apply to the selected robot, drive gains, physics backend, and cadence.

Next steps
----------

- :doc:`/tutorial/trajectory_planning` for trajectory diagnostics.
- :doc:`/overview/sim/motion/planners/index` for alternative planning backends.
- :doc:`/overview/sim/atomic_actions/index` for task-level planning and execution.
- :doc:`MotionGenerator API </api_reference/embodichain/embodichain.lab.sim.motion.motion_generator>`
  for complete options and helper methods.
