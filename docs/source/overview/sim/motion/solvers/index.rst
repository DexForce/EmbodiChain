Solvers
========

Kinematic solvers map joint positions to end-effector poses (forward
kinematics, FK) and find joint positions for target poses (inverse kinematics,
IK). Use :doc:`../motion_generator` when you need a timed trajectory between
poses.

Choose a solver
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Solver
     - Use it for
   * - :doc:`pytorch_solver`
     - Numerical IK with batched PyTorch computation and multiple seeds.
   * - :doc:`differential_solver`
     - Jacobian-based joint updates for differential pose control.
   * - :doc:`pink_solver`
     - Optimization with frame tasks and null-space posture objectives.
   * - :doc:`pinocchio_solver`
     - Single-target numerical IK using Pinocchio.
   * - :doc:`opw_solver`
     - Analytical IK for 6-DOF arms matching the OPW parameterization.
   * - :doc:`srs_solver`
     - Analytical IK for 7-DOF spherical-rotational-spherical arms.
   * - :doc:`ur_solver`
     - Analytical IK for supported Universal Robots models.
   * - :doc:`neural_ik_solver`
     - Experimental learned IK with a trained Franka Panda checkpoint.

Analytical solvers require matching robot geometry. Numerical solvers support
more general chains but depend on seed quality and convergence settings.

.. _motion-solver-conventions:

Shared FK/IK conventions
------------------------

Configure the robot model, joint order, root link, end link, and optional TCP
through the solver's configuration. Joint positions use radians; pose matrices
are homogeneous transforms. Direct solver calls use the solver's root/TCP
conventions. For simulation targets, use the robot's IK methods and their
documented pose frame; they handle conversion to the solver root frame.

For solvers that support batches, joint inputs have shape ``(B, DOF)`` and
pose inputs have shape ``(B, 4, 4)``. A basic FK-to-IK call is:

.. code-block:: python

   target_pose = solver.get_fk(qpos_seed)
   success, solutions = solver.get_ik(
       target_xpos=target_pose,
       qpos_seed=qpos_seed,
   )

``get_ik()`` returns validity/success first and joint solutions second. Check
success before using a solution. Output shapes and the meaning of
``return_all_solutions`` differ by solver; the pages below document these
exceptions. PinocchioSolver currently solves one target per call.

For full signatures and configuration fields, see the
:doc:`solver API </api_reference/embodichain/embodichain.lab.sim.motion.solvers>`.

.. toctree::
   :maxdepth: 1

   pytorch_solver
   differential_solver
   pink_solver
   pinocchio_solver
   opw_solver
   srs_solver
   ur_solver
   neural_ik_solver
