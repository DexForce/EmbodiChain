
.. _tutorial_solver:

Create a solver
===============

.. currentmodule:: embodichain.lab.sim.solvers

Overview
~~~~~~~~

The ``solver`` module in EmbodiChain provides a unified and extensible interface for robot kinematics computation, including forward kinematics (FK), inverse kinematics (IK), and Jacobian calculation. It supports multiple solver backends (e.g., Pinocchio, OPW, SRS, PINK, PyTorch) and is designed for both simulation and real-robot applications.

Key Features
------------
- **Unified API**: Abstract base class (`BaseSolver`) defines a common interface for all solvers.
- **Multiple Backends**: Supports Pinocchio, OPW, SRS, PINK, PyTorch, and differential solvers.
- **Flexible Configuration**: Easily switch solver type and parameters via configuration.
- **Batch and Single Query**: Supports both batch and single FK/IK/Jacobian queries.
- **Extensible**: New solvers can be added by subclassing `BaseSolver` and implementing required methods.

Example: Using PinkSolver
~~~~~~~~~~~~~~~~~~~~~~~~~


.. code-block:: python

   from embodichain.lab.sim.solvers import PinkSolverCfg
   from embodichain.lab.sim.objects.robot import Robot

   # 1. Configure PinkSolver
   pink_cfg = PinkSolverCfg(
       urdf_path="/path/to/robot.urdf",
       joint_names=[
           "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
           "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
       ],
       end_link_name="ee_link",
       root_link_name="base_link"
   )
   # 2. Assign solver config to robot config
   robot_cfg.solver_cfg = pink_cfg
   # 3. Instantiate robot (solver will be initialized automatically)
   robot = Robot(cfg=robot_cfg, entities=[], device="cpu")

   # 4. Use FK/IK/Jacobian
   qpos = [0.0, -1.57, 1.57, 0.0, 1.57, 0.0]  # 6-DOF joint angles (radians)
   ee_pose = robot.compute_fk(qpos)  # Forward kinematics, returns 4x4 matrix
   print("End-effector pose (FK):\n", ee_pose)

   import numpy as np
   target_pose = np.array([
       [0, -1, 0, 0.5],
       [1,  0, 0, 0.2],
       [0,  0, 1, 0.3],
       [0,  0, 0, 1.0]
   ])
   success, qpos_sol = robot.compute_ik(target_pose, joint_seed=qpos)
   print("IK success:", success)
   print("IK solution:", qpos_sol)

   J = robot.get_solver().get_jacobian(qpos)
   print("Jacobian:\n", J)

**Note**

- robot.compute_fk(qpos) internally calls the bound solver's get_fk method.
- robot.compute_ik(target_pose, joint_seed) internally calls the solver's get_ik method.

API Reference
~~~~~~~~~~~~~

**BaseSolver**

.. code-block:: python

   class BaseSolver:
       def get_fk(self, qpos, **kwargs) -> torch.Tensor:
           """Compute forward kinematics for the end-effector."""

       def get_ik(self, target_pose, joint_seed=None, num_samples=None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
           """Compute inverse kinematics for a given pose."""

       def get_jacobian(self, qpos, locations=None, jac_type="full") -> torch.Tensor:
           """Compute the Jacobian matrix for the given joint positions."""

- **set_ik_nearst_weight**: Set weights for IK nearest neighbor search.
- **set_position_limits / get_position_limits**: Set or get joint position limits.
- **set_tcp / get_tcp**: Set or get the tool center point (TCP) transformation.

**PinkSolver**

- Implements all BaseSolver methods using the Pink library.
- Supports custom task lists, solver type selection, and joint limit handling.
- See PinkSolverCfg for all configuration options.

Configuration
~~~~~~~~~~~~~

- All solvers are configured via a `SolverCfg` or its subclass (e.g., `PinkSolverCfg`).
- Key config fields: `urdf_path`, `joint_names`, `end_link_name`, `root_link_name`, `tcp`, and solver-specific parameters.
- Use `cfg.init_solver()` to instantiate the solver, or assign to `robot_cfg.solver_cfg` for automatic integration.

Notes & Best Practices
~~~~~~~~~~~~~~~~~~~~~
- Always ensure URDF and joint/link names match your robot model.
- For IK, providing a good `qpos_seed` improves convergence and solution quality.
- Use `set_iteration_params` (if available) to tune solver performance for your application.
- For custom robots or new algorithms, subclass `BaseSolver` and register your solver.

See Also
~~~~~~~~
- :ref:`tutorial_motion_generator` — Motion Generator
- :ref:`tutorial_basic_env` — Basic Environment Setup
