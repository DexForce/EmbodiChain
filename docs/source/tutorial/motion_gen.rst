.. _tutorial_motion_generator:

Motion Generator
================

.. currentmodule:: embodichain.lab.sim.planners.motion_generator

The ``MotionGenerator`` class in EmbodiChain provides a unified and extensible interface for robot trajectory planning. It supports time-optimal trajectory generation (currently via TOPPRA), joint/Cartesian interpolation, and is designed for easy integration with RL, imitation learning, and classical control scenarios.

Key Features
------------

- **Unified API**: One interface for multiple planning strategies (time-optimal, interpolation, etc.)
- **Constraint Support**: Velocity/acceleration constraints configurable per joint
- **Flexible Input**: Supports both joint space and Cartesian space waypoints
- **Extensible**: Easy to add new planners (RRT, PRM, etc.)
- **Integration Ready**: Can be used in RL, imitation learning, or classical pipelines


The Code
~~~~~~~~

The tutorial corresponds to the ``motion_generator.py`` script in the ``scripts/tutorials/sim`` directory.

.. dropdown:: Code for motion_generator.py
    :icon: code

    .. literalinclude:: ../../../scripts/tutorials/sim/motion_generator.py
        :language: python
        :linenos:

Typical Usage
~~~~~~~~~~~~~

.. code-block:: python

   from embodichain.lab.sim.planners import MotionGenerator, MotionGenCfg, ToppraPlannerCfg
   from embodichain.lab.sim.planners.toppra_planner import ToppraPlanOptions
   from embodichain.lab.sim.planners.utils import PlanState, TrajectorySampleMethod, MoveType

   # Assume you have a robot instance and arm_name
   # Constraints are now specified in ToppraPlanOptions, not in ToppraPlannerCfg
   motion_cfg = MotionGenCfg(
       planner_cfg=ToppraPlannerCfg(
           robot_uid=robot.uid,
       )
   )
   motion_gen = MotionGenerator(cfg=motion_cfg)

   # Create options with constraints and planning parameters
   plan_opts = ToppraPlanOptions(
       constraints={
           "velocity": 0.2,
           "acceleration": 0.5,
       },
       sample_method=TrajectorySampleMethod.TIME,
       sample_interval=0.01
   )

   # Create motion generation options
   motion_opts = MotionGenOptions(
       plan_opts=plan_opts,
       control_part=arm_name,
       is_interpolate=True,
       interpolate_nums=10,
   )

   # Plan a joint-space trajectory (use generate() method instead of plan())
   target_states = [
       PlanState(move_type=MoveType.JOINT_MOVE, qpos=torch.tensor([0.5, 0.2, 0., 0., 0., 0.]))
   ]
   plan_result = motion_gen.generate(
       target_states=target_states,
       options=motion_opts
   )
   success = plan_result.success
   positions = plan_result.positions
   velocities = plan_result.velocities
   accelerations = plan_result.accelerations
   duration = plan_result.duration

API Reference
~~~~~~~~~~~~~

**Initialization**

.. code-block:: python

   from embodichain.lab.sim.planners.toppra_planner import ToppraPlanOptions

   motion_cfg = MotionGenCfg(
       planner_cfg=ToppraPlannerCfg(
           robot_uid=robot.uid,
       )
   )
   MotionGenerator(cfg=motion_cfg)

- ``cfg``: MotionGenCfg instance, containing the specific planner's configuration (like ``ToppraPlannerCfg``)
- ``robot_uid``: Robot unique identifier
- ``constraints``: Now specified in ``ToppraPlanOptions`` (passed via ``MotionGenOptions.plan_opts``)

**MotionGenOptions**

.. code-block:: python

   motion_opts = MotionGenOptions(
       plan_opts=ToppraPlanOptions(...),  # Options for the underlying planner
       control_part=arm_name,              # Robot part to control (e.g., 'left_arm')
       is_interpolate=False,               # Whether to pre-interpolate trajectory
       interpolate_nums=10,                # Number of interpolation points between waypoints
       is_linear=False,                    # Use Cartesian linear interpolation if True, else joint space
       interpolate_position_step=0.002,    # Step size for Cartesian interpolation (meters)
       interpolate_angle_step=np.pi/90,   # Step size for joint interpolation (radians)
       start_qpos=torch.tensor([...]),     # Optional starting joint configuration
   )

**generate** (formerly ``plan``)

.. code-block:: python

   generate(
       target_states: List[PlanState],
       options: MotionGenOptions = MotionGenOptions(),
   ) -> PlanResult

- Generates a time-optimal trajectory (joint space), returning a ``PlanResult`` data class.
- Uses ``target_states`` (list of PlanState) and ``options`` (MotionGenOptions) instead of individual parameters.

**interpolate_trajectory**

.. code-block:: python

   interpolate_trajectory(
       control_part: str | None = None,
       xpos_list: torch.Tensor | None = None,
       qpos_list: torch.Tensor | None = None,
       options: MotionGenOptions = MotionGenOptions(),
   ) -> Tuple[torch.Tensor, torch.Tensor | None]

- Interpolates trajectory between waypoints (joint or Cartesian), auto-handles FK/IK.

**estimate_trajectory_sample_count**

.. code-block:: python

   estimate_trajectory_sample_count(
       xpos_list=None,
       qpos_list=None,
       step_size=0.01,
       angle_step=np.pi/90,
       control_part=None,
   ) -> torch.Tensor

- Estimates the number of samples needed for a trajectory.

**plan_with_collision**

.. code-block:: python

   plan_with_collision(...)

- (Reserved) Plan trajectory with collision checking (not yet implemented).

Notes & Best Practices
~~~~~~~~~~~~~~~~~~~~~

- Only collision-free planning is currently supported; collision checking is a placeholder.
- Input/outputs are numpy arrays or torch tensors; ensure type consistency.
- Robot instance must implement get_joint_ids, compute_fk, compute_ik, get_proprioception, etc.
- For custom planners, extend the PlannerType Enum and _create_planner methods.
- Constraints (velocity, acceleration) are now specified in ``ToppraPlanOptions``, not in ``ToppraPlannerCfg``.
- Use ``PlanState.qpos`` (not ``position``) for joint positions.
