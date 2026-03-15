
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

   # Assume you have a robot instance and arm_name
   motion_cfg = MotionGenCfg(
       planner_cfg=ToppraPlannerCfg(
           robot_uid=robot.uid,
           control_part=arm_name,
           constraints={
               "velocity": 0.2,
               "acceleration": 0.5,
           },
       )
   )
   motion_gen = MotionGenerator(cfg=motion_cfg)

   # Plan a joint-space trajectory
   current_state = PlanState(position=np.array([0., 0., 0., 0., 0., 0.]))
   target_states = [PlanState(position=np.array([0.5, 0.2, 0., 0., 0., 0.]))]
   plan_result = motion_gen.plan(
       current_state=current_state,
       target_states=target_states
   )
   success = plan_result.success
   positions = plan_result.positions
   velocities = plan_result.velocities
   accelerations = plan_result.accelerations
   times = plan_result.t_series
   duration = plan_result.duration

   # Generate a discrete trajectory (joint or Cartesian)
   qpos_list, xpos_list = motion_gen.create_discrete_trajectory(
       qpos_list=[[0,0,0,0,0,0],[0.5,0.2,0,0,0,0]],
       sample_num=20
   )

API Reference
~~~~~~~~~~~~~

**Initialization**

.. code-block:: python

   motion_cfg = MotionGenCfg(
       planner_cfg=ToppraPlannerCfg(
           robot_uid=robot.uid,
           control_part=arm_name,
           constraints={
               "velocity": 0.2,
               "acceleration": 0.5,
           },
       )
   )
   MotionGenerator(cfg=motion_cfg)

- ``cfg``: MotionGenCfg instance, containing the specific planner's configuration (like ``ToppraPlannerCfg``)
- ``robot_uid``: Robot unique identifier
- ``control_part``: The specific part or arm you're controlling
- ``constraints``: Dictionary constraints of matching dimensions for each joint

**plan**

.. code-block:: python

   plan(
       current_state: PlanState,
       target_states: List[PlanState],
       sample_method=TrajectorySampleMethod.TIME,
       sample_interval=0.01,
       **kwargs
   ) -> PlanResult

- Plans a time-optimal trajectory (joint space), returning a ``PlanResult`` data class.

**create_discrete_trajectory**

.. code-block:: python

   create_discrete_trajectory(
       xpos_list=None,
       qpos_list=None,
       is_use_current_qpos=True,
       is_linear=False,
       sample_method=TrajectorySampleMethod.QUANTITY,
       sample_num=20,
       qpos_seed=None,
       **kwargs
   ) -> Tuple[List[np.ndarray], List[np.ndarray]]

- Generates a discrete trajectory between waypoints (joint or Cartesian), auto-handles FK/IK.

**estimate_trajectory_sample_count**

.. code-block:: python

   estimate_trajectory_sample_count(
       xpos_list=None,
       qpos_list=None,
       step_size=0.01,
       angle_step=np.pi/90,
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
