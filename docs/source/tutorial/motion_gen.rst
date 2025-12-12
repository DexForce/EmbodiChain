
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

   from embodichain.lab.sim.planners.motion_generator import MotionGenerator

   # Assume you have a robot instance and uid
   motion_gen = MotionGenerator(
       robot=robot,
       uid="arm",
       default_velocity=0.2,
       default_acceleration=0.5
   )

   # Plan a joint-space trajectory
   current_state = {"position": [0, 0, 0, 0, 0, 0]}
   target_states = [{"position": [0.5, 0.2, 0, 0, 0, 0]}]
   success, positions, velocities, accelerations, times, duration = motion_gen.plan(
       current_state=current_state,
       target_states=target_states
   )

   # Generate a discrete trajectory (joint or Cartesian)
   qpos_list, xpos_list = motion_gen.create_discrete_trajectory(
       qpos_list=[[0,0,0,0,0,0],[0.5,0.2,0,0,0,0]],
       sample_num=20
   )

API Reference
~~~~~~~~~~~~~

**Initialization**

.. code-block:: python

   MotionGenerator(
       robot: Robot,
       uid: str,
       sim=None,
       planner_type="toppra",
       default_velocity=0.2,
       default_acceleration=0.5,
       collision_margin=0.01,
       **kwargs
   )

- ``robot``: Robot instance, must support get_joint_ids, compute_fk, compute_ik
- ``uid``: Unique robot identifier (e.g., "arm")
- ``planner_type``: Planner type (default: "toppra")
- ``default_velocity``, ``default_acceleration``: Default joint constraints

**plan**

.. code-block:: python

   plan(
       current_state: Dict,
       target_states: List[Dict],
       sample_method=TrajectorySampleMethod.TIME,
       sample_interval=0.01,
       **kwargs
   ) -> Tuple[bool, positions, velocities, accelerations, times, duration]

- Plans a time-optimal trajectory (joint space), returns trajectory arrays and duration.

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
       **kwargs
   ) -> int

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
