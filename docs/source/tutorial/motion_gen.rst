.. _tutorial_motion_generator:

Motion Generator
================

.. currentmodule:: embodichain.lab.sim.motion.motion_generator

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

   from embodichain.lab.sim.motion.motion_generator import MotionGenerator, MotionGenCfg
   from embodichain.lab.sim.motion.planners import ToppraPlannerCfg
   from embodichain.lab.sim.motion.planners.toppra_planner import ToppraPlanOptions
   from embodichain.lab.sim.motion.planners.utils import PlanState, TrajectorySampleMethod, MoveType

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
       strategy="motion_gen",
       plan_opts=plan_opts,
       control_part=arm_name,
       is_interpolate=True,
       interpolate_nums=10,
   )

   # Plan a joint-space trajectory (use generate() method instead of plan())
   target_states = [
       PlanState.from_qpos(torch.tensor([[0.5, 0.2, 0., 0., 0., 0.]]))
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

Timed playback
~~~~~~~~~~~~~~

``PlanResult.positions``, ``velocities``, and ``dt`` form one timed command.
During physical playback, send position and velocity sample ``i``, then advance
physics by ``dt[i + 1]`` before measuring the state at sample ``i + 1``. After
the final waypoint, write its position with a zero velocity target so the controller holds the terminal
position instead of retaining the last feed-forward command. The complete
tutorial script demonstrates this loop and resamples fractional planner
intervals onto a uniform cadence no larger than the configured physics step.

To compare the controller under position-only and position-plus-velocity
commands with identical initial state, gains, reference, and cadence, run::

   python examples/sim/motion/trajectory_velocity_tracking.py --headless \
       --device cpu --output-dir trajectory_velocity_results

The script restores the robot's configured initial state and clears its dynamics
before each trial. The position-only trial explicitly sends zero target velocity. It writes ``tracking.csv`` with samples
timestamped after each control interval and ``tracking.png`` with the aggregate
joint error, and prints RMSE, P95, and maximum absolute joint error. Treat the
numbers as measurements for the selected robot, drive gains, physics backend,
and cadence; velocity targets do not imply universal improvement.

DexSim must support both target-position and target-velocity writes. The
example uses :meth:`Robot.set_qpos` and :meth:`Robot.set_qvel`, which map to
DexSim's articulation target APIs. It is headless by default and requires the
normal simulation assets and runtime.

API Reference
~~~~~~~~~~~~~

**Initialization**

.. code-block:: python

   from embodichain.lab.sim.motion.planners.toppra_planner import ToppraPlanOptions

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
       strategy="motion_gen",               # "motion_gen" or "ik_interp"
       sample_count=None,                    # Optional normalized output length
       interpolation_dt=None,                # Required for deterministic interpolation
       plan_opts=ToppraPlanOptions(...),  # Options for the underlying planner
       control_part=arm_name,              # Robot part to control (e.g., 'left_arm')
       is_interpolate=False,               # Whether to pre-interpolate trajectory
       interpolate_nums=10,                # Number of interpolation points between waypoints
       is_linear=False,                    # Use Cartesian linear interpolation if True, else joint space
       interpolate_position_step=0.002,    # Step size for Cartesian interpolation (meters)
       interpolate_angle_step=np.pi/90,   # Step size for joint interpolation (radians)
       start_qpos=torch.tensor([...]),     # Optional starting joint configuration
   )

**generate**

.. code-block:: python

   generate(
       target_states: list[PlanState],
       options: MotionGenOptions | None = None,
   ) -> PlanResult

- ``strategy="motion_gen"`` delegates to the configured backend; ``strategy="ik_interp"`` performs deterministic waypoint IK and joint interpolation and requires ``interpolation_dt``.
- Returns a normalized, environment-batched ``PlanResult`` with explicit ``dt``
  and derived ``duration`` whenever positions are present. Its ``velocities``
  carry the matching velocity targets. Missing timing raises immediately.
- Uses ``target_states`` (list of PlanState) and ``options`` (MotionGenOptions) instead of individual parameters.

**interpolate_trajectory**

.. code-block:: python

   interpolate_trajectory(
       control_part: str | None = None,
       xpos_list: torch.Tensor | None = None,
       qpos_list: torch.Tensor | None = None,
       options: MotionGenOptions | None = None,
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

Notes & Best Practices
~~~~~~~~~~~~~~~~~~~~~~

- TOPPRA and NeuralPlanner do not maintain a collision world. Select the optional
  cuRobo V2 backend for collision-aware planning and exact joint-trajectory
  collision validation; see :doc:`/overview/sim/motion/planners/curobo_planner`.
- Planning inputs and outputs use environment-batched PyTorch tensors.
- Robot instance must implement get_joint_ids, compute_fk, compute_ik, get_proprioception, etc.
- Custom planners subclass ``BasePlanner`` with a matching ``BasePlannerCfg`` and
  declare ``supported_move_types``. Register the pair with
  ``MotionGenerator.register_planner_type(name, planner_class, planner_cfg_class)``.
- Constraints (velocity, acceleration) are now specified in ``ToppraPlanOptions``, not in ``ToppraPlannerCfg``.
- Use ``PlanState.qpos`` (not ``position``) for joint positions.
