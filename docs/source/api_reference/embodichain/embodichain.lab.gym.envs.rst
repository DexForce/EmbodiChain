embodichain.lab.gym.envs
====================================

.. automodule:: embodichain.lab.gym.envs

Overview
--------

The Gymnasium-compatible environment framework for embodied manipulation
tasks. :class:`BaseEnv` extends ``gym.Env`` with multi-environment (vectorized)
support and owns the :class:`~embodichain.lab.sim.sim_manager.SimulationManager`,
robot, sensors, and action/observation spaces. :class:`EmbodiedEnv` builds on
``BaseEnv`` and is the modular base class for concrete tasks: it wires in the
event, observation, reward, action, and dataset managers via the
functor/``FunctorCfg`` pattern. Tasks are registered with
:func:`~embodichain.lab.gym.utils.registration.register_env` and instantiated
through :func:`~embodichain.lab.gym.utils.registration.make`.

``EmbodiedEnvCfg.rl_task`` optionally enables measured RL outcomes on this same
host. ``TerminationManager`` supplies cached success/failure/truncation results,
and the environment records reset-target context for dynamic replay. Task Program
adapters and acceptance keep their own contract; use separate deployments.

   .. rubric:: Submodules

   .. autosummary::

      demo
      differentiable_env
      expert_trajectory
      task_program
      managers
      types
      wrapper

.. toctree::
   :hidden:

   embodichain.lab.gym.envs.task_program

.. currentmodule:: embodichain.lab.gym.envs

Environment Classes
-------------------

.. currentmodule:: embodichain.lab.gym.envs

.. autoclass:: BaseEnv
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: EnvCfg
    :members:
    :exclude-members: __init__, class_type

.. autoclass:: EmbodiedEnv
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: EmbodiedEnvCfg
    :members:
    :exclude-members: __init__, class_type

Differentiable Environment
--------------------------

``DifferentiableEnv`` provides one entry point with explicit execution modes.
The default ``step_mode="kinematic"`` records task-defined action, kinematics
and output hooks in a Warp tape without advancing the Newton solver. Tracked
terminal steps expose deferred reset IDs; callers finish backward before reset.

``step_mode="dynamics"`` maps actions to controls and advances the configured
solver through the Scene functional interface. Its independent state/history
supports automatic selected-row reset, graph-connected terminal observations
and ``detach_state()`` rollout boundaries. ``dynamics_state`` and
``dynamics_parameters()`` expose solver-owned tensors in this mode. Kinematic
tasks own their recurrence and must define ``detach_state()`` when used with a
segmented trainer. The mode is selected by the Python task constructor and is
independent of the solver name.

.. currentmodule:: embodichain.lab.gym.envs.differentiable_env

.. autoclass:: DifferentiableEnv
    :members:
    :inherited-members:
    :show-inheritance:

Expert Trajectories
-------------------

Expert trajectory configuration is source-neutral: handwritten generators,
motion generation, and Task Program execution use the same environment-owned
joint command mode. Position-only control remains the default. The optional
position-velocity mode records a stable flat ``[qpos, qvel]`` action layout
without changing the policy-facing Gym action space.

.. currentmodule:: embodichain.lab.gym.envs.expert_trajectory

.. autosummary::
   :nosignatures:

   EXPERT_TRAJECTORY_SCHEMA_VERSION
   ExpertActionSpec
   ExpertJointTrajectory
   ExpertTrajectoryCfg
   JointCommandMode
   build_expert_action_spec
   encode_expert_action
   prepare_expert_joint_trajectory

.. autodata:: EXPERT_TRAJECTORY_SCHEMA_VERSION

.. autoclass:: ExpertActionSpec
    :members:

.. autoclass:: ExpertJointTrajectory
    :members:

.. autoclass:: ExpertTrajectoryCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict, validate

.. autodata:: JointCommandMode

.. autofunction:: build_expert_action_spec

.. autofunction:: encode_expert_action

.. autofunction:: prepare_expert_joint_trajectory

Controller-ready Actions
------------------------

``ControllerAction`` marks controller-ready qpos/qvel/qf commands. The
environment keeps the envelope through preprocessing, validates it, and applies
it directly without invoking ``ActionManager``. Ordinary flat policy tensors
are processed and applied by the manager before physics advances.

.. currentmodule:: embodichain.lab.gym.envs.types

.. autoclass:: ControllerAction
    :members:

Demonstration Episodes
----------------------

The segment-aware demonstration API represents a complete task as one episode
containing one or more semantic subtasks. Segment action iterables may be lazy,
and the common executor records per-environment lengths, terminal status, and
segment spans.

``execute_demo_episode(on_segment_completed=...)`` optionally observes finalized
segment results after recording and before the next segment, including failure
and cancellation. Observers must not step or reset the environment; their errors
propagate to the caller. The default ``None`` retains ordinary execution.

Overall language belongs to the task: Task Programs declare a root
``instruction``, and handwritten/direct-control tasks use
``EmbodiedEnvCfg.task_instruction``. An explicit ``execute_demo_episode``
instruction overrides these defaults for one episode. The executor freezes the
resolved text and its source before planning, so disk recorders and online
language batches consume the same semantics even if configuration later changes.
Legacy recorder instructions remain a fallback. Segment annotations distinguish
explicit labels from overall-task fallback through ``instruction_source``.

.. currentmodule:: embodichain.lab.gym.envs.demo

.. autoclass:: DemoExecutionCfg
    :members:

.. autodata:: DemoOutputMode

.. autodata:: DemoSegmentOutcomeKind

.. autoclass:: DemoSegment
    :members:

.. autoclass:: DemoSegmentResult
    :members:

.. autoclass:: DemoEpisodeResult
    :members:

.. autofunction:: execute_demo_episode

.. autofunction:: resolve_demo_segments

.. autofunction:: resolve_demo_instruction

Dynamic Settling
----------------

The shared settling monitor is used by both reset events and Task Program
post-policies, so they apply the same row-local stability semantics.

.. currentmodule:: embodichain.lab.gym.envs.settling

.. autoclass:: DynamicSettleMonitorCfg
    :members:

.. autoclass:: DynamicSettleSample
    :members:

.. autoclass:: DynamicSettleState
    :members:

.. autoclass:: DynamicSettleMonitor
    :members:

Wrappers
--------

.. currentmodule:: embodichain.lab.gym.envs

.. autoclass:: NoFailWrapper
    :members:
    :show-inheritance:

.. autoclass:: ReplayWrapper
    :members:
    :show-inheritance:
