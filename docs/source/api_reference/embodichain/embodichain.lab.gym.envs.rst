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

   .. rubric:: Submodules

   .. autosummary::

      demo
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

Generation Preparation
----------------------

``BaseEnv.acquire_generation_lease(owner)`` reserves the complete environment
batch, suppresses automatic resets, and rejects ordinary resets until the same
owner calls ``release_generation_lease``. ``generation_epoch`` invalidates old
runtime bindings when ownership or episode preparation changes. A lease is an
ownership contract; callers must still serialize host access. Gym leases and
direct simulation generation hosts register the same simulator-batch ownership,
so they cannot reserve that batch for different owners.

``EmbodiedEnv.prepare_generation_episode`` requires the active lease and four
explicit preparation, restoration, settling, and verification callbacks. It
discards buffered recordings, so the caller must freeze the old episode first.
After settling it resets episode managers before verification, so validation
observes the new episode's state. It obtains the initial observation and seeds
recording history only after verification succeeds. Failed preparation leaves
stepping disabled; normal reset events and RNG reseeding are not part of this path.

The shared
:class:`~embodichain.lab.trajectory_generation.initial_state.FixedSceneHost`
assembles this lifecycle with physical state restoration and a trusted task
profile. See :doc:`/overview/trajectory_generation` for usage and limitations.

``BaseEnv.observe_generation_commands(owner, observer)`` installs one serialized
callback for the active lease. It runs after controller submission and before
physics, allowing the collector to copy actual controller targets. An exception
occurs after the command was submitted, so the attempt still counts and the
collector must stop safely.

Controller-ready Actions
------------------------

``ControllerAction`` marks commands that already crossed the raw-policy
preprocessing boundary. The environment validates these commands and skips
``ActionManager`` terms in ``pre`` mode while retaining the normal Gym step and
``post`` processing lifecycle.

.. currentmodule:: embodichain.lab.gym.envs.types

.. autoclass:: ControllerAction
    :members:

Demonstration Episodes
----------------------

The segment-aware demonstration API represents a complete task as one episode
containing one or more semantic subtasks. Segment action iterables may be lazy,
and the common executor records per-environment lengths, terminal status, and
segment spans.

``execute_demo_episode`` accepts ``step_observer`` to consume the existing
``env.step`` result and its active-row mask before terminal handling, without
querying observations again. With explicit segments, ``row_step_limits`` provides
one command count per row; zero skips a row and finished rows receive holds while
remaining rows finish. Their post-completion holds are not recorded as training
frames. Omitting these arguments preserves ordinary segment execution.

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
