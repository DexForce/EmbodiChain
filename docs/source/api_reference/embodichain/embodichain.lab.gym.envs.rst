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
      managers
      wrapper

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

Demonstration Episodes
----------------------

The segment-aware demonstration API represents a complete task as one episode
containing one or more semantic subtasks. Segment action iterables may be lazy,
and the common executor records per-environment lengths, terminal status, and
segment spans.

.. autoclass:: DemoSegment
    :members:

.. autoclass:: DemoSegmentResult
    :members:

.. autoclass:: DemoEpisodeResult
    :members:

.. autofunction:: execute_demo_episode

.. autofunction:: resolve_demo_segments

Wrappers
--------

.. autoclass:: NoFailWrapper
    :members:
    :show-inheritance:

.. autoclass:: ReplayWrapper
    :members:
    :show-inheritance:
