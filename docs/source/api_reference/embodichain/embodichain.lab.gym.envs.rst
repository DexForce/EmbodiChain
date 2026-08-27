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
      differentiable_env
      expert_program
      managers
      types
      wrapper

.. toctree::
   :hidden:

   embodichain.lab.gym.envs.expert_program

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

``DifferentiableEmbodiedEnv`` keeps the standard environment lifecycle while
bridging Newton trajectories into PyTorch autograd for analytic policy-gradient
tasks. Dynamics and explicit kinematics subclasses provide the action and
output kernels; the base class owns tape-aware stepping and deferred resets.

.. currentmodule:: embodichain.lab.gym.envs.differentiable_env

.. autoclass:: DifferentiableEmbodiedEnv
    :members:
    :inherited-members:
    :show-inheritance:

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

.. currentmodule:: embodichain.lab.gym.envs.demo

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

The shared settling monitor is used by both reset events and Expert Program
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
