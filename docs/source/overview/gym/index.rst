Gym
===================

.. currentmodule:: embodichain.lab.gym

The ``gym`` module provides a comprehensive framework for creating robot learning environments. It extends the Gymnasium interface to support multi-environment parallel execution, custom observations, and robotic-specific functionality.

Environment Classes
-------------------

Base Environments
~~~~~~~~~~~~~~~~~

- :class:`envs.BaseEnv` - Foundational environment class that provides core functionality for all EmbodiChain RL environments
- :class:`envs.EnvCfg` - Configuration class for basic environment settings

Embodied Environments
~~~~~~~~~~~~~~~~~~~~~

- :class:`envs.EmbodiedEnv` - Advanced environment class for complex Embodied AI tasks with configuration-driven architecture
- :class:`envs.EmbodiedEnvCfg` - Configuration class for Embodied Environments

.. toctree::
   :maxdepth: 1

   env.md