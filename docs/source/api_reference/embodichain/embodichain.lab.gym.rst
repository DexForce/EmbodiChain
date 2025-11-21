embodichain.lab.gym
===============================

.. automodule:: embodichain.lab.gym

   .. rubric:: Submodules

   .. autosummary::
      :toctree: .

      envs
      utils

.. currentmodule:: embodichain.lab.gym

Overview
--------

The ``gym`` module provides a comprehensive framework for creating robot learning environments. It extends the Gymnasium interface to support multi-environment parallel execution,
custom observations, and robotic-specific functionality.

Key Features:

* **Multi-Environment Support**: Run multiple environment instances in parallel for efficient training
* **Gymnasium Integration**: Full compatibility with the Gymnasium API and ecosystem
* **Robotic Focus**: Built-in support for robot control, sensors, and manipulation tasks
* **Extensible Architecture**: Easy to create custom environments and tasks
* **GPU Acceleration**: Leverage GPU computing for high-performance simulation

Environments Module (envs)
---------------------------

.. currentmodule:: embodichain.lab.gym.envs

Base Environment Classes
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: BaseEnv
    :members:
    :inherited-members:
    :show-inheritance:

    The foundational environment class that provides the core functionality for all EmbodiChain RL environments.
    This class extends the Gymnasium ``Env`` interface with multi-environment support and robotic-specific features.

.. autoclass:: EnvCfg
    :members:
    :exclude-members: __init__, class_type

    Configuration class for basic environment settings including simulation parameters and environment count.

Embodied Environment Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: EmbodiedEnv
    :members:
    :inherited-members:
    :show-inheritance:

    An advanced environment class that provides additional features for embodied AI research, including
    sophisticated observation management, event handling, and multi-modal sensor integration.

.. autoclass:: EmbodiedEnvCfg
    :members:
    :exclude-members: __init__, class_type

    Configuration class for embodied environments with extended settings for lighting, observation management,
    and advanced simulation features.

Utilities Module (utils)
-------------------------

.. currentmodule:: embodichain.lab.gym.utils

Registration System
~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain.lab.gym.utils.registration

.. autoclass:: EnvSpec
    :members:
    :show-inheritance:

    Specification class for environment registration, containing environment metadata and creation parameters.

.. autofunction:: register

    Register a new environment class with the EmbodiChain environment registry.

    :param name: Unique identifier for the environment
    :param cls: Environment class (must inherit from BaseEnv or BaseEnv)  
    :param max_episode_steps: Maximum steps per episode (optional)
    :param default_kwargs: Default keyword arguments for environment creation

.. autofunction:: register_env

    Decorator function for registering environment classes. This is the recommended way to register environments.

    :param uid: Unique identifier for the environment
    :param max_episode_steps: Maximum steps per episode (optional)
    :param override: Whether to override existing environment with same ID
    :param kwargs: Additional registration parameters
    
    Example:
        .. code-block:: python

            @register_env("MyEnv-v1", max_episode_steps=1000)
            class MyCustomEnv(BaseEnv):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)

.. autofunction:: make

    Create an environment instance from a registered environment ID.

    :param env_id: Registered environment identifier
    :param kwargs: Additional keyword arguments for environment creation
    :returns: Environment instance

.. autoclass:: TimeLimitWrapper
    :members:
    :show-inheritance:

    Gymnasium wrapper that adds episode time limits to environments.

Gymnasium Utilities
~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain.lab.gym.utils.gym_utils

    Helper functions and utilities for Gymnasium environment integration.

Miscellaneous Utilities
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain.lab.gym.utils.misc

    Miscellaneous utility functions for environment development and debugging.

