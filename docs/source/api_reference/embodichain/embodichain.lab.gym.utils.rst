embodichain.lab.gym.utils
=====================================

.. automodule:: embodichain.lab.gym.utils

Overview
--------

Utilities for the environment framework: the registration system
(:func:`register_env` decorator, :func:`make` factory, and the
:class:`EnvSpec`/:class:`TimeLimitWrapper` helpers), Gymnasium integration
helpers, miscellaneous environment utilities, and the
:class:`~embodichain.lab.gym.utils.profiler.EnvProfiler` for per-step / per-reset
and per-functor timing.

Registration System
-------------------

.. currentmodule:: embodichain.lab.gym.utils.registration

.. autoclass:: EnvSpec
    :members:
    :show-inheritance:

.. autofunction:: register

.. autofunction:: register_env

.. autofunction:: make

.. autofunction:: get_env_spec

.. autofunction:: build_env

.. autofunction:: make_vec

.. autofunction:: register_env_function

.. autoclass:: TimeLimitWrapper
    :members:
    :show-inheritance:

.. autodata:: REGISTERED_ENVS

Task Package Discovery
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: discover_task_packages

.. autofunction:: execute_init_hooks

Utility Modules
---------------


Gymnasium Utilities
~~~~~~~~~~~~~~~~~~~

.. automodule:: embodichain.lab.gym.utils.gym_utils


Miscellaneous
~~~~~~~~~~~~~

.. automodule:: embodichain.lab.gym.utils.misc


Profiling
~~~~~~~~~

.. automodule:: embodichain.lab.gym.utils.profiler
