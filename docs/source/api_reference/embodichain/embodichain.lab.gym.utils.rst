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

.. autoclass:: TimeLimitWrapper
    :members:
    :show-inheritance:

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

   