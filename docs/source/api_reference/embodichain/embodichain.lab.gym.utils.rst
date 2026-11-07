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

Execution-free component composition
------------------------------------

``resolve_gym_components`` expands environment, embodiment and scene component
selectors through the existing strict decoders. Calls own their nested output
mappings and preserve field ownership and relative paths. They neither register
Gym environments nor construct runtime configurations or simulation resources.
Task Program selectors remain declarations: compose them separately through
``load_configured_task_program_deployment`` using the returned skill profile.
Complete runtime validation remains with the normal environment loader.

The optional ``component_sources`` mapping supplies an owned snapshot of every
selected physical component, keyed by its owner-relative or absolute path. Both
``resolve_gym_components`` and ``config_to_cfg`` consume it without reopening
component files. Missing selections and duplicate resolved paths reject; nested
declarations remain independent of caller mutations. Asset paths retain each
component's owner directory. Callers retain responsibility for verifying the
snapshot's provenance and external asset identity.

``ResolvedGymComponents.validate_scene_bindings`` checks named semantic native
roots against its resolved physical scene without constructing simulation.

.. currentmodule:: embodichain.lab.gym.utils

.. autosummary::

   ResolvedGymComponents
   resolve_gym_components

.. currentmodule:: embodichain.lab.gym.utils.composition

.. autoclass:: ResolvedGymComponents
   :members: validate_scene_bindings

.. autofunction:: resolve_gym_components
