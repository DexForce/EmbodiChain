embodichain.learning.rl.utils
=============================

.. automodule:: embodichain.learning.rl.utils

Overview
--------

The ``utils`` package contains helper utilities for RL configuration,
data conversion, and training orchestration. It exposes the
:class:`AlgorithmCfg` config and the data-conversion helpers
:func:`dict_to_tensordict` and :func:`flatten_dict_observation`, backed by the
``config``, ``helper``, and ``trainer`` submodules.

.. rubric:: Classes

.. autosummary::

   AlgorithmCfg

.. rubric:: Functions

.. autosummary::

   dict_to_tensordict
   flatten_dict_observation

.. rubric:: Submodules

.. autosummary::

   config
   helper
   trainer

Configuration Helpers
---------------------

.. automodule:: embodichain.learning.rl.utils.config
   :members:
   :undoc-members:
   :show-inheritance:

General Helpers
---------------

.. automodule:: embodichain.learning.rl.utils.helper
   :members:
   :undoc-members:
   :show-inheritance:

Trainer Utilities
-----------------

.. automodule:: embodichain.learning.rl.utils.trainer
   :members:
   :undoc-members:
   :show-inheritance:

   