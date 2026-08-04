embodichain.learning.rl.buffer
==============================

.. automodule:: embodichain.learning.rl.buffer

Overview
--------

The ``buffer`` package provides the on-policy rollout buffer used by RL
algorithms. :class:`RolloutBuffer` owns the preallocated ``TensorDict`` storage
that collectors fill and algorithms consume; :func:`iterate_minibatches` and
:func:`transition_view` are helpers for slicing minibatches out of a rollout.

.. rubric:: Classes

.. autosummary::

   RolloutBuffer

.. rubric:: Functions

.. autosummary::

   iterate_minibatches
   transition_view

.. rubric:: Submodules

.. autosummary::

   standard_buffer
   utils

.. currentmodule:: embodichain.learning.rl.buffer

Rollout Buffer Classes
----------------------

.. automodule:: embodichain.learning.rl.buffer.standard_buffer
   :members:
   :undoc-members:
   :show-inheritance:

Buffer Utilities
----------------

.. automodule:: embodichain.learning.rl.buffer.utils
   :members:
   :undoc-members:
   :show-inheritance:

   