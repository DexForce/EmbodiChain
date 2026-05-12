embodichain.agents.rl.collector
================================

.. automodule:: embodichain.agents.rl.collector

Overview
--------

Collectors are responsible for interacting with vectorized environments and
assembling rollout data into a preallocated ``TensorDict`` layout.

.. rubric:: Classes

.. autosummary::

   BaseCollector
   SyncCollector

.. currentmodule:: embodichain.agents.rl.collector

BaseCollector
-------------

.. autoclass:: BaseCollector
   :members:
   :show-inheritance:

SyncCollector
-------------

.. autoclass:: SyncCollector
   :members:
   :show-inheritance:
