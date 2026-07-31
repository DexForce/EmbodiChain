embodichain.lab.sim.common
======================================

.. automodule:: embodichain.lab.sim.common

Overview
--------

:class:`BatchEntity` is the abstract base class for every batched scene entity
in EmbodiChain. Rigid objects, articulations (and therefore robots), cameras,
lights, and sensors all derive from it. A ``BatchEntity`` wraps a list of
underlying DexSim entities and exposes their state as batched tensors with a
leading ``B = num_envs`` dimension, so a single instance represents one entity
replicated across all parallel environments.

.. rubric:: Classes

.. autosummary::

   BatchEntity

.. currentmodule:: embodichain.lab.sim.common

.. autoclass:: BatchEntity
   :members:
   :undoc-members:
   :show-inheritance:
