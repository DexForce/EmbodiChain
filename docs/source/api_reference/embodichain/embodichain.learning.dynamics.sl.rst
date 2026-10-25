NeRD supervised-learning adapters
=================================

NeRD uses the shared supervised optimizer update while retaining its physical
codec, normalized loss and numerical execution policy. The standalone recipe
connects immutable NeRD windows to the common trainer; its checkpoints use the
SL format. Historical NeRD training checkpoints remain owned by the existing
``dynamics`` workflow.

Physical objective and batches
------------------------------

.. currentmodule:: embodichain.learning.dynamics.supervised

.. autosummary::

   NeRDObjective
   make_nerd_batch

.. autoclass:: NeRDObjective
   :members:

.. autofunction:: make_nerd_batch

Offline source and training recipe
----------------------------------

.. currentmodule:: embodichain.learning.dynamics.sl_recipe

.. autosummary::

   NeRDWindowSource
   build_supervised_trainer

.. autoclass:: NeRDWindowSource
   :members:

.. autofunction:: build_supervised_trainer
