NeRD supervised-learning adapters
================================

NeRD uses the shared supervised optimizer update while retaining its physical
codec, normalized loss and numerical execution policy. The standalone recipe
connects immutable NeRD windows to the common trainer; its checkpoints use the
SL format. Historical NeRD training checkpoints remain owned by the existing
``dynamics`` workflow.

.. automodule:: embodichain.learning.dynamics.supervised
   :members: NeRDObjective, make_nerd_batch

.. automodule:: embodichain.learning.dynamics.sl_recipe
   :members: NeRDWindowSource, build_supervised_trainer
