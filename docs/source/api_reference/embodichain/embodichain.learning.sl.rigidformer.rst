Rigid-body dynamics learning
============================

A lazy optional adapter binds the pinned RigidFormer implementation to causal
point-cloud windows and shared supervised updates. Reference geometry and point
correspondence are explicit; this adapter does not declare action-gradient or
native-solver capabilities. Install the rigidformer extra only for this model.

embodichain.data_pipeline.datasets.rigid_body
---------------------------------------------

.. currentmodule:: embodichain.data_pipeline.datasets.rigid_body

.. autosummary::

   RigidBodySequenceSource

.. autoclass:: RigidBodySequenceSource
   :members:

embodichain.learning.models.rigidformer
---------------------------------------

.. currentmodule:: embodichain.learning.models.rigidformer

.. autosummary::

   RIGIDFORMER_REVISION
   RigidFormerCfg
   RigidFormerModel

.. autoclass:: RIGIDFORMER_REVISION
   :members:

.. autoclass:: RigidFormerCfg
   :members:

.. autoclass:: RigidFormerModel
   :members:

embodichain.learning.sl.rigidformer
-----------------------------------

.. currentmodule:: embodichain.learning.sl.rigidformer

.. autosummary::

   RigidFormerObjective
   rigidformer_batch
   build_rigidformer_trainer

.. autoclass:: RigidFormerObjective
   :members:

.. autofunction:: rigidformer_batch

.. autofunction:: build_rigidformer_trainer
