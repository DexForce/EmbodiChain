Supervised learning (SL)
========================

The SL layer composes objective-owned losses with a common optimizer update and
offline run lifecycle. Batches retain their tensor-tree structure and causal
masks; source adapters, objectives and downstream execution define their semantics.
Importing this package does not initialize simulation or reinforcement learning.

embodichain.learning.sl
-----------------------

.. currentmodule:: embodichain.learning.sl

.. autosummary::

   LossResult
   SupervisedBatch
   SupervisedObjective
   SupervisedLearner
   SupervisedSource
   TensorBatchSource
   SupervisedTrainer
   SupervisedTrainerCfg

.. autoclass:: LossResult
   :members:

.. autoclass:: SupervisedBatch
   :members:

.. autoclass:: SupervisedObjective
   :members:

.. autoclass:: SupervisedLearner
   :members:

.. autoclass:: SupervisedSource
   :members:

.. autoclass:: TensorBatchSource
   :members:

.. autoclass:: SupervisedTrainer
   :members:

.. autoclass:: SupervisedTrainerCfg
   :members:

embodichain.learning.sl.learner
-------------------------------

.. currentmodule:: embodichain.learning.sl.learner

.. autosummary::

   SupervisedLearner

.. autoclass:: SupervisedLearner
   :members:

embodichain.learning.sl.sources
-------------------------------

.. currentmodule:: embodichain.learning.sl.sources

.. autosummary::

   SupervisedSource
   TensorBatchSource

.. autoclass:: SupervisedSource
   :members:

.. autoclass:: TensorBatchSource
   :members:

embodichain.learning.sl.trainer
-------------------------------

.. currentmodule:: embodichain.learning.sl.trainer

.. autosummary::

   SupervisedTrainerCfg
   SupervisedTrainer

.. autoclass:: SupervisedTrainerCfg
   :members:

.. autoclass:: SupervisedTrainer
   :members:

embodichain.learning.sl.types
-----------------------------

.. currentmodule:: embodichain.learning.sl.types

.. autosummary::

   SupervisedBatch
   LossResult
   SupervisedObjective

.. autoclass:: SupervisedBatch
   :members:

.. autoclass:: LossResult
   :members:

.. autoclass:: SupervisedObjective
   :members:

embodichain.learning.sl.workflow
--------------------------------

.. currentmodule:: embodichain.learning.sl.workflow

.. autosummary::

   train_from_config
   cli

.. autofunction:: train_from_config

.. autofunction:: cli
