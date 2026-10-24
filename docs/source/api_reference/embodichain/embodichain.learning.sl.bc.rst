Behavior cloning
================

State and visual action-chunk policies use the shared SL update and offline
trainer. Frozen normalization and declared target semantics travel with model
state; an executor owns causal observation history and per-environment queues.
The fixed-corpus reader binds views and target-transform versions to source
identity and returns only causal history images alongside future supervision.

embodichain.data_pipeline.datasets.bc_hdf5
------------------------------------------

.. currentmodule:: embodichain.data_pipeline.datasets.bc_hdf5

.. autosummary::

   BCWindowDataset

.. autoclass:: BCWindowDataset
   :members:

embodichain.learning.models.chunk_bc
------------------------------------

.. currentmodule:: embodichain.learning.models.chunk_bc

.. autosummary::

   ChunkBCModelCfg
   ChunkBCModel
   ChunkBCExecutor

.. autoclass:: ChunkBCModelCfg
   :members:

.. autoclass:: ChunkBCModel
   :members:

.. autoclass:: ChunkBCExecutor
   :members:

embodichain.learning.models.resnet_encoder
------------------------------------------

.. currentmodule:: embodichain.learning.models.resnet_encoder

.. autosummary::

   ResNet10Encoder

.. autoclass:: ResNet10Encoder
   :members:

embodichain.learning.sl.bc
--------------------------

.. currentmodule:: embodichain.learning.sl.bc

.. autosummary::

   ChunkBCObjective
   HDF5BCSource
   PrefetchedBCSource

.. autoclass:: ChunkBCObjective
   :members:

.. autoclass:: HDF5BCSource
   :members:

.. autoclass:: PrefetchedBCSource
   :members:
