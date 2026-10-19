embodichain.data_pipeline.datasets
==================================

.. automodule:: embodichain.data_pipeline.datasets

Overview
--------

Datasets and samplers for live simulation and bounded resident numeric data.
:class:`OnlineDataset` consumes trajectory data streamed through the
:mod:`~embodichain.data_pipeline.engine`, while the chunk samplers
(:class:`UniformChunkSampler`, :class:`ChunkSizeSampler`,
:class:`GMMChunkSampler`) carve that stream into training chunks.

   .. rubric:: Classes

   .. autosummary::

      OnlineDataset
      UniformChunkSampler
      ChunkSizeSampler
      GMMChunkSampler
      SequenceSpecCfg
      BatchSource
      OnlineDataSource
      TensorDataSource
      LeRobotFieldMappingCfg
      LeRobotDataSource
      OfflineDataset
      DataBudgetExhaustedError

.. automodule:: embodichain.data_pipeline.datasets
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: embodichain.data_pipeline.datasets.online_data
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: embodichain.data_pipeline.datasets.sampler
   :members:
   :undoc-members:
   :show-inheritance:

Reusable causal training batches
--------------------------------

``BatchSource`` implementations return independently owned TensorDicts with
``obs``, ``actions``, ``valid``, ``loss_mask`` and available source metadata.
``SequenceSpecCfg`` defines history and prediction windows, accepted-segment
boundaries and optional generation filtering. Sampling is episode-uniform;
invalid or discontinuous spans pad rather than become training transitions.

``OnlineDataSource`` preserves the engine's lock, worker checks and shared
delivery quotas. Its caller continues to own start, refill, waits and stop.
``TensorDataSource`` serves explicit resident CPU episodes; ``LeRobotDataSource``
loads numeric columns through the optional official SDK with an explicit
resident-memory limit and schema/column mapping. SDK caches and bounded staging
are separate from that resident tensor limit. Large out-of-core visual loading
is outside this backend's contract.

``OfflineDataset`` supplies item or prebatched PyTorch iteration. Stored
cardinality does not imply an epoch length; set ``max_batches`` for finite
iteration. ``OnlineDataset`` retains its existing chunk interface and accepts
an opt-in ``sequence`` plus explicit ``schema`` for the common context contract.
Quota exhaustion and permanent source failures propagate without hidden retry.

.. automodule:: embodichain.data_pipeline.datasets.sequence
   :members:

.. automodule:: embodichain.data_pipeline.datasets.sources
   :members:

.. automodule:: embodichain.data_pipeline.datasets.lerobot
   :members:

.. automodule:: embodichain.data_pipeline.datasets.offline_data
   :members:

Offline dataset inspection
--------------------------

Validate saved episode/frame metadata, feature values, segment boundaries,
and optional media evidence without constructing a simulator. Quality-aware
split manifests group source episodes and their derived fragments so one
lineage cannot cross train, validation, and test partitions.

See :doc:`/guides/offline_expert_datasets` for CLI and Python examples,
diagnostic reports, quality evidence and split-manifest consumption.

.. automodule:: embodichain.data_pipeline.datasets.inspection
   :members:
