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
      SequenceDataset
      DataBudgetExhaustedError
      DataUnavailableError

   .. rubric:: Schema utilities

   .. autosummary::

      validate_data_schema
      data_schema_fingerprint

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

``SequenceDataset`` supplies common item or prebatched PyTorch iteration for
live and resident sources. ``OfflineDataset`` is its compatibility alias. Stored
cardinality does not imply an epoch length; set ``max_batches`` for finite
iteration. ``OnlineDataset`` retains its existing chunk interface and accepts
an opt-in ``sequence`` plus explicit ``schema`` for the common context contract.
It delegates context iteration to ``SequenceDataset``. Quota exhaustion,
``DataUnavailableError`` and permanent source failures propagate without hidden
retry. Production, bounded waiting and refill remain caller-owned.

All source constructors use ``validate_data_schema``: action semantics (or
representation) and a positive control period are required. Optional observation
timing must be pre-action. Feature names, dtypes and declared versions are
validated consistently. ``data_schema_fingerprint`` hashes finite JSON schema
metadata, including ordered names and supplied transformation/normalization
versions; it does not identify trajectory content or infer missing semantics.

.. automodule:: embodichain.data_pipeline.datasets.sequence
   :members:

.. automodule:: embodichain.data_pipeline.datasets.sources
   :members:

.. automodule:: embodichain.data_pipeline.datasets.lerobot
   :members:

.. automodule:: embodichain.data_pipeline.datasets.offline_data
   :members:

.. automodule:: embodichain.data_pipeline.datasets.sequence_data
   :members:

.. automodule:: embodichain.data_pipeline.datasets.schema
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
