embodichain.data_pipeline.datasets
==================================

.. automodule:: embodichain.data_pipeline.datasets

Overview
--------

Datasets and samplers for online streaming training from live simulation.
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

Offline dataset inspection
--------------------------

Validate saved episode/frame metadata, feature values, segment boundaries,
and optional media evidence without constructing a simulator. Quality-aware
split manifests group source episodes and their derived fragments so one
lineage cannot cross train, validation, and test partitions.

.. automodule:: embodichain.data_pipeline.datasets.inspection
   :members:
