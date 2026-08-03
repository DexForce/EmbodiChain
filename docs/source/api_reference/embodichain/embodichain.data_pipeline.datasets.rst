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
