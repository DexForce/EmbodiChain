embodichain.data_pipeline
=========================

.. automodule:: embodichain.data_pipeline

Overview
--------

Online data streaming and recording for live-simulation training. The package
has three parts: :mod:`~embodichain.data_pipeline.datasets` (online datasets
and samplers that stream trajectories from a running simulation),
:mod:`~embodichain.data_pipeline.engine` (a process-safe shared buffer that
decouples simulation producers from training consumers), and
:mod:`~embodichain.data_pipeline.depth_video` (compressed depth-sidecar
storage for LeRobot datasets on Python 3.10/3.11).

   .. rubric:: Submodules

   .. autosummary::

      datasets
      depth_video
      engine

Datasets
--------

.. automodule:: embodichain.data_pipeline.datasets
   :members:
   :undoc-members:
   :show-inheritance:

   .. autosummary::

      online_data
      sampler

Depth Video
-----------

.. toctree::
   :maxdepth: 1

   embodichain.data_pipeline.depth_video

Online Data Engine
------------------

.. automodule:: embodichain.data_pipeline.engine
   :members:
   :undoc-members:
   :show-inheritance:

   .. autosummary::

      data
