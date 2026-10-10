embodichain.data_pipeline
=========================

.. automodule:: embodichain.data_pipeline

Overview
--------

Online data streaming, recording, and offline expert-dataset inspection. The
online components are :mod:`~embodichain.data_pipeline.datasets` (online datasets
and samplers that stream trajectories from a running simulation),
:mod:`~embodichain.data_pipeline.engine` (a process-safe shared buffer that
decouples simulation producers from training consumers), and
:mod:`~embodichain.data_pipeline.depth_video` (compressed depth-sidecar
storage for LeRobot datasets on Python 3.10--3.12).

   .. rubric:: Submodules

   .. autosummary::

      datasets
      depth_video
      engine
      recording

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
   :exclude-members: SharedLanguageRegistry
   :undoc-members:
   :show-inheritance:

   .. autosummary::

      data

Recording integrity
-------------------

The offline recording utilities import without the simulator or training
stack. They maintain stable episode identities, configuration fingerprints,
and durable evidence for conservative metadata repair.

.. toctree::
   :maxdepth: 1

   embodichain.data_pipeline.recording
