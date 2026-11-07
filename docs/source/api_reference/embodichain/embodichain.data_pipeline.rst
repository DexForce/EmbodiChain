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
      runtime

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
   :no-index:
   :exclude-members: SharedLanguageRegistry
   :undoc-members:
   :show-inheritance:

   .. autosummary::

      data

Owner-side flow control
-----------------------

The application starts and stops its manual-refill engine. The optional
controller applies a measured plan, drains publication events, checks health,
and bounds generation/data/quota waits. A plan or production limit requires an
explicit producer count dividing cache size. Sources and Datasets retain
immediate sampling/error behavior. See :doc:`/features/ods/configuration` for
matching engine settings, callback clocks and an integration example.

.. automodule:: embodichain.data_pipeline.runtime

.. autosummary::

   embodichain.data_pipeline.runtime.ODSFlowControllerCfg
   embodichain.data_pipeline.runtime.ODSFlowController

.. autoclass:: embodichain.data_pipeline.runtime.ODSFlowControllerCfg
   :members:
   :show-inheritance:

.. autoclass:: embodichain.data_pipeline.runtime.ODSFlowController
   :members:
   :show-inheritance:

Recording integrity
-------------------

The offline recording utilities import without the simulator or training
stack. They maintain stable episode identities, configuration fingerprints,
and durable evidence for conservative metadata repair.

The usage guide :doc:`/guides/offline_expert_datasets` covers validation,
quality splits and interrupted commits. :doc:`/guides/lerobot_export` describes
the independent Python 3.12 converter for official LeRobot 0.6.1 language/depth.

.. toctree::
   :maxdepth: 1

   embodichain.data_pipeline.recording
