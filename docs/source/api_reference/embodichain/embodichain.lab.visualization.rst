embodichain.lab.visualization
=============================

.. automodule:: embodichain.lab.visualization

Overview
--------

Browser-based visualization of simulation scenes. The
:class:`SceneExporter` reads simulation assets on the simulation thread and
produces detached, backend-neutral CPU snapshots (:class:`SceneManifest`,
:class:`SceneFrame`, :class:`CameraImageFrame`). A background
:class:`VisualizationRuntime` owns latest-frame queues, rate limiting, health,
and telemetry, and pushes snapshots to a visualization backend - currently the
Viser server (:class:`ViserServerCfg`) that publishes an interactive 3D view
to the browser. The stack supports interactive gizmos (with optional command
write-back to the simulation), scene overlays (targets, trajectories, point
clouds), and live RGB camera preview. CLI helpers
(:func:`add_viser_args_to_parser`, :func:`visualization_cfg_from_args`) wire
the standard ``--viser*`` arguments into launchers.

Configuration
-------------

.. autoclass:: VisualizationCfg
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: ViserServerCfg
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict, validate

Runtime
-------

.. autoclass:: VisualizationRuntime
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: LatestFrameQueue
   :members:
   :undoc-members:

.. autoclass:: GizmoCommandQueue
   :members:
   :undoc-members:

.. autoclass:: RuntimeHealth
   :members:
   :undoc-members:

.. autoclass:: RuntimeStats
   :members:
   :undoc-members:

Scene Export
------------

.. autoclass:: SceneExporter
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: SceneManifest
   :members:
   :undoc-members:

.. autoclass:: SceneFrame
   :members:
   :undoc-members:

.. autoclass:: SceneNode
   :members:
   :undoc-members:

.. autoclass:: MeshGeometry
   :members:
   :undoc-members:

.. autoclass:: DynamicMeshUpdate
   :members:
   :undoc-members:

.. autoclass:: GizmoSpec
   :members:
   :undoc-members:

.. autoclass:: GizmoState
   :members:
   :undoc-members:

.. autoclass:: GizmoCommand
   :members:
   :undoc-members:

.. autoclass:: CaptureResult
   :members:
   :undoc-members:

Overlays and Camera Preview
---------------------------

.. autoclass:: SceneOverlays
   :members:
   :undoc-members:

.. autoclass:: FrameOverlay
   :members:
   :undoc-members:

.. autoclass:: TargetOverlay
   :members:
   :undoc-members:

.. autoclass:: TrajectoryOverlay
   :members:
   :undoc-members:

.. autoclass:: PointCloudOverlay
   :members:
   :undoc-members:

.. autoclass:: CameraSpec
   :members:
   :undoc-members:

.. autoclass:: CameraImage
   :members:
   :undoc-members:

.. autoclass:: CameraImageFrame
   :members:
   :undoc-members:

.. autoclass:: CameraImageCaptureResult
   :members:
   :undoc-members:

Pose Conversion
---------------

.. autofunction:: pose_to_position_wxyz

Command-Line Helpers
--------------------

.. autofunction:: add_viser_args_to_parser

.. autofunction:: visualization_cfg_from_args
