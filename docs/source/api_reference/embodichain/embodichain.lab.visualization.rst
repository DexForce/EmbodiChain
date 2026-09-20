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
to the browser. The stack supports interactive gizmos and scalar articulation
joint controls (with optional command write-back to the simulation), scene
overlays (targets, trajectories, point clouds), and live RGB camera preview.
CLI helpers
(:func:`add_viser_args_to_parser`, :func:`visualization_cfg_from_args`) wire
the standard ``--viser*`` arguments into launchers.

Render-only marker groups provide reusable primitive or caller-supplied mesh
geometry for native and browser visualization. They do not create physics
bodies or advance simulation. See
:class:`embodichain.lab.visualization.markers.MarkerGroupCfg` and
:class:`embodichain.lab.visualization.markers.MarkerGroup` for creation,
batched updates, visibility, and cleanup.

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

Markers
-------

Groups default to ``scope="env"`` and use ``SimulationManager.num_envs``.
Positions are local to each sub-environment; its origin is added once during
snapshot generation. Use ``scope="world"`` for a single global batch.
There is no ``arena_index`` on ``MarkerGroupCfg``.

``update`` accepts position/scale arrays shaped ``(E, M, 3)``, xyzw quaternion
and RGBA arrays shaped ``(E, M, 4)``, and prototype/visibility arrays shaped
``(E, M)``. With ``env_ids``, ``E`` is the number of selected environments in
that order. A single selected environment also accepts ``(M, ...)`` arrays.
Changing a selected environment's marker count resets its omitted fields;
other environments retain their state. ``count`` is the total instance count,
and ``counts`` reports counts in environment order.

.. code-block:: python

   markers = sim.add_marker_group(MarkerGroupCfg(
       name="targets", prototypes={"sphere": MarkerPrototypeCfg(shape="sphere")}
   ))
   markers.update(translations=positions)  # (sim.num_envs, M, 3)
   markers.update(translations=selected_positions, env_ids=[1, 3])  # (2, M, 3)
   markers.set_visibility(False, env_ids=[1])
   markers.clear(env_ids=[3])

Attach to a registered rigid object or articulation root by UID, or to a robot/
articulation link using ``link_name``. The target must already be prepared.
Existing marker poses become parent-relative offsets; subsequent host updates
and explicit render-state synchronization refresh the transforms without
marker operations stepping physics. Attachment currently requires env scope.

.. code-block:: python

   markers.attach("robot", link_name="tool0")
   markers.attach("box", env_ids=[1])
   markers.detach(env_ids=[1])  # Keep the current world pose by default.

Prototype colors and per-instance overrides are RGBA in [0, 1]. Frame markers
keep their RGB axis colors while following alpha. Native rendering requires
DexSim's generic ``RenderBody`` and ``MaterialInst`` overlay properties.
The adapter creates an ordinary Arena-owned ``MeshObject`` without adding
physics components. It configures overlay routing, disables shadow and picking,
adds the mesh without automatic building, assigns an owned unlit RGBA material,
and then builds and attaches the object. Creation failures remove the incomplete
object. No Spawn factory or descriptor is required.

Unlit and alpha-mode controls require DexSim's shared native RT material type;
Filament materials reject them. Mesh geometry is supplied as arrays. Existing
``MeshObject`` APIs provide pose, scale, color, visibility and removal without
preparing physics.

Hybrid, FastRT and OfflineRT support these overlays. Offscreen output excludes
them by default; explicit camera-group opt-in enables supported targets. Hybrid
NRD offscreen with DLSS disabled inherits the engine's overlay-composition
limitation even with opt-in. This batched state API currently
uses per-object native handles; native batch submission and GPU instancing are
tracked separately in DexSim issue 227.

.. autoclass:: embodichain.lab.visualization.markers.MarkerPrototypeCfg
   :members:
   :undoc-members:
   :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: embodichain.lab.visualization.markers.MarkerGroupCfg
   :members:
   :undoc-members:
   :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: embodichain.lab.visualization.markers.MarkerGroup
   :members:
   :undoc-members:

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

.. autoclass:: JointControlCommandQueue
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

``PickCommand`` identifies a node in a specific simulation run and scene
revision. An empty selection releases only the gizmo created by click picking;
explicitly configured gizmos retain their ownership.

.. autoclass:: PickCommand
   :members:
   :undoc-members:

.. autoclass:: JointControlSpec
   :members:
   :undoc-members:

.. autoclass:: JointControlState
   :members:
   :undoc-members:

.. autoclass:: JointControlCommand
   :members:
   :undoc-members:

.. autoclass:: JointControlProvider
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

.. autoclass:: MeshMarkerOverlay
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
