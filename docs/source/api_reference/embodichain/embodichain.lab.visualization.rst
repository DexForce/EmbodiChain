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

Prototype colors and per-instance overrides are sRGB RGBA in [0, 1]. Frame
markers retain RGB axis colors while following alpha. Mesh prototypes may carry
UV coordinates and a uint8 RGB/RGBA texture, with repeat, clamp or mirror
sampling. Geometry is copied and validated once per prototype, then shared as
read-only arrays by its instance snapshots.

Native markers use Scene-owned visual objects, shared geometry and independent
material state. Batch updates preserve selected-environment ordering without
stepping physics. Offscreen cameras exclude these visuals by default; opt in
with ``SpawnedCamera.set_visual_overlay_enabled(True)`` for diagnostic capture.
Appearance depends on the renderer. Transparent overlap and lighting can differ
between native and browser visualization.

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

Visualization markers and robot visuals
---------------------------------------

``VisualizationMarkers`` adapts flat Isaac-style arrays to ``MarkerGroup``.
Its orientations are **xyzw**, matching Isaac Lab 3.0 and EmbodiChain. Marker
names determine prototype indices. ``environment_ids`` explicitly assigns each
instance to an environment; omitted IDs place every instance in environment 0.
The default coordinate space is world. Use ``coordinate_space="env"`` to add
environment origins. The adapter never infers environment membership from count.

``RobotVisual`` reads URDF visuals or one selected USD articulation/visual
object. Multiple visuals per link, visual origins, mesh scales and material
parts are preserved. ``visualize_link_poses`` accepts ``(E, L, 7)`` xyz+xyzw or
``(E, L, 4, 4)`` matrices; inserting ``K`` after ``E`` displays K references in
each environment. ``link_names`` gives the required link order. ``env_ids``
selects rows to update while other environments retain their references.

USD sources can use ``source=UsdSourceCfg(file_path=..., source_prim_path=...)``,
sharing source selection and package texture resolution with ``VisualObject``.
Use either ``source`` or ``asset_path``/``prim_path``. URDF inputs use
``asset_path``. A ghost owns extracted package textures until removal; ordinary
USD texture files are referenced directly.

URDF sources also provide ``visualize_qpos`` and ``joint_names``. Callers supply
every joint in that order, including mimic joints; FK does not expand mimic
relations. USD sources use external link poses. Motion loading, time sampling,
root alignment and target selection remain in the task.

``RobotVisualAppearanceCfg(mode="override")`` replaces the base-color factor while
retaining source textures, roughness, metallic, normal maps and other imported
material parameters. ``mode="asset"`` retains the original factor and multiplies
source alpha by ``opacity``. Texture RGB still modulates the new factor: dark
painted details remain dark when the factor is green.

Opaque robot visuals preserve imported materials and respond to scene lighting.
Normals, material UV coordinates and extracted texture files remain available
for the visual's lifetime. Transparency is adjustable, but translucent shading
can be simpler than opaque shading and does not promise the same result across
renderers. ``RobotVisualAppearanceCfg(unlit=True)`` selects a uniform color;
general marker prototypes are unlit by default.

``set_appearance`` changes the ghost's independent material state without editing
the source asset. ``set_visibility`` retains instances; ``remove`` releases them.

.. code-block:: python

   from embodichain.lab.visualization.markers import (
       RobotVisualAppearanceCfg, RobotVisual, RobotVisualCfg,
   )

   sim.prepare()
   ghost = RobotVisual(RobotVisualCfg(
       name="reference", asset_path=robot_urdf,
       coordinate_space="env",
       appearance=RobotVisualAppearanceCfg(mode="asset", opacity=0.4),
   ), sim)
   ghost.visualize_qpos(reference_qpos, root_poses=reference_root_poses)
   with sim.render_frame():
       pass
   ghost.remove()

The executable example ``examples/sim/robot_visual.py`` accepts ``--asset``,
``--layers``, ``--mode``, ``--opacity`` and ``--viser``. Its generated reference
motion does not advance physics. ``--headless --output-dir`` saves native images
and separates creation, update/submission and camera-render timings.

Recorded task trajectories
--------------------------

The trajectory example defaults to ``--camera-mode follow_xy``. It fits the
whole clip once and follows the first trajectory's root in world X/Y, keeping
world height, orbit orientation and distance fixed. Use ``--camera-mode fixed``
for a stationary camera. Live mouse orbit adjustments are preserved while
tracking. This follows ``PolicyViewerCameraCfg`` ground-plane tracking semantics.

``examples/sim/reference_trajectory.py`` displays recorded policy link poses
through ``RobotVisual``. It enables ghost visuals, path markers and optional
``VisualObject`` task cubes by default. The example does not step physics; a task
or policy evaluator owns state sampling and checkpoint execution.

The NPZ contains ``poses`` shaped ``(T, K, L, 7)`` in world xyz+xyzw order,
``link_names`` shaped ``(L,)`` and a positive scalar ``dt`` in seconds. Each
trajectory has its own link poses. Optional fields are end-effector ``tips``
``(T,K,3)``, task ``objects`` ``(T,K,7)``, ``targets`` ``(T,K,3)``, and body-frame
``command``/``velocity`` ``(T,3)``. Unequal source durations should be sampled onto
one time grid by the task; holding the final state is suitable for completed
manipulation episodes.

``--layout overlap`` keeps common robot origins. ``--layout offset --spacing 1.1``
adds a Y offset to each complete trajectory, its object, target and path. This
preserves the spatial relationship within each task. ``--original-first`` uses
source appearance for the first trajectory; other trajectories use colored
translucent materials with ``--opacity``. ``--live`` opens a bounded native
viewer, while the default writes H.264 video, PNG frames and capture metadata to
``--output``. ``--seconds`` and ``--fps`` set duration and sampling rate.

Robot visuals and task objects use the existing scene ownership and pose APIs.
Regular task objects participate in camera images; robot visuals require
explicit camera visualization opt-in.

.. autoclass:: embodichain.lab.visualization.markers.VisualizationMarkersCfg
   :members:
   :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: embodichain.lab.visualization.markers.VisualizationMarkers
   :members:
   :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: embodichain.lab.visualization.markers.RobotVisualAppearanceCfg
   :members:
   :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: embodichain.lab.visualization.markers.RobotVisualCfg
   :members:
   :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: embodichain.lab.visualization.markers.RobotVisual
   :members:
   :exclude-members: __init__, copy, replace, to_dict, validate

Implementation modules expose the same contracts:

.. autosummary::

   markers.visualization_markers.VisualizationMarkers
   markers.visualization_markers.VisualizationMarkersCfg
   markers.robot_visual.RobotVisualAppearanceCfg
   markers.robot_visual.RobotVisualCfg
   markers.robot_visual.RobotVisual

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
