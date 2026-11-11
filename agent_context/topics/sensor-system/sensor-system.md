# Sensor System

## Find the owner

| Change or question | Start here |
|---|---|
| Sensor config decoding and buffer interface | `embodichain/lab/sim/sensors/base_sensor.py` → `SensorCfg`, `BaseSensor` |
| Exported sensor/config names | `embodichain/lab/sim/sensors/__init__.py` |
| Image configuration, data flags, extrinsics | `embodichain/lab/sim/sensors/camera.py` → `CameraCfg`, `Camera` |
| Two-eye transforms and attachment | `embodichain/lab/sim/sensors/stereo.py` → `StereoCameraCfg`, `StereoCamera` |
| Parent-name resolution | `embodichain/lab/sim/sensors/attachment.py` → `resolve_parent_nodes` |
| Creation, preparation and attachment coordination | `embodichain/lab/sim/sim_manager.py` → `add_sensor`, `prepare` |
| Contact query adaptation and actor metadata | `embodichain/lab/sim/sensors/contact/sensor.py` |
| Joint and contact acquisition windows | `embodichain/lab/sim/sensors/contact/observation.py` |
| Native support observations and pair ownership | `embodichain/lab/sim/sensors/support_observation.py` |
| Fixed contact-buffer scatter | `embodichain/lab/sim/sensors/contact/_warp/scatter.py` |

Sensors inherit `BaseSensor`/`BatchEntity` and expose per-environment data through
a TensorDict buffer. `SensorCfg.from_dict()` resolves `sensor_type + "Cfg"`
from the sensor registry, so names are case-sensitive. Read concrete config
classes for image dimensions, enabled outputs, intrinsics and filter defaults;
these are source-owned values.

## Ownership and resolution

Create sensors through `SimulationManager.add_sensor()`. The explicit manager
owns the World, ordered Arenas, preparation and semantic parent resolution;
sensors must not rediscover a singleton manager. Cameras are render features
on both physics backends. Contact availability is a backend capability checked
before preparation and again after DexUni's scene-based dispatch.

Camera/StereoCamera intrinsics use Scene camera batches, preserving CPU/CUDA
input selection order, negative indices and last-write-wins duplicate rows.
Pose reads/writes, look-at and extrinsics reset retain
their per-instance native paths. Shared image targets already batch their view
buffers; ordered publication keeps preceding parameter writes visible to the
next photograph. Stereo calibration and logical center poses remain sensor-owned.

Robot-mounted cameras are declared in `RobotCfg.sensors` with robot-local
parent links. `SimulationManager.add_robot()` initializes them only when the
robot's sensor switch is enabled, qualifies parents using the final robot UID,
and registers them through `add_sensor()`. Sensor UIDs remain scene-wide;
multiple robots need distinct sensor UIDs. Camera creation and rebuilds retain
the same manager-owned lifecycle. Removing a robot destroys its automatically
initialized cameras before rebuilding the physical scene.

Componentized Gym deployments select the robot and independent sensors through
an embodiment component. The component/inline exclusion rules, Gym acquisition
switch and effective suite belong to
[environment configuration](../env-framework/configuration.md), not sensor
classes. Use
[add-embodiment-component](../../../.agents/skills/add-embodiment-component/SKILL.md)
when changing embodiment selection or independent sensor declarations; change
the robot configuration for a reusable mounted camera suite.

Shared pose and backend contracts belong to
[simulation](../simulation-system/simulation-system.md). Camera offsets are
parent-relative; `parent=None` leaves manager-created views in arena space.
The camera's extrinsics decoder owns look-at versus explicit-pose precedence.
For explicit camera poses, `Camera.reset()` passes parent-mounted transforms
to the native OpenGL view unchanged (-Z forward, +Y up). Unparented explicit
poses are converted from optical axes (+Z forward, +Y down) by flipping Y/Z.
Convert optical-frame mount calibration before storing a parented `pos/quat`;
do not infer image direction from `get_arena_pose()` without this conversion.

For Gaussian scenes, `Camera.capture_metadata` retains completed-frame identity,
calibration and output semantics separately from image tensors. Gym exposes the
snapshot through `sensor_capture_metadata`. Capability checks and publication
ordering follow [Gaussian composition](../simulation-system/rendering.md#gaussian-composition).
USD-native Gaussian PLY references use the same capture-status path; source prim
and local visual transform appear in `gaussian_instances`. The separate
`get_capture_metadata()` method also includes the ground visibility policy.

## Camera ground visibility

`CameraCfg.ground_plane_visible` and `Camera.set_ground_plane_visibility(...,
env_ids=...)` filter the manager-owned shared Mesh ground per camera instance;
stereo applies the same policy to both eyes. The global manager switch remains
a master gate. Collision, picking, Viser grids and floor embedded in scanned
assets are separate. The manager reserves category bit 1 for its default plane.

Filtering requires the DexSim Mesh-category/camera-exclusion APIs; the default
visible configuration remains usable with older SDKs. Camera reset restores the
configured policy for selected environments. Parent reattachment resets only
extrinsics, preserving runtime visibility through topology changes. The ground
category belongs to the Spawn descriptor, so rematerialization retains it.

`render_camera_group` snapshots policy after rendering; `Camera.update` publishes
that metadata only after successful image reads. Direct camera updates snapshot
their own render. `get_capture_metadata()` returns a copy, separately from image
TensorDict values. A policy change invalidates the camera's published metadata;
a failed render/read cannot be presented as a new successful capture. Callers
using `fetch_only=True` should render through the manager to obtain metadata.

## Camera attachment boundary

`resolve_parent_nodes()` parses a canonical link name or
`<asset_uid>/<link_name>`, disambiguates registered assets, and checks instance
counts. It queries public `Articulation.get_link_render_nodes()`; neither the
resolver nor camera should inspect private entity lists, guess clone suffixes,
or search global native nodes.

Cameras attach only to concrete per-environment render nodes and report success
after applying parent-relative extrinsics. Stereo attachment covers both eyes.
Direct camera construction requires explicit attachment. A manager topology
rebuild must detach old views before native skeletons are removed, then resolve
and attach the new nodes after binding. See
[sensor lifecycle contracts](lifecycle.md#camera-rebuilds) when changing rebuild
ordering or supporting another parent type.

## Contact boundary

`ContactSensor` consumes one Spawn `ContactQuery` after manager preparation.
It uses arena-frame positions, explicit environment IDs and a per-environment
quota; native actor ordering does not determine environment assignment.
`user_ids` are query actor identities, resolved through `get_actor_info()`;
render user IDs are not interchangeable. Consumers must honor `is_valid` and
per-environment counts because unused fixed-buffer values are unspecified.

For substep histories and control-interval accumulation, read
[contact history](contact-history.md).

Read [contact lifecycle and capabilities](lifecycle.md#contact-queries) before
changing filtering, force assumptions, topology handling or actor metadata.
Geometry-only contacts and multiple backend-emitted rows per actor pair are
valid; sensor adaptation does not synthesize a common contact manifold.

The contact scatter kernel remains sensor-owned because its columns, IDs and
quota are integration contracts. Generic image tiling belongs to
`embodichain/compute/image/_warp/tiling.py`; compatibility exports under
`utils.warp.kernels` do not change those owners.

## Focused validation

| Changed boundary | Existing coverage |
|---|---|
| Parent parsing and instance validation | `tests/sim/sensors/test_attachment.py` |
| Native render-node query | `tests/sim/objects/test_articulation.py` |
| Camera/stereo attachment and buffers | `tests/sim/sensors/test_camera.py`, `test_stereo.py` in that directory |
| Manager coordination and rebuilds | `tests/sim/test_sim_manager.py` |
| Query IDs, frames, quota and scatter | `tests/sim/sensors/test_contact_query_sensor.py`, `test_contact_kernels.py` in that directory |
| Backend contact behavior | `tests/sim/sensors/test_contact_default_e2e.py`, `test_contact_newton_e2e.py` in that directory |

For missing camera output, inspect enabled fields and `ViewFlags` together.
Camera and stereo images exclude visual markers by default. Set
`include_visual_overlays=True` for an explicit preview camera.
For missing contacts, inspect backend capabilities, the query filter and quota
before changing buffer allocation. Pure resolver tests do not require a renderer.
