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
| Contact query adaptation and actor metadata | `embodichain/lab/sim/sensors/contact_sensor.py` |
| Fixed contact-buffer scatter | `embodichain/lab/sim/sensors/_warp/contact.py` |

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
before preparation and again after Newton AutoSolver selection.

Componentized Gym deployments select the robot and its sensor suite together
through an embodiment component. The component/inline exclusion rules and
resolver belong to [environment configuration](../env-framework/configuration.md),
not sensor classes. Use
[add-embodiment-component](../../../.agents/skills/add-embodiment-component/SKILL.md)
when changing a reusable mounted sensor suite.

Shared pose and backend contracts belong to
[simulation](../simulation-system/simulation-system.md). Camera offsets are
parent-relative; `parent=None` leaves manager-created views in arena space.
The camera's extrinsics decoder owns look-at versus explicit-pose precedence.

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
For missing contacts, inspect backend capabilities, the query filter and quota
before changing buffer allocation. Pure resolver tests do not require a renderer.
