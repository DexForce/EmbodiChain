# Simulation System

## Entry Points

| Responsibility | Source |
|---|---|
| Public simulation facade | `embodichain/lab/sim/__init__.py` |
| World, registries, readiness and stepping | `embodichain/lab/sim/sim_manager.py` → `SimulationManager`, `SimulationManagerCfg` |
| Declaration, source resolution and binding | `embodichain/lab/sim/spawn/scene.py`, `spawn/source.py`, `spawn/descriptors.py` |
| Fresh static USD reads and owned authored link meshes | `embodichain/lab/sim/spawn/source_geometry.py` |
| Manager-level backend hooks and capabilities | `embodichain/lab/sim/physics/` |
| Object-level Scene batch adapters | `embodichain/lab/sim/objects/backends/scene.py`, `newton.py` |
| Object, renderer and physics configuration | `embodichain/lab/sim/cfg/` (`cfg/__init__.py` is the public facade) |
| Visual material configuration, templates and native instances | `embodichain/lab/sim/material.py`; [material identity and lifetime](materials.md) |
| Gym lifecycle and task scene construction | `embodichain/lab/gym/envs/base_env.py`, `embodied_env.py` |

`embodichain.cli.sim` owns standalone simulation arguments without importing
runtime dependencies. Gym launchers compose configuration separately; follow
[environment configuration](../env-framework/configuration.md) for file-owned
physics selection and overrides. Numeric defaults and example recipes belong
to their source configs and callers.

## Ownership

`SimulationManager` owns a DexSim `World`, a `SpawnScene`, resource registries,
and readiness for each committed topology revision. DexSim `SceneBuilder` and
`Scene` own descriptor revisions, native materialization, replicated arenas,
and backend handles. `add_*()` returns a declared facade; `prepare()` binds
that same object in place. UIDs identify resources in the manager registry.

`VisualObject` owns render-only root poses, visibility and material assignments,
with no rigid-body buffers or collision API. Its config lives in
`cfg/visual_object.py`; mesh files use `MeshCfg.fpath`, as for rigid objects.
`cfg/source.py:UsdSourceCfg` owns the source file and exact `source_prim_path`;
visual, rigid and articulation configs accept it through `source`, exclusively
with their existing shape/file input. `ObjectBaseCfg` keeps runtime identity
and initial pose independent of file formats. `add_visual_object()` compiles mesh/primitive appearance or
selects a USD object's appearance. Whole-scene `add_usd()`
preserves physical types and wraps only objects without physics and colliders
as visuals. Conflicting visual UIDs receive the scene-name prefix while returned
keys preserve source prim paths. Other import UID conflicts fail before declaration.
Imported facades retain their source address in `cfg.source`. This metadata
does not request a second spawn or bind an unrelated existing scene entity.
Visuals retain facade-owned pose/visibility/material-assignment state across native
topology rebuilds, with separate reset baselines. They use the same prepare,
replication, selective reset, removal and teardown owners as physical assets. They do not enter physical scene snapshots or planners.
Pose I/O uses pure-visual `World.fetch_object_transform` / `apply_object_transform`
tensor batches through the private Scene visual adapter. That boundary converts
world matrices to arena-local poses and refreshes parent transforms and native
IDs after topology changes.

Explicit `visuals` replace root/body appearance; `link_visuals` replaces named
articulation links. `cfg/visuals.py` owns the resource and element configs;
`spawn/_visuals.py` compiles companion Mesh/Gaussian descriptors under the same
SpawnScene and publishes their owner-relative transforms at the existing render
boundary. Default physics is currently required. Source physics is retained;
`RigidObject.get_semantic_mesh()` preserves its original surface or an explicit
`semantic_shape` before source render meshes are cleared. Do not use visual
vertices as an implicit semantic fallback for Gaussian-only appearance.
The qualified DexSim integration build must supply `Scene.clear_visual_meshes`
and shared Gaussian template replication; missing APIs fail before creation.
`BatchEntity.set_gaussian_visual_state` stages per-element/per-environment gains,
uniform geometry scale, visibility and LoD in the binding; reset baselines stay
independent from current values across rebuilds. Capture metadata owns copies of
that state. Flat-PLY point opacity/SH overrides create immutable resource variants
at birth; selecting prepared elements changes appearance without uploading point
arrays each frame. Variants remain separately resident and share only within
their clone group; modifying points with an existing LoD sidecar is rejected.
DexSim's native USD loader accepts a visual Xform referencing an external
Gaussian PLY beside independently authored collision geometry. `add_usd()` and
`source=UsdSourceCfg(...)` retain the USD owner, source prim and local visual
offset. This path is qualified in one Arena with Default physics and Hybrid
rendering; native Arena replication of USD Gaussian renders is rejected before
declaration. Existing explicit Gaussian visuals remain the multi-Arena route.
`GaussianSplatCfg.file_path` itself still accepts only PLY/GSQ. Standard USD
ParticleField import and native Gaussian source-appearance replacement are not
implied by this reference route. NoRender drops the Gaussian render descriptor
before native materialization while retaining the USD body, collision, source
configuration and typed authored-visual metadata across Arenas. A Gaussian-only
USD needs `semantic_shape` for semantic mesh consumers.

`objects/backends/collision.py` owns physical collision descriptors and strict
cooked convex/box snapshots through `capture_physical_collision_geometry`.
Strict snapshots preserve float64 geometry and requested native body offsets;
ordinary descriptor getters keep their existing precision and shape support.
[Scene services](../scene-services/scene-services.md) assemble observed frames and
native ownership around those values without duplicating SDK geometry reads.

`spawn/_usd_assets.py` materializes package textures into `SpawnScene`-owned
files. They survive topology rebuilds and are deleted after native Scene closure;
facade source configs retain the original package address.

`BaseEnv` owns Gym reset/step timing; `EmbodiedEnv` translates task composition
into manager declarations. Scene replication and collision isolation belong
to Spawn, not per-object collision-group calculations. Backend differences
belong in physics hooks/capabilities or object adapters, not task-side native
access. Contact sensors use the finalized Scene's `ContactQuery` boundary.

`spawn.source_geometry` owns `UsdSourceSnapshot`, `read_usd_source` and
`LinkMesh` for Object source adapters and static Scene consumers. Reads use fresh
anonymous USD layers and recheck captured bytes. Explicit snapshots can be
shared within one assembly; there is no process-wide source cache. Polygon
inputs and fan-triangulated owner-local meshes preserve order and apply the
deployment scale once. Source meshes retain their input identity and do not
certify native cooked shapes. Current poses, effective joint limits and physical
collision shapes remain Object/Articulation queries; control-role interpretation
and stricter surface qualification remain Scene services.

[Scene services](../scene-services/scene-services.md) compose existing configs,
propose layouts and read/change physical state through these owners. They do
not add a registry or commit topology. Runtime scene services currently accept
only B=1 selections; Spawn still owns replication and isolation.

## Lifecycle

```text
construct manager and World
  → declare initial scene and optional Newton runtime controls
  → prepare: resolve/configure source → commit → prepare runtime → bind
             → publish render state → attach cameras → mark ready
  → metadata-dependent setup
  → update: explicit physics steps and state publication
  → reset selected environments
  → destroy: deferred native cleanup
```

Default can materialize handles during declaration; Newton defers physical
materialization. Both require the same `prepare()` boundary before metadata,
state, or physics use. `BaseEnv` supplies it between scene setup and dependent
initialization; standalone callers supply it explicitly. `update()` also
prepares defensively. See [lifecycle](lifecycle.md) for retries, replacement,
reset ordering, runtime controls and teardown.

`simulation_time` is a read-only manager-relative clock: successful `update()`
substeps add their actual durations before `after_substep` callbacks. Explicit
warmup updates count; preparation/render synchronization at zero duration does
not advance time, and object/environment reset does not rewind it. A fresh
manager starts at zero. Episode-relative Task Program clocks can have an offset;
physical evidence must validate that mapping instead of mixing timestamp domains.

`physics_step_index` counts completed manager substeps before each
`after_substep` callback, including runs without rendering. Reads, selected
resets and topology rebuilds leave it unchanged. Consumers that cache
topology-dependent state must also validate their binding, not only this count.

## Quaternion and pose convention

EmbodiChain domain-facing quaternions are `(x, y, z, w)` (`xyzw`); public 7D
poses are `xyz + xyzw`, with identity quaternion `(0, 0, 0, 1)`. This covers
object/link/COM state, FK/IK, sensor offsets, task configuration and math.

Convert exactly once at external boundaries. DexSim Scene pose buffers use
`xyzw + xyz` (layout permutation only). DexSim Spawn mass descriptors store a
body-frame mat33 and COM position; EmbodiChain composes principal moments and
`xyzw` orientation before that boundary. Legacy `PhysicalAttr` retains `wxyz`
and requires explicit quaternion conversion.
Newton/Warp transforms already use position plus `xyzw`. Explicit external
formats such as visualization `*wxyz`, cuRobo poses and GenSim
`rotation_quaternion_wxyz` must be converted before EmbodiChain math. Adapter
tests need non-symmetric rotations to expose component-order mistakes.

## Invariants

- Configure arenas, devices, renderer and backend before manager construction.
- Keep `prepare()` convergent and retryable; a committed scene is not ready
  until binding, render publication and camera attachment all succeed.
- Physics advances through `update()` only. Rendering, diagnostics and marker
  publication must not step or finalize an empty standalone scene.
- Keep physics timesteps fixed during trajectory playback; retime planner
  samples to the command grid. See [motion execution](../motion-planning/motion-planning.md).
- Use Scene batch adapters for state selection. Preserve untouched rows/DOFs
  and keep articulation control selections on their tensor device.
- Keep public articulation state order distinct from source traversal order;
  map by joint name. Configure source overlays before Newton's first build.
- Preserve resolved initialization snapshots during runtime property writes;
  reset only selected environments and honor supported exclusions.
- Declare deformables and Newton controls before finalization. Deformables
  require Newton, CUDA and a supported particle solver without gradient mode.
- Flush deferred cleanup after caller-owned scene locals unwind when using
  `destroy(exit_process=False)`; release child views before native parents.

## Where to Make Changes

| Request | Read next / change owner |
|---|---|
| Preparation, topology changes, reset, cleanup, controls | [Lifecycle](lifecycle.md); manager and `spawn/scene.py` |
| Backend/device selection, source overlays, collision/mass policy | [Physics configuration](physics-configuration.md); `cfg/`, `spawn/descriptors.py` |
| Joint state order, drive lowering, mimic coupling, native adaptation | [Articulation and batch adapters](articulation-adapters.md); `objects/backends/`, `objects/articulation.py` |
| Volume/surface state, topology and kinematic nodes | [Deformables](deformables.md); `cfg/deformable.py`, `objects/deformable/` |
| Camera synchronization, native window/offscreen denoising, startup readiness | [Rendering and diagnostics](rendering.md); manager, `cfg/simulation.py`, `_startup_summary.py` |
| Robot definition, control parts, robot presets | [Robot system](../robot-system/robot-system.md) |
| Camera attachment resolution or contact sensors | [Sensor system](../sensor-system/sensor-system.md) |
| Browser export or native gizmos | [Visualization](../sim-visualization/sim-visualization.md), [native gizmos](../sim-visualization/native-gizmos.md) |
| Solvers, planners, workspace, augmentation | [IK](../ik-solvers/ik-solvers.md), [motion](../motion-planning/motion-planning.md), [workspace](../robot-workspace/robot-workspace.md) |
| Semantic geometry and action execution | [Atomic Actions](../atomic-actions/atomic-actions.md), [Task Programs](../task-programs/task-programs.md) |

`motion/__init__.py` and workspace analyzer exports remain lazy: Robot imports
solver/workspace types while planners resolve robots through the manager.
Specialized APIs belong to their subpackages. Augmentation owns generation
bookkeeping; host integrations own execution, reset and persistence.

## Common Failure Modes

| Symptom | First check |
|---|---|
| Missing metadata or stale state after declaration/mutation | Successful `prepare()` for the current topology revision |
| Newton Viser shows only a grid or non-finite link poses | Empty visualization startup must not finalize the scene |
| Wrong execution or render GPU | Compute `device` and renderer `gpu_id` are distinct; inspect resolved configuration |
| Unexpected control rate | Gym control/physics ratio and fixed command grid |
| Reset changes untouched worlds or loses physical baseline | Batch selections, immutable `default_*` snapshots and complete Newton articulation clear |
| Native leaks/crash during teardown | Deferred queue flushing and child-before-parent release order |

Focused validation is linked with each detail page. The primary lifecycle
coverage is `tests/sim/test_sim_manager.py` and `tests/sim/spawn/test_scene.py`;
configuration and adapter coverage should be selected for the changed contract.
