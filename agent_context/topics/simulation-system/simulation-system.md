# Simulation System

## Entry Points

| Responsibility | Source |
|---|---|
| Public simulation facade | `embodichain/lab/sim/__init__.py` |
| World, registries, readiness and stepping | `embodichain/lab/sim/sim_manager.py` → `SimulationManager`, `SimulationManagerCfg` |
| Declaration, source resolution and binding | `embodichain/lab/sim/spawn/scene.py`, `spawn/source.py`, `spawn/descriptors.py` |
| USD import recipes, preflight and source-path results | `cfg/usd.py`, `spawn/usd_import.py`, `spawn/usd_result.py` |
| Fresh static USD reads and owned authored link meshes | `embodichain/lab/sim/spawn/source_geometry.py` |
| Manager-level backend hooks and capabilities | `embodichain/lab/sim/physics/` |
| Object-level Scene batch adapters | `embodichain/lab/sim/objects/backends/scene.py`, `newton.py` |
| Object, renderer and physics configuration | `embodichain/lab/sim/cfg/` (`cfg/__init__.py` is the public facade) |
| Visual material configuration, templates and native instances | `embodichain/lab/sim/material.py`; [material identity and lifetime](materials.md) |
| Gym lifecycle and task scene construction | `embodichain/lab/gym/envs/base_env.py`, `embodied_env.py` |

`cli.sim` supplies standalone arguments without importing runtime dependencies.
[Environment configuration](../env-framework/configuration.md) owns runnable
file/CLI composition and physics selection. Concrete defaults stay in configs.

## Ownership

`SimulationManager` owns a World, SpawnScene, registries and readiness per
committed topology revision. DexSim SceneBuilder/Scene owns descriptors, native
materialization, arenas and backend handles. `add_*` returns declared facades;
`prepare()` binds those same objects in place. UIDs are manager-scoped.

Backend differences belong in physics hooks or object adapters; tasks do not
perform native access. Spawn owns replication/isolation and source overlays.
Contact sensors use the finalized Scene query boundary. Cameras/lights use the
same Scene, including its pre-finalization prepared-Arena phase.

`VisualObject` owns appearance, visibility and pure-visual poses without physics
buffers or collision API. Source selectors are exclusive with ordinary file/
shape inputs. Explicit `visuals` and articulation `link_visuals` replace appearance
through `cfg/visuals.py` and `spawn/_visuals.py`, retaining source physics and an
independent semantic mesh. Single Gaussian resources normalize to an identity
element; multi-element composition keeps explicit IDs. Gaussian admission,
native USD reference limits and capture contracts are in [rendering](rendering.md).
Whole-scene USD imports own namespaces, placement and sparse
source-path overrides; singleton placement remains asset-owned. Returned source
identities do not request a second spawn. Imported visuals/deformables keep their
physical type and supported lifecycle. Single-articulation USD imports via
`add_robot()`/`add_articulation()` preserve constraints between selected-prim
links. `spawn/usd.py` maps endpoints/names to the runtime UID while retaining
body-local frames and scale. Declare constrained assets before prepare on
Default; cross-object/world constraints require `add_usd()`.
Source, texture and visual-rebuild rules
are in [articulation adapters](articulation-adapters.md),
[rendering](rendering.md) and [scene geometry](../scene-services/geometry.md).

`spawn.source_geometry` owns fresh authored reads and `LinkMesh`, with optional
per-assembly snapshots and no global source cache. Captured identity/order and
one-time scale survive conversion. Authored source geometry never certifies
native cooked shapes; collision adapters own strict physical reads. Scene
services assemble explicit geometry/state without adding registries or topology.
Their batching is API-specific; physical views support explicit ordered rows,
while native capture/query adapters retain their declared B1 limits.
Whole-scene inspection accepts `manager=None` without a World or runtime facades;
the import plan retains authored visual metadata for runtime tracking, including
NoRender imports that omit native Gaussian resources.

## Lifecycle

```text
construct → declare source/assets/controls → prepare and bind
          → publish render state and attach cameras → metadata-dependent setup
          → explicit update substeps → selected reset → deferred teardown
```

Default may materialize handles during declaration; Newton defers physical
materialization. Both require successful prepare for the current revision before
metadata/state/physics use. Preparation stays convergent/retryable and a committed
scene is not ready until all binding/publication/attachment succeeds.
`update()` prepares defensively; rendering/diagnostics do not step physics or
finalize an empty scene. See [lifecycle](lifecycle.md) for readiness, replacement,
reset, runtime controls and child-before-parent deferred destruction.

`simulation_time` and `physics_step_index` count successful manager substeps,
including explicit warmup, before `after_substep`. Preparation, render sync,
reads, topology rebuilds and resets do not advance or rewind these clocks.
Episode clocks may have an offset; physical evidence must validate the mapping.
Caches also validate binding/topology identity, not only the step count.

## Compound geometry and prepared variants

Procedural rigid geometry can be a `CompoundShapeCfg` of flat body-local
`ShapePartCfg` entries. Spawn emits one physical body and multiple independent
colliders; independent visual parts cannot become collision geometry. Explicit
mass/COM/inertia applies once to the parent. `CylinderCfg` retains a native
cylinder descriptor, but cooking fidelity remains backend-specific.

Preparation-time `RigidObjectCfg.env_variants` requires Newton and a supporting
SDK. MuJoCo Warp requires compatible collider topology within a batch; arbitrary
per-arena piece counts are not qualified. Spawn checks descriptor support for
variants and raw Newton `solref`/`solimp`/`solmix` before materialization rather
than silently discarding requested settings. A dependency version alone does
not qualify every geometry or solver profile.
The default ground is separately owned by `SimulationManagerCfg`: disable it
for constrained probes or author `ground_plane_attrs` to match a reference
contact profile. Omitting that profile preserves the existing scene defaults.

## Quaternion and pose convention

Domain quaternions are `xyzw`; 7D poses are `xyz + xyzw`. Convert once at external
boundaries. Scene pose buffers use `xyzw + xyz`; legacy `PhysicalAttr` uses
`wxyz`; Newton/Warp already use position + `xyzw`. cuRobo, GenSim and viewer fields
that explicitly name `wxyz` need adaptation. Mass descriptors instead use a
body-frame matrix/COM with principal-inertia composition at the adapter boundary.
Use non-symmetric test rotations to detect order mistakes.

## Invariants

- Select backend, devices, renderer and arenas before construction. Keep physics
  dt fixed during playback and retime samples to the command clock.
- Batch adapters preserve unselected rows/DOFs and device-local selections.
  Public articulation order is distinct from source traversal; map by name.
- Runtime property writes retain resolved initialization snapshots. Selected
  resets honor exclusions and complete Newton articulation clearing.
- Deformables require Newton/CUDA and supported non-gradient solvers. Cables
  require DexUni/CUDA without gradients, pre-prepare declaration and fixed
  finalized topology; reset belongs to their Scene.
- Native write/teardown boundaries are capability-specific. Unwind caller scene
  locals and tracebacks before draining deferred cleanup; release child views
  before parents.

## Where to Make Changes

| Request | Read next / change owner |
|---|---|
| Preparation, topology changes, reset, cleanup, controls | [Lifecycle](lifecycle.md); manager and `spawn/scene.py` |
| Backend/device selection, source overlays, collision/mass policy | [Physics configuration](physics-configuration.md); `cfg/`, `spawn/descriptors.py` |
| Joint state order, drive lowering, mimic coupling, native adaptation | [Articulation and batch adapters](articulation-adapters.md); `objects/backends/`, `objects/articulation.py` |
| Initial-fill fluid configuration, state and rendering | [Fluids](fluids.md); `cfg/fluid.py`, `objects/fluid.py` |
| Volume/surface state, topology and kinematic nodes | [Deformables](deformables.md); `cfg/deformable.py`, `objects/deformable/` |
| Cable declaration, segment state and external loads | [Cable objects](deformables.md#cable-objects); `cfg/cable.py`, `objects/cable/` |
| Camera synchronization, native window/offscreen denoising, startup readiness | [Rendering and diagnostics](rendering.md); manager, `cfg/simulation.py`, `_startup_summary.py` |
| Robot definition, control parts, robot presets | [Robot system](../robot-system/robot-system.md) |
| Camera attachment resolution or contact sensors | [Sensor system](../sensor-system/sensor-system.md) |
| Browser export or native gizmos | [Visualization](../sim-visualization/sim-visualization.md), [native gizmos](../sim-visualization/native-gizmos.md) |
| Solvers, planners, workspace, augmentation | [IK](../ik-solvers/ik-solvers.md), [motion](../motion-planning/motion-planning.md), [workspace](../robot-workspace/robot-workspace.md) |
| Semantic geometry and action execution | [Atomic Skills](../atomic-actions/atomic-actions.md), [Task Programs](../task-programs/task-programs.md) |

## Common Failure Modes

For missing metadata, verify readiness at the current topology; for empty Newton
viewers, check that startup did not finalize an undeclared scene. Compute device
and render GPU are distinct. Reset leakage points to selections/snapshots; native
teardown failures point to ownership/unwind order. Motion namespaces stay lazy.

Select focused tests from the owning detail page. Lifecycle coverage starts in
`tests/sim/test_sim_manager.py` and `tests/sim/spawn/test_scene.py`; source,
configuration and adapter changes use their matching contract tests.
