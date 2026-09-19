# Simulation System

## Entry Points

| Responsibility | Source |
|---|---|
| Public simulation facade | `embodichain/lab/sim/__init__.py` |
| World, registries, readiness and stepping | `embodichain/lab/sim/sim_manager.py` → `SimulationManager`, `SimulationManagerCfg` |
| Declaration, source resolution and binding | `embodichain/lab/sim/spawn/scene.py`, `spawn/source.py`, `spawn/descriptors.py` |
| Manager-level backend hooks and capabilities | `embodichain/lab/sim/physics/` |
| Object-level Scene batch adapters | `embodichain/lab/sim/objects/backends/scene.py`, `newton.py` |
| Object, renderer and physics configuration | `embodichain/lab/sim/cfg/` (`cfg/__init__.py` is the public facade) |
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

`BaseEnv` owns Gym reset/step timing; `EmbodiedEnv` translates task composition
into manager declarations. Scene replication and collision isolation belong
to Spawn, not per-object collision-group calculations. Backend differences
belong in physics hooks/capabilities or object adapters, not task-side native
access. Contact sensors use the finalized Scene's `ContactQuery` boundary.

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

## Quaternion and pose convention

EmbodiChain domain-facing quaternions are `(x, y, z, w)` (`xyzw`); public 7D
poses are `xyz + xyzw`, with identity quaternion `(0, 0, 0, 1)`. This covers
object/link/COM state, FK/IK, sensor offsets, task configuration and math.

Convert exactly once at external boundaries. DexSim Scene pose buffers use
`xyzw + xyz` (layout permutation only); DexSim mass/COM descriptors use `wxyz`
and require explicit `quat_xyzw_to_wxyz()`/`quat_wxyz_to_xyzw()` conversions.
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
| Camera synchronization, native window, DLSS, startup readiness | [Rendering and diagnostics](rendering.md); manager, `cfg/simulation.py`, `_startup_summary.py` |
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
