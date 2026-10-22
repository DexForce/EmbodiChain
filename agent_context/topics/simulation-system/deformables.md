# Deformable topology and state

Read this for Newton volume/surface declaration, particle mutation and render
binding. Return to the [simulation overview](simulation-system.md).

## Owners and supported boundary

`cfg/deformable.py` owns the discriminated `DeformableObjectCfg` hierarchy:
`VolumeDeformableObjectCfg` and `SurfaceDeformableObjectCfg`.
`objects/deformable/` supplies their shared `DeformableObject` and concrete
`DeformableObjectData` contract. `SimulationManager.add_deformable_object()`
uses one registry and backend dispatch for both topologies.

The supported path is Newton on CUDA, without gradient mode. Declarations
accept unresolved AutoSolver or a supported particle solver (`xpbd`,
`semi_implicit`, `vbd`, `dexuni`); they reject unsupported backends/solvers and
post-finalization additions. Default soft/cloth compatibility is absent.
Future backend support belongs at dispatch, not inside a monolithic config.

`spawn/descriptors.py` translates volume/surface configs to typed `SoftBodyDesc`
and `ClothDesc`; volume meshing uses `SoftBodyMeshingDesc`. Runtime mutation
uses `Scene.create_particle_set_batch()`, not renderer buffers or legacy native
SoftBody/ClothBody objects. Consumers use `data.nodal_pos_w`, `nodal_vel_w`,
`nodal_state_w`, `get_surface_vertices()` and `get_surface_triangles()`.

## Physical topology is distinct from visual topology

`shape` owns the physical mesh. `particle_flags` follows simulation-particle
order; node selection for a runtime control is not a render vertex selection.
For surfaces requiring stable authored node indices, array-backed `MeshCfg`
preserves the supplied order; native file importers can reorder vertices.
Volume voxelization creates tetrahedral particles whose indices do not match
the source mesh vertices.

A surface's optional `visual_shape` can have independent indexing and UV seams;
`visual_binding_mode` controls mapping from particles. Visual material/UVs
belong on `visual_shape` when present. Volumes likewise keep collision topology
from the typed particle handle separate from their visualization surface.
Do not infer physical node identity from a render mesh or write render vertices
to mutate simulation. Browser export belongs to
[visualization](../sim-visualization/sim-visualization.md).

Physics groups use deformable `attrs`, not the rigid `ObjectBaseCfg.attrs`
parser. Volume-only `meshing` and common `surface_props` preserve topology
ownership. Volume elasticity uses Young's modulus/Poisson's ratio, translated
to Lamé coefficients at Spawn; density units differ (kg/m³ volume, kg/m²
surface). Exact fields and defaults stay in the typed config source.

## Kinematic node controls

Clear the Newton `ACTIVE` bit for all driven nodes in `particle_flags` before
model construction. Register through
`SimulationManager.register_kinematic_nodal_trajectory()` after declaration
and before `prepare()`. Offsets are relative to finalized initial world
positions; runtime initialization resolves and validates the particle set.
Host-side writes disable graph replay. See [runtime controls](lifecycle.md#newton-controls-before-preparation)
for layouts, interpolation and related material schedules.

If a replicated file-backed volume fails render upload, compare cloned render
vertex counts with template embedding topology before changing particle state
code; physical and render meshes need compatible binding, not identical indices.

## Change sites and validation

Use `cfg/deformable.py` for parsing, `spawn/descriptors.py` for typed topology,
`objects/deformable/base.py` and `data.py` for shared state, and `volume.py` or
`surface.py` for specialization. Manager declarations/capability checks remain
in `sim_manager.py`; nodal motion is in `_runtime_controls.py`.

Focused coverage: `tests/sim/test_deformable_cfg.py`,
`tests/sim/objects/test_deformable_object.py`, `test_soft_object.py`,
`test_cloth_object.py`, `tests/sim/spawn/test_descriptors.py`, and
`tests/sim/test_runtime_controls.py`.

## Cable objects

`CableObjectCfg` and `CableObject(BatchEntity)` are independent of particle-set
deformables. `SimulationManager.add_cable_object()` lowers through
`spawn/descriptors.py` into `CableDesc`, declares in the existing `SpawnScene`,
and binds the same facade in place during `prepare()`. Backend capabilities
reject Default, CPU, AutoSolver and gradient mode; use explicit DexUni on CUDA.

`CableObjectData` uses `Scene.create_cable_batch()` and exposes independent
world snapshots for equal-topology replicated segments. Public poses are
`xyz + xyzw`; the adapter permutes DexSim's quaternion-first batch output
without adding Arena offsets again. Authoring material quaternions are likewise
xyzw and convert once to CableDesc's wxyz. Defaults come from the engine's
immutable creation state, not a live observation at binding.

External force/torque and pending-load clearing use public `SpawnedCable`
methods. `write_segment_state_to_sim()` accepts world states in the same public
layout and uses `CableBatch.apply_state()` for both engine buffers. This teleport
clears history for each owning world, including other cables, while preserving
controls and loads. Object reset restores that cable's defaults and clears its
loads. It is not a complete episode reset or a solver-history checkpoint.

`EmbodiedEnvCfg.cable_object` declares Gym cables before preparation and belongs
to the physical environment component. Cable recordings and replay restrictions
are owned by [Gym execution](../env-framework/execution.md).

Newton's pre-build backend hook preflights cable attachment compatibility and
registers nonempty `init_qpos` through public model-initialization and batch-target
hooks before engine creation defaults are frozen. Input vectors are copied to
finite host float32 values, including CUDA tensors. The initializer does not
overwrite runtime commands; episode reset restores nonzero poses and targets.
Permanent attachment to an articulation with nonzero `init_qpos` is rejected
before commit: the current engine resolves attachments before model-initialization
hooks. Supporting that combination requires an engine authoring API that applies
initial joint poses before attachment construction.

Newton's scene-reset backend hook uses one public `Scene.reset_arenas()` call for all physical assets,
targets, loads and supported controls in selected environments. The manager then
resets selected lights and sensors once. Resolve Arena
indices by name: the Scene's default Arena means environment IDs are not Arena
indices. Physical exclusions are rejected before mutation. Runtime property edits
are outside this reset contract. Engine preflight rejects unsupported controls,
selected particles and unsupported solver modes. Global time remains monotonic;
contact queries need another update. For seeded initialization, reset the manager
then write caller-owned perturbed default segment states, preserving attachment
anchors. Finalized physical topology and individual removal remain unsupported.
Do not access backend arrays to implement missing engine operations.

Focused coverage: `tests/sim/test_cable_cfg.py`,
`tests/sim/objects/test_cable_object.py`, and
`tests/sim/objects/test_cable_object_cuda.py` (explicit serial CUDA run).
