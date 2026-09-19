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
