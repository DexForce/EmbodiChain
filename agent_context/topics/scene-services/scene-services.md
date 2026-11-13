# Scene services

`embodichain.lab.sim.scene` composes existing configurations, proposes geometry,
and reads/restores physical state. It owns no World, entity registry, semantic
scene graph, generation client or task lifecycle. Its facade loads lazily.

## Entry points and boundaries

| Request | Owner |
|---|---|
| Compose existing configurations and anchors | `embodichain/lab/sim/scene/composition.py` |
| Seeded proposals and measured XY layout checks | `embodichain/lab/sim/scene/procedural.py` |
| UID-based support geometry and group placement | `embodichain/lab/sim/scene/geometry.py`, `placement.py` |
| Fixed-orientation XY constraints and solution status | `embodichain/lab/sim/scene/layout.py` |
| Existing binding selection and cloned poses | `embodichain/lab/sim/scene/query.py` |
| Exact native planar footprint observations | `embodichain/lab/sim/scene/planar_observation.py` |
| Immutable physical capture and restoration | `embodichain/lab/sim/scene/state.py` |
| Exact native hand/rigid shape and physical-offset capture | `embodichain/lab/sim/scene/native_hand.py` |
| Preflighted allowed rigid-body pose changes | `embodichain/lab/sim/scene/operations.py` |
| Explicit evidence, state comparison and caller-driven settling | `embodichain/lab/sim/scene/validation.py` |
| Static USD root/link geometry and explicit proxy fitting | `embodichain/lab/sim/scene/articulation_geometry.py` |
| Source-declared prismatic, press and revolute physical bindings | `embodichain/lab/sim/scene/articulation_inputs.py`, `press_inputs.py`, `revolute_inputs.py` |
| Explicit control roles, collider inputs and source lineage | `embodichain/lab/sim/scene/control_semantics.py`, `revolute_geometry.py`, `control_source.py`, `control_mass_source.py` |
| Shared authored source reads and link-local collider inputs | `embodichain/lab/sim/spawn/source_geometry.py`; Scene retains its `LinkMesh` import |
| Batched tensor geometry kernels | `embodichain/compute/geometry/` |
| CPU support, packing and constrained-layout strategies | `embodichain/lab/sim/scene/_support_cpu.py`, `_placement_cpu.py`, `_packing_cpu.py`, `_layout_cpu.py` |
| Robot-aware object proposals and full-pose IK/FK checks | `embodichain/lab/sim/scene/workspace.py` |
| Existing-entity expansion proposals and host integration | [Scene expansion](../scene-expansion/scene-expansion.md) |

Read [geometry and capture](geometry.md) for source versus cooked meshes,
collision providers, tensor batching and CPU strategies. Numeric algorithms
belong to `compute.geometry`; live acquisition and ownership stay in Lab.

## Construction and proposal flow

Resolved configs → owned `SceneFragment` → anchor/frame composition →
`SceneRecipe` → seeded XY proposal → Spawn declaration or allowed pose changes
→ host physics → measured checks.

Fragments copy configs and preserve scales; composition does not choose physics.
`declare_fragment` uses the existing manager, whose caller owns preparation and
native-failure recovery. Seeds and logical ordinals are independent of slots.
Recipes change XY translations while retaining authored height/orientation.
An unsuccessful solve has no applicable fragment or variant.

Group proxies cover all members under a configured root. Consumers expand every
member from one baseline, include all changed UIDs in movable permissions and
validate internal structure; the shared solver traverses no GenSim graph.
Layout uses complete AABB bounds and arena-frame relations. `solved`,
`no_solution`, `budget_exhausted` and `unsupported` are distinct outcomes;
failure is not proof of mathematical infeasibility. A layout proposal does not
establish collision freedom, support, stability, reachability or task success.

## Frames and runtime invariants

- Geometry is scaled arena-local Z-up metres; poses use proper transforms or
  `xyz + xyzw`. Replication offsets do not enter proposals. GenSim owns Y-up
  conversion and its editable model.
- `SceneView` supports rigid objects, robots and articulations. Selected
  `env_ids` are nonempty, unique and ordered; views reject changed capacity or
  bindings. Rigid groups, deformables and explicit rigid constraints remain
  unsupported by physical snapshots.
- Snapshots own poses, velocities, joints, targets and efforts. Restore checks
  original ownership, backend, topology and all selected rows before mutation,
  leaving peers untouched. It is not a solver checkpoint or task-state reset.
  Joint targets stay strict; explicit position tolerance permits bounded
  clipping followed by host round-trip checks.
- `restore_render_state` publishes complete measured states without physics,
  temporarily admitting measured solver excursions and restoring exact limits.
  It must not become a physical reset. Episode/time identity stays host-owned.
- `scene.physics` owns selected-row mass/inertia/friction baselines for ready
  dynamic rigid bodies on Default/Newton. Changes derive from the original
  snapshot, validate all requests first and verify readback. Damping is outside
  this contract; native/readback failures require stopping the host batch.
- Change sets preflight permissions, set rigid poses, clear dynamics and publish
  render state without stepping. Native failures have no guaranteed rollback;
  hosts honor planner/observation invalidations even at unchanged simulation time.
  `apply_changes_batch` accepts distinct per-env changes and preflights all rows.
- Checks distinguish passed, failed and unverified. Empty/unverified evidence
  cannot accept a candidate. Fixed-step settling alone does not prove stability;
  convergence uses consecutive measurements across the full-world selection
  because a step callback advances every row.
- `check_layout` uses actual settled poses. Rotated objects need measured arena
  AABBs; missing poses/proxies or changed support yield unverified evidence.
  Moving fixed entities fails. Group checks use member-union bounds and verify
  relative structure, not just the root pose.

## Robot workspace adapter

`RobotSceneWorkspace` combines an explicit control part, object-local grasp and
workspace samples, then checks actual targets with IK, effective limits and
full-pose FK. It returns evidence without applying poses, stepping, resetting,
choosing semantic roles or accepting task success. Task Program adapts that
report to its execution contract.

`scripts/tutorials/scene/robot_workspace.py` demonstrates geometry checks and
optional Task Program trials. Viewer/recording presentation is separate from
physics and measured reachability; cache density is not an IK acceptance result.

## Consumers and change sites

GenSim owns understanding, graph traversal, asset generation and exports; its
adapters call these services. Task Program/scene expansion owns Gym reset,
observations, bridge identity and persistence gates. Spawn remains topology
and replication owner. See [scene expansion](../scene-expansion/scene-expansion.md)
and [Expansion Engine](../expansion-engine/expansion-engine.md).

CPU strategies require Shapely/SciPy/Trimesh. The `geometry` extra exposes
standalone mesh consumers without adding FCL or simulation; Torch-only bounds
stay importable without Trimesh.

## Focused validation

Use `tests/sim/scene/` for state, changes, geometry and workspace adapters;
`tests/compute/geometry/` for pure kernels; and the selected GenSim/Task Program
host tests for integration. Real physics, rendering and task outcomes require
separate runtime qualification. Tutorial files remain examples, not test APIs.
