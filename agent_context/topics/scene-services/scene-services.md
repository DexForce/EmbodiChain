# Scene services

`embodichain.lab.sim.scene` supplies configuration composition, procedural
proposals, geometric layout, physical state and validation. It consumes existing
entity UIDs and configurations; it owns no native scene, resource registry,
semantic scene graph, generation model client or task lifecycle.

## Entry points and boundaries

| Request | Owner |
|---|---|
| Compose existing configurations and anchors | `embodichain/lab/sim/scene/composition.py` |
| Seeded proposals and measured XY layout checks | `embodichain/lab/sim/scene/procedural.py` |
| UID-based support geometry and group placement | `embodichain/lab/sim/scene/geometry.py`, `placement.py` |
| Fixed-orientation XY constraints and solution status | `embodichain/lab/sim/scene/layout.py` |
| Existing binding selection and cloned poses | `embodichain/lab/sim/scene/query.py` |
| Immutable physical capture and restoration | `embodichain/lab/sim/scene/state.py` |
| Preflighted allowed rigid-body pose changes | `embodichain/lab/sim/scene/operations.py` |
| Explicit evidence, state comparison and caller-driven settling | `embodichain/lab/sim/scene/validation.py` |
| Batched tensor geometry kernels | `embodichain/compute/geometry/` |
| CPU support, packing and constrained-layout strategies | `embodichain/lab/sim/scene/_support_cpu.py`, `_placement_cpu.py`, `_packing_cpu.py`, `_layout_cpu.py` |
| Existing-entity expansion proposals and host integration | [Scene expansion](../scene-expansion/scene-expansion.md) |

The package facade resolves services lazily. Expansion has its own
`scene.expansion` namespace. `compute.geometry` takes Torch tensors, never entity UIDs,
polygon objects, simulation or GenSim models. Scene CPU strategies own support
selection, polygon search, greedy packing and SciPy SLSQP optimization.

## Construction and proposal flow

Resolved asset configurations → owned `SceneFragment` → optional anchor/frame
composition → `SceneRecipe` → seeded XY proposal → existing Spawn declaration
or allowed state changes → host physics steps → measured checks.

Fragments deep-copy configurations and preserve body scales. Composition may
namespace entity/anchor names and align a named anchor, but does not select or
override physics. `declare_fragment` declares through the existing manager;
its caller owns `prepare()` and any recovery after a native declaration failure.

Recipes provide explicit proxy geometry, versioned identity, constraints and
asset choices. Seed and logical ordinal determine proposal sampling independently
of environment slots. Generation changes XY translations only; authored height
and orientation remain. Virtual fixed entries can represent keep-out regions.
An unsuccessful layout has no fragment or applicable variant.

For group proxies, the layout UID names a configured root and the AABB covers
all members. Generation moves only listed movable entities by their solved
center displacement; consumers expand the root and every member from a common
baseline and include all changed UIDs in the movable permissions. The shared
recipe does not traverse a GenSim support graph.

Scene layout uses complete AABB bounds, fixed centers, planar relations and
weighted displacement preferences. Only `frame="arena"` relations are supported;
unsupported frames are reported instead of reinterpreted. `solved`,
`no_solution`, `budget_exhausted` and `unsupported` distinguish outcomes.
Failure does not prove mathematical infeasibility. Geometry checks do not prove
mesh collision freedom, vertical support, stability, reachability or task success.

## Tensor batching and CPU strategies

The geometry kernels preserve CPU/CUDA device and float32/float64 dtype. They
compute AABB overlap and containment violations, affine AABB bounds, triangle
normals/areas, and masked group-height reductions. Shape/dtype/device checks use
host metadata; numeric invalidity is represented by outputs on the input device.

`B` indexes independent layouts; `N` indexes objects in one layout. An `(N, 2, 3)`
AABB collection is one object set, not N independent environments. Kernels
broadcast leading dimensions; callers introduce separate object-pair axes when
requesting pairwise comparisons. Group-height reduction consumes `(..., N, 2, 3)`
and an optional boolean `(..., N)` validity mask, retaining independent leading
batches. Padding is excluded; empty/all-padding batches return invalid evidence.

Scene strategies still run on CPU and use Shapely, Trimesh and SciPy. Their
Torch-kernel calls do not offload polygon search or SLSQP to CUDA. CUDA kernel
support does not enable B>1 physics capture, edits or host stepping.

## Frames and runtime invariants

- Geometry is already scaled/oriented in arena-local Z-up metres. Rigid poses
  use proper 4x4 transforms or position plus `xyzw`; replication offsets do not
  enter proposals. [GenSim](../gen-sim/gen-sim.md) owns its Y-up conversion and
  editable scene model.
- Mesh support detection currently selects the largest outer contour at one
  main horizontal level. Explicit polygon support inputs retain holes and
  disconnected regions. Group translation and height alignment preserve
  relative poses; independent AABB separation uses a separate strategy.
- `SceneView` wraps existing rigid objects, robots and articulations. The current
  runtime rejects worlds other than B=1 and selections other than `(0,)` before
  writes. Poses and physical snapshots still expose an environment dimension.
  Rigid groups, deformables and explicit rigid constraints are unsupported.
- Physical snapshots own numeric poses, velocities, joints, targets and efforts;
  restore checks original ownership, binding, backend and topology. It does not
  restore solver internals, task progress or a full backend checkpoint. Host
  integrations retain control-period and episode identity contracts.
  Restoration rejects saved joint positions or targets outside current limits
  during preflight, before any writes; the articulation writer would clip them.
- Change sets preflight all UIDs and movable permissions, apply rigid poses,
  clear dynamics and publish render state without stepping. Native write errors
  propagate without guaranteed rollback. The host handles reported planner and
  observation invalidations even when the simulation clock has not advanced.
- Required checks are `passed`, `failed` or `unverified`; empty or unverified
  reports cannot accept a candidate. Fixed-step settling is unverified stability.
  Opt-in convergence checks consecutive low-velocity measurements. Supplied
  step callbacks advance the whole world; no per-selection isolation is implied.
- `check_layout` reads actual poses after settling. Supplied measured arena XY
  AABBs update individual proxies and allow checking changed orientations.
  Missing poses, changed support, or changed orientation without a measured
  proxy yield unverified evidence. Moving fixed entities fails validation;
  vertical support and task acceptance still need separate measurements.
  Group consumers supply the union of every member's measured AABB under the
  root UID and separately verify internal structure and fixed member poses.
  A root-only pose check cannot establish that the whole group moved correctly.

## Consumers and change sites

[GenSim](../gen-sim/gen-sim.md) keeps image/text understanding, assets, edit graph,
model calls, temporary simulation ownership, diagnostics and export. Its
existing support/group optimizers adapt the shared geometry algorithms; the
tabletop and parent-surface optimizers adapt shared `solve_layout`. Its layout
constructor retains hierarchy traversal and pose propagation.

[Task Program expansion](../scene-expansion/scene-expansion.md) accepts host-owned
physical preparation callbacks. The concrete Gym host and workspace adaptation
are provided separately by [PR #750](https://github.com/DexForce/EmbodiChain/pull/750);
that layer retains reset order, observation refresh, bridge recreation and
initial-state provenance while using these physical services. General
[simulation lifecycle](../simulation-system/simulation-system.md) remains owned
by Spawn and SimulationManager.

CPU strategies use SciPy and Trimesh supplied by DexSim; EmbodiChain's core
dependency requires Shapely 2 or newer. No scene-specific extra is needed. The standalone
`scripts/tutorials/scene/tabletop_recipe.py` demonstrates a workbench, two
panel-built containers, parts and a keep-out region with native preparation,
settling, measured layout and restoration; it requires the installed simulation
runtime and does not call a model provider.

The standalone `scripts/benchmark/scene/geometry_kernels.py` measures geometry
kernels separately from solver/host execution. Report its actual device, dtype,
batch shape and synchronization policy; do not treat kernel throughput as
end-to-end layout or simulator throughput.

## Focused validation

Use `tests/compute/geometry/` and `tests/sim/scene/` for
geometry, constraints, recipes and runtime contracts. GenSim adapter regressions
live in `tests/gen_sim/scene_engine/`; generic Task Program expansion regressions
live in `tests/gym/envs/task_program/test_scene_expansion.py`. Concrete Gym-host
qualification belongs to the separate PR #750 integration. Native settling and
restoration need an explicit supported simulator run beyond CPU port tests.
