# Scene services

`embodichain.lab.sim.scene` supplies configuration composition, procedural
proposals, geometric layout, physical state and validation. It consumes existing
entity UIDs and configurations; it owns no native scene, resource registry,
semantic scene graph, generation model client or task lifecycle.

## Entry points and boundaries

| Request | Owner |
|---|---|
| Explicit source property/pose/scale ingress | `source_physics.py`, `source_transforms.py`; owned declarations only, no physical acquisition or host lifecycle |
| Static source component/pose/contact/prismatic declarations | `component_declarations.py`, `frozen_declarations.py`, `source_contact.py`, `prismatic_declarations.py`; explicit owned configuration only |
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
| Selected native root/prismatic-link spatial geometry | `embodichain/lab/sim/scene/spatial_reference.py` |
| Static USD root/link geometry and explicit proxy fitting | `embodichain/lab/sim/scene/articulation_geometry.py` |
| Explicit source control resize/support/press calibration | `control_preparation.py`, `control_support.py`, `source_measurement.py`, `press_calibration.py`; static source data, never live acquisition |
| Source-declared prismatic, press and revolute physical bindings | `embodichain/lab/sim/scene/articulation_inputs.py`, `press_inputs.py`, `revolute_inputs.py` |
| Explicit control roles, collider inputs and source lineage | `embodichain/lab/sim/scene/control_semantics.py`, `revolute_geometry.py`, `control_source.py`, `control_mass_source.py` |
| Static source support/container/Pour target affordances and horizontal permission | `declaration_placement.py`; explicit source views/reserves, no semantic graph or runtime planning |
| Enabled authored collision meshes in owning-link frames | `source_collisions.py`; shared source snapshot, ordered polygon-fan topology and uniform-scale boundary |
| Declared world source meshes and conservative slider envelopes | `source_geometry.py`, `source_envelopes.py`; static source/scale/pose only, no native collision claim |
| Declared local source vertices | `source_geometry.read_source_vertices`; authored mesh/articulation data or an explicitly scaled local override, with no native observation |
| Read-only source glTF basis/top inputs | `source_geometry.py`; original scene transforms, scale and topology, with no native acquisition |
| Shared authored source reads and link-local collider inputs | `embodichain/lab/sim/spawn/source_geometry.py`; Scene retains its `LinkMesh` import |
| Explicit robot mounting frames and URDF collision inputs | `source_robot_geometry.py`, `utility/urdf_collision.py` |

| Explicit Default/B1 gravity, native Pour inputs and immutable receiver snapshots | `embodichain/lab/sim/scene/physical_parameters.py`, `pour_inputs.py`, `pour_region.py` |
| Batched tensor geometry kernels | `embodichain/compute/geometry/` |
| CPU support, packing and constrained-layout strategies | `embodichain/lab/sim/scene/_support_cpu.py`, `_placement_cpu.py`, `_packing_cpu.py`, `_layout_cpu.py` |
| Robot-aware object proposals and full-pose IK/FK checks | `embodichain/lab/sim/scene/workspace.py` |
| Existing-entity expansion proposals and host integration | [Scene expansion](../scene-expansion/scene-expansion.md) |

The package facade resolves services lazily. Expansion has its own
`scene.expansion` namespace. `compute.geometry` takes numerical geometry, never
simulation or GenSim models. Its `contact_support.py` computes a contact patch
and COM distance from arrays; native acquisition and witness lifetimes belong
to the [sensor integration](../sensor-system/contact-history.md). Its `landing.py`
CPU routines additionally consume Trimesh meshes and Shapely polygons for
triangle-union floor and silhouette measurements. Scene CPU strategies own
support selection, polygon search, greedy packing and SciPy SLSQP optimization.

Pure control dimension candidates, ordered XY clearance proposals and source
press release-bias coefficients also live under `compute.geometry`; they consume
explicit numeric dimensions/bounds, never a robot, task graph or physical state.
Their callers own calibrated inputs, asset adaptation and qualification.

Pure grasp samples, release clearances, ordered footprint proposals, source-axis
adaptation, held-object alignment targets and distal staging also live in
`compute.geometry`. They consume explicit arrays, tensors or mesh/polygon inputs
and retain no scene, Robot or semantic graph. `alignment.py` takes supplied poses,
axes and held/root transforms. Runtime acquisition, policy selection, ownership
and physical acceptance remain with their consuming subsystems.
`pose_checks.as_rigid_matrix` only validates one input transform; it neither
repairs it nor invokes FK/IK.

`physical_parameters.observed_gravity` reads actual configured Default/B1 gravity,
without inventing missing values. `pour_inputs` captures native local mesh data
and the selected solver root; `pour_region` owns immutable arrays and delegates
staging arithmetic to Compute using fresh caller-supplied poses. These planning
inputs do not create native resources or establish physical acceptance.

`rigid_mesh.capture_rigid_mesh` snapshots scaled native input vertices and
matching triangles as owned CPU data with the environment axis intact. It reads
no poses, preserves duplicates/winding/order and never substitutes a hull or box
for missing topology. Callers recapture after model/scale changes. This separate
API leaves the lightweight mesh-envelope path unchanged and supplies
`NativePlanarPlacement` and exact silhouette/Pour consumers.

`native_hand.py` captures cooked convex/box geometry and explicitly requested
rest/contact offsets for the exact simulation-owned Robot and native rigid UIDs.
It requires Default/B1/env0, checks named FK against native frames, and validates
original owners before subsequent rigid reads. Strict shape conversion and offset
reads belong to `objects/backends/collision.py`; closing FK reuses
[hand geometry](../robot-system/robot-system.md#sampled-hand-geometry).
It creates no collision query and selects no grasp policy. Geometric qualification
belongs to [GraspKit](../motion-planning/motion-planning.md); query composition and
cleanup belong to `ActionPlanningServices` in [Atomic Actions](../atomic-actions/atomic-actions.md).
Captured shapes and nominal queries do not establish physical acquisition/release.

`native_hand_scene.py` and `native_motion_scene.py` expose opt-in FCL queries
on the same physical observation. Hand capture retains collision-only links,
planes and native contact/rest offsets; motion capture retains ordered rigid and
articulation obstacles, arm parts and named-joint FK. Both preserve the original
Default/B1/env0 coverage and reject stale or replaced owners rather than refresh.
Callers supply the observation guard and close each owned query. These are
discrete geometric checks, not continuous-path or physical-effect evidence.
`convex_clearance.py` checks conservative input-mesh solids at explicit frame
indices, preserving order and first rejection; it also catches full containment.
It takes explicit margins and never substitutes for a native contact witness.

Static control readers consume explicit authored roles, keep source mesh frames
and identity, and never build semantic graphs or select execution/recovery
budgets. Acceptance thresholds and semantic link IDs belong to their callers.
They consume the shared [Object/Spawn source reader](../simulation-system/simulation-system.md)
and may explicitly share one `UsdSourceSnapshot` within an assembly. Source
freshness checks reuse captured identities without repeating role parsing at
the end of each inspection. Scene retains control-specific qualification,
source-label targets and geometry derivation; live state uses Object APIs.
`control_mass_source` retains the Default-only two-attribute mass compensation
in a content-addressed derived source; it never edits the original asset.
`control_source` rejects source/sidecar drift, including newly present optional
sidecars. Source mesh proofs do not certify native cooked shapes or paths.

`source_robot_geometry.py` resolves declared fixed mounting chains and arm-base
anchors from source configuration; explicit component transforms take precedence.
`utility.urdf_collision` reads exact selected mesh/box colliders into the existing
Spawn `LinkMesh` type, preserving source order and scaling origins/vertices once.
It rejects unresolved URIs and visual-only links. Vertex-envelope mode remains
explicit and supplies no triangle-surface certificate. These generic readers
construct no Robot, simulator or IK solver and select no concrete robot preset.

Static source declarations reuse Spawn's USD translator without creating a
World. `frozen_declarations` checks raw matrix/position/intrinsic-XYZ redundancy
before descriptor conversion and retains configured scale precision in override
and overlay modes. `component_declarations` serializes source poses and collider
choices into owned component mappings; it selects no physics backend.
`source_contact` exposes Default descriptor contact margins for static fitting.
`prismatic_declarations` applies the existing passive source calibration only to
explicit selected UIDs while preserving declared drive and separate mass owners.
The optional prismatic reset label and authored handle-center readers remain in
`articulation_inputs`; neither is an observed or settled physical state. Task
Engine may bind these public descriptions, but owns no implementation here.

`NativePlanarPlacement` owns geometry for two explicit native rigid-object UIDs;
recreate it after replacing either object or changing its model/scale. Queries
take explicit observed poses and return footprint containment, area and vertical
gap, without accepting a task or inferring contact support. Exact silhouette mode
retains concavities and holes; convex-hull mode remains explicit. `source_geometry()` returns independent subject vertices and floor copies after
checking the original bindings. `capture_planar_contents` reads explicitly ordered
logical/native rigid contents at B=1; both supply geometry without reading poses
or certifying retention. Acquisition errors remain unavailable evidence. Callers
own thresholds, invocation/episode association, observation timing and evidence
publication.

`mesh_snapshot.py` captures scaled rigid visual vertices and ordered articulation
collision-input components from explicit UID/link mappings. It reads no live
poses and keeps unavailable geometry distinct from usable components. Neutral
planner roles are metadata: `none` does not remove a selected physical input.
Connected nonconvex inputs remain unavailable in this capture path. The separate
`collision_sources.py` link-box adapter retains its conservative component AABB
policy; it is not a native cooked-shape/contact oracle.
`snapshot_collision_sources()` combines common shape descriptors and explicitly
supplied same-observation poses through `world_to_frame`. Batched inputs require
matching pose/transform batch axes and equal projected obstacle poses; otherwise
the helper rejects them rather than selecting row zero. It returns B=1 providers
for the declared shared geometry/world, not per-environment world providers.
It preserves topology/shape order,
applies no second scale, and supplies owned providers to existing planner APIs.
See [collision worlds](../motion-planning/collision-worlds.md) for cuRobo query lifetimes.

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

`compute.geometry._sampling` additionally supplies counter-addressed rectangular
AABB rejection batches on CPU/CUDA. It returns numeric XY proposals and validity
masks only. GenSim owns their group/asset adaptation, candidate materialization
and persistence; this sampler does not replace the constrained SLSQP strategy.
`_hierarchical` adds integer-indexed postorder subtree XY/yaw batches with whole
support-cell coverage certificates and dynamic outer envelopes. It consumes
numeric geometry only; fixed-order matrix reductions and area scans preserve batch-prefix
reproducibility on CUDA. `_support_grid.py` owns the parent-local CPU policy:
explicit support transforms, child-contact-guided near-planar layers, unions with
holes/islands, evidence-checked overrides, conservative layer height bounds and
whole-cell rasterization. The hierarchical sampler partitions free AABB center
rectangles and samples by area without placement retries.
The table detector retains its default outer-contour behavior. Shared scene
expansion enables `SupportSurfaceConfig.preserve_holes` and caches a rectangle
covered by the selected mesh component, excluding openings during preparation.

`spatial_reference.read_spatial_reference` resolves the selected catalog body
and its scaled link-local vertices through the existing authored articulation
reader. A fixed grasp child cannot replace that physical reference. Root pose,
source units and horizontal prismatic limits retain their original contracts;
missing/unsupported geometry is unavailable. The returned arrays are owned and
read-only. Semantic role selection, signed-axis requests and acceptance remain
with callers.

## Frames and runtime invariants

- Geometry is already scaled/oriented in arena-local Z-up metres. Rigid poses
  use proper 4x4 transforms or position plus `xyzw`; replication offsets do not
  enter proposals. [GenSim](../gen-sim/gen-sim.md) owns its Y-up conversion and
  editable scene model.
- Mesh support detection selects the largest contour at one main horizontal
  level; `preserve_holes` retains its mesh openings. Explicit polygon inputs retain holes and
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
  by default. An explicit `qpos_tolerance` allows bounded measured-position
  clipping followed by caller-owned round-trip validation; targets remain strict.
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

## Robot workspace adapter

`RobotSceneWorkspace` uses an explicit Robot control part and object-local grasp
to derive scene poses from valid workspace samples. It independently verifies
actual targets using IK, current joint limits and full-pose FK residuals,
returning `SceneCheckReport`. It does not choose task roles, apply poses, reset,
step or accept task success. Task Program retains a compatibility adapter that
converts checks to its execution validation contract.

The UR5 demonstration and its private helpers live in `scripts/tutorials/scene/`;
run `robot_workspace.py`. They call the scene services directly and are not SDK
APIs. Scene mode exercises these services
without importing Gym or Task Program; task mode adds preparation, real Pick/Place,
first-frame and measured outcome checks. Views are Viser, native window,
offscreen camera/video and no-render. Native renderer/device availability is
separate from geometric and task correctness. A tabletop grid checks upright
object origins with the same fixed grasp, current joint limits and full-pose FK.
The shared physical input is a 2.4 m diameter round table with a central
pedestal and robot. Support and grid inclusion check the entire cube footprint
inside the faceted circular collision surface, rather than its bounding box.
`workspace_reachability.npz` saves actual green/pass and red/fail results; they
are not inferred from FK-cache density. Viser and the native Hybrid renderer
show the same red/green point cloud. The cube initializes at a red point first
and is rejected before execution, then initializes at green points for successful
pose checks and Task Program trials. Offscreen runs retain the control-cadence
`preview.mp4` and post-process saved camera evidence into
`workspace_explained.mp4` with labelled inspection pauses. Presentation does not
advance physics.

The user-facing showcase, recorded video, commands and measured acceptance are
documented in `docs/source/tutorial/robot_workspace.rst`.

## Consumers and change sites

[GenSim](../gen-sim/gen-sim.md) keeps image/text understanding, assets, edit graph,
model calls, temporary simulation ownership, diagnostics and export. Its
existing support/group optimizers adapt the shared geometry algorithms; the
tabletop and parent-surface optimizers adapt shared `solve_layout`. Its layout
constructor retains hierarchy traversal and pose propagation.

[Task Program expansion](../scene-expansion/scene-expansion.md) delegates
physical state and changes here while keeping Gym reset order, observations,
bridge recreation, initial-state identity and persistence eligibility. General
[simulation lifecycle](../simulation-system/simulation-system.md) remains owned
by Spawn and SimulationManager.

CPU strategies use SciPy and Trimesh supplied by DexSim; EmbodiChain's core
dependency requires Shapely 2 or newer. The `geometry` extra declares Trimesh
for standalone numerical mesh consumers; it adds no FCL or simulation backend.
Torch-only `mesh_bounds` entry points do not load that optional dependency.
No scene-specific extra is needed. The standalone
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
live in `tests/gen_sim/scene_engine/`; Task Program host regressions live in
`tests/gym/envs/task_program/test_simulation_scene_expansion.py`. Native settling
and restoration need an explicit supported simulator run beyond CPU port tests.

`compute.geometry.pouring` owns explicit numeric tilt arcs, gravity-top staging
and per-candidate grasp projection. It accepts arrays/tensors, actual pad lines
and configured clearance, not scene/robot/task objects. It preserves owned
snapshots, source order and masks and establishes no physical pour success.
`compute.geometry.projection.triangle_mesh_xy_projection` retains all positive
area triangles in authored order, including holes and disconnected regions;
it does not reduce the mesh to a convex hull or choose a support floor.
