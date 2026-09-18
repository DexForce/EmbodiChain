# Articulation and Scene batch adapters

Read this when changing state layout, source joint configuration or backend
adaptation. Return to the [simulation overview](simulation-system.md).

## Source configuration before binding

`SimulationManager._declare_spawn_articulation()` supplies
`spawn/descriptors.py:configure_articulation_desc()` as its configuration
callback. It resolves exact source names before applying link/joint overlays
and compiling `qpos_limits`. Default uses loaded native metadata; Newton
resolves source metadata before its first immutable build. Do not implement
initial configuration as finalize-then-rebuild or duplicate descriptor writes
inside `Articulation._apply_spawn_config()`.

`ArticulationRootPropertiesCfg` separates root intent from link physics.
Fixed-base/self-collision settings enter common descriptors, independently of
the source overlay policy; explicit `None` preserves USD intent and selects
URDF import defaults. Default-only sleep/solver-iteration settings must reach
the native articulation before Direct GPU initialization and initial reset.
Iteration counts are configured together because the native setter is atomic.
Applying them only during facade binding is too late for captured GPU settings.

`joint_drive_props` is the sole joint-property entry point; sparse fields retain
source values. `drive_type` describes the response and `target_mode` the active
command components. Lowering masks inactive gains: none/effort clear both,
velocity clears position gain. Newton rejects active acceleration drives;
non-MuJoCo position mode is gain-based emulation with zero velocity target.
Exact field options and defaults remain in `cfg/articulation.py`.

## State order and batched writes

Public `joint_names` follows finalized Scene qpos/qvel order, which may differ
from source traversal. Map initial state, control groups, limits and mimic
metadata by joint name into that order. Source-name queries serve topology
resolution, not state indexing.

`objects/backends/scene.py` owns `SceneRigidBodyView.from_entities()` and
`SceneArticulationView.from_entities()`, the object layer's batch factories.
Data objects require finalized Scene views. Delegate environment selection to
Scene batches; keep articulation joint selections on device, using reusable
full-batch tensors when partial writes must preserve other rows/DOFs. Avoid
DexSim's host-materialized selected-DOF path. Pose conversions follow the
[public quaternion contract](simulation-system.md#quaternion-and-pose-convention).

Newton root-pose writes filter unchanged rows before forwarding genuine
changes, preserving CUDA graphs on ordinary fixed-root reset. Intentional
initial anchor changes should occur after `prepare()` and before the first
`update()` so initial graph capture sees final anchors.

Newton standalone rigid-body pose/velocity writes also synchronize reduced
FREE-joint state in both runtime buffers. Keep that compatibility access in
`objects/backends/newton.py`; its selection cache is invalidated by topology
revision. Remove the workaround only when the public Scene batch guarantees
the same synchronization.

Newton physical-property reads use the finalized Scene batch and reusable
output tensors. Public inertia/COM remain principal moments plus `xyz + xyzw`;
scalar descriptors expose body-frame mat33. Runtime writes compose the pair
before DexSim, and reset restores the saved pair together to avoid reordering
principal axes between two writes.

## Mimic coupling

MuJoCo-Warp mimic joints retain native equality constraints and contact-force
coupling. `_configure_newton_mimic_compliance()` adjusts only the selected
articulation's equality rows, respecting the solver's time-constant safety
floor; a weak follower drive stabilizes updates. Command targets propagate the
authored leader relation. Never copy measured follower state or disable the
native equality, which would turn the follower into an independent servo.
Keep private Newton runtime access in the backend helper, with generic
Articulation owning name mappings, resets and target propagation.

Solvers without configured mimic compliance project reset positions onto the
authored relation; the MuJoCo-Warp compliance path preserves authored current
positions. Manager-level complete-world reset rules are in [lifecycle](lifecycle.md).

## Domain-neutral topology and geometry

`get_parent_joint_chain()` returns copied, immediate-parent-first
`ArticulationJointKinematics` values. Do not expose backend-native joint-info
objects or reach through `BatchEntity._entities` in consumers.
`get_link_render_nodes()` returns live nodes in environment order; their
lifetime ends with the asset. Camera name resolution belongs to sensors.

`get_link_vert_face()` and named-state `compute_fk()` stay deterministic and
domain-neutral; stochastic affordance sampling and semantic keys belong to
Atomic Action adapters. Newton link mesh export concatenates every render mesh
with triangle offsets, and returns empty arrays for zero-mesh links.

USD `build_pk_chain` uses resolved Spawn descriptors in
`objects/backends/_kinematics.py`; URDF retains its source chain. Fixed,
revolute and prismatic tree joints preserve
`origin_pose @ motion(q) @ inverse(target_pose)`. Non-tree/unsupported joints
fail explicitly; simulation-only users can disable chain construction.
Full-width Jacobian inputs mean public joint order, including equal-width
chains; narrower chain-sized inputs retain serial order. Output columns use
selected chain parameter order.

## Validation

Use `tests/sim/objects/test_scene_backend.py` for batch selections, pose order,
FREE-joint synchronization, mesh export and no-op root writes;
`test_articulation_drive_compat.py` for mimic/drive behavior;
`tests/sim/objects/backends/test_kinematics.py` and
`tests/sim/objects/test_articulation.py` for topology/FK. Source configuration
and root timing are covered by `tests/sim/spawn/test_descriptors.py` and
`tests/sim/test_sim_manager.py`.
