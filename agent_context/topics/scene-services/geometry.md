# Scene geometry and capture

Return to [scene services](scene-services.md) for proposal/state ownership.

## Source and native geometry

`spawn/source_geometry.py` owns fresh authored USD reads and `LinkMesh`.
Static control readers consume explicit roles and source lineage, without
building semantic graphs or choosing policy/recovery budgets. They may share
one `UsdSourceSnapshot` per assembly; live state remains Object-owned.
`control_source` rejects source/optional-sidecar drift. `control_mass_source`
creates content-addressed Default-only compensation without editing originals.
Source proofs do not certify cooked shapes or physical paths.

`rigid_mesh.capture_rigid_mesh` owns scaled semantic vertices and matching
triangles as CPU data with the env axis intact. It reads no poses, preserves
order/winding/duplicates, and substitutes no hull or box for missing topology.
Recapture after model or scale changes. `get_semantic_mesh()` retains source
geometry or an explicit `semantic_shape` across Gaussian appearance replacement;
legacy facades without that API use their visual mesh. Gaussian-only appearance
does not supply an implicit semantic surface.

`native_hand.py` captures exact cooked convex/box shapes and requested offsets
for simulation-owned Robot/rigid UIDs. It requires Default/B1/env0, checks FK
against native frames and validates owners before later reads. Strict conversion
and offset reads belong to `objects/backends/collision.py`; closing sampling
shares [Robot hand geometry](../robot-system/robot-system.md#sampled-hand-geometry).
Capture creates no query or grasp policy. GraspKit qualifies geometry; Atomic
services compose query lifetime and hosts interpret physical effects.

`mesh_snapshot.py` instead captures scaled rigid semantic vertices and ordered
articulation collision-input components from explicit UID/link mappings. It
reads no live poses; missing or connected-nonconvex inputs remain unavailable.
Planner role metadata never removes a selected physical input. The separate
`collision_sources.py` link-box adapter keeps its conservative AABB policy.
`snapshot_collision_sources` combines descriptors with explicit same-observation
poses and `world_to_frame`; batch axes must match and projected obstacle poses
must agree. It returns B=1 shared-world providers, without selecting row zero,
rescaling or reordering topology. See
[collision worlds](../motion-planning/collision-worlds.md) for provider lifetime.

`NativePlanarPlacement` binds two explicit rigid UIDs and accepts observed poses.
Recreate it after replacement/model/scale changes. It reports containment, area
and vertical gap; exact silhouettes preserve holes/concavities and hull mode is
explicit. Errors remain unavailable evidence. Hosts own thresholds, observation
identity and acquisition/release acceptance.

## Tensor kernels and CPU strategies

`compute.geometry` consumes numerical inputs and retains no scene/Robot/semantic
state. It owns AABB overlap/containment, affine bounds, triangle metrics, masked
height reductions, grasp/release proposals, alignment and distal staging.
`pose_checks.as_rigid_matrix` validates one transform without repair or FK/IK.
Contact-support calculations consume arrays; [sensors](../sensor-system/contact-history.md)
own acquisition and witness lifetimes. Landing geometry may consume supplied
Trimesh meshes/Shapely polygons.

`B` means layouts and `N` means objects: `(N, 2, 3)` is one object set, not N
arenas. CPU/CUDA kernels preserve float32/64, device and broadcast batch axes;
callers introduce pair axes explicitly. Height reduction uses `(..., N, 2, 3)`
with optional `(..., N)` validity; padding is excluded and empty/all-padding
batches report invalid evidence. Numeric invalidity stays on-device.

Scene support, polygon search, packing and SLSQP remain CPU strategies. Calling
Torch kernels does not move those algorithms to CUDA. `_sampling` supplies
counter-addressed XY rejection batches, without materialization or persistence.
`_hierarchical` supplies postorder subtree XY/yaw sampling, dynamic envelopes and
whole support-cell certificates. Fixed-order reductions preserve CUDA batch
prefixes; rectangle sampling uses area rather than placement retries.
`_support_grid.py` owns parent-local near-planar layers, holes/islands,
evidence-checked overrides and conservative rasterization.

Default support detection keeps its outer-contour policy. Explicit
`preserve_holes` retains openings; polygon inputs retain disconnected regions.
Scene expansion caches a rectangle covered by the chosen mesh component.
Relative group poses remain fixed under shared translation/height alignment;
independent AABB separation is a distinct operation.
