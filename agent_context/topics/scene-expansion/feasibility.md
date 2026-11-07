# Joint scene and robot feasibility

`embodichain.lab.sim.scene.feasibility` owns fixed task/robot/scene search
contracts, joint XY/yaw proposals, branch screening, conditional regions and
independent fixed-count validation. It owns no simulator, semantic registry,
model client or data execution lifecycle. Numerical planar composition lives in
`compute.geometry.planar`.

The configured entry is `search-scene-feasibility`, dispatched to
`lab.task_program.integrations.feasibility`. It strictly decodes an explicit
request and a trusted existing Task Program deployment. Robot, resolved runtime
policy, physical configuration and acceptance thresholds enter its identities.
The configured adapter supports one runtime-policy mode; Python integrations
can bind multiple legal operation modes explicitly. An explicit workspace phase,
part and group can enable `RobotWorkspaceProposalSource`: it owns valid-only FK
copies after a warm host is destroyed, retains independent uniform exploration,
and never certifies projected height/orientation. Fixed coordinates canonicalize
to zero; proposal prefixes require the same recorded feedback stream.

`integrations/simulation/feasibility.py` owns the B=1 Gym host. It rebuilds an
installation from copied configuration, prepares through
`SimulationSceneExpansionHost`, screens the phase graph, restores and identifies
the actual execution start, then uses `execute_scene_variant`. Runtime controller
provenance selects phase contact permissions; measured object poses replace
projected attachment poses during actual execution. `ExpansionSession` remains
the owner of attempts, reservations, accepted episodes and persistence receipts.
Only confirmed writes count as committed data. Required `observation_keys` use
actual reset/step returns, preserving T+1 history without extra observation
updates. Raw recording bytes are bounded. Independent variation events are
rejected by the configured adapter; known reset settling is permitted, and
physical geometry signatures are rechecked across preparation/restoration.
Initial collision checks include static layout overlap with explicit support
contacts. Per-instance simulation-update guards bound preparation/settling loops
without blocking cleanup; required observations use the qpos-only session schema.

`planning.py` retains full-joint branches, including synchronized targets and
mimic constraints. Hypothetical whole-robot FK supplies solver-root transforms;
using live roots for projected torso/base states is incorrect. Complete path
screening uses explicit physical geometry and phase contact permissions.
`collision.py` lazily builds a Newton `CollisionPipeline` over isolated query
states. The physical extractor preserves native convex shapes, includes the
global ground, and rejects triangle meshes, SDFs, non-robot articulations and
unsupported robot asset scales/formats. No collider is silently convexified.
Halfspaces use analytic physical-shape support. Phase masks do not mutate live
filters, and the query never advances the simulation clock. Shape/pair/contact
storage is bounded; saturation and nonfinite results remain unverified.
The host closes the owned query arrays/graphs after every attempt.

The checker separately exposes `contact_query` native signed-surface facts and
`contact_pairs` for all active strict-contact physical-ID pairs from one collide.
The tolerance-aware `check` and capped interval retain their existing semantics.
Exactly represented native touching can decide zero rest offset; conversion
uncertainty spanning a threshold remains unavailable. Triangle meshes require
explicit `allow_triangle_meshes=True`; this uses actual native triangle surfaces,
never hull substitution or a containment certificate. The default physical
extractor continues to reject unsupported geometry. Mesh midphase/contact
overflow is unavailable, and returned native facts own their data.
The mesh-group adapter's `clearance_lower_bound` reuses the same geometry/pose
conversion preparation for conservative frozen-AABB culling without a model or
collide. Only a decided bound strictly above the original threshold can skip a
native query; near/unsafe inputs remain zero or unavailable. It never supplies
contact evidence or replaces the native predicate.
Mixed mesh/primitive absence of contacts is not a separation certificate: the
native SDK can omit such contacts for intersecting inputs. `check`,
`contact_query` and `contact_pairs` require native contact evidence or a canonical
AABB separation proof; otherwise they retain unavailable clearance with an
explicit reason. Convex-only defaults retain their original semantics.
All mesh/mesh negative decisions require canonical AABB separation. Bounded
native searches can omit deeply contained components even while reporting
positive halo contacts for another component. Missing contacts or a positive
reported minimum are not complete solid-clearance certificates. All three
checker entries retain unavailable evidence without that proof. Native contact
facts can still decide positively. Near separated pairs and empty concave gaps
with overlapping AABBs can remain unavailable without changing their inputs.

Query transforms are rebased before composing local offsets; precision budgets
cover conversion, scale and nonfinite failures before no-contact acceptance.
The owned tolerance array is refreshed before graph replay. Checker calls are
serialized; unsupported precision cannot become successful geometry evidence.

Collision queries default to CPU, independently of the physical backend and
proposal device. CUDA graph replay is an explicit query option. Clearance is a
capped contact-band estimate, not global minimum distance; execution evidence
records the backend version, precision policy and query identity. Configuration
request identities include the query contract so old checks are not reused as
new evidence. Discrete path/control-boundary checks do not certify continuous
collision freedom.

`motion/_budget.py` supplies context-local cooperative hooks at Robot IK,
MotionGenerator generation and Atomic planning/recovery boundaries. No active
scope means no budget is installed. The host shares quotas with screening,
reserves validation capacity, and preserves exhaustion even when a runtime
converts an exception to a failed action. A native call is checked on return;
cooperative cancellation cannot preempt its internal work.

Regions retain joint coordinate conditions and operation mode. Sampled points,
workspace/IK checks, path screening and measured task success are distinct
outcomes. Independent validation freezes its distribution before trials and
uses separate IID streams. Only completed fixed allocations have marginal
binomial intervals; interruption leaves raw counts without an interval.
Sampling new layouts reruns semantic/geometric checks and never inherits task
success. Region projections must not be recombined into independent ranges.

Use `tests/sim/scene/feasibility`, the geometry planar tests and
`tests/gym/envs/task_program/test_feasibility.py` for production contracts. The
UR5 job in `examples/expansion/scene_robot_feasibility` exercises the configured
path; native qualification and recorded outcomes remain separate from CPU ports.
See the [public API guide](../../../docs/source/api_reference/embodichain/embodichain.lab.sim.scene.feasibility.rst).
