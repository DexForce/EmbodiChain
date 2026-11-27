# Rolling workload execution

The public owner is `embodichain.lab.expansion.workloads`. This optional port is
independent of task names and downstream operator configuration. Existing
`ExpansionEngine.from_job` and cohort callers retain their interfaces and defaults.

## Capacity and ownership

`WorkloadExecutionCfg` declares W active groups and B physical candidate rows
per group; the supplied environment must contain exactly W times B rows. The
executor creates no simulation World, process or planning thread. It runs once
on the thread that owns its World. `world_info` records actual process, World,
backend, state and solver devices; selecting a visible CUDA device or renderer
does not select the physics device.

`ExpansionWorkloadCfg` declares a unique group, qualified target, attempt cap and
admission wall-time budget. Candidate ordinals and seeds belong to that group,
not its lane or physical row. Admission includes in-flight candidates in the
remaining quota, avoiding extra speculative trajectories credited beyond target.
Another queued group takes the lane when the previous group has drained. A
difficult group does not silently borrow another group's B-row allocation.

## Source ports and native accounting

`run_task_program_expansion` binds a deployment and a closed scene/motion
configuration to `run_gym_expansion_workloads`. It owns bounded pose/follower
proposals, grasp-policy snapshots, original-grid plan transforms, common measured
acceptance and aggregate results. Configuration/viewer loaders do not import the
simulator. Integrators retain task values and provenance inputs rather than
copying these binding components. `derive_grasp_expansion_deployment` exposes the
same immutable-source preparation for configuration preflight.
`resolve_task_program_expansion_bindings` validates sealed fitted policies
and attaches the same four artifact hashes for callers that assemble their own
workloads. Row-local transform factories can adjust one call's typed skill options
before planning, leaving the deployment and other rows unchanged.

`launch_expansion_preview`, `preview_expansion_results` and
`stop_expansion_preview` manage a bounded Linux native service with recorded PID
ownership. The preview reader accepts native aggregate results and derives path
endpoints from recorded motion; downstream workflow-root unwrapping stays with
the integrator. These APIs never recollect or advance physical time.

`run_gym_expansion_workloads` is the public native composition for Task Program
scene/plan workloads. `GymExpansionWorkload` supplies a pinned deployment,
independent collection budget, artifacts and row callbacks. The facade opens one
World, captures the original seeded nominal scene in row zero, and composes the
existing session, frame collector, episode assembler and confirmed artifact sink.
Different integration/grasp policies may share that World only when their
physical configuration, program, cadence and expert action schema agree. The
effective scene-preparation declaration and its parameters must also match:
explicit configuration wins over the environment's registered default, following
`EmbodiedEnv` construction. The common nominal scene is prepared once before
constructing the World.
Command mode, ActionManager declarations and controlled-joint selections cannot
vary between workloads in one World. Selections belong to the resolved environment
configurations, not the shared registered environment implementation class. Physical/visual/prototype
operators remain on the cohort API; unsupported payloads fail explicitly.
Explicit contact sampling conventions may differ per integration through
`SimulationTaskProgramAdapterFactory.create_adapter(grasp_contact_frames=...)`;
the declared frame must match its grasp service, while TCP, hardware geometry
and drive/command bindings remain strict and the live robot is unchanged.
Only the initial nominal reset may run scene-settling events. The facade then
freezes reset events; rolling preparation is checked against the physical step
counter so another row cannot advance outside the World-owner loop.
Frozen groups export through one bounded writer thread without World access.
Fresh output is required; this facade does not resume interrupted collection.

`WorkloadExecutionCfg.reuse_idle_rows` optionally lends unused physical rows to
unfinished groups; capacity remains exactly W times B. Queued groups keep their
normal lane priority. Healthy candidates fill only their remaining quota. Each
observed unsuccessful attempt permits one additional competing candidate, bounded
by the existing attempt, admission-time, settling, step and pending-byte limits.
The low-level flag and native Task Program job both default false. Enable
reuse explicitly with `engine.reuse_idle_rows: true`. On
CPU-bound planning, competing retries may increase total work despite higher
row occupancy; validate qualified output per wall time before enabling reuse.

This mode requires a source `cancel(result, *, reason)` port. Only a confirmed
receipt fills quota. Remaining live candidates use the ordinary safe-stop
handshake; unstarted warming rows are cancelled without fabricated commands.
Already completed peers retain their canonical program acceptance in cancelled
diagnostics; they do not become physical failures or successful persisted rows.
The native source retains measured T+1 partial buffers and plan traces, then
releases the native assignment. Cancelled rows earn no receipt or quota credit.

`WorkloadSource.prepare` runs inside the selected-row `episode_prepare` reset
callback. It must not step physics, alter an active peer or mutate World-wide
appearance/resources. Incompatible topology or shared resources require a
different deployment, rather than pretending they are independent rows.

After at least the declared physical settling substeps, `activate` captures the
initial measured boundary, performs the source's required preparation checks,
registers a native candidate and uses `ExpansionEngine.admit_row`. It returns a
fresh bridge bound exactly to that row and the environment's control cadence.
Rejected preparation returns None and never earns quota. The native source retires
warming rows when the admission time or independent rollout budget is exhausted.
It records the stop reason before proposal, including exhaustion detected during
initial-state validation. The group can finish incomplete while already-admitted
peers continue and export normally.

Settling rows remain controller-active so Gym consumes their explicit reset
hold targets. They are not live trajectories: only cursor rows receive source
observations, command acknowledgements, trajectory lengths or receipt credit.
Using Gym's inactive measured-qpos substitution during settling would remove
restoring drive error on every tick and change the admitted initial posture.

`started` calls `mark_row_started` after the first actual command. `observe`
captures every active T+1 boundary and performs the original physical checks;
it returns only active rows needing native safe rejection. The executor retains
the bridge acknowledgement/hold handshake. Preparing and completed peers cannot
fabricate actions or observations for another trajectory.

`complete` receives the canonical bridge result and exact first/last World
control indices. It assembles the original source evidence and calls
`ExpansionEngine.complete_row`. Only that confirmed receipt earns a qualified
count. None is a rejection, never a placeholder success. Unavailable or ambiguous
writes retain the ordinary native stage/reconciliation contract and stop the
rolling owner from issuing unsupported quota credit.
`ExpansionSession.take_ready(speculative=True)` may reserve competing running
rows while quota is unfilled. Required physical checks are unchanged, and
pending writes plus confirmed commits can never exceed the original target.
`ExpansionEngine.cancel_row` cannot release pending persistence; its existing
reconciliation path still owns uncertain or failed writes. Safe-stop controller
values retain the interrupted invocation/phase metadata for ordinary physical
validators, while their cancellation kind and reason remain explicit.
When a runtime is already terminal with a final unconsumed motion, the joint
controller owner supplies its actual already-applied position targets and zero
velocity through the abort callback's optional `controller_hold`. The bridge
prefers a runtime-generated hold, never relabels pending motion, and advances
the clock only after the fallback hold receives a real `env.step`. Without an
explicit owner-proven hold, the existing fail-closed abort behavior remains.

Frozen groups can call `export_expansion_episodes` from bounded writer threads.
The callback must not retain mutable live state or operate on the World.
Partial quotas remain explicit failures even when some episodes were exported.

## Planning and validation

The optional cooperative planner merges compatible fixed-grid motion requests
and pure FK/IK/Jacobian calls. Budget-bearing whole-motion requests keep their
original execution so charge semantics are preserved. Solvers declaring the
strong sequential-path capability also merge compatible native path requests,
retaining each row's preceding-valid seed and independently owned results.
Context variables and
candidate random streams remain row-local; results are independently owned.
Default CPU command gathering shares native numeric reads through a bounded
World-owner scope in `sim/objects/backends/_numeric_observations.py`. It ends
before completion callbacks and physical stepping, and starts after reset and
activation. Checked native writes, physics steps and binding topology changes
invalidate samples. Requests for matrices of a named articulation link can read
just the selected CPU arenas; the result uses the same arena frame and float32
rotation conversion as the full link batch. Derived dynamic rigid poses share
the same boundary and invalidation, with separate CPU autocast representations.
Consumers receive independent output
buffers; semantic
state, episode clocks and validation remain row-owned. CUDA and other backends
retain native reads. Raw SDK mutation or concurrent World writes inside the
read boundary are outside this internal side-effect-free planning contract.
Semantic programs and some trajectory processing still run per row. Per-control
planning readiness is a barrier; this API does not claim complete vectorization
or universal throughput improvement.

`NewtonCollisionChecker.check_batch` uses the same physical geometry, pair
exemptions, halfspace supports and precision/saturation rules as scalar checks.
It returns discrete row reports, without claiming continuous clearance. Its
query model is never stepped and is not another simulation World.
`MeasuredTaskMonitor.batch_scope` shares measured numeric inputs while preserving
each native event state machine and receipts. `ContactPermissionChecker` checks
all valid native contact slots with the original applied-impulse and loss gates;
hosts still own sampling at every required physics substep.

`MeasuredExpansionValidator` uses these same batch interfaces. Compatible rows
share one immutable geometry declaration/checker; selected rows are compacted
into bounded Newton queries and mapped back in their original order. Different
geometry retains separate checker groups. Query models never step physics.
Predicate inputs share the current complete measured snapshot, while events,
consecutive counts, failures and contact holds remain row-owned. Shape snapshots
are needed only by declared support-bound predicates; native-contact rows do not
gain a homogeneous-topology requirement.

The built-in joint transport writes all of a frame's destinations into one owned
output. Single commands, subclasses, replaced encoder methods and custom
transports retain normal dispatch and registration order. Numerical/frame checks,
inactive holds and position-velocity evidence remain unchanged. Episode assembly
shares validated immutable segment metadata internally; mutable readback still
owns its containers. Durable JSON uses the same finite sorted representation and
fsync/atomic publication, with one encoded text allocation.

Default GPU dynamics use Direct GPU reporting, whose static contact identity
limitations remain explicit admission constraints. Physics devices and task
qualification must be reported separately. Existing CPU deployments do not
switch devices.

## Optional task-pose proposal feedback

For native Task Program jobs, `trajectory.affordance.branch_selection` defaults
to `round_robin`. With `outcome_guided`, the workload assigns each non-nominal
grasp rank once before using physical outcomes to rank future choices. Once a failure
has been observed, ranks are scored by (successes + 1) / (trials + 2).
If every rank has a physical failure and none has finished successfully, the
one with the most verified Task Program segments takes priority.
`scheduling.exploration_fraction` (default 0.25) reserves a fraction of the
remaining selections for the least-used ranks. Each motion recipe records its
chosen rank before execution. After a successful branch, exploitation can also
record its source ordinal and accepted object-relative grasp matrices. Each
selected call looks for an equivalent non-nominal pose in its freshly generated
valid candidates and replans against the new scene. Exploration uses fresh
ranked choices. Selected calls in the recipe reuse its declared branch.
Only physical outcomes with a recorded plan transform count once per attempt.
Preparation failures, queue limits, and cancelled rows supply no evidence.
Feedback is local to one workload run.

Task-facing expansion configs select `scene_sampling.strategy` (`bounded_pose`
or `workspace_joint`) independently of `scheduling.policy` (`fifo` or
`task_pose_coverage`). Absent settings preserve the legacy bounded sampler.
`engine.nominal_seed` optionally fixes the authored scene reset independently of
the expansion seed. Without it, both retain the historical shared seed. Runtime
candidate streams and proposal streams still use the expansion seed/ordinal;
the reference manifest records both seeds. This prevents a sampler comparison
from silently changing the nominal scene's reset randomization and moving its
declared local domain. Rolling resets use the same explicit nominal reset seed.
The raw pose budget enforces the original admission deadline. An admitted
trajectory's planning view retains shared IK/planning call limits and explicit
cancellation without treating that deadline as a new mid-trajectory hard stop.
The original max_steps, physical checks and safe-stop handshake still apply.
`_pose_sampling.py` owns bounded candidate snapshots, source feedback and budgets;
`task_pose_coverage.TaskPoseCoverage` implements the session coverage protocol.
Measured rigid initial/terminal poses contribute only after native sink receipts.
An optional `pose_coverage.measurements` mapping selects entity states at initial,
terminal or an explicitly named successful segment end. Its grids use the same
names. This supports held-object and multi-stage tasks without creating fake
place-reference entities. `fixed: true` contributes one category inside its
declared grid; tracking noise does not create placement novelty. Proposal ranking
predicts only initial cells; later boundaries require measured T+1 evidence.
Separate SceneCase identities aggregate within a declared configuration/grid
fingerprint; installation changes remain separate domains. FIFO can omit coverage.

Joint sampling uses an explicit cache or joint-uniform FK prior and repeats
full-pose IK/FK checks after planar projection. Fixed pick grasp policy and prior
must agree; unsupported dynamic grasp replacement rejects. Normal scene preparation,
planning, measured acceptance and persistence still determine success. Feedback
checkpoints do not restore a live World or replace job receipt reconciliation.
The configured acceptance task is
`embodichain/extensions/tasks/configs/tasks/manipulation/workspace_pose_pick_place`;
its four-way runner is `scripts/benchmark/expansion/task_pose_coverage.py`.

## Conditional layout proposals and task screening

`scene_sampling.strategy: scene_first` adds a prepared support-tree layout source
before ordinary native execution. `_scene_first_sampling.py` reuses shared scene
preparation/conditional generation and `RobotFeasibilityBackend`/`plan_task_mode`;
it creates no physics World and owns only the isolated collision-query resources.
`layout` and `task_mode` are explicit, validated task declarations. Root layout
variables currently use XY only; selected nested members retain declared XY/yaw.
Placement checks group bounds on the main support. For a bounded leaf root
overhanging that rectangle, preparation can verify the projected centroid
against contact patches at the authored height on the named support. Full
projected extents still determine clearance to other groups.
All layout-screened operators use native convex shapes and primitives. Physical
triangle colliders on rigid objects and robot links are decomposed in their scaled local frame for the isolated
Newton phase/path query. V-HACD handles volumetric concavities; planar meshes
retain their footprint and holes. `scene-first-checks.json` records part counts
by physical collider in `mesh_convex_parts`. Rollout contacts use the
simulator's original physical shapes.
Homogeneous replicas support rolling W times B collection, including eight rows.
Preparation snapshots the common prototype geometry, installation and nominal
full-joint planning start before any row executes. The feasibility adapter uses
an explicit solver row and never treats an active peer's live posture as its
planning start. It creates no additional physical World. A `bounded_pose` job may opt into the
same layout/task-mode screens for a controlled comparison without changing
ordinary simple-sampling defaults. Screening budgets are bounded and separate
from fixed physical quality thresholds. Prepared candidates stay immutable by
ordinal; failed screening cannot turn into an accepted simple fallback.
Static policy and task-mode decoding use shared simulator-free declarations.
Authored anchors must satisfy the prepared layout domain. An out-of-domain
bounded draw rejects only that proposal. Followers outside the support tree
inherit the final parent's displacement before task screening.

`robot_first` anchors one movable support-tree root per proposal from the
workspace of a declared `robot_anchors` phase/part target. It uses the target's
initial rigid relation to the root and projects current-arena TCP XY onto the
root's allowed XY domain, preserving authored height and rotation. Per-part FK
populations are prepared once and selected across occupied cells. The layout
generator reserves the anchor and completes the remaining groups. A prior
without an in-domain sample is a budgeted rejection.

`bidirectional` alternates scene-first and robot-first by raw ordinal, including
rejections. Attempt budgets, FIFO or task-pose coverage selection, execution and
receipt rules stay in their shared owners. The source records the direction of
each draw and selection, and records the requested anchor separately from the
generated and measured poses.

The current geometry, joint limits, full-pose FK residuals, full-state/mimic
branches and synchronized path collision checks remain in their existing owners.
Actual execution replans and measured acceptance/persistence remain mandatory.
`scene-first-checks.json` retains phase results, workspace prior and anchor
evidence, and the unchanged World clock;
the query is closed before the owner environment, including setup failures.
The reproducible cross-task runner is
`scripts/benchmark/expansion/scene_first.py`; per-task fixtures are under the
existing grab/place, stack and dual-arm RGB sorting task configurations. The
sorting program operates the two arms sequentially and retains its original
segment validators; receipt-based coverage counts both manipulated blocks and
does not turn fixed-goal tracking noise into placement diversity.

## Focused validation

`tests/lab/expansion/test_workloads_sim.py` runs a real four-row task through
native session admission and artifact receipts. It covers rejected-row safe
holds, refill, queued groups, no missing control boundaries, scalar/batched task
predicates and compatible whole-motion result parity. Geometry and contact
coverage lives in the existing native feasibility/contact suites. Per-episode
export uses the real LeRobot SDK and checks portable T+1 evidence.
