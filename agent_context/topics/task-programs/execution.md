# Task Program assurance and execution

Read this when the request needs these details. [Topic overview](task-programs.md).

## Execution occurrence identity

Cached semantic analyses may serve several repeated calls. The runtime supplies
the execution occurrence ID when grounding each original call; effect monitors,
guards and commands inherit that ID rather than the cached analysis index.
Direct compiler callers may omit the ID to retain workflow/index identity.
Recovery-generated calls retain their separately analyzed workflow identity.

`TaskProgramEnvironmentAdapter.create_bridge` accepts immutable
`InvocationPolicyBinding` declarations sealing a named profile preset to a
compiled-program fingerprint and one `program_id/segment_id:segment_call_index`
execution ID. `task_program_fingerprint` preserves the existing nominal-source
identity algorithm. `BoundInvocationPolicies` rejects stale programs, duplicate
or unknown occurrences and parallel-branch targets before execution. Unbound
parallel execution retains its existing behavior.

Static preflight and the bridge's cached sequential analysis use the same
occurrence-aligned preset selections. The compiler links each selected call
again before analysis and grounding; it does not patch an already bound call or
mutate the Atomic engine. Selected presets must preserve the original effect
assurance. Their existing action options, motion, tracking, recovery and runner
settings own that call; unselected calls retain the host's runner override and
normal preset resolution. Re-grounding an original call retains its selection
and occurrence ID; derived workflow-recovery calls use their own identity and
normal policy selection. These are typed integration declarations, not new
Task Program language fields or configured-loader extensions.
The manifest check owns profile/preset validation. Scene registry IDs belong to
the host selection, not `SceneManifest`; the standard adapter checks them, while
a host constructing `TaskProgramDemoBridge` directly must validate scene ownership.

## Effect assurance and acceptance

Every `SkillPolicyPreset` explicitly selects one authority:

- `verified`: semantic state advances only from measured effect evidence;
  curated Pick/Place/HandOver calls require monitor mappings.
- `projected`: monitor mappings are forbidden and the action plan's expected
  symbolic effect is projected after command completion.

Projected execution proves the intended trajectory path completed; it does not
prove physical task success.

Keep these boundaries separate:

1. effect monitor: one Semantic Call's physical postcondition and recovery;
2. segment post-policy: environment advancement such as settling; and
3. segment validator: task or dataset acceptance.

Native grip evidence is an explicit simulation integration in
`integrations/simulation/grip_evidence.py`. Its factory owns immutable object/pad
declarations and creates the standard control-part provider through its public
observer hook. The semantic adapter translates clauses to the physical
`atomic_actions.grip_evidence.GripContactObserver`; sensors own acquisition/history
and compute owns bilateral contact arithmetic. It retains provider IDs and physical algorithm revisions; revision
2's open-hand interpretation requires explicit driver calibration and opt-in.
No task labels or control-part suffixes select a policy. Sampling/timeline lifecycle
and per-row availability belong to the [contact boundary](../sensor-system/contact-history.md).
Raw traces are observations, while report serialization and TaskSpec verdicts
remain frontend responsibilities.

Stateful segment validators capture their incoming evidence through the optional
`begin_segment` port immediately before runtime planning starts. Lift checks use
that segment's measured baseline, including for lazy and repeated segments;
they do not observe at compile time or reset physical state. Results remain
row-local and exclude rows that were ineligible when the segment started.

`HandOver` starting from free resources covers pickup, transfer, delivery and
final release. Its intermediate acquisition gates require attachment, while
the terminal effect requires both participants detached. With an already
verified source hold and an explicit policy preset selecting
`HandOverOptions.release_at_target=False`, it transfers to the destination and
leaves that resource held. The compiler never rewrites this option from live
state. Static analysis and fresh grounding reject a mismatched holder or missing
held state in eligible rows. Check evidence against the `effect_spec.clauses`
for that trace's boundary; the participant name and overall call status do not
determine the expected raw value.

`TaskProgramDemoBridge` never calls `env.step()`. It yields lazy
`DemoSegment` values so the ordinary Gym executor retains stepping, recording,
reward, reset, and persistence. Final success is published only after every
segment lifecycle completes normally.

Named geometric/held/support stability conditions execute through the standard
`SimulationSegmentPolicyPort`; they do not install another constraints registry.
The initial sample does not count toward duration. Ordinary drift resets the
object/reference anchors, while a held-state failure stays latched for the row.
Joint stability uses the actual joint coordinate and full-window range, holding
the original observed robot position. Its receipt does not claim contact safety.
The pure ``compute.trajectory.PositionRangeMonitor`` owns candidate-window
statistics; the port owns observation and hold scheduling. An execution clock
adds the final raw sample and a bounded ``joint_window`` record with actual
steps/times, per-sample validity/errors and policy/segment/native identity.
The pre-sample range anchor is retained separately as numerical initialization.
Incomplete retention is explicit and cannot prove full coverage. Unclocked
default execution retains its legacy receipt without fabricated timestamps.
Execution termination and per-row condition status remain separate. The default
invalid-row mode aborts the batch; explicit isolation lets other rows continue.
Cancelled/interrupted waits keep partial diagnostics and no success result.
Native support and qualified planar-region inputs remain distinct measurements;
missing acquisition cannot become geometric success. Pure task/report matching
stays outside Lab and can consume owned condition data without a runtime.
Mesh-envelope math belongs to Compute; qualified contact-height adaptation
belongs to sensors. Empty attachment selections perform no Robot query, and
multi-arm attachment checks reuse one inverse without changing FK or sampling.

The simulation factory owns support observer reuse and installs the same native
sensor instances in the public environment suite. A support-enabled factory
rejects overlapping unfinished bridges before replacing the clock. The adapter
checks ownership before live assembly and binds the actual returned bridge.
`execution_finished` distinguishes safely ended iteration from successful
`program_completed`: outstanding lazy segments and unacknowledged actions keep
ownership until consumed/validated or explicitly aborted. Non-support factories
retain existing assembly. Returned effect providers, live scene snapshots and
drive-target holds share the factory lifetime guard. Closing an
environment keeps the demo executor's existing abort handshake, discards/finalizes
recorders, closes runtime hooks, then destroys simulation. Close never guesses an
action acknowledgement or steps physics; incomplete manual execution remains
incomplete, and closed adapters/factories reject further assembly/acquisition.

Sequential segments reuse one static workflow analysis per uninterrupted
sequential stretch; parallel barriers stop this reuse across branches. Each
executed call still grounds from a fresh observation and checks compiler/catalog
ownership. Runtime results capture an append-only history prefix and materialize
independent history tuples when read or serialized. `SemanticCallExecutor.advance`
advances the same due cycle as `step` without constructing a result facade;
the bridge reads a result at terminal audit boundaries. Synchronous observer or clock failures abort through
the runner's cancel-then-hold boundary and retain any safe-stop failure details.

The environment-owned expert `joint_command_mode` is propagated through the
simulation factory and adapter into `RuntimeCommandFrameEncoder`. Position mode
retains the tensor qpos action. Position-velocity mode emits structured qpos and
qvel targets, requires velocity-bearing joint payloads, and assigns zero qvel to
idle, wait, inactive-row, and abort-safe holds. This changes neither Task Program
language nor the policy action space.

On active rows, normal joint-position frames retain unaddressed qpos targets
by stable environment ID. Simulation factories default to
`hold_qpos_source="observed"`, reusing captured qpos without extra robot reads.
Explicit `drive_target` mode supplies full controller targets for unaddressed
and inactive joints. Other providers retain the measured-qpos fallback unless
they supply `hold_qpos`. Inactive rows
invalidate retained targets; explicit safe stops still rebase them to the
observed qpos. A new episode creates a fresh encoder.

Simulation planning observations optionally capture the arena-local robot root
pose through the public `capture_root_pose` keyword. It is acquired in the same
observation and copied into `RobotObservation`; the default performs no root
query and leaves that field unset.

The bridge declares `progress_total_steps` only for deterministic open-loop
Pick/Place segments with fixed interpolation samples and no recovery, runner
holds, feedback settling, post-policies, or parallel execution. It links only
the current segment's calls to their policy presets; progress display never
performs full workflow analysis or downstream look-ahead. Handwritten
trajectory tasks use the same `DemoSegment` field for a known fixed trajectory
and settle suffix; paths whose emitted action count depends on runtime state
remain indeterminate in both execution styles.

## Demonstration outcome and persistence

The bridge's accepted mask remains the sole segment-quality authority. The
common demo executor records it in `DemoSegmentResult.successes`, classifies the
first authoritative failure phase in `outcome_kinds`, and writes
`segment_accepted`, `segment_attempt_id`, and `continuity_id` for every real
rollout frame. Task Program segment metadata uses `task_program_id`; LeRobot
fragment sidecars copy it to the provider-neutral `source_program_id` field.

`DemoExecutionCfg` is collector-owned, not part of the Task Program language.
Its default `continuous` mode preserves episode semantics. Its
`segment_fragments` mode persists eligible naturally executed segments as
independent LeRobot episodes, with failed fragments requiring explicit opt-in.
Current execution remains fail-closed and every frame has `continuity_id == 0`;
checkpoint capture, state restore, and suffix resume are deferred. Recorder
commit, deduplication and partial-failure behavior are owned by
[data persistence](../data-pipeline/persistence.md).

## Parallel and registered calls

Parallel execution requires disjoint resource claims and runtime targets,
control-grid alignment, conflict-free symbolic writes, an authoritative
`ParallelCommandSafetyValidator`, and an explicit fail-fast barrier. Resource
disjointness alone is not physical safety evidence. Nested parallel blocks are
rejected.

A registered call payload is declarative and executable-free. The immutable
integration must provide exactly one matching lowerer factory and fingerprint
its call ID, revision, and target Atomic Skill descriptor. Use this extension
to expose shared Atomic Skills; do not place task-local motion generators in a
lowerer.
Registered held effects use compiler-owned expectations, monitor grounding,
acquisition/release gates and retention guards. An attached retention-only
contract consumes an exact verified input hold for every eligible row, requires
a verified monitor, and guards the declared segments without a new acquisition
gate. Routed held-object transport uses this contract and its named Atomic goal
requests terminal retained-attachment verification.
A registered release may retain
its incoming verified hold through declared approach phases and require measured
detachment before retracting. Relative Place declares the same checks as ordinary
Place. Runtime validates every guard phase against each successful final plan,
after optional transforms and before dispatch, including retries. Failed plans
retain their original planning diagnostics. A lowerer may explicitly preserve
symbolic state only when it declares no effect contract. Declaring a held
effect or phase does not certify physical attachment or weaken the selected
assurance; verified acquisition still requires its measured monitor. A release
or transfer target contributes to pickup look-ahead and then ends that chain.

## MLLM boundary

`embodichain.agents.mllm.task_program` accepts untrusted JSON, reuses the
canonical decoder/validator/compiler, and requires a trusted host adapter. The
model cannot choose integrations, robot resources, live providers, or
executable lowerers. The current frontend permits the constrained node/call
subset enforced by its decoder; Task Program itself supports the complete
language.

`TaskProgramPlanLifecycleFactory` is the provider-free selection protocol for
an explicit grounded occurrence. Its composite permits at most one lifecycle
owner and rejects duplicate factory registration. Runtime hosts bind this port
explicitly; source-specific lowerers and services remain with their owning
integration layers. Semantic preflight checks planning recovery against the
actual target SkillDescriptor before grounding or physical acquisition.

## Evidence ownership across integrations

Task Engine selects declared evidence requirements and named policies from skill,
effect and TaskSpec contracts. It emits deployment declarations through the public
configured deployment loader; it does not construct sensors, infer pad links from
robot names, interpret impulses, or mutate runtime provider state. Robot/hand
calibration belongs to embodiment runtime services; task object mappings belong
to the integration and execution thresholds belong to the selected policy.

The integration resolves those declarations against exact native owners and
adapts semantic clauses to shared physical requests. Sensor acquisition/history
lives in `lab.sim.sensors`; physical grip interpretation lives in
`lab.sim.atomic_actions.grip_evidence`, including `DriverOpeningBinding.bind`
for calibrated hand-opening observations. Factories bind those declarations
before creating sensor resources; release predicates and experiment labels
remain outside raw acquisition. Contact arithmetic lives in
`compute.geometry.grip_contacts` and `compute.geometry.contact_support`.
`lab.sim.evidence` is the common frame/batch/request owner. Existing Atomic public
imports refer to the same types. No raw acquisition module imports Task Program
or Task Engine. Invocation and episode associations are maintained by the
execution host rather than fields on shared raw frames.

Changes after the contact-evidence extraction must import the sensor modules
rather than the removed integration-local contact_evidence/contact_timeline paths.
Factories may create or reuse matching sensor declarations; lifecycle, scope,
capacity, unavailable data and selective resets remain sensor contracts. Do not
turn missing acquisition into a measured negative or a successful effect.

For declared support conditions, incomplete acquisition resets only the affected
row's stable streak and keeps yielding holds until a qualified window or timeout.
`measurements.support_acquisition` retains the physical channel's validity,
errors and timestamp/row/revision identity separately from source support metadata.
The result records missing-observation counts and the last reasons. A row that
never reestablishes a successful window after missing evidence reports
`support_result_available=false`; consumers must not interpret that false result
as a measured task failure. Source-global errors still invalidate every witness.

`SimulationSegmentPolicyPort.observation_clock` exposes only the requesting
session's current observation binding; an unbound port returns None and a
superseded session rejects new reads. Same-clock binding is idempotent and the
generation belongs to the existing owner lifetime, not a global active-run
registry. Default execution, cancellation and cleanup retain their own protocols.
The environment adapter requests a fresh policy/validator session for every
bridge, including ordinary execution without nominal sources. Creating a later
bridge cannot replace the clock stored by an earlier session. Position-only
post-policy holds pass through the configured Gym action encoder, preserving the
captured positions and supplying zero velocity targets in position-velocity mode.
