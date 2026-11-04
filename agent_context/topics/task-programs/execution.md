# Task Program assurance and execution

Read this when the request needs these details. [Topic overview](task-programs.md).

## Execution occurrence identity

Cached semantic analyses may serve several repeated calls. The runtime supplies
the execution occurrence ID when grounding each original call; effect monitors,
guards and commands inherit that ID rather than the cached analysis index.
Direct compiler callers may omit the ID to retain workflow/index identity.
Recovery-generated calls retain their separately analyzed workflow identity.

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

`HandOver` covers pickup, transfer, delivery and final release. Its intermediate
acquisition gates require attachment, while the terminal effect requires both
participants detached. Check serialized evidence against the `effect_spec.clauses`
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
Native support and qualified planar-region inputs remain distinct measurements;
missing acquisition cannot become geometric success. Pure task/report matching
stays outside Lab and can consume owned condition data without a runtime.

The simulation factory owns support observer reuse across bridges and installs
the same native sensor instances in the public environment suite. Closing an
environment keeps the demo executor's existing abort handshake, discards/finalizes
recorders, closes runtime hooks, then destroys simulation. Close never guesses an
action acknowledgement or steps physics; incomplete manual execution remains
incomplete, and closed adapters/factories reject further assembly/acquisition.

Sequential segments reuse one static workflow analysis per uninterrupted
sequential stretch; parallel barriers stop this reuse across branches. Each
executed call still grounds from a fresh observation and checks compiler/catalog
ownership. Runtime results capture an append-only history prefix and materialize
independent history tuples when read or serialized, keeping history copying out
of status-only control steps. Synchronous observer or clock failures abort through
the runner's cancel-then-hold boundary and retain any safe-stop failure details.

The environment-owned expert `joint_command_mode` is propagated through the
simulation factory and adapter into `RuntimeCommandFrameEncoder`. Position mode
retains the tensor qpos action. Position-velocity mode emits structured qpos and
qvel targets, requires velocity-bearing joint payloads, and assigns zero qvel to
idle, wait, inactive-row, and abort-safe holds. This changes neither Task Program
language nor the policy action space.

On active rows, normal joint-position frames retain unaddressed qpos targets
by stable environment ID. Inactive rows use measured holds and invalidate
retained targets; explicit safe stops rebase them to measured qpos. A new
episode creates a fresh encoder.

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

## MLLM boundary

`embodichain.agents.mllm.task_program` accepts untrusted JSON, reuses the
canonical decoder/validator/compiler, and requires a trusted host adapter. The
model cannot choose integrations, robot resources, live providers, or
executable lowerers. The current frontend permits the constrained node/call
subset enforced by its decoder; Task Program itself supports the complete
language.

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
`lab.sim.atomic_actions.grip_evidence`; contact arithmetic lives in
`compute.geometry.grip_contacts`. `lab.sim.evidence` is the common frame/batch/request
owner. Existing Atomic public imports refer to the same types. No raw acquisition
module imports Task Program or Task Engine. Invocation and episode associations
are maintained by the execution host rather than fields on shared raw frames.

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
