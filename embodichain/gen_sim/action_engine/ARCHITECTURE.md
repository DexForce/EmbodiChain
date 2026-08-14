# Action Engine v2 Architecture

Action Engine v2 uses a task-first protocol and executes a direct
`AtomicAction` graph. The persisted graph is symbolic and coordinate-free;
simulator geometry is resolved immediately before each action executes.

The phase-one collaboration entry point wraps that existing pipeline with
three narrow owners:

1. `TaskAgent` produces three scene-independent `TaskDraft` candidates and
   deterministically derives each `SceneRequest` and `SuccessSpec`.
2. `SceneAdapter` binds one verified candidate to an existing scene or exact
   content-addressed `ScenePackage`, producing `SceneManifest`, `RoleBindings`,
   and a complete `BindingReport`.
3. `ActionAgent` lowers the selected `GroundedTaskPlan` to the existing
   `action_engine_seed_graph_v3`, performs executable capability preflight,
   runs it through `ProgramExecutor`, and emits a tensor-free
   `ExecutionReport`.

The public CLI is `embodichain gen-sim-task import-scene|prepare|run`. This
layer does not modify Scene Engine and continues to publish all legacy bundle
artifacts for existing runners.

## Package Ownership

The collaboration workflow is split by ownership rather than nested under
Action Engine:

- `embodichain.gen_sim.task_engine` owns scene-independent interpretation,
  E1-E9 semantic ontology, `TaskDraft`, `SceneRequest`, `SuccessSpec`, and
  `TaskAgent`.
- `embodichain.gen_sim.scene_engine` remains the existing scene generation
  subsystem and is not modified by the collaboration workflow.
- `embodichain.gen_sim.action_engine.agent` owns `ActionAgent`; Action Engine's
  existing `domain`, `planning`, and `runtime` packages remain authoritative
  for graph compilation and execution.
- `embodichain.gen_sim.collaboration` owns cross-engine contracts, scene
  adaptation, the content-addressed scene store, orchestration, artifacts, and
  the unified CLI.

The former `embodichain.gen_sim.action_engine.collaboration` namespace is a
deprecated import bridge. It contains no workflow implementation and may be
removed after downstream callers migrate to the owning packages above.

## Data Flow

1. `TaskFactory` or a caller creates a validated `TaskSpec`.
2. Action Engine emits `SceneRequirements` for the external Scene Engine.
3. After scene generation, Action Engine validates role-to-UID bindings,
   affordances, initial state, cameras, and spatial requirements.
4. Offline recipes and the online planner independently create complete
   `SeedGraph` candidates whose nodes are `AtomicAction` calls.
5. Runtime preflight checks the capability catalog and rejects planning-only
   actions before simulator motion starts.
6. `ActionGrounder` reads live robot, object, articulation, and camera state and
   materializes the typed goal and immutable action options just in time.
7. `ProgramExecutor` schedules the DAG, executes vectorized action masks, and
   verifies semantic postconditions from live state.

There is no persisted semantic task graph between `TaskSpec` and `SeedGraph`.
The mature five-task compiler remains only as an input migration adapter for
regenerating current tasks; it publishes a v2 graph and never publishes
`task_agent.json`.

## Protocols

### TaskSpec

`TaskSpec` owns the level, public instruction, E1-E9 task instances,
dependencies, path-independent success conditions, and a private oracle. The
online planner receives `public_task_spec(...)`, which removes the oracle and,
for L4, the hidden reference task instances. Online L4 TaskGroups are inferred
from the instruction and observations rather than matched to an oracle path.

Levels classify how the task is specified, not action count:

- L1: one E instance.
- L2: two or more instances of the same E type.
- L3: two or more different E types explicitly composed.
- L4: an abstract instruction that requires memory, visual semantics, pattern,
  logic, common-sense, or constraint reasoning.

Free-language L1-L3 generation uses two structured model calls. The first sees
only the instruction and E1-E9 catalog and emits typed steps whose scene
selectors are open natural-language references. The second sees those
references plus a coordinate-free semantic inventory and may return only
existing scene UIDs, status, and confidence. Local validation enforces complete
request coverage, candidate roles, cardinality, confidence, and non-self
targets; unresolved or ambiguous references fail instead of being guessed.
The optional `deterministic` instruction parser is an explicitly selected,
finite-vocabulary offline compatibility adapter. It is not imported by either
LLM stage and is never used as an implicit fallback.

The older public `planning.plan_task` adapter still accepts an LLM-produced
`TaskAgent`, but it does not reinterpret the instruction after that structured
output exists. Axis, orientation, and arm-allocation fields come only from the
validated model result. Its former keyword fallback is rejected; callers that
need the bounded offline parser must select `tasks.deterministic` explicitly.

### SceneRequirements

`SceneRequirements` is the JSON hand-off to the external Scene Engine. It
declares object roles, counts, categories, affordances, initial states, spatial
constraints, camera requirements, and distractors. Scene results are never
silently repaired. Structural contradictions and explicit affordance
contradictions invalidate the task instance; an absent affordance declaration
remains unknown and is deferred to runtime physical validation.

The current tabletop importer has one deliberately narrow structural contract:
exactly one `background` object is the support surface and receives runtime UID
`table`; every movable object is assumed to begin on that surface. Zero or
multiple backgrounds are rejected rather than resolved from position, UID, or
description text. Semantic `category`, `color`, and `attributes` come only from
their explicit scene fields. Physics `attrs` are not semantic metadata.

For task-first inputs, explicit role bindings are authoritative unless they
contradict metadata that the scene actually declares. Automatic role binding
requires a unique match with complete structured category, attribute, state,
and affordance evidence. Object names and descriptions remain available to the
LLM grounding call, but deterministic validation never searches them for
semantic substrings.

### SeedGraph

Every node directly names an `atomic_action`, scene `object_uid`, symbolic
`target_binding`, actor, control, dependencies, resources, pre/postconditions,
motion policy, E type, and `task_instance_id`. `TaskGroup` groups all nodes of
one E instance with `role=primary|recovery`; it is metadata over the same DAG,
not a second graph.

Validation guarantees:

- node and TaskGroup dependencies are DAGs;
- every node belongs to exactly one TaskGroup;
- E groups contain their required core actions;
- concurrent nodes do not claim the same exclusive arm/object resource;
- object references resolve to scene UIDs;
- world poses, qpos, trajectories, grasp poses, and waypoints are rejected
  recursively;
- hashes use canonical strict JSON and are stable across processes.

The production loader accepts v2 graphs only. A v1 graph, whether supplied as
JSON or an in-memory mapping, receives an explicit regeneration error rather
than an implicit migration.

## Capability Boundary

`AtomicCapabilityRegistry` is the single runtime catalog. A descriptor declares
the action/option types, accepted symbolic bindings and controls, resource mode,
held-object state effect, target and config materializers, verifier, failure
classifier, retry mode, and runtime availability.

The executable catalog currently contains:

- `PickUp`, `MoveHeldObject`, `MoveEndEffector`, `MoveJoints`, and `Place`
- `Press`
- `CoordinatedPickment` and `CoordinatedPlacement`
- `HandOver`

`Pour`, `PullArticulatedPart`, `PushArticulatedPart`, and `TurnKnob` are
planning-only until matching lower-level implementations exist. They can be
generated and statically checked, but preflight fails before any motion with
the descriptor's unavailable reason.

Adding an executable skill consists of registering its descriptor and reusable
materializer/verifier hooks plus focused tests. Planner and executor dispatch
do not maintain a parallel action-class table.

## Offline And Online Planning

Offline recipes deterministically instantiate E1-E9 task instances. Current
task mappings are:

- `place_relative -> E1`
- `orient_object -> E2`
- `coordinated_transport -> E5`
- every member of `build_stack` and `arrange_line` -> one E1 instance

E5 uses `coordinated_transport` only as the semantic task-group operator. Its
motion graph contains one `CoordinatedPickment`; a `place` terminal behavior
adds synchronized left/right `MoveJoints(gripper_open)` nodes. The executor
clears coordinated hold state only after both grippers are observed open.

The online path first extracts auditable visual facts from multi-view RGB and,
when available, depth and camera calibration. Facts contain only known UIDs,
normalized bboxes/keypoints, canonical spatial relations, task predicates, and
confidence. Spatial relations use a shared ontology and fixed participant
order. Task-level judgments such as visual or pattern completion are accepted
only when the current `TaskSpec.success` explicitly requests them. A second
structured call produces a complete direct `AtomicAction` graph. Prompts
request facts and graph JSON only; hidden chain-of-thought is neither requested
nor stored.

Image-space constraints may use normalized keypoints, masks, bboxes, and
relative relations. The Grounder uses live depth and camera calibration to
convert them to world targets. The SeedGraph never stores that result.

## JIT Grounding

Each action is grounded again immediately before planning/execution. Grounding
therefore observes object displacement, current qpos, current held-object
ownership, articulation state, and fresh camera measurements. Coordinated and
handover actions are grounded as synchronized execution units. Automatic arm
selection, collision checks, live arrangement slots, and current predicate
semantics remain deterministic runtime responsibilities.

## Mainline Planning Contract

The runtime keeps only an Action Engine-local `ExecutionState` for full-robot
qpos and held-object relations. Each plan converts that state to the mainline
`PlanningContext` (`RobotObservation`, `TaskState`, and `SceneSnapshot`) and
submits an `ActionInvocation` to `AtomicActionEngine`. The returned
`StateDelta` remains speculative until physical and semantic verification; only
verified vectorized rows are committed.

Single-arm arm motion uses cuRobo `motion_gen` by default. Hand-only and
coordinated dual-arm actions use `ik_interp`, because mainline coordinated
primitives do not support cuRobo motion generation. A failed single-arm cuRobo
row may fall back to `ik_interp` without replacing successful rows. Generated
background objects form the static cuRobo collision world; dynamic obstacles
are an explicit runtime-policy opt-in.

Generated mesh objects carry V-HACD settings in both the current shape-level
schema and legacy top-level fields. Before antipodal grasp construction, the
runtime prepares a checksummed V-HACD payload at the shared collision-checker
cache path so the unchanged mainline checker does not silently recompute CoACD.

## A/B Evaluation

Test mode retains both candidates. `run_strict_ab` creates distinct offline and
online environments with the same task, scene configuration, seed, Grounder,
verifiers, and retry policy. Both environments reset before execution and a
digest over robot qpos and object state must match exactly; a mismatch aborts
before either branch executes.

Artifacts are written under `offline/` and `online/`, with a shared
`comparison.json`. The comparison records graph hashes/differences, action and
path lengths, success, retries, recoveries, revisions, latency, record paths,
and planner/VLM metadata supplied by each candidate.

L4 A/B runs must supply a private-oracle evaluator. The built-in evaluator
checks memory reconstruction, visual completion, pattern completion, numeric
selection, functional placement, and stable/unobstructed goals from the final
state only. The comparison labels whether success came from runtime step
postconditions or the private oracle.

## Dynamic Recovery

The persisted `SeedGraph` is immutable. `RuntimeGraph` keeps a detached working
copy and an ordered revision log. One failed `AtomicAction` can be freshly
grounded and retried twice, for three total attempts, and only while its live
precondition remains true.

Failures use the bounded taxonomy `plan_failed`, `grasp_missed`,
`object_fallen`, `object_dropped`, and `postcondition_failed`. Known recoverable
states can insert a complete `role=recovery` TaskGroup, such as an E2 upright
group. After recovery, the selected route replans only the unfinished suffix.
Offline and online dynamic replanners are explicit, separate modes. Revision,
recovery-action, transition, and retry budgets bound every loop.

## Selection And Fusion

Product mode statically scores offline and online candidates using schema
validity, capability availability, UID validity, task coverage, visual
confidence, and estimated action cost. Exact mature-template matches favor the
offline route; L4 visual tasks favor sufficiently confident online results.

Fusion is conservative. It may choose only complete `TaskGroup` units, rewires
dependencies at group boundaries, and rejects unordered state changes to the
same object. It never splits one E instance across candidates.

## Artifacts

A normal generated bundle contains:

- `task_spec.json`
- `scene_requirements.json`
- `seed_task_graph.json`
- `seed_task_graph.png`
- `agent_config.json`
- `fast_gym_config.json`

Strict A/B adds branch-local graph/result artifacts and `comparison.json`.
Review graphs, runtime records, and videos never become execution inputs.

## Invariants

- SeedGraph nodes are direct AtomicActions, not E-level operators.
- E labels are subgraph grouping semantics only.
- Planning artifacts contain no grounded motion coordinates.
- Online planning never receives the private oracle.
- Runtime uses one capability registry for preflight, Grounding, config
  construction, execution, verification policy, and recovery policy.
- Required arms are never silently replaced.
- Failed or inactive vectorized rows preserve their last valid state.
- Current five task families preserve their v1 AtomicAction topology and live
  Grounding behavior after regeneration.
