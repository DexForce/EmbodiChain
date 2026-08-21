# GenSim PR #532-#538 and Semantic Skill Integration Plan

- Status: proposed
- Last updated: 2026-08-21
- Guiding principle: **Semantic Skill is the only unified lower-level execution architecture**
- Related Semantic Skill PR: [#492](https://github.com/DexForce/EmbodiChain/pull/492)
- Related GenSim PRs:
  [#532](https://github.com/DexForce/EmbodiChain/pull/532),
  [#533](https://github.com/DexForce/EmbodiChain/pull/533),
  [#534](https://github.com/DexForce/EmbodiChain/pull/534),
  [#535](https://github.com/DexForce/EmbodiChain/pull/535),
  [#536](https://github.com/DexForce/EmbodiChain/pull/536),
  [#537](https://github.com/DexForce/EmbodiChain/pull/537), and
  [#538](https://github.com/DexForce/EmbodiChain/pull/538)
- Related Semantic Skill stack:
  [#495](https://github.com/DexForce/EmbodiChain/pull/495),
  [#496](https://github.com/DexForce/EmbodiChain/pull/496),
  [#497](https://github.com/DexForce/EmbodiChain/pull/497),
  [#498](https://github.com/DexForce/EmbodiChain/pull/498),
  [#500](https://github.com/DexForce/EmbodiChain/pull/500),
  [#501](https://github.com/DexForce/EmbodiChain/pull/501),
  [#504](https://github.com/DexForce/EmbodiChain/pull/504), and
  [#480-#483](https://github.com/DexForce/EmbodiChain/pull/480)

## 1. Executive summary

PR #532-#538 should not introduce a second Action Engine beside the existing
Semantic Skill and Atomic Action stack. The adjusted design keeps the parts of
GenSim that provide real additional value:

- task interpretation and typed `TaskSpec` values;
- offline and online task-plan candidates;
- an immutable task DAG and revision log;
- candidate selection and TaskGroup-level fusion;
- unfinished-suffix replanning;
- task-level recovery, audit artifacts, and A/B evaluation;
- cross-engine orchestration and final task inspection.

The following duplicate implementations should be removed from GenSim:

- an independent atomic capability registry;
- action-specific motion policies and robot-part routing;
- a second live action grounder;
- an adapter that directly materializes `ActionInvocation` values;
- a second physical execution loop and trajectory scheduler;
- independent held-object, effect, and recovery truth.

The final layering is:

```text
Task Engine
    owns TaskSpec, SemanticTaskGraph, candidates, task scheduling,
    task-level recovery, and final success
                         |
                         v
Semantic Skill
    owns semantic calls, scene/profile binding, JIT grounding,
    physical effects, workflow recovery, and the application runtime
                         |
                         v
Atomic Actions
    owns typed goals, planning, command execution, tracking recovery,
    transport acknowledgement, and safe stop
```

This is not a compatibility exercise for the current GenSim runtime. PR
#533-#538 should be structurally rewritten around the canonical Semantic Skill
interfaces instead of accumulating adapters around the current implementation.

## 2. Scope and assumptions

### 2.1 In scope

- Reassign responsibilities across PR #532-#538.
- Define the target task graph and execution boundary.
- Identify GenSim components to keep, transform, move, or remove.
- Define dynamic-task and error-recovery behavior.
- Define the landing order and review gates.
- Define architecture and end-to-end acceptance tests.

### 2.2 Out of scope

- Replacing `AtomicActionEngine`, `ExecutionSession`, or `ExecutionRunner`.
- Replacing the canonical `SceneRegistry` or `RobotSkillProfile`.
- Implementing a second controller or simulation stepping loop in GenSim.
- Persisting robot trajectories, qpos, grasp poses, or controller routing in a
  task-planning artifact.
- Allowing task-graph rewrites during an unverified physical-effect boundary.
- Claiming arbitrary per-environment task-program divergence in the first
  adjusted version.

### 2.3 Canonical runtime naming

The current #492 head contains `SemanticSkillRuntime`, while downstream #496
defines the intended canonical `SkillRuntime`. Before adapting #536 and #537,
these implementations must be consolidated into one public runtime. This plan
uses `SkillRuntime` as the conceptual name; the exact final class name is less
important than having exactly one implementation and one ownership boundary.

## 3. Target architecture

```text
Scene Engine
  GeneratedSceneGraph / SceneAuthoringGraph
             |
             v
Task Engine SceneAdapter
  SceneManifest + SceneRegistry registration plan + evidence sidecar
             |
             +------------------------------------------+
                                                        |
TaskSpec                                                |
   |                                                    |
   v                                                    v
Offline / Online Semantic Task Planner      Semantic Integration Catalog
   |                                        - SceneManifest
   |                                        - RobotSkillProfile
   |                                        - SemanticCallCatalog
   |                                        - provider declarations
   |                                        - integration fingerprint
   v
SemanticTaskGraph candidates
   |
   v
Candidate selection / TaskGroup fusion / preflight
   |
   v
TaskGraphScheduler
   |
   | selected route and semantic look-ahead suffix
   v
SkillRuntime / ParallelSkillRuntime
   |
   v
SemanticSkillCompiler
   |
   v
ActionInvocation
   |
   v
AtomicActionEngine -> ExecutionRunner -> controller or simulation
```

The main runtime resolution path is:

```text
SemanticTaskGraph node
    -> canonical SemanticCallCfg decoder
    -> SemanticCallSpec
    -> SemanticSkillCompiler.analyze()
    -> fresh observation
    -> SemanticSkillCompiler.ground()
    -> ActionInvocation
    -> AtomicActionEngine / ExecutionRunner
    -> verified SkillResult and TaskState
    -> TaskGraphScheduler transition
```

## 4. Ownership matrix

| Concern | Canonical owner | Allowed GenSim responsibility |
|---|---|---|
| Generated scene authoring | Scene Engine | Generate, edit, import, export, and preserve provenance |
| Provider-free scene identity | `SceneManifest` | Adapt generated scene data once into canonical IDs |
| Live scene identity and metadata | `SceneRegistry` | Hold only references to the canonical integration |
| Dynamic scene state | `SceneSnapshot` and `SceneProvider` | Consume snapshots through Semantic Skill ports |
| Semantic call discovery | `SemanticCallCatalog` | Consume a JSON-safe planner projection |
| Robot capabilities and resources | `RobotSkillProfile` | Express semantic participant constraints only |
| Exact physical resource claims | Bound `RobotSkillProfile` | Use claims returned by canonical preflight/runtime |
| Semantic grounding | `SemanticSkillCompiler` and registered grounders | Bind language roles to canonical scene references |
| Motion and action lowering | `SemanticSkillCompiler` and Atomic Actions | Do not materialize goals, options, or invocations |
| Physical execution | `SkillRuntime` and `ExecutionRunner` | Invoke a narrow semantic execution port |
| Effect verification | Semantic effect monitors and evidence collectors | Consume verified results and evidence summaries |
| Tracking and transport recovery | `ExecutionSession` and `ExecutionRunner` | Observe terminal typed failures only |
| Reacquisition and workflow recovery | `SkillRuntime` | Wait until workflow recovery is exhausted |
| DAG scheduling and route selection | `TaskGraphScheduler` | Own node readiness, route selection, and graph revisions |
| Task-level recovery | `TaskGraphScheduler` | Replace a TaskGroup or unfinished suffix at a safe boundary |
| Task or scene regeneration | Task Engine orchestration | Start a new safe execution transaction |
| Final task success | `SuccessSpec` and final inspection | Inspect final observed state without duplicating call effects |

## 5. SemanticTaskGraph contract

### 5.1 Purpose

`SemanticTaskGraph` replaces the current direct-AtomicAction `SeedGraph`. It is
an immutable, JSON-safe task-planning artifact. It describes semantic intent,
dependencies, candidate routes, and task-level recovery, but contains no
grounded execution data.

Each node contains exactly one canonical semantic-call payload. A scheduler
may give the selected future suffix to `SkillRuntime` for static look-ahead
while executing only the current node.

### 5.2 Conceptual schema

```json
{
  "schema_version": "semantic_task_graph/v1",
  "task_id": "place_cube",
  "instruction": "Place the cube on the tray",
  "planner_route": "selected",
  "integration_fingerprint": "<sha256>",
  "nodes": [
    {
      "id": "pick_cube",
      "call": {
        "kind": "pick",
        "object": "cube"
      },
      "depends_on": [],
      "task_instance_id": "e1_0",
      "task_type": "E1",
      "role": "primary"
    },
    {
      "id": "place_cube",
      "call": {
        "kind": "place",
        "object": "cube",
        "on": "tray"
      },
      "depends_on": ["pick_cube"],
      "task_instance_id": "e1_0",
      "task_type": "E1",
      "role": "primary"
    }
  ],
  "task_groups": [
    {
      "id": "e1_0",
      "task_type": "E1",
      "node_ids": ["pick_cube", "place_cube"],
      "depends_on": [],
      "success": {
        "kind": "object_supported_by",
        "object": "cube",
        "support": "tray"
      }
    }
  ],
  "success": {
    "kind": "all_task_groups"
  }
}
```

The concrete schema should reuse the exact `SemanticCallCfg` and decoder from
the Expert Program stack. GenSim must not define another semantic-call JSON
format.

### 5.3 Allowed node data

- Stable node and TaskGroup IDs.
- A versioned `SemanticCallCfg` payload.
- Canonical scene-reference IDs.
- Task dependencies and TaskGroup membership.
- Task-level route guards and completion importance.
- Bounded task-level failure routes.
- Planner provenance and confidence.
- Optional semantic resource selections when explicitly bound to the exact
  integration fingerprint.

### 5.4 Forbidden node data

- Atomic action class or implementation name.
- `ActionInvocation`, typed atomic goal, or action options.
- Arm, hand, control-part, solver, or controller route.
- `MotionPolicy`, planner backend, or recovery thresholds.
- qpos, EEF pose, grasp pose, waypoint, trajectory, or command frame.
- Runtime `ResourceClaim` snapshots.
- Held-object state or speculative physical effects.
- Action-level preconditions and postconditions already owned by the semantic
  compiler or effect monitor.

Task-level object pose constraints may be stored when they are part of the
task intent. Grounded robot or motion coordinates may not be stored.

### 5.5 Look-ahead without premature execution

For a selected `Pick -> Place` route, the scheduler should pass both calls to
the canonical runtime analysis window but execute only the first call:

```python
runtime.start(
    pick_call,
    place_call,
    execution_prefix_length=1,
)
```

This lets the compiler use the Place target during grasp selection while
retaining a verified physical boundary after Pick. If the route changes after
Pick, the scheduler can replace only the unfinished suffix.

If the final canonical runtime does not expose an analysis-window/execution-
prefix boundary, that capability belongs in Semantic Skill, not in GenSim.

## 6. PR-by-PR adjustment plan

### 6.1 PR #532: Scene Authoring only

Recommended title:

> `feat(scene-engine): add generated-scene authoring and editing`

Keep:

- scene generation and editing pipelines;
- layout, gravity settling, and asset preparation;
- image and geometry service boundaries;
- scene import/export;
- semantic evidence and provenance.

Adjust:

- Rename the public `SceneGraph` concept to `GeneratedSceneGraph` or
  `SceneAuthoringGraph`, or make its authoring-only status explicit.
- Emit stable canonical IDs, parent relationships, affordance evidence,
  geometry metadata, and physics provenance.
- Do not construct or own a live `SceneRegistry`.
- Keep richer generation evidence in an audit-only sidecar rather than a
  second execution manifest.
- Put conversion to `SceneManifest` in #538's cross-engine adapter.

Exit criteria:

- The generated artifact converts deterministically to one canonical
  `SceneManifest`.
- Import/export preserves canonical identity and ancestry.
- No authoring artifact contains live simulator handles or pose readers.
- Scene Engine types cannot become a second runtime scene authority.

### 6.2 PR #533: TaskSpec and SemanticTaskGraph contracts

Recommended title:

> `feat(task-engine): add TaskSpec and semantic task-graph contracts`

Keep:

- `TaskSpec`, E1-E9 ontology, and `SuccessSpec`;
- strict JSON validation and stable hashing;
- DAG validation and TaskGroup metadata;
- planner provenance and bounded failure policy.

Transform:

- Replace direct-AtomicAction `SeedGraph` with `SemanticTaskGraph`.
- Make every graph node contain one canonical `SemanticCallCfg`.
- Replace `capability_catalog_hash` with the canonical semantic integration
  fingerprint.
- Validate TaskGroup semantic coverage using calls and success conditions,
  rather than hard-coded action sequences.

Remove:

- `AtomicCapabilityRegistry`;
- the parallel generic `CapabilityRegistry` when it represents the same
  executable surface;
- `build_atomic_capability_registry()`;
- GenSim-owned runtime policy and motion-policy defaults;
- node fields for actor, control, atomic action, and motion policy.

Unsupported tasks:

- A TaskSpec may still express a task such as Pour.
- If the semantic catalog has no executable call, planning returns a structured
  `unsupported_semantic_capability` result.
- The planner must not emit planning-only fake AtomicAction nodes.

Exit criteria:

- Graph JSON round-trips through the same Expert Program semantic-call decoder.
- Forbidden grounded or atomic fields are rejected recursively.
- Graph hashes are deterministic.
- There is one capability/fingerprint source.

### 6.3 PR #534: Task interpretation and canonical scene binding

Recommended title:

> `feat(task-engine): add task interpretation and canonical scene binding`

Keep:

- typed instruction interpretation;
- multiple TaskDraft candidates;
- explicit role grounding and ambiguity diagnostics;
- `SceneRequirements` and task-level success conditions;
- strict structured-model boundaries where an MLLM is used.

Adjust:

- Bind scene roles only to canonical `SceneObjectRef`,
  `SceneArticulationRef`, `SceneLinkRef`, and `SceneAffordanceRef` identities.
- Resolve references through the canonical `SceneManifest`.
- Consume a planner projection generated from `RobotSkillProfile` and
  `SemanticCallCatalog`.
- Keep language grounding separate from physical goal grounding.

Remove:

- robot/action configuration construction from
  `generation/config_builder.py`;
- hard-coded `left_arm`, `right_arm`, gripper states, and packaged robot-action
  policy profiles;
- keyword-based affordance or object inference;
- knowledge of planner backends, solvers, controllers, or atomic goal types.

Exit criteria:

- Interpretation produces only TaskSpec, role bindings, and semantic call
  candidates.
- Canonical identity is resolved once and ambiguity fails explicitly.
- MLLM output cannot bypass the same local schema and catalog validation used
  by deterministic callers.

### 6.4 PR #535: Semantic graph planning and bundle generation

Recommended title:

> `feat(task-engine): add semantic task-graph planning and bundles`

Keep:

- deterministic offline recipes;
- online semantic task planning;
- candidate scoring and selection;
- conservative TaskGroup-level fusion;
- stable graph loader, hashing, visualization, and bundle artifacts;
- candidate-local preparation failures.

Adjust:

- Make all recipes produce Semantic Calls rather than AtomicActions.
- Compile and validate only `SemanticTaskGraph` topology and task semantics.
- Preflight calls through the canonical Expert Program and Semantic Skill
  catalogs.
- Derive exact resource conflicts from bound `ResourceClaim` values instead of
  persisting claims in the graph.
- Move robot, sensor, and light templates to their owning scene/robot
  integration packages.

Recommended bundle:

```text
task_spec.json
scene_requirements.json
semantic_task_graph.json
semantic_task_graph.png
integration_fingerprint.json
planner_report.json
```

Exit criteria:

- Offline and online candidates use the same graph schema.
- Candidate fusion occurs only at complete TaskGroup boundaries.
- Every executable candidate passes Semantic Skill provider-free preflight.
- No graph compiler imports or constructs atomic-action implementation types.

### 6.5 PR #536: Thin Semantic Skill runtime binding

Recommended title:

> `feat(task-engine): bind semantic task graphs to SkillRuntime`

Remove:

- `ActionGrounder`;
- `AtomicActionAdapter`;
- `atomic_compat.py`;
- `robot_parts.py`;
- `solver_compat.py`;
- GenSim-owned frame-to-action goal lowering;
- GenSim motion-policy materialization;
- duplicate qpos and held-object execution state.

Replace with a narrow semantic execution boundary:

1. Decode one graph call through the canonical semantic-call decoder.
2. Validate it against the exact semantic integration fingerprint.
3. Build the selected route's semantic analysis window.
4. Call `SkillRuntime` with an execution prefix.
5. Return the canonical `SkillResult` without reconstructing physical truth.

Move remaining responsibilities:

| Current responsibility | New owner |
|---|---|
| camera/depth target materialization | registered Semantic Skill target provider or lowerer |
| live object and articulation goal grounding | `SemanticSkillCompiler` |
| robot arm and endpoint selection | `RobotSkillProfile` |
| grasp collision cache | Atomic Action/graspkit planning service |
| action effect predicate | semantic effect monitor/evidence collector |
| task completion predicate | Task Engine final inspection |

Exit criteria:

- One graph node creates one canonical semantic runtime execution boundary.
- Every executed node captures a fresh observation.
- Pick look-ahead can inspect the selected suffix without executing it.
- GenSim does not construct `ActionInvocation` values.
- GenSim does not read qpos or held-object state as an independent authority.

### 6.6 PR #537: TaskGraphScheduler, graph recovery, and reporting

Recommended title:

> `feat(task-engine): add graph scheduling, recovery, and reporting`

Keep:

- immutable original graph;
- a detached runtime graph and ordered revision log;
- bounded route and transition budgets;
- unfinished-suffix replanning;
- execution recording and tensor-free reporting;
- shared-state A/B evaluation contracts.

Replace `ProgramExecutor` with `TaskGraphScheduler`.

`TaskGraphScheduler` owns only:

- DAG readiness and deterministic tie-breaking;
- selected-route and TaskGroup transitions;
- a shared task-node barrier with row-local masks;
- calls to `SkillRuntime` or `ParallelSkillRuntime`;
- consumption of verified `SkillResult` and `TaskState` values;
- task-level route substitution after runtime recovery is exhausted;
- graph revision, transition, and task-recovery records.

Remove:

- merged-trajectory and per-arm command scheduling;
- controller dispatch and simulation stepping;
- action retry and grasp retry;
- physical effect verification;
- raw qpos and held-object execution state;
- failure classification from exception strings;
- command-level timeout and safe-stop logic.

Parallel behavior:

- Delegate physical concurrency to the canonical `ParallelSkillRuntime` and its
  required safety validator.
- If those contracts are unavailable, serialize independent ready nodes
  deterministically.
- Never merge command frames or trajectories in `TaskGraphScheduler`.

Naming:

- Rename `ActionAgent` to `SemanticTaskPlanner` or `TaskPlanAgent`.
- Avoid exposing `gen_sim.action_engine` as a second public execution system.
- Prefer moving planning/runtime-graph code under
  `embodichain.gen_sim.task_engine` while the PRs remain unmerged.

Exit criteria:

- Graph recovery begins only after a canonical runtime terminal result.
- Verified `TaskState` is the only physical-state transition input.
- Scheduler cancellation delegates safe stop to the active runtime.
- Parallel paths use the canonical parallel safety boundary.
- Reports retain original Semantic Skill events and failures without replacing
  them with string-derived classifications.

### 6.7 PR #538: End-to-end orchestration

Recommended title:

> `feat(task-engine): add semantic orchestration and end-to-end integration`

Keep:

- Scene/Task/Planner orchestration;
- run-directory isolation;
- feasibility reports;
- final task inspection;
- CLI and artifact publication;
- task or scene regeneration;
- end-to-end benchmarks.

Adjust:

- Make `SceneAdapter` produce one canonical `SceneManifest`, a
  SceneRegistry-registration plan, and optional audit evidence.
- Make `FeasibilityBroker` consume the Semantic Integration Catalog rather than
  an atomic capability registry.
- Make the coordinator invoke `TaskGraphScheduler` rather than
  `ProgramExecutor` or `AtomicActionAdapter`.
- Keep final inspection at TaskSpec success level; do not reimplement grasp,
  release, handover, or articulation effect checks.
- Treat `StaticSceneManifest`, redacted manifests, and
  `ConservativeSceneGraph` as derived evidence or reports, not execution
  identity sources.

Split out unrelated changes:

- upright-grasp ranking changes in `pick_up.py`;
- yaw-equivalent downstream-reachability changes;
- graspkit implementation changes;
- corresponding Atomic Action tests.

These are reusable lower-layer enhancements and should land as a focused
Atomic Action prerequisite before the adjusted task-planning stack.

Exit criteria:

- One end-to-end path runs from generated scene data through canonical scene
  integration, SemanticTaskGraph, SkillRuntime, and final inspection.
- No orchestration component sends controller commands directly.
- Infeasible tasks publish structured failure artifacts without stale bundles.
- Scene/task regeneration starts only after the active runtime reaches a safe
  terminal boundary.

## 7. Dynamic-task support

Dynamic behavior has several different meanings and must be assigned to the
correct layer.

### 7.1 Supported behavior

- A moving target can invalidate and replan the current action through
  `SceneEntityPose`, scene dependencies, and `ExecutionSession`.
- Collision-world pose revisions can trigger row-local replanning through the
  canonical scene/planner integration.
- Every Semantic Call is observed and JIT-grounded again before execution.
- Pick can use the selected downstream suffix for grasp look-ahead.
- The TaskGraphScheduler can replace an unfinished suffix after a verified
  semantic-call boundary.
- Offline and online candidates can be selected or fused at TaskGroup
  boundaries.
- The immutable source graph can retain an auditable sequence of runtime
  revisions.
- Final task inspection can trigger a bounded repair TaskGroup or new route.

### 7.2 Explicit limitations

- An active call cannot be replaced by an unrelated skill in place.
- A graph cannot be rewritten while a physical effect is pending verification.
- Changing runtime controller destinations requires a new invocation and safe
  ownership transition.
- Dynamic obstacle pose updates are supported only where the registered scene
  provider and planner support them.
- Entity add/remove and geometry changes require a new scene integration and
  runtime session; they are not an in-place pose revision.
- The first adjusted version should retain a shared task-node barrier. It may
  use row-local success, failure, eligibility, and recovery masks, but should
  not claim arbitrary divergent graph program counters per environment.

## 8. Error-recovery ownership

| Failure class | Owner | Behavior |
|---|---|---|
| tracking error | `ExecutionSession` / `ExecutionRunner` | bounded row-local replan |
| moving semantic target | `ExecutionSession` | re-ground the same invocation revision within policy |
| collision-world revision | `ExecutionSession` | invalidate and replan affected rows |
| planner failure | Atomic Action runtime | retry within the action policy and emit typed failure |
| controller rejection or timeout | `ExecutionRunner` | cancel addressed targets, then hold safely |
| semantic effect not achieved | `SkillRuntime` | verify evidence and apply the semantic workflow policy |
| held relation lost | `SkillRuntime` | perform bounded physical reacquisition where configured |
| current TaskGroup route exhausted | `TaskGraphScheduler` | choose a different TaskGroup or unfinished suffix |
| final task predicate failed | Task Engine | add a bounded repair route or regenerate the plan |
| task or scene infeasible | Task Engine orchestration | publish failure or create a new transaction |

Important constraints:

- The graph scheduler must not insert its own Pick merely because it infers
  that an object was dropped. Canonical workflow recovery owns real
  reacquisition.
- Task-level recovery begins only after `SkillRuntime` exhausts its own action
  and workflow budgets.
- Original typed failures and evidence remain in reports. A task-level category
  may summarize them but must not replace them.
- Every layer owns a separate bounded budget so nested recovery cannot form an
  unbounded loop.

Recommended budget hierarchy:

```text
Atomic recovery budget
    inside one ActionInvocation

Semantic workflow recovery budget
    retry or physical reacquisition of one semantic call

Task graph revision budget
    alternate TaskGroup or unfinished suffix

Task orchestration regeneration budget
    new task or scene transaction
```

## 9. Components to keep, transform, move, and remove

### 9.1 Keep

- `TaskSpec`, E1-E9 ontology, and `SuccessSpec`.
- Offline and online candidate generation.
- TaskGroup grouping and conservative fusion.
- Immutable graph hash, loader, visualization, and artifacts.
- Runtime graph revision log.
- Task-level route recovery and suffix replanning.
- Final task inspection.
- Run directories, recording, reporting, and A/B comparison.

### 9.2 Transform

| Current concept | Target concept |
|---|---|
| direct AtomicAction SeedGraph | `SemanticTaskGraph` |
| ActionAgent | `SemanticTaskPlanner` or `TaskPlanAgent` |
| ProgramExecutor | `TaskGraphScheduler` |
| capability registry | planner projection of Semantic Integration Catalog |
| runtime graph node action | canonical `SemanticCallCfg` |
| action postcondition | semantic effect result or task success predicate |
| static scene execution manifest | canonical `SceneManifest` plus evidence sidecar |

### 9.3 Move

- Visual target providers to registered Semantic Skill grounders/providers.
- Robot and endpoint selection to `RobotSkillProfile`.
- Grasp collision preparation to Atomic Action/graspkit planning services.
- Atomic grasp-quality improvements to a focused lower-layer prerequisite PR.
- Task success predicates to Task Engine final inspection.
- CLI-only execution assembly to the final orchestration layer.

### 9.4 Remove

- `AtomicCapabilityRegistry` and repeated registry builders.
- `ActionGrounder`.
- `AtomicActionAdapter`.
- `ProgramExecutor` as a physical executor.
- GenSim qpos and held-object execution truth.
- GenSim motion policy and robot-part compatibility layers.
- Independent command scheduling, effect verification, and safe-stop logic.
- Hard-coded arm names and robot action templates.
- Planning-only fake actions in executable graph artifacts.

## 10. Dependency and landing plan

### Gate 0: consolidate the Semantic Skill stack

Before adapting the GenSim runtime layers:

1. Decide whether runtime remains in #492 or is owned by #496.
2. Produce exactly one canonical runtime API.
3. Restack #495-#504 and #480-#483 on the current #492 head.
4. Preserve or land the following shared contracts:
   - canonical Semantic Call config/decoder;
   - Semantic Integration Catalog and fingerprint;
   - physical effect/evidence runtime;
   - workflow recovery and reacquisition;
   - parallel runtime and safety validator if physical parallelism is required.

### Recommended merge order

```text
Semantic Skill runtime consolidation
             |
             +------ focused Atomic Action grasp enhancement
             |
             +------ adjusted PR #532 Scene Authoring
             |
             v
PR #533 SemanticTaskGraph contracts
             |
             v
PR #534 task interpretation and scene binding
             |
             v
PR #535 semantic graph planning and bundles
             |
             v
PR #536 thin SkillRuntime binding
             |
             v
PR #537 graph scheduling and task-level recovery
             |
             v
PR #538 end-to-end orchestration
```

Operationally, #533-#538 should be marked Draft while their contracts and
branches are rewritten. Their PR bodies should be updated with the new
dependency chain, ownership rules, and explicit removal of the duplicate
execution architecture.

## 11. Validation plan

### 11.1 Architecture guards

- `embodichain/gen_sim/task_engine` must not directly import
  `embodichain.lab.sim.atomic_actions`.
- GenSim must not define `AtomicCapabilityRegistry`, `ActionGrounder`,
  `AtomicActionAdapter`, or a physical `ProgramExecutor`.
- An architecture test must reject direct controller, simulator-step, or
  command-frame ownership in Task Engine.
- A graph-schema test must recursively reject atomic action, motion policy,
  qpos, grasp pose, and trajectory fields.
- Capability and integration fingerprints must come from one canonical
  Semantic Integration Catalog.

### 11.2 Contract tests

- Semantic graph calls round-trip through the Expert Program decoder.
- Canonical scene aliases normalize exactly once.
- Unknown or ambiguous references fail with pathful diagnostics.
- Unsupported semantic capabilities reject a candidate before bundle
  publication.
- Integration fingerprint drift fails before execution.
- Offline and online candidates use the same graph and call schema.

### 11.3 Runtime-boundary tests

- One graph node creates one semantic runtime execution boundary.
- A fresh observation is captured before every call.
- A selected future suffix participates in compiler look-ahead without
  premature execution.
- Verified `TaskState` is the only state adopted by the graph scheduler.
- A controller rejection invokes canonical cancel-then-hold before graph
  recovery.
- Workflow reacquisition completes or exhausts before TaskGraphScheduler route
  substitution.
- Parallel nodes either use canonical `ParallelSkillRuntime` with a safety
  validator or execute serially.

### 11.4 End-to-end tests

At minimum, cover:

1. deterministic Pick -> Place through a generated scene;
2. dual-resource HandOver through one SemanticTaskGraph;
3. moving target recovery during one semantic call;
4. real held-object loss followed by canonical SkillRuntime reacquisition;
5. exhausted semantic recovery followed by TaskGroup route replacement;
6. unsupported task capability producing a preparation failure artifact;
7. offline/online A/B candidates starting from identical state and using the
   same Semantic Integration Catalog;
8. final task inspection failing and producing a bounded repair or terminal
   report.

## 12. Stack-wide acceptance criteria

The adjusted stack is ready only when all of the following are true:

- Semantic Skill is the only path from semantic intent to `ActionInvocation`.
- Atomic Action and `ExecutionRunner` are the only owners of command execution
  and safe stop.
- GenSim artifacts contain semantic calls and task dependencies, not physical
  action configuration.
- Scene Engine authoring data converts once into the canonical scene
  integration.
- Robot-specific behavior comes from `RobotSkillProfile`, not task recipes.
- Each graph node is JIT-grounded from a fresh observation.
- Effects are committed only from verified Semantic Skill results.
- Graph recovery occurs only at a safe semantic boundary.
- Task-level dynamic replanning preserves completed physical effects and
  revises only unfinished work.
- Reports preserve call-, effect-, recovery-, graph-, and final-task evidence
  without duplicating physical truth.
- The end-to-end tutorials and benchmarks use the same runtime path as Python,
  Expert Program, and model-generated callers.

## 13. Expected result

After this adjustment, the two systems become complementary:

- Semantic Skill supplies one reliable, robot-generic, effect-aware, and
  recoverable physical execution architecture.
- GenSim supplies task interpretation, candidate planning, immutable task
  graphs, dynamic suffix replanning, task-level recovery, audit artifacts, and
  A/B evaluation.

The final system has one physical truth, one capability source, one grounding
path, and one execution path, while retaining GenSim's task-level planning and
dynamic orchestration strengths.
