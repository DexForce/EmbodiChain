(semantic-skills)=

# Semantic skills

```{currentmodule} embodichain.lab.sim.skills
```

Semantic skills are the application-facing layer above
{doc}`atomic actions <atomic_actions/index>`. A semantic call names an object,
relation, and optional robot participant without exposing joint groups, planner
instances, raw controller commands, or an `ActionBinding`. The semantic layer
validates that declaration against one scene and robot embodiment, then lowers
it to the same typed atomic-action runtime used by direct Python callers.

This boundary is useful for MLLM agents, task planners, configuration-driven
applications, and users who want robot-independent task code. It is not an
agent loop: the application still owns task selection, perception policy,
physical-effect verification, and any task-level fallback strategy.

```text
SemanticCallSpec values
        |
        v
SemanticIntegrationManifest
  +-- SceneManifest          canonical IDs and affordance metadata
  +-- RobotSkillProfile      resources, commands, presets, providers
  `-- SemanticCallCatalog    discoverable call schemas
        |
        v bind live registry + action engine
BoundSemanticIntegration
        |
        v
SemanticSkillCompiler
  analyze() -> SemanticWorkflow       provider-free validation and look-ahead
  ground()  -> GroundedSemanticCall   latest observation -> ActionInvocation
        |
        v
SemanticSkillRuntime / SemanticTask
        |
        v
ExecutionRunner -> controller transports -> verified effects
```

## Semantic skills or direct atomic actions?

Both paths use `AtomicActionEngine` and therefore share planning, controller
authorization, recovery, and effect semantics.

| Choose | When it is the better boundary |
|---|---|
| Semantic skills | Task code should be robot-independent; an agent or planner emits object-centric calls; scene and resource validation should happen before controller work; dynamic task segments must retain verified state. |
| Direct atomic actions | A scripted application already knows exact goals, bindings, policies, and options; low-level tuning or a custom controller contract is part of the application. |

For a user writing a small fixed robot script, direct atomic actions usually
have fewer integration objects. For an MLLM agent or an application targeting
multiple robot profiles, semantic skills provide the safer and more stable
interface.

## Public semantic calls

The built-in catalog returned by {func}`builtin_semantic_call_catalog` exposes
three curated call values:

| Call | Intent | Main lowering behavior |
|---|---|---|
| {class}`Pick` | Acquire a registered object, optionally through an explicit grasp affordance. | Selects a capability-compatible grasp affordance and lowers to atomic `pick_up`. |
| {class}`Place` | Release a held object at an absolute pose, on a support, or inside a container. | Requires exactly one of `at`, `on`, or `inside`; relation targets use an explicitly installed typed grounder. |
| {class}`HandOver` | Transfer a held object to another robot resource. | Uses a robot-profile-selected provider for the middle and default final pose; an explicit `final_target` overrides the latter. |

{class}`SemanticPose` expresses an absolute object-space pose with a position
and normalized WXYZ quaternion. Scene objects and affordances use typed
{class}`SceneObjectRef` and {class}`SceneAffordanceRef` values, so aliases are
resolved at the registry boundary instead of being propagated into execution.

Extensions use {class}`RegisteredSemanticCall`. Its argument tree accepts only
declarative values; tensors, callables, classes, modules, and live simulator
objects are rejected. A registered descriptor must identify one exact
agent-visible atomic target, and the compiler must install a matching
{class}`RegisteredSemanticLowerer` with the same call ID and schema version.
Curated calls cannot be remapped through this extension mechanism.

## Static integration

Create the static declaration before execution:

```python
from embodichain.lab.sim.skills import (
    SceneManifest,
    SemanticIntegrationManifest,
    builtin_semantic_call_catalog,
)

manifest = SemanticIntegrationManifest(
    scene=SceneManifest.from_registry(scene_registry),
    robot_profile=robot_profile,
    call_catalog=builtin_semantic_call_catalog(),
)
```

`SceneManifest` is provider-free: creating it does not observe simulation or
perception. It snapshots canonical identity, aliases, topology, affordance
capabilities and revisions, and collision-world mode. `manifest.bind(...)`
requires the live {class}`SceneRegistry` to match that snapshot and binds the
profile to the exact action engine. Replacing an installed agent-visible action,
changing the bound profile, or changing scene metadata invalidates the old
integration rather than silently reusing stale contracts.

Policy preset selection is deterministic:

1. `SemanticIntegrationManifest.runtime_preset`, when configured;
2. `RobotSkillProfile.skill_presets[atomic_skill_id]`;
3. `RobotSkillProfile.default_preset`.

A missing or unknown preset is a validation error. The selected
{class}`SkillPolicyPreset` owns the motion policy, recovery policy, and
`ExecutionRunnerCfg` used by that call.

## Analyze first, ground from fresh state

{meth}`SemanticSkillCompiler.analyze` performs provider-free work:

- catalog and schema discovery;
- canonical scene and affordance resolution;
- robot-resource and preset selection;
- verified-held-object flow analysis;
- first-release look-ahead for Pick grasp selection;
- validation that required lowerers and grounders are installed.

It returns an immutable {class}`SemanticWorkflow`. No scene provider is read and
no planner is run at this stage.

{meth}`SemanticSkillCompiler.ground` lowers exactly one analyzed call from the
latest {class}`~embodichain.lab.sim.atomic_actions.PlanningContext`. It resolves
late-bound relation or handover targets and returns a
{class}`GroundedSemanticCall` containing an `ActionInvocation` and an owned
per-environment `eligible_mask`:

```python
workflow = compiler.analyze(calls, workflow_id="sort_workpiece")
grounded = compiler.ground(
    workflow,
    call_index=0,
    context=latest_context,
    eligible_mask=active_rows,
)
session = engine.start(
    (grounded.invocation,),
    latest_context,
    eligible_mask=grounded.eligible_mask,
)
```

The runtime performs this JIT grounding automatically before every call. Known
calls should be submitted together when possible: a `Pick -> Place` or
`Pick -> HandOver` segment lets analysis pass the first downstream object target
into grasp selection. Splitting those calls into separate dynamic segments is
valid, but removes that look-ahead information from the earlier Pick.

## Construct a runtime

Use {meth}`SemanticSkillRuntime.from_simulation` for the standard simulation
path. It creates a registry-backed planning scene provider, a
`SimulationExecutionAdapter`, an `AtomicActionEngine` with built-ins, and the
semantic manifest/compiler:

```python
runtime = SemanticSkillRuntime.from_simulation(
    simulation=sim,
    robot=robot,
    motion_generator=motion_generator,
    scene_registry=scene_registry,
    robot_profile=robot_profile,
    effect_verifier=verify_effect,
    control_dt=4 * sim.sim_config.physics_dt,
)
```

Use {meth}`SemanticSkillRuntime.bind` when the application owns custom
observation, command, clock, endpoint-adapter, or hardware ports. Only one
{class}`SemanticTask` may own a runtime at a time; this layer does not implement
a resource scheduler or lease manager.

`runtime.runner_cfg`, when supplied, overrides the runner configuration from
every selected skill preset. When omitted, each grounded call uses its own
preset's runner configuration. `control_dt` is the command cadence and is
independent of the simulation physics period.

## Execute a fixed workflow

The minimal robot-independent program names only the registered object and its
desired final pose:

```python
from embodichain.lab.sim.skills import Pick, Place, SceneObjectRef, SemanticPose

workpiece = SceneObjectRef("workpiece")
calls = (
    Pick(object=workpiece),
    Place(
        object=workpiece,
        at=SemanticPose(
            position=(-0.40, 0.48, 0.025),
            quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
    ),
)

result = runtime.run(
    calls,
    task_id="pick_and_place",
    effect_verifier=verify_effect,
)
result.require_all_succeeded()
```

{meth}`SemanticSkillRuntime.run` is blocking and requires a
`SemanticEffectVerifier`. Use {meth}`SemanticSkillRuntime.start` plus
{meth}`SemanticExecution.step` or
{meth}`SemanticExecution.run_until_blocked` when physical verification arrives
asynchronously. A pending effect produces
`SemanticExecutionStatus.WAITING_FOR_EFFECT`; resume it with a boolean
per-environment `effect_success` mask.

## Dynamic tasks

Dynamic task construction is supported at completed semantic-segment
boundaries. A {class}`SemanticTask` carries verified `TaskState`, the latest
observation, and a sticky eligible cohort across those decisions:

```python
with runtime.open_task("clear_table") as task:
    first = task.run_segment(
        (Pick(object=workpiece),),
        segment_id="acquire",
        effect_verifier=verify_effect,
    )

    next_calls = decide_next_calls(first.task_state, task.latest_context)
    task.run_segment(
        next_calls,
        segment_id="agent_decision_1",
        effect_verifier=verify_effect,
    )
    result = task.finish()
```

The following boundaries are intentional:

- only one segment executes at a time;
- successful segments leave the task open until `finish()` or `cancel()`;
- failed or cancelled segments are terminal and release runtime ownership;
- environment rows that become ineligible remain excluded in later calls and
  segments;
- `revise_current()` can update a compatible in-flight call, but it cannot
  replace the semantic skill, logical invocation, or runtime endpoint addresses.

The runtime does not automatically choose a replacement skill, re-run an agent,
or reconcile symbolic state after an uncertain physical effect. Implement those
task-level policies in the application at a safe segment boundary.

## Recovery and physical success

Each grounded call runs through the existing closed-loop `ExecutionRunner`.
Depending on its `RecoveryPolicy`, it can detect and recover from tracking
errors, supported scene-target motion, collision-world revisions, timeouts, and
per-environment planning failure. Recovery is bounded and emits structured
atomic-action events retained in {class}`SemanticCallRecord` and aggregated by
{class}`SemanticSegmentResult` and {class}`SemanticTaskResult`.

Dynamic target recovery follows the atomic primitive's dependency contract.
For example, Pick monitors its object/grasp dependency only through the
`approach` segment; contact-, close-, and lift-induced object movement is not
treated as an external target update. Atomic HandOver can monitor
`SceneEntityPose` values supplied for its middle and final option poses.

Planning success is not physical success. Attachment, release, and ownership
transfer are committed only after the application verifier accepts the pending
effect for each environment. A failed verification follows the configured
atomic recovery budget; if the runner terminates unsuccessfully, the semantic
segment and task fail. There is no implicit success assumption.

Final task status is:

- `SemanticTaskStatus.SUCCEEDED` when all initially eligible rows remain;
- `SemanticTaskStatus.PARTIAL_SUCCESS` when a non-empty subset remains;
- `SemanticTaskStatus.FAILED` when execution fails or no row remains;
- `SemanticTaskStatus.CANCELLED` after explicit cancellation.

Call {meth}`SemanticTaskResult.require_all_succeeded` when partial batch success
is not acceptable.

## Diagnostics and extension points

{meth}`SemanticSkillRuntime.validate` exposes static analysis without observing,
planning, or executing. Static integration and grounding errors use
{class}`SemanticValidationError`, whose {class}`SemanticDiagnostic` contains a
stable code, a complete path, a human-readable message, and sorted candidates.
Agents should consume the structured fields rather than parse the exception
string.

Three explicit extension points keep executable objects outside semantic calls:

- {class}`RegisteredSemanticLowerer` lowers a catalog-registered call;
- {class}`RelationTargetGrounder` converts a capability-, payload-type-, and
  revision-matched relation into an object pose;
- {class}`HandOverPoseProvider` supplies embodiment-appropriate middle and
  default final object targets and is selected through
  `RobotSkillProfile.grounding_providers["hand_over"]`.

See {doc}`/tutorial/semantic_skills` for complete runnable Place and dual-arm
HandOver examples, {doc}`scene_registry` for affordance registration, and
{doc}`atomic_actions/robot_skill_profiles` for embodiment resource binding.
