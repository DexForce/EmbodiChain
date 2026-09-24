# Repeated Pick/Place Configured Generation T1-T4 Design

## Objective

Complete issue #670's T1-T4 slice on top of PR #681 and demonstrate it with
the configuration-defined `repeated_pick_place` Task Program.  The demo must be
driven by a separate Generation Profile, generate one nominal and two
deterministic same-grid spatial candidates through the source-neutral
`CandidateCoordinator`, rebuild complete Atomic `ActionPlan` values, and execute
the selected candidate through the existing `TaskProgramDemoBridge` and ordinary
Gym `env.step()` boundary.

The deliverable is a foundation/MVP demonstration, not fixed-scene collection.
It does not claim measured task success, coverage confirmation, durable dataset
persistence, multi-slot throughput, or observation fan-out.

## Relationship to the Foundation Hardening Work

This design depends on
`2026-09-24-pr681-foundation-hardening-design.md`.  Its Atomic-plan integrity,
`PlanResult` normalization, session-owned generation streams, atomic proposal
admission, FIFO-only scheduling, and host-runner ownership fixes are
prerequisites rather than duplicated here.

Together, the two changes complete the intended T1-T4 boundary:

- the hardening change makes the existing foundation safe and internally
  consistent;
- this change adds the missing Generation Profile loading/resolution path,
  completes the four Source Adapters, connects Task Program planning to the
  coordinator, and supplies the configured demonstration.

The existing unmerged PR APIs may be changed directly.  Incorrect interim
interfaces do not become compatibility surfaces.

## Selected Approach

Use an independent profile file and an episode-level integration argument.

The rejected alternatives are:

1. **Put generation fields in `program.yaml`.** This violates the provider-
   independent Task Program language and issue #670's explicit exclusion.
2. **Put generation fields in `env.yaml` or the embodiment component.** This
   makes a physical environment or robot own candidate-generation policy.
3. **Add general `run-env` collector routing now.** This expands T4 into the T5+
   collection/host layer and was deliberately excluded from PR #681.

The chosen demo launcher accepts two independent inputs:

```text
task.franka.yaml       physical environment + embodiment + Task Program
generation.demo.yaml  source + expansion + logical scheduling profile
```

It constructs the ordinary configured environment, then passes the decoded
Generation Profile and selected candidate ordinal to `execute_demo_episode()`.
`EmbodiedEnv.create_demo_segments()` treats those as trusted episode-level host
arguments and forwards them to Task Program runtime assembly.  A normal caller
that omits them follows the unchanged path.

## Ownership Boundaries

### Environment and deployment

The existing task files retain their current owners:

- `env.yaml`: backend, `num_envs`, simulation cadence, physical cube, and Gym
  lifecycle;
- `task.franka.yaml`: environment, embodiment, Task Program, and execution-
  policy component selection;
- `program.yaml`: repeat/segment/semantic-call flow and cyclic target poses;
- `integration.yaml`: semantic scene binding and Pick/Place defaults;
- embodiment and execution-policy components: robot resources, motion,
  tracking, recovery, runner, and assurance.

No generation field is added to any of those files.

### Generation Profile

`generation.demo.yaml` is a separate, callable-free, strictly decoded sidecar.
It owns:

- source kind and stable source identity;
- action-unit selection and explicit editable phase permissions;
- Affordance and trajectory proposal budgets;
- enabled same-grid spatial operators and deterministic seed;
- logical candidate count and FIFO scheduling;
- bounded in-flight settings; and
- post-rollout observation profile names as candidate metadata only.

It must not contain backend, `num_envs`, `control_dt`, robot, sensors, simulator,
or recorder construction.  Those values are supplied from the already-resolved
environment/runtime and cannot be overridden by the profile.

### Runtime integration

Task Program continues to lower calls to Atomic Skills.  The generation
integration observes one already-grounded Atomic plan, converts it through the
Task Program Source Adapter, asks `CandidateCoordinator` for logical candidates,
selects one requested ordinal, and returns a completely rebuilt `ActionPlan`.

It does not create a second Task Program executor, command sink, environment
step loop, reset state machine, or persistence state machine.

## Generation Profile Contract

The typed representation remains `TrajectoryGenerationJobCfg` for this PR, but
its public documentation identifies it as the source-neutral Generation Profile.
The new loader is:

```python
load_generation_profile(path: str | Path) -> TrajectoryGenerationJobCfg
```

It uses the repository's safe `load_config()` helper and delegates closed-field
validation to `TrajectoryGenerationJobCfg.from_mapping()`.

T1 completes the profile fields needed by issue #670:

- `source.kind`: `handwritten`, `motion_generator`, `atomic_action`, or
  `task_program`; the transitional `atomic` spelling is removed;
- `source.source_id` and `source.source_revision`;
- `source.unit_scope`, initially fixed to `action` for Atomic and Task Program
  sources;
- `source.template_id`, used as the selected Atomic `skill_id` in this demo;
- explicit `source.phase_permissions` and `source.phase_kinds` keyed by Atomic
  segment name;
- `augmentation.max_variants_per_reference`;
- Affordance and scheduling budgets;
- `execution.max_inflight`, initially required to be `1` for the T4 runner;
- `observation.enabled` and unique `observation.profiles`, retained as
  `CandidateSpec` metadata without implementing fan-out.

Unsupported T5+ behavior fails early.  In particular:

- `coverage_per_cost` is rejected until T9;
- timing variation is rejected by the configured T4 Task Program demo;
- Affordance expansion is disabled in the demo profile;
- observation profiles do not create physical candidate identities; and
- environment-owned fields are unknown profile keys and therefore fail strict
  decoding.

The task-local profile is:

```yaml
source:
  kind: task_program
  source_id: repeated_cube_pick_place
  source_revision: config:repeated_pick_place_v1
  unit_scope: action
  template_id: place
  phase_permissions:
    retract: [joint_residual, via_points]
  phase_kinds:
    approach: free
    release: contact
    retract: free

augmentation:
  seed: 7
  max_variants_per_reference: 3
  factors:
    spatial:
      enabled: true
      method: [joint_residual, via_points]
      joint_offset_scale: 0.05
    timing:
      enabled: false

affordance:
  enabled: false

scheduling:
  policy: fifo
  candidate_budget: 3
  reference_family_budget: 1
  exploration_fraction: 0.0

execution:
  max_inflight: 1

observation:
  enabled: false
  profiles: []
```

Fields omitted from this example keep the existing semantically valid T4
defaults.

## T2: Complete Atomic Plan/Template Adaptation

The hardening change enforces transform isolation, source-plan binding, and an
identical timing grid.  This change makes the Atomic adapter implement the
common source protocol structurally:

```python
class ActionPlanTemplateAdapter:
    kind = "atomic_action"

    def export_template(
        self,
        source: ActionPlan,
        *,
        context: SourceContext,
    ) -> TrajectoryTemplate: ...
```

`SourceContext` owns source ID, source revision, and unit/template identity.
The adapter owns only trusted joint order, phase kinds, phase permissions,
controlled-joint indices, and validator ID.  Its existing direct `export()`
helper may delegate to `export_template()` only when passed an explicit context;
it must not retain a competing source-identity protocol.

For `repeated_pick_place`, only the Place `retract` segment is editable.
`approach` remains a fixed free phase and `release` is a fixed contact phase.
The adapter rejects missing segment declarations, unknown declarations, timing
changes, B != 1, unsuccessful plans, or plans without retained full-joint qpos.

Rebuilding continues through `AtomicActionEngine.rebuild_plan_from_trajectory()`
so commands, qvel references, tracking, named segments, scene dependencies,
effects, guards, and recovery metadata are regenerated together.

## T3: Four Source Adapters

All four kinds implement the same structural `SourceAdapter` contract:

1. `TemplateSourceAdapter` for handwritten, already annotated qpos templates;
2. `PlanResultSourceAdapter` for successful single-row MotionGenerator results;
3. `ActionPlanTemplateAdapter` for one grounded Atomic `ActionPlan`; and
4. `TaskProgramSourceAdapter` for Task Program calls through the Atomic adapter.

`TaskProgramSourceAdapter` lives under Task Program integrations, not motion
core.  It is deliberately thin: its source value is the grounded Atomic
`ActionPlan`, and it delegates template conversion to
`ActionPlanTemplateAdapter`.  Its `kind` is `task_program`; it owns no Task
Program syntax, compiler, physical slots, reset, coverage, or persistence.

This placement prevents `motion.expansion` from importing Task Program or
Atomic runtime packages and avoids circular dependencies.

## Task Program Transform Injection

The call-scoped plan transform is threaded through existing owners:

```text
EmbodiedEnv.create_demo_segments(generation_profile, candidate_index)
  -> TaskProgramEnvironmentAdapter.create_bridge(...)
  -> SemanticCallExecutor(plan_transform_factory=...)
  -> AtomicActionEngine.start(plan_transform=...)
  -> ExecutionSession
  -> AtomicActionEngine._plan_request(..., plan_transform=...)
```

The factory sees trusted compiled-program and grounded-call metadata and returns
either `None` or one Atomic `PlanTransform`.  It returns `None` for every skill
except `profile.source.template_id` (`place` in the demo).  No transform is
installed globally on a shared engine.

The transform is scoped to one Atomic execution session.  Every planning call
inside that session uses the same transform; the demo execution policy has zero
replans and retries, so only the initial plan is transformed.  General recovery
semantics remain explicit and tested rather than inferred.

## T4: Minimal Coordinator Integration

For each selected Place plan, the transform:

1. derives a stable `SceneCase` from the compiled program/integration/profile
   identity and the current B=1 planning context;
2. reads full joint names and limits from the exact Atomic engine robot;
3. creates/registers a value-only `GenerationSession`;
4. constructs `SourceContext` for the grounded action unit;
5. constructs `CandidateCoordinator` with `TaskProgramSourceAdapter`;
6. enqueues `augmentation.max_variants_per_reference` candidates;
7. selects the caller-requested FIFO candidate ordinal;
8. validates the ordinal and selected candidate identity; and
9. rebuilds and returns the selected complete `ActionPlan`.

The hardening work supplies session-owned proposal RNG, exact duplicate
filtering, unique physical geometry lineage, atomic proposal admission, and
FIFO-only decoding.

`CandidateSpec.observation_profiles` is populated from the profile.  The
compatibility key includes scene case, initial state, joint order, phase layout,
control period, selected backend identity, validator, and required observation
profile tuple.  It remains metadata in the B=1 demo; multi-slot bucketing is T7.

The generic synchronous B=1 host runner and memory sink remain covered by motion
execution tests from the hardening change.  The configured Task Program demo
does not route its physical rollout through that runner because doing so would
duplicate `TaskProgramDemoBridge` execution.  T5 will adapt fixed-scene host,
measured validation, and durable EpisodeSink ownership to the shared lifecycle.

## Demo Workflow

Add:

```text
embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/
  generation.demo.yaml

examples/sim/motion/
  repeated_pick_place_generation_showcase.py
```

The launcher accepts:

```text
--task-config       defaults to task.franka.yaml
--generation-profile defaults to generation.demo.yaml
--output-dir        required
--candidate-indices defaults to 0 1 2
```

For each candidate index it:

1. creates or resets the normal configured environment;
2. executes one complete Task Program episode with the decoded profile and
   candidate ordinal;
3. discards recorder state because T5 persistence is out of scope;
4. records generation metadata and Task Program result metadata; and
5. writes comparison plots and, when rendering is available, replay video to
   the requested output directory.

All three candidates run through the same `program.yaml`: three Pick/Place
repetitions per episode, with every Place call using the selected candidate
ordinal for its current reference plan.  Candidate 0 is nominal; candidates 1
and 2 are deterministic spatial variants.  Contact/release samples and all
phase endpoints remain exact.

The example may report physical completion observed from projected assurance,
but it must not call that measured task success or confirmed coverage.

## Error Handling

Validation fails before simulation side effects whenever possible:

- profile decoding rejects unknown fields and unsupported modes;
- profile source ID/program ID or selected skill mismatches fail before bridge
  execution;
- candidate ordinal outside the generated FIFO set fails before command
  dispatch;
- missing or inconsistent phase declarations fail during template export;
- changed timing or malformed rebuilt plans fail before session installation;
- candidate generation failure leaves no partial session or coordinator state;
- early demo termination follows the existing bridge cancellation/safe-hold
  path; and
- the demo always discards uncommitted recordings on failure or close.

Errors include profile path, source unit, candidate ordinal, and candidate ID
when those values have been allocated.

## Validation Strategy

Tests are written before production changes and remain simulator-free unless a
specific integration requires the live runtime.

### T1 configuration

- strict load and round-trip of `generation.demo.yaml`;
- rejection of environment-owned fields, unknown keys, unsupported scheduling,
  enabled timing, and enabled Affordance for the demo route;
- unique observation profiles and bounded candidate/max-inflight settings;
- package-data inclusion of the new profile.

### T2/T3 adapters

- all four adapters satisfy `SourceAdapter` structurally;
- Atomic and Task Program adapters produce identical templates for the same
  plan/context;
- explicit phase kinds/permissions are preserved;
- ordinary Atomic/Task Program calls without a profile are unchanged;
- complete rebuilt command payload, qvel, tracking, segments, dependencies,
  effects, and guards remain internally consistent.

### T4 coordinator/runtime

- profile-selected candidate count and ordinal;
- nominal plus two deterministic, distinct spatial candidates;
- exact contact/hold/endpoints and identical `dt`;
- session-owned identities and no partial proposal state;
- observation profile propagation and strict compatibility key;
- transform threading through `SemanticCallExecutor` and `ExecutionSession`;
- configured `repeated_pick_place` static deployment inspection and a
  simulator-free vertical slice using the existing fake environment/runtime.

### Demo/static validation

- example import and argument forwarding without simulator construction;
- output paths remain caller-owned;
- no tutorial script is imported as a production test target;
- existing Task Program deployment inspector passes for Franka and UR5; and
- live simulator execution is run when available or explicitly reported as not
  run.

Focused validation includes:

```bash
/root/miniconda3/envs/py311/bin/python -m pytest -q \
  tests/sim/motion/expansion \
  tests/sim/atomic_actions \
  tests/lab/task_program \
  tests/gym/envs/task_program/test_configured_integration.py \
  tests/gym/envs/task_program/test_task_vertical_slices.py \
  tests/test_task_program_package_data.py

/root/miniconda3/envs/py311/bin/python \
  .agents/skills/add-task-program/scripts/inspect_deployment.py \
  embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.franka.yaml \
  embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml
```

The final proportional pre-commit selection is governed by the project
`pre-commit-check` skill.

## Explicit Exclusions

- Task Program DSL or `program.yaml` changes;
- environment, embodiment, physics, sensor, or recorder ownership changes;
- geometry-changing Affordance proposals and replanning;
- timing variants or phase-index remapping;
- parallel semantic calls or heterogeneous phase schedules;
- fixed-scene snapshot/restore verification from PR #591;
- measured effect/guard integration from PR #594;
- durable LeRobot EpisodeSink integration;
- m-family/n-slot scheduling, compatibility batching, caches, CPU/GPU overlap,
  and writer backpressure;
- coverage-per-cost scheduling; and
- observation/perception fan-out or confirmed observation coverage.

## Completion Criteria

T1-T4 are complete for this slice when:

1. a separate strict Generation Profile loads from the task directory without
   duplicating environment-owned values;
2. handwritten, MotionGenerator, Atomic Action, and Task Program adapters use
   one source protocol;
3. the configured Task Program installs only call-scoped transforms for Place;
4. `CandidateCoordinator` produces nominal plus two deterministic same-grid
   candidates with session-owned identity and no partial state;
5. the selected candidate is rebuilt into a complete valid `ActionPlan` and
   executes only through `TaskProgramDemoBridge`;
6. the ordinary deployment remains behaviorally unchanged when no profile is
   supplied;
7. the task-local demo exposes candidate provenance and comparison artifacts;
8. static Task Program inspection and focused CPU tests pass; and
9. live physical qualification is reported accurately rather than inferred
   from projected assurance.
