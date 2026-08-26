# Expert Programs

## Entry points and ownership

The human-facing reference is
`docs/source/overview/sim/atomic_actions/expert_programs.md`.

The canonical public package is
`embodichain.lab.gym.envs.expert_program`. Its main composition roots are:

- `load_expert_program()` and `decode_expert_program()` for strict serialized
  input;
- `ExpertProgramCompiler` for provider-free compilation;
- `SimulationExpertProgramRegistration` for immutable task integration;
- `SimulationExpertProgramAdapterFactory` for binding that registration to an
  initialized simulation environment;
- `configured_runtime.py` for decoding supported callable-free Gym runtime
  declarations and registering the existing `EmbodiedEnv` under a config-owned
  ID;
- `create_simulation_expert_program_adapter()` for live simulation assembly;
- `EmbodiedEnv.compile_expert_program()` and
  `EmbodiedEnv.create_expert_program_bridge()` for environment delegation;
- `EmbodiedEnv.create_demo_segments(expert_program=...)` for static-config or
  episode-injected program execution;
- `AtomicDemoBridge` for lazy `DemoSegment` production.

An Expert Program is a frontend to the semantic-skill and atomic-action stack.
It does not own a second planner, scheduler, effect system, or simulation loop.
The dependency path is:

```text
JSON/YAML or programmatic ExpertProgramCfg
  -> strict schema validation
  -> ExpertProgramCompiler + provider-free SceneManifest
  -> immutable CompiledProgram
  -> ExpertProgramEnvironmentAdapter provider-aware preflight
  -> AtomicDemoBridge
  -> SkillRuntime -> ExecutionRunner -> AtomicActionEngine
  -> DemoSegment actions consumed through env.step()
```

## Configuration and compilation contract

The external Expert Program formats intentionally have no `schema_version`.
This includes the top-level program, opaque registered-call payloads, and the
optional Gym `expert_program_runtime` declaration. Development-history version
fields are rejected as unknown input rather than accepted or defaulted. The
catalog fingerprint's private encoding marker is an internal implementation
detail, not a user configuration field.

The top-level config owns `program_id`, exact scene/profile/preset integration
IDs, named targets, and one program node. Supported nodes are `SequenceCfg`,
`RepeatCfg`, `SegmentCfg`, `InvokeCfg`, and `ParallelCfg`. A `BarrierCfg` belongs
to its `ParallelCfg`; it is not a standalone program node. Nested parallel
blocks are rejected. Built-in call configs are `PickCfg`, `PlaceCfg`, and
`HandOverCfg`; `RegisteredSemanticCallCfg` is the explicit catalog extension.

Both programmatic config construction and untrusted decoding enforce exact
types, discriminators, references, finite numeric values, and bounded depth,
node count, repeat count, and expanded call count. The loader additionally
enforces bounded bytes, valid UTF-8/JSON/YAML, and duplicate-key rejection.
Registered payloads must be acyclic and executable-free; imports, expressions,
callables, modules, and dotted environment traversal are forbidden.

The Expert Program YAML intentionally contains no live integration or Python
provider. The surrounding Gym config may add a strict
`expert_program_runtime` declaration composed from a required `scene`, a
required `robot_profile`, and optional `runtime_services`. There is no
task-level `kind` dispatcher. The scene and profile decode typed simulation
bindings, resources, action options, policies, and monitor selections;
allowlisted service leaves create antipodal grasp generators, configured
hand-over poses, articulation-link Slide lowerers, or joint-position constraint
evidence. Dotted imports, arbitrary callable names, and task module lookup are
not accepted.

Configured antipodal generators accept either a built-in parallel-jaw model ID
or an inline geometry mapping. The immutable model catalog is owned by
`embodichain.toolkits.graspkit.pose_generator`; it contains grasp-planning
geometry, returns a fresh config per lookup, and is not an asset-download or
URDF registry. Omitted algorithm, collision, and annotation fields defer to the
toolkit defaults. Configured runtimes always use whole-mesh annotation, so they
do not expose an interactive Viser port. For concise scene declarations, a
root entity's `simulation_uid` defaults to its `entity_id`; an antipodal
affordance's `native_name` defaults to its `entity_id` and its revision defaults
to `"1"`.

Configured articulation-link Slide lowerers can resolve their target as either
a live `SceneEntityPose` or an invocation-time pose snapshot. The snapshot mode
reads the current link pose once for each episode and installs no dynamic-goal
scene dependency, which keeps trajectory-only demonstrations deterministic
across resets. Live mode remains available for integrations that intentionally
combine target-motion monitoring with a replan policy.

`ExpertProgramCompiler` observes no live state. It resolves canonical scene
references, expands bounded repeats and cyclic targets, assigns stable segment
and call indices, preserves parallel branches, and returns one already
materialized `CompiledProgram`. Consecutive sequential segments share static
downstream-goal look-ahead; a parallel barrier splits that analysis.

## Simulation registration contract

`SimulationExpertProgramRegistration` is the sole extension owner on the
standard simulation path. A task registration contains:

- provider-free scene and robot-profile bindings;
- the semantic call catalog and settling presets;
- relation grounders and hand-over pose providers;
- endpoint adapters and deterministically ordered runtime transports;
- optional parallel-safety and control-part-evidence factories;
- registered semantic-lowerer factories.

All declarations enter one immutable integration fingerprint. Adapter
construction and live runtime assembly call `assert_unchanged()` and revalidate
the declared scene, profile, engine skill catalog, endpoint resolution, and
extension routes. `SimulationExpertProgramFactory` and
`create_simulation_expert_program_adapter()` require that registration and do
not expose separate scene/profile or per-extension override parameters.

Hand-over pose providers declare only the final object-space delivery target.
The unified atomic action computes its middle transfer pose from the two bound
arm roots; no provider-side middle-pose field or deferred look-ahead target is
part of the current contract.

Every registered call descriptor requires exact one-to-one coverage by a
`RegisteredSemanticLowererFactory`. Its frozen provider-free declaration fixes
the call ID and factory revision; the call catalog fixes the payload schema and
target descriptor. Each runtime assembly must receive a fresh lowerer whose
live declaration matches the call ID and target descriptor.
Stateful factories must expose recursively immutable configuration; do not put
live simulator objects or mutable task state in their declaration.

`config_to_cfg()` can obtain the registration directly from a decoded
`expert_program_runtime`, use its catalog for program loading and preflight,
and register `EmbodiedEnv` only after the rest of the Gym config parses. The
top-level `id` is therefore runtime-selectable. Re-loading the same ID and
identical declaration is idempotent; an existing unrelated ID, a changed
runtime declaration, or changed episode limit is rejected instead of silently
overridden. `gym.make(id)` is available only after this config-loading step.

`SimulationSceneBinding` explicitly declares registered rigid objects,
articulations, links, affordances, collision roles, and live pose sources. It
does not scan environment attributes. `SimulationRobotSkillProfileBinding`
accepts control-part convenience bindings or generic core `RobotResource`
values. Custom endpoint families also need matching endpoint adapters, runtime
payloads, and transports. On the standard path, non-control-part endpoints are
timed/open-loop until registration-owned closed-loop feedback, projection,
metric, and effect-evidence factory contracts exist for that endpoint family.

## Runtime and acceptance

`EmbodiedEnv` owns adapter binding and Expert Program action generation. An
`EnvSpec` may carry one `ExpertProgramAdapterFactory`; after `BaseEnv` has built
the live simulation and robot, `EmbodiedEnv` asks that factory for the exact
adapter. A standard simulation factory also exposes its immutable registration,
so `EnvSpec` derives the pre-simulation validation catalog from the same object
instead of requiring a second declaration. Advanced environments may still
override the adapter property.

Compilation validates integration selection against the provider-free
manifest. Bridge creation constructs fresh live scene/evidence/lowerer/safety
providers, performs semantic and parallel preflight, and then creates one
`AtomicDemoBridge`. `create_demo_segments()` normally uses
`cfg.expert_program`; its keyword override accepts an `ExpertProgramCfg` or an
already canonical `CompiledProgram` for one episode. The latter is the runtime
handoff for a trusted MLLM frontend and does not mutate environment config.

The bridge never calls `env.step()` itself. It yields lazy `DemoSegment`
actions; the shared demo executor consumes them through normal environment
steps. `BaseEnv.step_dt` is the control grid, and off-grid runtime durations
fail instead of being silently resampled.

Keep these three boundaries separate:

- semantic effect monitors verify one physical call and participate in
  recovery;
- segment post-policies advance environment behavior such as settling;
- segment validators decide application/dataset acceptance.

For each row, segment acceptance is the conjunction of runtime success, every
post-policy result, and every validator result. `EmbodiedEnv.is_task_success()`
publishes the final bridge mask only after all lazy segment lifecycles complete
normally. Reset lets the base environment consume that result before clearing
the completed bridge.

Physical evidence must come from measured providers. An accepted command is
not evidence. Held-object guards and phase-effect gates may block named atomic
segments or remove action-authorized invalid symbolic relations, but they do
not create constraints, freeze bodies, or override poses. Bounded workflow
recovery retries from fresh observations and may execute a real semantic Pick;
it does not patch task state directly.

`SkillPolicyPreset.effect_monitors` is the single semantic verification
configuration. Omitting it installs the built-in Pick/Place/HandOver monitors;
an explicit empty mapping selects trajectory-only execution, while a partial
mapping verifies only the named calls. Calls without a monitor install no
effect spec, gate, or held-object guard; the runtime skips physical verification
and projects the action plan's expected symbolic state when commands finish.
This mode is useful for lightweight demonstrations analogous to handwritten
trajectories, but command completion is not physical task acceptance. Configure
timed tracking and zero local/workflow recovery budgets separately when the
whole example should remain open-loop.

The two trajectory-only reference profiles also set the runner's
`minimum_cycle_time` to zero. Their call-state transitions therefore proceed
without an extra passive hold, while every emitted trajectory command remains
on the environment's authoritative control-time grid. Monitored integrations
instead use a positive cycle time aligned to `BaseEnv.step_dt`.

Parallel blocks require disjoint resource claims, disjoint runtime targets,
shared-clock alignment, conflict-free symbolic writes, and an authoritative
`ParallelCommandSafetyValidator`. Missing or inconclusive physical-safety
evidence fails closed. Resource disjointness alone is never sufficient.

## Reference integrations

- `ExpertProgramRepeatedPickPlace-v1` is a lightweight trajectory-only example:
  bounded repeat and cyclic targets lower to built-in Pick/Place, with no
  contact sensor, effect verification, settling, segment validator, tracking
  check, or retry budget. It has no task environment module or subclass; its
  Gym JSON declares `expert_program_runtime`, and `config_to_cfg()` binds that
  declaration directly to `EmbodiedEnv` under the configured ID.
- `ExpertProgramOpenDrawer-v1` declares an articulation-link lowerer service
  that lowers its registered simulation call to built-in `Slide`. It snapshots
  the current handle pose for each invocation, so the trajectory-only example
  does not install dynamic-target monitoring or recovery. It also does not wait
  for articulation settling or validate passive joint displacement. Its runtime
  lowerer is created by the decoded registration after scene initialization.
- `HandOver-v1` exercises the coordinated built-in call over disjoint source
  and destination resources with configured hand-over poses, measured gripper
  evidence, settling, and final-target validation.

Do not infer physical qualification from the two trajectory-only examples or
from unit/fake-port tests. Physical acceptance belongs in dedicated validation
integrations with measured evidence, controlled multi-seed/randomization runs,
and task validators. `EmbodiedEnv` is the common execution environment for all
three references. Their Gym configs own the complete supported composition
roots, including the allowlisted services needed by Open Drawer and Hand Over;
none has a task environment module or subclass.

## Recommended change sites

| Change | Owning location |
| --- | --- |
| Schema shape, bounds, node/call config | `embodichain/lab/gym/envs/expert_program/cfg.py` |
| Untrusted JSON/YAML parsing | `loader.py` and `decoder.py` in the same package |
| Static expansion and compiled program structure | `compiler.py` |
| Registration catalog and fingerprints | `catalog.py` and `extensions.py` |
| Generic environment assembly/preflight | `environment.py` |
| Simulation-backed live providers and ports | `simulation.py`, `simulation_environment.py`, and `simulation_policies.py` |
| Lazy Gym action/segment lifecycle | `bridge.py` and `embodichain/lab/gym/envs/demo.py` |
| Environment adapter binding, episode program selection, success/reset | `embodichain/lab/gym/envs/embodied_env.py` and `embodichain/lab/gym/utils/registration.py` |
| Config-created simple runtime and dynamic ID binding | `embodichain/lab/gym/envs/expert_program/configured_runtime.py` and `embodichain/lab/gym/utils/gym_utils.py` |
| Reference scene/profile/runtime values | `embodichain_tasks/configs/tasks/manipulation/{repeated_pick_place,open_drawer,hand_over}/env.json` |
| Configured live-service implementations | `embodichain/lab/gym/envs/expert_program/_configured_runtime_services.py` |

Prefer changing the narrow owner. Do not add task-local motion generators,
duplicate semantic runtimes, process-wide action registries, task-local success
methods, or live helper overrides around a standard registration.

## Common failure modes

- Integration ID mismatch: the program selected a scene registry, profile, or
  preset not owned by the environment registration.
- Fingerprint drift: a supposedly immutable registration or nested declaration
  changed after adapter construction.
- Registered-call mismatch: catalog coverage, call ID, target
  descriptor, or fresh-instance identity does not match the lowerer factory.
- Missing provider: a relation target, hand-over rendezvous, physical effect,
  custom endpoint, or parallel block lacks its exact typed provider.
- Timing mismatch: a command duration cannot map exactly to
  `BaseEnv.step_dt`.
- Lifecycle misuse: a consumer requests the next segment before exhausting the
  current action iterator and invoking its validator.
- False physical success: an integration treats command acceptance or an
  open-loop `Slide` completion as evidence of the application effect.

## Focused validation surface

Run the narrow tests first:

- `tests/gym/envs/expert_program/`;
- `tests/gym/envs/test_demo.py`;
- `tests/gym/envs/test_embodied_env_expert_program.py`;
- `tests/gym/envs/test_settling.py`;
- relevant `tests/sim/skills/` runtime, evidence, and parallel tests;
- CLI/config-path tests under `tests/lab/scripts/` and `tests/gym/utils/`.

Public-contract changes also require API-documentation coverage and a Sphinx
build. Physical acceptance requires a real environment run demonstrating
normal `env.step()` consumption, measured evidence, settling, validation,
row-local outcomes, safe cancellation, and persisted deterministic metadata.
