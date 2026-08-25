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
  -> strict schema-v2 validation
  -> ExpertProgramCompiler + provider-free SceneManifest
  -> immutable CompiledProgram
  -> ExpertProgramEnvironmentAdapter provider-aware preflight
  -> AtomicDemoBridge
  -> SkillRuntime -> ExecutionRunner -> AtomicActionEngine
  -> DemoSegment actions consumed through env.step()
```

## Configuration and compilation contract

`EXPERT_PROGRAM_SCHEMA_VERSION == 2` is the only top-level schema version.
`REGISTERED_SEMANTIC_CALL_SCHEMA_VERSION == 1` independently versions opaque
registered-call payloads; do not couple the two revisions.

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
extension routes. When a registration is supplied, helper arguments cannot
replace registration-owned catalogs, adapters, lowerers, transports, ports,
runner policy, or safety/evidence providers.

Every registered call descriptor requires exact one-to-one coverage by a
`RegisteredSemanticLowererFactory`. Its frozen provider-free declaration fixes
the call ID and factory revision; the call catalog fixes the payload schema and
target descriptor. Each runtime assembly must receive a fresh lowerer whose
live declaration matches the call ID, schema version, and target descriptor.
Stateful factories must expose recursively immutable configuration; do not put
live simulator objects or mutable task state in their declaration.

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

Parallel blocks require disjoint resource claims, disjoint runtime targets,
shared-clock alignment, conflict-free symbolic writes, and an authoritative
`ParallelCommandSafetyValidator`. Missing or inconclusive physical-safety
evidence fails closed. Resource disjointness alone is never sufficient.

## Reference integrations

- `ExpertProgramRepeatedPickPlace-v1` exercises bounded repeat, cyclic targets,
  built-in Pick/Place, contact-backed effect evidence, rigid-object settling,
  and object-near-target validation. Its task class supplies only the bundled
  default program; adapter construction is delegated to `EmbodiedEnv`.
- `ExpertProgramOpenDrawer-v1` owns a versioned registered-lowerer factory that
  lowers its task call to built-in `Slide`; articulation settling and passive
  joint validation remain application-level acceptance. Its live drawer
  lowerer is created by the registration-owned runtime factory after scene
  initialization.
- `HandOver-v1` exercises the coordinated built-in call over disjoint source
  and destination resources with measured final-target validation.

Do not infer broad physical qualification from unit or fake-port tests. Wider
qualification still requires controlled multi-seed/randomization runs for the
repeated-cube and Open Drawer tasks, plus an environment-qualified parallel
safety factory before migrating concurrent tasks. `EmbodiedEnv` is the common
execution environment; concrete task composition roots still own assets,
sensors, evidence sources, and provider-free registrations.

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
| Concrete task declarations | `embodichain_tasks/embodichain_tasks/expert_program/` |

Prefer changing the narrow owner. Do not add task-local motion generators,
duplicate semantic runtimes, process-wide action registries, task-local success
methods, or live helper overrides around a standard registration.

## Common failure modes

- Integration ID mismatch: the program selected a scene registry, profile, or
  preset not owned by the environment registration.
- Fingerprint drift: a supposedly immutable registration or nested declaration
  changed after adapter construction.
- Registered-call mismatch: catalog coverage, call ID, schema version, target
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
