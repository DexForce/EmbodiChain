(expert-programs)=

# Declarative Expert Programs

Expert Programs let a task describe semantic intent without implementing a
task-local motion generator. A program names registered scene entities, robot
profiles, runtime presets, semantic calls, post-policies, and validators. The
shared compiler lowers every call just in time through the same
`SemanticSkillCompiler`, `AtomicActionEngine`, and `SkillRuntime` used by the
Python semantic API.

Use an Expert Program when later motion depends on the physical result of an
earlier call. Each call receives a fresh scene observation, owns one
`ExecutionSession`, verifies its physical effect, and commits verified symbolic
state before the next call is grounded.

## Architecture and ownership

An Expert Program is a strict frontend to the semantic-skill runtime, not a
second planner, scheduler, effect system, or simulation loop:

```text
JSON / YAML
    -> strict schema-v2 decoder
    -> ExpertProgramCfg
    -> ExpertProgramCompiler + provider-free SceneManifest
    -> CompiledProgram
    -> ExpertProgramEnvironmentAdapter provider-aware preflight
    -> AtomicDemoBridge
    -> SkillRuntime -> ExecutionRunner -> AtomicActionEngine
    -> DemoSegment actions consumed by normal env.step()
```

Python semantic calls and serialized programs converge at
`SemanticCallSpec`. They use the same `SemanticSkillCompiler`, runtime effect
monitors, robot profile, and typed atomic-action core.

| Responsibility | Canonical owner |
| --- | --- |
| Serialized structure, discriminators, and bounds | Expert Program config, loader, and decoder |
| Static scene identity and aliases | `SceneManifest`, projected from `SceneRegistry` |
| Semantic analysis, resource binding, and lowering | `SemanticSkillCompiler` |
| Robot resources, endpoint capabilities, and policy presets | `RobotSkillProfile` |
| Atomic planning and bounded action recovery | `AtomicActionEngine` and `ExecutionRunner` |
| Semantic-call lifecycle and verified symbolic state | `SkillRuntime` |
| Physical evidence and effect decisions | Typed evidence providers and effect monitors |
| Gym action buffering and step handshake | `AtomicDemoBridge` and the demo executor |
| Settling and application acceptance | Segment post-policy and validator ports |
| Final task success | Completed bridge acceptance exposed by `EmbodiedEnv` |

These dependencies are one-way. The simulation semantic-skill package does not
import Gym frontend types, and the bridge does not reproduce planning,
scheduling, effect verification, recovery, or safe-stop behavior.

## Author a program

Schema version 2 is the only accepted schema. It supports bounded `sequence`,
`repeat`, `segment`, and `invoke` nodes plus deterministic `parallel` blocks.
Each parallel block owns its explicit `barrier`; a barrier is not a standalone
program node. Unknown fields, unsupported discriminators,
unbounded structures, executable values, and dotted environment traversal are
rejected before physical execution or command emission.

`RegisteredSemanticCall` is an opaque extension boundary in these schema
versions. An extension with a physical effect must also register its typed
compiler/effect contract; a serialized call ID alone cannot manufacture effect
verification semantics.

The top-level shape is:

```text
ExpertProgramCfg
  schema_version: 2
  program_id
  integration
    robot_profile
    scene_registry
    runtime_preset
  targets
  program
```

The supported program nodes are `SequenceCfg`, `RepeatCfg`, `SegmentCfg`,
`InvokeCfg`, and `ParallelCfg`. `BarrierCfg` belongs to one parallel block and
is not a standalone node. Built-in calls are `PickCfg`, `PlaceCfg`, and
`HandOverCfg`; `RegisteredSemanticCallCfg` is the explicit extension boundary.
The standard segment validators check either a rigid object's measured position
against a target or a named articulation joint against inclusive position
bounds.

The registered-call payload has its own schema revision, independent of the
top-level program version. Config construction enforces the same structural
invariants as decoding, while provider-aware validation separately resolves
catalog, scene, profile, preset, post-policy, and validator IDs. The loader and
decoder reject duplicate keys, unknown fields and discriminators, non-finite
numbers, invalid UTF-8, executable values, environment traversal, excessive
nesting, excessive nodes, excessive repeats, and excessive expanded calls.

The repeated-cube task is configured entirely as semantic calls:

```yaml
schema_version: 2
program_id: repeated_cube_pick_place
integration:
  robot_profile: expert_program_ur5_pick_place
  scene_registry: expert_program_repeated_pick_place
  runtime_preset: safe
targets:
  drop_pose:
    kind: cyclic_pose
    values:
      - position: [-0.40, 0.48, 0.10]
        quaternion_wxyz: [1.0, 0.0, 0.0, 0.0]
program:
  kind: repeat
  count: 3
  body:
    kind: segment
    name: move_cube
    steps:
      kind: sequence
      items:
        - kind: invoke
          call: {kind: pick, object: cube}
        - kind: invoke
          call:
            kind: place
            object: cube
            at: {kind: target_ref, target: drop_pose}
    post:
      - {kind: wait_stable, entity: cube, preset: rigid_object}
    validators:
      - kind: object_near_target
        object: cube
        target: drop_pose
        position_tolerance: 0.12
```

The top-level Gym configuration selects the file with a path relative to that
configuration file:

```json
{
  "expert_program_path": "../../expert_program/my_task.yaml"
}
```

Alternatively, `run_env` can load a program explicitly:

```bash
python -m embodichain.lab.scripts.run_env \
  --gym_config path/to/gym.json \
  --expert-program path/to/program.yaml
```

Future model-facing frontends must feed the same strict decoder and
`ExpertProgramEnvironmentAdapter.compile` boundary. They must not introduce a
second schema, compiler, or runtime, and the trusted host must continue to own
integration selection and executable extensions.

## Integrate a simulation task

Task code supplies typed scene and robot integration declarations once, while
the external Expert Program configuration owns task sequence and targets. The
task then delegates runtime assembly to the shared factory; it does not
construct approach, grasp, pull, or placement trajectories:

```python
MY_EXPERT_PROGRAM_REGISTRATION = SimulationExpertProgramRegistration(
    scene_binding=create_my_scene_binding(),
    robot_profile_binding=create_my_robot_profile_binding(),
)


class MyTaskEnv(EmbodiedEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._expert_program_adapter = create_simulation_expert_program_adapter(
            self,
            registration=MY_EXPERT_PROGRAM_REGISTRATION,
        )

    @property
    def expert_program_adapter(self):
        return self._expert_program_adapter
```

The scene binding is authoritative for semantic identity, live pose sources,
geometry, affordances, and collision roles. The robot profile owns reusable
resources, endpoint capabilities, semantic commands, policy presets, and effect
monitor selection. `SimulationRobotSkillProfileBinding` accepts generic
core `RobotResource` declarations containing arbitrary typed `ResourceEndpoint`
values directly; `ControlPartResourceBinding` is the joint-backed convenience.
Endpoint adapters and runtime transports are the extension boundary for
mobile-base, whole-body, or non-joint controllers. On the standard path they
are owned by `SimulationExpertProgramRegistration`, not passed as live helper
overrides. Their exact target, payload, route, and transport declarations enter
the catalog fingerprint, while transport tuple order defines deterministic
Gym-action composition order. Task programs keep the same semantic calls and
do not gain controller-shaped fields.

The standard registration currently installs built-in joint tracking and
effect-evidence routes only for `ControlPartEndpoint`. A custom endpoint adapter
must declare empty tracking and evidence routes, so it uses timed/open-loop
completion. Non-joint closed-loop projectors, feedback sources, metric
evaluators, and effect-evidence backends still require registration-owned
provider-factory contracts. Whole-body controllers expressed through existing
joint control parts continue to use the built-in joint route.

`SimulationExpertProgramRegistration` is the standard task-owned trust
boundary. Besides the provider-free scene/profile declarations, it owns the
call catalog, settling presets, relation grounders, hand-over pose providers,
endpoint adapters, ordered runtime transports, and optional parallel-safety,
control-part-evidence, and registered-lowerer factories. These declarations
enter one immutable integration fingerprint and are checked again against the
live engine, endpoints, and scene registry during runtime assembly. Supplying a
registration forbids helper arguments from replacing its components after
preflight.

Every registered semantic-call descriptor must have exactly one matching
`RegisteredSemanticLowererFactory`. Its frozen provider-free declaration fixes
the call ID and factory revision; the call catalog fixes the payload schema and
target descriptor. Each runtime assembly gets a fresh lowerer and revalidates
its call ID, schema version, and target descriptor, preventing mutable
task-local lowerers from becoming an untracked runtime side channel.

Relation and rendezvous semantics are also explicit integration capabilities.
`Place(on=...)` and `Place(inside=...)` require an exact typed/versioned
`RelationTargetGrounder` for the selected affordance payload. `HandOver`
requires the profile-selected `HandOverPoseProvider`. A direct `Place(at=...)`
does not require a relation grounder. Missing, ambiguous, or stale providers
fail during provider-aware program preflight, before the first physical action.

The same preflight rejects a reachable `safe` preset for a dynamic scene before
the first observation when the active motion generator cannot provide the
required dynamic collision world.

## Execution and physical effects

`AtomicDemoBridge` yields lazy `DemoSegment` actions. Every command and settling
hold is consumed by normal `env.step()`, so action managers, recorders, rewards,
timing, and dataset boundaries remain authoritative. `BaseEnv.step_dt` is the
only control cadence; a command duration that is not representable on that grid
fails instead of being silently resampled.

Compilation returns one already materialized bounded program. Creating a bridge
performs provider-aware semantic analysis before any command can be emitted. Sequential
stretches retain downstream object-target look-ahead across segment boundaries;
an explicit parallel block is a conservative look-ahead barrier. Runtime still
re-observes and grounds each call just in time after prior verified effects.

The standard simulation integration never treats an accepted controller command
as physical evidence. Grasp, release, and hand-over verification consume live
pose evidence together with registered physical evidence routes. The standard
registration path does not accept task-side contact, constraint, force, or
wrench callback overrides; missing physical channels remain invalid and fail
closed. Advanced integrations may still assemble explicit providers directly.

Program/demo-segment metadata records runtime call results, named trajectory
segments, effect decisions, recovery events, scene and collision revisions,
settling outcomes, and validator results in deterministic JSON-safe values.
Trajectory segments are trace ranges inside one atomic plan; they do not create
independent recovery or timeout boundaries.

Three success boundaries remain distinct:

| Boundary | Meaning |
| --- | --- |
| Atomic or semantic effect monitor | Decides whether one physical call succeeded and participates in recovery |
| Program-segment post-policy | Advances environment behavior such as settling after motion completion |
| Program-segment validator | Decides whether the application or dataset segment is acceptable |

For each participating row, segment acceptance is the conjunction of semantic
runtime success, every post-policy result, and every validator result. Final
Expert Program success is published only after all lazy segment lifecycles have
been consumed normally.

Curated manipulation calls may also install physical boundaries inside an
atomic plan. A held-object guard detects a lost relation before a dependent
named segment, while a phase-effect gate blocks entry until the preceding
attachment or release transition has physical evidence. They can reconcile
only action-authorized state removals and never create simulator constraints,
freeze bodies, or override poses. After terminal reconciliation, a bounded
workflow recovery policy may retry from still-verified state or perform a real
semantic `Pick` before retrying the original call. Successful peer rows remain
at the shared call barrier.

Parallel blocks additionally require an authoritative
`ParallelCommandSafetyValidator`. Resource-claim disjointness is necessary but
is not treated as proof of physical safety. If no validator is installed, the
parallel block refuses to start; the standard simulation adapter intentionally
does not invent one from resource names. Its task registration must declare a
safety factory for the exact transport set, and each runtime assembly receives
a fresh validator from that factory. Every parallel frame must occupy
exactly one `BaseEnv.step_dt`; shorter lanes repeat their last safe target as
hold padding, while fractional frames are rejected rather than resampled.
The runtime also uses strict symbolic key-level conflict detection at the barrier:
two branches may not commit the same task-state key, even when their physical
changes occurred in disjoint environment rows.

## Python semantic calls

Standalone applications can use the same compiler and runtime through
`AtomicSkills`:

```python
skills = AtomicSkills.from_env(runtime_provider, preset="safe")
cube = skills.scene.object("cube")
tray = skills.scene.object("tray")
result = skills.run(Pick(object=cube), Place(object=cube, on=tray))
```

In this example, `runtime_provider` owns the typed relation grounder for the
tray's placement affordance. Applications without such a provider can use a
direct `SemanticPose` through `Place(at=...)`.

`from_env` requires an explicit `SkillRuntimeProvider`; it never scans arbitrary
environment attributes. Gym demonstration environments intentionally use the
lazy bridge instead, because a synchronous runtime would bypass the required
`env.step()` handshake. Advanced applications may use
`AtomicSkills.from_components(...)` with explicit observation, command,
evidence, and clock ports.

## Reference integrations

The packaged tasks demonstrate three different integration paths:

| Environment | Declarative path | Acceptance boundary |
| --- | --- | --- |
| `ExpertProgramRepeatedPickPlace-v1` | Bounded repeat of built-in `Pick` and `Place` calls with cyclic targets | Rigid-object settling plus measured object-near-target validation |
| `ExpertProgramOpenDrawer-v1` | Registered task call lowered by a registration-owned factory to the built-in `Slide` primitive | Articulation settling plus measured passive-joint position validation |
| `HandOver-v1` | Built-in coordinated `HandOver` over disjoint source and destination resources | Rigid-object settling plus measured object-near-target validation |

Open Drawer intentionally keeps articulation success at the application
boundary: `Slide` proves motion completion, not drawer travel. A dedicated
drawer-specific atomic primitive would duplicate that existing motion contract.

For the lower-level planning and execution contracts, see {doc}`index`. For
robot resource and endpoint declarations, see {doc}`robot_skill_profiles`.

## Capability status

| Surface | Shared contract | Standard simulation integration |
| --- | --- | --- |
| `Pick` | Compiler, runtime, effect verification | Antipodal grasp binding plus motion/grasp resources |
| `Place(at=...)` | Object-centric lowering with verified held state | Direct semantic pose target |
| `Place(on=...)` / `Place(inside=...)` | Exact typed relation dispatch | Integration must install the matching `RelationTargetGrounder` |
| `HandOver` | Coordinated call, state flow, and effect contract | Embodiment must install its named `HandOverPoseProvider` and evidence sources |
| Registered calls | Typed call catalog and explicit lowerer | Physical extensions must add an explicit effect contract |
| Mobile/whole-body extensions | Generic resources, claims, endpoint targets, command frames, and routing | Requires a reusable semantic skill/lowerer plus matching adapter, payload, transport, and effect integration; no curated navigation or whole-body skill is installed today |
| Parallel blocks | Shared-clock coordinator and strict barrier merge | Requires an authoritative `ParallelCommandSafetyValidator`; none is inferred by default |

The table separates implemented reusable contracts from embodiment-specific
providers. It is not a claim that every row has completed task-level physical
simulation acceptance.

Articulated tasks should reuse the existing motion-centric `Slide` primitive and
verify articulation completion at the application boundary. A dedicated drawer
or `OperateArticulation` semantic path is intentionally not part of this API.

## Qualification boundaries

The reusable contracts and focused tests do not by themselves qualify every
physical task. Before broader task-level claims, run the repeated-cube and Open
Drawer integrations across controlled seeds and the intended randomization
envelope, then inspect their persisted segment metadata. Concurrent task
migration additionally requires an environment-qualified
`ParallelCommandSafetyValidator`; resource disjointness alone is insufficient.

A generic `DeclarativeExpertProgramEnv` is intentionally deferred. Concrete
tasks still own scene assets, sensors, evidence sources, and their immutable
integration registration. Custom non-control-part endpoints on the standard
path currently use timed/open-loop completion until registration-owned
closed-loop feedback, projection, metric, and evidence-provider factories are
defined for those endpoint families.
