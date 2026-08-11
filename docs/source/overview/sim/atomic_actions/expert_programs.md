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

## Author a program

Schema version 1 supports bounded `sequence`, `repeat`, `segment`, and `invoke`
nodes. Schema version 2 additionally supports deterministic `parallel` blocks
and explicit `barrier` nodes. Unknown fields, unsupported discriminators,
unbounded structures, executable values, and dotted environment traversal are
rejected before physical execution or command emission.

`RegisteredSemanticCall` is an opaque extension boundary in these schema
versions. An extension with a physical effect must also register its typed
compiler/effect contract; a serialized call ID alone cannot manufacture effect
verification semantics.

The repeated-cube task is configured entirely as semantic calls:

```yaml
schema_version: 1
program_id: repeated_cube_pick_place
integration:
  robot_profile: ur5_parallel_gripper_v1
  scene_registry: multi_segments_cube_v1
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

`run_env` can override it explicitly:

```bash
python -m embodichain.lab.scripts.run_env \
  --gym_config path/to/gym.json \
  --expert-program path/to/program.yaml
```

## Accept untrusted model output

Model-generated programs use the same decoder and compiler, but enter through
the narrower MLLM frontend. The trusted host owns the scene, robot profile, and
runtime preset; the model response must omit `integration` entirely:

```python
from embodichain.agents.mllm import compile_mllm_expert_program
from embodichain.lab.gym.envs.expert_program import ExpertProgramIntegrationCfg

compiled = compile_mllm_expert_program(
    model_response,
    adapter=adapter,
    integration=ExpertProgramIntegrationCfg(
        robot_profile="my_robot_v1",
        scene_registry="my_scene_v1",
        runtime_preset="safe",
    ),
)
```

This entry point accepts exactly one bounded JSON document. It rejects duplicate
keys, non-finite or overflowing numeric values, invalid Unicode, Markdown
fences, trailing text, and every normal schema violation. Its initial policy is
deliberately smaller than the file format: only schema version 1 and curated
`pick`, `place`, `hand_over`, and `operate_articulation` calls are admitted.
The model cannot select `resources`, a hand-over `receiver`, a runtime preset,
or an explicit articulation position/displacement; articulation operations must
use a host-declared named target. Registered calls and parallel nodes remain
host-authored extensions.

`compile_mllm_expert_program` delegates to the existing
`ExpertProgramEnvironmentAdapter.compile` method. It neither creates a second
compiler nor assembles a runtime while validating model output.

## Integrate a simulation task

Task code supplies typed scene and robot integration declarations once, while
the external Expert Program configuration owns task sequence and targets. The
task then delegates runtime assembly to the shared factory; it does not
construct approach, grasp, pull, or placement trajectories:

```python
class MyTaskEnv(ExpertProgramEnvironmentMixin, EmbodiedEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._expert_program_adapter = create_simulation_expert_program_adapter(
            self,
            scene_binding=create_my_scene_binding(),
            robot_profile_binding=create_my_robot_profile_binding(),
        )

    @property
    def expert_program_adapter(self):
        return self._expert_program_adapter
```

The scene binding is authoritative for semantic identity, live pose sources,
geometry, affordances, and collision roles. The robot profile owns reusable
resources, endpoint capabilities, semantic commands, policy presets, and effect
monitor selection. `SimulationRobotSkillProfileBinding` accepts generic
`RobotResourceBinding` declarations containing arbitrary typed
`ResourceEndpoint` values; `ControlPartResourceBinding` is its stricter
joint-backed convenience. Endpoint adapters and runtime transports are the
extension boundary for mobile-base, whole-body, or non-joint controllers and
are accepted by the standard simulation helper. Task programs keep the same
semantic calls and do not gain controller-shaped fields.

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

Creating a bridge materializes the bounded segment stream and performs
provider-aware semantic analysis before any command can be emitted. Sequential
stretches retain downstream object-target look-ahead across segment boundaries;
an explicit parallel block is a conservative look-ahead barrier. Runtime still
re-observes and grounds each call just in time after prior verified effects.

The standard simulation integration verifies grasp and release with two pieces
of evidence:

- the last exact open/grasp command accepted by the buffered Gym command sink,
  tracked independently for every stable environment ID; and
- the live object-to-endpoint pose relation from the shared scene snapshot.

The command-state update is transactional: encoder, buffer, cancellation, or
safe-stop failures invalidate it. An integration with contact, constraint,
force, or wrench sensing can install typed evidence callbacks without changing
the semantic call or program.

Program/demo-segment metadata records runtime call results, named trajectory
segments, effect decisions, recovery events, scene and collision revisions,
settling outcomes, and validator results in deterministic JSON-safe values.
Trajectory segments are trace ranges inside one atomic plan; they do not create
independent recovery or timeout boundaries.

Schema-version-2 parallel blocks additionally require an authoritative
`ParallelCommandSafetyValidator`. Resource-claim disjointness is necessary but
is not treated as proof of physical safety. If no validator is installed, the
parallel block refuses to start; the standard simulation adapter intentionally
does not invent one from resource names. Every parallel frame must occupy
exactly one `BaseEnv.step_dt`; shorter lanes repeat their last safe target as
hold padding, while fractional frames are rejected rather than resampled.
Version 2 also uses strict symbolic key-level conflict detection at the barrier:
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

For the lower-level planning and execution contracts, see {doc}`index`. For
robot resource and endpoint declarations, see {doc}`robot_skill_profiles`.

## Capability status

| Surface | Shared contract | Standard simulation integration |
| --- | --- | --- |
| `Pick` | Compiler, runtime, effect verification | Antipodal grasp binding plus motion/grasp resources |
| `Place(at=...)` | Object-centric lowering with verified held state | Direct semantic pose target |
| `Place(on=...)` / `Place(inside=...)` | Exact typed relation dispatch | Integration must install the matching `RelationTargetGrounder` |
| `HandOver` | Coordinated call, state flow, and effect contract | Embodiment must install its named `HandOverPoseProvider` and evidence sources |
| `OperateArticulation` | Named/absolute/displacement target and joint effect | Link, joint, operation-affordance, and interaction endpoint bindings |
| Registered calls | Typed call catalog and explicit lowerer | Physical extensions must add an explicit effect contract |
| Mobile/whole-body extensions | Generic resources, claims, endpoint targets, command frames, and routing | Requires a reusable semantic skill/lowerer plus matching adapter, payload, transport, and effect integration; no curated navigation or whole-body skill is installed today |
| Parallel blocks | Shared-clock coordinator and strict barrier merge | Requires an authoritative `ParallelCommandSafetyValidator`; none is inferred by default |

The table separates implemented reusable contracts from embodiment-specific
providers. It is not a claim that every row has completed task-level physical
simulation acceptance.

The Open Drawer vertical slice has completed its supported-simulation physical
run and reached the configured drawer joint target. Repeated cube pick/place has
completed one physical Pick/Place/settle/validator cycle; its full three-cycle
run remains in threshold calibration.
