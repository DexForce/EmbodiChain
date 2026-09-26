# Isaac-Lab-Style Action Manager Refactor Design

## Objective

Replace EmbodiChain's shared-`EnvAction` preprocessing pipeline with an
Isaac-Lab-style action manager. The manager owns one ordered flat policy action,
splits it into action-term slices, lets each term process its own slice, and
lets each term apply commands only to the joints or other resources it owns.

The refactor removes the combined arm-and-gripper action classes introduced by
the current branch. Arm, parallel gripper, dexterous-hand joint, and future
tendon actions become independent terms with explicit dimensions and resource
ownership.

This is an intentional breaking change. No runtime compatibility adapter,
legacy action-space layout, old class aliases, or post-action term lifecycle is
retained.

## Motivation

The current `ActionManager` asks terms to return tensors or a shared
`TensorDict` containing controller keys such as `qpos`, then asks
`EmbodiedEnv` to apply the combined result. That design has several limits:

- multiple terms that produce `qpos` do not have an explicit merge contract;
- action dimensions and spaces can disagree with dynamically resolved joints;
- a combined EEF-plus-gripper term hard-codes one robot's arm width and control
  part names;
- mimic followers can receive stale targets when a term returns a full qpos
  vector but updates only independent joints;
- policy action slices and dataset feature semantics are inferred rather than
  declared; and
- post terms transform a shared action after execution, obscuring whether the
  value represents a policy request, processed command, or executed target.

The official main-branch usage does not justify preserving this architecture.
Seven logical tasks use ActionManager: CartPole and six locomotion tasks. All
use exactly one pre term and already expose a flat `Box`. The current branch
adds two RLinf repeated-pick-place variants, which have not been released and
can migrate directly.

## Scope

This change includes:

- a new `ActionTerm` process/apply lifecycle;
- ordered flat action slicing in `ActionManager`;
- direct term ownership of selected robot joints;
- migration and renaming of all built-in action implementations;
- an independent parallel-gripper action;
- removal of the combined EEF/gripper and joint/gripper actions;
- removal of pre/post modes and the legacy `process_action()` protocol;
- action IO descriptors suitable for dataset metadata;
- migration of every official YAML configuration and Python consumer;
- preservation of the direct `ControllerAction` expert/Task Program path; and
- proportional unit, integration, configuration, dataset, and simulator
  validation.

## Non-Goals

This change does not implement:

- fixed-tendon backend APIs;
- hand-synergy decoders;
- per-physics-substep action application;
- automatic migration of third-party configurations;
- old class-name aliases;
- mixed old/new action terms;
- a legacy Dict action space; or
- controller-qpos dataset features.

The new interfaces must leave room for fixed-tendon and synergy terms without
implementing them prematurely.

## Public Contracts

### ActionTerm

`ActionTerm` remains the abstract manager-owned base class. Concrete
implementations end in `Action`, not `ActionTerm`.

```python
class ActionTerm(Functor):
    @property
    def action_dim(self) -> int:
        ...

    @property
    def action_space(self) -> gym.spaces.Box:
        ...

    @property
    def raw_actions(self) -> torch.Tensor:
        ...

    @property
    def processed_actions(self) -> torch.Tensor:
        ...

    @property
    def command_type(self) -> str:
        ...

    @property
    def controlled_joint_ids(self) -> tuple[int, ...]:
        ...

    @property
    def descriptor(self) -> ActionTermDescriptor:
        ...

    def process_actions(self, actions: torch.Tensor) -> None:
        ...

    def apply_actions(self) -> None:
        ...

    def reset(self, env_ids: torch.Tensor | list[int] | None = None) -> None:
        ...
```

Each term allocates a fixed `[num_envs, action_dim]` raw-action buffer.
Processed-action shape is term-owned and may differ: an EEF term turns a 6D
pose into selected-arm qpos, while a one-dimensional gripper action turns into
one command per independent gripper joint. `process_actions()` updates both
buffers once per environment control step. `apply_actions()` writes only the
term-owned joint IDs or resource IDs.

`ActionTermCfg` keeps `func`, `params`, and `extra`, but removes `mode`. The
terms resolve part names, explicit joint names, regex selections, joint order,
and mimic exclusion once during construction.

### ActionTermDescriptor

Every term exposes JSON-compatible semantics independently of dataset code:

```python
@dataclass(frozen=True, slots=True)
class ActionTermDescriptor:
    representation: str
    feature_names: tuple[str, ...]
    units: tuple[str, ...]
    normalization: str | None
    joint_names: tuple[str, ...]
    metadata: Mapping[str, JSONValue]
```

The manager binds the term name and flat slice to produce an
`ActionDescriptor`:

```python
@dataclass(frozen=True, slots=True)
class ActionDescriptor:
    name: str
    start: int
    stop: int
    term: ActionTermDescriptor
```

The descriptor is the source of truth for action feature names, ordering,
units, normalization, and dataset metadata. Dataset code must not infer these
values from concrete class names.

### ActionManager

`ActionManager` owns the current and previous flat policy actions:

```python
class ActionManager:
    @property
    def total_action_dim(self) -> int:
        ...

    @property
    def single_action_space(self) -> gym.spaces.Box:
        ...

    @property
    def action(self) -> torch.Tensor:
        ...

    @property
    def previous_action(self) -> torch.Tensor:
        ...

    @property
    def descriptors(self) -> tuple[ActionDescriptor, ...]:
        ...

    def process_action(self, action: torch.Tensor) -> None:
        ...

    def apply_action(self) -> None:
        ...
```

The manager validates the complete batch shape and device, stores the flat raw
action, slices it in configuration order, and calls `term.process_actions()`.
`apply_action()` calls each term's `apply_actions()`.

The manager rejects overlapping controlled joint IDs for terms that write the
same robot command type (`qpos`, `qvel`, or `qf`). Overlap errors identify both
term names and the joint names. Terms that explicitly own a different resource
index space, such as a future fixed tendon, do not collide with joint IDs.

The manager action space is always a flat `gym.spaces.Box`. It concatenates
each term's low and high arrays in descriptor order. Dict and TensorDict policy
actions are no longer accepted.

## Environment Lifecycle

### Policy path

The normal policy path becomes:

```text
env.step(flat_action)
  -> ActionManager.process_action(flat_action)
  -> ActionManager.apply_action()
  -> physics substeps
  -> observations / info / rewards
  -> rollout and dataset hooks receive the flat raw policy action
```

Term commands are applied once before the physics substeps. Position, velocity,
and effort targets persist across substeps as they do today. Applying actions
before every physics substep is a separate future change.

No post-action mode remains. Observations, rewards, and datasets obtain raw or
processed action state through manager or term properties.

### ControllerAction path

Task Program and expert execution retain the controller-ready envelope:

```text
env.step(ControllerAction)
  -> skip ActionManager
  -> validate qpos/qvel/qf batch, width, dtype, and device
  -> apply through the existing direct controller boundary
  -> physics substeps
  -> expert rollout stores the ExpertActionSpec representation
```

`ControllerAction` remains distinct through preprocessing so `_step_action()`
can choose the direct path without relying on mutable flags or tensor-shape
inference.

## Built-In Actions

The refactor replaces the current concrete classes with:

| New class | Responsibility |
|---|---|
| `JointPositionAction` | Absolute selected-joint position targets |
| `JointPositionToLimitsAction` | Per-joint normalized action mapped to limits |
| `JointVelocityAction` | Selected-joint velocity targets |
| `JointEffortAction` | Selected-joint effort targets |
| `RelativeJointPositionAction` | Current/default-relative position targets |
| `DefaultJointPositionAction` | Default-offset locomotion action with encoder bias |
| `EefPoseAction` | Selected-arm EEF pose through IK |
| `ParallelGripperAction` | One scalar controlling an explicit parallel-gripper mapping |

The following names are removed without aliases:

```text
QposTerm
QposDenormalizedTerm
QposNormalizedTerm
QvelTerm
QfTerm
DeltaQposTerm
DefaultJointPositionTerm
EefPoseTerm
EefPoseGripperTerm
JointPositionGripperTerm
```

### Joint actions

Joint actions accept `part_name` or `joint_names`, plus `preserve_order` where
order affects the policy interface. `action_dim` equals the number of resolved,
non-mimic controlled joints. Each action applies only its selected IDs.

`DefaultJointPositionAction` exposes `raw_actions`, `previous_raw_actions`, and
`position_bias`. Locomotion consumers migrate to those names.

### EefPoseAction

`EefPoseAction` accepts a selected arm part and an explicit pose
representation. The first delivery supports canonical absolute `xyz_rpy` for
the RLinf policy and preserves existing 6D/7D IK support where explicitly
configured.

The term stores raw pose actions, computes IK once per control step, stores the
selected-arm qpos target and `ik_success`, and applies only the arm joints. IK
failure holds the current arm qpos for the failed row.

### ParallelGripperAction

`ParallelGripperAction` owns a gripper part and always has `action_dim == 1`.
It supports two modes:

- `continuous`: map a normalized scalar in `[-1, 1]` between configured lower
  and upper commands for every independent gripper joint;
- `binary`: choose configured open or close commands from the sign of the raw
  scalar.

Commands are resolved by joint-name expressions and must cover every selected
independent joint. Opposing joint directions are expressed in the configured
command maps, not inferred from one shared joint limit convention. Mimic
followers are excluded from command writes.

Validation remains device-native in the control hot path. The term uses clamp,
`torch.where`, or tensor math; it does not convert CUDA tensors to Python
booleans every step. Strict dataset validation runs on the CPU snapshot at the
persistence boundary.

### Dexterous hands

Dexterous hands use `JointPositionToLimitsAction`, one action dimension per
independent actuated joint. A scalar parallel-gripper term is never broadcast
implicitly over a dexterous hand.

A future `HandSynergyAction` must declare `synergy_id`, input dimension,
output joint names, and a deterministic decoder. A future
`FixedTendonPositionAction` uses a separate tendon index space after simulation
assets expose tendon lookup and write APIs.

## Dataset Contract

Policy-controlled datasets record the flat raw action received by
`ActionManager`. The recorder receives the ordered manager descriptors and
stores them in action feature metadata. Controller qpos is not recorded.

An EEF plus parallel-gripper descriptor is equivalent to:

```json
[
  {
    "name": "arm_action",
    "slice": [0, 6],
    "representation": "eef_pose",
    "feature_names": ["x", "y", "z", "roll", "pitch", "yaw"],
    "units": ["m", "m", "m", "rad", "rad", "rad"]
  },
  {
    "name": "gripper_action",
    "slice": [6, 7],
    "representation": "parallel_gripper",
    "feature_names": ["gripper"],
    "normalization": "minus_one_to_one"
  }
]
```

The composed dataset representations are:

```text
eef_pose_parallel_gripper
joint_position_parallel_gripper
joint_position
joint_position_velocity
hand_joint_position
```

Task Program and expert `ControllerAction` rollouts do not manufacture a
policy descriptor. They continue to use `ExpertActionSpec` and the existing
joint-position or joint-position-velocity schema.

For a dexterous hand, the descriptor includes one feature name and one joint
name per controlled dimension. A future synergy representation additionally
includes its `synergy_id`. Different descriptor lists are distinct dataset
schemas and cannot be merged without an explicit transform.

## Official Task Migration

All official uses migrate in this change:

- CartPole uses `RelativeJointPositionAction`.
- ANYmal C, G1, Go1, Go2, H1_2, and MicroDuck use
  `DefaultJointPositionAction` in both Default and Newton configurations.
- RLinf EEF repeated pick-and-place uses `EefPoseAction` plus
  `ParallelGripperAction`.
- RLinf joint repeated pick-and-place uses `JointPositionAction` plus
  `ParallelGripperAction`.

The Franka RLinf deployments explicitly select `control_parts: [arm, hand]` so
active joint order excludes mimic followers. No action term emits or writes a
stale full-qpos follower target.

The locomotion environment changes its direct term-state access from
`action`/`previous_action` to `raw_actions`/`previous_raw_actions` and keeps
`position_bias` under `DefaultJointPositionAction`.

## Error Handling

The manager rejects:

- wrong batch rank, row count, total action width, dtype, or device;
- non-finite policy actions when a term contract requires finite values;
- action-space bounds at the term's declared validation boundary;
- unresolved or empty joint selections;
- duplicate term names;
- overlapping joint command ownership;
- descriptor shape or name-count mismatches; and
- unsupported Dict/TensorDict policy actions.

Each error names the owning term and the invalid shape, joint, or resource.
Validation that requires reading a CUDA scalar is not performed on every
control step unless the control path already synchronizes for another reason.

## Breaking Migration

This is a breaking public API change. The PR description and migration notes
must list:

- every removed class name and its replacement;
- removal of `ActionTermCfg.mode`;
- removal of `process_action()` and `input_key`;
- removal of Dict/TensorDict policy actions;
- removal of post action terms;
- the new flat term-order action layout;
- the locomotion property renames; and
- dataset representation and metadata changes.

No aliases or deprecation window are provided. Unknown old function names fail
through the existing strict callable resolver with a migration-oriented error
where practical.

## Validation

Pure unit tests cover:

- ordered flat slicing and concatenated action bounds;
- wrong shape, dtype, device, finite-value, and bounds failures;
- per-term raw, previous, and processed buffers;
- selected-row reset behavior;
- joint ownership overlap rejection;
- six- and seven-DoF arm dimensions;
- parallel gripper continuous and binary mappings;
- opposing gripper joint directions;
- mimic follower exclusion;
- EEF IK success/failure row behavior;
- direct qpos, qvel, and effort application;
- ControllerAction bypass; and
- descriptor validation and serialization.

Focused integration tests cover:

- CartPole action equivalence;
- every locomotion configuration's action width and term state;
- Franka EEF `6 + 1` and joint `7 + 1` layouts;
- W1/BrainCo six-joint hand ordering through
  `JointPositionToLimitsAction`;
- policy and expert rollout separation;
- synchronous, asynchronous, and fragment dataset action round trips;
- official task loading and package contents; and
- Default and Newton low-resource simulator smoke tests.

GPU validation confirms that the control hot path contains no per-step Python
scalar conversion or implicit device-to-host transfer. Benchmarking is
required only if static inspection cannot establish the absence of a new
synchronization.

Repository gates include Black 26.3.1, focused pytest suites, API documentation
coverage, context checks, compileall, and `git diff --check`. The full suite is
reserved for the completed cross-subsystem implementation because the change
affects shared environment control behavior.

## Documentation and Project Context

The manager/functor context becomes the owner of the process/apply lifecycle,
flat term slicing, joint ownership, and descriptors. The environment context
documents the separate policy and ControllerAction paths. The data-pipeline
context documents descriptor-owned policy schemas and ExpertActionSpec-owned
expert schemas. Robot context remains the owner of control parts, joint order,
mimic relationships, and future tendon resources.

Public API docs are regenerated for the new action classes and removed for the
deleted names. The repeated-pick-place README documents composed term order and
the resulting flat dataset action layout.
